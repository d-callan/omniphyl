"""Kaggle GPU kernel: multi-genome Stage-2 refiner training.

Trains OmniPhylRefiner against frozen v6 Stage-1 features on a
clade-diverse genome set (Apicomplexa + kinetoplastid + amoeba +
fungus from VEuPathDB, plus E. coli / Halobacterium from RefSeq).
Each genome arrives as a mounted Kaggle dataset containing one
genome fasta + one GFF; per-genome Stage-1 caches (.res.pkl) are
mounted either inside those datasets or as separate cache datasets
-- every *.res.pkl under /kaggle/input is seeded into the cache dir
before training so no Stage-1 inference runs unless a genome lacks
a cache.

Dataset layout convention (per genome, one Kaggle dataset):
  <anything>_Genome.fasta | *.fna | *.fasta   genome sequence
  *.gff | *.gff3                              CDS annotation
  <fasta_stem>.res.pkl                        optional Stage-1 cache

Holdout: set HELDOUT env (or edit the constant below) to a substring
of a dataset directory name to route that genome to --val-data
(whole-genome holdout) instead of --data. Additionally any seqids in
VAL_SEQIDS are held out of --data pairs via --val-seqids.

Outputs to /kaggle/working:
  refiner_cache/*.res.pkl   frozen Stage-1 features per fasta stem
  refiner_multi.pt          trained refiner ckpt (best val nucF1)
  results.json              per-epoch metrics + held-out decode
                            benchmark (refiner vs heuristic)
"""

import glob
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM -- verify phone number at "
        "kaggle.com/settings (required for internet + GPU), then "
        "re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
RESULT = WORK / "results.json"
OUT_PT = WORK / "refiner_multi.pt"
CACHE = WORK / "refiner_cache"
os.environ.setdefault("OMP_NUM_THREADS", "4")

# substring of a dataset dir name -> that genome goes to --val-data
# (whole-genome holdout; decode benchmark below evaluates it)
HELDOUT = os.environ.get("HELDOUT", "")
# substring of a dataset dir name -> drop that genome ENTIRELY (not
# even validation). Used for data-regime ablations, e.g. EXCLUDE=
# pfalciparum for the no-Plasmodium trial.
EXCLUDE = os.environ.get("EXCLUDE", "")
# seqids held out of --data pairs (chromosome-level holdout for a
# genome that otherwise stays in training). Pf3D7 chr1 is the
# default early-stopping metric; the honest generalization eval is
# the post-run decode on the unmounted refiner-pfhb3 genome.
VAL_SEQIDS = (os.environ.get("VAL_SEQIDS", "Pf3D7_01_v3").split()
              or [])
# ablation switches, passed through to train_refiner.py:
#   COND_MODE         full | reduced | none | regularize
#   BALANCE_GENOMES   "1" -> --balance-genomes (per-genome equal
#                     chunk sampling via WeightedRandomSampler)
#   EPOCHS            override the default 20 (trial runs use 10)
COND_MODE = os.environ.get("COND_MODE", "full")
BALANCE_GENOMES = os.environ.get("BALANCE_GENOMES", "") == "1"
EPOCHS = os.environ.get("EPOCHS", "20")

FASTA_GLOBS = ("*_Genome.fasta", "*.fna", "*.fasta")
GFF_GLOBS = ("*.gff", "*.gff3")


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(
            f"missing {pattern}; mounted={os.listdir('/kaggle/input')}")
    return hits[0]


subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)
sys.path.insert(0, str(WORK / "omniphyl/scripts"))
sys.path.insert(0, str(WORK / "omniphyl/src"))

CKPT = Path(find_one("w64_e15_best.pt"))

# pair each fasta with a gff in the SAME containing directory --
# datasets mount under /kaggle/input/datasets/<owner>/<slug>/, so
# depth is not fixed and top-level grouping is wrong
fas = []
for pat in FASTA_GLOBS:
    fas += glob.glob(f"/kaggle/input/**/{pat}", recursive=True)
pairs = []
for fa in sorted(set(fas)):
    d = Path(fa).parent
    gffs = []
    for pat in GFF_GLOBS:
        gffs += sorted(d.glob(pat))
    gffs = [g for g in gffs if "out" not in g.stem.lower()
            and "pred" not in g.stem.lower()]
    if gffs:
        pairs.append((str(d.relative_to("/kaggle/input")),
                      Path(fa), gffs[0]))
if not pairs:
    raise SystemExit("no fasta+gff dataset pairs under /kaggle/input")
pairs = [p for p in pairs if not EXCLUDE or EXCLUDE not in p[0]]
if EXCLUDE:
    print(f"EXCLUDE={EXCLUDE!r}: removed "
          f"{len([p for p in pairs if EXCLUDE in p[0]])} pair(s)",
          flush=True)
tr_pairs = [p for p in pairs if not HELDOUT or HELDOUT not in p[0]]
val_pairs = [p for p in pairs if HELDOUT and HELDOUT in p[0]]
print("data pairs:")
for name, fa, gff in pairs:
    tag = "VAL" if (name, fa, gff) in val_pairs else "train"
    print(f"  [{tag}] {name}: {fa.name} + {gff.name}", flush=True)

# seed every mounted Stage-1 cache into the shared cache dir; the
# filename must be <fasta stem>.res.pkl to be picked up per genome.
# The pf3d7-stage1-cache dataset predates the res.taxon fix
# (tracks_only caches lacked taxonomy calls), so it is NOT seeded --
# it regenerates alongside the other genomes with a real TaxonCall.
CACHE.mkdir(parents=True, exist_ok=True)
for pkl in glob.glob("/kaggle/input/**/*.res.pkl", recursive=True):
    if "pf3d7-stage1-cache" in pkl:
        print(f"skipping stale cache (no taxon): {Path(pkl).name}",
              flush=True)
        continue
    dst = CACHE / Path(pkl).name
    if not dst.exists():
        shutil.copy(pkl, dst)
        print(f"seeded stage-1 cache: {dst.name}", flush=True)

# ---------------- Step 1: stage-1 features + refiner training ---------

cmd = [sys.executable,
       str(WORK / "omniphyl/scripts/train_refiner.py"),
       "--model", str(CKPT),
       "--cache-dir", str(CACHE),
       "--out", str(OUT_PT),
       "--epochs", EPOCHS, "--batch-size", "16", "--num-workers", "4",
       "--cond-mode", COND_MODE]
if BALANCE_GENOMES:
    cmd += ["--balance-genomes"]
print(f"trial config: COND_MODE={COND_MODE} "
      f"BALANCE_GENOMES={BALANCE_GENOMES} EPOCHS={EPOCHS} "
      f"EXCLUDE={EXCLUDE!r} HELDOUT={HELDOUT!r}", flush=True)
for _name, fa, gff in tr_pairs:
    cmd += ["--data", str(fa), str(gff)]
for _name, fa, gff in val_pairs:
    cmd += ["--val-data", str(fa), str(gff)]
if VAL_SEQIDS:
    cmd += ["--val-seqids"] + VAL_SEQIDS
p = subprocess.run(cmd, capture_output=True, text=True, timeout=28800)
log = p.stdout + p.stderr
print(log[-4000:], flush=True)
if p.returncode:
    raise SystemExit(f"train_refiner failed rc={p.returncode}")

epochs = []
best = None
for line in log.splitlines():
    if line.startswith("ep "):
        # "ep 3: train 0.1234 val 0.4567 nucF1 0.901 peakAcc 0.92 ..."
        tok = line.replace(":", "").split()
        rec = {"epoch": int(tok[1])}
        for i, k in enumerate(tok):
            if k in ("train", "val", "nucF1", "peakAcc"):
                rec[k] = float(tok[i + 1])
        epochs.append(rec)
        if best is None or rec["nucF1"] > best["nucF1"]:
            best = rec
out = {"best": best, "epochs": epochs, "decode": {}}

# ---------------- Step 2: held-out decode benchmark -------------------
# Decode the first contig of each val genome (or the first --data
# genome when nothing is held out) from the cached res.pkl -- no
# Stage-1 re-inference -- and compare refiner vs heuristic decoders.

try:
    import pickle  # noqa: E402
    import torch  # noqa: E402
    from eval_genes import (parse_cds, _structures, _metrics, _merge,
                            _overlap_len)  # noqa: E402
    from omniphyl.annotate import (_decode_genes, _exon_bounds,
                                   STOPS)  # noqa: E402
    from omniphyl.refiner import (load_refiner,
                                  decode_with_refiner)  # noqa: E402

    eval_pairs = val_pairs or tr_pairs[:1]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = load_refiner(str(OUT_PT), device=dev)
    for name, fa, gff in eval_pairs:
        pkl_path = CACHE / f"{fa.stem}.res.pkl"
        if not pkl_path.exists():
            print(f"no cache for {name}, skipping decode eval",
                  flush=True)
            continue
        with open(pkl_path, "rb") as fh:
            pkl = pickle.load(fh)
        seqid = pkl["results"][0].contig
        res = pkl["results"][0]
        seq = pkl["contig_seqs"][seqid]
        gcode = pkl.get("gcode", 1)
        del pkl

        ref_rows = {seqid: parse_cds(str(gff))[seqid]}
        ref_ex, ref_tx = _structures(ref_rows)
        r_iv = _merge([iv for _, _, iv in ref_rows[seqid]])

        def score(genes):
            rows = [(f"g{i}", g["strand"], (es, ee + 1))
                    for i, g in enumerate(genes)
                    for es, ee in _exon_bounds(g)]
            ex, tx = _structures({seqid: rows})
            p_iv = _merge([iv for _, _, iv in rows])
            return {
                "n_pred_genes": len(genes),
                "nucleotide": _metrics(
                    _overlap_len(r_iv, p_iv),
                    sum(e - s for s, e in r_iv),
                    sum(e - s for s, e in p_iv)),
                "exon": _metrics(len(ref_ex & ex), len(ref_ex),
                                 len(ex)),
                "transcript": _metrics(len(ref_tx & tx), len(ref_tx),
                                       len(tx)),
            }

        out["decode"][name] = {"seqid": seqid}
        out["decode"][name]["refiner"] = score(decode_with_refiner(
            ref, seq, res, stops=STOPS, euk=True, gcode=gcode,
            device=dev))
        out["decode"][name]["heuristic"] = score(_decode_genes(
            seq, res, stops=STOPS, euk=True, w_ev=0.25,
            min_cds_len=90, coding_threshold=0.45,
            boundary_threshold=0.25, min_intron=35,
            max_intron=5000, intron_hints=[]))
        for tag in ("refiner", "heuristic"):
            print(f"[{name}/{tag}] "
                  f"{out['decode'][name][tag].get('nucleotide')}",
                  flush=True)
except Exception as e:  # decode eval is best-effort; don't fail the run
    out["decode"]["error"] = str(e)
    print(f"decode eval failed: {e}", flush=True)

# ---------------- Step 3: intron/class panel on eval genomes ---------
# Whole-contig intron recall + posterior-signal + gene-class summary
# for every genome that went through decode eval. This is the metric
# that exposed the pvivax intron-state suppression; run it whenever
# eval_refiner_panel.py is in the cloned repo (older refs skip it).

PANEL_PY = WORK / "omniphyl/scripts/eval_refiner_panel.py"
try:
    eval_pairs
    dev
except NameError:
    eval_pairs, dev = val_pairs or tr_pairs[:1], "cpu"
if PANEL_PY.exists():
    for name, fa, gff in eval_pairs:
        pkl_path = CACHE / f"{fa.stem}.res.pkl"
        if not pkl_path.exists():
            continue
        p = subprocess.run(
            [sys.executable, str(PANEL_PY),
             "--ckpt", str(OUT_PT),
             "--panel", f"{name},{pkl_path},{gff}",
             "--device", dev, "--out", "/tmp/_panel.json"],
            capture_output=True, text=True, timeout=7200)
        print(p.stdout[-2000:], flush=True)
        try:
            out.setdefault("panel", {}).update(
                json.loads(Path("/tmp/_panel.json").read_text())
                ["panels"])
        except Exception:
            out.setdefault("panel", {})[name] = {
                "error": (p.stderr or "panel eval failed")[-500:]}

RESULT.write_text(json.dumps(out, indent=2))
print("wrote results.json", flush=True)
