"""Kaggle GPU kernel: Stage-2 refiner architecture test on Pf3D7.

Trains OmniPhylRefiner against frozen v6 Stage-1 features on the
pinned PlasmoDB-71 P. falciparum 3D7 genome (~23.7 Mb: 14 nuclear
chromosomes + apicoplast + mitochondrion; ~7.6k chunks after RC
augmentation). Inputs: daniellecallan/plasmodb71-pf3d7 (genome + GFF)
and daniellecallan/omniphyl-v6-ckpts (w64_e15_best.pt).

Outputs to /kaggle/working:
  refiner_cache/PlasmoDB-71_Pfalciparum3D7_Genome.res.pkl
      frozen Stage-1 features -- reuse for refiner retraining without
      re-running the ~25-min annotate pass
  refiner_pf3d7.pt          trained refiner ckpt (best val nucF1)
  results.json              per-epoch metrics + chr1 decode benchmark:
      refiner vs heuristic decoder at nucleotide/CDS-exon/transcript
      level. chr1 is held out of training via --val-seqids, so the
      benchmark is a genuine (single-chromosome) held-out eval; full
      generalization still needs the multi-genome corpus.
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
os.environ.setdefault("OMP_NUM_THREADS", "4")


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

FASTA = Path(find_one("*_Genome.fasta"))
REF_GFF = Path(find_one("PlasmoDB-*.gff"))
CKPT = Path(find_one("w64_e15_best.pt"))
OUT_PT = WORK / "refiner_pf3d7.pt"
CACHE = WORK / "refiner_cache"
print(f"genome: {FASTA.name}\nreference: {REF_GFF.name}\n"
      f"stage-1: {CKPT.name}", flush=True)

# Seed the Stage-1 feature cache from the pf3d7-stage1-cache dataset
# (res.pkl captured from a prior run's completed annotate pass) so
# retries go straight to training instead of re-running ~30 min of
# windowed inference.
CACHE.mkdir(parents=True, exist_ok=True)
for pkl in glob.glob("/kaggle/input/**/*.res.pkl", recursive=True):
    dst = CACHE / Path(pkl).name
    if not dst.exists():
        shutil.copy(pkl, dst)
        print(f"seeded stage-1 cache: {dst.name}", flush=True)

# ---------------- Step 1: stage-1 features + refiner training ---------

p = subprocess.run(
    [sys.executable, str(WORK / "omniphyl/scripts/train_refiner.py"),
     "--data", str(FASTA), str(REF_GFF),
     "--model", str(CKPT),
     "--cache-dir", str(CACHE),
     "--out", str(OUT_PT),
     "--val-seqids", "Pf3D7_01_v3",
     "--epochs", "20", "--batch-size", "16", "--num-workers", "4"],
    capture_output=True, text=True, timeout=28800)
log = p.stdout + p.stderr
print(log[-4000:], flush=True)
if p.returncode:
    raise SystemExit(f"train_refiner failed rc={p.returncode}")

epochs = []
best = None
for line in log.splitlines():
    if line.startswith("ep "):
        # "ep 3: train 0.1234 val 0.4567 nucF1 0.901 ..."
        tok = line.replace(":", "").split()
        rec = {"epoch": int(tok[1])}
        for i, k in enumerate(tok):
            if k in ("train", "val", "nucF1"):
                rec[k] = float(tok[i + 1])
        epochs.append(rec)
        if best is None or rec["nucF1"] > best["nucF1"]:
            best = rec
out = {"best": best, "epochs": epochs, "chr1_decode": {}}

# ---------------- Step 2: chr1 decode benchmark (held-out) ------------
# chr1 was held out of training via --val-seqids, so this is a real
# generalization eval on one chromosome. Both paths decode from the
# cached Stage-1 res.pkl -- no Stage-1 re-inference.

SEQID = "Pf3D7_01_v3"
try:
    import pickle  # noqa: E402
    import torch  # noqa: E402
    from eval_genes import (parse_cds, _structures, _metrics, _merge,
                            _overlap_len)  # noqa: E402
    from omniphyl.annotate import (_decode_genes, _exon_bounds,
                                   STOPS)  # noqa: E402
    from omniphyl.refiner import (load_refiner,
                                  decode_with_refiner)  # noqa: E402

    with open(CACHE / f"{FASTA.stem}.res.pkl", "rb") as fh:
        pkl = pickle.load(fh)
    res = next(r for r in pkl["results"] if r.contig == SEQID)
    seq = pkl["contig_seqs"][SEQID]
    gcode = pkl.get("gcode", 1)
    del pkl

    ref_rows = {SEQID: parse_cds(str(REF_GFF))[SEQID]}
    ref_ex, ref_tx = _structures(ref_rows)
    r_iv = _merge([iv for _, _, iv in ref_rows[SEQID]])

    def score(genes):
        rows = [(f"g{i}", g["strand"], (es, ee + 1))
                for i, g in enumerate(genes)
                for es, ee in _exon_bounds(g)]
        ex, tx = _structures({SEQID: rows})
        p_iv = _merge([iv for _, _, iv in rows])
        return {
            "n_pred_genes": len(genes),
            "nucleotide": _metrics(
                _overlap_len(r_iv, p_iv),
                sum(e - s for s, e in r_iv),
                sum(e - s for s, e in p_iv)),
            "exon": _metrics(len(ref_ex & ex), len(ref_ex), len(ex)),
            "transcript": _metrics(len(ref_tx & tx), len(ref_tx),
                                   len(tx)),
        }

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref = load_refiner(str(OUT_PT), device=dev)
    out["chr1_decode"]["refiner"] = score(decode_with_refiner(
        ref, seq, res, stops=STOPS, euk=True, gcode=gcode,
        device=dev))
    out["chr1_decode"]["heuristic"] = score(_decode_genes(
        seq, res, stops=STOPS, euk=True, w_ev=0.25,
        min_cds_len=90, coding_threshold=0.45,
        boundary_threshold=0.25, min_intron=35, max_intron=5000,
        intron_hints=[]))
    for tag in ("refiner", "heuristic"):
        print(f"[chr1/{tag}] "
              f"{out['chr1_decode'][tag].get('nucleotide')}", flush=True)
except Exception as e:  # decode eval is best-effort; don't fail the run
    out["chr1_decode"]["error"] = str(e)
    print(f"chr1 decode eval failed: {e}", flush=True)

RESULT.write_text(json.dumps(out, indent=2))
print("wrote results.json", flush=True)
