"""Kaggle kernel: end-to-end annotate eval for the multi-genome refiner.

Runs `omniphyl annotate --refiner refiner_multi.pt` on every mounted
genome (fasta + reference CDS gff3 pair), then scores the prediction
two ways per contig from the dumped res.pkl:

  refiner    = res.genes (annotate already ran decode_with_refiner)
  heuristic  = _decode_genes() replayed on the same Stage-1 tracks

Scoring uses eval_genes helpers (_structures/_merge/_overlap_len) at
nucleotide, exon, and transcript level vs the reference GFF. Outputs:

  <fasta_stem>.gff3        refiner predictions (from annotate)
  <fasta_stem>.res.pkl     Stage-1 cache (reusable)
  results.json             metrics per genome/decoder
"""

import glob
import json
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM -- verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
os.environ.setdefault("OMP_NUM_THREADS", "4")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)
SRC = str(WORK / "omniphyl/src")
sys.path.insert(0, str(WORK / "omniphyl/scripts"))
sys.path.insert(0, SRC)


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern} under /kaggle/input; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return hits[0]


MODEL = find_one("w64_e15_best.pt")
REFINER = find_one("refiner_multi.pt")

# every mounted fasta paired with the gff in its own directory
pairs = []
for fa in sorted(glob.glob("/kaggle/input/**/*.f*",
                           recursive=True)):
    if not fa.lower().endswith((".fasta", ".fna", ".fa")):
        continue
    gffs = glob.glob(os.path.join(os.path.dirname(fa), "*.gff*"))
    if gffs:
        pairs.append((fa, gffs[0]))
if not pairs:
    raise SystemExit("no fasta+gff pairs mounted")
for fa, gff in pairs:
    print(f"  {Path(fa).name} + {Path(gff).name}", flush=True)

# ---------------- Step 1: annotate (subprocess per genome) -------------
for fa, _gff in pairs:
    stem = Path(fa).stem
    spec = json.dumps({
        "model_path": MODEL, "fasta_path": fa,
        "out_prefix": str(WORK / stem),
        "device": "auto", "batch_size": 256,
        "refiner_path": REFINER, "verbose": True})
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys, json\n"
         "from pathlib import Path\n"
         "from omniphyl.annotate import run_annotate\n"
         "kw = json.loads(sys.argv[1])\n"
         "kw['out_prefix'] = Path(kw['out_prefix'])\n"
         "run_annotate(**kw)\n",
         spec],
        env={**os.environ, "OMNIPHYL_DUMP_RES": "1",
             "PYTHONPATH": SRC + os.pathsep
             + os.environ.get("PYTHONPATH", "")})
    if r.returncode != 0 or not Path(f"{WORK / stem}.res.pkl").exists():
        print(f"{stem}: annotate FAILED rc={r.returncode} -- skipping",
              flush=True)

# ---------------- Step 2: score refiner vs heuristic -------------------
from eval_genes import (parse_cds, _structures, _metrics, _merge,
                        _overlap_len)  # noqa: E402
from omniphyl.annotate import _decode_genes, _exon_bounds, STOPS  # noqa


def genome_score(ref_rows, genes_by_seqid):
    """ref_rows: parse_cds dict. genes_by_seqid: seqid -> gene dicts."""
    seqids = set(ref_rows) | set(genes_by_seqid)
    ref_iv, pred_iv = {}, {}
    pred_rows = {}
    n_genes = 0
    for sq in seqids:
        genes = genes_by_seqid.get(sq, [])
        n_genes += len(genes)
        rows = [(f"{sq}_g{i}", g["strand"], (es, ee + 1))
                for i, g in enumerate(genes)
                for es, ee in _exon_bounds(g)]
        pred_rows[sq] = rows
        ref_iv[sq] = _merge([iv for _, _, iv in ref_rows.get(sq, [])])
        pred_iv[sq] = _merge([iv for _, _, iv in rows])
    ref_ex, ref_tx = _structures({s: ref_rows.get(s, [])
                                for s in seqids})
    pred_ex, pred_tx = _structures(pred_rows)
    hit = sum(_overlap_len(ref_iv[s], pred_iv[s]) for s in seqids)
    rb = sum(e - s for iv in ref_iv.values() for s, e in iv)
    pb = sum(e - s for iv in pred_iv.values() for s, e in iv)
    return {"n_pred_genes": n_genes,
            "nucleotide": _metrics(hit, rb, pb),
            "exon": _metrics(len(ref_ex & pred_ex), len(ref_ex),
                             len(pred_ex)),
            "transcript": _metrics(len(ref_tx & pred_tx),
                                   len(ref_tx), len(pred_tx))}


out = {}
for fa, gff in pairs:
    stem = Path(fa).stem
    pk = Path(f"{WORK / stem}.res.pkl")
    if not pk.exists():
        continue
    with open(pk, "rb") as fh:
        pkl = pickle.load(fh)
    gcode = pkl.get("gcode", 1)
    ref_rows = parse_cds(gff)

    refiner_genes = {res.contig: (res.genes or [])
                     for res in pkl["results"]}
    heur_genes = {}
    for res in pkl["results"]:
        seq = pkl["contig_seqs"][res.contig]
        heur_genes[res.contig] = _decode_genes(
            seq, res, stops=STOPS, euk=True, w_ev=0.25,
            min_cds_len=90, coding_threshold=0.45,
            boundary_threshold=0.25, min_intron=35,
            max_intron=5000, intron_hints=[])
    del pkl

    out[stem] = {
        "refiner": genome_score(ref_rows, refiner_genes),
        "heuristic": genome_score(ref_rows, heur_genes),
    }
    for tag in ("refiner", "heuristic"):
        m = out[stem][tag]
        print(f"[{stem}/{tag}] genes={m['n_pred_genes']} "
              f"nuc F1={m['nucleotide']['f1']:.3f} "
              f"exon F1={m['exon']['f1']:.3f} "
              f"tx F1={m['transcript']['f1']:.3f}", flush=True)

with open(WORK / "results.json", "w") as fh:
    json.dump(out, fh, indent=2)
print("done -> results.json", flush=True)
