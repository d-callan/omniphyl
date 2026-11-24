"""Kaggle CPU kernel: eval the nopfal trial ckpt on cached pvivax tracks.

Loads the staged pvivax P01 stage-1 res.pkl (pvivax-stage1-cache),
recomputes refiner tracks once for refiner_nopfal.pt on PvP01_12_v2,
then decodes twice:

  default  production decoder gates (comparable to the 9.0/9.5/12.2
           ablation table)
  sweep    best config from the decoder sweep: gamma=0.45 p_int
           temperature, min_intron=25, DEC_FALLBACK_LEN=400,
           DEC_PAIR_LL=-4.5, site_threshold=0.05

Question: did dropping pf3d7 from the corpus shift the learned
intron regime toward vivax-like recovery, beyond what decoder
calibration alone buys?
"""

import glob
import json
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

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
sys.path.insert(0, str(WORK / "omniphyl/scripts"))
sys.path.insert(0, str(WORK / "omniphyl/src"))


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern}; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return hits[0]


from eval_genes import parse_cds                       # noqa: E402
from eval_errors import classify_contig, summarize     # noqa: E402
from omniphyl.annotate import STOPS                    # noqa: E402
import omniphyl.refiner as R                           # noqa: E402
from eval_refiner_panel import _true_introns, _intron_diag  # noqa: E402

RES_PKL = find_one("PlasmoDB-71_PvivaxP01_Genome.res.pkl")
GFF = find_one("PlasmoDB-71_PvivaxP01.gff")
CKPT = find_one("refiner_nopfal.pt")
CONTIG = "PvP01_12_v2"

pkl = pickle.load(open(RES_PKL, "rb"))
res = next(r for r in pkl["results"] if r.contig == CONTIG)
seq = pkl["contig_seqs"][CONTIG].upper()
L = len(seq)
gcode = pkl.get("gcode", 1)
euk = (res.taxon.names.get("domain")
       if getattr(res, "taxon", None) else None) == "Eukaryota"
print(f"{CONTIG} L={L} euk={euk} gcode={gcode}", flush=True)

ref_rows = parse_cds(GFF)
true_intr = _true_introns(ref_rows[CONTIG], L, seq)
print(f"true introns: {len(true_intr)}", flush=True)

ref = R.load_refiner(CKPT, device="cpu")
print(f"ckpt {CKPT} cond_mode={ref.cond_mode}", flush=True)
print("computing refiner tracks...", flush=True)
p_state0, p_site = R.refiner_tracks(ref, seq, res, gcode=gcode,
                                    device="cpu")
pi = np.maximum(p_state0[:, R.ST_INT_P], p_state0[:, R.ST_INT_M])
print(f"tracks done: p_int>0.5 frac={(pi > 0.5).mean():.4f}",
      flush=True)


def scale_pint(ps, gamma):
    if gamma == 1.0:
        return ps
    ps = ps.astype(np.float64).copy()
    for st in (R.ST_INT_P, R.ST_INT_M):
        ps[:, st] = np.clip(ps[:, st], 1e-9, 1) ** gamma
    ps /= ps.sum(1, keepdims=True)
    return ps.astype(np.float32)


def run(tag, gamma=1.0, site_thr=0.05, pair_ll=-3.0, fb_len=0,
        min_intron=35):
    R.DEC_PAIR_LL = pair_ll
    R.DEC_FALLBACK_LEN = fb_len
    ps = scale_pint(p_state0, gamma)
    genes = R.decode_with_refiner(
        None, seq, res, euk=euk, gcode=gcode, starts=("ATG",),
        min_intron=min_intron, site_threshold=site_thr,
        p_state=ps, p_site=p_site, device="cpu")
    pred = [(d, a) for g in genes for d, a in g.get("introns", [])]
    intr = _intron_diag(true_intr, pred, p_state0, p_site)
    s = summarize({CONTIG: classify_contig(CONTIG, ref_rows,
                                           genes, seq)})
    rc = s["ref_classes"]
    print(f"{tag:12s} match {intr['n_matched']}/{intr['n_true']} "
          f"({100 * intr['n_matched'] / intr['n_true']:.1f}%)  "
          f"pred_intr={len(pred)} "
          f"exact={rc.get('exact', 0)} "
          f"wrong={rc.get('wrong_struct', 0)} "
          f"missed={rc.get('missed', 0)} "
          f"causes={intr['unmatched_causes']}", flush=True)
    return {"tag": tag, "intron": intr,
            "ref_classes": rc, "n_pred_introns": len(pred)}


rows = [run("default")]
rows.append(run("sweep-best", gamma=0.45, site_thr=0.05,
                pair_ll=-4.5, fb_len=400, min_intron=25))

json.dump(rows, open(WORK / "nopfal_pvivax.json", "w"), indent=1,
          default=str)
print("wrote nopfal_pvivax.json", flush=True)
