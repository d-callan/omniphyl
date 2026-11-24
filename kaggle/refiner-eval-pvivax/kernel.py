"""Kaggle kernel: fixed-panel refiner eval on P. vivax P01.

Runs Stage-1 (tracks_only) once via train_refiner._stage1_cache -- the
contig-split subprocess path -- then eval_refiner_panel.py for each
mounted refiner checkpoint:

  v5_full   baseline full-cond multi-genome refiner (v5)
  redbal    COND_MODE=reduced + genome-balanced sampling trial
  reduced   COND_MODE=reduced trial

Outputs:
  refiner_cache/PlasmoDB-71_PvivaxP01_Genome.res.pkl   reusable cache
  results.json                                       panel report per ckpt
"""

import glob
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


MODEL = find_one("w64_e15_best.pt")
FA = find_one("PlasmoDB-71_PvivaxP01_Genome.fasta")
GFF = find_one("PlasmoDB-71_PvivaxP01.gff")

CKPTS = {
    "v5_full": find_one("refiner_multi.pt"),
    "redbal": find_one("refiner_redbal.pt"),
    "reduced": find_one("refiner_reduced.pt"),
}
for tag, ck in CKPTS.items():
    print(f"{tag}: {ck}", flush=True)

# ---------------- Step 1: Stage-1 cache (contig-split subprocesses) ----
import train_refiner  # noqa: E402

args = SimpleNamespace(window=500, stride=100, bin_width=0,
                       stage1_batch=256, device="auto")
CACHE = WORK / "refiner_cache"
res_pkl = CACHE / (Path(FA).stem + ".res.pkl")
if train_refiner._stage1_cache(Path(FA), MODEL, CACHE, args) is None:
    raise SystemExit("stage-1 failed -- see log")

# ---------------- Step 2: panel eval per ckpt --------------------------
results = {}
for tag, ck in CKPTS.items():
    print(f"ckpt {tag}: {ck}", flush=True)
    out = WORK / f"_panel_{tag}.json"
    subprocess.run(
        [sys.executable,
         str(WORK / "omniphyl/scripts/eval_refiner_panel.py"),
         "--ckpt", ck,
         "--panel", f"pvivax,{res_pkl},{GFF}",
         "--device", "cuda",
         "--out", str(out)],
        env={**os.environ,
             "PYTHONPATH": str(WORK / "omniphyl/scripts") + os.pathsep
             + str(WORK / "omniphyl/src")})
    if out.exists():
        results[tag] = json.load(open(out))

json.dump(results, open(WORK / "results.json", "w"), indent=1)
print("wrote results.json", flush=True)
