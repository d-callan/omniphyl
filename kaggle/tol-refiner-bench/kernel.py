"""Kaggle T4 kernel: Mini-ToL Refiner v2-vs-v1 benchmark (§4.2).

Mounts:
  daniellecallan/omniphyl-tol-mini        refiner_panel/ from
                                        tol-mini-corpus
  daniellecallan/omniphyl-tol-stage1      stage1_mini_best.pt from
                                        tol-stage1-bench
  daniellecallan/omniphyl-stage1-cache    stage1_cache/*.res.pkl
                                        from tol-panel-cache

Runs `scripts/tol_arch_bench.py refiner`:
  1. loads precomputed Stage-1 .res.pkl tracks (tol-panel-cache);
     annotates any genomes missing a cache in-run
  2. trains OmniPhylRefiner v2 (default arch) and v1 (--v1-arch),
     10 epochs, --balance-genomes, fp16, batch 4
  3. eval_refiner_panel decode on the held-out species
  4. emits refiner_gate_report.json with gates S2-A/B/C
"""

import glob
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
os.environ.setdefault("OMP_NUM_THREADS", "4")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)


def find_one(pattern: str) -> Path:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern}; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return Path(hits[0])


def find_dir(name: str) -> Path:
    hits = sorted(p for p in glob.glob("/kaggle/input/**",
                                       recursive=True)
                  if Path(p).name == name and Path(p).is_dir())
    if not hits:
        raise SystemExit(f"missing dir {name}")
    return Path(hits[0])


panel = find_dir("refiner_panel")
s1 = find_one("stage1_mini_best.pt")

# precomputed stage-1 res.pkl caches from tol-panel-cache runs
# (mounted ds is read-only; copy into working so _stage1_cache can
# also write any genomes missing from the cache)
cache = WORK / "stage1_cache"
cache.mkdir(exist_ok=True)
for pk in glob.glob("/kaggle/input/**/*.res.pkl", recursive=True):
    dst = cache / Path(pk).name
    if not dst.exists():
        shutil.copy2(pk, dst)
print(f"stage-1 cache mounted: {len(list(cache.glob('*.res.pkl')))} "
      f"genomes")

argv = [sys.executable,
        str(WORK / "omniphyl/scripts/tol_arch_bench.py"), "refiner",
        "--panel-dir", str(panel), "--stage1-model", str(s1),
        "--out", str(WORK / "refiner"),
        "--cache-dir", str(cache),
        "--track-stride", "500",
        "--epochs", os.environ.get("EPOCHS", "10"),
        "--batch-size", "4", "--cond-mode", "reduced",
        "--amp", "fp16", "--device", "cuda"]
subprocess.run(argv, check=True)
print("done -> /kaggle/working/refiner/")
