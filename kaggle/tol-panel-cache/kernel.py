"""Stage-1 track cache for the refiner panel (tol-refiner-bench).

Runs _stage1_caches over a SUBSET of the panel genomes and writes one
res.pkl per genome to /kaggle/working/stage1_cache/.  The bench kernel
mounts the resulting dataset via --cache-dir and skips straight to
refiner training -- this split exists because stride-100 annotate
needed ~6h per 30Mb genome (~40h total) vs the ~12h kernel cap.

Inputs:
  daniellecallan/omniphyl-tol-mini        refiner_panel/{train,heldout}
  daniellecallan/omniphyl-tol-stage1      stage1_mini_best.pt
Output:
  stage1_cache/<name>.res.pkl per genome in SUBSET
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

SUBSET = "train"          # "train" | "heldout" -- set per kernel push
STRIDE = 500              # stage-1 track stride (bp): 100 = 5 votes/bin
                          # but ~6h per 30Mb genome; 500 = 1 vote, fits
                          # the ~12h kernel cap
WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"

os.environ.setdefault("OMP_NUM_THREADS", "4")

try:
    import requests  # noqa: F401 -- clone needs outbound HTTPS
    requests.get("https://github.com", timeout=10)
except Exception:
    raise SystemExit(
        "NO INTERNET on this kernel VM -- verify phone number at "
        "kaggle.com/settings (required), then re-push.")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)
sys.path.insert(0, str(WORK / "omniphyl" / "scripts"))

from tol_arch_bench import _panel_pairs, _stage1_caches  # noqa: E402


def _find(pattern: str) -> Path:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern}; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return Path(hits[0])


panel = _find("refiner_panel")          # a dir containing train/ heldout/
while not (panel / "train").is_dir():   # glob may land on an ancestor
    panel = panel.parent
s1 = _find("stage1_mini_best.pt")

pairs = _panel_pairs(panel / SUBSET)
print(f"{SUBSET}: {len(pairs)} genomes")
cache = WORK / "stage1_cache"
pks = _stage1_caches(pairs, s1, cache, "cuda", stride=STRIDE)
print(f"cached {len(pks)}/{len(pairs)} -> {cache}")
if len(pks) < len(pairs):
    missing = [n for n, _, _ in pairs if n not in pks]
    print(f"MISSING: {missing}")
