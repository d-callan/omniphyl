"""Kaggle T4 kernel: train OmniPhylRefiner v2 + v1 on cached stage-1
tracks.

Parallel arm of the split bench: runs while tol-panel-cache v2
annotates the heldout subset, saving ~2h wall-clock.  Heldout caches
don't exist yet, so --val-data is omitted: _split_items carves a
deterministic val_frac share of the 13 train genomes instead of the
intended heldout-val.  Deviation is identical across both arms, so the
v2-vs-v1 gate comparison stays fair.

Inputs:
  daniellecallan/omniphyl-tol-mini        refiner_panel/train/*
  daniellecallan/omniphyl-tol-stage1      stage1_mini_best.pt
  daniellecallan/omniphyl-stage1-cache    stage1_cache/*.res.pkl
                                        (stride 500, train subset)
Output:
  refiner_v2.pt, refiner_v1.pt -> ds for the bench kernel's
  --v1-ckpt/--v2-ckpt
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"

os.environ.setdefault("OMP_NUM_THREADS", "4")

try:
    import requests  # noqa: F401
    requests.get("https://github.com", timeout=10)
except Exception:
    raise SystemExit(
        "NO INTERNET on this kernel VM -- verify phone number at "
        "kaggle.com/settings (required), then re-push.")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)
sys.path.insert(0, str(WORK / "omniphyl" / "scripts"))

from tol_arch_bench import _panel_pairs  # noqa: E402


def _find(pattern: str) -> Path:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern}; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return Path(hits[0])


panel = _find("refiner_panel")
while not (panel / "train").is_dir():
    panel = panel.parent
s1 = _find("stage1_mini_best.pt")

train = _panel_pairs(panel / "train")
cache = WORK / "stage1_cache"
cache.mkdir(exist_ok=True)
for pk in glob.glob("/kaggle/input/**/*.res.pkl", recursive=True):
    dst = cache / Path(pk).name
    if not dst.exists():
        shutil.copy2(pk, dst)
print(f"train={len(train)} cached={len(list(cache.glob('*.res.pkl')))}")

base = [sys.executable,
        str(WORK / "omniphyl/scripts/train_refiner.py"),
        "--model", str(s1), "--cache-dir", str(cache),
        "--epochs", "10", "--batch-size", "4",
        "--balance-genomes", "--cond-mode", "reduced",
        "--amp", "fp16", "--device", "cuda"]
data = []
for _n, fna, gff in train:
    data += ["--data", str(fna), str(gff)]

subprocess.run(base + ["--out", str(WORK / "refiner_v2.pt")] + data,
               check=True)
subprocess.run(base + ["--out", str(WORK / "refiner_v1.pt"),
                       "--v1-arch"] + data, check=True)
print("done -> /kaggle/working/refiner_v{1,2}.pt")
