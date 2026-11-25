"""Kaggle T4 kernel: Mini-ToL Stage-1 benchmark (tol-arch §4.2).

Mounts the `omniphyl-tol-mini` dataset produced by tol-mini-corpus
(corpus.packed + screened fragments/labels) and runs
`scripts/tol_arch_bench.py stage1`:

  arm baseline  v6-style arch (resnet w64, gated-aa, flat heads)
  arm tol       locked ToL arch (msfcgr w96, dna_stats, aa_extra_v2,
                replicon+struct+domain heads, annot-ctx, bottleneck)
  gates         S1-A selective FDR + mammalian/plant species acc +
                human-window screen retention
                S1-B synthetic C_species=120k head_bottleneck VRAM
                probe (<16 GB at batch 512 bf16)
                S1-C gap-lineage reconciliation consistency

Outputs to /kaggle/working:
  stage1/stage1_mini_best.pt      ToL-arm checkpoint (refiner bench
                                  mounts this via a dataset)
  stage1/stage1_gate_report.json  gate verdicts + arm summaries
"""

import glob
import os
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

# zip-mode dataset uploads mount as archives, not extracted files;
# corpus.packed additionally ships as a nested zip inside
import zipfile
_done = set()
for _ in range(3):
    zips = [z for z in Path("/kaggle/input").rglob("*.zip")
            if z not in _done]
    if not zips:
        break
    for z in zips:
        _done.add(z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(z.parent)


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


packed = find_dir("corpus.packed")
fna = find_one("fragments_screened.fna")
lab = find_one("labels_screened.tsv")

subprocess.run(
    [sys.executable,
     str(WORK / "omniphyl/scripts/tol_arch_bench.py"), "stage1",
     "--packed", str(packed), "--labels", str(lab),
     "--fragments", str(fna),
     "--out", str(WORK / "stage1"),
     "--epochs", os.environ.get("EPOCHS", "5"),
     "--batch-size", "256", "--device", "cuda",
     "--probe"],
    check=True)
print("done -> /kaggle/working/stage1/")
