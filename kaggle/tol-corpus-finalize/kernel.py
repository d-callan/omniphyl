"""Kaggle CPU kernel: finalize the ToL-mini corpus for stage-1 bench.

Mounts daniellecallan/omniphyl-tol-mini-partial (fragments.fna +
patched labels.tsv -- run ONLY after tol-reannotate's output is folded
in) and produces the artifacts tol-stage1-bench expects:

  fragments_screened.fna / labels_screened.tsv   contaminant screen
  corpus.packed/                                 mmap binary store
  screen_manifest.json                           rejection log

Promote /kaggle/working/finalized/ to the `omniphyl-tol-mini` dataset
so tol-stage1-bench can mount it.
"""

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
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
OUT = WORK / "finalized"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)

# zip-mode dataset uploads mount as archives, not extracted files
import zipfile
for z in Path("/kaggle/input").rglob("*.zip"):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(z.parent)

def _find(name):
    hits = [p for p in Path("/kaggle/input").rglob(name)
            if p.name == name]
    if not hits:
        raise SystemExit(f"{name} not found under /kaggle/input")
    return hits[0]

DATA = WORK / "corpus"
DATA.mkdir(exist_ok=True)
shutil.copy2(_find("fragments.fna"), DATA / "fragments.fna")
shutil.copy2(_find("labels.tsv"), DATA / "labels.tsv")

subprocess.run([sys.executable,
                str(WORK / "omniphyl/scripts/screen_contaminants.py"),
                "--fragments", str(DATA / "fragments.fna"),
                "--labels", str(DATA / "labels.tsv"),
                "--outdir", str(DATA)], check=True)

fna = DATA / "fragments_screened.fna"
lab = DATA / "labels_screened.tsv"
if not (fna.exists() and lab.exists()):
    raise SystemExit("screen did not produce screened fragments/labels")

subprocess.run([sys.executable,
                str(WORK / "omniphyl/scripts/pack_corpus.py"),
                "--fragments", str(fna), "--labels", str(lab),
                "--out", str(OUT / "corpus.packed")], check=True)

for p in (fna, lab):
    shutil.copy2(p, OUT / p.name)
for extra in ("screen_manifest.json", "manifest.json"):
    p = DATA / extra
    if p.exists():
        shutil.copy2(p, OUT / p.name)

# transient screened/raw copies already live under finalized/ — free the
# workdir so the output capture isn't carrying them twice
shutil.rmtree(DATA, ignore_errors=True)
print("done -> /kaggle/working/finalized/")
