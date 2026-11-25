"""Kaggle CPU kernel: build the Mini-ToL Refiner panel.

Clones the repo and runs scripts/build_refiner_panel.py: downloads the
~17 NCBI assembly zips (GENOME_FASTA+GFF), slices each panel seqid,
caps length, keeps in-cap CDS rows, emits
/kaggle/working/refiner_panel/{train,heldout}/<name>.fna + .gff3 +
panel.json.

Upload the refiner_panel/ dir into daniellecallan/omniphyl-tol-mini
(next dataset version) so tol-refiner-bench can mount it.
"""

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
        "kaggle.com/settings (required), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
os.environ.setdefault("OMP_NUM_THREADS", "4")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)

import glob
_veup = sorted(glob.glob("/kaggle/input/**/pfalciparum.fna",
                         recursive=True))
argv = [sys.executable,
        str(WORK / "omniphyl/scripts/build_refiner_panel.py"),
        "--out", str(WORK / "refiner_panel"),
        "--dl", str(WORK / "_panel_dl")]
if _veup:
    argv += ["--veupath-dir", str(Path(_veup[0]).parent)]
    print(f"veupath overrides: {Path(_veup[0]).parent}")
else:
    print("no veupath mount found -- all entries via NCBI")
subprocess.run(argv, check=True)

print("done -> /kaggle/working/refiner_panel/")
