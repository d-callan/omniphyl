"""Kaggle CPU kernel: re-annotate reconstructed ToL-mini labels rows.

The timeout-harvested corpus carries ~84k labels rows written with
label_source=reconstructed -- real fids/coords/taxonomy but NA for every
annotation-derived stat (coding_frac, boundary, n_boundaries,
boundary_offset, intron_frac, rna_frac, active_frame, struct_flags,
domain_ids). This kernel re-derives them from the same NCBI GFF /
VEuPathDB GenArk sources and code paths as the original emit
(scripts/reannotate_labels.py).

Input : daniellecallan/omniphyl-tol-mini-partial (labels.tsv, frags.fna)
        + reannotate_map.tsv (accession->ftp_path, shipped in this dir)
Output: /kaggle/working/labels_patched.tsv + labels_patched.txt
        (.txt copy because Kaggle capture has dropped .tsv outputs)
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
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)

# zip-mode dataset uploads mount as archives, not extracted files
import zipfile
for z in Path("/kaggle/input").rglob("*.zip"):
    with zipfile.ZipFile(z) as zf:
        zf.extractall(z.parent)

lab_src = next(Path("/kaggle/input").rglob("labels.tsv"), None)
if lab_src is None:
    raise SystemExit("labels.tsv not found under /kaggle/input: "
                     f"{[str(p) for p in Path('/kaggle/input').rglob('*')]}")
CORP = WORK / "corpus"
CORP.mkdir(exist_ok=True)
(CORP / "_ncbi_cache").mkdir(exist_ok=True)
shutil.copy2(lab_src, CORP / "labels.tsv")

MAP = next((p for p in (
        Path(__file__).resolve().parent / "reannotate_map.tsv",
        Path.cwd() / "reannotate_map.tsv",
        WORK / "reannotate_map.tsv",
        *Path("/kaggle/input").rglob("reannotate_map.tsv"))
        if p.exists()), None)
if MAP is None:
    raise SystemExit("reannotate_map.tsv not bundled with the kernel")

subprocess.run([
    sys.executable,
    str(WORK / "omniphyl/scripts/reannotate_labels.py"),
    "--corpus", str(CORP),
    "--map", str(MAP),
    "--out", str(WORK / "labels_patched.tsv"),
    "--work", str(WORK / "reannotate_work"),
], check=True)

# .tsv output has been dropped by Kaggle's capture before; belt+braces.
shutil.copy2(WORK / "labels_patched.tsv", WORK / "labels_patched.txt")
print("done -> labels_patched.tsv + labels_patched.txt")
