"""Kaggle kernel entrypoint: ablation arm on the frozen 5group-mini corpus.

Mounts the daniellecallan/omniphyl-5group-mini-corpus dataset (Kaggle
unpacks the uploaded tarball on ingest, so the mount is a plain dir of
fragments.fna/labels.tsv/manifest.json) and runs remote/run_pilot.sh
with SKIP_FETCH=1 — every arm trains on identical fragments/labels;
only env knobs differ.

Per-arm config lives in the os.environ block below.  Clone this dir per arm
(A0 flat / A1 tree / ...), flip HEAD or other knobs, push as its own slug.
"""

import glob
import os
import socket
import subprocess
import sys

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet + GPU), then re-push.")

WORK = "/kaggle/working"
# dataset_sources mount under /kaggle/input/datasets/<user>/<slug> on
# script kernels (not /kaggle/input/<slug>) — locate it by content.
hits = glob.glob("/kaggle/input/**/fragments.fna", recursive=True)
if not hits:
    print("/kaggle/input contents:", file=sys.stderr)
    for root, dirs, files in os.walk("/kaggle/input"):
        print(f"  {root}: {dirs} {files}", file=sys.stderr)
    raise SystemExit("no fragments.fna under /kaggle/input — "
                     "corpus dataset not mounted?")
CORPUS = os.path.dirname(hits[0])
if not os.path.exists(f"{CORPUS}/labels.tsv"):
    raise SystemExit(f"missing labels.tsv next to {hits[0]}")
print(f"corpus mount: {CORPUS}: {sorted(os.listdir(CORPUS))}")

os.environ.setdefault("WORKDIR", f"{WORK}/omniphyl-work")
os.environ.setdefault("OUTDIR", WORK)
os.environ.setdefault("SKIP_FETCH", "1")
os.environ.setdefault("CORPUS_DIR", CORPUS)
# --- arm config -------------------------------------------------------
# A3f: missing factorial cell — aux branches under the FLAT head.
# A2->A3f vs A2.5->A3 crosses aux x head so the aux contribution is
# measured both with and without chain lock-out.
os.environ.setdefault("AUX", "hash,aa")      # 11-mer bag + AA 3-mer
os.environ.setdefault("BACKBONE", "msfcgr")
os.environ.setdefault("HEAD", "flat")
os.environ.setdefault("PRIOR_TAU", "1.0")
os.environ.setdefault("EPOCHS", "10")
os.environ.setdefault("WIDTH", "32")
os.environ.setdefault("DROPOUT", "0.2")
os.environ.setdefault("LABEL_SMOOTHING", "0.05")
os.environ.setdefault("WD", "1e-2")
os.environ.setdefault("VAL_FRAC", "0.1")
os.environ.setdefault("HOLDOUT_GENUS_FRAC", "0.1")  # clade holdout per plan
# block-buffered stdout hides progress in the kernel log viewer
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
