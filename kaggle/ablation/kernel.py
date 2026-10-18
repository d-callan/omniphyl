"""Kaggle kernel entrypoint: ablation arm on the frozen 5group-mini corpus.

Mounts the fetch kernel's output (omniphyl-corpus-*.tar.gz) as a dataset
source, extracts it, and runs remote/run_pilot.sh with SKIP_FETCH=1 —
every arm trains on identical fragments/labels; only env knobs differ.

Per-arm config lives in the os.environ block below.  Clone this dir per arm
(A0 flat / A1 tree / ...), flip HEAD or other knobs, push as its own slug.

Requires the fetch kernel's output to have been promoted to the
daniellecallan/omniphyl-5group-mini-corpus dataset (pull the tarball with
`kaggle kernels output`, then `kaggle datasets create` on it — same path
as the v4 corpus dataset).
"""

import glob
import os
import socket
import subprocess
import tarfile

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet + GPU), then re-push.")

WORK = "/kaggle/working"
CORPUS = f"{WORK}/corpus"

# --- extract the frozen corpus tarball mounted from the corpus dataset ---
tarballs = sorted(
    glob.glob("/kaggle/input/omniphyl-5group-mini-corpus/*.tar.gz"))
if not tarballs:
    raise SystemExit(
        "no omniphyl-corpus-*.tar.gz under "
        "/kaggle/input/omniphyl-5group-mini-corpus — promote the fetch "
        "kernel's output to that dataset first")
os.makedirs(CORPUS, exist_ok=True)
with tarfile.open(tarballs[-1]) as tf:  # newest if several
    tf.extractall(CORPUS)
print(f"extracted {tarballs[-1]} -> {CORPUS}: {sorted(os.listdir(CORPUS))}")

os.environ.setdefault("WORKDIR", f"{WORK}/omniphyl-work")
os.environ.setdefault("OUTDIR", WORK)
os.environ.setdefault("SKIP_FETCH", "1")
os.environ.setdefault("CORPUS_DIR", CORPUS)
# --- arm config -------------------------------------------------------
os.environ.setdefault("HEAD", "flat")     # A0 control; A1 -> "tree"
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
