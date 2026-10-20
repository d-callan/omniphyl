"""Kaggle kernel entrypoint: E1b arm — coding-gated aa fusion.

Identical corpus and config to ablation-a3f (msfcgr + flat + hash,aa on
the frozen 5group-mini corpus); AA_MODE=gated appends the 18 six-frame
stop/ORF/TGA features to the aa input, gates the aa projection with a
z_dna-driven scalar, and moves coding_frac onto a z_dna-only head so the
aa spectrum cannot leak the coding label.

Runs remote/run_pilot.sh with SKIP_FETCH=1 on the mounted corpus.
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
# E1b: gated aa fusion (gate input = [z_dna, detached coding logit,
# 18 frame/stop feats]). Same everything else as a3f.
os.environ.setdefault("AUX", "hash,aa")
os.environ.setdefault("AA_MODE", "gated")
os.environ.setdefault("BACKBONE", "msfcgr")
os.environ.setdefault("HEAD", "flat")
os.environ.setdefault("PRIOR_TAU", "1.0")
os.environ.setdefault("EPOCHS", "10")
os.environ.setdefault("WIDTH", "32")
os.environ.setdefault("DROPOUT", "0.2")
os.environ.setdefault("LABEL_SMOOTHING", "0.05")
os.environ.setdefault("WD", "1e-2")
os.environ.setdefault("VAL_FRAC", "0.1")
os.environ.setdefault("HOLDOUT_GENUS_FRAC", "0.1")
# block-buffered stdout hides progress in the kernel log viewer
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
