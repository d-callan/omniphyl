"""Kaggle kernel entrypoint: E3 arm — region-composition head.

Base config matches the a3f/e1 arm (msfcgr + flat + hash,aa) so the only
delta is REGION_HEAD=1: a 4-way soft head predicting
[coding, intron, rna, intergenic] fractions with 3x weight on
non-coding windows (91% coding imbalance in annotated corpora).

Requires the annot-mini corpus (fetch-annot-mini output promoted to a
Kaggle Dataset): only the refseq_gff tier emits intron_frac/rna_frac.

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
# E3: region-composition head; everything else identical to a3f.
os.environ.setdefault("REGION_HEAD", "1")
os.environ.setdefault("AUX", "hash,aa")
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
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
