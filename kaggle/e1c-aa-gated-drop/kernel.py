"""Kaggle kernel entrypoint: E1c arm — gated aa fusion + branch dropout.

Identical to e1b-aa-gated except AA_DROP=0.2: during training the gated
aa branch is zeroed for a Bernoulli(0.2) fraction of each batch, forcing
the classifier to keep a DNA-only fallback pathway and preventing the
aa spectrum from dominating the fused representation.

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
# E1c: gated aa fusion + 20% branch dropout. AUX=aa only — hash was
# dropped from the plan.
os.environ.setdefault("AUX", "aa")
os.environ.setdefault("AA_MODE", "gated")
os.environ.setdefault("AA_DROP", "0.2")
os.environ.setdefault("BACKBONE", "msfcgr")
os.environ.setdefault("HEAD", "flat")
os.environ.setdefault("FLAT_PRIOR_TAU", "1.0")
os.environ.setdefault("FRAG_VAL", "0.1")
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

# --- E1 gate: --zero-aux aa eval ---------------------------------------
# E1c's hypothesis is that the 20% branch dropout forces a DNA-only
# pathway; the zeroed-aa eval is the arm's primary result, not a side
# check. Pass gate: >=0.65 single-frag / >=0.80 M20 Domain accuracy.
REPO = f"{WORK}/omniphyl"
CKPT = f"{WORK}/omniphyl-work/run/best.pt"
for frag, lab in [
        (f"{WORK}/omniphyl-work/screened/fragments_screened.fna",
         f"{WORK}/omniphyl-work/screened/labels_screened.tsv"),
        (f"{CORPUS}/fragments_screened.fna",
         f"{CORPUS}/labels_screened.tsv"),
        (f"{CORPUS}/fragments.fna", f"{CORPUS}/labels.tsv")]:
    if os.path.exists(frag) and os.path.exists(lab):
        break
else:
    raise SystemExit("no usable fragments/labels for zero-aux eval")
outdir = f"{WORK}/omniphyl-work/run/eval_zeroaa"
print(f"=== eval --zero-aux aa -> {outdir} ===", flush=True)
subprocess.run(
    [sys.executable, f"{REPO}/scripts/eval_pilot.py",
     "--model", CKPT, "--fragments", frag, "--labels", lab,
     "--outdir", outdir, "--num-workers", "4",
     "--val-only", "--zero-aux", "aa"],
    check=True)
