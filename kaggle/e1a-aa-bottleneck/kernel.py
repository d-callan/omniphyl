"""Kaggle kernel entrypoint: E1a arm — bottlenecked aa fusion.

Identical corpus and config to ablation-a3f (msfcgr + flat + hash,aa on
the frozen 5group-mini corpus); the only difference is AA_MODE=bottleneck,
which compresses the 9261-d aa tripeptide spectrum through a 64->128
bottleneck instead of the plain 9261->128 projection.

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
# E1a: aa bottleneck (9261 -> 64 -> 128). AUX=aa only — the hash
# EmbeddingBag was dropped from the plan (72% of params, memorized
# training genomes, starved msfcgr). FLAT_PRIOR_TAU drives the
# logit-adjusted CE on flat heads (PRIOR_TAU is tree-head-only).
os.environ.setdefault("AUX", "aa")
os.environ.setdefault("AA_MODE", "bottleneck")
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
# The arm only passes if accuracy survives with the aa branch zeroed —
# that is the whole point of bottlenecking/gating (msfcgr must not be
# starved). Prefer the screened corpus the runner actually trained on.
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
