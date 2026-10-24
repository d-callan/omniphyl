"""Kaggle kernel entrypoint: E2 arm — variable read-length training.

E1 winner (gated aa + 20% branch dropout; flip AA_MODE/AA_DROP env if the
E1 arms pick a different winner) plus VAR_LEN_PROB=0.35, MIN_TRAIN_LEN=150:
during training 35% of fragments get a stochastic crop (70% uniform
150-500bp sub-window, 30% unmerged PE150 with an N junction).

After run_pilot.sh finishes (full-length eval included), this kernel
runs eval_pilot.py on the trained checkpoint at each test length:
SE150, SE250, PE150, PE250 — outputs land under run/eval_len*/.
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
# E2: var-len training on top of the E1 winner base, gated+drop.
os.environ.setdefault("AUX", "aa")
os.environ.setdefault("AA_MODE", "gated")
os.environ.setdefault("AA_DROP", "0.2")
os.environ.setdefault("MIN_TRAIN_LEN", "150")
os.environ.setdefault("VAR_LEN_PROB", "0.35")
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

# --- multi-length eval sweep ------------------------------------------
# run_pilot.sh already wrote the full-length (500bp) eval to run/eval.
# Benchmark the same checkpoint at each simulated read length.
REPO = f"{WORK}/omniphyl"
CKPT = f"{WORK}/omniphyl-work/run/best.pt"
# prefer the screened corpus run_pilot trained on, else the raw mount
frag = f"{WORK}/omniphyl-work/screened/fragments_screened.fna"
lab = f"{WORK}/omniphyl-work/screened/labels_screened.tsv"
if not os.path.exists(frag):
    frag, lab = f"{CORPUS}/fragments.fna", f"{CORPUS}/labels.tsv"

for name, extra in [
    ("len150", ["--eval-crop-len", "150"]),
    ("len250", ["--eval-crop-len", "250"]),
    ("pe150", ["--eval-paired-len", "150"]),
    ("pe250", ["--eval-paired-len", "250"]),
]:
    outdir = f"{WORK}/omniphyl-work/run/eval_{name}"
    print(f"=== eval {name} -> {outdir} ===", flush=True)
    subprocess.run(
        [sys.executable, f"{REPO}/scripts/eval_pilot.py",
         "--model", CKPT, "--fragments", frag, "--labels", lab,
         "--outdir", outdir, "--num-workers", "4", *extra],
        check=True)
print("multi-length eval sweep complete")
    outdir = f"{WORK}/omniphyl-work/run/eval_{name}"
    print(f"=== eval {name} -> {outdir} ===", flush=True)
    subprocess.run(
        [sys.executable, f"{REPO}/scripts/eval_pilot.py",
         "--model", CKPT, "--fragments", frag, "--labels", lab,
         "--outdir", outdir, "--num-workers", "4", *extra],
        check=True)
print("multi-length eval sweep complete")
