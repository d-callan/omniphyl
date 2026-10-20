"""Kaggle kernel: P0 false-positive-first baseline eval on frozen checkpoints.

Runs scripts/eval_pilot.py --val-only for each checkpoint in the mounted
omniphyl-pilot-checkpoints dataset, against the corpus each arm trained on:

    arm    checkpoint           corpus
    a2     mini (msfcgr flat)   mini/ screened corpus (bundled in dataset)
    a3f    mini (msfcgr+aux)    mini/ screened corpus (bundled in dataset)
    v4     v4  (resnet flat)    omniphyl-v4-corpus dataset (mounted)

Outputs land under /kaggle/working/eval_<arm>/ — results.json,
selective_precision.tsv, ancestor_veto.tsv, predictions.tsv, etc.

The kernel clones the repo at HEAD; the eval machinery added in P0
(selective metrics, reconciliation) must be pushed before this kernel
can produce those outputs — an older clone silently skips the new
sections.
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
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
EVAL = f"{WORK}/omniphyl/scripts/eval_pilot.py"


def find(pattern):
    hits = glob.glob(f"/kaggle/input/**/{pattern}", recursive=True)
    if not hits:
        raise SystemExit(f"missing {pattern} under /kaggle/input — "
                         "checkpoint/corpus datasets mounted?")
    return hits[0]


# mini corpus: screened copy bundled inside the checkpoint dataset
MINI_LABELS = find("mini/labels.tsv")
MINI_FRAGS = find("mini/fragments.fna")
# v4 corpus: mounted as its own dataset
V4_LABELS = find("labels.tsv")
if V4_LABELS == MINI_LABELS:
    V4_LABELS = next(h for h in glob.glob(
        "/kaggle/input/**/labels.tsv", recursive=True) if "mini" not in h)
V4_FRAGS = os.path.join(os.path.dirname(V4_LABELS), "fragments.fna")

ARMS = {
    "a2": (find("a2/best.pt"), MINI_LABELS, MINI_FRAGS),
    "a3f": (find("a3f/best.pt"), MINI_LABELS, MINI_FRAGS),
    "v4": (find("v4/best.pt"), V4_LABELS, V4_FRAGS),
}

for arm, (ckpt, labels, frags) in ARMS.items():
    outdir = f"{WORK}/eval_{arm}"
    print(f"\n=== eval {arm}: {ckpt} -> {outdir} ===", flush=True)
    subprocess.run(
        [sys.executable, EVAL,
         "--model", ckpt, "--labels", labels, "--fragments", frags,
         "--outdir", outdir, "--val-only",
         "--batch-size", "512", "--num-workers", "2"],
        check=True)

print("\nall arms evaluated; outputs under", WORK)
