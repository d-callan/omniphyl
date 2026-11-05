"""Kaggle CPU kernel: stride x bin-width grid on RefSeq chr1.

Runs `omniphyl annotate` over a (S, B) grid on the mounted
omniphyl-infer-test dataset (v5_w64 best.pt + NC_004325.2 fasta/gff3)
so aggregation effects are measured under identical weights/code/VM.
W stays 500 — the model was trained on fixed 500bp fragments, so other
window sizes are out-of-distribution and not informative. B=0 means the
code default gcd(S,50). OMNIPHYL_DUMP_RES=1 writes .res.pkl per arm so
decode parameters can be iterated locally without re-inference.

Outputs: /kaggle/working/chr1_s{S}_b{B}.* — pull with
kaggle kernels output daniellecallan/omniphyl-infer-stride50
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
        "kaggle.com/settings (required for internet), then re-push.")

WORK = "/kaggle/working"
hits = glob.glob("/kaggle/input/**/best.pt", recursive=True)
if not hits:
    raise SystemExit("no best.pt under /kaggle/input — "
                     f"dataset not mounted? input: {os.listdir('/kaggle/input')}")
INP = os.path.dirname(hits[0])
if not os.path.exists(f"{INP}/NC_004325.2.fasta"):
    raise SystemExit(f"missing NC_004325.2.fasta next to {hits[0]}")

# Kaggle CPU image already ships torch+numpy; install is a no-op guard.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "numpy"], check=False)
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/d-callan/omniphyl.git",
                f"{WORK}/omniphyl"], check=True)

env = dict(os.environ,
           PYTHONPATH=f"{WORK}/omniphyl/src",
           OMNIPHYL_DUMP_RES="1",
           PYTHONUNBUFFERED="1",
           OMP_NUM_THREADS="4")

# (stride, bin_width): B=0 -> code default gcd(S,50).
GRID = [(100, 0), (100, 25), (50, 25), (25, 0), (50, 10)]
for stride, bw in GRID:
    tag = f"s{stride}_b{bw or 'auto'}"
    print(f"\n=== annotate stride={stride} bin_width={bw} ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "omniphyl", "annotate",
         "--model", f"{INP}/best.pt",
         "--fasta", f"{INP}/NC_004325.2.fasta",
         "--out", f"{WORK}/chr1_{tag}",
         "--device", "cpu",
         "--batch-size", "256",
         "--stride", str(stride),
         "--bin-width", str(bw)],
        env=env, check=True)

print("\ndone — outputs in /kaggle/working", flush=True)
