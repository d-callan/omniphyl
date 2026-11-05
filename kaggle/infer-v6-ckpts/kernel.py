"""Kaggle CPU kernel: v6 w32/w64 checkpoints -> res.pkl on RefSeq chr1.

The v6-scale training kernels (omniphyl-train-v5-w32-e15, -w64-e15) hit
the Kaggle time limit mid-run; their best.pt checkpoints were pulled and
uploaded as the omniphyl-v6-ckpts dataset. This kernel runs
`omniphyl annotate` on NC_004325.2 (mounted from omniphyl-infer-test)
with OMNIPHYL_DUMP_RES=1 so the cached per-bin outputs can be decoded
locally with redecode.py and compared to the v5 checkpoint numbers.

Outputs: /kaggle/working/chr1_v6{w32,w64}_s100_bauto.res.pkl — pull with
kaggle kernels output daniellecallan/omniphyl-infer-v6-ckpts
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

# fasta/gff3 live in omniphyl-infer-test next to the v5 best.pt
fasta_hits = glob.glob("/kaggle/input/**/NC_004325.2.fasta",
                       recursive=True)
if not fasta_hits:
    raise SystemExit("no NC_004325.2.fasta under /kaggle/input — "
                     "is omniphyl-infer-test mounted?")
FASTA = fasta_hits[0]

ckpts = sorted(glob.glob("/kaggle/input/**/*_e15_best.pt",
                         recursive=True))
if not ckpts:
    raise SystemExit("no *_e15_best.pt under /kaggle/input — "
                     "is omniphyl-v6-ckpts mounted?")

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

for ck in ckpts:
    # w32_e15_best.pt -> v6w32 ; w64_e15_best.pt -> v6w64
    tag = "v6" + os.path.basename(ck).split("_")[0]
    print(f"\n=== annotate {tag} ({ck}) ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "omniphyl", "annotate",
         "--model", ck,
         "--fasta", FASTA,
         "--out", f"{WORK}/chr1_{tag}_s100_bauto",
         "--device", "cpu",
         "--batch-size", "256",
         "--stride", "100",
         "--bin-width", "0"],
        env=env, check=True)

print("\ndone — res.pkl files in /kaggle/working", flush=True)
