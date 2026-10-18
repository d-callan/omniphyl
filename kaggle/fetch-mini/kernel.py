"""Kaggle kernel entrypoint: 5group-mini fetch-only corpus build.

Same five groups and family-level sampling as v4, but ~30 frags/genome
(NT_BUDGET=15000) and one assembly per family — target ~500-1000 genomes,
~20-30k fragments.  GenBank gap-fill runs for the weak groups
(protozoa/archaea/fungi); GenBank picks are marked seq_source=genbank and
forced train-only at split time.

FETCH_ONLY=1: fetch + contaminant screen + bundle, exit before training.
Output tarball lands in /kaggle/working; pull with `kaggle kernels output`
or promote to a Kaggle Dataset so ablation kernels can mount it.

GPU is disabled for this kernel (fetch is network-bound) — the T4/P100
quota is reserved for the ablation arms.
"""

import os
import socket
import subprocess

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet + GPU), then re-push.")

WORK = "/kaggle/working"
os.environ.setdefault("WORKDIR", f"{WORK}/omniphyl-work")
os.environ.setdefault("OUTDIR", WORK)
# 5group-mini: same groups as v4, 1 assembly per family, ~30 frags/genome.
os.environ.setdefault("GROUPS", "protozoa fungi bacteria viral archaea")
os.environ.setdefault("LEVELS", "Complete Genome,Chromosome,Scaffold")
os.environ.setdefault("ONE_PER_RANK", "family")
os.environ.setdefault("PER_RANK_N", "1")
os.environ.setdefault("NT_BUDGET", "15000")
os.environ.setdefault("VEUPATH", "1")
# GenBank gap-fill for the underrepresented groups only (train-only at
# split time via seq_source=genbank); genbank-extra stays 0 = fill only
# rank values RefSeq missed, no supplement on covered values.
os.environ.setdefault("GENBANK_GROUPS", "archaea protozoa fungi")
os.environ.setdefault("GENBANK_EXTRA", "0")
# corpus capture only — no training, no GPU
os.environ.setdefault("FETCH_ONLY", "1")
# contaminant screen stays on (default); runs pre-bundle so the dataset
# ships both raw and screened fragments
# block-buffered stdout hides fetch progress in the kernel log viewer
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# os.environ["REPO_URL"] = "https://<token>@github.com/d-callan/omniphyl.git"

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
