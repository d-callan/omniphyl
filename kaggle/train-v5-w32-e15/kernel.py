"""Kaggle kernel entrypoint: v5 corpus main run — width 32, 15 epochs (width-ablation arm vs train-v5-w64-e15).

Locked-in architecture (e5-domains arm config): msfcgr backbone with
positional channels, gated AA branch (20% branch dropout), flat heads
with flat_prior_tau=1.0, region + struct + domain-256 annotation heads.
On the ~21k-genome / ~900k-fragment v5 corpus (--one-per-rank genus
per_rank_n=2 + GenBank gap-fill + GTDB ranks + Pfam fragment labels).

CACHE_SEQS=1 loads all fragment sequences into a single in-RAM bytes
blob at dataset init, eliminating the per-sample open()+seek() that
capped the dataloader at ~165 frags/s on Kaggle's 4-vCPU VMs. With the
cache, epoch time is GPU-bound (~21-30 min/epoch at width 64), keeping
15 epochs + eval inside the 12h session limit.

Runs remote/run_pilot.sh with SKIP_FETCH=1 on the mounted corpus dataset.
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
# corpus ships already contaminant-screened
os.environ.setdefault("SCREEN_CONTAM", "0")
# --- locked-in arch (e5-domains arm) -----------------------------------
os.environ.setdefault("REGION_HEAD", "1")
os.environ.setdefault("STRUCT_HEAD", "1")
os.environ.setdefault("DOMAIN_HEAD_K", "256")
os.environ.setdefault("AUX", "aa")
os.environ.setdefault("AA_MODE", "gated")
os.environ.setdefault("AA_DROP", "0.2")
os.environ.setdefault("BACKBONE", "msfcgr")
os.environ.setdefault("HEAD", "flat")
os.environ.setdefault("FLAT_PRIOR_TAU", "1.0")
os.environ.setdefault("FRAG_VAL", "0.05")
os.environ.setdefault("DROPOUT", "0.2")
os.environ.setdefault("LABEL_SMOOTHING", "0.05")
os.environ.setdefault("WD", "1e-2")
os.environ.setdefault("VAL_FRAC", "0.05")
os.environ.setdefault("HOLDOUT_GENUS_FRAC", "0.02")
# --- v6 main-run schedule ----------------------------------------------
# v6 is ~1.6x v5 (1.06M vs ~670k frags), so ~74min/epoch becomes
# ~117min pre-speedup. 6 epochs is the ceiling inside the 12h cap
# with room for eval; last.pt is saved atomically every epoch so a
# shortfall still yields a usable checkpoint.
os.environ.setdefault("WIDTH", "32")
os.environ.setdefault("EPOCHS", "6")
os.environ.setdefault("CACHE_SEQS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
