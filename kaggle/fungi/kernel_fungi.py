"""Kaggle kernel entrypoint: clone repo, run remote/run_pilot.sh.

FUNGI CAPACITY PROBE: single-group, species-level corpus to test whether
v4's deep-rank failure (val family 0.11 / species 0.21) is model capacity
vs corpus dilution. ~461 genomes / ~456 species classes, undivided
capacity. If val species stays ~0.1-0.2 here, the bottleneck is the
500bp FCGR representation itself.

Outputs (checkpoint, eval json, labels) land under /kaggle/working and
are pulled back with `kaggle kernels output`. Env vars forward through
this file's os.environ defaults below.
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
# probe: fungi only, ALL strains per species (cap 2), deep 300k nt/genome
os.environ.setdefault("GROUPS", "fungi")
os.environ.setdefault("LEVELS", "Complete Genome,Chromosome,Scaffold")
os.environ.setdefault("ONE_PER_RANK", "species")
os.environ.setdefault("PER_RANK_N", "2")
os.environ.setdefault("NT_BUDGET", "300000")
os.environ.setdefault("EPOCHS", "10")
os.environ.setdefault("DROPOUT", "0.2")
os.environ.setdefault("LABEL_SMOOTHING", "0.05")
os.environ.setdefault("WD", "1e-2")
os.environ.setdefault("WIDTH", "32")
os.environ.setdefault("VEUPATH", "1")
# block-buffered stdout hides fetch progress in the kernel log viewer
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# os.environ["REPO_URL"] = "https://<token>@github.com/d-callan/omniphyl.git"

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
