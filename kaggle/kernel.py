"""Kaggle kernel entrypoint: clone repo, run remote/run_pilot.sh.

Outputs (checkpoint, eval json, labels) land under /kaggle/working and
are pulled back with `kaggle kernels output`. Env vars forward through
this file's os.environ defaults below — tweak there or locally in the
script. See docs/remote-runs.md.
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
# v4: 5 groups incl. archaea, 2 assemblies per FAMILY (~3.6k genomes)
# -> val asks "new species in a known family"; 75k nt/accession ->
# ~430k frags; regs on (dropout/label-smoothing/wd), 10 epochs.
os.environ.setdefault("GROUPS", "protozoa fungi bacteria viral archaea")
os.environ.setdefault("LEVELS", "Complete Genome,Chromosome,Scaffold")
os.environ.setdefault("ONE_PER_RANK", "family")
os.environ.setdefault("PER_RANK_N", "2")
os.environ.setdefault("NT_BUDGET", "75000")
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
