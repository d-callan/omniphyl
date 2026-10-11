"""Kaggle kernel entrypoint: clone repo, run remote/run_pilot.sh.

Outputs (checkpoint, eval json, labels) land under /kaggle/working and
are pulled back with `kaggle kernels output`. Env vars forward through
this file's os.environ defaults below — tweak there or locally in the
script. See docs/remote-runs.md.
"""

import os
import subprocess

WORK = "/kaggle/working"
os.environ.setdefault("WORKDIR", f"{WORK}/omniphyl-work")
os.environ.setdefault("OUTDIR", WORK)
# os.environ["REPO_URL"] = "https://<token>@github.com/d-callan/omniphyl.git"

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
