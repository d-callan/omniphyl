"""Kaggle smoke kernel: clone repo, run tests/test_smoke.py on CPU.

Validates the fetch/train/eval machinery on Kaggle VMs without burning
GPU quota — same checks as `run_pilot.sh`'s SMOKE stage. Push:
`cd kaggle/smoke && kaggle kernels push`; read output via
`kaggle kernels output daniellecallan/omniphyl-smoke` (or the web log).
"""

import socket
import subprocess

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet + GPU), then re-push.")

WORK = "/kaggle/working"
subprocess.run(
    ["git", "clone", "--depth", "1",
     "https://github.com/d-callan/omniphyl.git", f"{WORK}/omniphyl"],
    check=True)
subprocess.run(
    ["python", "-c",
     "import torch, numpy; print('torch', torch.__version__)"],
    check=True)
subprocess.run(["python", "tests/test_smoke.py"],
               cwd=f"{WORK}/omniphyl", check=True)
print("SMOKE PASS")
