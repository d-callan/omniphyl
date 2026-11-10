"""omniphyl-clade-bench — cross-clade contig benchmark.

Runs scripts/clade_bench.py from a fresh clone: fetches FASTA+GFF for
held-out organisms (NCBI datasets REST for bsubtilis / scerevisiae /
tbrucei / ddiscoideum / mjannaschii; P. vivax P01 is mounted as a
dataset since VEuPathDB downloads need a session), slices the most
CDS-dense contig per organism, runs `omniphyl annotate` with the
multi-genome refiner, scores vs reference GFF, and emits the
eval_errors error-class breakdown per organism.

Datasets:
  - daniellecallan/omniphyl-v6-ckpts     (w64_e15_best.pt base model)
  - daniellecallan/omniphyl-refiner-multi (refiner_multi.pt)
  - daniellecallan/refiner-pvivax        (PlasmoDB-71 PvivaxP01 fna+gff)

Output: /kaggle/working/clade_eval/{results.json,results.md,
         <org>/errors.json,...}
"""
import glob
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

try:
    urllib.request.urlopen("https://github.com", timeout=10)
except (urllib.error.URLError, socket.timeout):
    raise SystemExit(
        "NO INTERNET on this kernel VM -- verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
os.environ.setdefault("OMP_NUM_THREADS", "4")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)
SRC = str(WORK / "omniphyl/src")


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern} under /kaggle/input; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return hits[0]


MODEL = find_one("w64_e15_best.pt")
REFINER = find_one("refiner_multi.pt")

# stage P. vivax into the workdir cache so clade_bench.fetch() skips
# the (session-gated) VEuPathDB download entirely
data_dir = WORK / "clade_eval/data"
data_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(find_one("*PvivaxP01_Genome.fasta"), data_dir / "pvivax.fna")
shutil.copy(find_one("*PvivaxP01.gff"), data_dir / "pvivax.gff")

r = subprocess.run(
    [sys.executable,
     str(WORK / "omniphyl/scripts/clade_bench.py"),
     "--workdir", str(WORK / "clade_eval"),
     "--model", MODEL, "--refiner", REFINER],
    env={**os.environ, "PYTHONPATH": SRC})
sys.exit(r.returncode)
