"""Kaggle kernel: taxonomic check of PvSY43 / PvSY56 ("vivax-like"?).

GCA_003402195.1 (PvSY43_v1) and GCA_003402215.1 (PvSY56_v1) are
submitted to NCBI as Plasmodium vivax but have been described as a
distinct "vivax-like" clade. This kernel downloads both genomes from
the NCBI datasets API and runs `omniphyl profile` in contig mode
(per-contig reconciled calls, base-weighted abundance) with the v6
w64 checkpoint, so we can see where the model actually places them.

Outputs: /kaggle/working/{pv43,pv56}.{precontigs.tsv,precontigs.fasta,
precontigs.pep.fasta,taxonomic_profile.tsv,functional_profile.tsv,
summary.json} — pull with
kaggle kernels output daniellecallan/omniphyl-pvlike-profile
"""

import glob
import os
import socket
import subprocess
import sys
import urllib.request
import zipfile

try:
    socket.create_connection(("api.ncbi.nlm.nih.gov", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = "/kaggle/working"

GENOMES = {
    "pv43": "GCA_003402195.1",
    "pv56": "GCA_003402215.1",
}

ckpt_hits = sorted(glob.glob("/kaggle/input/**/w64_e15_best.pt",
                             recursive=True))
if not ckpt_hits:
    ckpt_hits = sorted(glob.glob("/kaggle/input/**/*_e15_best.pt",
                                 recursive=True))
if not ckpt_hits:
    raise SystemExit("no *_e15_best.pt under /kaggle/input — "
                     "is omniphyl-v6-ckpts mounted?")
CKPT = ckpt_hits[0]

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "numpy"], check=False)
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/d-callan/omniphyl.git",
                f"{WORK}/omniphyl"], check=True)

import shutil
device = "cuda" if shutil.which("nvidia-smi") else "cpu"
env = dict(os.environ,
           PYTHONPATH=f"{WORK}/omniphyl/src",
           PYTHONUNBUFFERED="1",
           OMP_NUM_THREADS="4")

for tag, acc in GENOMES.items():
    zpath = f"{WORK}/{acc}.zip"
    url = (f"https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/"
           f"accession/{acc}/download"
           "?include_annotation_type=GENOME_FASTA&hydrated=FULLY_HYDRATED")
    print(f"downloading {acc} -> {tag}", flush=True)
    urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        fna = [n for n in z.namelist()
               if n.endswith("_genomic.fna")][0]
        z.extract(fna, WORK)
        fasta = f"{WORK}/{fna}"
    print(f"\n=== omniphyl profile {tag} ({acc}) on {device} ===",
          flush=True)
    subprocess.run(
        [sys.executable, "-m", "omniphyl", "profile",
         "--model", CKPT,
         "--out", f"{WORK}/{tag}",
         "--mode", "contig",
         "--device", device,
         "--batch-size", "256",
         fasta],
        env=env, check=True)

print("\ndone — pv43.*/pv56.* outputs in /kaggle/working", flush=True)
