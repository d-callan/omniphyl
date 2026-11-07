"""Kaggle kernel: ZymoBIOMICS D6300 mock community -> omniphyl profile.

Streams a prefix subsample of SRR12324253 (Illumina NovaSeq 150bp PE,
ZymoBIOMICS D6300 "even" community: 8 bacteria @ ~12%, S. cerevisiae +
C. neoformans @ ~2%) from ENA, then runs `omniphyl profile` in
precontig mode (paired-end join) with the v6 w64 checkpoint.

Ground truth for sanity checks (cell fractions, Zymo spec):
  Pseudomonas aeruginosa      12%   Salmonella enterica        12%
  Escherichia coli            12%   Lactobacillus fermentum    12%
  Enterococcus faecalis       12%   Staphylococcus aureus      12%
  Listeria monocytogenes      12%   Bacillus subtilis          12%
  Saccharomyces cerevisiae     2%   Cryptococcus neoformans     2%

Outputs: /kaggle/working/zymo.{precontigs.tsv,precontigs.fasta,
precontigs.pep.fasta,taxonomic_profile.tsv,functional_profile.tsv,
summary.json,reads.tsv} — pull with
kaggle kernels output daniellecallan/omniphyl-zymo-profile
"""

import gzip
import glob
import os
import shutil
import socket
import subprocess
import sys
import urllib.request

try:
    socket.create_connection(("www.ebi.ac.uk", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = "/kaggle/working"
N_PAIRS = int(os.environ.get("ZYMOPROF_N_PAIRS", "400000"))

ENA_BASE = ("https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/053/"
            "SRR12324253")
R1_URL = f"{ENA_BASE}/SRR12324253_1.fastq.gz"
R2_URL = f"{ENA_BASE}/SRR12324253_2.fastq.gz"


def stream_prefix(url: str, out_path: str, n_recs: int):
    """Stream-decompress a gz FASTQ URL, keep the first n_recs records.

    Writes raw text (no re-gzip) so iter_sequences sees plain FASTQ,
    then truncates the tail to the last complete 4-line record.
    """
    want = 4 * n_recs
    n_lines = 0
    with urllib.request.urlopen(url) as resp, \
            gzip.GzipFile(fileobj=resp) as gz, \
            open(out_path, "wb") as out:
        while n_lines < want:
            chunk = gz.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            n_lines += chunk.count(b"\n")
    # drop the partial tail: keep only the first `want` complete lines
    n = 0
    with open(out_path, "r+b") as f:
        while n < want:
            b = f.read(1 << 20)
            if not b:
                break
            nl = b.count(b"\n")
            if n + nl > want:
                # position of the want-th newline within this chunk
                need = want - n
                pos = -1
                for _ in range(need):
                    pos = b.find(b"\n", pos + 1)
                cut = f.tell() - len(b) + pos + 1
                f.truncate(cut)
                break
            n += nl
    with open(out_path, "rb") as f:
        f.seek(0, 2)
        sz = f.tell()
    print(f"{out_path}: ~{sz // 350:,} reads ({sz:,} B)", flush=True)


ckpt_hits = sorted(glob.glob("/kaggle/input/**/w64_e15_best.pt",
                             recursive=True))
if not ckpt_hits:
    ckpt_hits = sorted(glob.glob("/kaggle/input/**/*_e15_best.pt",
                                 recursive=True))
if not ckpt_hits:
    raise SystemExit("no *_e15_best.pt under /kaggle/input — "
                     "is omniphyl-v6-ckpts mounted?")
CKPT = ckpt_hits[0]

print(f"downloading first {N_PAIRS:,} pairs of SRR12324253 "
      "(Zymo D6300, NovaSeq 150bp PE)", flush=True)
r1, r2 = f"{WORK}/zymo_r1.fq", f"{WORK}/zymo_r2.fq"
stream_prefix(R1_URL, r1, N_PAIRS)
stream_prefix(R2_URL, r2, N_PAIRS)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "numpy"], check=False)
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/d-callan/omniphyl.git",
                f"{WORK}/omniphyl"], check=True)

device = "cuda" if shutil.which("nvidia-smi") else "cpu"
env = dict(os.environ,
           PYTHONPATH=f"{WORK}/omniphyl/src",
           PYTHONUNBUFFERED="1",
           OMP_NUM_THREADS="4")

print(f"\n=== omniphyl profile [precontig] on {device} ===",
      flush=True)
subprocess.run(
    [sys.executable, "-m", "omniphyl", "profile",
     "--model", CKPT,
     "--out", f"{WORK}/zymo",
     "--mode", "precontig",
     "--reads2", r2,
     "--device", device,
     "--batch-size", "256",
     r1],
    env=env, check=True)

print("\ndone — zymo.* outputs in /kaggle/working", flush=True)
