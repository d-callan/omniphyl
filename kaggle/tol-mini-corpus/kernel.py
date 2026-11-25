"""Kaggle CPU kernel: Mini-ToL corpus + refiner panel (tol-arch §4.2).

Phase 1  fetch the 9-group Mini-ToL Stage-1 corpus via run_pilot.sh
         (FETCH_ONLY): ~150k windows across archaea/bacteria/fungi/
         protozoa/plant/invertebrate/vertebrate_other/
         vertebrate_mammalian/viral, family-subsampled ~2/family,
         clade-aware contaminant screen, then pack to
         corpus.packed/ (mmap store) via pack_corpus.py.
Phase 2  fetch the Mini-ToL Refiner panel: 9 training species
         (one chromosome/scaffold each + vertebrate chrM) and
         held-out decode species incl. organelle + alt-code taxa,
         sliced to the target seqid, written as
         refiner_panel/{train,heldout}/<name>.fna + .gff3 + panel.json.

Outputs land in /kaggle/working/omniphyl-tol-mini/ — promote the
directory to a Kaggle Dataset so tol-stage1-bench / tol-refiner-bench
can mount it.
"""

import io
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM -- verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
OUT = WORK / "omniphyl-tol-mini"
OUT.mkdir(exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)
sys.path.insert(0, str(WORK / "omniphyl/src"))
sys.path.insert(0, str(WORK / "omniphyl/scripts"))

# ---- Phase 1: 9-group corpus fetch + clade-aware screen --------------
os.environ["WORKDIR"] = str(WORK / "omniphyl-work")
os.environ["OUTDIR"] = str(OUT)
os.environ["GROUPS"] = ("archaea bacteria fungi protozoa plant "
                        "invertebrate vertebrate_other "
                        "vertebrate_mammalian viral")
os.environ["LEVELS"] = "Complete Genome,Chromosome,Scaffold"
os.environ["ONE_PER_RANK"] = "family"
os.environ["PER_RANK_N"] = "2"
# ~8.5 Mb/group -> ~150k 500-bp windows total
os.environ["GROUP_NT_TOTAL"] = "8500000"
os.environ["NT_BUDGET"] = "400000"
os.environ["VEUPATH"] = "1"
os.environ["GENBANK_GROUPS"] = "archaea protozoa fungi plant invertebrate"
os.environ["GENBANK_EXTRA"] = "0"
os.environ["FETCH_ONLY"] = "1"
# 8 concurrent euk genomes blew the ~20GB quota back when results were
# buffered until the end; streaming emit keeps disk bounded now. RAM is
# the limit on the big-euk tail -- a MemoryError'd worker just marks its
# accession skipped (retried by the next resume run).
os.environ["FETCH_WORKERS"] = "6"
# VEuPathDB only covers eukaryotic pathogens/vectors; skipping UCSC
# lookups for the other ~6k accessions saves ~2 HTTP calls each
os.environ["VEUPATH_GROUPS"] = "protozoa fungi invertebrate"
# genome downloads are deleted right after slicing; anything still
# in-flight at a timeout kill is dead weight in the output capture
# (~4.4GB of a 5GB pull). Container /tmp is not captured.
os.environ["OMNIPHYL_GENOMES_DIR"] = "/tmp/omniphyl-genomes"

# resume: if a partial-corpus dataset is attached (output of a previous
# run that hit Kaggle's 12h cap), seed it into the fetch workdir and set
# BASE_CORPUS so selection skips every accession already in labels.tsv.
CORPUS_DIR = Path(os.environ["WORKDIR"]) / "corpus"
for _d in Path("/kaggle/input").iterdir():
    if (_d / "fragments.fna").exists() and (_d / "labels.tsv").exists():
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(_d, CORPUS_DIR, dirs_exist_ok=True)
        os.environ["BASE_CORPUS"] = str(CORPUS_DIR)
        print(f"resuming corpus from {_d.name}")
        break

subprocess.run(["bash", str(WORK / "omniphyl/remote/run_pilot.sh")],
               check=True)

# run_pilot.sh lands the corpus in $WORKDIR/corpus (fresh fetch; if the
# corpus dir were mounted read-only the screen would go to
# $WORKDIR/screened). OUTDIR only gets the corpus tarball.
def locate(pat: str) -> Path:
    hits = sorted(WORK.glob(f"omniphyl-work/**/{pat}"))
    if not hits:
        raise SystemExit(f"fetch produced no {pat} under omniphyl-work")
    return hits[0]


try:
    fna = locate("fragments_screened.fna")
    lab = locate("labels_screened.tsv")
except SystemExit:
    fna = locate("fragments.fna")
    lab = locate("labels.tsv")
subprocess.run([sys.executable,
                str(WORK / "omniphyl/scripts/pack_corpus.py"),
                "--fragments", str(fna), "--labels", str(lab),
                "--out", str(OUT / "corpus.packed")], check=True)

# keep the raw screened pair too -- eval_pilot reads raw FASTA/TSV
for p in (fna, lab):
    (OUT / p.name).write_bytes(p.read_bytes())
for extra in ("screen_manifest.json", "manifest.json"):
    try:
        (OUT / extra).write_bytes(locate(extra).read_bytes())
    except SystemExit:
        pass

# the corpus now lives under OUT; drop the workdir copy so the panel
# downloads + tarball have headroom
shutil.rmtree(fna.parent, ignore_errors=True)

# ---- Phase 2: refiner panel ------------------------------------------
# canonical implementation: scripts/build_refiner_panel.py;
# do not fork the PANEL spec here.
subprocess.run([sys.executable,
                str(WORK / "omniphyl/scripts/build_refiner_panel.py"),
                "--out", str(OUT / "refiner_panel"),
                "--dl", str(WORK / "_panel_dl")], check=True)

# tarball for `kaggle kernels output` download -- it duplicates OUT on
# disk, so skip it rather than ENOSPC the output dir when space is tight
out_bytes = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
free_bytes = shutil.disk_usage(WORK).free
if free_bytes > out_bytes * 1.2:
    subprocess.run(["tar", "-czf", str(WORK / "omniphyl-tol-mini.tar.gz"),
                    "-C", str(WORK), "omniphyl-tol-mini"], check=True)
    print(f"done -> {OUT} (+ omniphyl-tol-mini.tar.gz)")
else:
    print(f"done -> {OUT} (skipped tarball: "
          f"{out_bytes/1e9:.1f} GB out, {free_bytes/1e9:.1f} GB free)")
