"""Kaggle CPU kernel: Option B (~20k genomes, ~900k fragments) v5 corpus build.

Pipeline:
  1. RefSeq assemblies across protozoa, fungi, archaea, bacteria, viral
     (Complete Genome, Chromosome, Scaffold) with GFF required on RefSeq
     so E3/E4 region/frame/splice supervision is dense.
  2. GTDB r220 taxonomy override (bac120 + ar53) for bacteria and archaea,
     replacing polyphyletic NCBI ranks with monophyletic GTDB ranks.
  3. Genus-level sampling (--one-per-rank genus --per-rank-n 2) with
     round-robin species interleaving inside each genus, plus GenBank
     gap-fill (--genbank-groups archaea protozoa fungi) for genera absent
     from RefSeq (~21.3k selected assemblies total; GenBank rows stay
     train-only via seq_source=genbank).
  4. Widened GFF domain regex (InterPro/Pfam/CDD/COG/TIGRFAM/PROSITE) +
     post-slice pyhmmer Pfam-A hmmsearch (--pfam-hmms) across all emitted
     fragments so E5 domain supervision is uniform across all clades.
  5. Contaminant screen (phiX174 + UniVec_Core).
  6. Screened corpus files + dataset-metadata.json are written directly to
     /kaggle/working (and optionally published via Kaggle API if
     KAGGLE_USERNAME/KAGGLE_KEY or KAGGLE_API_TOKEN are set).
"""

import glob
import os
import shutil
import socket
import subprocess
from pathlib import Path

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet + GPU), then re-push.")

WORK = "/kaggle/working"
WORKDIR = f"{WORK}/omniphyl-work"
os.environ.setdefault("WORKDIR", WORKDIR)
os.environ.setdefault("OUTDIR", WORK)

# If a pre-sliced raw corpus is mounted in /kaggle/input, copy it into
# WORKDIR/corpus and set SKIP_FETCH=1 so run_pilot.sh runs only the
# pyhmmer Pfam-A scan and contaminant screen.
raw_hits = glob.glob("/kaggle/input/**/fragments.fna", recursive=True)
if raw_hits:
    raw_dir = Path(raw_hits[0]).parent
    dst_corpus = Path(WORKDIR) / "corpus"
    dst_corpus.mkdir(parents=True, exist_ok=True)
    for fname in ("fragments.fna", "labels.tsv", "manifest.json", "skipped.tsv"):
        src_f = raw_dir / fname
        if src_f.exists():
            shutil.copy2(src_f, dst_corpus / fname)
    os.environ["SKIP_FETCH"] = "1"
    os.environ["CORPUS_DIR"] = str(dst_corpus)
    print(f"==> Resuming from mounted pre-sliced corpus at {raw_dir}")

# Option B (~20k genomes, ~900k fragments at 25kb/genome):
os.environ.setdefault("GROUPS", "protozoa fungi archaea bacteria viral")
os.environ.setdefault("LEVELS", "Complete Genome,Chromosome,Scaffold")
os.environ.setdefault("ONE_PER_RANK", "genus")
os.environ.setdefault("PER_RANK_N", "2")
os.environ.setdefault("NT_BUDGET", "25000")
os.environ.setdefault("VEUPATH", "0")
os.environ.setdefault("REQUIRE_GFF", "1")
os.environ.setdefault("BOUNDARY_FRAC", "0.3")
os.environ.setdefault("GENBANK_GROUPS", "archaea protozoa fungi")
os.environ.setdefault("GENBANK_EXTRA", "0")
os.environ.setdefault("GTDB", "1")
os.environ.setdefault("PFAM_HMMS", "1")
os.environ.setdefault("FETCH_ONLY", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

DATASET_ID = os.environ.get(
    "PUBLISH_DATASET_ID", "daniellecallan/omniphyl-v5-20k")
DATASET_TITLE = os.environ.get(
    "PUBLISH_DATASET_TITLE", "omniphyl-v5-20k")

if (os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")
        or Path.home().joinpath(".kaggle/kaggle.json").exists()
        or Path.home().joinpath(".kaggle/access_token").exists()):
    os.environ.setdefault("PUBLISH_DATASET_ID", DATASET_ID)
    os.environ.setdefault("PUBLISH_DATASET_TITLE", DATASET_TITLE)

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)

# Stage top-level dataset files in /kaggle/working and clean up the
# transient workdir so the kernel output itself can be promoted directly
# to a private Kaggle Dataset ("New Dataset from Kernel Output") or
# pulled with `kaggle kernels output`.
corpus_dir = Path(WORKDIR) / "corpus"
for src_name, dst_name in [
    ("fragments_screened.fna", "fragments.fna"),
    ("labels_screened.tsv", "labels.tsv"),
    ("manifest.json", "manifest.json"),
    ("screen_manifest.json", "screen_manifest.json"),
    ("skipped.tsv", "skipped.tsv"),
]:
    src = corpus_dir / src_name
    if not src.exists() and src_name.endswith("_screened.fna"):
        src = corpus_dir / "fragments.fna"
    if not src.exists() and src_name.endswith("_screened.tsv"):
        src = corpus_dir / "labels.tsv"
    if src.exists():
        shutil.copy2(src, Path(WORK) / dst_name)

Path(WORK, "dataset-metadata.json").write_text(
    '{\n'
    f'  "title": "{DATASET_TITLE}",\n'
    f'  "id": "{DATASET_ID}",\n'
    '  "licenses": [{"name": "CC0-1.0"}]\n'
    '}\n'
)

shutil.rmtree(WORKDIR, ignore_errors=True)
shutil.rmtree(f"{WORK}/omniphyl", ignore_errors=True)
print(f"==> staged dataset files in {WORK} for {DATASET_ID}")
