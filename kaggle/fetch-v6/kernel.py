"""Kaggle kernel entrypoint: v6 corpus build (incremental extension over v5-20k).

Mounts daniellecallan/omniphyl-v5-20k (18,915 assemblies, 683,622 screened
fragments) via --base-corpus so existing assemblies and Pfam hits are reused
without re-downloading or re-scanning, and fetches/slices only the v6 delta:
  1. Protozoa & Fungi expanded to SPECIES-level sampling (including Contig
     assemblies, eukaryotic GenBank MAGs/SAGs/environmental isolates via
     --include-mags, and UCSC GenArk VEuPathDBGeneModels via --veupath),
     with 50kb/genome for protozoa and 35kb/genome for fungi.
  2. Deficit-aware GenBank top-up for singleton RefSeq taxa across archaea,
     protozoa, and fungi.
  3. Uncultured bacterial MAG/SAG diversity from GenBank + GTDB r220 sampled
     at family level (GROUP_GENBANK_RANK="bacteria=family",
     GROUP_GENBANK_N="bacteria=1") to cover the ~5,188 GTDB bacterial
     families absent from RefSeq.
  4. pyhmmer Pfam-A scan runs strictly on the newly added delta fragments
     (only_fids) and merges into labels.tsv, followed by contaminant
     screening and staging in /kaggle/working for dataset promotion.
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

raw_hits = glob.glob("/kaggle/input/**/fragments.fna", recursive=True)
if raw_hits:
    raw_dir = Path(raw_hits[0]).parent
    os.environ.setdefault("BASE_CORPUS", str(raw_dir))
    print(f"==> Extending mounted v5 base corpus at {raw_dir}")
else:
    raise SystemExit("no base corpus found under /kaggle/input — "
                     "is daniellecallan/omniphyl-v5-20k mounted?")

os.environ.setdefault("GROUPS", "protozoa fungi archaea bacteria viral")
os.environ.setdefault("LEVELS", "Complete Genome,Chromosome,Scaffold")
os.environ.setdefault("ONE_PER_RANK", "genus")
os.environ.setdefault("PER_RANK_N", "2")
os.environ.setdefault("NT_BUDGET", "25000")
os.environ.setdefault("GROUP_ONE_PER_RANK", "protozoa=species,fungi=species")
os.environ.setdefault("GROUP_PER_RANK_N", "protozoa=2,fungi=1")
os.environ.setdefault("GROUP_GENBANK_RANK", "bacteria=family")
os.environ.setdefault("GROUP_GENBANK_N", "bacteria=1")
os.environ.setdefault("GROUP_NT_BUDGET", "protozoa=50000,fungi=35000")
os.environ.setdefault(
    "GROUP_LEVELS",
    "protozoa=Complete Genome|Chromosome|Scaffold|Contig,"
    "fungi=Complete Genome|Chromosome|Scaffold|Contig")
os.environ.setdefault("VEUPATH", "1")
os.environ.setdefault("REQUIRE_GFF", "1")
os.environ.setdefault("BOUNDARY_FRAC", "0.3")
os.environ.setdefault("GENBANK_GROUPS", "archaea protozoa fungi bacteria")
os.environ.setdefault("GENBANK_EXTRA", "0")
os.environ.setdefault("INCLUDE_MAGS", "1")
os.environ.setdefault("GTDB", "1")
os.environ.setdefault("PFAM_HMMS", "1")
os.environ.setdefault("FETCH_ONLY", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

DATASET_ID = os.environ.get(
    "PUBLISH_DATASET_ID", "daniellecallan/omniphyl-v6")
DATASET_TITLE = os.environ.get(
    "PUBLISH_DATASET_TITLE", "omniphyl-v6")

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
