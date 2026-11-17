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
NCBI_DL = ("https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
           "{acc}/download?include_annotation_type=GENOME_FASTA,"
           "GENOME_GFF&filename={acc}.zip")

# name -> (accession, seqid substring, clade, expected gcode, role,
#          max bases kept from seq start, chrM seqid substring)
PANEL = [
    # ---- training species (single chrom/scaffold each) ----
    ("ecoli",      "GCF_000005845.2",  "NC_000913", "bacteria", None,
     "train", 5_000_000),
    ("hsalinarum", "GCF_000006805.1",  "NC_002607", "archaea", None,
     "train", 3_000_000),
    ("scerevisiae", "GCF_000146045.2", "NC_001136", "fungi", None,
     "train", 2_000_000),
    ("pfalciparum", "GCF_000002765.2", "Pf3D7_14", "apicomplexa",
     None, "train", 4_000_000),
    ("athaliana",  "GCF_000001735.4",  "NC_003070", "plant", None,
     "train", 30_000_000),
    ("celegans",   "GCF_000002985.6",  "NC_003279", "invertebrate",
     None, "train", 16_000_000),
    ("dmelanogaster", "GCF_000001215.4", "NT_033779", "arthropod",
     None, "train", 24_000_000),
    ("drerio",     "GCF_000002035.6",  "NC_007112", "vertebrate", None,
     "train", 30_000_000),
    ("mmusculus",  "GCF_000001635.27", "NC_000085", "vertebrate", None,
     "train", 30_000_000),
    # vertebrate chrM in the training set (code 2)
    ("mmus_chrm",  "GCF_000001635.27", "NC_005089", "vertebrate", 2,
     "train", 0),
    # ---- held-out decode species ----
    ("bsubtilis",  "GCF_000009045.1",  "NC_000964", "bacteria", None,
     "heldout", 5_000_000),
    ("pvivax",     "GCF_000002415.2",  "PvP01_12", "apicomplexa",
     None, "heldout", 4_000_000),
    ("osativa",    "GCF_001433935.1",  "NC_029256", "plant", None,
     "heldout", 30_000_000),
    ("agambiae",   "GCF_000005575.2",  "NT_078266", "arthropod",
     None, "heldout", 24_000_000),
    ("ggallus",    "GCF_000002315.6",  "NC_006093", "vertebrate", None,
     "heldout", 30_000_000),
    ("hsapiens",   "GCF_000001405.40", "NC_000022", "vertebrate", None,
     "heldout", 10_000_000),
    ("hsap_chrm",  "GCF_000001405.40", "NC_012920", "vertebrate", 2,
     "heldout", 0),
    # alt-code taxa for gate S2-C
    ("tetrahymena", "GCF_000189635.1", "scaffold", "ciliate", 6,
     "heldout", 3_000_000),
    ("mycoplasma", "GCF_000027325.1",  "NC_000908", "bacteria", 4,
     "heldout", 1_000_000),
]

PANEL_DIR = OUT / "refiner_panel"
(PANEL_DIR / "train").mkdir(parents=True, exist_ok=True)
(PANEL_DIR / "heldout").mkdir(parents=True, exist_ok=True)
DL = WORK / "panel_dl"
DL.mkdir(exist_ok=True)
manifest = {}


def fetch_asm(acc: str) -> tuple[Path, Path]:
    """-> (genomic.fna, genomic.gff) for an assembly accession."""
    fna_hits = sorted(DL.glob(f"{acc}/**/[!r]*genomic.fna",
                              recursive=True)) or \
        sorted(DL.glob(f"{acc}/**/*.fna", recursive=True))
    gff_hits = sorted(DL.glob(f"{acc}/**/*.gff", recursive=True))
    if fna_hits and gff_hits:
        return fna_hits[0], gff_hits[0]
    zpath = DL / f"{acc}.zip"
    if not zpath.exists():
        print(f"  download {acc}", flush=True)
        req = urllib.request.Request(
            NCBI_DL.format(acc=acc),
            headers={"User-Agent": "omniphyl-tol-mini"})
        with urllib.request.urlopen(req, timeout=600) as r:
            zpath.write_bytes(r.read())
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DL / acc)
    fna_hits = sorted(p for p in DL.glob(f"{acc}/**/*.fna",
                                         recursive=True)
                      if "rna" not in p.name
                      and "cds" not in p.name.lower())
    gff_hits = sorted(DL.glob(f"{acc}/**/*.gff", recursive=True))
    if not fna_hits or not gff_hits:
        raise SystemExit(f"{acc}: no fasta/gff inside zip")
    return fna_hits[0], gff_hits[0]


def read_fna(path: Path) -> list:
    seqs, h, buf = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if h is not None:
                seqs.append((h, "".join(buf)))
            h, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if h is not None:
        seqs.append((h, "".join(buf)))
    return seqs


def write_panel(name, acc, seqmatch, clade, gcode, role, cap):
    fna, gff = fetch_asm(acc)
    target = None
    for h, s in read_fna(fna):
        sid = h.split()[0]
        if seqmatch in sid or seqmatch in h:
            target = (sid, s)
            break
    if target is None and "scaffold" in seqmatch:
        # tetrahymena-style assemblies: take the largest scaffold
        target = max(read_fna(fna), key=lambda x: len(x[1]))
    if target is None:
        print(f"  {name}: seqid *{seqmatch}* not in {acc}, skipped")
        return
    sid, seq = target
    if cap and len(seq) > cap:
        seq = seq[:cap]
        cap_end = cap
    else:
        cap_end = len(seq)
    rows = []
    for line in open(gff):
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 9 or c[0] != sid or c[2] != "CDS":
            continue
        s, e = int(c[3]), int(c[4])
        if s >= 1 and e <= cap_end:
            rows.append(line.rstrip("\n"))
    if not rows:
        print(f"  {name}: no CDS rows on {sid}, skipped")
        return
    d = PANEL_DIR / role
    with open(d / f"{name}.fna", "w") as f:
        f.write(f">{sid}\n" + "\n".join(
            seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    with open(d / f"{name}.gff3", "w") as f:
        f.write("##gff-version 3\n")
        f.writelines(r + "\n" for r in rows)
    manifest[name] = {"clade": clade, "gcode": gcode, "acc": acc,
                      "seqid": sid, "len": len(seq),
                      "n_cds": len(rows), "role": role}
    print(f"  {name}: {sid} {len(seq):,} bp, {len(rows)} CDS -> {role}",
          flush=True)


for i, spec in enumerate(PANEL):
    try:
        write_panel(*spec)
    except Exception as ex:
        print(f"  {spec[0]}: FAILED ({ex}) -- skipping", flush=True)
    acc = spec[1]
    if acc not in {s[1] for s in PANEL[i + 1:]}:
        # last panel entry for this accession: release its zip + extract
        shutil.rmtree(DL / acc, ignore_errors=True)
        (DL / f"{acc}.zip").unlink(missing_ok=True)

(PANEL_DIR / "panel.json").write_text(json.dumps(manifest, indent=1))
print(f"\npanel manifest: {len(manifest)} genomes "
      f"({sum(1 for m in manifest.values() if m['role'] == 'train')} "
      f"train / "
      f"{sum(1 for m in manifest.values() if m['role'] == 'heldout')} "
      f"heldout)")

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
