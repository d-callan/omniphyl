"""Kaggle CPU kernel: gene-level annotation benchmark on held-out Plasmodium.

Picks a Plasmodium accession from the leak-free v6 val cohort (clean_val_accs
logic = eval-v5-v6 kernel), slices the chromosome with the most CDS features,
and compares predicted gene structures against the VEuPathDB reference
(RefSeq GFF as secondary) at nucleotide / CDS-exon / transcript levels via
scripts/eval_genes.py.

Predictors:
  - OmniPhyl v6 w32 + w64, full model plus each --ablate stream
  - Tiberius (closest available clade: Diatoms/Bacillariophyta; no
    apicomplexan model exists)
  - ANNEVO (Zenodo precomputed predictions when the accession is in their
    566-assembly set, else local run with the Fungi lineage model; no
    apicomplexan lineage exists)

All tool steps are wrapped; failures degrade to status="failed" in
results.json rather than killing the kernel.
"""

import glob
import gzip
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
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
RESULT = WORK / "results.json"
os.environ.setdefault("OMP_NUM_THREADS", "4")

subprocess.run(["git", "clone", "--depth", "1", REPO, str(WORK / "omniphyl")],
               check=True)
sys.path.insert(0, str(WORK / "omniphyl/scripts"))
sys.path.insert(0, str(WORK / "omniphyl/src"))

from eval_genes import evaluate  # noqa: E402
from eval_pilot import load_labels  # noqa: E402
from train_omniphyl import build_split  # noqa: E402

status = {}          # tool -> "ok" / "skipped: reason" / "failed: reason"
out = {"status": status, "results": {}}


def note(tool: str, st: str):
    status[tool] = st
    print(f"[{tool}] {st}", flush=True)


def sh(cmd: list, env: dict = None, cwd=None, timeout: int = 7200):
    """Run cmd, return (rc, tail). Never raises on nonzero rc."""
    print("+", " ".join(str(c) for c in cmd)[:300], flush=True)
    try:
        p = subprocess.run(cmd, env=env or os.environ.copy(), cwd=cwd,
                           timeout=timeout, capture_output=True, text=True)
        tail = (p.stdout + p.stderr)[-2000:]
        if p.returncode:
            print(tail[-1000:], flush=True)
        return p.returncode, tail
    except Exception as e:
        return 1, str(e)


def fetch(url: str, dest: Path, timeout: int = 600) -> Path:
    print(f"GET {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    return dest


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern}; mounted={os.listdir('/kaggle/input')}")
    return hits[0]


def find_corpus(tag: str) -> str:
    hits = sorted(glob.glob("/kaggle/input/**/labels.tsv", recursive=True))
    for h in hits:
        if tag in h:
            return h
    raise SystemExit(f"corpus '{tag}' labels.tsv not found")


# ---------------- Step 1: held-out Plasmodium accession ------------------

v5_lab = find_corpus("v5-20k")
v6_lab = find_corpus("omniphyl-v6")

SPLIT_COLS = {"accession", "rank_unresolved", "genus", "species",
              "seq_source", "seqid", "start"}
v5_meta = load_labels(Path(v5_lab), cols=SPLIT_COLS)
v5_val_accs, v5_hold_gen, _ = build_split(v5_meta, 0.1, 0.1, 0, 0.1,
                                          legacy=True)
v5_all = {m["accession"] for m in v5_meta.values()}
v5_train = {m["accession"] for m in v5_meta.values()
            if m["accession"] not in v5_val_accs
            and m.get("genus", "") not in v5_hold_gen}
del v5_meta

v6_meta = load_labels(Path(v6_lab), cols=SPLIT_COLS)
v6_val_accs, _, _ = build_split(v6_meta, 0.05, 0.02, 0, 0.05, legacy=False)
clean_val_accs = set(v6_val_accs) - v5_train

plasmo = sorted({m["accession"]: (m.get("genus", ""), m.get("species", ""))
                 for m in v6_meta.values()
                 if m["accession"] in clean_val_accs
                 and "plasmodium" in
                 (m.get("genus", "") + " " + m.get("species", "")).lower()}
                .items(),
                # P. falciparum first (best-curated VEuPathDB reference)
                key=lambda kv: ("falciparum" not in kv[1][1].lower(), kv[0]))
del v6_meta
if not plasmo:
    raise SystemExit("no held-out Plasmodium accession in clean_val_accs")
print(f"held-out Plasmodium candidates: {len(plasmo)}", flush=True)

# ---------------- Step 2: genome FASTA + reference GFFs ------------------

DATA = WORK / "data"
DATA.mkdir(exist_ok=True)
ACC = FASTA = REFSEQ_GFF = None
SPECIES = ""

for acc, (genus, species) in plasmo:
    z = DATA / f"{acc}.zip"
    try:
        fetch(f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
              f"{acc}/download?include_annotation_type=GENOME_FASTA,"
              f"GENOME_GFF", z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(DATA / acc)
        base = DATA / acc / "ncbi_dataset" / "data" / acc
        fasta = next(base.glob("*_genomic.fna"))
        gff = next((DATA / acc).rglob("genomic.gff"), None)
        if gff is None:
            continue
        ACC, FASTA, REFSEQ_GFF, SPECIES = acc, fasta, gff, species
        break
    except Exception as e:
        print(f"  {acc}: {e}", flush=True)
        continue
if ACC is None:
    raise SystemExit("no Plasmodium accession downloaded successfully")
print(f"benchmark genome: {ACC} ({SPECIES})", flush=True)
out["genome"] = {"accession": ACC, "species": SPECIES}

# VEuPathDB GTF via UCSC GenArk (same source as corpus labels tier 1),
# seqids bridged chromAlias -> RefSeq names used in the FASTA.
VEUP_GFF = DATA / "ref_veupathdb.gff"
try:
    from fetch_refseq_corpus import _ucsc_gtf_url, load_chrom_alias
    url = _ucsc_gtf_url(ACC)
    if url:
        fetch(url, DATA / "veupath.gtf.gz")
        aliases = load_chrom_alias(ACC, DATA)
        n_bridged = n_lines = 0
        with gzip.open(DATA / "veupath.gtf.gz", "rt") as fi, \
                open(VEUP_GFF, "w") as fo:
            fo.write("##gff-version 3\n")
            for line in fi:
                if line.startswith("#"):
                    continue
                c = line.rstrip("\n").split("\t")
                if len(c) < 9:
                    continue
                grp = aliases.get(c[0], frozenset())
                hit = next((a for a in grp if a.startswith("NC_")), None)
                if hit:
                    c[0] = hit
                    n_bridged += 1
                fo.write("\t".join(c) + "\n")
                n_lines += 1
        note("veupathdb_ref", f"ok ({n_bridged}/{n_lines} bridged)")
    else:
        note("veupathdb_ref", "skipped: no veupathGenes.gtf on hub")
except Exception as e:
    note("veupathdb_ref", f"failed: {e}")
    VEUP_GFF = None

# ---------------- Step 3: pick region (chromosome with most CDS) ---------

sys.path.insert(0, str(WORK / "omniphyl/src"))
from eval_genes import parse_cds  # noqa: E402

ref = REFSEQ_GFF
if VEUP_GFF and VEUP_GFF.exists():
    cds = parse_cds(VEUP_GFF)
    if sum(len(v) for v in cds.values()) >= 50:
        ref = VEUP_GFF
out["reference_gff"] = str(ref)
ref_cds = parse_cds(ref)
target_seq = max(ref_cds, key=lambda k: len(ref_cds[k]))
MAX_BP = 3_000_000

seqs, cur, name = {}, [], None
with open(FASTA) as fh:
    for line in fh:
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(cur)
            name, cur = line[1:].split()[0], []
        else:
            cur.append(line.strip())
    if name:
        seqs[name] = "".join(cur)
if target_seq not in seqs:
    target_seq = max(seqs, key=lambda k: len(seqs[k]))
region_seq = seqs[target_seq][:MAX_BP]
REGION_FA = DATA / "region.fna"
with open(REGION_FA, "w") as fh:
    fh.write(f">{target_seq}\n")
    for i in range(0, len(region_seq), 70):
        fh.write(region_seq[i:i + 70] + "\n")
REGION_GFF = DATA / "region_ref.gff"
with open(ref) as fi, open(REGION_GFF, "w") as fo:
    fo.write("##gff-version 3\n")
    for line in fi:
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) >= 9 and c[0] == target_seq \
                and int(c[3]) <= len(region_seq):
            fo.write(line)
n_ref_cds = len(ref_cds.get(target_seq, []))
print(f"region: {target_seq} ({len(region_seq):,} bp, "
      f"{n_ref_cds} ref CDS exons)", flush=True)
out["region"] = {"seqid": target_seq, "length": len(region_seq),
                 "n_ref_cds_exons": n_ref_cds}

PRED_GFFS: dict[str, Path] = {}


def add_eval(tool: str, gff: Path):
    if gff is None or not Path(gff).exists() or Path(gff).stat().st_size < 50:
        note(tool, "failed: empty/missing GFF")
        return
    PRED_GFFS[tool] = Path(gff)
    try:
        out["results"][tool] = evaluate(str(REGION_GFF), str(gff))
        note(tool, "ok")
    except Exception as e:
        note(tool, f"failed eval: {e}")


# ---------------- Step 4: OmniPhyl (full + ablations) --------------------

from omniphyl.annotate import run_annotate  # noqa: E402

CKPTS = {"v6_w64": find_one("w64_e15_best.pt"),
         "v6_w32": find_one("w32_e15_best.pt")}
ABLATIONS = ["", "frame", "flags", "region", "coding", "boundary"]

for ck_name, ckpt in CKPTS.items():
    for abl in ABLATIONS:
        tag = f"omniphyl_{ck_name}" + (f"_no-{abl}" if abl else "")
        outp = WORK / f"omniphyl_{ck_name}{'_' + abl if abl else ''}"
        try:
            run_annotate(ckpt, str(REGION_FA), outp,
                         kingdom="eukaryote", genetic_code="auto",
                         ablate=abl, verbose=False)
            add_eval(tag, Path(str(outp) + ".gff3"))
        except Exception as e:
            note(tag, f"failed: {e}")

# ---------------- Step 5: Tiberius (Diatoms clade — closest SAR) ---------

try:
    if sys.version_info < (3, 12):
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = WORK / "tibvenv"
        rc, _ = sh([sys.executable, "-m", "uv", "venv", "--python", "3.12",
                    str(uv)])
        if rc == 0:
            sh([str(uv / "bin/pip"), "install", "-q", "tensorflow-cpu"])
            rc, _ = sh(["git", "clone", "--depth", "1",
                        "https://github.com/Gaius-Augustus/Tiberius",
                        str(WORK / "Tiberius")])
            if rc == 0:
                sh([str(uv / "bin/pip"), "install", "-q",
                    str(WORK / "Tiberius")])
        tpy = str(uv / "bin/python")
    else:
        tpy = sys.executable
        sh(["git", "clone", "--depth", "1",
            "https://github.com/Gaius-Augustus/Tiberius",
            str(WORK / "Tiberius")])
        sh([tpy, "-m", "pip", "install", "-q", str(WORK / "Tiberius")])

    # pick a clade cfg: prefer diatoms/bacillariophyta (SAR supergroup,
    # closest to Apicomplexa among Tiberius's 7 clades), else fungi
    rc, cfgs = sh([tpy, str(WORK / "Tiberius/tiberius.py"), "--list_cfg"])
    cand = [c for c in ("bacillariophyta", "diatoms", "chlorophyta",
                        "fungi") if c in cfgs.lower()]
    if not cand:
        cand = ["fungi"]
    rc, tail = sh([tpy, str(WORK / "Tiberius/tiberius.py"),
                   "--genome", str(REGION_FA),
                   "--model_cfg", cand[0],
                   "--out", str(WORK / "tiberius.gtf")],
                  timeout=10800)
    if rc == 0:
        add_eval("tiberius_" + cand[0], WORK / "tiberius.gtf")
    else:
        note("tiberius", f"failed rc={rc}")
except Exception as e:
    note("tiberius", f"failed: {e}")

# ---------------- Step 6: ANNEVO -----------------------------------------
# (a) Zenodo precomputed predictions on 566 RefSeq assemblies (their own
#     benchmark output — authoritative); (b) local run, Fungi lineage.

annevo_done = False
try:
    j = json.loads(urllib.request.urlopen(
        "https://zenodo.org/api/records/16749615", timeout=60).read())
    for f in j.get("files", []):
        fname = f.get("key", "")
        if ACC.split(".")[0] in fname:
            zf_path = DATA / fname
            fetch(f["links"]["self"], zf_path, timeout=1800)
            if fname.endswith(".zip"):
                with zipfile.ZipFile(zf_path) as zf:
                    zf.extractall(DATA / "annevo_pred")
                hits = list((DATA / "annevo_pred").rglob("*.gff*"))
            else:
                hits = [zf_path]
            for h in hits:
                if h.suffix == ".gz":
                    with gzip.open(h, "rt") as fi, \
                            open(DATA / "annevo.gff", "w") as fo:
                        fo.write(fi.read())
                    h = DATA / "annevo.gff"
                add_eval("annevo_zenodo", h)
                annevo_done = True
                break
            break
except Exception as e:
    print(f"zenodo lookup failed: {e}", flush=True)

if not annevo_done:
    try:
        sh(["git", "clone", "--depth", "1",
            "https://github.com/xjtu-omics/ANNEVO.git",
            str(WORK / "ANNEVO")])
        models = glob.glob(str(WORK / "ANNEVO/saved_model/*.pt"))
        if not models:  # git-lfs pointers: fetch via raw URL
            for lin in ("Fungi",):
                fetch(f"https://github.com/xjtu-omics/ANNEVO/raw/main/"
                      f"saved_model/ANNEVO_{lin}.pt",
                      WORK / f"ANNEVO_{lin}.pt", timeout=1800)
            models = [str(WORK / "ANNEVO_Fungi.pt")]
        sh([sys.executable, "-m", "pip", "install", "-q",
            "h5py", "numba", "bcbio-gff", "torchmetrics",
            "biopython", "pandas"], timeout=1800)
        model = next((m for m in models if "Fungi" in m), models[0])
        rc, tail = sh([sys.executable,
                       str(WORK / "ANNEVO/annotation.py"),
                       "-g", str(REGION_FA), "-m", model,
                       "-l", "Fungi",
                       "-o", str(WORK / "annevo.gff"),
                       "-t", "4", "--batch_size", "8", "--show_log"],
                      timeout=10800)
        if rc == 0:
            add_eval("annevo_fungi", WORK / "annevo.gff")
        else:
            note("annevo", f"failed rc={rc}")
    except Exception as e:
        note("annevo", f"failed: {e}")

# ---------------- Step 7: write outputs -----------------------------------

(WORK / "pred_gffs.json").write_text(
    json.dumps({k: str(v) for k, v in PRED_GFFS.items()}, indent=2))
RESULT.write_text(json.dumps(out, indent=2))
print("\n==== SUMMARY ====", flush=True)
for tool, res in out["results"].items():
    if "error" in res:
        continue
    t = res["transcript"]
    print(f"{tool:32s} txSn={t['sn']:.3f} txSp={t['sp']:.3f} "
          f"exonSn={res['exon']['sn']:.3f} nucSn={res['nucleotide']['sn']:.3f}",
          flush=True)
print(json.dumps(status, indent=2), flush=True)
shutil.rmtree(WORK / "omniphyl", ignore_errors=True)
