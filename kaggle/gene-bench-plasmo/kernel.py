"""Kaggle CPU kernel: gene-level annotation benchmark on Pf3D7.

Reference + genome are PINNED inputs: PlasmoDB release-71 P. falciparum
3D7 Genome.fasta + .gff uploaded as the daniellecallan/plasmodb71-pf3d7
dataset. Held-out status verified locally against the corpus labels:
GCF_000002765.6 is in v6_val (never trained on by v6 checkpoints) —
it is in v5_train, so v5 checkpoints could NOT be benchmarked here,
but none are run. External tools were never trained on it.

Region: the largest chromosome that fits entirely within MAX_BP (no
truncation), evaluated vs the VEuPathDB GFF at nucleotide / CDS-exon /
transcript levels via scripts/eval_genes.py.

Predictors:
  - OmniPhyl v6 w32 + w64, full model plus each --ablate stream
  - Tiberius (Fungi model — closest available clade; no apicomplexan
    model exists)
  - ANNEVO (local run, Fungi lineage; no apicomplexan lineage exists)

All tool steps are wrapped; failures degrade to status="failed" in
results.json rather than killing the kernel.
"""

import glob
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
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

from eval_genes import evaluate, parse_cds  # noqa: E402

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


# ---------------- Step 1: pinned genome + VEuPathDB reference ------------

FASTA = Path(find_one("*_Genome.fasta"))
REF_GFF = Path(find_one("PlasmoDB-*.gff"))
print(f"genome: {FASTA.name}\nreference: {REF_GFF.name}", flush=True)
out["genome"] = {
    "file": FASTA.name, "species": "Plasmodium falciparum 3D7",
    "refseq_accession": "GCF_000002765.6",
    # verified locally vs corpus labels.tsv + train_omniphyl.build_split
    "held_out": {"v6_val": True, "v5_train": True,
                 "note": "clean for v6 checkpoints; v5 ckpts saw it "
                         "(not evaluated here)"},
}
out["reference"] = {"file": REF_GFF.name,
                    "source": "PlasmoDB release-71 (VEuPathDB)"}

# ---------------- Step 2: pick region (largest chr <= MAX_BP, whole) -----

MAX_BP = 1_600_000  # CPU-feasible for all tools; selects a FULL chromosome

ref_cds = parse_cds(REF_GFF)
n_cds = {k: len(v) for k, v in ref_cds.items()}

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

# prefer the most-CDS chromosome that fits whole; else truncate biggest
fits = [s for s in seqs if len(seqs[s]) <= MAX_BP and n_cds.get(s, 0) > 0]
if fits:
    target_seq = max(fits, key=lambda s: n_cds[s])
    region_seq = seqs[target_seq]
    truncated = False
else:
    target_seq = max(seqs, key=lambda k: n_cds.get(k, 0))
    region_seq = seqs[target_seq][:MAX_BP]
    truncated = True

DATA = WORK / "data"
DATA.mkdir(exist_ok=True)
REGION_FA = DATA / "region.fna"
with open(REGION_FA, "w") as fh:
    fh.write(f">{target_seq}\n")
    for i in range(0, len(region_seq), 70):
        fh.write(region_seq[i:i + 70] + "\n")
REGION_GFF = DATA / "region_ref.gff"
with open(REF_GFF) as fi, open(REGION_GFF, "w") as fo:
    fo.write("##gff-version 3\n")
    for line in fi:
        if line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) >= 9 and c[0] == target_seq \
                and int(c[3]) <= len(region_seq):
            fo.write(line)
print(f"region: {target_seq} ({len(region_seq):,} bp, "
      f"{n_cds.get(target_seq, 0)} ref CDS exons, truncated={truncated})",
      flush=True)
out["region"] = {"seqid": target_seq, "length": len(region_seq),
                 "n_ref_cds_exons": n_cds.get(target_seq, 0),
                 "truncated": truncated}

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


# ---------------- Step 3: OmniPhyl (full + ablations) --------------------

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

# ---------------- Step 4: Tiberius (Fungi clade — closest available) ---

try:
    if sys.version_info < (3, 12):
        sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
        uv = WORK / "tibvenv"
        rc, _ = sh([sys.executable, "-m", "uv", "venv", "--python", "3.12",
                    str(uv)])
        if rc == 0:
            # Tiberius core deps omit bricks2marble (needed by
            # tiberius.main) — it lives in the from_source extra
            sh([str(uv / "bin/pip"), "install", "-q", "tensorflow-cpu",
                "bricks2marble[tf]"])
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
        sh([tpy, "-m", "pip", "install", "-q", str(WORK / "Tiberius"),
            "bricks2marble[tf]"])

    # pick a clade cfg: Fungi first (user call — no protozoan model
    # exists for either tool), then SAR/protist clades as fallback
    rc, cfgs = sh([tpy, str(WORK / "Tiberius/tiberius.py"), "--list_cfg"])
    cand = [c for c in ("fungi", "bacillariophyta", "diatoms",
                        "chlorophyta") if c in cfgs.lower()]
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

# ---------------- Step 5: ANNEVO (local run, Fungi lineage) --------------
# Zenodo precomputed predictions are on RefSeq assemblies (NC_* seqids)
# and would not match the Pf3D7_* seqids of our pinned FASTA, so we run
# ANNEVO locally on the same region file as every other tool.

try:
    sh(["git", "clone", "--depth", "1",
        "https://github.com/xjtu-omics/ANNEVO.git",
        str(WORK / "ANNEVO")])
    models = glob.glob(str(WORK / "ANNEVO/saved_model/*.pt"))
    if not models:  # git-lfs pointers: fetch via raw URL
        fetch("https://github.com/xjtu-omics/ANNEVO/raw/main/"
              "saved_model/ANNEVO_Fungi.pt",
              WORK / "ANNEVO_Fungi.pt", timeout=1800)
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

# ---------------- Step 6: write outputs -----------------------------------

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
