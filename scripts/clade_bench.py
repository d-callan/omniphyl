#!/usr/bin/env python3
"""Cross-clade contig benchmark for the OmniPhyl refiner.

For each genome in GENOMES (all held out of refiner training):
  1. fetch FASTA + GFF (NCBI datasets v2 REST, or VEuPathDB release
     download URLs), caching under --workdir/data
  2. slice to the seqid carrying the most CDS bases ("the contig") and
     write region.fna / region_ref.gff
  3. run `omniphyl annotate` in a subprocess (OMNIPHYL_DUMP_RES=1 so a
     res.pkl cache lands next to the GFF3)
  4. score refiner genes vs the sliced reference (nucleotide / exon /
     transcript F1 via eval_genes) and run the eval_errors error-class
     breakdown

Outputs <workdir>/results.json + results.md; every row records its
source URL and the slice transform (provenance).

Usage:
  python scripts/clade_bench.py --workdir /tmp/clade_eval \
      --model ~/omniphyl-work/v6-ckpts-dataset/w64_e15_best.pt \
      --refiner /tmp/v5_out/refiner_multi.pt
"""
import argparse
import gzip
import io
import json
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_genes import evaluate  # noqa: E402

NCBI_DL = ("https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
           "{acc}/download?include_annotation_type=GENOME_FASTA,"
           "GENOME_GFF&filename={acc}.zip")

# All held out of the 8-genome refiner training set.
GENOMES = [
    {"name": "bsubtilis", "clade": "bacteria", "acc": "GCF_000009045.1"},
    {"name": "scerevisiae", "clade": "fungi", "acc": "GCF_000146045.2"},
    {"name": "tbrucei", "clade": "kinetoplastida", "acc": "GCF_000002445.2"},
    {"name": "ddiscoideum", "clade": "amoebozoa", "acc": "GCF_000004695.1"},
    {"name": "mjannaschii", "clade": "archaea", "acc": "GCF_000091665.1"},
    {"name": "pvivax", "clade": "apicomplexa",
     "veupathdb": {"base": "https://plasmodb.org/common/downloads/release-71",
                   "org": "PvivaxP01", "proj": "PlasmoDB-71"}},
]


def _url(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"  fetch {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return dest


def fetch(spec, data_dir):
    """Return (fna_path, gff_path)."""
    if "veupathdb" in spec:
        v = spec["veupathdb"]
        base = f"{v['base']}/{v['org']}"
        fna = _url(f"{base}/fasta/data/{v['proj']}_{v['org']}_Genome.fasta",
                   data_dir / f"{spec['name']}.fna")
        gff = _url(f"{base}/gff/data/{v['proj']}_{v['org']}.gff",
                   data_dir / f"{spec['name']}.gff")
        return fna, gff
    z = _url(NCBI_DL.format(acc=spec["acc"]), data_dir / f"{spec['acc']}.zip")
    with zipfile.ZipFile(z) as zf:
        fna_n = next(n for n in zf.namelist()
                     if n.endswith("_genomic.fna") and "cds_from" not in n)
        gff_n = next(n for n in zf.namelist() if n.endswith("genomic.gff"))
        fna = data_dir / f"{spec['name']}.fna"
        gff = data_dir / f"{spec['name']}.gff"
        if not fna.exists():
            fna.write_bytes(zf.read(fna_n))
        if not gff.exists():
            gff.write_bytes(zf.read(gff_n))
    return fna, gff


def pick_contig(gff_path):
    """Seqid carrying the most CDS bases (gene-dense = hard test)."""
    bases = {}
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) >= 9 and p[2] == "CDS":
                bases[p[0]] = bases.get(p[0], 0) + (int(p[4]) - int(p[3]) + 1)
    return max(bases, key=bases.get)


def slice_region(fna_path, gff_path, seqid, out_dir):
    """Write region.fna + region_ref.gff for one seqid."""
    region_fna = out_dir / "region.fna"
    region_gff = out_dir / "region_ref.gff"
    seq, found = [], False
    with open(fna_path) as f:
        for line in f:
            if line.startswith(">"):
                if found:
                    break
                found = line[1:].split()[0] == seqid
                if found:
                    seq.append(line)
            elif found:
                seq.append(line)
    if not found:
        raise RuntimeError(f"{seqid} not in {fna_path}")
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt", errors="replace") as f, \
            open(region_gff, "w") as o:
        o.writelines(l for l in f
                     if l.startswith("#") or l.split("\t", 1)[0] == seqid)
    with open(region_fna, "w") as o:
        o.writelines(seq)
    return region_fna, region_gff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--refiner", required=True, type=Path)
    ap.add_argument("--only", default=None,
                    help="comma list of names to run (default all)")
    a = ap.parse_args()
    wd = a.workdir
    (wd / "data").mkdir(parents=True, exist_ok=True)
    omniphyl = "/home/danielle/miniconda3/envs/omniphyl/bin/omniphyl"
    results = {}

    only = set(a.only.split(",")) if a.only else None
    for spec in GENOMES:
        if only and spec["name"] not in only:
            continue
        name = spec["name"]
        odir = wd / name
        odir.mkdir(exist_ok=True)
        print(f"\n=== {name} ({spec['clade']}) ===", file=sys.stderr)
        try:
            fna, gff = fetch(spec, wd / "data")
            seqid = pick_contig(gff)
            rf, rg = slice_region(fna, gff, seqid, odir)
            if not (odir / f"{name}.res.pkl").exists():
                anno_kw = json.dumps({
                    "model_path": str(a.model), "fasta_path": str(rf),
                    "out_prefix": str(odir / name),
                    "refiner_path": str(a.refiner), "verbose": True})
                subprocess.run(
                    [sys.executable, "-c",
                     "import sys, json\n"
                     "from pathlib import Path\n"
                     "from omniphyl.annotate import run_annotate\n"
                     "kw = json.loads(sys.argv[1])\n"
                     "kw['out_prefix'] = Path(kw['out_prefix'])\n"
                     "run_annotate(**kw)\n",
                     anno_kw],
                    env=env, check=True)
            res = {"clade": spec["clade"], "seqid": seqid,
                   "src": spec.get("acc") or spec["veupathdb"]["base"],
                   "metrics": evaluate(str(rg), str(odir / f"{name}.gff3"))}
            err = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "eval_errors.py"),
                 str(odir / f"{name}.res.pkl"), str(rg), str(a.refiner),
                 "--out", str(odir / "errors.json")],
                capture_output=True, text=True, env=env)
            res["errors"] = err.stdout if err.returncode == 0 else err.stderr
            results[name] = res
            m = res["metrics"]
            print(f"  {seqid}: nuc {m['nucleotide']['f1']:.3f} "
                  f"exon {m['exon']['f1']:.3f} tx {m['transcript']['f1']:.3f}")
        except Exception as e:
            results[name] = {"clade": spec["clade"], "error": str(e)}
            print(f"  FAILED: {e}", file=sys.stderr)

    (wd / "results.json").write_text(json.dumps(results, indent=1))
    lines = ["| organism | clade | contig | nuc F1 | exon F1 | tx F1 |",
             "|---|---|---|---|---|---|"]
    for n, r in results.items():
        if "metrics" in r:
            m = r["metrics"]
            lines.append(f"| {n} | {r['clade']} | {r['seqid']} | "
                         f"{m['nucleotide']['f1']:.3f} | "
                         f"{m['exon']['f1']:.3f} | {m['transcript']['f1']:.3f} |")
        else:
            lines.append(f"| {n} | {r['clade']} | - | {r['error'][:60]} | | |")
    (wd / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
