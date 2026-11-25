#!/usr/bin/env python3
"""Re-annotate `label_source=reconstructed` labels rows with real stats.

Rows recovered from timeout-killed kernels (labels.tsv lost to Kaggle's
output capture) carry real fids/coords/taxonomy but NA for every
annotation-derived stat. This re-derives them from the same sources and
functions slice_assembly uses, so patched rows are emit-identical:

  group in VEUPATH_GROUPS and hub hit:
      fetch_veupathdb_cds + load_chrom_alias -> veupath_intervals
      -> coding_stats_iv            (label_source = veupathdb)
  otherwise:
      {ftp_dir}/{name}_genomic.gff.gz -> load_gff_features
      -> feature_stats              (label_source = refseq_gff)
  no annotation:                    (label_source = none, stats NA)

Caveats:
  * replicon is NOT recoverable without the genome FASTA header;
    reconstructed rows keep their placeholder value.
  * gcode upgrades to the GFF per-seqid gcodes majority when a GFF is
    present (fetcher-exact); otherwise keeps the taxid-LUT value.

Usage:
  python scripts/reannotate_labels.py --corpus DIR --map reannotate_map.tsv \
      [--out labels_patched.tsv] [--work /tmp/reannotate_work]
      [--only-accs GCF_000001,GCA_000002]   # debug subset
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_refseq_corpus as F

VEUPATH_GROUPS = {"protozoa", "fungi", "invertebrate"}

# labels.tsv column indexes (see fetch_refseq_corpus emit order)
C_FID, C_ACC, C_SEQID, C_S, C_E = 0, 1, 2, 3, 4
C_CODING, C_SRC = 10, 11
C_FRAC, C_BND, C_NB, C_BOFF = 12, 13, 14, 15
C_INTR, C_RNA, C_FRAME, C_FLAGS, C_DOMS = 23, 24, 25, 26, 27
C_GCODE = 29


def _apply(row, stats, tier):
    """Write emit-formatted stat columns into a labels row in place."""
    row[C_SRC] = tier
    if stats is None:
        row[C_CODING] = "NA"
        row[C_FRAC] = row[C_BND] = row[C_NB] = row[C_BOFF] = ""
        for c in (C_INTR, C_RNA, C_FRAME, C_FLAGS, C_DOMS):
            row[c] = "NA"
        return
    if isinstance(stats, dict):
        frac = stats["frac"]
        row[C_FRAC] = f"{frac:.4f}"
        row[C_BND] = "1" if 0 < frac < 1 else "0"
        row[C_NB] = str(stats["n_bnd"])
        row[C_BOFF] = (f"{stats['boff']:.4f}"
                       if stats["boff"] is not None else "")
        row[C_INTR] = f"{stats['intron']:.4f}"
        row[C_RNA] = f"{stats['rna']:.4f}"
        row[C_FRAME] = str(stats["frame"])
        row[C_FLAGS] = str(stats["flags"])
        row[C_DOMS] = ",".join(stats["domains"])
    else:
        frac, n_bnd, boff = stats
        row[C_FRAC] = f"{frac:.4f}"
        row[C_BND] = "1" if 0 < frac < 1 else "0"
        row[C_NB] = str(n_bnd)
        row[C_BOFF] = f"{boff:.4f}" if boff is not None else ""
        for c in (C_INTR, C_RNA, C_FRAME, C_FLAGS, C_DOMS):
            row[c] = "NA"
    row[C_CODING] = "coding" if frac >= 0.5 else "non-coding"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True,
                    help="dir containing labels.tsv (+ _ncbi_cache)")
    ap.add_argument("--map", required=True,
                    help="accession/group/seq_source/ftp_path TSV")
    ap.add_argument("--out", help="default <corpus>/labels_patched.tsv")
    ap.add_argument("--work", default="/tmp/reannotate_work")
    ap.add_argument("--only-accs", default="")
    args = ap.parse_args()

    corp = Path(args.corpus)
    cache = corp / "_ncbi_cache"
    cache.mkdir(exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else corp / "labels_patched.tsv"

    only = set(args.only_accs.split(",")) if args.only_accs else None

    meta = {}
    with open(args.map) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["ftp_path"] and (only is None or r["accession"] in only):
                meta[r["accession"]] = r
    print(f"{len(meta)} accessions to re-annotate", file=sys.stderr)

    with open(corp / "labels.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        hdr = next(r)
        rows = [row for row in r if len(row) == len(hdr)]
    # acc -> seqid -> [(s, e, row)]
    todo: dict[str, dict[str, list]] = {}
    n_todo = 0
    for row in rows:
        if row[C_SRC] != "reconstructed" or row[C_ACC] not in meta:
            continue
        todo.setdefault(row[C_ACC], {}).setdefault(
            row[C_SEQID], []).append((int(row[C_S]), int(row[C_E]), row))
        n_todo += 1
    print(f"{n_todo} reconstructed rows across {len(todo)} accessions",
          file=sys.stderr)

    tiers = {"veupathdb": 0, "refseq_gff": 0, "none": 0, "failed": 0}
    n_fixed = 0
    for i, (acc, seqrows) in enumerate(todo.items()):
        m = meta[acc]
        try:
            # slice_assembly order: fetch the GFF first on every accession
            # (it also feeds per-seqid gcode votes even when the veupath
            # tier emits the stats), then decide the stats tier.
            feats = {}
            ftp_dir = m["ftp_path"].replace("ftp://", "https://").rstrip("/")
            name = ftp_dir.rsplit("/", 1)[-1]
            gff = work / f"{acc}.gff.gz"
            try:
                F.download(f"{ftp_dir}/{name}_genomic.gff.gz", gff)
                feats = F.load_gff_features(gff)
            except Exception:
                pass
            gcode_of = (lambda sid:
                        (v.most_common(1)[0][0]
                         if (v := feats.get(sid, {}).get("gcodes"))
                         else None))
            veu = (F.fetch_veupathdb_cds(acc, cache)
                   if m["group"] in VEUPATH_GROUPS else None)
            if veu is not None:
                tier = "veupathdb"
                alias = F.load_chrom_alias(acc, cache)
                stats_of = (lambda sid, s, e:
                            None if (iv := F.veupath_intervals(
                                veu, alias, sid)) is None
                            else F.coding_stats_iv(iv, s, e))
            elif feats:
                tier = "refseq_gff"
                stats_of = lambda sid, s, e: F.feature_stats(
                    feats, sid, s, e)
            else:
                tier, stats_of = "none", (lambda sid, s, e: None)
            for sid, frs in seqrows.items():
                gc = gcode_of(sid)
                for s, e, row in frs:
                    _apply(row, stats_of(sid, s, e), tier)
                    if gc is not None:
                        row[C_GCODE] = str(gc)
                    n_fixed += 1
            tiers[tier] += 1
        except Exception as e:
            tiers["failed"] += 1
            print(f"  FAILED {acc}: {e}", file=sys.stderr)
        finally:
            (work / f"{acc}.gff.gz").unlink(missing_ok=True)
            (work / f"{acc}.gff.gz.part").unlink(missing_ok=True)
            for suf in (".veupathGenes.gtf.gz", ".chromAlias.txt",
                        ".veupath_genes.json"):
                (cache / f"{acc}{suf}").unlink(missing_ok=True)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(todo)} accs, {n_fixed} rows patched",
                  file=sys.stderr)

    with open(out, "w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(hdr)
        w.writerows(rows)
    print(f"patched {n_fixed} rows -> {out}")
    print("tiers:", tiers)


if __name__ == "__main__":
    main()
