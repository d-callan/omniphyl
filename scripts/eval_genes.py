#!/usr/bin/env python3
"""Gene-level annotation eval: predicted GFF3/GTF vs a reference GFF3/GTF.

Reports the three standard levels used by Tiberius/ANNEVO-style papers:

  nucleotide  union of all CDS intervals, per-base overlap
  exon        CDS segment (seqid,strand,start,end) exact match
  transcript  CDS structure of a transcript matches exactly (same set of
              CDS segments; UTRs ignored)

Sensitivity = matched_ref / n_ref, Specificity = matched_pred / n_pred,
F1 = harmonic mean. Transcripts are grouped by the CDS feature's parent
attribute (Parent= for GFF3, transcript_id= for GTF); CDS rows without a
resolvable parent are each treated as a single-exon transcript.

Usage:
    python eval_genes.py --ref ref.gff3 --pred pred.gff3 [--json out.json]
    # repeat --pred for multiple tools:
    python eval_genes.py --ref ref.gff3 \
        --pred tiberius.gff --pred annevo.gff --pred omniphyl.gff3 --tsv
"""

import argparse
import json
import re
import sys
from collections import defaultdict


def _attrs(s: str) -> dict:
    out = {}
    for tok in s.strip().strip(";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:                       # GFF3
            k, v = tok.split("=", 1)
        elif " " in tok:                     # GTF: key "value"
            k, v = tok.split(None, 1)
            v = v.strip('"')
        else:
            continue
        out[k.strip()] = v.strip()
    return out


def _parent(attr: dict, feat_id: str) -> str:
    # GFF3: Parent=rna-X; multiple parents -> take first. GTF: transcript_id.
    p = attr.get("Parent") or attr.get("transcript_id")
    if p:
        return p.split(",")[0]
    return feat_id


def parse_cds(path: str) -> dict[str, list[tuple[str, str, tuple]]]:
    """{seqid: [(tx_id, strand, (start,end)), ...]} with 0-based half-open
    coords. Only 'CDS' rows (plus 'CDS'-like: 'cds' feature name used by
    some tools) are kept."""
    feats = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "CDS":
                continue
            seqid, strand = c[0], c[6]
            s, e = int(c[3]) - 1, int(c[4])         # -> 0-based half-open
            if e <= s:
                continue
            a = _attrs(c[8])
            tx = _parent(a, f"{seqid}:{s}-{e}")
            feats[seqid].append((tx, strand, (s, e)))
    return feats


def _merge(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _overlap_len(a: list, b: list) -> int:
    """Total bp shared between two merged interval lists."""
    i = j = tot = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def _structures(feats: dict) -> tuple[set, set]:
    """Return (exon_set, transcript_set) over all seqids.
    exon: (seqid, strand, start, end). transcript: (seqid, strand,
    tuple(sorted exons))."""
    exons, txs = set(), set()
    for seqid, rows in feats.items():
        by_tx = defaultdict(list)
        for tx, strand, iv in rows:
            exons.add((seqid, strand, iv[0], iv[1]))
            by_tx[(tx, strand)].append(iv)
        for (tx, strand), ivs in by_tx.items():
            txs.add((seqid, strand, tuple(sorted(set(ivs)))))
    return exons, txs


def _metrics(n_match: int, n_ref: int, n_pred: int) -> dict:
    sn = n_match / n_ref if n_ref else float("nan")
    sp = n_match / n_pred if n_pred else float("nan")
    f1 = (2 * sn * sp / (sn + sp)) if (sn + sp) > 0 else float("nan")
    return {"sn": round(sn, 4), "sp": round(sp, 4), "f1": round(f1, 4),
            "match": n_match, "n_ref": n_ref, "n_pred": n_pred}


def evaluate(ref_path: str, pred_path: str,
             seqid_filter: set = None) -> dict:
    ref = parse_cds(ref_path)
    pred = parse_cds(pred_path)
    if seqid_filter:
        ref = {k: v for k, v in ref.items() if k in seqid_filter}
        pred = {k: v for k, v in pred.items() if k in seqid_filter}

    # nucleotide level: merged CDS unions, overlap counted per seqid
    seqids = set(ref) | set(pred)
    ref_bp = pred_bp = hit_bp = 0
    for sq in seqids:
        r_iv = _merge([iv for _, _, iv in ref.get(sq, [])])
        p_iv = _merge([iv for _, _, iv in pred.get(sq, [])])
        ref_bp += sum(e - s for s, e in r_iv)
        pred_bp += sum(e - s for s, e in p_iv)
        hit_bp += _overlap_len(r_iv, p_iv)

    ref_ex, ref_tx = _structures(ref)
    pred_ex, pred_tx = _structures(pred)

    return {
        "nucleotide": _metrics(hit_bp, ref_bp, pred_bp),
        "exon": _metrics(len(ref_ex & pred_ex), len(ref_ex), len(pred_ex)),
        "transcript": _metrics(len(ref_tx & pred_tx), len(ref_tx),
                               len(pred_tx)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference GFF3/GTF")
    ap.add_argument("--pred", required=True, action="append",
                    help="predicted GFF3/GTF (repeatable)")
    ap.add_argument("--names", nargs="*", default=None,
                    help="tool names matching --pred order (default: file stems)")
    ap.add_argument("--seqids", nargs="*", default=None,
                    help="restrict eval to these seqids")
    ap.add_argument("--tsv", action="store_true", help="print TSV table")
    ap.add_argument("--json", default=None, help="write full results JSON")
    args = ap.parse_args()

    names = args.names or [re.sub(r"\.(gff3?|gtf)(\.gz)?$", "", p.rsplit("/", 1)[-1])
                           for p in args.pred]
    if len(names) != len(args.pred):
        sys.exit("--names must match --pred count")

    seqid_filter = set(args.seqids) if args.seqids else None
    out = {}
    for name, pp in zip(names, args.pred):
        try:
            out[name] = evaluate(args.ref, pp, seqid_filter)
        except Exception as e:
            out[name] = {"error": str(e)}

    hdr = ["tool", "level", "sn", "sp", "f1", "match", "n_ref", "n_pred"]
    rows = [hdr]
    for name, res in out.items():
        if "error" in res:
            rows.append([name, "ERROR", res["error"], "", "", "", "", ""])
            continue
        for lvl in ("nucleotide", "exon", "transcript"):
            m = res[lvl]
            rows.append([name, lvl, m["sn"], m["sp"], m["f1"],
                         m["match"], m["n_ref"], m["n_pred"]])
    w = [max(len(str(r[i])) for r in rows) for i in range(len(hdr))]
    sep = "\t" if args.tsv else "  "
    for r in rows:
        print(sep.join(str(x).ljust(w[i]) for i, x in enumerate(r)).rstrip())

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
