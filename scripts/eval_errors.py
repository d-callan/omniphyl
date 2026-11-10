#!/usr/bin/env python3
"""Error-class analysis for refiner predictions.

Decodes genes from a cached res.pkl (stage-1 tracks) with the refiner,
then classifies every predicted transcript and every reference transcript
into error classes and stratifies errors by gene properties.

Pred-side classes (precedence order):
  exact          exon structures identical to a ref tx
  merged         merged CDS span overlaps >=2 distinct ref txs
  split_part     shares a ref tx with >=1 other pred gene
  exon_count     same ref tx, different exon count
  boundary       same ref tx, same exon count, different boundaries
  extra          zero CDS overlap with any ref tx

Ref-side classes:
  exact          matched exactly by a pred
  wrong_struct   overlapped by a pred but structure differs
  missed         no pred CDS overlap at all

Strata for wrong/missed genes: intron count (0 / 1 / 2+), ref CDS length
quartiles, local gene density (bp to nearest ref CDS on the contig).

Usage:
  python scripts/eval_errors.py res.pkl ref.gff refiner.pt [--out errors.json]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_genes import parse_cds, _merge, _overlap_len
from omniphyl.annotate import _exon_bounds, STOPS
from omniphyl.refiner import load_refiner, decode_with_refiner


def _spans(rows):
    """txid/strand/merged span per transcript row group."""
    out = {}
    for txid, strand, iv in rows:
        if txid not in out:
            out[txid] = {"strand": strand, "exons": []}
        out[txid]["exons"].append(iv)
    for t in out.values():
        t["exons"].sort()
        t["span"] = (min(s for s, _ in t["exons"]), max(e for _, e in t["exons"]))
    return out


def _iv_overlap(a, b):
    """Overlap bp between two sorted interval lists."""
    return _overlap_len(_merge(a), _merge(b))


def classify_contig(seqid, ref_rows, genes, seq):
    ref = _spans(ref_rows.get(seqid, []))
    preds = []
    for i, g in enumerate(genes):
        exons = [(s, e) for s, e in _exon_bounds(g)]
        preds.append({"gid": f"g{i}", "strand": g["strand"], "exons": exons,
                      "span": (min(s for s, _ in exons), max(e for _, e in exons))})

    # pred -> overlapping ref txs (same strand only for structure match)
    for p in preds:
        p["ref_hits"] = [tid for tid, t in ref.items()
                         if t["strand"] == p["strand"]
                         and _iv_overlap(p["exons"], t["exons"]) > 0]
        p["anti_hits"] = [tid for tid, t in ref.items()
                          if t["strand"] != p["strand"]
                          and _iv_overlap(p["exons"], t["exons"]) > 0]

    # ref -> overlapping preds
    for tid, t in ref.items():
        t["pred_hits"] = [p for p in preds
                          if p["strand"] == t["strand"]
                          and _iv_overlap(p["exons"], t["exons"]) > 0]
        t["exact"] = any(sorted(p["exons"]) == sorted(t["exons"])
                         for p in t["pred_hits"])

    for p in preds:
        p["class"] = "extra"
        if p["ref_hits"]:
            if len(p["ref_hits"]) >= 2:
                p["class"] = "merged"
            elif len(ref[p["ref_hits"][0]]["pred_hits"]) >= 2:
                p["class"] = "split_part"
            else:
                t = ref[p["ref_hits"][0]]
                if sorted(p["exons"]) == sorted(t["exons"]):
                    p["class"] = "exact"
                elif len(p["exons"]) != len(t["exons"]):
                    p["class"] = "exon_count"
                else:
                    p["class"] = "boundary"
        elif p["anti_hits"]:
            p["class"] = "antisense_only"

    for tid, t in ref.items():
        t["class"] = "exact" if t["exact"] else (
            "wrong_struct" if t["pred_hits"] else "missed")

    # strata
    n_ex = lambda t: len(t["exons"])
    tids = list(ref)
    for tid in tids:
        t = ref[tid]
        t["n_intron"] = n_ex(t) - 1
        t["cds_len"] = sum(e - s for s, e in t["exons"])
        s0, e0 = t["span"]
        t["nearest_gap"] = min(
            (abs(s0 - u["span"][1]) if u["span"][1] <= s0 else
             abs(u["span"][0] - e0) if u["span"][0] >= e0 else 0)
            for u in (ref[u2] for u2 in tids if u2 != tid)
        ) if len(tids) > 1 else None
        if seq:
            t["gc"] = (seq[s0:e0].count("G") + seq[s0:e0].count("C")) / max(1, e0 - s0)

    return {"preds": preds, "ref": ref}


def _len_bin(n):
    return ("<300" if n < 300 else "300-900" if n < 900 else
            "900-1800" if n < 1800 else ">=1800")


def _gap_bin(n):
    return ("overlap/0" if n == 0 else "1-50" if n <= 50 else
            "51-500" if n <= 500 else ">500")


def summarize(per_contig):
    pred_classes, ref_classes = {}, {}
    wrong_ref = {"n_intron": {"0": 0, "1": 0, "2+": 0},
                 "cds_len": {}, "gap": {}, "gc_bin": {}}
    missed_ref = {"n_intron": {"0": 0, "1": 0, "2+": 0},
                  "cds_len": {}, "gap": {}, "gc_bin": {}}
    totals = {"ref": 0, "ref_intron0": 0, "pred": 0}
    boundary_bp_err, exon_delta = [], []

    for c in per_contig.values():
        ref, preds = c["ref"], c["preds"]
        totals["pred"] += len(preds)
        totals["ref"] += len(ref)
        totals["ref_intron0"] += sum(1 for t in ref.values() if t["n_intron"] == 0)
        for p in preds:
            pred_classes[p["class"]] = pred_classes.get(p["class"], 0) + 1
            if p["class"] == "boundary":
                t = ref[p["ref_hits"][0]]
                errs = sorted(abs(a - b) for a, b in
                              zip([s for s, _ in sorted(p["exons"])],
                                  [s for s, _ in sorted(t["exons"])]))
                boundary_bp_err.extend(errs[:1])  # 5' boundary shift magnitude
            if p["class"] == "exon_count":
                exon_delta.append(len(p["exons"]) - len(ref[p["ref_hits"][0]]["exons"]))
        for t in ref.values():
            ref_classes[t["class"]] = ref_classes.get(t["class"], 0) + 1
            if t["class"] in ("wrong_struct", "missed"):
                d = missed_ref if t["class"] == "missed" else wrong_ref
                k = "0" if t["n_intron"] == 0 else ("1" if t["n_intron"] == 1 else "2+")
                d["n_intron"][k] += 1
                lb = _len_bin(t["cds_len"])
                d["cds_len"][lb] = d["cds_len"].get(lb, 0) + 1
                gb = _gap_bin(t["nearest_gap"]) if t["nearest_gap"] is not None else "overlap/0"
                d["gap"][gb] = d["gap"].get(gb, 0) + 1
                if t.get("gc") is not None:
                    gcb = "<0.2" if t["gc"] < 0.2 else "0.2-0.3" if t["gc"] < 0.3 else ">=0.3"
                    d["gc_bin"][gcb] = d["gc_bin"].get(gcb, 0) + 1

    return {"totals": totals, "pred_classes": pred_classes,
            "ref_classes": ref_classes, "wrong_struct_strata": wrong_ref,
            "missed_strata": missed_ref,
            "boundary_shift_bp": sorted(boundary_bp_err),
            "exon_count_deltas": exon_delta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("res_pkl")
    ap.add_argument("ref_gff")
    ap.add_argument("refiner_pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    pkl = pickle.load(open(a.res_pkl, "rb"))
    ref_rows_all = parse_cds(a.ref_gff)
    gcode = pkl.get("gcode", 1)
    ref = load_refiner(a.refiner_pt, device=a.device)
    seqs = pkl["contig_seqs"]

    per_contig = {}
    for r in pkl["results"]:
        if r.contig not in ref_rows_all:
            continue
        genes = decode_with_refiner(ref, seqs[r.contig], r, stops=STOPS,
                                    euk=True, gcode=gcode, device=a.device)
        per_contig[r.contig] = classify_contig(r.contig, ref_rows_all, genes,
                                               seqs[r.contig])

    s = summarize(per_contig)
    n_pred, n_ref = s["totals"]["pred"], s["totals"]["ref"]
    print(f"pred genes: {n_pred}   ref tx: {n_ref} "
          f"(intronless ref: {s['totals']['ref_intron0']})")
    print("\npred classes:")
    for k, v in sorted(s["pred_classes"].items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:6d}  ({100*v/n_pred:.1f}%)")
    print("ref classes:")
    for k, v in sorted(s["ref_classes"].items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:6d}  ({100*v/n_ref:.1f}%)")
    for name, strata in (("missed", s["missed_strata"]),
                         ("wrong_struct", s["wrong_struct_strata"])):
        print(f"\n{name} strata:")
        for k, d in strata.items():
            if d:
                print(f"  {k}: {d}")
    if s["boundary_shift_bp"]:
        b = s["boundary_shift_bp"]
        print(f"\nboundary 5' shift bp: median={b[len(b)//2]} "
              f"p90={b[int(len(b)*0.9)]} n={len(b)}")
    if s["exon_count_deltas"]:
        d = s["exon_count_deltas"]
        print(f"exon-count delta (pred-ref): mean={sum(d)/len(d):+.2f} "
              f"n={len(d)}  dist={ {x: d.count(x) for x in sorted(set(d))} }")
    if a.out:
        json.dump(s, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
