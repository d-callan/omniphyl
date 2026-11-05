"""Kaggle CPU kernel: LexicMap threshold / decode sweep on chr1 (NC_004325.2).

Runs decode-only experiments on the cached `chr1_s100_bauto.res.pkl` +
`NC_004325.2.gff3` + `chr5_annot.bed*` (mounted via omniphyl-infer-test)
so multi-million-row BED filtering and threshold sweeps never run on the
local machine.

Each arm calls the repository's OWN `load_lexicmap_evidence` (two-tier
unitig-aware filtering, softmask-safe sequence screens, streaming parse)
so sweep results are faithful to production `run_annotate` behaviour.

Outputs: /kaggle/working/sweep_results.tsv
"""
import copy
import glob
import pickle
import subprocess
import sys

WORK = "/kaggle/working"
subprocess.run(["git", "clone", "--depth", "1",
                "https://github.com/d-callan/omniphyl.git",
                f"{WORK}/omniphyl"], check=True)
sys.path.insert(0, f"{WORK}/omniphyl/src")

import numpy as np
from omniphyl.annotate import _decode_genes, _gcode_stops, _exon_bounds
from omniphyl.io import load_lexicmap_evidence

pkl_hits = glob.glob("/kaggle/input/**/chr1_s100_bauto.res.pkl", recursive=True)
gff_hits = glob.glob("/kaggle/input/**/NC_004325.2.gff3", recursive=True)
bed_hits = (glob.glob("/kaggle/input/**/chr5_annot.bed*", recursive=True)
            + glob.glob("/kaggle/input/**/rs_full*.bed*", recursive=True))
if not (pkl_hits and gff_hits and bed_hits):
    raise SystemExit(f"Missing inputs in /kaggle/input: pkl={pkl_hits} "
                     f"gff={gff_hits} bed={bed_hits}")

PKL_PATH = pkl_hits[0]
GFF_PATH = gff_hits[0]
BED_PATH = bed_hits[0]


def load_ref(path):
    genes, introns = [], set()
    cds_by_parent = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 9:
            continue
        if f[2] == "gene":
            genes.append((int(f[3]) - 1, int(f[4]), f[6]))
        elif f[2] == "CDS":
            parent = ""
            for kv in f[8].split(";"):
                if kv.startswith("Parent="):
                    parent = kv.split("=", 1)[1]
            cds_by_parent.setdefault(parent, []).append(
                (int(f[3]) - 1, int(f[4]), f[6]))
    for exons in cds_by_parent.values():
        exons.sort()
        for a, b in zip(exons, exons[1:]):
            if b[0] > a[1]:
                introns.add((a[1], b[0], a[2]))
    return genes, introns, cds_by_parent


def eval_preds(genes_pred, ref_genes, ref_introns, ref_cds, clen):
    matched_r, matched_p = set(), set()
    for ri, r in enumerate(ref_genes):
        best, best_ov = -1, 0
        for pi, p in enumerate(genes_pred):
            if p["strand"] != r[2]:
                continue
            ov = max(0, min(r[1], p["end"]) - max(r[0], p["start"]))
            if ov > best_ov:
                best, best_ov = pi, ov
        if best >= 0:
            p = genes_pred[best]
            if (best_ov >= 0.5 * (r[1] - r[0])
                    and best_ov >= 0.25 * (p["end"] - p["start"])):
                matched_r.add(ri)
                matched_p.add(best)
    pred_introns = {
        (d, acc, g["strand"])
        for g in genes_pred for d, acc in g.get("introns", [])
    }
    exact_introns = len(pred_introns & ref_introns)
    cov = np.zeros(clen, bool)
    for p in genes_pred:
        for es, ee in _exon_bounds(p):
            cov[max(0, es):min(clen, ee + 1)] = True
    cds_bp = hit_bp = 0
    for exons in ref_cds.values():
        for a, b, _s in exons:
            cds_bp += b - a
            hit_bp += int(cov[a:b].sum())
    return {
        "preds": len(genes_pred),
        "matched": len(matched_r),
        "recall": len(matched_r) / max(1, len(ref_genes)),
        "precision": len(matched_p) / max(1, len(genes_pred)),
        "multi_exon": sum(1 for g in genes_pred if g.get("introns")),
        "pred_introns": len(pred_introns),
        "exact_introns": exact_introns,
        "cds_cov": hit_bp / max(1, cds_bp),
    }


def fuse_evidence(res, ev, nbins):
    """Mirror run_annotate's aggregation-time fusion exactly."""
    d = ev.density[:nbins]
    if len(d) < nbins:
        d = np.pad(d, (0, nbins - len(d)))
    l1p = np.log1p(d)
    p90 = np.percentile(l1p[l1p > 0], 90) if (l1p > 0).any() else 1.0
    dtilde = np.minimum(1.0, l1p / max(p90, 1.0))
    med = np.median(d[d > 0]) if (d > 0).any() else 0.0
    if med > 0:
        p_crna = res.p_region[:, 0] + res.p_region[:, 2]
        damp = (d > 10 * med) & (p_crna < 0.35)
        dtilde = np.where(damp, np.minimum(dtilde, 0.15), dtilde)
    res.dtilde = dtilde
    if ev.endpoints is not None:
        e_lex = ev.endpoints[:nbins]
        if len(e_lex) < nbins:
            e_lex = np.pad(e_lex, (0, nbins - len(e_lex)))
        res.b_tracks[:, 2] += e_lex
        res.b_tracks[:, 3] += e_lex


def main():
    d = pickle.load(open(PKL_PATH, "rb"))
    base_res = d["results"][0]
    contig = base_res.contig
    seq = d["contig_seqs"][contig]
    stops = _gcode_stops(d["gcode"])
    ref_genes, ref_introns, ref_cds = load_ref(GFF_PATH)
    ref_iv_xy = {(a, b) for a, b, _s in ref_introns}
    print(f"Loaded {contig} ({base_res.length} bp), "
          f"ref_genes={len(ref_genes)}, ref_introns={len(ref_introns)}",
          flush=True)

    rows = []
    # Baseline no-evidence arm
    res0 = copy.deepcopy(base_res)
    g0 = _decode_genes(seq, res0, stops=stops, euk=True, w_ev=0.0,
                       min_cds_len=90, coding_threshold=0.45,
                       boundary_threshold=0.25, min_intron=35,
                       max_intron=5000, intron_hints=[])
    m0 = eval_preds(g0, ref_genes, ref_introns, ref_cds, base_res.length)
    print("NO_EV:", m0, flush=True)
    rows.append(("no_ev", 0, 0, 0, 0.0, m0))

    for min_single in (300, 400, 600, 900):
        for min_partner in (150, 250):
            for min_sup in (1, 2, 4):
                ev_map = load_lexicmap_evidence(
                    bed_path=BED_PATH, source="rnaseq",
                    min_score=min_partner,
                    min_single_score=min_single,
                    min_intron_support=min_sup,
                    bin_width=base_res.bin_width,
                    contig_seqs={contig: seq})
                ev = ev_map.get(contig)
                hints = ev.intron_candidates if ev else []
                cov_bins = int((ev.density > 0).sum()) if ev else 0
                hint_tp = sum(
                    1 for hd, ha in hints
                    if any(abs(hd - rd) <= 12 and abs(ha - ra) <= 12
                           for rd, ra in ref_iv_xy))
                for w_ev in (0.25, 0.35):
                    res = copy.deepcopy(base_res)
                    if ev is not None:
                        fuse_evidence(res, ev, res.n_bins)
                    genes = _decode_genes(
                        seq, res, stops=stops, euk=True, w_ev=w_ev,
                        min_cds_len=90, coding_threshold=0.45,
                        boundary_threshold=0.25, min_intron=35,
                        max_intron=5000, intron_hints=hints)
                    m = eval_preds(genes, ref_genes, ref_introns,
                                   ref_cds, base_res.length)
                    m["n_hints"] = len(hints)
                    m["hint_tp"] = hint_tp
                    m["cov_bins"] = cov_bins
                    tag = (f"s{min_single}_p{min_partner}"
                           f"_sup{min_sup}_w{w_ev}")
                    print(tag, m, flush=True)
                    rows.append((tag, min_single, min_partner,
                                 min_sup, w_ev, m))

    with open(f"{WORK}/sweep_results.tsv", "w") as out:
        out.write("tag\tmin_single\tmin_partner\tmin_sup\tw_ev\tn_hints\t"
                  "hint_tp\tcov_bins\tpreds\tmatched\trecall\tprecision\t"
                  "multi_exon\tpred_introns\texact_introns\tcds_cov\n")
        for tag, ms, mp, msup, wev, m in rows:
            out.write(
                f"{tag}\t{ms}\t{mp}\t{msup}\t{wev}\t"
                f"{m.get('n_hints', 0)}\t{m.get('hint_tp', 0)}\t"
                f"{m.get('cov_bins', 0)}\t"
                f"{m['preds']}\t{m['matched']}\t{m['recall']:.4f}\t"
                f"{m['precision']:.4f}\t{m['multi_exon']}\t"
                f"{m['pred_introns']}\t{m['exact_introns']}\t"
                f"{m['cds_cov']:.4f}\n")


if __name__ == "__main__":
    main()
