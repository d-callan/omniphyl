#!/usr/bin/env python3
"""Fixed-panel refiner checkpoint evaluation.

For each ``--panel name,res.pkl,ref.gff[,contig]`` entry: loads the
refiner ckpt, recomputes per-base posteriors + decodes genes fresh
(never uses cached r.genes), then reports

  intron recall   true GFF introns whose donor/acceptor are matched
                  within 12 bp by a predicted intron
  signal stats    p_int_max / run_cov / p_don / p_acceptor quantiles
                  on matched vs unmatched true introns, plus the
                  unmatched-bucket breakdown (no_state_signal /
                  no_sites / no_motif / gated_elsewhere)
  gene classes    eval_errors reference/prediction class summary

Usage:
  python scripts/eval_refiner_panel.py --ckpt refiner.pt \
      --panel pvivax,/tmp/.../pvivax.res.pkl,/tmp/.../region_ref.gff \
      --panel hb3,/tmp/eval_out/...HB3....res.pkl,/tmp/eval_refs/....gff,PfHB3_09 \
      --out panel_report.json [--device cpu]

Contig defaults to the largest annotated contig present in both the
res.pkl results and the GFF. Prokaryotic panels decode with the full
STARTS codon set (matching run_annotate); --redecode in eval_errors
is ATG-only and understates bacteria/archaea.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_genes import parse_cds                       # noqa: E402
from eval_errors import classify_contig, summarize     # noqa: E402
from omniphyl.annotate import STOPS, STARTS            # noqa: E402
from omniphyl.refiner import (load_refiner, refiner_tracks,  # noqa: E402
                              decode_with_refiner,
                              ST_INT_P, ST_INT_M, ST_CDS_P, ST_CDS_M,
                              S_DON_P, S_ACC_P, S_DON_M, S_ACC_M)


def _true_introns(rows, L, seq):
    """[(plus, donor_gap, acceptor_gap, canon, ilen)] from CDS rows."""
    txs = {}
    for txid, strand, (s, e) in rows:
        txs.setdefault((txid, strand), []).append((s, e))
    out = []
    for (_txid, strand), ex in txs.items():
        ex = sorted(ex)
        plus = strand == "+"
        for i in range(len(ex) - 1):
            d, a = ex[i][1], ex[i + 1][0]
            ilen = a - d
            if ilen < 10 or a > L:
                continue
            if plus:
                don2, acc2 = seq[d:d + 2], seq[a - 2:a]
            else:
                don2 = {"AC": "GT", "GC": "GC"}.get(seq[a - 2:a], "??")
                acc2 = {"CT": "AG"}.get(seq[d:d + 2], "??")
            out.append((plus, d, a, don2 in ("GT", "GC")
                        and acc2 == "AG", ilen))
    return out


def _intron_diag(true_introns, pred_introns, p_state, p_site):
    stats = []
    for plus, d, a, canon, ilen in true_introns:
        st_i, st_c = (ST_INT_P, ST_CDS_P) if plus else (ST_INT_M,
                                                      ST_CDS_M)
        s_d, s_a = (S_DON_P, S_ACC_P) if plus else (S_DON_M, S_ACC_M)
        pi, pc = p_state[d:a, st_i], p_state[d:a, st_c]
        don_pos, acc_pos = (d, a - 1) if plus else (a - 1, d)
        mask = (pi >= 0.15) & (pi >= 0.35 * pc)
        stats.append({
            "len": ilen, "canon": canon,
            "p_int_max": float(pi.max()),
            "run_cov": float(mask.mean()),
            "p_don": float(p_site[max(0, don_pos - 3):don_pos + 4,
                                  s_d].max()),
            "p_acc": float(p_site[max(0, acc_pos - 3):acc_pos + 4,
                                  s_a].max()),
            "matched": any(abs(d - pd) <= 12 and abs(a - pa) <= 12
                           for pd, pa in pred_introns),
        })
    q = lambda v: [round(x, 3) for x in
                   np.percentile(v, [10, 50, 90])] if v else []
    rep = {"n_true": len(stats),
           "n_matched": sum(s["matched"] for s in stats)}
    for label, sub in (("matched", [s for s in stats if s["matched"]]),
                       ("unmatched", [s for s in stats
                                      if not s["matched"]])):
        if sub:
            rep[label] = {k: q([s[k] for s in sub]) for k in
                          ("p_int_max", "run_cov", "p_don", "p_acc")}
            rep[label]["n"] = len(sub)
            rep[label]["canon"] = sum(s["canon"] for s in sub)
    buckets = {"no_state_signal": 0, "no_sites": 0, "no_motif": 0,
               "gated_elsewhere": 0}
    for s in stats:
        if s["matched"]:
            continue
        if s["p_int_max"] < 0.25 or s["run_cov"] < 0.5:
            buckets["no_state_signal"] += 1
        elif s["p_don"] < 0.05 or s["p_acc"] < 0.05:
            buckets["no_sites"] += 1
        elif not s["canon"]:
            buckets["no_motif"] += 1
        else:
            buckets["gated_elsewhere"] += 1
    rep["unmatched_causes"] = buckets
    return rep


def _pick_contig(res_list, ref_rows, contig=None):
    avail = [r for r in res_list if r.contig in ref_rows]
    if contig:
        r = next((r for r in res_list if r.contig == contig), None)
        if r is None:
            sys.exit(f"contig {contig} not in res.pkl")
        return r
    if not avail:
        sys.exit("no res.pkl contig is annotated in the GFF")
    return max(avail, key=lambda r: r.length)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True,
                    help="refiner .pt checkpoint")
    ap.add_argument("--panel", action="append", required=True,
                    metavar="NAME,RES_PKL,REF_GFF[,CONTIG]",
                    help="repeatable panel entry")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = load_refiner(args.ckpt, device=args.device)
    print(f"ckpt {args.ckpt}  cond_mode="
          f"{getattr(ref, 'cond_mode', 'full')}", flush=True)
    report = {"ckpt": args.ckpt,
              "cond_mode": getattr(ref, "cond_mode", "full"),
              "panels": {}}
    for spec in args.panel:
        parts = spec.split(",")
        if len(parts) < 3:
            sys.exit(f"bad --panel {spec!r}: want name,res_pkl,gff"
                     "[,contig]")
        name, res_pkl, gff = parts[:3]
        contig = parts[3] if len(parts) > 3 else None
        pkl = pickle.load(open(res_pkl, "rb"))
        ref_rows_all = parse_cds(gff)
        res = _pick_contig(pkl["results"], ref_rows_all, contig)
        seq = pkl["contig_seqs"][res.contig]
        seq_u = seq.upper()
        L = len(seq_u)
        gcode = pkl.get("gcode", 1)
        dom = (res.taxon.names.get("domain")
               if getattr(res, "taxon", None) else None)
        euk = dom == "Eukaryota"
        starts = ("ATG",) if euk else STARTS

        p_state, p_site = refiner_tracks(ref, seq_u, res,
                                         gcode=gcode,
                                         device=args.device)
        genes = decode_with_refiner(ref, seq, res, stops=STOPS,
                                    euk=euk, gcode=gcode,
                                    p_state=p_state, p_site=p_site,
                                    starts=starts,
                                    device=args.device)
        pred_intr = [(d, a) for g in genes
                     for d, a in g.get("introns", [])]
        intr = _intron_diag(
            _true_introns(ref_rows_all[res.contig], L, seq_u),
            pred_intr, p_state, p_site)
        s = summarize({res.contig: classify_contig(
            res.contig, ref_rows_all, genes, seq)})
        pi = np.maximum(p_state[:, ST_INT_P], p_state[:, ST_INT_M])
        row = {"contig": res.contig, "domain": dom, "euk": euk,
               "n_pred": len(genes), "n_pred_introns": len(pred_intr),
               "frac_bases_p_int>0.5": round(float((pi > 0.5).mean()),
                                             4),
               "intron": intr,
               "ref_classes": s["ref_classes"],
               "pred_classes": s["pred_classes"]}
        report["panels"][name] = row
        match = (f"{intr['n_matched']}/{intr['n_true']} "
                 f"({100 * intr['n_matched'] / max(intr['n_true'], 1):.1f}%)")
        print(f"{name}/{res.contig}: introns {match}  "
              f"pred_introns={len(pred_intr)}  "
              f"p_int>0.5 bases={row['frac_bases_p_int>0.5']}  "
              f"ref_exact={s['ref_classes'].get('exact', 0)} "
              f"wrong={s['ref_classes'].get('wrong_struct', 0)} "
              f"miss={s['ref_classes'].get('missed', 0)}  "
              f"extra={s['pred_classes'].get('extra', 0)} "
              f"split={s['pred_classes'].get('split_part', 0)}",
              flush=True)
    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
