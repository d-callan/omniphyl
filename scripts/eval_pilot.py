#!/usr/bin/env python3
"""Evaluate an OmniPhyl pilot checkpoint against a labelled fragment set.

Metrics emitted:
  - Per-rank top-1 accuracy and macro-F1 (taxonomy ranks + coding head)
  - Split metrics: the training genome split is reproduced from the
    checkpoint's saved args; metrics are reported separately for train vs
    held-out val genomes (memorization vs generalization).
  - Deepest-confident-rank: deepest rank whose top-1 softmax probability
    clears a confidence threshold (tau sweep). Real abstention — the mask
    class 0 is never a learned prediction.
  - Per-group accuracy breakdown (domain/phylum/... accuracy split by RefSeq
    group: bacteria/viral/fungi/...)
  - Leave-phylum-out generalisation: separate accuracy for fragments from phyla
    that were NOT in the training vocabulary (unknown-mass test).
  - RefSeq coverage gap report: which phyla in the labelled set have only a
    single assembly represented (culture-bias flag for the full corpus plan).
  - Confusion matrices (saved as TSV) for domain and phylum.
  - Calibration: ECE + reliability bins over held-out genome fragments
    (calibration.tsv), and risk-vs-coverage AURC (risk_coverage.tsv).
  - Per-class recall tables (per_class_<rank>.tsv) on held-out genomes.
  - Genome consensus: pooled log-prob accuracy per accession at M=1/5/20/all
    fragments (genome_consensus in results.json).
  - Per-fragment predictions dump (predictions.tsv).

Subset flags: --val-only restricts inference to held-out val genomes
(recommended for CPU evals); --max-genomes N caps how many held-out
accessions are evaluated/probability-stored (0 = all).

Reads:
  best.pt         OmniPhyl checkpoint (includes vocab, targets, k, args)
  labels.tsv      fragment labels (including group, phylum, coding columns)
  fragments.fna   fragment sequences

Deps: torch, numpy.

Usage:
  python eval_pilot.py \\
      --model   data/pilot/model/best.pt \\
      --labels  data/pilot/labels_screened.tsv \\
      --fragments data/pilot/fragments_screened.fna \\
      --outdir  data/pilot/eval
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add scripts dir to path so we can import from siblings
sys.path.insert(0, str(Path(__file__).parent))
from train_omniphyl import (OmniPhyl, FragmentDataset, build_cell_lut,
                             build_split, RANKS, CONT_TARGETS,
                             encode_cont_labels, MS_KS, rc_index_map)

RANK_ORDER = RANKS  # domain -> ... -> species
TOPK_CONS = 512     # sparse log-prob entries stored per frag per rank


def load_labels(path: Path, keep: set | None = None,
                cols: set | None = None) -> dict[str, dict]:
    """Load labels.tsv. ``keep`` restricts to a set of frag_ids; ``cols``
    restricts the stored fields per row (frag_id is the key, not stored).
    Both are RAM savers for --val-only CPU evals on large corpora."""
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            fid = r["frag_id"]
            if keep is not None and fid not in keep:
                continue
            if cols is not None:
                r = {c: r.get(c, "") for c in cols}
            rows[fid] = r
    return rows


def encode_labels_from_vocab(meta: dict[str, dict], vocab: dict,
                              all_targets: list[str]) -> dict[str, dict]:
    out = {}
    for fid, m in meta.items():
        rec = {}
        for t in all_targets:
            name = m.get(t, "")
            v = vocab[t].get(name, -1) if name else -1
            rec[t] = (v + 1 if v >= 0 else 0)
        out[fid] = rec
    return out


def macro_f1(pred: list[int], true: list[int],
             n_classes: int) -> float:
    """Macro-averaged F1 ignoring class 0 (unknown)."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p, t in zip(pred, true):
        if t == 0:
            continue
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    f1s = []
    for c in range(1, n_classes):
        prec = tp[c] / (tp[c] + fp[c] + 1e-9)
        rec = tp[c] / (tp[c] + fn[c] + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s)) if f1s else 0.0


def confusion_matrix_tsv(pred: list[int], true: list[int],
                          id2name: dict[int, str]) -> str:
    classes = sorted(set(true + pred) - {0})
    names = [id2name.get(c, str(c)) for c in classes]
    mat = defaultdict(lambda: defaultdict(int))
    for p, t in zip(pred, true):
        if t != 0:
            mat[t][p] += 1
    lines = ["\t" + "\t".join(names)]
    for c, name in zip(classes, names):
        row = [name] + [str(mat[c][c2]) for c2 in classes]
        lines.append("\t".join(row))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--fragments", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--val-only", action="store_true",
                    help="restrict inference to held-out val genomes")
    ap.add_argument("--max-genomes", type=int, default=0,
                    help="cap held-out accessions evaluated (0 = all)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    # Load checkpoint
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    vocab: dict[str, dict] = ckpt["vocab"]
    all_targets: list[str] = ckpt.get("targets", RANKS)
    k: int = ckpt["k"]
    train_args = ckpt.get("args", {})

    # Classification targets have vocab entries; continuous heads
    # (coding_frac/boundary/offset) are scalar outputs, n_classes=1.
    cls_targets = [t for t in all_targets if t in vocab]
    n_classes = {t: len(vocab[t]) + 1 for t in cls_targets}
    n_classes.update({t: 1 for t in all_targets if t not in n_classes})
    base_ch = len(MS_KS) if train_args.get("backbone") == "msfcgr" else 1
    model = OmniPhyl(n_classes,
                     width=train_args.get("width", 64),
                     cin=base_ch + 2 if train_args.get("pos_channels")
                     else base_ch,
                     backbone=train_args.get("backbone", "resnet"),
                     side=1 << k,
                     targets=all_targets,
                     dropout=train_args.get("dropout", 0.0),
                     stem_stride=train_args.get("stem_stride", 2),
                     head=train_args.get("head", "flat"),
                     tree=ckpt.get("tree"))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Build reverse vocab (id -> name) for reporting
    id2name: dict[str, dict[int, str]] = {}
    for t, v in vocab.items():
        id2name[t] = {idx + 1: name for name, idx in v.items()}
        id2name[t][0] = "<unknown>"

    # Load labels; reproduce the training genome split up front so
    # --val-only can filter the eval set before dataset construction.
    # Two-pass read under --val-only: pass 1 keeps only the columns
    # build_split needs (461k full rows = ~2 GB; slim rows ~50 MB); pass 2
    # loads full rows for eval fragments only.
    SPLIT_COLS = {"accession", "rank_unresolved", "genus"}
    if args.val_only:
        meta_split = load_labels(args.labels, cols=SPLIT_COLS)
        val_accs, _ = build_split(
            meta_split,
            float(train_args.get("val_frac", 0.1)),
            float(train_args.get("holdout_genus_frac", 0.0)),
            int(train_args.get("seed", 0)))
        eval_accs = set(sorted(val_accs)[:args.max_genomes or None])
        keep = {fid for fid, m in meta_split.items()
                if m.get("accession") in eval_accs}
        meta = load_labels(args.labels, keep=keep)
        fid_acc = {fid: m.get("accession", "")
                   for fid, m in meta_split.items()}
        del meta_split
    else:
        meta = load_labels(args.labels)
        val_accs, _ = build_split(
            meta,
            float(train_args.get("val_frac", 0.1)),
            float(train_args.get("holdout_genus_frac", 0.0)),
            int(train_args.get("seed", 0)))
        eval_accs = set(sorted(val_accs)[:args.max_genomes or None])
        fid_acc = {fid: m.get("accession", "") for fid, m in meta.items()}
    encoded = encode_labels_from_vocab(meta, vocab, cls_targets)
    cont = encode_cont_labels(meta)

    lut = build_cell_lut(k)
    ms_maps = ({kk: build_cell_lut(kk) for kk in MS_KS},
               {kk: rc_index_map(kk) for kk in MS_KS}) \
        if train_args.get("backbone") == "msfcgr" else (None, None)
    ds = FragmentDataset(args.fragments, encoded, k, lut,
                         pos=train_args.get("pos_channels", False),
                         targets=cls_targets, cont=cont,
                         seq=train_args.get("backbone") == "seq",
                         seq_len=int(train_args.get("seq_len", 500)),
                         ms_luts=ms_maps[0], ms_canons=ms_maps[1])
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    # Run inference
    all_preds: dict[str, list[int]] = {t: [] for t in cls_targets}
    all_true: dict[str, list[int]] = {t: [] for t in cls_targets}
    all_confs: dict[str, list[float]] = {t: [] for t in cls_targets}
    cont_out: dict[str, list[float]] = {t: [] for t in CONT_TARGETS}
    cont_true: list[tuple] = []
    frag_ids = ds.ids
    # Frag positions (held-out genomes) whose full log-probs get stored for
    # calibration/consensus metrics; float16 rows keep RAM bounded.
    consensus_pos = {i for i, fid in enumerate(frag_ids)
                     if fid_acc.get(fid) in eval_accs}
    cons_pos_list: list[int] = []
    cons_logp: dict[str, list] = {t: [] for t in cls_targets}
    # per-sibling conditional entropy (tree heads): the open-set signal —
    # unseen child clades of a known parent should produce near-uniform
    # conditionals over that parent's children
    tree_po = (ckpt.get("tree") or {}).get("parent_of")
    sib_ent: dict[str, list] = {t: [] for t in cls_targets}

    with torch.no_grad():
        pos = 0
        for x, y, yc in loader:
            x = x.to(device)
            z, logits = model(x)
            if tree_po:
                cond = model.cond_logps(z, tau=0.0)
                for ri in range(1, len(RANK_ORDER)):
                    t, par_t = RANK_ORDER[ri], RANK_ORDER[ri - 1]
                    if t not in cond or par_t not in logits:
                        continue
                    po_t = torch.as_tensor(tree_po[t][1:],
                                           device=z.device)
                    pred_par = logits[par_t].argmax(1)
                    for b in range(x.size(0)):
                        sel = po_t == pred_par[b]
                        n = int(sel.sum())
                        if n < 2:
                            continue
                        l = cond[t][b, sel]
                        sib_ent[t].append(
                            float(-(l.exp() * l).sum() / math.log(n)))
            logp = {t: F.log_softmax(logits[t], 1).cpu()
                    for t in cls_targets}
            for i, t in enumerate(cls_targets):
                all_preds[t].extend(logits[t].argmax(1).cpu().tolist())
                all_confs[t].extend(
                    logits[t].softmax(1).max(1).values.cpu().tolist())
                all_true[t].extend(y[:, i].tolist())
            for t in CONT_TARGETS:
                if t in logits:
                    cont_out[t].extend(
                        torch.sigmoid(logits[t][:, 0]).cpu().tolist())
            cont_true.extend(yc.tolist())
            for j in range(x.size(0)):
                if pos + j in consensus_pos:
                    cons_pos_list.append(pos + j)
                    for t in cls_targets:
                        kk = min(TOPK_CONS, logp[t].size(1))
                        v, i = torch.topk(logp[t][j], kk)
                        cons_logp[t].append(
                            (i.numpy().astype(np.int32),
                             v.numpy().astype(np.float16)))
            pos += x.size(0)

    # -----------------------------------------------------------------------
    # 1. Per-target accuracy + macro-F1
    # -----------------------------------------------------------------------
    all_idxs = list(range(len(frag_ids)))
    frac_true = np.array([c[0] for c in cont_true])
    off_true = np.array([c[1] for c in cont_true])

    def cls_metrics(idxs: list[int]) -> dict:
        out = {}
        for t in cls_targets:
            known = [i for i in idxs if all_true[t][i] != 0]
            if not known:
                out[t] = {"accuracy": None, "macro_f1": None, "n": 0}
                continue
            acc = sum(all_preds[t][i] == all_true[t][i]
                      for i in known) / len(known)
            f1 = macro_f1([all_preds[t][i] for i in known],
                          [all_true[t][i] for i in known], n_classes[t])
            out[t] = {"accuracy": round(acc, 4),
                      "macro_f1": round(f1, 4), "n": len(known)}
        return out

    def cont_metrics(idxs: list[int]) -> dict:
        """coding_frac / boundary / offset metrics restricted to idxs.

        Boundary recall vs flag rate is the cascade-economics gate
        (open question 1 in docs/annotation-cascade.md).
        """
        sel = np.asarray(idxs, dtype=np.int64)
        ft, ot = frac_true[sel], off_true[sel]
        fmk, omk = ft >= 0, ot >= 0
        out = {}
        if fmk.any() and cont_out.get("coding_frac"):
            p = np.asarray(cont_out["coding_frac"])[sel][fmk]
            t_ = ft[fmk]
            r = (float(np.corrcoef(p, t_)[0, 1])
                 if t_.std() > 0 else float("nan"))
            out["coding_frac"] = {
                "mae": round(float(np.abs(p - t_).mean()), 4),
                "pearson_r": round(r, 4), "n": int(fmk.sum())}
        if fmk.any() and cont_out.get("boundary"):
            p = np.asarray(cont_out["boundary"])[sel][fmk]
            tb = ((ft[fmk] > 0) & (ft[fmk] < 1))
            sweep = {}
            for rate in (0.05, 0.10, 0.20, 0.30, 0.50):
                thr = np.quantile(p, 1 - rate)
                flagged = p >= thr
                sweep[f"flag_{int(rate*100)}pct"] = round(
                    float((flagged & tb).sum() / max(tb.sum(), 1)), 4)
            out["boundary"] = {
                "accuracy": round(float(((p > 0.5) == tb).mean()), 4),
                "recall_at_flag_rate": sweep,
                "n": int(fmk.sum()), "n_pos": int(tb.sum())}
        if omk.any() and cont_out.get("offset"):
            p = np.asarray(cont_out["offset"])[sel][omk]
            out["offset"] = {
                "mae": round(float(np.abs(p - ot[omk]).mean()), 4),
                "n": int(omk.sum())}
        return out

    results: dict = {}
    results["per_target"] = cls_metrics(all_idxs)
    for t, m in results["per_target"].items():
        print(f"  {t:12s}  acc={m['accuracy']}  macro-F1={m['macro_f1']}  "
              f"n={m['n']}")
    results["annotation_heads"] = cont_metrics(all_idxs)
    for t, m in results["annotation_heads"].items():
        print(f"  {t:12s}  {m}")

    # -----------------------------------------------------------------------
    # 1c. Honest split: reproduce the training genome split (deterministic
    #     given the saved args) and score train vs held-out val fragments
    #     separately. per_target mixes memorized train frags with held-out
    #     frags; the val column is the real generalization readout.
    # -----------------------------------------------------------------------
    val_set = {i for i, fid in enumerate(frag_ids)
               if fid_acc.get(fid) in val_accs}
    val_idxs = sorted(val_set)
    train_idxs = [i for i in all_idxs if i not in val_set]
    results["split_metrics"] = {
        "n_train": len(train_idxs), "n_val": len(val_idxs),
        "val_genomes": sorted(val_accs),
        "train": {"per_target": cls_metrics(train_idxs),
                  "annotation_heads": cont_metrics(train_idxs)},
        "val": {"per_target": cls_metrics(val_idxs),
                "annotation_heads": cont_metrics(val_idxs)}}
    print(f"\nSplit: {len(train_idxs)} train / {len(val_idxs)} val-genome "
          f"frags ({len(val_accs)} held-out genomes)")
    for t, m in results["split_metrics"]["val"]["per_target"].items():
        print(f"  val {t:12s}  acc={m['accuracy']}  n={m['n']}")

    # -----------------------------------------------------------------------
    # 1b. Predicted-class distribution: how concentrated argmax calls are
    # across the vocab (the v4 "41.7% -> Uroviricota" collapse metric),
    # plus mean per-sibling conditional entropy for tree heads.
    # -----------------------------------------------------------------------
    inv_name = {t: {i + 1: n for n, i in vocab[t].items()}
                for t in cls_targets}
    results["predicted_class_entropy"] = {}
    print("\nPredicted-class distribution (collapse check):")
    for t in cls_targets:
        preds = np.asarray(all_preds[t])
        preds = preds[preds != 0]
        if len(preds) == 0:
            continue
        cnt = np.bincount(preds)
        nz = cnt[cnt > 0]
        h = float(-(nz / nz.sum() * np.log(nz / nz.sum())).sum())
        top = int(cnt.argmax())
        ent = {"n_predicted": int(nz.size),
               "top_class": inv_name[t].get(top, str(top)),
               "top_share": round(float(cnt[top] / len(preds)), 4),
               "entropy_nats": round(h, 4),
               "entropy_norm": round(h / math.log(nz.size), 4)
               if nz.size > 1 else 0.0}
        results["predicted_class_entropy"][t] = ent
        print(f"  {t:10s}  H={ent['entropy_nats']} "
              f"({ent['entropy_norm']} norm)  top={ent['top_class']} "
              f"{ent['top_share']:.1%} of {nz.size} predicted")
    if tree_po:
        results["sibling_entropy"] = {
            t: {"mean_norm_entropy": round(float(np.mean(v)), 4),
                "n": len(v)}
            for t, v in sib_ent.items() if v}
        print("\nPer-sibling conditional entropy (normalized):")
        for t, m in results["sibling_entropy"].items():
            print(f"  {t:10s}  H_norm={m['mean_norm_entropy']} "
                  f"n={m['n']}")

    # -----------------------------------------------------------------------
    # 2. Deepest-confident-rank: deepest RANK where pred != 0 (unknown)
    #    Only applies to taxonomy ranks, not the coding head.
    # -----------------------------------------------------------------------
    # Descend the cascade while a rank's top-1 softmax prob clears tau.
    # (The old pred != unknown check never abstained: class 0 is a mask
    # label the model never emits, so every fragment reached 'species'.)
    results["deepest_confident_rank"] = {}
    print("\nDeepest confident rank distribution:")
    for tau in (0.5, 0.8, 0.9):
        dc: dict[str, int] = defaultdict(int)
        for i in range(len(frag_ids)):
            depth = "none"
            for t in reversed(RANK_ORDER):  # species -> ... -> domain
                if (t in all_targets and all_preds[t][i] != 0
                        and all_confs[t][i] >= tau):
                    depth = t
                    break
            dc[depth] += 1
        results["deepest_confident_rank"][f"tau_{tau}"] = dict(dc)
        print(f"  tau={tau}: " + "  ".join(
            f"{r}:{dc.get(r, 0)}" for r in RANK_ORDER + ["none"]))

    # -----------------------------------------------------------------------
    # 3. Per-group accuracy (by RefSeq group column)
    # -----------------------------------------------------------------------
    fid_to_group = {fid: meta[fid].get("group", "") for fid in frag_ids}
    groups = sorted(set(fid_to_group.values()))
    results["per_group"] = {}
    print("\nPer-group domain accuracy:")
    for grp in groups:
        idxs = [i for i, fid in enumerate(frag_ids) if fid_to_group[fid] == grp]
        t = "domain"
        if t not in all_targets:
            continue
        known = [i for i in idxs if all_true[t][i] != 0]
        if not known:
            continue
        acc = sum(all_preds[t][i] == all_true[t][i] for i in known) / len(known)
        results["per_group"][grp] = {"domain_accuracy": round(acc, 4),
                                      "n": len(known)}
        print(f"  {grp:30s}  domain_acc={acc:.3f}  n={len(known)}")

    # -----------------------------------------------------------------------
    # 4. Leave-phylum-out generalisation
    #    Phyla with true label == 0 (not in training vocab) are the held-out set
    # -----------------------------------------------------------------------
    if "phylum" in all_targets:
        ti = all_targets.index("phylum")
        in_vocab_idxs = [i for i, v in enumerate(all_true["phylum"]) if v != 0]
        unk_idxs = [i for i, v in enumerate(all_true["phylum"]) if v == 0]

        def _rank_acc(idxs, t):
            if not idxs or t not in all_targets:
                return None
            known = [i for i in idxs if all_true[t][i] != 0]
            if not known:
                return None
            return round(sum(all_preds[t][i] == all_true[t][i]
                             for i in known) / len(known), 4)

        results["leave_phylum_out"] = {
            "n_in_vocab_phyla": len(in_vocab_idxs),
            "n_held_out_phyla": len(unk_idxs),
        }
        print(f"\nLeave-phylum-out: {len(unk_idxs)} fragments from held-out phyla")
        for t in RANK_ORDER:
            acc_in = _rank_acc(in_vocab_idxs, t)
            acc_out = _rank_acc(unk_idxs, t)
            results["leave_phylum_out"][t] = {
                "in_vocab_acc": acc_in, "held_out_acc": acc_out}
            print(f"  {t:12s}  in-vocab={acc_in}  held-out={acc_out}")

    # -----------------------------------------------------------------------
    # 5. Coverage gap report: phyla with only 1 assembly in the labelled set
    # -----------------------------------------------------------------------
    phylum_to_accs: dict[str, set[str]] = defaultdict(set)
    for fid, m in meta.items():
        ph = m.get("phylum", "")
        if ph:
            phylum_to_accs[ph].add(m.get("accession", ""))
    singleton_phyla = sorted(ph for ph, accs in phylum_to_accs.items()
                              if len(accs) == 1)
    absent_phyla: list[str] = []  # can be cross-referenced against a reference list
    results["coverage_gaps"] = {
        "singleton_phyla": singleton_phyla,
        "n_singleton": len(singleton_phyla),
        "n_phyla_total": len(phylum_to_accs),
    }
    print(f"\nCoverage gaps: {len(singleton_phyla)}/{len(phylum_to_accs)} "
          f"phyla represented by a single assembly")
    if singleton_phyla:
        print("  " + ", ".join(singleton_phyla[:10]) +
              ("..." if len(singleton_phyla) > 10 else ""))

    # -----------------------------------------------------------------------
    # 6. Confusion matrices (domain + phylum)
    # -----------------------------------------------------------------------
    for t in ["domain", "phylum"]:
        if t not in all_targets:
            continue
        p = all_preds[t]
        gt = all_true[t]
        cm = confusion_matrix_tsv(p, gt, id2name[t])
        cm_path = args.outdir / f"confusion_{t}.tsv"
        cm_path.write_text(cm)
        print(f"wrote {cm_path}")

    # -----------------------------------------------------------------------
    # 7. Calibration: 10-bin ECE + reliability table; risk-coverage AURC.
    #    Computed on held-out-genome fragments (eval_accs subset).
    # -----------------------------------------------------------------------
    calib_pos = cons_pos_list
    edges = np.linspace(0.0, 1.0, 11)
    calib_rows = ["rank\tbin_lo\tbin_hi\tn\tacc\tconf"]
    rc_rows = ["rank\tcoverage\trisk\tn"]
    ece_by_rank: dict[str, float] = {}
    aurc_by_rank: dict[str, float] = {}
    for t in cls_targets:
        conf = np.array([all_confs[t][p] for p in calib_pos])
        true = np.array([all_true[t][p] for p in calib_pos])
        pred = np.array([all_preds[t][p] for p in calib_pos])
        known = true != 0
        if not known.any():
            continue
        conf, pred, true = conf[known], pred[known], true[known]
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf <= hi if hi >= 1.0 else conf < hi)
            if not m.any():
                continue
            acc_b = float((pred[m] == true[m]).mean())
            conf_b = float(conf[m].mean())
            calib_rows.append(f"{t}\t{lo:.2f}\t{hi:.2f}\t{int(m.sum())}\t"
                              f"{acc_b:.4f}\t{conf_b:.4f}")
            ece += m.sum() / len(conf) * abs(acc_b - conf_b)
        ece_by_rank[t] = round(float(ece), 4)
        order = np.argsort(-conf)
        corr = (pred[order] == true[order]).astype(float)
        risks = []
        for c in np.arange(0.05, 1.0001, 0.05):
            k_ = max(1, int(round(c * len(order))))
            r = float(1 - corr[:k_].mean())
            risks.append(r)
            rc_rows.append(f"{t}\t{c:.2f}\t{r:.4f}\t{k_}")
        aurc_by_rank[t] = round(float(np.mean(risks)), 4)
    (args.outdir / "calibration.tsv").write_text("\n".join(calib_rows) + "\n")
    (args.outdir / "risk_coverage.tsv").write_text("\n".join(rc_rows) + "\n")
    results["calibration_ece"] = ece_by_rank
    results["risk_coverage_aurc"] = aurc_by_rank
    print("\nCalibration (held-out frags):")
    for t in cls_targets:
        if t in ece_by_rank:
            print(f"  {t:10s}  ECE={ece_by_rank[t]}  AURC={aurc_by_rank[t]}")

    # -----------------------------------------------------------------------
    # 8. Per-class recall tables on held-out frags
    # -----------------------------------------------------------------------
    for t in cls_targets:
        sup: dict[int, list] = defaultdict(lambda: [0, 0, 0])
        for p in calib_pos:
            tr, pr = all_true[t][p], all_preds[t][p]
            if tr == 0:
                continue
            sup[tr][0] += 1
            sup[tr][1] += int(pr == tr)
            sup[pr][2] += 1
        rows = ["class\tname\tsupport\trecall\tn_predicted"]
        for c, (n, tp, npred) in sorted(sup.items(), key=lambda kv: -kv[1][0]):
            rows.append(f"{c}\t{id2name[t].get(c, str(c))}\t{n}\t"
                        f"{tp / max(n, 1):.4f}\t{npred}")
        (args.outdir / f"per_class_{t}.tsv").write_text("\n".join(rows) + "\n")
    print(f"wrote {args.outdir}/per_class_<rank>.tsv")

    # -----------------------------------------------------------------------
    # 9. Per-fragment predictions dump (all evaluated frags)
    # -----------------------------------------------------------------------
    hdr = "frag_id\taccession\tgroup\t" + "\t".join(
        f"{t}_true\t{t}_pred\t{t}_conf" for t in cls_targets)
    pred_rows = [hdr]
    for i, fid in enumerate(frag_ids):
        cols = [fid, fid_acc.get(fid, ""), meta[fid].get("group", "")]
        for t in cls_targets:
            cols += [id2name[t].get(all_true[t][i], str(all_true[t][i])),
                     id2name[t].get(all_preds[t][i], str(all_preds[t][i])),
                     f"{all_confs[t][i]:.4f}"]
        pred_rows.append("\t".join(cols))
    (args.outdir / "predictions.tsv").write_text("\n".join(pred_rows) + "\n")
    print(f"wrote {args.outdir/'predictions.tsv'}")

    # -----------------------------------------------------------------------
    # 10. Genome consensus: pooled log-probs per accession, M-frag sweep.
    #     Tests whether contig-level aggregation rescues per-frag weakness.
    # -----------------------------------------------------------------------
    acc_to_local: dict[str, list[int]] = defaultdict(list)
    for li, gp in enumerate(cons_pos_list):
        acc_to_local[fid_acc.get(frag_ids[gp], "")].append(li)
    acc_group = {acc: meta[frag_ids[cons_pos_list[l[0]]]].get("group", "")
                 for acc, l in acc_to_local.items()}
    rng = np.random.default_rng(0)

    # fragments-per-genome distribution per group: contextualizes M=20 vs
    # "all" — a small-genome group where the corpus only sampled ~10 frags
    # makes M=20 effectively whole-genome, which inflates apparent lift.
    fpg_rows = ["group\tn_genomes\tfrags_mean\tfrags_median\tfrags_max"]
    fpg_stats: dict[str, dict] = {}
    for g in sorted(set(acc_group.values())):
        ns = [len(l) for acc, l in acc_to_local.items() if acc_group[acc] == g]
        if not ns:
            continue
        fpg_stats[g] = {"n_genomes": len(ns), "mean": round(np.mean(ns), 1),
                        "median": int(np.median(ns)), "max": max(ns)}
        fpg_rows.append(f"{g}\t{len(ns)}\t{np.mean(ns):.1f}\t"
                        f"{int(np.median(ns))}\t{max(ns)}")
    (args.outdir / "frags_per_genome.tsv").write_text(
        "\n".join(fpg_rows) + "\n")
    results["frags_per_genome"] = fpg_stats
    print("\nFragments per held-out genome:")
    for r in fpg_rows:
        print(f"  {r}")

    def pool_vote(t, take) -> int:
        pooled: dict[int, float] = defaultdict(float)
        for li in np.atleast_1d(take):
            idx, val = cons_logp[t][li]
            # topk values are sorted desc: val[-1] is the floor
            # every non-topK class would contribute. Subtract it
            # so stored entries are logp - floor; classes absent
            # from a frag's topK then get 0 marginal advantage.
            floor = float(val[-1])
            for c, v in zip(idx.tolist(), val.tolist()):
                pooled[c] += v - floor
        return max(pooled.items(), key=lambda kv: kv[1])[0] if pooled else 0

    consensus: dict[str, dict] = {}
    consensus_by_group: dict[str, dict] = defaultdict(dict)
    cbg_rows = ["group\trank\tM\tacc\tn_genomes"]
    print("\nGenome consensus accuracy (held-out accessions):")
    for t in cls_targets:
        if not cons_logp[t]:
            continue
        per_m = {}
        for m in (1, 5, 20, 0):
            correct = total = 0
            grp_stat: dict[str, list] = defaultdict(lambda: [0, 0])
            for acc, locs in acc_to_local.items():
                gi = cons_pos_list[locs[0]]
                if all_true[t][gi] == 0:
                    continue
                take = (locs if m == 0 else
                        rng.choice(locs, min(m, len(locs)), replace=False))
                pred_c = pool_vote(t, take)
                hit = int(pred_c == all_true[t][gi])
                correct += hit
                total += 1
                grp_stat[acc_group[acc]][0] += hit
                grp_stat[acc_group[acc]][1] += 1
            key = "all" if m == 0 else f"M{m}"
            per_m[key] = {"acc": round(correct / total, 4) if total else None,
                          "n_genomes": total}
            for g, (hit, n) in sorted(grp_stat.items()):
                consensus_by_group[g].setdefault(t, {})[key] = {
                    "acc": round(hit / n, 4) if n else None, "n_genomes": n}
                cbg_rows.append(f"{g}\t{t}\t{key}\t"
                                f"{round(hit / n, 4) if n else ''}\t{n}")
        consensus[t] = per_m
        print(f"  {t:10s}  " + "  ".join(
            f"{k}={v['acc']}" for k, v in per_m.items()))
    results["genome_consensus"] = consensus
    results["genome_consensus_by_group"] = dict(consensus_by_group)
    (args.outdir / "consensus_by_group.tsv").write_text(
        "\n".join(cbg_rows) + "\n")
    print(f"wrote {args.outdir/'consensus_by_group.tsv'}")
    print("\nConsensus by group (all-frags pooled):")
    for g, ranks in sorted(consensus_by_group.items()):
        print(f"  {g:20s}  " + "  ".join(
            f"{t}:{ranks[t]['all']['acc']}(n={ranks[t]['all']['n_genomes']})"
            for t in cls_targets if t in ranks))

    # -----------------------------------------------------------------------
    # Write full results JSON
    # -----------------------------------------------------------------------
    results_path = args.outdir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
