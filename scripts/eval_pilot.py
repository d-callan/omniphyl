#!/usr/bin/env python3
"""Evaluate an OmniPhyl pilot checkpoint against a labelled fragment set.

Metrics emitted:
  - Per-rank top-1 accuracy and macro-F1 (taxonomy ranks + coding head)
  - Deepest-confident-rank: for each fragment, find the deepest rank where the
    model's top-1 prediction is within-vocabulary (not the unknown class 0);
    report the distribution and mean depth.
  - Per-group accuracy breakdown (domain/phylum/... accuracy split by RefSeq
    group: bacteria/viral/fungi/...)
  - Leave-phylum-out generalisation: separate accuracy for fragments from phyla
    that were NOT in the training vocabulary (unknown-mass test).
  - RefSeq coverage gap report: which phyla in the labelled set have only a
    single assembly represented (culture-bias flag for the full corpus plan).
  - Confusion matrices (saved as TSV) for domain and phylum.

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
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add scripts dir to path so we can import from siblings
sys.path.insert(0, str(Path(__file__).parent))
from train_omniphyl import (OmniPhyl, FragmentDataset, build_cell_lut,
                             RANKS, CONT_TARGETS, encode_cont_labels)

RANK_ORDER = RANKS  # domain -> ... -> species


def load_labels(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["frag_id"]] = r
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
    model = OmniPhyl(n_classes,
                     width=train_args.get("width", 64),
                     cin=3 if train_args.get("pos_channels") else 1,
                     backbone=train_args.get("backbone", "resnet"),
                     side=1 << k,
                     targets=all_targets)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Build reverse vocab (id -> name) for reporting
    id2name: dict[str, dict[int, str]] = {}
    for t, v in vocab.items():
        id2name[t] = {idx + 1: name for name, idx in v.items()}
        id2name[t][0] = "<unknown>"

    # Load labels
    meta = load_labels(args.labels)
    encoded = encode_labels_from_vocab(meta, vocab, cls_targets)
    cont = encode_cont_labels(meta)

    lut = build_cell_lut(k)
    ds = FragmentDataset(args.fragments, encoded, k, lut,
                         pos=train_args.get("pos_channels", False),
                         targets=cls_targets, cont=cont)
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    # Run inference
    all_preds: dict[str, list[int]] = {t: [] for t in cls_targets}
    all_true: dict[str, list[int]] = {t: [] for t in cls_targets}
    cont_out: dict[str, list[float]] = {t: [] for t in CONT_TARGETS}
    cont_true: list[tuple] = []
    frag_ids = ds.ids

    with torch.no_grad():
        for x, y, yc in loader:
            x = x.to(device)
            _, logits = model(x)
            for i, t in enumerate(cls_targets):
                all_preds[t].extend(logits[t].argmax(1).cpu().tolist())
                all_true[t].extend(y[:, i].tolist())
            for t in CONT_TARGETS:
                if t in logits:
                    cont_out[t].extend(
                        torch.sigmoid(logits[t][:, 0]).cpu().tolist())
            cont_true.extend(yc.tolist())

    # -----------------------------------------------------------------------
    # 1. Per-target accuracy + macro-F1
    # -----------------------------------------------------------------------
    results: dict = {"per_target": {}}
    for t in cls_targets:
        p, gt = all_preds[t], all_true[t]
        known_mask = [i for i, v in enumerate(gt) if v != 0]
        if not known_mask:
            results["per_target"][t] = {"accuracy": None, "macro_f1": None,
                                         "n": 0}
            continue
        acc = sum(p[i] == gt[i] for i in known_mask) / len(known_mask)
        f1 = macro_f1([p[i] for i in known_mask],
                      [gt[i] for i in known_mask], n_classes[t])
        results["per_target"][t] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "n": len(known_mask),
        }
        print(f"  {t:12s}  acc={acc:.3f}  macro-F1={f1:.3f}  n={len(known_mask)}")

    # -----------------------------------------------------------------------
    # 1b. Continuous annotation heads (coding_frac / boundary / offset)
    #     Boundary recall vs flag rate is the cascade-economics gate
    #     (open question 1 in docs/annotation-cascade.md).
    # -----------------------------------------------------------------------
    frac_true = np.array([c[0] for c in cont_true])
    off_true = np.array([c[1] for c in cont_true])
    fm = frac_true >= 0
    om = off_true >= 0
    results["annotation_heads"] = {}
    if fm.any() and cont_out.get("coding_frac"):
        p_frac = np.array(cont_out["coding_frac"])[fm]
        t_frac = frac_true[fm]
        mae = float(np.abs(p_frac - t_frac).mean())
        r = (float(np.corrcoef(p_frac, t_frac)[0, 1])
             if t_frac.std() > 0 else float("nan"))
        results["annotation_heads"]["coding_frac"] = {
            "mae": round(mae, 4), "pearson_r": round(r, 4), "n": int(fm.sum())}
        print(f"  coding_frac   MAE={mae:.3f}  r={r:.3f}  n={int(fm.sum())}")
    if fm.any() and cont_out.get("boundary"):
        p_bnd = np.array(cont_out["boundary"])[fm]
        t_bnd = ((frac_true[fm] > 0) & (frac_true[fm] < 1))
        acc = float(((p_bnd > 0.5) == t_bnd).mean())
        sweep = {}
        for rate in (0.05, 0.10, 0.20, 0.30, 0.50):
            thr = np.quantile(p_bnd, 1 - rate)
            flagged = p_bnd >= thr
            rec = float((flagged & t_bnd).sum() / max(t_bnd.sum(), 1))
            sweep[f"flag_{int(rate*100)}pct"] = round(rec, 4)
        results["annotation_heads"]["boundary"] = {
            "accuracy": round(acc, 4), "recall_at_flag_rate": sweep,
            "n": int(fm.sum()), "n_pos": int(t_bnd.sum())}
        print(f"  boundary      acc={acc:.3f}  recall@flag={sweep}  "
              f"n_pos={int(t_bnd.sum())}")
    if om.any() and cont_out.get("offset"):
        p_off = np.array(cont_out["offset"])[om]
        mae = float(np.abs(p_off - off_true[om]).mean())
        results["annotation_heads"]["offset"] = {
            "mae": round(mae, 4), "n": int(om.sum())}
        print(f"  offset        MAE={mae:.3f}  n={int(om.sum())}")

    # -----------------------------------------------------------------------
    # 2. Deepest-confident-rank: deepest RANK where pred != 0 (unknown)
    #    Only applies to taxonomy ranks, not the coding head.
    # -----------------------------------------------------------------------
    depth_counts: dict[str, int] = defaultdict(int)
    for i in range(len(frag_ids)):
        depth = "none"
        for t in reversed(RANK_ORDER):  # species -> ... -> domain
            if t in all_targets:
                idx = all_targets.index(t)
                if all_preds[t][i] != 0:
                    depth = t
                    break
        depth_counts[depth] += 1

    results["deepest_confident_rank"] = dict(depth_counts)
    rank_order_with_none = RANK_ORDER + ["none"]
    print("\nDeepest confident rank distribution:")
    for r in rank_order_with_none:
        n = depth_counts.get(r, 0)
        print(f"  {r:12s}  {n:6d}  ({100*n/max(len(frag_ids),1):.1f}%)")

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
    # Write full results JSON
    # -----------------------------------------------------------------------
    results_path = args.outdir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
