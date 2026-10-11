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
                             build_split, RANKS, CONT_TARGETS,
                             encode_cont_labels)

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
    all_confs: dict[str, list[float]] = {t: [] for t in cls_targets}
    cont_out: dict[str, list[float]] = {t: [] for t in CONT_TARGETS}
    cont_true: list[tuple] = []
    frag_ids = ds.ids

    with torch.no_grad():
        for x, y, yc in loader:
            x = x.to(device)
            _, logits = model(x)
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
        sel = np.asarray(idxs)
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
    val_accs, _ = build_split(
        meta,
        float(train_args.get("val_frac", 0.1)),
        float(train_args.get("holdout_genus_frac", 0.0)),
        int(train_args.get("seed", 0)))
    val_set = {i for i, fid in enumerate(frag_ids)
               if meta[fid].get("accession") in val_accs}
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
    # Write full results JSON
    # -----------------------------------------------------------------------
    results_path = args.outdir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {results_path}")


if __name__ == "__main__":
    main()
