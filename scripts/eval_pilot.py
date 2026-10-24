#!/usr/bin/env python3
"""Evaluate an OmniPhyl pilot checkpoint against a labelled fragment set.

Metrics emitted:
  - Per-rank top-1 accuracy and macro-F1 (taxonomy ranks + coding head)
  - Split metrics: the training genome split is reproduced from the
    checkpoint's saved args; metrics are reported separately for train vs
    held-out val genomes vs within-genome frag-val vs genus-holdout
    (memorization vs generalization vs leave-clade-out).
  - Deepest-confident-rank: deepest rank whose top-1 softmax probability
    clears a confidence threshold (tau sweep). Real abstention — the mask
    class 0 is never a learned prediction.
  - Selective precision / FDR among emitted calls (selective_precision.tsv):
    per-rank coverage, precision, and FDR at tau in {0.5,...,0.95} plus
    coverage@P90/P95 (max coverage whose running precision clears the
    target). Emitted calls on out-of-vocabulary fragments (true == 0)
    count as FALSE POSITIVES, not misses — the asymmetric cost model
    where a confident wrong name is worse than abstaining.
  - NCBI cross-rank reconciliation (default on, --no-reconcile-tree):
    ancestor-path mean-logp rescoring of flat-head calls, a contradiction
    veto (suppress a rank call when any shallower ancestor prob < tau_anc),
    and reconciled variants of accuracy/selective metrics.
  - Unseen-clade safety (genus-holdout fragments): overcall rate = fraction
    emitted at genus/species where truth is out-of-vocab (hallucination);
    safe_ancestor_stop = correct family call with genus/species abstained.
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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add scripts dir to path so we can import from siblings
sys.path.insert(0, str(Path(__file__).parent))
from train_omniphyl import (OmniPhyl, FragmentDataset, build_cell_lut,
                             build_split, RANKS, CONT_TARGETS,
                             encode_cont_labels, encode_annot_labels,
                             MS_KS, HASH_K, HASH_PAD,
                             rc_index_map)

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


def auroc(y: np.ndarray, s: np.ndarray):
    """Tie-aware Mann-Whitney AUROC; None when the truth is single-class."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    if y.min() == y.max() or len(y) == 0:
        return None
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    npos = y.sum()
    nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2)
                 / (npos * nneg))


def auprc(y: np.ndarray, s: np.ndarray):
    """Average-precision AUPRC; None when no positives exist."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    if y.sum() == 0:
        return None
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[o])
    return float((tp / np.arange(1, len(y) + 1) * y[o]).sum() / y.sum())


STOPS = {"TAA", "TAG", "TGA"}


def scan_transition(seq: str, frame: int, kind: str,
                    pred_pos: float):
    """Canonical triplet edge coordinate nearest pred_pos in `frame`.

    Stage-2 of the frame-scan pathway: given the predicted frame and a
    predicted transition coordinate (window-relative bp), return the
    genomic edge position of the closest ATG (start) or stop codon in
    that frame, or None when no canonical triplet exists. Edge coords
    match the fetcher's convention: CDS-interval start/end boundary,
    transcript-oriented on the minus strand.
    """
    rev = frame >= 3
    s2 = seq[::-1].translate(str.maketrans("ACGT", "TGCA")) if rev else seq
    L = len(s2)
    off = (frame - 3) if rev else frame
    cands = {"ATG"} if kind == "start" else STOPS
    best, bd = None, 1e18
    for i in range(off, L - 2, 3):
        if s2[i:i + 3] not in cands:
            continue
        edge = (L - i if kind == "start" else L - i - 3) if rev \
            else (i if kind == "start" else i + 3)
        if abs(edge - pred_pos) < bd:
            bd, best = abs(edge - pred_pos), edge
    return best


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


def build_parent_of(rows_iter, vocab: dict, ranks: list[str]) -> dict:
    """Majority-vote parent map when the checkpoint lacks ckpt["tree"].

    parent_of[r][c] = shifted class id of the most common parent-rank
    name on label rows carrying class c; 0 = parent missing/out-of-vocab.
    Identical construction to train_omniphyl.build_tree's parent vote.
    """
    votes = {r: defaultdict(Counter) for r in ranks[1:]}
    for m in rows_iter:
        for ri in range(1, len(ranks)):
            c_name = m.get(ranks[ri], "")
            if c_name in vocab[ranks[ri]]:
                votes[ranks[ri]][c_name][m.get(ranks[ri - 1], "")] += 1
    parent_of = {}
    for ri, r in enumerate(ranks):
        po = np.zeros(len(vocab[r]) + 1, dtype=np.int64)
        if ri:
            for name, c in vocab[r].items():
                pv = votes[r].get(name)
                if pv:
                    pname = pv.most_common(1)[0][0]
                    po[c + 1] = vocab[ranks[ri - 1]].get(pname, -1) + 1
        parent_of[r] = po
    return parent_of


def ancestor_chains(parent_of: dict, ranks: list[str]) -> dict:
    """chains[r] -> int64 [n_r, ri+1]: col j = ancestor class id at
    ranks[j] for each class id at rank r (col ri = self; 0 = ancestor
    exists but is out-of-vocab/mask class; -1 = no parent known)."""
    chains = {}
    for ri, r in enumerate(ranks):
        n = len(parent_of[r])
        cols = [np.arange(n, dtype=np.int64)]
        cur = np.arange(n, dtype=np.int64)
        for k in range(ri, 0, -1):
            po_k = parent_of[ranks[k]]
            nxt = np.where(cur > 0, po_k[np.maximum(cur, 0)], -1)
            cols.insert(0, nxt)
            cur = nxt
        chains[r] = np.stack(cols, axis=1)
    return chains


def path_scores(logp: dict, chain: np.ndarray,
                ranks: list[str]) -> torch.Tensor:
    """Mean log-probability of each rank-r class's full ancestor path.

    chain: [C_r, L] numpy ancestor ids (col L-1 = self, -1 = missing).
    logp[rk]: [B, C_rk] log-softmax tensors. Ancestor id 0 (the
    out-of-vocab mask class) is treated as missing — its logit is an
    untrained free parameter and carries no signal.
    Returns [B, C_r] path-consistent scores (geometric-mean log-space).
    """
    ch = torch.from_numpy(chain)
    valid = ch > 0                     # mask of usable ancestor evidence
    denom = valid.sum(1).float()
    acc = torch.zeros(next(iter(logp.values())).size(0), ch.size(0))
    for j in range(ch.size(1)):
        ids = ch[:, j].clamp(min=0)
        g = logp[ranks[j]][:, ids]     # [B, C_r]
        acc += g * valid[:, j].float()
    # classes with zero usable ancestor evidence (the mask class 0, or
    # fully-orphaned lineages) must not get score 0.0 — that would beat
    # every real (negative) mean-logp path score.
    score = acc / denom.clamp(min=1).unsqueeze(0)
    score[:, denom == 0] = float("-inf")
    return score


FIXED_TAUS = (0.5, 0.7, 0.8, 0.9, 0.95)


def selective_metrics(idxs: list[int], preds: dict, confs: dict,
                      trues: dict, cls_targets: list[str],
                      min_anc: dict | None = None,
                      anc_taus: tuple = (0.1, 0.2, 0.4)) -> dict:
    """Precision/FDR among emitted calls + coverage at target precision.

    False positives include calls on unseen-clade fragments (true == 0):
    an emitted confident call where the truth is out-of-vocabulary is a
    hallucination, not a miss, so it counts in the FP denominator.
    coverage = fraction of ALL cohort fragments emitted at tau.
    cov_at_pT = max coverage over the confidence-sorted list whose
    running precision stays >= T. If min_anc is provided, also sweeps
    ancestor-consistency vetoes: a call at rank t is additionally
    vetoed when any shallower-rank ancestor of the argmax has prob <
    anc_tau (cross-rank contradiction -> false-positive suppressor).
    """
    sel = np.asarray(sorted(idxs), dtype=np.int64)
    out: dict = {}
    for t in cls_targets:
        c = np.asarray(confs[t])[sel]
        p = np.asarray(preds[t])[sel]
        y = np.asarray(trues[t])[sel]
        n_all = max(len(sel), 1)
        rows: dict = {}
        for tau in FIXED_TAUS:
            called = c >= tau
            nc = int(called.sum())
            tp = int(((p == y) & (y > 0) & called).sum())
            rows[f"tau_{tau}"] = {
                "coverage": round(nc / n_all, 4),
                "precision": round(tp / nc, 4) if nc else None,
                "fdr": round(1 - tp / nc, 4) if nc else None,
                "n_called": nc, "n_fp": nc - tp}
        order = np.argsort(-c)
        corr = ((p[order] == y[order]) & (y[order] > 0)).astype(float)
        cump = np.cumsum(corr) / np.arange(1, len(corr) + 1)
        covf = np.arange(1, len(corr) + 1) / n_all
        for tgt in (90, 95):
            ok = cump >= tgt / 100
            rows[f"cov_at_p{tgt}"] = (round(float(covf[ok].max()), 4)
                                      if ok.any() else 0.0)
        if min_anc is not None and t in min_anc:
            ma = np.asarray(min_anc[t])[sel]
            veto_rows = {}
            for at in anc_taus:
                nc = 0
                tp = 0
                for tau in FIXED_TAUS:
                    called = (c >= tau) & (ma >= at)
                    n2 = int(called.sum())
                    t2 = int(((p == y) & (y > 0) & called).sum())
                    nc += n2
                    tp += t2
                    veto_rows[f"t{tau}_a{at}"] = {
                        "coverage": round(n2 / n_all, 4),
                        "precision": round(t2 / n2, 4) if n2 else None}
            rows["ancestor_veto"] = veto_rows
        out[t] = rows
    return out


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
    ap.add_argument("--infer-tau", type=float, default=None,
                    help="tree head: prior offset at inference. Default "
                         "= the checkpoint's training --prior-tau (MAP "
                         "decisions); 0 gives prior-balanced conditionals")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--jitter-rate", type=float, default=0.0,
                    help="SNP substitutions per position applied to eval "
                         "fragments; sweep this to test instance- vs "
                         "spectrum-feature reliance (1 SNP kills ~k "
                         "overlapping k-mers but preserves spectrum)")
    ap.add_argument("--zero-aux", type=str, default="",
                    help="comma list of aux branches to zero out at "
                         "inference (e.g. 'hash' or 'hash,aa') — branch "
                         "attribution")
    ap.add_argument("--shuffle-aux", type=str, default="",
                    help="comma list of aux branches whose per-sample "
                         "inputs are shuffled across the batch — "
                         "destroys label alignment while preserving "
                         "input distribution (permutation control)")
    ap.add_argument("--val-only", action="store_true",
                    help="restrict inference to held-out val genomes")
    ap.add_argument("--max-genomes", type=int, default=0,
                    help="cap held-out accessions evaluated (0 = all)")
    ap.add_argument("--eval-crop-len", type=int, default=0,
                    help="truncate each fragment to this many bases "
                         "before encoding (0 = full length); simulates "
                         "150/250bp single-end reads")
    ap.add_argument("--eval-paired-len", type=int, default=0,
                    help="simulate unmerged paired-end reads: prefix and "
                         "suffix mates of this length joined by an 'N' "
                         "junction (e.g. 150 -> PE150)")
    ap.add_argument("--no-reconcile-tree", action="store_true",
                    help="skip NCBI cross-rank path-score reconciliation "
                         "and ancestor-veto metrics (saves the per-batch "
                         "ancestor-chain gathers on large corpora)")
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
                     tree=ckpt.get("tree"),
                     aux={a for a in train_args.get("aux", "").split(",")
                          if a},
                     aa_mode=train_args.get("aa_mode", "plain"),
                     aa_drop=float(train_args.get("aa_drop", 0.0) or 0.0),
                     region_head=bool(train_args.get("region_head")),
                     struct_head=bool(train_args.get("struct_head")),
                     domain_k=len(ckpt.get("domain_vocab", {})))
    model.load_state_dict(ckpt["model"])
    # The tree head is trained so softmax(logits + tau*log nu) = P(y);
    # evaluating at that same tau restores the posterior the model was
    # fit to. tau=0 instead reports P/nu (rare-class-inflated ratios).
    model.infer_tau = (args.infer_tau if args.infer_tau is not None
                       else float(train_args.get("prior_tau", 0.0))
                       if train_args.get("head") == "tree" else 0.0)
    if train_args.get("head") == "tree":
        print(f"tree head: infer_tau={model.infer_tau}")
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
    SPLIT_COLS = {"accession", "rank_unresolved", "genus", "seq_source",
                  "seqid", "start"}
    split_args = (float(train_args.get("val_frac", 0.1)),
                  float(train_args.get("holdout_genus_frac", 0.0)),
                  int(train_args.get("seed", 0)),
                  float(train_args.get("frag_val", 0.0)))
    if args.val_only:
        meta_split = load_labels(args.labels, cols=SPLIT_COLS)
        val_accs, holdout_genera, val_frags = build_split(
            meta_split, *split_args)
        eval_accs = set(sorted(val_accs)[:args.max_genomes or None])
        # genus-holdout fragments sit on train-side accessions but were
        # never trained on; keep them so the unseen-clade safety metrics
        # (overcall / safe ancestor stop) can be scored under --val-only
        keep = {fid for fid, m in meta_split.items()
                if m.get("accession") in eval_accs or fid in val_frags
                or m.get("genus", "") in holdout_genera}
        meta = load_labels(args.labels, keep=keep)
        fid_acc = {fid: m.get("accession", "")
                   for fid, m in meta_split.items()}
        del meta_split
    else:
        meta = load_labels(args.labels)
        val_accs, holdout_genera, val_frags = build_split(
            meta, *split_args)
        eval_accs = set(sorted(val_accs)[:args.max_genomes or None])
        fid_acc = {fid: m.get("accession", "") for fid, m in meta.items()}
    encoded = encode_labels_from_vocab(meta, vocab, cls_targets)
    cont = encode_cont_labels(meta)
    # P3 annot labels are needed only when the checkpoint carries the
    # matching heads; domain_vocab pins the multi-hot layout.
    domain_vocab = ckpt.get("domain_vocab", {})
    annot_on = (bool(train_args.get("region_head"))
                or bool(train_args.get("struct_head"))
                or bool(domain_vocab))
    annot = (encode_annot_labels(meta, domain_vocab)
             if annot_on else None)

    lut = build_cell_lut(k)
    ms_maps = ({kk: build_cell_lut(kk) for kk in MS_KS},
               {kk: rc_index_map(kk) for kk in MS_KS}) \
        if train_args.get("backbone") == "msfcgr" else (None, None)
    aux = {a for a in train_args.get("aux", "").split(",") if a}
    ds = FragmentDataset(args.fragments, encoded, k, lut,
                         pos=train_args.get("pos_channels", False),
                         targets=cls_targets, cont=cont,
                         seq=train_args.get("backbone") == "seq",
                         seq_len=int(train_args.get("seq_len", 500)),
                         ms_luts=ms_maps[0], ms_canons=ms_maps[1],
                         aux=aux,
                         hash_canon=rc_index_map(HASH_K)
                         if "hash" in aux else None,
                         jitter=args.jitter_rate,
                         aa_extra=train_args.get("aa_mode") == "gated",
                         eval_crop=args.eval_crop_len,
                         eval_paired=args.eval_paired_len,
                         aa_frame=bool(train_args.get("struct_head")),
                         annot=annot)
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False,
        num_workers=args.num_workers)

    # Run inference
    all_preds: dict[str, list[int]] = {t: [] for t in cls_targets}
    all_true: dict[str, list[int]] = {t: [] for t in cls_targets}
    all_confs: dict[str, list[float]] = {t: [] for t in cls_targets}
    cont_out: dict[str, list[float]] = {t: [] for t in CONT_TARGETS}
    cont_true: list[tuple] = []
    # P3 annotation-head outputs, collected only when the model has them
    ann_out: dict[str, list] = {
        k: [] for k in ("region_fracs", "active_frame",
                        "struct_flags", "domains")}
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

    # --- NCBI cross-rank reconciliation setup ---------------------------
    # parent_of[r][c] maps class id c at rank r to its parent class id at
    # rank r-1. New checkpoints save it for every head; older flat
    # checkpoints rebuild it by majority vote over labels.tsv rank
    # columns (same logic as build_tree's parent vote).
    reconcile = not args.no_reconcile_tree
    chains: dict | None = None
    if reconcile:
        if tree_po is None:
            print("no parent map in checkpoint; rebuilding from "
                  "labels.tsv (majority-vote parents, vocab-restricted)")
            rows_iter = (meta.values() if not args.val_only else
                         load_labels(args.labels, cols=set(RANKS)).values())
            tree_po = {r: build_parent_of(rows_iter, vocab,
                                          cls_targets)[r].tolist()
                       for r in cls_targets}
            del rows_iter
        chains = {r: np.asarray(tree_po[r]) for r in cls_targets}
        chains = ancestor_chains(chains, cls_targets)
        n_orphan = {r: int((chains[r][:, :-1] == -1).all(1).sum())
                    for r in cls_targets if chains[r].shape[1] > 1}
        print(f"reconciling ranks over NCBI parent map; "
              f"rootless classes per rank: {n_orphan}")
    recon_pred: dict[str, list] = {t: [] for t in cls_targets}
    recon_conf: dict[str, list] = {t: [] for t in cls_targets}
    min_anc: dict[str, list] = {t: [] for t in cls_targets}

    zero_aux = {a for a in args.zero_aux.split(",") if a}
    shuffle_aux = {a for a in args.shuffle_aux.split(",") if a}

    with torch.no_grad():
        pos = 0
        for x, y, yc in loader:
            x = ({kk: v.to(device) for kk, v in x.items()}
                 if isinstance(x, dict) else x.to(device))
            bs = y.size(0)
            if isinstance(x, dict):
                for a in zero_aux & set(x):
                    # all-padding bag -> padding_idx -> zero embedding;
                    # vector branches -> literal zeros
                    x[a] = (x[a].fill_(HASH_PAD) if a == "hash"
                            else torch.zeros_like(x[a]))
                for a in shuffle_aux & set(x):
                    x[a] = x[a][torch.randperm(bs, device=x[a].device)]
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
                    for b in range(bs):
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
            if chains is not None:
                for t in cls_targets:
                    ch = chains[t]
                    # path-consistent call: mean logp along the full
                    # NCBI ancestor path; col 0 (unknown) masked out
                    s = path_scores(logp, ch, cls_targets)
                    s[:, 0] = -1e9
                    rp, rs = s.max(1)
                    recon_pred[t].extend(rp.tolist())
                    recon_conf[t].extend(torch.exp(rs).tolist())
                    # veto evidence: weakest ancestor prob of the raw
                    # argmax call (any shallower-rank rejection)
                    pmax = logp[t].argmax(1)
                    crow = torch.from_numpy(ch)[pmax]
                    minp = torch.ones(bs)
                    for j in range(crow.size(1) - 1):
                        ids = crow[:, j]
                        m = ids > 0
                        ap = logp[cls_targets[j]].exp().gather(
                            1, ids.clamp(min=0)[:, None]).squeeze(1)
                        minp = torch.where(m, torch.minimum(minp, ap),
                                           minp)
                    min_anc[t].extend(minp.tolist())
            for t in CONT_TARGETS:
                if t in logits:
                    cont_out[t].extend(
                        torch.sigmoid(logits[t][:, 0]).cpu().tolist())
            if "region_fracs" in logits:
                ann_out["region_fracs"].extend(
                    logits["region_fracs"].softmax(1).cpu().tolist())
            if "active_frame" in logits:
                ann_out["active_frame"].extend(
                    logits["active_frame"].argmax(1).cpu().tolist())
            if "struct_flags" in logits:
                ann_out["struct_flags"].extend(
                    logits["struct_flags"].sigmoid().cpu().tolist())
            if "domains" in logits:
                ann_out["domains"].extend(
                    logits["domains"].sigmoid().cpu().tolist())
            cont_true.extend(yc.tolist())
            for j in range(bs):
                if pos + j in consensus_pos:
                    cons_pos_list.append(pos + j)
                    for t in cls_targets:
                        kk = min(TOPK_CONS, logp[t].size(1))
                        v, i = torch.topk(logp[t][j], kk)
                        cons_logp[t].append(
                            (i.numpy().astype(np.int32),
                             v.numpy().astype(np.float16)))
            pos += bs

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

    # P3 annot truth block: yc cols 2.. are [region(4) | frame(1) |
    # flags(4) | domains(K) | dom_mask(1)] — only present when the
    # checkpoint carries annotation heads.
    ann_true = (np.asarray(cont_true)[:, 2:]
                if cont_true and len(cont_true[0]) > 2 else None)
    dom_id2name = {v: k for k, v in domain_vocab.items()}

    def annot_metrics(idxs: list[int], scan: bool = False) -> dict:
        """E3-E5 annotation-head metrics restricted to idxs.

        The frame-scan gate (E4 stage-2) resolves each predicted
        start/stop flag to the nearest canonical triplet in the
        predicted frame and scores |pred - true edge| <= 3bp — the
        operationally meaningful end-to-end rate, not just head acc.
        """
        if ann_true is None:
            return {}
        sel = np.asarray(idxs, dtype=np.int64)
        ya = ann_true[sel]
        out: dict = {}
        if ann_out["region_fracs"]:
            rm = ya[:, 0] >= 0
            if rm.any():
                p = np.asarray(ann_out["region_fracs"])[sel][rm]
                t_ = ya[rm, :4]
                coding_bin = (t_[:, 0] >= 0.5).astype(float)
                nonc = t_[:, 0] < 0.5
                pred_nc = p.argmax(1) != 0
                tp = float((pred_nc & nonc).sum())
                fp = float((pred_nc & ~nonc).sum())
                fn = float((~pred_nc & nonc).sum())
                f1 = 2 * tp / max(2 * tp + fp + fn, 1)
                m = {"mae": round(float(np.abs(p - t_).mean()), 4),
                     "coding_auroc": auroc(coding_bin, p[:, 0]),
                     "noncoding_f1": round(f1, 4),
                     "n": int(rm.sum())}
                # eukaryote subset: introns make the composition task
                # strictly harder than the bacterial majority
                euk = np.array([meta[frag_ids[i]].get("group") in
                                ("fungi", "protozoa")
                                for i in sel[rm]])
                if euk.any():
                    m["euk_coding_auroc"] = auroc(
                        coding_bin[euk], p[euk, 0])
                    m["euk_n"] = int(euk.sum())
                out["region_fracs"] = m
        if ann_out["active_frame"]:
            fm = ya[:, 4] >= 0
            if fm.any():
                p = np.asarray(ann_out["active_frame"])[sel][fm]
                t_ = ya[fm, 4]
                acc = float((p == t_).mean())
                single = t_ < 6
                m = {"accuracy": round(acc, 4), "n": int(fm.sum())}
                if single.any():
                    m["single_frame_acc"] = round(
                        float((p[single] == t_[single]).mean()), 4)
                    m["n_single_frame"] = int(single.sum())
                out["active_frame"] = m
        if ann_out["struct_flags"]:
            sm = ya[:, 5] >= 0
            if sm.any():
                p = np.asarray(ann_out["struct_flags"])[sel][sm]
                t_ = ya[sm, 5:9]
                bits = {}
                names = ("start", "stop", "splice_donor",
                         "splice_acceptor")
                for b in range(4):
                    a = auroc(t_[:, b], p[:, b])
                    bits[names[b]] = {"auroc": a,
                                      "n_pos": int(t_[:, b].sum())}
                vals = [v["auroc"] for v in bits.values()
                        if v["auroc"] is not None]
                out["struct_flags"] = {
                    "per_bit": bits,
                    "macro_auroc": round(float(np.mean(vals)), 4)
                    if vals else None,
                    "n": int(sm.sum())}
        if ann_out["domains"]:
            dm = ya[:, -1] >= 0
            if dm.any():
                p = np.asarray(ann_out["domains"])[sel][dm]
                K = p.shape[1]
                t_ = ya[dm, 9:9 + K]
                per = [(auroc(t_[:, j], p[:, j]),
                        auprc(t_[:, j], p[:, j]))
                       for j in range(K)]
                aus = [a for a, _ in per if a is not None]
                aps = [b for _, b in per if b is not None]
                top5 = np.argsort(-p, 1)[:, :min(5, K)]
                pos_rows = t_.sum(1) > 0
                r5 = (float(np.mean(
                    [t_[i, top5[i]].any() for i in range(len(t_))
                     if pos_rows[i]])) if pos_rows.any() else None)
                out["domains"] = {
                    "macro_auroc": round(float(np.mean(aus)), 4)
                    if aus else None,
                    "macro_auprc": round(float(np.mean(aps)), 4)
                    if aps else None,
                    "recall_at5": round(r5, 4) if r5 is not None else None,
                    "K": K, "n": int(dm.sum()),
                    "n_pos_rows": int(pos_rows.sum())}
        # ---- E4 stage-2 frame-scan gate ------------------------------
        # has_start/has_stop windows with exactly one CDS boundary:
        # head predicts bit + frame + coordinate; scanner resolves the
        # coordinate to a canonical triplet in that frame; hit within 3bp.
        if scan and ann_out["struct_flags"] \
                and ann_out["active_frame"]:
            rows = {"start": [0, 0], "stop": [0, 0]}
            n_single_bnd = 0
            for pos_i, i in enumerate(sel):
                fid = frag_ids[i]
                if meta[fid].get("n_boundaries", "") != "1" \
                        or off_true[i] < 0:
                    continue
                n_single_bnd += 1
                flg = ann_out["struct_flags"][i]
                fr = ann_out["active_frame"][i]
                if fr >= 6:
                    continue
                seq = ds._read_seq(fid)
                flen = len(seq)
                for bit, kind in ((0, "start"), (1, "stop")):
                    if ya[pos_i, 5 + bit] < 0.5 or flg[bit] <= 0.5:
                        continue
                    pred_off = (cont_out["offset"][i] * flen
                                if cont_out["offset"] else flen / 2)
                    edge = scan_transition(seq, int(fr), kind, pred_off)
                    if edge is None:
                        continue
                    rows[kind][1] += 1
                    if abs(edge - off_true[i] * flen) <= 3:
                        rows[kind][0] += 1
            if n_single_bnd:
                out["frame_scan"] = {
                    "n_single_boundary_windows": n_single_bnd,
                    **{k: {"hit_rate_3bp": round(v[0] / v[1], 4)
                           if v[1] else None,
                           "n": v[1]}
                       for k, v in rows.items()}}
        return out

    results: dict = {}
    results["per_target"] = cls_metrics(all_idxs)
    for t, m in results["per_target"].items():
        print(f"  {t:12s}  acc={m['accuracy']}  macro-F1={m['macro_f1']}  "
              f"n={m['n']}")
    results["annotation_heads"] = cont_metrics(all_idxs)
    for t, m in results["annotation_heads"].items():
        print(f"  {t:12s}  {m}")
    if ann_true is not None:
        results["annot_heads"] = annot_metrics(all_idxs)
        for t, m in results["annot_heads"].items():
            print(f"  {t:12s}  {m}")

    # -----------------------------------------------------------------------
    # 1c. Honest split: reproduce the training genome split (deterministic
    #     given the saved args) and score train vs held-out val fragments
    #     separately. per_target mixes memorized train frags with held-out
    #     frags; the val column is the real generalization readout.
    # -----------------------------------------------------------------------
    val_set = {i for i, fid in enumerate(frag_ids)
               if fid_acc.get(fid) in val_accs}
    fragval_set = {i for i, fid in enumerate(frag_ids)
                   if fid in val_frags and i not in val_set}
    # genus-holdout fragments were excluded from training by is_val() in
    # train_omniphyl (leave-clade-out), but they sit on train-side
    # accessions — without this bucket they silently pollute the "train"
    # metrics, which understates the memorization gap
    genushold_set = {i for i, fid in enumerate(frag_ids)
                     if meta[fid].get("genus", "") in holdout_genera}
    val_genus_set = genushold_set - val_set
    val_idxs = sorted(val_set)
    fragval_idxs = sorted(fragval_set)
    val_genus_idxs = sorted(val_genus_set)
    train_idxs = [i for i in all_idxs
                  if i not in val_set and i not in fragval_set
                  and i not in val_genus_set]
    results["split_metrics"] = {
        "n_train": len(train_idxs), "n_val": len(val_idxs),
        "n_val_frag": len(fragval_idxs), "n_val_genus": len(val_genus_idxs),
        "val_genomes": sorted(val_accs),
        "holdout_genera": sorted(holdout_genera),
        "train": {"per_target": cls_metrics(train_idxs),
                  "annotation_heads": cont_metrics(train_idxs)},
        "val": {"per_target": cls_metrics(val_idxs),
                "annotation_heads": cont_metrics(val_idxs),
                "annot_heads": annot_metrics(val_idxs, scan=True)},
        "val_frag": {"per_target": cls_metrics(fragval_idxs),
                     "annotation_heads": cont_metrics(fragval_idxs)},
        "val_genus": {"per_target": cls_metrics(val_genus_idxs),
                      "annotation_heads": cont_metrics(val_genus_idxs)}}
    print(f"\nSplit: {len(train_idxs)} train / {len(val_idxs)} val-genome "
          f"frags ({len(val_accs)} held-out genomes) / "
          f"{len(fragval_idxs)} within-genome held-out frags / "
          f"{len(val_genus_idxs)} genus-holdout frags "
          f"({len(holdout_genera)} genera)")
    for t, m in results["split_metrics"]["val"]["per_target"].items():
        print(f"  val {t:12s}  acc={m['accuracy']}  n={m['n']}")
    if fragval_idxs:
        for t, m in results["split_metrics"]["val_frag"]["per_target"].items():
            print(f"  fragval {t:8s}  acc={m['accuracy']}  n={m['n']}")

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
    # 2b. False-positive-first evaluation: precision/FDR among EMITTED
    #     calls (conf >= tau), coverage @ target precision, and the
    #     unseen-clade overcall/hallucination rate. Calls on fragments
    #     whose truth is out-of-vocabulary (true == 0) count as false
    #     positives — under the asymmetric cost model a confident wrong
    #     name is worse than abstaining.
    # -----------------------------------------------------------------------
    cohorts = {"val": val_idxs, "val_frag": fragval_idxs,
               "val_genus": val_genus_idxs}
    sp_rows = ["cohort\trank\ttau\tcoverage\tprecision\tfdr\tn_called\tn_fp"]
    results["selective"] = {}
    for cname, cidx in cohorts.items():
        if not cidx:
            continue
        results["selective"][cname] = selective_metrics(
            cidx, all_preds, all_confs, all_true, cls_targets)
        for t, rows in results["selective"][cname].items():
            for tau in FIXED_TAUS:
                r = rows[f"tau_{tau}"]
                sp_rows.append(
                    f"{cname}\t{t}\t{tau}\t{r['coverage']}\t"
                    f"{r['precision']}\t{r['fdr']}\t{r['n_called']}\t"
                    f"{r['n_fp']}")
    print("\nSelective precision among emitted calls "
          "(held-out genomes; unseen-clade calls count as FP):")
    for t in cls_targets:
        v = results["selective"].get("val", {}).get(t)
        if not v:
            continue
        print(f"  {t:10s}  P@tau0.8={v['tau_0.8']['precision']} "
              f"FDR={v['tau_0.8']['fdr']} "
              f"cov={v['tau_0.8']['coverage']}  "
              f"cov@P90={v['cov_at_p90']}  cov@P95={v['cov_at_p95']}")

    # -----------------------------------------------------------------------
    # 2c. Reconciled calls: NCBI ancestor-path scoring + contradiction veto
    # -----------------------------------------------------------------------
    if chains is not None:
        results["reconciled"] = {}
        results["reconciled"]["per_target_val"] = {
            t: {"accuracy": (round(sum(recon_pred[t][i] == all_true[t][i]
                                       for i in val_idxs
                                       if all_true[t][i] != 0)
                                   / max(sum(1 for i in val_idxs
                                             if all_true[t][i] != 0), 1), 4)),
                "n": sum(1 for i in val_idxs if all_true[t][i] != 0)}
            for t in cls_targets}
        print("\nReconciled (path-score) val accuracy vs raw flat argmax:")
        for t in cls_targets:
            ra = results["reconciled"]["per_target_val"][t]["accuracy"]
            raw = results["split_metrics"]["val"]["per_target"][t][
                "accuracy"]
            print(f"  {t:10s}  raw={raw}  reconciled={ra}")
        # selective metrics on reconciled calls + ancestor-veto on raw
        # argmax calls (veto = emit call only if weakest ancestor prob
        # clears anc_tau)
        results["reconciled"]["selective_val"] = selective_metrics(
            val_idxs, recon_pred, recon_conf, all_true, cls_targets)
        results["ancestor_veto_val"] = selective_metrics(
            val_idxs, all_preds, all_confs, all_true, cls_targets,
            min_anc=min_anc)
        for t in cls_targets:
            rows = results["reconciled"]["selective_val"][t]
            for tau in FIXED_TAUS:
                r = rows[f"tau_{tau}"]
                sp_rows.append(
                    f"val_reconciled\t{t}\t{tau}\t{r['coverage']}\t"
                    f"{r['precision']}\t{r['fdr']}\t{r['n_called']}\t"
                    f"{r['n_fp']}")
        print("\nReconciled selective precision (val):")
        for t in cls_targets:
            v = results["reconciled"]["selective_val"][t]
            print(f"  {t:10s}  P@tau0.8={v['tau_0.8']['precision']} "
                  f"cov={v['tau_0.8']['coverage']}  "
                  f"cov@P90={v['cov_at_p90']}")
        veto_rows = ["rank\ttau_conf\ttau_anc\tcoverage\tprecision"]
        for t in cls_targets:
            av = results["ancestor_veto_val"][t].get("ancestor_veto", {})
            for k, r in av.items():
                tconf, tanc = k.lstrip("t").split("_a")
                veto_rows.append(f"{t}\t{tconf}\t{tanc}\t"
                                 f"{r['coverage']}\t{r['precision']}")
        (args.outdir / "ancestor_veto.tsv").write_text(
            "\n".join(veto_rows) + "\n")
    (args.outdir / "selective_precision.tsv").write_text(
        "\n".join(sp_rows) + "\n")
    print(f"wrote {args.outdir / 'selective_precision.tsv'}")

    # -----------------------------------------------------------------------
    # 2d. Unseen-clade safety: on genus-holdout fragments (true genus and
    #     species are out-of-vocab by construction), a confident deep call
    #     is a hallucination; the safe behaviour is to stop at the correct
    #     family (or shallower).
    # -----------------------------------------------------------------------
    if val_genus_idxs and {"family", "genus", "species"} <= set(cls_targets):
        safety = {}
        print("\nUnseen-clade safety (genus-holdout fragments):")
        for tau in FIXED_TAUS:
            n = len(val_genus_idxs)
            over = sum(1 for i in val_genus_idxs
                       if (all_confs["genus"][i] >= tau
                           or all_confs["species"][i] >= tau))
            safe = sum(1 for i in val_genus_idxs
                       if all_confs["genus"][i] < tau
                       and all_confs["species"][i] < tau
                       and all_confs["family"][i] >= tau
                       and all_preds["family"][i] == all_true["family"][i])
            safety[f"tau_{tau}"] = {
                "n": n,
                "overcall_rate": round(over / n, 4),
                "safe_ancestor_stop": round(safe / n, 4)}
            print(f"  tau={tau}: overcall={over / n:.3f}  "
                  f"safe-stop={safe / n:.3f}  (n={n})")
        results["unseen_clade_safety"] = safety
        if chains is not None:
            rsafety = {}
            for tau in FIXED_TAUS:
                n = len(val_genus_idxs)
                over = sum(1 for i in val_genus_idxs
                           if (recon_conf["genus"][i] >= tau
                               or recon_conf["species"][i] >= tau))
                safe = sum(1 for i in val_genus_idxs
                           if recon_conf["genus"][i] < tau
                           and recon_conf["species"][i] < tau
                           and recon_conf["family"][i] >= tau
                           and recon_pred["family"][i]
                           == all_true["family"][i])
                rsafety[f"tau_{tau}"] = {
                    "n": n, "overcall_rate": round(over / n, 4),
                    "safe_ancestor_stop": round(safe / n, 4)}
            results["unseen_clade_safety"]["reconciled"] = rsafety
            print("  reconciled: " + "  ".join(
                f"tau={t}: over={v['overcall_rate']} safe={v['safe_ancestor_stop']}"
                for t, v in rsafety.items()))

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
