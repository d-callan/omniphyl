#!/usr/bin/env python3
"""Train the Stage-2 OmniPhylRefiner on (genome, GFF3) pairs.

Two halves per plan docs/neural-refiner-plan.md §5:

  inputs    Stage-1 ContigResult tracks (from a cached .res.pkl or by
            running `omniphyl annotate` stage-1 via OMNIPHYL_DUMP_RES)
            interpolated to per-base + nucleotide one-hot + clade cond
  targets   7-state per-base labels + 8 boundary-site channels (sigma=3
            gaussian smoothing) built from reference CDS rows

The Stage-1 model is FROZEN -- this script never touches its weights.
Chunks are 8192 bp with 6144 stride; RC augmentation doubles each
contig (labels and ContigResult tracks are strand-flipped, see
revcomp_result / rc_labels); evidence-dropout augmentation zeros the
dtilde/e_lex channels for 25% of chunks.

  python scripts/train_refiner.py \
      --data genome.fna genes.gff3 [--data ...] \
      --model model.pt --out refiner.pt [--cache-dir cache/] \
      [--bundle-model model.pt --bundle-out model+v2.pt]

  # smoke test on cached Stage-1 results (no model needed):
  python scripts/train_refiner.py \
      --data NC_004325.2.fasta NC_004325.2.gff3 \
      --res-pkl chr1_s100_bauto.res.pkl --out /tmp/refiner.pt --epochs 3
"""

import argparse
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniphyl.refiner import (OmniPhylRefiner, build_refiner_inputs,
                              build_contig_cond, save_refiner,
                              _stage1_tracks,
                              _chunk_starts, N_STATES, N_SITES, CHUNK,
                              CHUNK_STRIDE, ST_INTERGENIC, ST_CDS_P,
                              ST_INT_P, ST_CDS_M, ST_INT_M,
                              S_START_P, S_STOP_P, S_DON_P, S_ACC_P,
                              S_START_M, S_STOP_M, S_DON_M, S_ACC_M)
from omniphyl.annotate import ContigResult, run_annotate
from omniphyl.io import iter_sequences, revcomp
from eval_genes import parse_cds

SITE_SIGMA = 3.0
# state CE class weights: intergenic, CDS+, int+, CDS-, int-, rna, dual
STATE_W = torch.tensor([1.0, 1.5, 2.5, 1.5, 2.5, 1.0, 2.0])


# ----------------------------------------------------------- labels ----

def build_refiner_labels(rows, L):
    """(state [L] int64, site [8,L] float32) from CDS rows of one seqid.

    rows: [(tx_id, strand, (s,e)), ...] 0-based half-open, grouped by
    transcript id so introns come from consecutive exon gaps.
    Site conventions (see plan §3.3):
      +  START at cds_start; STOP at cds_end-3; DONOR at intron_start;
         ACCEPTOR at intron_end-1
      -  START at cds_end-1; STOP at cds_start+2; DONOR at
         intron_end-1; ACCEPTOR at intron_start
    Overlapping opposite-strand genes: last write wins on the state
    map (BOTH_STRAND_CDS is a plan v2 refinement)."""
    by_tx = defaultdict(list)
    for tx, strand, (s, e) in rows:
        by_tx[(tx, strand)].append((s, e))
    state = np.zeros(L, np.int64)
    site = np.zeros((N_SITES, L), np.float32)
    for (_tx, strand), ivs in by_tx.items():
        ivs = sorted(ivs)
        plus = strand == "+"
        cds_i, int_i = (ST_CDS_P, ST_INT_P) if plus \
            else (ST_CDS_M, ST_INT_M)
        s0, e0 = ivs[0][0], ivs[-1][1]
        for i, (s, e) in enumerate(ivs):
            state[s:e] = cds_i
            if i + 1 < len(ivs):
                state[e:ivs[i + 1][0]] = int_i
        if plus:
            site[S_START_P, s0] = 1.0
            site[S_STOP_P, max(e0 - 3, 0)] = 1.0
            for i in range(len(ivs) - 1):
                site[S_DON_P, ivs[i][1]] = 1.0
                site[S_ACC_P, ivs[i + 1][0] - 1] = 1.0
        else:
            site[S_START_M, e0 - 1] = 1.0
            site[S_STOP_M, min(s0 + 2, L - 1)] = 1.0
            for i in range(len(ivs) - 1):
                site[S_DON_M, ivs[i + 1][0] - 1] = 1.0
                site[S_ACC_M, ivs[i][1]] = 1.0
    # sigma=3 gaussian smoothing on site impulses
    k = np.exp(-0.5 * (np.arange(-9, 10) / SITE_SIGMA) ** 2)
    k /= k.max()
    site = np.stack([np.convolve(ch, k, mode="same") for ch in site])
    return state, site.astype(np.float32)


def revcomp_result(res: ContigResult) -> ContigResult:
    """ContigResult for the reverse-complemented contig: all bin tracks
    reversed; p_frame strand groups swapped with the genomic-frame
    remap (orig + frame g -> RC - frame 3+(L-g)%3, orig - frame 3+r
    -> RC + frame (L-r)%3); b_tracks donor/acceptor channels swap."""
    L = res.length
    pf = np.zeros_like(res.p_frame)
    for g in range(3):
        pf[:, 3 + (L - g) % 3] = res.p_frame[::-1, g]
        pf[:, (L - g) % 3] = res.p_frame[::-1, 3 + g]
    pf[:, 6] = res.p_frame[::-1, 6]
    pf[:, 7] = res.p_frame[::-1, 7]
    bt = res.b_tracks[::-1].copy()
    bt[:, [2, 3]] = bt[:, [3, 2]]
    return ContigResult(res.contig + "_rc", L, res.n_bins, res.bin_width,
                        p_region=res.p_region[::-1].copy(), p_frame=pf,
                        b_tracks=bt,
                        p_boundary=res.p_boundary[::-1].copy(),
                        p_coding=res.p_coding[::-1].copy(),
                        dtilde=res.dtilde[::-1].copy(),
                        e_lex=res.e_lex[::-1].copy())


def rc_labels(state, site):
    """Strand-flip label maps for the RC-augmented contig."""
    st_swap = {ST_CDS_P: ST_CDS_M, ST_CDS_M: ST_CDS_P,
               ST_INT_P: ST_INT_M, ST_INT_M: ST_INT_P}
    lut = np.array([st_swap.get(i, i) for i in range(N_STATES)],
                   np.int64)
    st = lut[state[::-1]]
    si = site[[S_START_M, S_STOP_M, S_DON_M, S_ACC_M,
               S_START_P, S_STOP_P, S_DON_P, S_ACC_P]][:, ::-1]
    return st.copy(), si.copy()


# ----------------------------------------------------------- dataset ---

class RefinerChunks(Dataset):
    """(seq, res, state, site, cond, tr, tr_no_ev) items -> chunks.

    tr_no_ev is the track stack with dtilde/e_lex zeroed;
    ~25% of chunks draw inputs from it (evidence dropout) so the
    refiner stays accurate when LexicMap evidence is absent."""

    def __init__(self, items, w_ev=0.25, p_no_ev=0.25, seed=0):
        self.items = items
        self.w_ev = w_ev
        self.p_no_ev = p_no_ev
        self.seed = seed
        self.chunks = []
        for i, it in enumerate(items):
            for a in _chunk_starts(len(it[0]), CHUNK, CHUNK_STRIDE):
                self.chunks.append((i, a, min(a + CHUNK, len(it[0]))))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        it, lo, hi = self.chunks[i]
        seq, res, state, site, cond, tr, tr_no_ev = self.items[it]
        # per-index RNG: identical evidence-dropout decisions regardless
        # of DataLoader worker partitioning
        tr = tr_no_ev if np.random.default_rng(
            self.seed + i).random() < self.p_no_ev else tr
        x = build_refiner_inputs(seq, res, lo, hi, self.w_ev, tracks=tr)
        ys, yt = state[lo:hi].copy(), site[:, lo:hi].copy()
        if hi - lo < CHUNK:
            # tail chunk at a contig end: pad to CHUNK so batch
            # tensors are uniform (default_collate's shared-storage
            # resize_ and torch.stack both require equal sizes).
            # state pad = intergenic so the site-loss mask stays off
            # the pad; site pad = no sites outside the contig.
            pad = CHUNK - (hi - lo)
            x = np.pad(x, ((0, 0), (0, pad)))
            ys = np.pad(ys, (0, pad), constant_values=ST_INTERGENIC)
            yt = np.pad(yt, ((0, 0), (0, pad)))
        return (torch.from_numpy(x), torch.from_numpy(cond),
                torch.from_numpy(ys), torch.from_numpy(yt))


def _chunk_collate(batch):
    """Uniform torch.stack collate (items are already fixed-size,
    but this avoids default_collate's shared-memory resize path)."""
    x, cond, ys, yt = zip(*batch)
    return (torch.stack(x), torch.stack(cond),
            torch.stack(ys), torch.stack(yt))


def _make_items(res_pkl: dict, labels_by_seqid: dict, gcode: int,
                w_ev: float = 0.25, include_rc=True):
    """Per-annotated-contig training items: (seq, res, state, site,
    cond, tracks, tracks_no_evidence)."""
    results = {r.contig: r for r in res_pkl["results"]}
    seqs = res_pkl["contig_seqs"]
    items = []
    for cid, rows in labels_by_seqid.items():
        if cid not in results or cid not in seqs:
            continue
        res, seq = results[cid], seqs[cid].upper()
        state, site = build_refiner_labels(rows, len(seq))
        no_ev = ContigResult(
            res.contig, res.length, res.n_bins, res.bin_width,
            p_region=res.p_region, p_frame=res.p_frame,
            b_tracks=res.b_tracks, p_boundary=res.p_boundary,
            p_coding=res.p_coding,
            dtilde=np.zeros_like(res.dtilde),
            e_lex=np.zeros_like(res.e_lex))
        cond = build_contig_cond(res, seq, gcode)
        items.append((seq, res, state, site, cond,
                      _stage1_tracks(res, w_ev),
                      _stage1_tracks(no_ev, w_ev)))
        if include_rc:
            res_rc = revcomp_result(res)
            st_rc, si_rc = rc_labels(state, site)
            no_ev_rc = revcomp_result(no_ev)
            items.append((revcomp(seq), res_rc, st_rc, si_rc, cond,
                          _stage1_tracks(res_rc, w_ev),
                          _stage1_tracks(no_ev_rc, w_ev)))
    return items


# ------------------------------------------------------------- loss ----

def refiner_loss(sl, tl, y_state, y_site, min_intron=35,
                 max_intron=5000):
    """L = L_state + L_site + 0.5 * L_phase (plan §5)."""
    L_state = F.cross_entropy(sl, y_state,
                              weight=STATE_W.to(sl.device))
    # focal BCE on sites, masked to +-500 bp around annotated sequence
    m = (y_state != ST_INTERGENIC).float()[:, None, :]
    m = F.max_pool1d(m, 1001, stride=1, padding=500)
    p = torch.sigmoid(tl)
    pt = p * y_site + (1.0 - p) * (1.0 - y_site)
    bce = F.binary_cross_entropy_with_logits(
        tl, y_site, reduction="none", pos_weight=torch.tensor(
            50.0, device=tl.device))
    L_site = (bce * (1.0 - pt).clamp(min=0) ** 2
              * m).sum() / m.sum().clamp(min=1.0) / N_SITES
    L_phase = _phase_loss(tl, y_site, min_intron, max_intron)
    return L_state + L_site + 0.5 * L_phase


def _phase_loss(tl, y_site, min_intron, max_intron):
    """Donor-acceptor co-firing: for each true donor, the best
    acceptor site inside the legal intron-length window should fire
    jointly (penalizes a lone confident donor with no partner)."""
    losses = []
    for don_ch, acc_ch in ((S_DON_P, S_ACC_P), (S_DON_M, S_ACC_M)):
        pd = torch.sigmoid(tl[:, don_ch])
        pa = torch.sigmoid(tl[:, acc_ch])
        for b in range(tl.shape[0]):
            for d in (y_site[b, don_ch] > 0.5).nonzero()[:, 0]:
                d = int(d)
                lo_w = d + min_intron
                hi_w = min(d + max_intron + 1, pa.shape[-1])
                pa_max = pa[b, lo_w:hi_w].max() if hi_w > lo_w \
                    else pa[b, d] * 0.0
                losses.append(1.0 - pd[b, d] * pa_max)
    if not losses:
        return tl.sum() * 0.0
    return torch.stack(losses).mean()


def _nuc_f1(sl, y_state):
    """Nucleotide CDS F1: predicted CDS+/- vs labelled CDS+/-."""
    pred = sl.argmax(1)
    p = (pred == ST_CDS_P) | (pred == ST_CDS_M)
    t = (y_state == ST_CDS_P) | (y_state == ST_CDS_M)
    tp = (p & t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    sn = tp / max(tp + fn, 1)
    sp = tp / max(tp + fp, 1)
    return f1, sn, sp


# ------------------------------------------------------------- main ----

def _stage1_cache(fasta: Path, model, cache_dir: Path, args):
    """ContigResult dict for a fasta: cached .res.pkl else a tracks-only
    annotate run with OMNIPHYL_DUMP_RES."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pk = cache_dir / (fasta.stem + ".res.pkl")
    if not pk.exists():
        if model is None:
            sys.exit(f"no Stage-1 cache {pk} and no --model given")
        os.environ["OMNIPHYL_DUMP_RES"] = "1"
        run_annotate(model_path=model, fasta_path=fasta,
                     out_prefix=cache_dir / fasta.stem,
                     window=args.window, stride=args.stride,
                     bin_width=args.bin_width,
                     batch_size=args.stage1_batch, device=args.device,
                     tracks_only=True, verbose=False)
    with open(pk, "rb") as fh:
        return pickle.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs=2, action="append", required=True,
                    metavar=("FASTA", "GFF3"),
                    help="(genome fasta, CDS gff3) pair; repeatable")
    ap.add_argument("--model", default=None,
                    help="Stage-1 ckpt to run/freeze for features")
    ap.add_argument("--res-pkl", type=Path, default=None,
                    help="cached res.pkl for the FIRST --data pair")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Stage-1 feature cache (.res.pkl per fasta)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bundle-model", type=Path, default=None,
                    help="Stage-1 ckpt to bundle the refiner into")
    ap.add_argument("--bundle-out", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader workers for chunk feature build")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of contigs held out (>=1 contig)")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--bin-width", type=int, default=0)
    ap.add_argument("--stage1-batch", type=int, default=256)
    ap.add_argument("--evidence-weight", type=float, default=0.25)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = ("cuda" if args.device == "auto"
              and torch.cuda.is_available() else
              "cpu" if args.device == "auto" else args.device)
    torch.manual_seed(args.seed)

    # ---- items per data pair -----------------------------------------
    items = []
    for pi, (fa_s, gff_s) in enumerate(args.data):
        fasta, gff = Path(fa_s), Path(gff_s)
        if pi == 0 and args.res_pkl:
            with open(args.res_pkl, "rb") as fh:
                res_pkl = pickle.load(fh)
        else:
            if args.model is None:
                sys.exit(f"{fasta}: no Stage-1 features -- pass "
                         "--model to run stage-1 or --res-pkl to load "
                         "a cache")
            cache = args.cache_dir or args.out.parent / "refiner_cache"
            res_pkl = _stage1_cache(fasta, args.model, cache, args)
        labels = parse_cds(str(gff))
        gcode = res_pkl.get("gcode", 1)
        got = _make_items(res_pkl, labels, gcode,
                          w_ev=args.evidence_weight)
        print(f"{fasta.name}: {len(got)} contig items "
              f"({sum(1 for _ in got)} incl. RC)", flush=True)
        items += got
    if not items:
        sys.exit("no labelled contigs (check seqid match fasta<->gff3)")

    # contig-level split: no chunk shares a contig across val/train
    idx = np.random.default_rng(args.seed).permutation(len(items))
    n_val = max(1, int(len(items) * args.val_frac))
    val_items = [items[i] for i in idx[:n_val]]
    trn_items = [items[i] for i in idx[n_val:]]
    trn = DataLoader(RefinerChunks(trn_items, args.evidence_weight,
                                   seed=args.seed),
                     batch_size=args.batch_size, shuffle=True,
                     num_workers=args.num_workers,
                     collate_fn=_chunk_collate)
    val = DataLoader(RefinerChunks(val_items, args.evidence_weight,
                                   p_no_ev=0.0, seed=args.seed),
                     batch_size=args.batch_size,
                     collate_fn=_chunk_collate)
    print(f"chunks: {len(trn.dataset)} train / {len(val.dataset)} val "
          f"({len(trn_items)} / {len(val_items)} contigs)", flush=True)

    refiner = OmniPhylRefiner().to(device)
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr,
                            weight_decay=0.01)
    best_f1, best_sd = -1.0, None
    for ep in range(args.epochs):
        refiner.train()
        tot, nb = 0.0, 0
        for x, cond, y_state, y_site in trn:
            x, cond = x.to(device), cond.to(device)
            y_state, y_site = y_state.to(device), y_site.to(device)
            sl, tl = refiner(x, cond)
            loss = refiner_loss(sl, tl, y_state, y_site)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        refiner.eval()
        f1s, vloss = [], []
        with torch.no_grad():
            for x, cond, y_state, y_site in val:
                x, cond = x.to(device), cond.to(device)
                y_state, y_site = y_state.to(device), y_site.to(device)
                sl, tl = refiner(x, cond)
                vloss.append(float(refiner_loss(sl, tl, y_state,
                                                y_site)))
                f1s.append(_nuc_f1(sl.cpu(), y_state.cpu()))
        f1 = float(np.mean([f[0] for f in f1s]))
        sn = float(np.mean([f[1] for f in f1s]))
        sp = float(np.mean([f[2] for f in f1s]))
        print(f"ep {ep + 1}: train {tot / max(nb, 1):.4f} "
              f"val {np.mean(vloss):.4f} nucF1 {f1:.3f} "
              f"(sn {sn:.3f} sp {sp:.3f})", flush=True)
        if f1 > best_f1:
            best_f1, best_sd = f1, {k: v.detach().clone()
                                    for k, v in
                                    refiner.state_dict().items()}
    if best_sd is not None:
        refiner.load_state_dict(best_sd)
    save_refiner(refiner, args.out,
                 extra={"nuc_f1": best_f1, "epochs": args.epochs,
                        "data": [f"{a} {b}" for a, b in args.data]})
    print(f"wrote {args.out} (val nucF1 {best_f1:.3f})")

    if args.bundle_model:
        if not args.bundle_out:
            sys.exit("--bundle-out required with --bundle-model")
        ck = torch.load(args.bundle_model, map_location="cpu",
                        weights_only=False)
        ck["refiner"] = {"model": refiner.state_dict(),
                         "config": {"c_in": 22, "cond_dim": 12,
                                    "hidden": 96, "n_states": N_STATES,
                                    "n_sites": N_SITES}}
        torch.save(ck, args.bundle_out)
        print(f"bundled -> {args.bundle_out}")


if __name__ == "__main__":
    main()
