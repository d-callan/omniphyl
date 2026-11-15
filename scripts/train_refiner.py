#!/usr/bin/env python3
"""Train the Stage-2 OmniPhylRefiner on (genome, GFF3) pairs.

Two halves per plan docs/neural-refiner-plan.md §5:

  inputs    Stage-1 ContigResult tracks (from a cached .res.pkl or by
            running `omniphyl annotate` stage-1 via OMNIPHYL_DUMP_RES)
            interpolated to per-base + nucleotide one-hot + clade cond
  targets   7-state per-base labels + 8 boundary-site channels (sigma=0.75
            gaussian smoothing + local peak CE) built from reference
            CDS rows

The Stage-1 model is FROZEN -- this script never touches its weights.
v2 chunks are 32768 bp with 24576 stride (config-driven; --v1-arch
reverts to 8192/6144); RC augmentation doubles each contig (labels
are rebuilt on strand-flipped canonical transcripts so codon-phase
targets stay exact, and ContigResult tracks are strand-flipped via
revcomp_result); evidence-dropout augmentation zeros the
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
import json
import math
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, Dataset,
                              WeightedRandomSampler)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omniphyl.refiner import (OmniPhylRefiner, build_refiner_inputs,
                              build_contig_cond, apply_cond_mode,
                              save_refiner, refiner_config, COND_MODES,
                              _stage1_tracks,
                              _chunk_starts, N_STATES, N_SITES,
                              N_COND, N_PHASE, CHUNK, CHUNK_STRIDE,
                              V1_CHUNK, V1_CHUNK_STRIDE,
                              ST_INTERGENIC, ST_CDS_P,
                              ST_INT_P, ST_CDS_M, ST_INT_M, ST_RNA,
                              ST_DUAL,
                              S_START_P, S_STOP_P, S_DON_P, S_ACC_P,
                              S_START_M, S_STOP_M, S_DON_M, S_ACC_M)
from omniphyl.annotate import ContigResult
from omniphyl.io import iter_sequences, revcomp
from eval_genes import parse_cds, _attrs
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

# Site-target smoothing width. sigma=3 saturated a ~13 bp plateau
# around each boundary, which made shifted decoy sites score as well
# as the true position during decode; sigma=0.75 keeps the impulse
# essentially local (<=1 bp context) so the site heads learn exact
# boundary placement. L_peak then adds the explicit argmax-at-site
# supervision the plateau erased.
SITE_SIGMA = 0.75
# state CE class weights: intergenic, CDS+, int+, CDS-, int-, rna, dual
STATE_W = torch.tensor([1.0, 1.5, 2.5, 1.5, 2.5, 1.0, 2.0])


# ----------------------------------------------------------- labels ----

# Transcript-level GFF feature types whose Parent links CDS rows to
# genes (isoform grouping) and whose own intervals label ST_RNA.
_TX_FEATURES = {"mRNA", "transcript", "lnc_RNA", "rRNA", "tRNA",
                "ncRNA", "snRNA", "snoRNA", "miRNA", "scRNA",
                "SRP_RNA", "RNase_P_RNA", "telomerase_RNA",
                "vault_RNA", "antisense_RNA", "tmRNA"}
_RNA_FEATURES = _TX_FEATURES - {"mRNA", "transcript"}


def parse_training_gff(path: str) -> dict:
    """{seqid: {"cds": {tx: {"strand": s, "ivs": [(s,e)]}},
                "tx2gene": {tx: gene_id}, "mane": {tx},
                "rna": [(s, e)]}}

    Three facts per seqid beyond eval_genes.parse_cds: tx->gene
    parents (isoform grouping), MANE tags (canonical-transcript
    choice), and noncoding-RNA intervals (ST_RNA state labels)."""
    out = defaultdict(lambda: {"cds": defaultdict(
        lambda: {"strand": "+", "ivs": []}),
        "tx2gene": {}, "mane": set(), "rna": []})
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9:
                continue
            seqid, ftype, strand = c[0], c[2], c[6]
            s, e = int(c[3]) - 1, int(c[4])
            if e <= s:
                continue
            a = _attrs(c[8])
            if ftype == "CDS":
                tx = (a.get("Parent") or "").split(",")[0]
                if tx:
                    rec = out[seqid]["cds"][tx]
                    rec["strand"] = strand
                    rec["ivs"].append((s, e))
            elif ftype in _TX_FEATURES:
                tid = a.get("ID")
                if not tid:
                    continue
                par = (a.get("Parent") or "").split(",")[0]
                if par:
                    out[seqid]["tx2gene"][tid] = par
                if "MANE" in (a.get("Dbxref", "")
                              + a.get("db_xref", "")):
                    out[seqid]["mane"].add(tid)
                if ftype in _RNA_FEATURES:
                    out[seqid]["rna"].append((s, e))
    return out


def canonical_txs(cds: dict, tx2gene: dict, mane: set) -> set:
    """One representative transcript per gene: MANE_Select when the
    annotation carries it, else the longest summed CDS length.
    Transcripts with no resolvable gene parent form their own
    singleton groups."""
    genes = defaultdict(list)
    for tx in cds:
        genes[tx2gene.get(tx, tx)].append(tx)
    keep = set()
    for txs in genes.values():
        mane_hits = [t for t in txs if t in mane]
        if mane_hits:
            keep.add(mane_hits[0])
        else:
            keep.add(max(
                txs,
                key=lambda t: sum(e - s for s, e in cds[t]["ivs"])))
    return keep


def build_refiner_labels(rows, L, rna_ivs=()):
    """(state [L] int64, site [8,L] float32, phase [L] int64) from
    canonical CDS rows of one seqid.

    rows: [(tx_id, strand, (s,e)), ...] 0-based half-open, grouped by
    transcript id so introns come from consecutive exon gaps.
    Site conventions (see plan §3.3):
      +  START at cds_start; STOP at cds_end-3; DONOR at intron_start;
         ACCEPTOR at intron_end-1
      -  START at cds_end-1; STOP at cds_start+2; DONOR at
         intron_end-1; ACCEPTOR at intron_start
    Opposite-strand CDS overlap is labelled ST_DUAL (v1's
    last-write-wins hid antisense genes entirely); noncoding-RNA
    intervals label ST_RNA only where no CDS covers them. phase
    classes are codon phase 0/1/2 (+ strand) and 3/4/5 (- strand),
    -1 off-CDS; minus-strand txs are written first so + wins on dual
    coverage."""
    by_tx = defaultdict(list)
    for tx, strand, (s, e) in rows:
        by_tx[(tx, strand)].append((s, e))
    cov = np.zeros((2, L), bool)
    int_ivs = []
    for (_tx, strand), ivs in by_tx.items():
        ivs = sorted(ivs)
        c = cov[0] if strand == "+" else cov[1]
        for i, (s, e) in enumerate(ivs):
            c[s:e] = True
            if i + 1 < len(ivs):
                int_ivs.append((e, ivs[i + 1][0], strand == "+"))
    both = cov[0] & cov[1]
    nocds = ~cov[0] & ~cov[1]
    state = np.zeros(L, np.int64)
    state[cov[0] & ~both] = ST_CDS_P
    state[cov[1] & ~both] = ST_CDS_M
    state[both] = ST_DUAL
    for s, e, plus in int_ivs:
        seg = state[s:e]
        seg[nocds[s:e]] = ST_INT_P if plus else ST_INT_M
    for s, e in rna_ivs:
        seg = state[max(s, 0):min(e, L)]
        seg[nocds[max(s, 0):min(e, L)]] = ST_RNA
    site = np.zeros((N_SITES, L), np.float32)
    phase = np.full(L, -1, np.int64)
    for (_tx, strand), ivs in sorted(
            by_tx.items(), key=lambda kv: kv[0][1] == "+"):
        ivs = sorted(ivs)
        plus = strand == "+"
        s0, e0 = ivs[0][0], ivs[-1][1]
        cum = 0
        for s, e in (ivs if plus else ivs[::-1]):
            if plus:
                phase[s:e] = (cum + np.arange(s, e) - s) % 3
            else:
                phase[s:e] = 3 + (cum + e - 1
                                  - np.arange(s, e)) % 3
            cum += e - s
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
    # sigma=0.75 gaussian smoothing on site impulses (see SITE_SIGMA)
    k = np.exp(-0.5 * (np.arange(-9, 10) / SITE_SIGMA) ** 2)
    k /= k.max()
    site = np.stack([np.convolve(ch, k, mode="same") for ch in site])
    return state, site.astype(np.float32), phase.astype(np.int8)


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

def _site_coords(site: np.ndarray) -> list[tuple]:
    """(C, L) site map -> per-channel (pos_i32, val_f32) pairs.

    Site targets are Gaussian-smoothed impulses (sigma=0.75 ramp over
    ~19 positions per site), so positions alone are not enough -- the
    BCE reads the soft values. ~19 nonzeros per site means this costs
    ~nothing vs 32 B/bp for the float32 bitmap."""
    return [(np.flatnonzero(site[c]).astype(np.int32),
             site[c][site[c] != 0].astype(np.float32))
            for c in range(site.shape[0])]


class RefinerChunks(Dataset):
    """(seq, bin_width, state_u8, site_coords, cond, tr) -> chunks.

    Evidence dropout: ~25% of chunks get dtilde/e_lex (output
    channels 20-21) zeroed, so the refiner stays accurate when
    LexicMap evidence is absent. Sites are stored as per-channel
    (pos, val) coordinate lists (sparse Gaussian impulses) and state
    as uint8 -- the full bitmap/int64 forms cost ~40 B/bp/item and
    OOM multi-genome corpora; this form is ~9 B/bp."""

    def __init__(self, items, w_ev=0.25, p_no_ev=0.25, seed=0,
                 cond_aug=False, chunk=CHUNK, chunk_stride=CHUNK_STRIDE):
        self.items = items
        self.w_ev = w_ev
        self.p_no_ev = p_no_ev
        self.seed = seed
        self.chunk = int(chunk)
        # cond_aug: training-only cond regularization ("regularize"
        # cond_mode). Inference/val items see the unmodified vector.
        self.cond_aug = cond_aug
        self.chunks = []
        # item[7] is the genome id used by --balance-genomes; items
        # without one fall back to their own index (never mixed).
        self.chunk_gid = []
        for i, it in enumerate(items):
            gid = it[7] if len(it) > 7 else i
            for a in _chunk_starts(len(it[0]), self.chunk,
                                   chunk_stride):
                self.chunks.append(
                    (i, a, min(a + self.chunk, len(it[0]))))
                self.chunk_gid.append(gid)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        it, lo, hi = self.chunks[i]
        seq, bw, state, sites, phase, cond, tr = self.items[it][:7]
        x = build_refiner_inputs(seq, None, lo, hi, self.w_ev,
                                 tracks=tr, bin_width=bw)
        # per-index RNG: identical evidence-dropout decisions regardless
        # of DataLoader worker partitioning; zeroing post-interp is
        # identical to interpolating a zeroed track
        if np.random.default_rng(self.seed + i).random() < self.p_no_ev:
            x[20:22] = 0.0
        if self.cond_aug:
            # regularize mode: prevent FiLM from keying on the exact
            # cond vectors of the small training corpus. p=0.3 fully
            # unconditioned; otherwise independently mask the
            # rank-confidence block (dims 12:20, the genome-identity
            # leak) and jitter the continuous track-stat dims.
            rng = np.random.default_rng(self.seed * 7919 + i + 31)
            cond = np.array(cond, dtype=np.float32, copy=True)
            if rng.random() < 0.30:
                cond[:] = 0.0
            else:
                cond[12:20] *= rng.random(8) >= 0.5
                cond[4:12] += rng.normal(0.0, 0.08, 8)
                cond[20] += rng.normal(0.0, 0.08)
        ys = state[lo:hi].astype(np.int64)
        yt = np.zeros((N_SITES, hi - lo), np.float32)
        for c, (pos, val) in enumerate(sites):
            m = (pos >= lo) & (pos < hi)
            if m.any():
                yt[c, pos[m] - lo] = val[m]
        # phase int8 with -1 off-CDS -> -100 (cross_entropy
        # ignore_index)
        yph = np.where(phase[lo:hi] < 0, -100,
                       phase[lo:hi]).astype(np.int64)
        if hi - lo < self.chunk:
            # tail chunk at a contig end: pad to chunk so batch
            # tensors are uniform (default_collate's shared-storage
            # resize_ and torch.stack both require equal sizes).
            # state pad = intergenic so the site-loss mask stays off
            # the pad; site pad = no sites outside the contig.
            pad = self.chunk - (hi - lo)
            x = np.pad(x, ((0, 0), (0, pad)))
            ys = np.pad(ys, (0, pad), constant_values=ST_INTERGENIC)
            yt = np.pad(yt, ((0, 0), (0, pad)))
            yph = np.pad(yph, (0, pad), constant_values=-100)
        return (torch.from_numpy(x), torch.from_numpy(cond),
                torch.from_numpy(ys), torch.from_numpy(yt),
                torch.from_numpy(yph))


def _chunk_collate(batch):
    """Uniform torch.stack collate (items are already fixed-size,
    but this avoids default_collate's shared-memory resize path)."""
    x, cond, ys, yt, yph = zip(*batch)
    return (torch.stack(x), torch.stack(cond),
            torch.stack(ys), torch.stack(yt), torch.stack(yph))


def _rc_rows(rows, L):
    """Strand-flipped canonical CDS rows for the RC-augmented contig:
    rebuilding labels on flipped rows (rather than array-splicing
    phase) keeps the codon-phase targets exact -- a reversed phase
    value is not recoverable without the transcript length."""
    flip = {"+": "-", "-": "+"}
    return [(tx, flip.get(strand, strand), (L - e, L - s))
            for tx, strand, (s, e) in rows]


def _make_items(res_pkl: dict, labels_by_seqid: dict, gcode: int,
                w_ev: float = 0.25, include_rc=True,
                cond_mode: str = "full", gid: int = 0):
    """Per-annotated-contig training items: (items, groups).

    items: (seq, bin_width, state_u8, site_coords, phase_i8, cond,
            tracks, gid)
    groups: parallel base-contig seqid for each item -- a contig and
    its reverse-complement partner share a group so a grouped split
    can never leak one strand into train and the other into val.
    gid is the genome id used by --balance-genomes. cond is stored
    with apply_cond_mode already applied; "regularize" stores the
    full vector because its jitter is stochastic per chunk."""
    results = {r.contig: r for r in res_pkl["results"]}
    seqs = res_pkl["contig_seqs"]
    items, groups = [], []
    for cid, lab in labels_by_seqid.items():
        if cid not in results or cid not in seqs:
            continue
        res, seq = results[cid], seqs[cid].upper()
        state, site, phase = build_refiner_labels(
            lab["rows"], len(seq), lab.get("rna", ()))
        cond = apply_cond_mode(build_contig_cond(res, seq, gcode),
                               cond_mode)
        bw = res.bin_width
        items.append((seq, bw, state.astype(np.uint8),
                      _site_coords(site), phase, cond,
                      _stage1_tracks(res, w_ev), gid))
        groups.append(cid)
        if include_rc:
            res_rc = revcomp_result(res)
            st_rc, si_rc, ph_rc = build_refiner_labels(
                _rc_rows(lab["rows"], len(seq)), len(seq),
                [(len(seq) - e, len(seq) - s)
                 for s, e in lab.get("rna", ())])
            items.append((revcomp(seq), bw, st_rc.astype(np.uint8),
                          _site_coords(si_rc), ph_rc, cond,
                          _stage1_tracks(res_rc, w_ev), gid))
            groups.append(cid)
    return items, groups


# ------------------------------------------------------------- loss ----

def refiner_loss(sl, tl, pl, y_state, y_site, y_phase,
                 min_intron=35, max_intron=5000):
    """L = L_state + L_site + 0.5*L_pair + L_peak + 0.5*L_phase_cls."""
    L_state = F.cross_entropy(sl, y_state,
                              weight=STATE_W.to(sl.device))
    # focal BCE on sites, masked to +-500 bp around annotated sequence
    m = (y_state != ST_INTERGENIC).float()[:, None, :]
    m = F.max_pool1d(m, 1001, stride=1, padding=500)
    p = torch.sigmoid(tl)
    pt = p * y_site + (1.0 - p) * (1.0 - y_site)
    # pos_weight 12 (was 50): with sharp sigma=0.75 targets the true
    # positive mass is ~1 bp per site, and 50 pushed the whole head
    # toward saturation -- every motif lit up as a confident boundary.
    bce = F.binary_cross_entropy_with_logits(
        tl, y_site, reduction="none", pos_weight=torch.tensor(
            12.0, device=tl.device))
    L_site = (bce * (1.0 - pt).clamp(min=0) ** 2
              * m).sum() / m.sum().clamp(min=1.0) / N_SITES
    L_pair = _phase_loss(tl, y_site, min_intron, max_intron)
    L_peak = _peak_loss(tl, y_site)
    L = L_state + L_site + 0.5 * L_pair + L_peak
    if pl is not None:
        L = L + 0.5 * F.cross_entropy(
            pl, y_phase, ignore_index=-100)
    return L


def _phase_loss(tl, y_site, min_intron, max_intron):
    """Donor-acceptor co-firing: for each true donor, the best
    acceptor site inside the legal intron-length window should fire
    jointly (penalizes a lone confident donor with no partner).

    The window direction is strand-aware: a minus-strand donor sits
    at the intron's RIGHT edge (transcript runs right-to-left), so
    its partner acceptor is at SMALLER genomic coords. The v1 loop
    applied the forward window to both strands, so the minus-strand
    term never saw the true acceptor."""
    losses = []
    for don_ch, acc_ch, fwd in ((S_DON_P, S_ACC_P, True),
                                (S_DON_M, S_ACC_M, False)):
        pd = torch.sigmoid(tl[:, don_ch])
        pa = torch.sigmoid(tl[:, acc_ch])
        for b in range(tl.shape[0]):
            for d in (y_site[b, don_ch] > 0.5).nonzero()[:, 0]:
                d = int(d)
                if fwd:
                    lo_w = d + min_intron
                    hi_w = min(d + max_intron + 1, pa.shape[-1])
                else:
                    lo_w = max(0, d - max_intron)
                    hi_w = d - min_intron + 1
                pa_max = pa[b, lo_w:hi_w].max() if hi_w > lo_w \
                    else pa[b, d] * 0.0
                losses.append(1.0 - pd[b, d] * pa_max)
    if not losses:
        return tl.sum() * 0.0
    return torch.stack(losses).mean()


def _peak_loss(tl, y_site, half=6):
    """Local peak-localization CE: at each annotated site, softmax over
    the +-half bp window of that channel's logits with the label
    window's argmax as target. BCE alone grades each base
    independently and is satisfied by a bump anywhere near the site;
    this term requires the channel's local argmax to sit exactly on
    the annotated boundary, which is what the decoder's site-maxima
    scan actually reads."""
    losses = []
    for b in range(tl.shape[0]):
        for ch in range(N_SITES):
            for p in (y_site[b, ch] > 0.5).nonzero()[:, 0]:
                p = int(p)
                lo = max(0, p - half)
                hi = min(tl.shape[-1], p + half + 1)
                if hi - lo < 3:
                    continue
                tgt = int(y_site[b, ch, lo:hi].argmax())
                losses.append(F.cross_entropy(
                    tl[b, ch, lo:hi][None],
                    torch.tensor([tgt], device=tl.device)))
    if not losses:
        return tl.sum() * 0.0
    return torch.stack(losses).mean()


def _peak_acc(tl, y_site, half=6):
    """Fraction of annotated sites whose predicted site-probability
    local argmax (+-half bp) sits on the exact labelled position --
    the decode-time quantity the sharp site loss optimizes."""
    p = torch.sigmoid(tl)
    n = ok = 0
    for b in range(tl.shape[0]):
        for ch in range(N_SITES):
            for pos in (y_site[b, ch] > 0.5).nonzero()[:, 0]:
                pos = int(pos)
                lo = max(0, pos - half)
                hi = min(p.shape[-1], pos + half + 1)
                n += 1
                ok += int(int(p[b, ch, lo:hi].argmax()) + lo == pos)
    return ok / max(n, 1)


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


def _split_items(tr_pairs, val_pairs, val_seqids, val_frac, seed):
    """(train, val) item lists from (item, group) pair lists.

    Selection order: --val-data items are already in val_pairs;
    --val-seqids moves whole base-contig groups out of tr_pairs; if
    nothing named a val set, a random val_frac share of distinct
    groups is held out. A group always moves as a unit so a contig's
    forward and reverse-complement items can never straddle the
    split.
    """
    tr_pairs, val_pairs = list(tr_pairs), list(val_pairs)
    val_seqids = set(val_seqids)
    if val_seqids:
        hit = {g for _it, g in tr_pairs} & val_seqids
        missing = val_seqids - hit
        if missing:
            print(f"warning: --val-seqids not in training pool: "
                  f"{sorted(missing)}", flush=True)
        val_pairs += [(it, g) for it, g in tr_pairs if g in val_seqids]
        tr_pairs = [(it, g) for it, g in tr_pairs
                    if g not in val_seqids]
    if not val_pairs:
        groups = sorted({g for _it, g in tr_pairs})
        idx = np.random.default_rng(seed).permutation(len(groups))
        n_val = max(1, int(len(groups) * val_frac))
        held = {groups[i] for i in idx[:n_val]}
        val_pairs = [(it, g) for it, g in tr_pairs if g in held]
        tr_pairs = [(it, g) for it, g in tr_pairs if g not in held]
    return ([it for it, _g in tr_pairs],
            [it for it, _g in val_pairs])


# ------------------------------------------------------------- main ----

# Genomes whose total bases exceed this are annotated in contig-group
# subprocesses (see _stage1_cache). 20 Mb per group: a 30 Mb part
# passed once then a sibling part OOM'd (rc=-9) on Kaggle, so 30 Mb
# sits right at the kill boundary; lmajor's 33 Mb whole-genome run
# also passed only marginally. ~2x headroom.
STAGE1_SPLIT_BP = 20_000_000


def _stage1_cache(fasta: Path, model, cache_dir: Path, args):
    """ContigResult dict for a fasta: cached .res.pkl else a tracks-only
    annotate run with OMNIPHYL_DUMP_RES.

    The annotate runs in a SUBPROCESS: stage-1 holds the whole genome's
    ContigResults plus multi-GB prediction batches, and Python/glibc do
    not reliably return that memory to the OS. Isolating it per genome
    keeps multi-genome runs from accumulating RSS until the OOM killer
    fires; the parent only ever touches the pickled cache.

    Returns the loaded res dict, or None if stage-1 failed (caller
    skips the genome; main() gates on how many failures to tolerate).

    Genomes over _SPLIT_BP total bases are annotated in contig-group
    subprocesses and merged: Stage-1 keeps every contig's tracks in
    RAM at once, and a ~65 Mb genome (tgondii ME49) was OOM-killed
    (rc=-9) on Kaggle while the ~33 Mb lmajor passed. Each ~30 Mb
    group's peak RSS stays below that bound."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pk = cache_dir / (fasta.stem + ".res.pkl")
    if pk.exists():
        with open(pk, "rb") as fh:
            return pickle.load(fh)
    if model is None:
        sys.exit(f"no Stage-1 cache {pk} and no --model given")

    # partition contigs into ~30 Mb groups (record_id is the first
    # header token, matching the GFF seqid either way)
    groups, cur, cur_bp = [], [], 0
    for cid, s, _cov in iter_sequences(fasta):
        if cur and cur_bp + len(s) > STAGE1_SPLIT_BP:
            groups.append(cur)
            cur, cur_bp = [], 0
        cur.append((cid, s))
        cur_bp += len(s)
    if cur:
        groups.append(cur)

    merged = {"results": [], "contig_seqs": {}, "stops": None,
              "gcode": 1}
    for gi, grp in enumerate(groups):
        if len(groups) == 1:
            fa_g, pk_g = fasta, pk
        else:
            fa_g = cache_dir / f"{fasta.stem}._p{gi}.fasta"
            pk_g = cache_dir / f"{fasta.stem}._p{gi}.res.pkl"
            if not pk_g.exists():
                with open(fa_g, "w") as out:
                    for cid, s in grp:
                        out.write(f">{cid}\n" + "\n".join(
                            s[i:i + 60]
                            for i in range(0, len(s), 60)) + "\n")
        if not pk_g.exists():
            spec = json.dumps({
                "model_path": str(model), "fasta_path": str(fa_g),
                "out_prefix": str(cache_dir / fa_g.stem),
                "window": args.window, "stride": args.stride,
                "bin_width": args.bin_width,
                "batch_size": args.stage1_batch,
                "device": args.device,
                "tracks_only": True, "verbose": False})
            print(f"{fa_g.name}: stage-1 annotate (subprocess)",
                  flush=True)
            r = subprocess.run(
                [sys.executable, "-c",
                 "import sys, json\n"
                 "from pathlib import Path\n"
                 "from omniphyl.annotate import run_annotate\n"
                 "kw = json.loads(sys.argv[1])\n"
                 "kw['out_prefix'] = Path(kw['out_prefix'])\n"
                 "run_annotate(**kw)\n",
                 spec],
                env={**os.environ, "OMNIPHYL_DUMP_RES": "1",
                     # child needs the repo's src on path; sys.path
                     # edits in this process don't propagate to
                     # subprocesses
                     "PYTHONPATH": str(ROOT / "src") + os.pathsep
                     + os.environ.get("PYTHONPATH", "")})
            if r.returncode != 0 or not pk_g.exists():
                print(f"{fa_g.name}: stage-1 FAILED "
                      f"rc={r.returncode} -- skipping this genome",
                      flush=True)
                return None
        with open(pk_g, "rb") as fh:
            part = pickle.load(fh)
        merged["results"] += part["results"]
        merged["contig_seqs"].update(part["contig_seqs"])
        if merged["stops"] is None:
            merged["stops"] = part.get("stops")
            merged["gcode"] = part.get("gcode", 1)
    if len(groups) > 1:
        with open(pk, "wb") as fh:
            pickle.dump(merged, fh)
        for gi in range(len(groups)):
            for suf in (".fasta", ".res.pkl"):
                p = cache_dir / f"{fasta.stem}._p{gi}{suf}"
                if p.exists():
                    p.unlink()
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", nargs=2, action="append", required=True,
                    metavar=("FASTA", "GFF3"),
                    help="(genome fasta, CDS gff3) pair; repeatable")
    ap.add_argument("--val-data", nargs=2, action="append", default=[],
                    metavar=("FASTA", "GFF3"),
                    help="(genome fasta, CDS gff3) pair reserved entirely "
                         "for validation; repeatable. When given, no "
                         "--data contigs are held out unless "
                         "--val-seqids also names some")
    ap.add_argument("--val-seqids", nargs="+", default=[],
                    help="base contig seqids from --data to hold out for "
                         "validation (both strands move together); "
                         "overrides --val-frac for those contigs")
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
    ap.add_argument("--cond-mode", choices=COND_MODES, default="full",
                    help="which cond dims FiLM sees: full (21 dims), "
                         "reduced (drop GC + rank-conf/depth/OOD leak "
                         "dims), none (no contig conditioning), "
                         "regularize (full + train-time dropout/jitter)")
    ap.add_argument("--balance-genomes", action="store_true",
                    help="WeightedRandomSampler so every genome "
                         "contributes equally per epoch regardless of "
                         "assembly size (default: flat chunk pool)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", choices=["none", "fp16", "bf16"],
                    default="bf16",
                    help="mixed precision autocast dtype (fp16 also "
                         "enables GradScaler; bf16 needs no scaler)")
    ap.add_argument("--arch-config", type=Path, default=None,
                    help="JSON of OmniPhylRefiner geometry overrides "
                         "(hidden, dilations, rnn_pool, tfm_pool, "
                         "tfm_layers, nhead, n_phase, chunk, "
                         "chunk_stride); merged over the v2 defaults")
    ap.add_argument("--v1-arch", action="store_true",
                    help="train the original v1 graph (6 blocks "
                         "d<=32, 4x BiGRU, 8192 chunks, no tfm/phase)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="resume-ckpt .pt (model+opt+scaler+epoch)")
    ap.add_argument("--resume-auto", action="store_true",
                    help="resume from <out>.resume.pt if present")
    args = ap.parse_args()

    # ------------------------------------------------------------- ddp ---
    rank, world, local = 0, 1, 0
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl" if torch.cuda.is_available()
                                else "gloo")
    is_main = rank == 0

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if world > 1 and dev == "cuda":
        torch.cuda.set_device(local)
        device = f"cuda:{local}"
    else:
        device = dev
    torch.manual_seed(args.seed)
    amp_dtype = {"fp16": torch.float16,
                 "bf16": torch.bfloat16}.get(args.amp)

    # v2 geometry defaults; --v1-arch reproduces the original graph
    arch = ({"hidden": 96,
             "dilations": [1, 2, 4, 8, 16, 32],
             "rnn_pool": 4, "tfm_pool": 64, "tfm_layers": 0,
             "nhead": 4, "n_phase": 0,
             "chunk": V1_CHUNK, "chunk_stride": V1_CHUNK_STRIDE}
            if args.v1_arch else
            {"hidden": 96,
             "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
             "rnn_pool": 8, "tfm_pool": 64, "tfm_layers": 2,
             "nhead": 4, "n_phase": N_PHASE,
             "chunk": CHUNK, "chunk_stride": CHUNK_STRIDE})
    if args.arch_config:
        arch.update(json.loads(args.arch_config.read_text()))
    chunk = int(arch["chunk"])
    chunk_stride = int(arch["chunk_stride"])

    # ---- items per data pair -----------------------------------------
    # Each entry: (item, group) where group is the base contig seqid --
    # a contig and its _rc partner share a group so a grouped split
    # can't leak the opposite strand across the train/val boundary.
    tr_pairs, val_pairs, failed = [], [], []

    def _load_pair(pi, fa_s, gff_s, is_val, gid=None):
        if gid is None:
            gid = pi + (1000 if is_val else 0)
        fasta, gff = Path(fa_s), Path(gff_s)
        if pi == 0 and args.res_pkl and not is_val:
            with open(args.res_pkl, "rb") as fh:
                res_pkl = pickle.load(fh)
        else:
            if args.model is None:
                sys.exit(f"{fasta}: no Stage-1 features -- pass "
                         "--model to run stage-1 or --res-pkl to load "
                         "a cache")
            cache = args.cache_dir or args.out.parent / "refiner_cache"
            res_pkl = _stage1_cache(fasta, args.model, cache, args)
            if res_pkl is None:
                return None
        gff_d = parse_training_gff(str(gff))
        gcode = res_pkl.get("gcode", 1)
        labels = {}
        for sid, d in gff_d.items():
            keep = canonical_txs(d["cds"], d["tx2gene"], d["mane"])
            rows = [(tx, rec["strand"], iv)
                    for tx, rec in d["cds"].items() if tx in keep
                    for iv in rec["ivs"]]
            labels[sid] = {"rows": rows, "rna": d["rna"]}
        items, groups = _make_items(res_pkl, labels, gcode,
                                    w_ev=args.evidence_weight,
                                    cond_mode=args.cond_mode, gid=gid)
        if is_main:
            print(f"{fasta.name}: {len(items)} contig items "
                  f"({'val' if is_val else 'train-pool'})", flush=True)
        return list(zip(items, groups))

    for pi, (fa_s, gff_s) in enumerate(args.data):
        pairs = _load_pair(pi, fa_s, gff_s, is_val=False)
        if pairs is None:
            failed.append(fa_s)
        else:
            tr_pairs += pairs
    for pi, (fa_s, gff_s) in enumerate(args.val_data):
        pairs = _load_pair(pi, fa_s, gff_s, is_val=True)
        if pairs is None:
            failed.append(fa_s)
        else:
            val_pairs += pairs
    if len(failed) > 1:
        sys.exit(f"{len(failed)} genomes failed stage-1 {failed} -- "
                 "more than 1 allowed; successful caches are on disk, "
                 "rerun to retry only the failures")
    if failed:
        print(f"WARNING: training without failed genome(s): "
              f"{[Path(f).name for f in failed]}", flush=True)
    if not tr_pairs:
        sys.exit("no labelled contigs (check seqid match fasta<->gff3)")

    # contig-group split: no chunk shares a base contig across
    # val/train, and a contig's forward+rc items always stay together
    trn_items, val_items = _split_items(
        tr_pairs, val_pairs, args.val_seqids, args.val_frac,
        args.seed)
    trn_ds = RefinerChunks(trn_items, args.evidence_weight,
                           seed=args.seed,
                           cond_aug=(args.cond_mode == "regularize"),
                           chunk=chunk, chunk_stride=chunk_stride)
    trn_kw = dict(batch_size=args.batch_size,
                  num_workers=args.num_workers,
                  collate_fn=_chunk_collate)
    if world > 1:
        trn_sampler = DistributedSampler(trn_ds, num_replicas=world,
                                         rank=rank, shuffle=True,
                                         seed=args.seed)
        trn = DataLoader(trn_ds, sampler=trn_sampler, **trn_kw)
    elif args.balance_genomes:
        # inverse-chunk-count weighting per genome: an 11k-chunk
        # L. major can no longer supply 25% of gradient steps while a
        # 0.8k-chunk Halobacterium supplies ~2%
        from collections import Counter
        cnt = Counter(trn_ds.chunk_gid)
        w = torch.tensor([1.0 / cnt[g] for g in trn_ds.chunk_gid],
                         dtype=torch.double)
        trn = DataLoader(trn_ds,
                         sampler=WeightedRandomSampler(
                             w, num_samples=len(trn_ds),
                             replacement=True),
                         **trn_kw)
    else:
        trn = DataLoader(trn_ds, shuffle=True, **trn_kw)
    val = DataLoader(RefinerChunks(val_items, args.evidence_weight,
                                   p_no_ev=0.0, seed=args.seed,
                                   chunk=chunk,
                                   chunk_stride=chunk_stride),
                     batch_size=args.batch_size,
                     collate_fn=_chunk_collate)
    if is_main:
        print(f"chunks: {len(trn.dataset)} train / "
              f"{len(val.dataset)} val "
              f"({len(trn_items)} / {len(val_items)} contigs) "
              f"arch={arch}", flush=True)

    refiner = OmniPhylRefiner(**{
        k: arch[k] for k in ("hidden", "dilations", "rnn_pool",
                             "tfm_pool", "tfm_layers", "nhead",
                             "n_phase")}).to(device)
    refiner.cond_mode = args.cond_mode
    refiner.chunk, refiner.chunk_stride = chunk, chunk_stride
    if world > 1:
        refiner = DistributedDataParallel(
            refiner, device_ids=[local] if dev == "cuda" else None)
    raw = refiner.module if world > 1 else refiner
    opt = torch.optim.AdamW(refiner.parameters(), lr=args.lr,
                            weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(amp_dtype == torch.float16))
    actx = (lambda: torch.autocast(
        device_type="cuda" if dev == "cuda" else "cpu",
        dtype=amp_dtype, enabled=amp_dtype is not None))

    best_f1, best_sd, start_ep = -1.0, None, 0
    resume_pt = args.out.parent / (args.out.stem + ".resume.pt")
    rp = args.resume or (resume_pt if args.resume_auto
                         and resume_pt.exists() else None)
    if rp is not None:
        ck_r = torch.load(rp, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck_r["model"])
        opt.load_state_dict(ck_r["opt"])
        scaler.load_state_dict(ck_r["scaler"])
        start_ep = int(ck_r["epoch"])
        best_f1 = float(ck_r.get("best_f1", -1.0))
        best_sd = ck_r.get("best_sd")
        if is_main:
            print(f"resumed {rp} at epoch {start_ep} "
                  f"(best nucF1 {best_f1:.3f})", flush=True)

    for ep in range(start_ep, args.epochs):
        if world > 1:
            trn_sampler.set_epoch(ep)
        refiner.train()
        tot, nb = 0.0, 0
        for x, cond, y_state, y_site, y_phase in trn:
            x, cond = x.to(device), cond.to(device)
            y_state, y_site = y_state.to(device), y_site.to(device)
            y_phase = y_phase.to(device)
            opt.zero_grad()
            with actx():
                sl, tl, pl = refiner(x, cond)
                loss = refiner_loss(sl, tl, pl, y_state, y_site,
                                    y_phase)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach())
            nb += 1
        refiner.eval()
        f1s, vloss, pacc = [], [], []
        with torch.no_grad():
            for x, cond, y_state, y_site, y_phase in val:
                x, cond = x.to(device), cond.to(device)
                y_state, y_site = y_state.to(device), y_site.to(device)
                y_phase = y_phase.to(device)
                with actx():
                    sl, tl, pl = refiner(x, cond)
                    vloss.append(float(refiner_loss(
                        sl, tl, pl, y_state, y_site, y_phase)))
                f1s.append(_nuc_f1(sl.float().cpu(), y_state.cpu()))
                pacc.append(_peak_acc(tl.float().cpu(),
                                      y_site.cpu()))
        f1 = float(np.mean([f[0] for f in f1s]))
        sn = float(np.mean([f[1] for f in f1s]))
        sp = float(np.mean([f[2] for f in f1s]))
        pk = float(np.mean(pacc)) if pacc else 0.0
        if is_main:
            print(f"ep {ep + 1}: train {tot / max(nb, 1):.4f} "
                  f"val {np.mean(vloss):.4f} nucF1 {f1:.3f} "
                  f"peakAcc {pk:.3f} (sn {sn:.3f} sp {sp:.3f})",
                  flush=True)
        if f1 > best_f1:
            best_f1, best_sd = f1, {k: v.detach().cpu().clone()
                                    for k, v in
                                    raw.state_dict().items()}
        if is_main:
            torch.save({"model": raw.state_dict(),
                        "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": ep + 1, "best_f1": best_f1,
                        "best_sd": best_sd, "arch": arch},
                       resume_pt)
        if world > 1:
            dist.barrier()
    if not is_main:
        dist.destroy_process_group()
        return
    if best_sd is not None:
        raw.load_state_dict(best_sd)
    save_refiner(raw, args.out,
                 extra={"nuc_f1": best_f1, "epochs": args.epochs,
                        "cond_mode": args.cond_mode,
                        "balance_genomes": args.balance_genomes,
                        "arch": arch,
                        "data": [f"{a} {b}" for a, b in args.data]})
    print(f"wrote {args.out} (val nucF1 {best_f1:.3f})")

    if args.bundle_model:
        if not args.bundle_out:
            sys.exit("--bundle-out required with --bundle-model")
        ck = torch.load(args.bundle_model, map_location="cpu",
                        weights_only=False)
        ck["refiner"] = {
            "format": "omniphyl-refiner-v2",
            "model": {k: v.detach().cpu().float()
                      for k, v in raw.state_dict().items()},
            "config": refiner_config(raw)}
        torch.save(ck, args.bundle_out)
        print(f"bundled -> {args.bundle_out}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
