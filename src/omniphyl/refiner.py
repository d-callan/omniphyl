"""Stage-2 neural boundary refiner (OmniPhylRefiner).

A small (~0.6M-param) per-nucleotide refiner conditioned on Stage-1
OmniPhyl bin tracks plus a contig-level clade vector (FiLM). Emits
base-exact state posteriors -- intergenic / CDS+- / intron+- /
rna / dual -- and 8 boundary-site channels (start/stop/donor/acceptor
per strand), decoded by a strand-normalized semi-Markov exon-chain
search into the same gene dicts as ``annotate._decode_genes``.

Spec: docs/neural-refiner-plan.md.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .annotate import (ContigResult, _coding_track, _fused_cscore,
                       _spliced, _exon_frames, _nonoverlap_select,
                       _intron_prior, STOPS)
from .encode import BASE2DIGIT
from .io import revcomp

# ------------------------------------------------------------- spec ----

CHUNK = 8192
CHUNK_STRIDE = 6144
N_CHANNELS = 22
N_COND = 12
N_STATES = 7
N_SITES = 8

# state posteriors (strand-normalized on -)
ST_INTERGENIC = 0
ST_CDS_P = 1
ST_INT_P = 2
ST_CDS_M = 3
ST_INT_M = 4
ST_RNA = 5
ST_DUAL = 6

# site channels: 0-3 + strand, 4-7 - strand (start/stop/donor/acceptor)
S_START_P, S_STOP_P, S_DON_P, S_ACC_P = 0, 1, 2, 3
S_START_M, S_STOP_M, S_DON_M, S_ACC_M = 4, 5, 6, 7


def _chunk_starts(L: int, chunk: int = CHUNK,
                  stride: int = CHUNK_STRIDE) -> list[int]:
    """Overlapping chunk offsets covering [0, L)."""
    if L <= chunk:
        return [0]
    starts = list(range(0, L - chunk + 1, stride))
    if starts[-1] != L - chunk:
        starts.append(L - chunk)
    return starts


# --------------------------------------------------------- features ----

def _interp_track(track: np.ndarray, lo: int, hi: int,
                  B: int) -> np.ndarray:
    """Linear interpolation of per-bin track values to per-base over
    [lo, hi). Bin centers sit at b*B + B/2; np.interp clamps at the
    edges. Handles [nbins] and [nbins, C] (returns [L] or [L, C])."""
    nb = track.shape[0]
    xc = np.arange(nb) * B + B / 2.0
    xb = np.arange(lo, hi) + 0.5
    if track.ndim == 1:
        return np.interp(xb, xc, track.astype(np.float64)).astype(
            np.float32)
    return np.stack([np.interp(xb, xc, track[:, c].astype(np.float64))
                     for c in range(track.shape[1])],
                    axis=1).astype(np.float32)


def build_refiner_inputs(seq: str, res: ContigResult, lo: int, hi: int,
                         w_ev: float = 0.25) -> np.ndarray:
    """(22, L) float32 input tensor for slice seq[lo:hi].

    Channels:
      0-3   one-hot ACGT (0.25 uniform on N/ambiguous)
      4     ambiguous-base mask
      5     p_coding (region head or raw)
      6     cscore (evidence-fused coding score)
      7     p_boundary
      8-15  p_frame; + channels (8-10) and - channels (11-13) rotated
            by lo % 3 so channel 8+f is always the CHUNK-LOCAL frame
            f = (x - lo) % 3
      16-19 tanh(b_tracks)  [start, stop, donor/acceptor impulses]
      20    dtilde (evidence coverage)
      21    tanh(e_lex) (evidence endpoints)
    """
    L = hi - lo
    x = np.zeros((N_CHANNELS, L), np.float32)
    codes = BASE2DIGIT[np.frombuffer(
        seq[lo:hi].encode("ascii", "replace"), dtype=np.uint8)]
    valid = codes >= 0
    x[0:4] = 0.25
    x[codes[valid], np.flatnonzero(valid)] = 1.0
    x[4] = (~valid).astype(np.float32)

    cscore = np.clip(_fused_cscore(res, w_ev), 0.0, 1.0)
    tracks = np.column_stack([
        _coding_track(res),            # 5
        cscore,                        # 6
        res.p_boundary,                # 7
        res.p_frame,                   # 8-15
        np.tanh(res.b_tracks),         # 16-19
        res.dtilde,                    # 20
        np.tanh(res.e_lex),            # 21
    ])
    x[5:] = _interp_track(tracks, lo, hi, res.bin_width).T
    rot = lo % 3
    if rot:
        # p_frame channel g encodes global frame x % 3 == g; after a
        # chunk offset the same bases carry local frame (x-lo) % 3,
        # so roll + and - frame groups down by rot.
        x[8:11] = np.roll(x[8:11], -rot, axis=0)
        x[11:14] = np.roll(x[11:14], -rot, axis=0)
    return x


def build_contig_cond(res: ContigResult, seq: str,
                      gcode: int = 1) -> np.ndarray:
    """(12,) float32 clade/contig-conditioning vector (FiLM input).

    0-3   domain one-hot (Euk/Bact/Arch/Vir) scaled by call confidence
    4     mean coding mass      5  coding fraction >= 0.5
    6     max-frame share on coding bins
    7     mean splice-impulse mass on coding bins
    8     contig GC             9  genetic code == 4
    10    evidence present      11 log10(length)/7
    """
    cond = np.zeros(N_COND, np.float32)
    tax = getattr(res, "taxon", None)
    names = getattr(tax, "names", None) or {}
    conf = getattr(tax, "path_conf", None) or {}
    dom_w = float(conf.get("domain", 0.0)) or 0.25
    for i, d in enumerate(("Eukaryota", "Bacteria", "Archaea",
                           "Viruses")):
        if names.get("domain") == d:
            cond[i] = dom_w
    if not cond[:4].any():
        cond[:4] = 0.25
    coding = _coding_track(res)
    cond[4] = float(coding.mean())
    cod_bins = coding >= 0.5
    cond[5] = float(cod_bins.mean())
    if res.p_frame[:, :6].sum() > 0 and cod_bins.any():
        cond[6] = float(res.p_frame[cod_bins, :6].max(1).mean())
        cond[7] = float(np.tanh(
            res.b_tracks[cod_bins, 2:4].sum(1)).mean())
    su = seq.upper()
    cond[8] = (su.count("G") + su.count("C")) / max(len(su), 1)
    cond[9] = 1.0 if gcode == 4 else 0.0
    cond[10] = 1.0 if len(res.dtilde) and float(res.dtilde.max()) > 0 \
        else 0.0
    cond[11] = min(1.0, math.log10(max(res.length, 100)) / 7.0)
    return cond


# ------------------------------------------------------------- model ---

class RefinerResBlock(nn.Module):
    """Dilated conv block with FiLM conditioning on the clade vector."""

    def __init__(self, channels: int, dilation: int, cond_dim: int,
                 p_drop: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 7,
                              padding=3 * dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, channels)
        self.film = nn.Linear(cond_dim, 2 * channels)
        self.out = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(p_drop)

    def forward(self, h, cond):
        gamma, beta = self.film(cond)[:, :, None].chunk(2, dim=1)
        z = self.norm(self.conv(h))
        z = (1.0 + gamma) * z + beta
        z = self.drop(F.gelu(z))
        return h + self.out(z)


class OmniPhylRefiner(nn.Module):
    """Per-nucleotide Stage-2 refiner.

    stem(22->96) -> 6 FiLM-conditioned dilated res blocks
    (d=1..32, receptive field ~1.3 kb) -> 4x-downsampled BiGRU
    (context ~4 kb both directions) -> linear upsample -> per-base
    heads: state softmax [7], boundary-site logits [8] (start, stop,
    donor, acceptor x strand).
    """

    def __init__(self, c_in: int = N_CHANNELS, cond_dim: int = N_COND,
                 hidden: int = 96, n_states: int = N_STATES,
                 n_sites: int = N_SITES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, hidden, 7, padding=3),
            nn.GroupNorm(8, hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            RefinerResBlock(hidden, d, cond_dim)
            for d in (1, 2, 4, 8, 16, 32))
        self.rnn = nn.GRU(hidden, hidden, batch_first=True,
                          bidirectional=True)
        self.rnn_proj = nn.Linear(2 * hidden, hidden)
        self.head_state = nn.Conv1d(hidden, n_states, 1)
        self.head_site = nn.Conv1d(hidden, n_sites, 3, padding=1)

    def forward(self, x, cond):
        h = self.stem(x)                              # [B, C, L]
        for blk in self.blocks:
            h = blk(h, cond)
        L = h.shape[-1]
        z = F.avg_pool1d(h, 4, 4)                     # [B, C, L/4]
        z, _ = self.rnn(z.transpose(1, 2))            # [B, L/4, 2C]
        z = self.rnn_proj(z).transpose(1, 2)          # [B, C, L/4]
        z = F.interpolate(z, size=L, mode="linear",
                          align_corners=False)
        h = h + z
        return self.head_state(h), self.head_site(h)


def save_refiner(refiner: OmniPhylRefiner, path, extra: dict = None):
    torch.save({"format": "omniphyl-refiner-v1",
                "model": refiner.state_dict(),
                "config": {"c_in": N_CHANNELS, "cond_dim": N_COND,
                           "hidden": 96, "n_states": N_STATES,
                           "n_sites": N_SITES},
                **(extra or {})}, path)


def load_refiner(source, device: str = "cpu") -> OmniPhylRefiner:
    """Load a refiner from a .pt file path, or from a checkpoint dict
    containing a ``"refiner"`` key (bundled in a Stage-1 ckpt)."""
    ck = (torch.load(source, map_location="cpu", weights_only=False)
          if isinstance(source, (str, Path)) else source)
    if "refiner" in ck:
        ck = ck["refiner"]
    cfg = ck.get("config", {})
    ref = OmniPhylRefiner(**{k: cfg[k] for k in
                             ("c_in", "cond_dim", "hidden", "n_states",
                              "n_sites") if k in cfg})
    ref.load_state_dict(ck["model"])
    return ref.to(device).eval()


@torch.no_grad()
def refiner_tracks(refiner: OmniPhylRefiner, seq: str,
                   res: ContigResult, w_ev: float = 0.25,
                   gcode: int = 1, device: str = "cpu",
                   batch_size: int = 8):
    """Slide the refiner over a contig in 8192-bp chunks (stride 6144),
    blending overlaps with a raised-cosine taper. Returns
    (p_state [L,7], p_site [L,8]) float32."""
    L = len(seq)
    cond_np = build_contig_cond(res, seq, gcode)
    cond = torch.from_numpy(cond_np)[None].to(device)
    starts = _chunk_starts(L)
    p_state = np.zeros((L, N_STATES), np.float64)
    p_site = np.zeros((L, N_SITES), np.float64)
    wsum = np.zeros(L, np.float64)
    for i in range(0, len(starts), batch_size):
        batch = starts[i:i + batch_size]
        xs = np.stack([build_refiner_inputs(
            seq, res, a, min(a + CHUNK, L), w_ev) for a in batch])
        sl, tl = refiner(torch.from_numpy(xs).to(device),
                         cond.repeat(len(batch), 1))
        ps = torch.softmax(sl.float(), 1).cpu().numpy()
        pt = torch.sigmoid(tl.float()).cpu().numpy()
        for a, p_s, p_t in zip(batch, ps, pt):
            n = p_s.shape[-1]
            w = np.clip(np.sin(np.pi * np.arange(n)
                               / max(n - 1, 1)), 1e-3, 1.0) ** 2
            p_state[a:a + n] += w[:, None] * p_s.T
            p_site[a:a + n] += w[:, None] * p_t.T
            wsum[a:a + n] += w
    return (p_state / wsum[:, None]).astype(np.float32), \
           (p_site / wsum[:, None]).astype(np.float32)


# ----------------------------------------------------------- decoder ---

def decode_with_refiner(refiner, seq: str, res: ContigResult,
                        stops=STOPS, euk: bool = True, gcode: int = 1,
                        w_ev: float = 0.25, min_cds_len: int = 90,
                        coding_threshold: float = 0.45,
                        min_intron: int = 35, max_intron: int = 5000,
                        intron_hints=(), site_threshold: float = 0.05,
                        device: str = "cpu", p_state=None, p_site=None):
    """Decode a contig with the neural refiner -> gene dicts.

    Same return schema as annotate._decode_genes so downstream GFF3 /
    domain / report code paths are unchanged. ``refiner`` may be None
    when precomputed ``p_state``/``p_site`` are supplied (tests,
    res.pkl replays).
    """
    seq_u = seq.upper()
    L = len(seq_u)
    if p_state is None or p_site is None:
        p_state, p_site = refiner_tracks(refiner, seq_u, res, w_ev,
                                         gcode, device)
    genes = []
    for strand in ("+", "-"):
        genes += _refiner_decode_strand(
            seq_u, L, p_state, p_site, res, strand, euk, stops,
            min_cds_len, coding_threshold, min_intron, max_intron,
            intron_hints, site_threshold)
    seen, uniq = set(), []
    for g in sorted(genes, key=lambda x: -x.get("score", 0.0)):
        key = (g["start"], g["end"], g["strand"])
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    uniq = _nonoverlap_select(uniq)
    return sorted(uniq, key=lambda g: (g["start"], g["end"]))


def _refiner_decode_strand(seq_u, L, p_state, p_site, res, strand,
                           euk, stops, min_cds_len, coding_threshold,
                           min_intron, max_intron, intron_hints,
                           site_threshold):
    """One strand, all in the strand's reading direction.

    For '-' the sequence is revcomp'd and every posterior row is
    reversed, so loci/introns/ORFs are scanned forward; results are
    mapped back to genomic coords at emit time.
    """
    if strand == "+":
        s_seq = seq_u
        p_cds = p_state[:, ST_CDS_P]
        p_int = p_state[:, ST_INT_P]
        p_other = p_state[:, ST_CDS_M]
        p_start, p_stop = p_site[:, S_START_P], p_site[:, S_STOP_P]
        p_don, p_acc = p_site[:, S_DON_P], p_site[:, S_ACC_P]
        hints = [iv[:2] for iv in intron_hints]
    else:
        s_seq = revcomp(seq_u)
        p_cds = p_state[::-1, ST_CDS_M]
        p_int = p_state[::-1, ST_INT_M]
        p_other = p_state[::-1, ST_CDS_P]
        p_start = p_site[::-1, S_START_M]
        p_stop = p_site[::-1, S_STOP_M]
        p_don = p_site[::-1, S_DON_M]
        p_acc = p_site[::-1, S_ACC_M]
        hints = [(L - a, L - d) for d, a in
                 (iv[:2] for iv in intron_hints)]
    p_bg = 1.0 - p_cds
    loci = _refiner_loci(p_cds, p_int, coding_threshold, L)
    iprior = _intron_prior(res)
    out = []
    for lo, hi in loci:
        out += _refiner_decode_locus(
            s_seq, lo, hi, strand, L, res, euk, stops, min_cds_len,
            coding_threshold, min_intron, max_intron, hints,
            site_threshold, p_cds, p_int, p_other, p_bg, p_start,
            p_stop, p_don, p_acc, iprior)
    return out


def _refiner_loci(p_cds, p_int, coding_threshold, L, pad=300):
    """Active gene loci: contiguous regions where the 100 bp-smoothed
    exon+intron posterior reaches 0.20 and the run's max CDS posterior
    passes coding_threshold. Padded +-300 bp and merged."""
    sig = np.convolve(p_cds + p_int, np.ones(100) / 100.0, mode="same")
    mask = sig >= 0.20
    loci, i = [], 0
    while i < L:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < L and mask[j + 1]:
            j += 1
        if p_cds[i:j + 1].max() >= coding_threshold:
            loci.append((max(0, i - pad), min(L, j + 1 + pad)))
        i = j + 1
    merged = []
    for lo, hi in loci:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return [iv for iv in merged if iv[1] - iv[0] > 100]


def _refiner_intron_candidates(s_seq, lo, hi, min_intron, max_intron,
                               p_don, p_acc, p_int, p_cds, hints,
                               site_threshold):
    """Scored motif-validated intron candidates on the strand's
    transcript coords inside [lo, hi), as (ll, d, acc) sorted by ll
    desc. The junction likelihood is the sum of the donor/acceptor
    site log-probs, the mean intron-state log-posterior over the
    candidate, minus the mean CDS log-posterior (a false intron
    inside an exon scores badly), plus +2.0 when a split-alignment
    hint matches within 6 bp."""
    hint_xy = list(hints)
    dons = [d for d in range(lo + 3, hi - min_intron)
            if s_seq[d:d + 2] in ("GT", "GC")
            and (p_don[d] >= site_threshold
                 or any(d - 6 <= hd <= d + 6 for hd, _ in hint_xy))]
    accs = [a for a in range(lo + min_intron + 2, hi)
            if s_seq[a - 2:a] == "AG"
            and (p_acc[a - 1] >= site_threshold
                 or any(a - 6 <= ha <= a + 6 for _, ha in hint_xy))]
    out = []
    for d in dons:
        for a in accs:
            if not (min_intron <= a - d <= max_intron):
                continue
            ll = (math.log(max(float(p_don[d]), 1e-6))
                  + math.log(max(float(p_acc[a - 1]), 1e-6))
                  + float(np.log(np.clip(p_int[d:a], 1e-6, 1)).mean())
                  - float(np.log(np.clip(p_cds[d:a], 1e-6, 1)).mean()))
            if any(abs(d - hd) <= 6 and abs(a - ha) <= 6
                   for hd, ha in hint_xy):
                ll += 2.0
            if ll > -3.0:
                out.append((ll, d, a))
    return sorted(out, key=lambda x: -x[0])


def _refiner_intron_chains(cands, iprior, top_k_cand=24, K=8):
    """Top-K non-overlapping intron subsets for ORF scanning.

    Same keep/skip subset problem as annotate._intron_sets -- each
    candidate's edge score is discounted by the taxon-conditioned
    intron-length prior -- but with overlap handling: mutually
    overlapping candidates are bucketed and a chain keeps at most the
    two best members per bucket. The empty chain and every singleton
    are always returned so unspliced and single-intron scans always
    reach arbitration."""
    mu, sigma, alpha = iprior
    cands = [(ll - alpha * abs((a - d) - mu) / sigma, d, a)
             for ll, d, a in cands[:top_k_cand]]
    if not cands:
        return [[]]
    buckets, cur, cur_hi = [], [], -1
    for ll, d, a in sorted(cands, key=lambda x: (x[1], x[2])):
        if d >= cur_hi:
            if cur:
                buckets.append(cur)
            cur, cur_hi = [], -1
        cur.append((ll, d, a))
        cur_hi = max(cur_hi, a)
    if cur:
        buckets.append(cur)
    best = [(0.0, ())]
    for bucket in buckets:
        opts = [(0.0, None)]
        opts += [(ll, (d, a))
                 for ll, d, a in sorted(bucket, key=lambda x: -x[0])[:2]]
        best = sorted((s + sc, st if iv is None else st + (iv,))
                      for s, st in best for sc, iv in opts)[:K]
        best = sorted(best, key=lambda x: -x[0])
    chains = [list(st) for _s, st in best]
    have = {tuple(st) for st in chains}
    for _ll, d, a in cands:
        if ((d, a),) not in have:
            have.add(((d, a),))
            chains.append([(d, a)])
    return chains


def _refiner_orfs(seq_s, pos, lo, hi, chain, strand, L, res, stops,
                  min_cds_len, s_seq, p_cds, p_int, p_other, p_bg,
                  p_start, p_stop, hint_xy, intron_ll):
    """ATG->stop ORFs on the (possibly spliced) forward transcript.

    ``pos`` maps transcript index -> strand-local genomic coord
    (revcomp space on '-'). Exon-chain phase and junction codons are
    implicit: a wrong intron yields an in-frame stop and the ORF
    truncates. Scores come from refiner state/site posteriors plus the
    Stage-1 frame-consistency term (each exon's implied global frame
    via annotate._exon_frames, so a mis-phased join loses support)."""
    n = len(seq_s)
    out = []
    for t in range(n - 2):
        if seq_s[t:t + 3] != "ATG":
            continue
        u = t + 3
        while u + 3 <= n and seq_s[u:u + 3] not in stops:
            u += 3
        if u + 3 > n:
            break  # no stop downstream for this or any later start
        a_t, b_t = t, u + 3            # CDS occupies [a_t, b_t) tx
        a, bnd = pos[a_t], pos[b_t - 1] + 1
        if b_t - a_t < min_cds_len or a >= hi or bnd <= lo:
            continue
        ivs = [iv for iv in chain if a < iv[0] and iv[1] <= bnd]
        tex = np.array(pos[a_t:b_t])
        cds_ll = float((np.log(np.clip(p_cds[tex], 1e-6, 1))
                        - np.log(np.clip(p_bg[tex], 1e-6, 1))).mean())
        share = float((p_cds[tex]
                       / (p_cds[tex] + p_other[tex] + 1e-9)).mean())
        if share < 0.25:
            continue
        g = {"strand": strand, "introns": [], "frame": None,
             "domains": []}
        if strand == "+":
            g["start"], g["end"], g["introns"] = a, bnd, ivs
        else:
            g["start"], g["end"] = L - bnd, L - a
            g["introns"] = [(L - acc, L - d) for d, acc in ivs]
        n_hint = sum(1 for d, acc in ivs
                     if any(abs(d - hd) <= 12 and abs(acc - ha) <= 12
                            for hd, ha in hint_xy))
        if res.p_frame[:, :6].sum() > 0:
            exf = _exon_frames(g, res)
            n_fb = sum(b1 - b0 for _g, b0, b1 in exf)
            frame_sup = (sum(float(res.p_frame[b0:b1, gx].sum())
                             for gx, b0, b1 in exf)
                         / max(n_fb, 1)) if n_fb else 0.0
        else:
            frame_sup = 0.0
        cds_len = b_t - a_t
        g["body"] = max(0.0, cds_ll + frame_sup)
        g["n_hint_introns"] = n_hint
        g["strand_share"] = share
        g["score"] = ((cds_ll + frame_sup) * math.log1p(cds_len)
                      + math.log(max(float(p_start[a]), 1e-6))
                      + math.log(max(float(p_stop[pos[b_t - 3]]), 1e-6))
                      + sum(intron_ll.get(iv, 0.0) for iv in ivs)
                      + 0.4 * n_hint)
        if g["score"] > 0:
            out.append(g)
    return out


def _refiner_decode_locus(s_seq, lo, hi, strand, L, res, euk, stops,
                          min_cds_len, coding_threshold, min_intron,
                          max_intron, hints, site_threshold, p_cds,
                          p_int, p_other, p_bg, p_start, p_stop,
                          p_don, p_acc, iprior):
    """All candidate gene calls inside one active locus on one
    strand: candidate introns -> top-K excision chains -> spliced-
    transcript ORF scans -> per-(start,end) best-score dict."""
    cands = (_refiner_intron_candidates(
        s_seq, lo, hi, min_intron, max_intron, p_don, p_acc, p_int,
        p_cds, hints, site_threshold) if euk else [])
    chains = _refiner_intron_chains(cands, iprior)
    intron_ll = {(d, a): ll for ll, d, a in cands}
    hint_xy = list(hints)
    genes = {}
    for chain in chains:
        seq_s, pos = _spliced(s_seq, lo, hi, chain)
        for g in _refiner_orfs(
                seq_s, pos, lo, hi, chain, strand, L, res, stops,
                min_cds_len, s_seq, p_cds, p_int, p_other, p_bg,
                p_start, p_stop, hint_xy, intron_ll):
            key = (g["start"], g["end"])
            if key not in genes or g["score"] > genes[key]["score"]:
                genes[key] = g
    return list(genes.values())
