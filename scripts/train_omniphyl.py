#!/usr/bin/env python3
"""Train OmniPhyl Tier-0: per-fragment FCGR -> multi-rank taxonomy.

Reads fragments.fna + labels.tsv emitted by fetch_refseq_corpus.py.
FCGRs are computed on the fly in the DataLoader workers (no big tensors
on disk): sequence -> k-mer counts -> quadtree pixel image [1, 2^k, 2^k].

Splits are by *genome accession* so train/val never share a genome.
Optional --holdout-genus-frac holds out whole genera entirely (the
leave-clade-out probe: does the model still call their family/order?).

Deps: torch, numpy. Usage:
  python train_omniphyl.py --fragments ../data/tier0/fragments.fna \
      --labels ../data/tier0/labels.tsv --out ../runs/tier0_v1 \
      --epochs 30 --batch-size 256 --holdout-genus-frac 0.1
"""
import argparse
import csv
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
CODING_TARGETS = ["coding", "non-coding"]  # legacy binary labels; superseded by fractions
# Continuous annotation heads (see docs/annotation-cascade.md):
#   coding_frac  mixture-aware fraction target (replaces binary coding)
#   boundary     candidacy: derived at loss time as 0 < coding_frac < 1
#   offset       normalized position of nearest interior CDS edge
CONT_TARGETS = ["coding_frac", "boundary", "offset"]
BASE2DIGIT = np.full(256, -1, dtype=np.int8)
for i, b in enumerate(b"ACGT"):
    BASE2DIGIT[b] = i
    BASE2DIGIT[b + 32] = i  # lowercase

# Quadtree pixel layout: digit -> (x,y) step. A=(0,0) C=(0,1) G=(1,0) T=(1,1)
# x = digit>>1, y = digit&1. Last base of the k-mer is the outermost pixel bit.


def build_cell_lut(k: int) -> np.ndarray:
    """kmer base-4 index -> flattened pixel index (row=y, col=x)."""
    side = 1 << k
    lut = np.zeros(4 ** k, dtype=np.int32)
    idx = np.arange(4 ** k, dtype=np.int64)
    x = np.zeros_like(idx)
    y = np.zeros_like(idx)
    # standard CGR addressing: the most recent base (least significant
    # base-4 digit) owns the outermost pixel bit — p_i = v_i/2 + v_{i-1}/4...
    for i in range(k):
        d = (idx >> (2 * i)) & 3  # base i positions back from the k-mer end
        x |= (d >> 1) << (k - 1 - i)
        y |= (d & 1) << (k - 1 - i)
    lut = (y * side + x).astype(np.int32)
    return lut


MS_KS = (4, 5, 6)  # coarse->fine FCGR pyramid scales (backbone="msfcgr")


def rc_index_map(k: int) -> np.ndarray:
    """base-4 kmer index -> canonical index = min(idx, RC(idx)).

    Complement of base-4 digit d is 3-d (A<->T, C<->G); RC reverses digit
    order. Mapping every observed kmer through its canonical index makes
    FCGR counts strand-invariant at zero runtime cost.
    """
    idx = np.arange(4 ** k, dtype=np.int64)
    rc = np.zeros_like(idx)
    for i in range(k):
        rc |= (3 - ((idx >> (2 * i)) & 3)) << (2 * (k - 1 - i))
    return np.minimum(idx, rc).astype(np.int32)


# vertex coords for CGR points; index by code+1 so N (-1) -> center pull
VTX = np.array([[0.5, 0.5],  # invalid/N
                [0.0, 0.0],  # A
                [0.0, 1.0],  # C
                [1.0, 0.0],  # G
                [1.0, 1.0]], dtype=np.float32)  # T


def fcgr_encode(seq: str, k: int, lut: np.ndarray,
                perm: np.ndarray | None = None,
                pos: bool = False,
                canon: np.ndarray | None = None,
                norm: str = "max") -> np.ndarray:
    """sequence -> FCGR tensor [C, side, side] float32.

    Channel 0 = normalized k-mer frequency. With pos=True, channels 1-2
    hold the mean CGR (x, y) coordinate per cell (PhageCGRNet): within-cell
    offsets summarize the context preceding each k-mer, extending effective
    context beyond k. Empty cells get their cell-center coordinate.

    perm: optional permutation of the 4 corner vertices (label-preserving
    augmentation, Dick & Green 2019) applied as a digit remap.
    canon: optional canonical index map (rc_index_map) making counts
    strand-invariant.
    norm: "max" = log1p then divide by cell max (v0 default);
          "enrich" = log1p(count / expected-under-uniform), insensitive
          to a single repeated k-mer.
    """
    side = 1 << k
    codes = BASE2DIGIT[np.frombuffer(seq.encode("ascii", "replace"),
                                     dtype=np.uint8)]
    if perm is not None:
        codes = np.where(codes >= 0, perm[np.clip(codes, 0, 3)], -1)
    out = np.zeros((3 if pos else 1, side, side), dtype=np.float32)
    if len(codes) < k:
        return out
    freq = np.zeros(side * side, dtype=np.float32)
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    valid = (win >= 0).all(axis=1)
    if not valid.any():
        return out
    w = win[valid]
    # base-4 index, leftmost base most significant
    kidx = (w * (4 ** np.arange(k - 1, -1, -1))).sum(axis=1)
    if canon is not None:
        kidx = canon[kidx]
    cells = lut[kidx]
    np.add.at(freq, cells, 1.0)
    if pos:
        # CGR point at each position: p_i = (p_{i-1} + v_i)/2 solved as a
        # weighted sliding sum truncated at J terms (error <= 2^-J), with
        # the leftover weight going to the (0.5,0.5) initial point.
        J = 25
        weights = 0.5 ** np.arange(1, J + 1)
        n = len(codes)
        resid = 0.5 * (0.5 ** (np.arange(n) + 1))
        px = np.convolve(VTX[codes + 1, 0], weights)[:n] + resid
        py = np.convolve(VTX[codes + 1, 1], weights)[:n] + resid
        ends = np.flatnonzero(valid) + (k - 1)  # position of each k-mer's last base
        sx = np.zeros(side * side, dtype=np.float32)
        sy = np.zeros(side * side, dtype=np.float32)
        np.add.at(sx, cells, px[ends])
        np.add.at(sy, cells, py[ends])
        ys, xs = np.divmod(np.arange(side * side), side)
        mx = np.where(freq > 0, sx / (freq + 1e-9), (xs + 0.5) / side)
        my = np.where(freq > 0, sy / (freq + 1e-9), (ys + 0.5) / side)
        out[1] = mx.reshape(side, side)
        out[2] = my.reshape(side, side)
    if norm == "enrich":
        # log enrichment over the uniform expectation (n_kmers / n_cells)
        img = np.log1p(freq / max(freq.sum(), 1.0) * side * side)
    else:
        img = np.log1p(freq)
        img /= img.max() + 1e-9
    out[0] = img.reshape(side, side)
    return out


def fcgr_encode_ms(seq: str, luts: dict[int, np.ndarray],
                   canons: dict[int, np.ndarray],
                   ks: tuple = MS_KS,
                   perm: np.ndarray | None = None,
                   pos: bool = False) -> np.ndarray:
    """Multi-scale RC-canonical FCGR pyramid -> [len(ks)+2*pos, S, S].

    Each scale is RC-canonicalized and log-enrichment normalized, then
    repeated up to the finest scale's resolution so the channels stack —
    the coarse maps land exactly on quadtree-aligned blocks. pos channels
    are computed at the finest k only.
    """
    k_hi = ks[-1]
    side_hi = 1 << k_hi
    out = np.zeros((len(ks) + (2 if pos else 0), side_hi, side_hi),
                   dtype=np.float32)
    for ci, k in enumerate(ks):
        want_pos = pos and k == k_hi
        img = fcgr_encode(seq, k, luts[k], perm, pos=want_pos,
                          canon=canons[k], norm="enrich")
        rep = side_hi // (1 << k)
        out[ci] = (img[0] if rep == 1 else
                   np.repeat(np.repeat(img[0], rep, axis=0), rep, axis=1))
        if want_pos:
            out[-2], out[-1] = img[1], img[2]
    return out


def seq_encode(seq: str, seq_len: int,
               perm: np.ndarray | None = None) -> np.ndarray:
    """sequence -> one-hot tensor [4, seq_len] float32; N -> all-zero col.

    Perm is a 4-digit base remap (same augmentation semantics as the FCGR
    vertex permute).
    """
    out = np.zeros((4, seq_len), dtype=np.float32)
    codes = BASE2DIGIT[np.frombuffer(seq.encode("ascii", "replace"),
                                     dtype=np.uint8)]
    if perm is not None:
        codes = np.where(codes >= 0, perm[np.clip(codes, 0, 3)], -1)
    n = min(len(codes), seq_len)
    idx = np.nonzero(codes[:n] >= 0)[0]
    out[codes[:n][idx], idx] = 1.0
    return out


class FragmentDataset(Dataset):
    """Line-offset-indexed FASTA -> (FCGR tensor, per-target label ids)."""

    def __init__(self, fasta: Path, labels: dict[str, dict], k: int,
                 lut: np.ndarray, augment: bool = False, pos: bool = False,
                 targets: list[str] | None = None,
                 cont: dict[str, tuple] | None = None,
                 seq: bool = False, seq_len: int = 500,
                 ms_luts: dict[int, np.ndarray] | None = None,
                 ms_canons: dict[int, np.ndarray] | None = None):
        self.fasta = Path(fasta)
        self.k = k
        self.lut = lut
        self.augment = augment
        self.pos = pos
        self.seq = seq
        self.seq_len = seq_len
        self.ms_luts = ms_luts      # set -> multi-scale RC-canonical FCGR
        self.ms_canons = ms_canons
        self.labels = labels  # frag_id -> {target: int}
        self.cont = cont or {}  # frag_id -> (coding_frac, boundary_offset)
        self.ids = list(labels.keys())
        self.targets = targets if targets is not None else RANKS
        idx_path = fasta.with_suffix(fasta.suffix + ".idx.json")
        if idx_path.exists():
            self.offsets = json.loads(idx_path.read_text())
        else:
            self.offsets = self._build_index(idx_path)

    def _build_index(self, idx_path: Path):
        offsets = {}
        with open(self.fasta, "rb") as f:
            pos = f.tell()
            line = f.readline()
            while line:
                if line.startswith(b">"):
                    fid = line[1:].strip().decode()
                    offsets[fid] = pos
                pos = f.tell()
                line = f.readline()
        idx_path.write_text(json.dumps(offsets))
        print(f"indexed {len(offsets)} records -> {idx_path}")
        return offsets

    def _read_seq(self, fid: str) -> str:
        off = self.offsets[fid]
        with open(self.fasta, "rb") as f:
            f.seek(off)
            f.readline()  # header
            chunks = []
            for line in f:
                if line.startswith(b">"):
                    break
                chunks.append(line.strip())
        return b"".join(chunks).decode("ascii")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        perm = np.random.permutation(4) if self.augment else None
        s = self._read_seq(fid)
        img = (seq_encode(s, self.seq_len, perm) if self.seq
               else fcgr_encode_ms(s, self.ms_luts, self.ms_canons,
                                   perm=perm, pos=self.pos)
               if self.ms_luts is not None
               else fcgr_encode(s, self.k, self.lut, perm, self.pos))
        lab = np.array([self.labels[fid].get(t, 0) for t in self.targets],
                       dtype=np.int64)
        c = self.cont.get(fid, (-1.0, -1.0))  # -1 = unlabeled/missing
        return (torch.from_numpy(img), torch.from_numpy(lab),
                torch.tensor(c, dtype=torch.float32))


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, qt=False):
        super().__init__()
        if qt and stride == 2:
            # quadtree-aligned downsample: a 2x2 block is exactly the four
            # extensions of one (k-1)-mer prefix; 3x3s2 convs straddle
            # quadtree boundaries and mix unrelated prefixes.
            self.c1 = nn.Conv2d(cin, cout, 2, 2, 0, bias=False)
            self.short = nn.Conv2d(cin, cout, 2, 2, 0, bias=False)
        else:
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.short = (nn.Conv2d(cin, cout, 1, stride, bias=False)
                          if stride != 1 or cin != cout else nn.Identity())
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        s = self.short(x)
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + s)


class ResNetEncoder(nn.Module):
    def __init__(self, cin: int, width: int, stem_stride: int = 2,
                 qt: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            # qt: 3x3 stride-1 stem preserves the finest FCGR resolution;
            # all halving happens in the quadtree-aligned stage blocks.
            nn.Conv2d(cin, width, 3 if qt else 5,
                      1 if qt else stem_stride,
                      1 if qt else 2, bias=False),
            nn.BatchNorm2d(width), nn.ReLU())
        self.stages = nn.Sequential(
            Block(width, width, qt=qt), Block(width, width, qt=qt),
            Block(width, width * 2, 2, qt),
            Block(width * 2, width * 2, qt=qt),
            Block(width * 2, width * 4, 2, qt),
            Block(width * 4, width * 4, qt=qt),
            Block(width * 4, width * 8, 2, qt),
            Block(width * 8, width * 8, qt=qt),
        )
        self.latent = width * 8

    def forward(self, x):
        return self.stages(self.stem(x)).mean(dim=(2, 3))


class SeqEncoder(nn.Module):
    """1D CNN over one-hot bases -- keeps motif + positional info the FCGR
    composition histogram discards.

    Global mean+max concat pooling: mean carries broad compositional
    signal, max carries presence of discriminative motifs anywhere in the
    window.
    """

    def __init__(self, width: int):
        super().__init__()

        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv1d(i, o, 9, s, 4, bias=False),
                                 nn.BatchNorm1d(o), nn.ReLU())

        self.net = nn.Sequential(
            blk(4, width),
            blk(width, width * 2, 2),
            blk(width * 2, width * 2),
            blk(width * 2, width * 4, 2),
            blk(width * 4, width * 4),
            blk(width * 4, width * 8, 2),
        )
        self.latent = width * 8 * 2

    def forward(self, x):
        h = self.net(x)
        return torch.cat([h.mean(2), h.amax(2)], dim=1)


class ViTEncoder(nn.Module):
    """Patch-embed + TransformerEncoder + CLS latent (PCVR-style ViT on FCGR).

    Global self-attention mixes distant quadtree regions directly; useful
    where discriminative k-mers are sparse and spread across the grid.
    No conv inductive bias -> relies on data scale, which simulated
    fragments provide for free.
    """

    def __init__(self, cin: int, side: int, dim: int = 256, depth: int = 6,
                 heads: int = 8, patch: int = 8):
        super().__init__()
        n = (side // patch) ** 2
        self.patch = nn.Conv2d(cin, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pe = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.latent = dim

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)  # B, N, D
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], 1)
        return self.norm(self.enc(t + self.pe))[:, 0]


def build_tree(meta: dict[str, dict], vocab: dict[str, dict],
               ranks: list[str]) -> dict:
    """Sibling-conditional tree structure for the A1 head.

    For each rank r and shifted class id c (1..C-1):
      parent_of[r][c] = shifted id of the parent-rank class seen on the
      training rows carrying name c; 0 = parent missing/out-of-vocab ->
      the class joins the catch-all root group.
      log_prior[r][c] = log nu(c | parent) with add-one smoothing inside
      each sibling group; applied as a tau*log(nu) logit offset during
      training only (removed at inference for balanced conditionals).

    Parent names are resolved by majority vote over training rows, which
    absorbs the occasional lineage inconsistency in RefSeq labels.
    """
    counts = {r: np.zeros(len(vocab[r]) + 1) for r in ranks}
    parent_votes = {r: defaultdict(Counter) for r in ranks[1:]}
    for m in meta.values():
        for ri, r in enumerate(ranks):
            c = vocab[r].get(m.get(r, ""))
            if c is None:
                continue
            counts[r][c + 1] += 1
            if ri:
                parent_votes[r][m[r]][m.get(ranks[ri - 1], "")] += 1
    parent_of: dict[str, torch.Tensor] = {}
    log_prior: dict[str, torch.Tensor] = {}
    for ri, r in enumerate(ranks):
        n = len(vocab[r]) + 1
        po = np.zeros(n, dtype=np.int64)
        for name, c in vocab[r].items():
            if ri and parent_votes[r].get(name):
                pname = parent_votes[r][name].most_common(1)[0][0]
                po[c + 1] = vocab[ranks[ri - 1]].get(pname, -1) + 1
        parent_of[r] = torch.tensor(po)
        groups = defaultdict(list)
        for c in range(1, n):
            groups[int(po[c])].append(c)
        lp = np.zeros(n)
        for cs in groups.values():
            tot = counts[r][cs].sum() + len(cs)  # add-one smoothing
            for c in cs:
                lp[c] = float(np.log((counts[r][c] + 1) / tot))
        log_prior[r] = torch.tensor(lp, dtype=torch.float32)
    return {"parent_of": parent_of, "log_prior": log_prior}


class TreeRankHead(nn.Module):
    """Cosine-prototype head for one taxonomy rank (A1 arm).

    Logit for class c is a scaled cosine similarity against unit-norm
    prototype w_c, which removes the weight-norm advantage majority
    classes get from plain Linear heads. Class probabilities normalize
    *within sibling groups* (classes sharing the same parent id), not
    across the whole rank, so predictions chain into a coherent tree.
    """

    def __init__(self, latent: int, n_classes: int,
                 parent_of, log_prior):
        super().__init__()
        self.proto = nn.Parameter(
            F.normalize(torch.randn(n_classes - 1, latent), dim=1))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.register_buffer("parent_of",
                             torch.as_tensor(parent_of, dtype=torch.long))
        self.register_buffer("log_prior",
                             torch.as_tensor(log_prior, dtype=torch.float32))

    def cond_logits(self, z, tau: float = 0.0):
        """log P(c | parent(c), z) for classes 1..C-1 as a [B, C-1] tensor."""
        zn = F.normalize(z, dim=1)
        w = F.normalize(self.proto, dim=1)
        # fp32: bf16 under autocast breaks scatter_add_ (exp is autocast to
        # fp32) and is too coarse for the logsumexp below anyway
        lg = (self.scale * (zn @ w.T)).float()
        if tau:
            lg = lg + tau * self.log_prior[1:]
        pidx = self.parent_of[1:].expand(lg.size(0), -1)
        n_groups = int(self.parent_of.max().item()) + 1
        denom = torch.zeros(lg.size(0), n_groups,
                            dtype=lg.dtype, device=lg.device)
        denom.scatter_add_(1, pidx, lg.exp())
        return lg - denom.gather(1, pidx).clamp_min(1e-30).log()

    def marginal_logp(self, z, prev_ext=None):
        """Chained log-marginals for classes 1..C-1.

        prev_ext: [B, C_prev] log-marginals of the previous rank with a
        leading 0 column (log 1) so orphan classes (parent_of==0) inherit
        root mass 1.0.
        """
        cond = self.cond_logits(z)  # inference strips the prior offset
        if prev_ext is None:
            return cond
        pidx = self.parent_of[1:].expand(z.size(0), -1)
        return cond + prev_ext.gather(1, pidx)


class OmniPhyl(nn.Module):
    def __init__(self, n_classes: dict[str, int], width: int = 64,
                 cin: int = 1, backbone: str = "resnet", side: int = 128,
                 targets: list[str] | None = None, dropout: float = 0.0,
                 stem_stride: int = 2, head: str = "flat", tree=None):
        super().__init__()
        if backbone == "vit":
            self.encoder = ViTEncoder(cin, side, dim=width * 4)
        elif backbone == "seq":
            self.encoder = SeqEncoder(width)
        elif backbone == "msfcgr":
            self.encoder = ResNetEncoder(cin, width, qt=True)
        else:
            self.encoder = ResNetEncoder(cin, width, stem_stride)
        self.latent = self.encoder.latent
        self.drop = nn.Dropout(dropout)
        keys = targets if targets is not None else list(n_classes.keys())
        tree_po = (tree or {}).get("parent_of", {}) if head == "tree" else {}
        self.heads = nn.ModuleDict(
            {t: (TreeRankHead(self.latent, n_classes[t],
                              tree["parent_of"][t], tree["log_prior"][t])
                 if t in tree_po else nn.Linear(self.latent, n_classes[t]))
             for t in keys})

    def cond_logps(self, z, tau: float = 0.0):
        """Per-rank sibling-conditional log-probs (tree heads only)."""
        return {t: h.cond_logits(z, tau) for t, h in self.heads.items()
                if isinstance(h, TreeRankHead)}

    def forward(self, x):
        z = self.drop(self.encoder(x))
        logits = {}
        prev_ext = None  # [B, C_prev] chained log-marginals, col 0 = log 1
        for t, h in self.heads.items():
            if isinstance(h, TreeRankHead):
                logm = h.marginal_logp(z, prev_ext)
                prev_ext = torch.cat(
                    [logm.new_zeros(logm.size(0), 1), logm], 1)
                # col 0 (unknown class) gets a large negative so argmax
                # never lands there; cols 1.. are log-marginals
                logits[t] = torch.cat(
                    [logm.new_full((logm.size(0), 1), -1e4), logm], 1)
            else:
                logits[t] = h(z)
        return z, logits


def load_labels(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["frag_id"]] = r
    return rows


def encode_cont_labels(rows: dict[str, dict]) -> dict[str, tuple]:
    """frag_id -> (coding_frac, boundary_offset), -1.0 when absent.

    Boundary flag is NOT stored: it is derived at loss time as
    0 < coding_frac < 1, matching the candidacy semantics in
    docs/annotation-cascade.md. Missing/non-numeric/out-of-range
    values become -1.0 and are masked out of the loss.
    """
    out = {}
    for fid, m in rows.items():
        vals = []
        for col in ("coding_frac", "boundary_offset"):
            try:
                v = float(m.get(col, "") or -1.0)
            except (TypeError, ValueError):
                v = -1.0
            vals.append(v if 0 <= v <= 1 else -1.0)
        out[fid] = tuple(vals)
    return out


def build_split(meta: dict[str, dict], val_frac: float, genus_holdout: float,
                seed: int):
    rng = random.Random(seed)
    # genomes whose requested rank was unresolved in taxonomy are
    # test-only: never trained on, always held out (OOD probe set)
    forced_val = {m["accession"] for m in meta.values()
                  if m.get("rank_unresolved") == "1"}
    # genbank-sourced genomes are train-only: their taxonomy is
    # submitter-declared (no RefSeq QC), so keeping the held-out set
    # purely RefSeq keeps the benchmark clean when measuring whether
    # genbank augmentation helps underrepresented groups
    forced_train = {m["accession"] for m in meta.values()
                    if m.get("seq_source") == "genbank"}
    forced_val -= forced_train
    gb_genera = {m["genus"] for m in meta.values()
                 if m["accession"] in forced_train}
    accs = sorted({m["accession"] for m in meta.values()}
                  - forced_val - forced_train)
    rng.shuffle(accs)
    n_val = max(1, int(len(accs) * val_frac))
    val_accs = set(accs[:n_val]) | forced_val

    holdout_genera = set()
    if genus_holdout > 0:
        # genera containing any train-only (genbank) member are excluded:
        # holding one out would drag its genbank members into val anyway
        train_genera = sorted({m["genus"] for m in meta.values()
                               if m["genus"] and m["genus"] not in gb_genera
                               and m["accession"] not in val_accs})
        holdout_genera = set(rng.sample(
            train_genera, max(1, int(len(train_genera) * genus_holdout))))
    return val_accs, holdout_genera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of genomes held out for val")
    ap.add_argument("--holdout-genus-frac", type=float, default=0.0,
                    help="fraction of train genera moved to val entirely "
                         "(leave-clade-out probe)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--vertex-permute-augment", action="store_true",
                    help="randomly permute CGR corner vertices per fragment "
                         "(label-preserving augmentation)")
    ap.add_argument("--pos-channels", action="store_true",
                    help="add mean-CGR-(x,y) channels per cell "
                         "(PhageCGRNet positional features)")
    ap.add_argument("--backbone", choices=["resnet", "vit", "seq", "msfcgr"],
                    default="resnet",
                    help="encoder: resnet=FCGR CNN (default), vit=FCGR "
                         "global attention, seq=1D CNN over one-hot bases "
                         "(keeps motif/positional info FCGR discards), "
                         "msfcgr=RC-canonical k4/5/6 pyramid + "
                         "quadtree-aligned 2x2 resnet (A2)")
    ap.add_argument("--stem-stride", type=int, default=2,
                    help="resnet stem conv stride; 1 keeps full FCGR "
                         "resolution (stride-2 merges each k-mer's "
                         "terminal-base digit on both grid axes)")
    ap.add_argument("--seq-len", type=int, default=500,
                    help="pad/truncate fragments to this length for "
                         "--backbone seq")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout on the encoder latent before heads")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CE label smoothing; keeps softmax confidence "
                         "honest for the abstention metric")
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="AdamW weight decay (default 1e-4)")
    ap.add_argument("--head", choices=["flat", "tree"], default="flat",
                    help="tree = cosine-prototype heads with sibling-"
                         "conditional softmax + prior logit adjustment")
    ap.add_argument("--prior-tau", type=float, default=1.0,
                    help="tau for the tau*log(nu) sibling-prior offset "
                         "(tree head, applied during training only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = load_labels(args.labels)
    val_accs, holdout_genera = build_split(
        meta, args.val_frac, args.holdout_genus_frac, args.seed)
    print(f"{len(val_accs)} val genomes, "
          f"{len(holdout_genera)} holdout genera")

    def is_val(m):
        return (m["accession"] in val_accs
                or m["genus"] in holdout_genera)

    train_meta = {k: v for k, v in meta.items() if not is_val(v)}
    val_meta = {k: v for k, v in meta.items() if is_val(v)}
    print(f"frags: {len(train_meta)} train / {len(val_meta)} val")

    # vocab from TRAIN rows only; val rows outside vocab -> -1 (unknown)
    vocab = {r: {} for r in RANKS}
    for m in train_meta.values():
        for r in RANKS:
            name = m.get(r, "")
            if name and name not in vocab[r]:
                vocab[r][name] = len(vocab[r])
    cls_targets = RANKS
    all_targets = cls_targets + CONT_TARGETS
    n_classes = {r: len(vocab[r]) + 1 for r in cls_targets}  # +1 = unknown
    n_classes.update({t: 1 for t in CONT_TARGETS})  # scalar heads
    tree = (build_tree(train_meta, vocab, cls_targets)
            if args.head == "tree" else None)

    def encode_labels(rows):
        out = {}
        for fid, m in rows.items():
            rec = {r: vocab[r].get(m.get(r, ""), -1)
                   if m.get(r, "") else -1 for r in cls_targets}
            # +1 offset unknown->0: shift all ids by 1 so 0 = unknown class
            out[fid] = {r: (v + 1 if v >= 0 else 0) for r, v in rec.items()}
        return out

    lut = build_cell_lut(args.k)
    cont_train = encode_cont_labels(train_meta)
    cont_val = encode_cont_labels(val_meta)

    use_seq = args.backbone == "seq"
    ms_maps = ({k: build_cell_lut(k) for k in MS_KS},
               {k: rc_index_map(k) for k in MS_KS}) \
        if args.backbone == "msfcgr" else (None, None)
    train_ds = FragmentDataset(args.fragments, encode_labels(train_meta),
                               args.k, lut,
                               augment=args.vertex_permute_augment,
                               pos=args.pos_channels,
                               targets=cls_targets, cont=cont_train,
                               seq=use_seq, seq_len=args.seq_len,
                               ms_luts=ms_maps[0], ms_canons=ms_maps[1])
    val_ds = FragmentDataset(args.fragments, encode_labels(val_meta),
                             args.k, lut, pos=args.pos_channels,
                             targets=cls_targets, cont=cont_val,
                             seq=use_seq, seq_len=args.seq_len,
                             ms_luts=ms_maps[0], ms_canons=ms_maps[1])
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          persistent_workers=args.num_workers > 0)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    base_ch = len(MS_KS) if args.backbone == "msfcgr" else 1
    model = OmniPhyl(n_classes, args.width,
                     cin=base_ch + 2 if args.pos_channels else base_ch,
                     backbone=args.backbone, side=1 << args.k,
                     targets=all_targets, dropout=args.dropout,
                     stem_stride=args.stem_stride,
                     head=args.head, tree=tree).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    steps = len(train_ld) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    amp = device == "cuda"

    def run(loader, train=True):
        model.train(train)
        tot_loss, tot_n = 0.0, 0
        correct = defaultdict(int)
        total = defaultdict(int)
        frac_err = frac_n = bnd_ok = bnd_n = off_n = 0
        off_err = 0.0
        for x, y, yc in loader:
            x, y, yc = x.to(device), y.to(device), yc.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp):
                z, logits = model(x)
                if args.head == "tree":
                    # sibling-conditional NLL with prior offset; the
                    # conditional softmax is the training target while
                    # accuracy below still reads the chained marginals
                    clp = model.cond_logps(z, tau=args.prior_tau)
                    terms = []
                    for i, t in enumerate(cls_targets):
                        m = y[:, i] != 0
                        if not m.any():
                            continue
                        ce = -clp[t][m].gather(
                            1, (y[m, i] - 1)[:, None]).squeeze(1)
                        if args.label_smoothing:
                            ce = ((1 - args.label_smoothing) * ce
                                  - args.label_smoothing
                                  * clp[t][m].mean(1))
                        terms.append(ce.mean())
                else:
                    # skip heads whose batch labels are all masked —
                    # cross_entropy(ignore_index=0) returns NaN there (val
                    # batches whose rank labels are entirely out-of-vocab)
                    terms = [F.cross_entropy(logits[t], y[:, i],
                                             ignore_index=0,
                                             label_smoothing=args.label_smoothing)
                             for i, t in enumerate(cls_targets)
                             if (y[:, i] != 0).any()]
                fm = yc[:, 0] >= 0          # coding_frac label present
                btgt = ((yc[:, 0] > 0) & (yc[:, 0] < 1)).float()
                if fm.any():
                    terms.append(F.binary_cross_entropy_with_logits(
                        logits["coding_frac"][:, 0][fm], yc[fm, 0]))
                    terms.append(F.binary_cross_entropy_with_logits(
                        logits["boundary"][:, 0][fm], btgt[fm]))
                om = yc[:, 1] >= 0          # boundary_offset label present
                if om.any():
                    terms.append(F.mse_loss(torch.sigmoid(
                        logits["offset"][:, 0][om]), yc[om, 1]))
                if not terms:
                    continue
                loss = torch.stack(terms).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            tot_loss += loss.item() * len(x)
            tot_n += len(x)
            for i, t in enumerate(cls_targets):
                m = y[:, i] != 0  # unknown/NA labels excluded from accuracy
                total[t] += int(m.sum())
                correct[t] += int((logits[t].argmax(1)[m]
                                   == y[:, i][m]).sum())
            if fm.any():
                p_frac = torch.sigmoid(logits["coding_frac"][:, 0][fm])
                frac_err += float((p_frac - yc[fm, 0]).detach().abs().sum())
                frac_n += int(fm.sum())
                p_bnd = torch.sigmoid(logits["boundary"][:, 0][fm]) > 0.5
                bnd_ok += int((p_bnd == btgt[fm].bool()).sum())
                bnd_n += int(fm.sum())
            if om.any():
                p_off = torch.sigmoid(logits["offset"][:, 0][om])
                off_err += float((p_off - yc[om, 1]).detach().abs().sum())
                off_n += int(om.sum())
        accs = {t: correct[t] / max(total[t], 1) for t in cls_targets}
        extras = {
            "frac_mae": frac_err / max(frac_n, 1),
            "boundary_acc": bnd_ok / max(bnd_n, 1),
            "offset_mae": off_err / max(off_n, 1),
        }
        return tot_loss / max(tot_n, 1), accs, extras

    best = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run(train_ld, train=True)
        va_loss, accs, extras = run(val_ld, train=False)
        msg = "  ".join(f"{t}:{a:.3f}" for t, a in accs.items())
        xmsg = "  ".join(f"{t}:{v:.3f}" for t, v in extras.items())
        print(f"ep {ep:03d} train {tr_loss:.4f} val {va_loss:.4f} | "
              f"{msg} | {xmsg} | {time.time()-t0:.0f}s")
        mean_acc = float(np.mean(list(accs.values())))
        if mean_acc > best:
            best = mean_acc
            ckpt = {"model": model.state_dict(), "vocab": vocab,
                    "k": args.k, "targets": all_targets,
                    "args": {k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()}}
            if tree is not None:
                ckpt["tree"] = {
                    "parent_of": {r: tree["parent_of"][r].tolist()
                                  for r in cls_targets},
                    "log_prior": {r: tree["log_prior"][r].tolist()
                                  for r in cls_targets}}
            torch.save(ckpt, args.out / "best.pt")
    (args.out / "vocab.json").write_text(json.dumps(vocab, indent=1))
    print(f"done. best mean val acc {best:.3f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
