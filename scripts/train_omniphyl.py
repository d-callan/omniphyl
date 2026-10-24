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
import zlib
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


# ------------------------------------------------------- aux branches ---

HASH_K = 11                    # canonical marker-kmer length (A3)
HASH_BUCKETS = 1 << 16         # EmbeddingBag rows
HASH_PAD = HASH_BUCKETS        # padding_idx for variable-length bags
HASH_LEN = 512                 # padded bag size (500bp -> ~490 11-mers)
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard AAs; index 20 = stop
AA_ALPHA = len(AA_LETTERS) + 1
AA_DIM = AA_ALPHA ** 3         # 9261-dim tripeptide spectrum
AA_EXTRA_DIM = 18              # gated mode: 6 frames x (stop density,
                               #   longest ORF fraction, TGA share of stops)


def _build_aa_lut() -> np.ndarray:
    """codon base-4 index (A,C,G,T digit order) -> AA letter idx; 20=stop."""
    # standard genetic code, rows = first base, cols = second*4+third
    # in T,C,A,G order
    rows = {"T": "FFLLSSSSYY**CC*W",
            "C": "LLLLPPPPHHQQRRRR",
            "A": "IIIMTTTTNNKKSSRR",
            "G": "VVVVAAAADDEEGGGG"}
    tcag = {0: 2, 1: 1, 2: 3, 3: 0}      # our digit -> T,C,A,G position
    first = {0: "A", 1: "C", 2: "G", 3: "T"}
    lut = np.full(64, -1, dtype=np.int64)
    for d0 in range(4):
        row = rows[first[d0]]
        for d1 in range(4):
            for d2 in range(4):
                aa = row[tcag[d1] * 4 + tcag[d2]]
                lut[d0 * 16 + d1 * 4 + d2] = (
                    AA_ALPHA - 1 if aa == "*" else AA_LETTERS.index(aa))
    return lut


_AA_LUT = _build_aa_lut()


def hash_encode(seq: str, canon: np.ndarray) -> np.ndarray:
    """sequence -> padded hash-bucket ids [HASH_LEN] int64 (A3 branch).

    Canonical 11-mers are multiplicatively hashed into HASH_BUCKETS rows;
    HASH_PAD fills the tail. Strand-invariant via the rc canon map.
    """
    out = np.full(HASH_LEN, HASH_PAD, dtype=np.int64)
    codes = BASE2DIGIT[np.frombuffer(seq.encode("ascii", "replace"),
                                     dtype=np.uint8)]
    if len(codes) < HASH_K:
        return out
    win = np.lib.stride_tricks.sliding_window_view(codes, HASH_K)
    valid = (win >= 0).all(axis=1)
    if not valid.any():
        return out
    w = win[valid]
    kidx = (w * (4 ** np.arange(HASH_K - 1, -1, -1, dtype=np.int64))).sum(1)
    cidx = canon[kidx]
    buckets = (cidx.astype(np.uint64) * np.uint64(2654435761)) \
        % np.uint64(HASH_BUCKETS)
    n = min(len(buckets), HASH_LEN)
    out[:n] = buckets[:n].astype(np.int64)
    return out


def aa_encode(seq: str, extra: bool = False) -> np.ndarray:
    """sequence -> 6-frame AA 3-mer spectrum [AA_DIM] float32 (A3 branch).

    Translates all 3 frames on both strands (stop = letter 20), counts
    tripeptides, log-enrichment normalized. Codons touching an N are
    dropped. Synonymous 3rd-position drift cancels inside each tripeptide
    bin, preserving signal where nucleotide k-mers saturate.

    extra=True appends AA_EXTRA_DIM per-frame summaries used by the
    gated fusion gate: for each of the 6 frames in fixed order
    (F0,F1,F2,R0,R1,R2) -> [stop_density, longest_orf_frac, tga_ratio].
    tga_ratio (share of TGA among stops) separates Code-4 Mycoplasma
    (TGA->Trp) from standard-code genomes.
    """
    codes = BASE2DIGIT[np.frombuffer(seq.encode("ascii", "replace"),
                                     dtype=np.uint8)]
    vec = np.zeros(AA_DIM, dtype=np.float32)
    feats: list[float] = []
    for c in (codes, 3 - codes[::-1]):      # forward + reverse-complement
        for off in range(3):
            n = (len(c) - off) // 3 * 3
            if n <= 0:
                if extra:
                    feats.extend((0.0, 0.0, 0.0))
                continue
            trip = c[off:off + n].reshape(-1, 3)
            ok = (trip >= 0).all(axis=1)
            cidx = np.clip(trip[:, 0] * 16 + trip[:, 1] * 4 + trip[:, 2],
                           0, 63)
            aa = np.where(ok, _AA_LUT[cidx], -1)
            aa = aa[aa >= 0]
            if extra:
                n_cod = len(aa)
                is_stop = aa == (AA_ALPHA - 1)
                n_st = int(is_stop.sum())
                sp = np.where(is_stop)[0]
                runs = np.diff(np.concatenate(([-1], sp, [n_cod]))) - 1
                # TGA codon id = 3*16 + 2*4 + 0 = 56 (stop #20)
                tga = int((cidx[ok][is_stop] == 56).sum())
                feats.extend((n_st / max(n_cod, 1),
                              float(runs.max()) / max(n_cod, 1)
                              if len(runs) else 0.0,
                              tga / max(n_st, 1)))
            if len(aa) < 3:
                continue
            w = np.lib.stride_tricks.sliding_window_view(aa, 3)
            idx = (w * np.array([AA_ALPHA ** 2, AA_ALPHA, 1])).sum(axis=1)
            np.add.at(vec, idx, 1.0)
    spec = np.log1p(vec / max(vec.sum(), 1.0) * AA_DIM)
    if extra:
        spec = np.concatenate(
            [spec, np.asarray(feats, dtype=np.float32)])
    return spec


AA_FRAME_DIM = 144    # 6 frames x (21 AA/stop freqs + 3 summaries)


def aa_frame_encode(seq: str) -> np.ndarray:
    """sequence -> per-frame AA summary [6, 24] -> [144] float32 (E4).

    Frame order matches aa_encode (F0,F1,F2,R0,R1,R2). Each frame row is
    21 single-AA + stop frequencies, then stop_density, longest_orf_frac,
    and codon third-position GC — the minimal per-frame signature a
    linear head needs for active-frame classification.
    """
    codes = BASE2DIGIT[np.frombuffer(seq.encode("ascii", "replace"),
                                     dtype=np.uint8)]
    out = np.zeros((6, 24), dtype=np.float32)
    fi = 0
    for c in (codes, 3 - codes[::-1]):
        for off in range(3):
            n = (len(c) - off) // 3 * 3
            if n > 0:
                trip = c[off:off + n].reshape(-1, 3)
                ok = (trip >= 0).all(axis=1)
                cidx = np.clip(trip[:, 0] * 16 + trip[:, 1] * 4
                               + trip[:, 2], 0, 63)
                aa = np.where(ok, _AA_LUT[cidx], -1)
                aa = aa[aa >= 0]
                n_cod = len(aa)
                if n_cod:
                    out[fi, :AA_ALPHA] = np.bincount(
                        aa, minlength=AA_ALPHA)[:AA_ALPHA] / n_cod
                    is_stop = aa == (AA_ALPHA - 1)
                    n_st = int(is_stop.sum())
                    sp = np.where(is_stop)[0]
                    runs = np.diff(
                        np.concatenate(([-1], sp, [n_cod]))) - 1
                    out[fi, 21] = n_st / n_cod
                    out[fi, 22] = (float(runs.max()) / n_cod
                                   if len(runs) else 0.0)
                    p3 = trip[ok, 2]             # 3rd codon position
                    out[fi, 23] = float((p3 >= 1).mean())  # C|G share
            fi += 1
    return out.reshape(-1)


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


def jitter_seq(s: str, rate: float, seed: int) -> str:
    """Apply Poisson(rate)-distributed point substitutions per position.

    Deterministic in ``seed`` so a jitter sweep is reproducible and a
    given fragment mutates identically across workers. Used at eval to
    probe whether predictions rely on exact long k-mers (instance
    features) or compositional spectrum: one SNP destroys ~k overlapping
    k-mers but leaves GC/codon/dinucleotide statistics nearly unchanged,
    so the accuracy-vs-rate curve traces instance-feature reliance."""
    a = np.frombuffer(s.upper().encode("ascii"), dtype=np.uint8).copy()
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate * len(a))
    if n == 0:
        return s
    pos = rng.choice(len(a), size=min(int(n), len(a)), replace=False)
    pos = pos[np.isin(a[pos], np.frombuffer(b"ACGT", dtype=np.uint8))]
    if len(pos) == 0:
        return s
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    alt = bases[rng.integers(0, 4, len(pos))]
    neq = alt != a[pos]
    while not neq.all():
        redo = np.nonzero(~neq)[0]
        alt[redo] = bases[rng.integers(0, 4, len(redo))]
        neq = alt != a[pos]
    a[pos] = alt
    return a.tobytes().decode("ascii")


PE_LEN = 150    # unmerged paired-end mate length for var-len augment


def varlen_crop(s: str, min_len: int) -> str:
    """Stochastic read-length crop (E2): 70% uniform contiguous
    sub-window in [min_len, |s|]; 30% unmerged paired-end
    (PE_LEN prefix + 'N' + PE_LEN suffix). The 'N' junction breaks
    cross-insert k-mers and codons: both fcgr_encode and aa_encode
    drop windows/codons containing it.
    """
    if len(s) <= min_len:
        return s
    if np.random.random() < 0.3 and len(s) >= 2 * PE_LEN:
        return s[:PE_LEN] + "N" + s[-PE_LEN:]
    L = int(np.random.randint(min_len, len(s) + 1))
    o = int(np.random.randint(0, len(s) - L + 1))
    return s[o:o + L]


def eval_crop_seq(s: str, crop: int = 0, paired: int = 0) -> str:
    """Deterministic eval-time read simulation (E2):
    paired>0 -> prefix/suffix mates of `paired` bases joined by 'N';
    crop>0 -> truncate to the first `crop` bases (150/250bp reads).
    Seqs already <= the target length pass through unchanged.
    """
    if paired and len(s) > 2 * paired:
        return s[:paired] + "N" + s[-paired:]
    if crop and len(s) > crop:
        return s[:crop]
    return s


class FragmentDataset(Dataset):
    """Line-offset-indexed FASTA -> (FCGR tensor, per-target label ids)."""

    def __init__(self, fasta: Path, labels: dict[str, dict], k: int,
                 lut: np.ndarray, augment: bool = False, pos: bool = False,
                 targets: list[str] | None = None,
                 cont: dict[str, tuple] | None = None,
                 seq: bool = False, seq_len: int = 500,
                 ms_luts: dict[int, np.ndarray] | None = None,
                 ms_canons: dict[int, np.ndarray] | None = None,
                 aux: set | None = None,
                 hash_canon: np.ndarray | None = None,
                 jitter: float = 0.0,
                 aa_extra: bool = False,
                 var_len_prob: float = 0.0,
                 min_train_len: int = 500,
                 eval_crop: int = 0,
                 eval_paired: int = 0,
                 aa_frame: bool = False,
                 annot: dict | None = None):
        self.fasta = Path(fasta)
        self.k = k
        self.lut = lut
        self.augment = augment
        self.pos = pos
        self.seq = seq
        self.seq_len = seq_len
        self.ms_luts = ms_luts      # set -> multi-scale RC-canonical FCGR
        self.ms_canons = ms_canons
        self.aux = aux or set()   # {"hash","aa"} extra input branches (A3)
        self.hash_canon = hash_canon
        self.jitter = jitter      # eval-only SNP rate; train keeps 0
        self.aa_extra = aa_extra  # gated aa mode: append frame/stop feats
        self.var_len_prob = var_len_prob   # E2: stochastic crop prob
        self.min_train_len = min_train_len
        self.eval_crop = eval_crop         # E2: deterministic read crop
        self.eval_paired = eval_paired     # E2: paired-end mate length
        self.aa_frame = aa_frame   # E4: emit per-frame aa summary vec
        self.annot = annot         # frag_id -> annot label tuple or None
        self._annot_default = (
            (-1.0,) * (ANNOT_BASE + annot["_k"]
                       + (1 if annot["_k"] else 0))
            if annot is not None else ())
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
        try:
            idx_path.write_text(json.dumps(offsets))
            print(f"indexed {len(offsets)} records -> {idx_path}")
        except OSError:
            # fasta dir may be read-only (mounted corpus); the in-memory
            # index is still valid, we just can't cache it to disk.
            print(f"indexed {len(offsets)} records (cache write skipped: "
                  f"{idx_path} not writable)")
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
        s = self._read_seq(fid)
        perm = np.random.permutation(4) if self.augment else None
        if perm is not None:
            # Permute at the SEQUENCE level so every encoder — FCGR
            # image, hash k-mers, AA histogram, frame summary — sees
            # the same permuted molecule. Digit-remap inside
            # fcgr_encode alone would leave the aux branches encoding
            # the unpermuted sequence (mismatched input branches).
            tbl = str.maketrans("ACGTacgt",
                                "".join("ACGT"[d] for d in perm) * 2)
            s = s.translate(tbl)
            perm = None
        if self.jitter:
            s = jitter_seq(s, self.jitter,
                           zlib.crc32(fid.encode("ascii")))
        cropped = False
        if self.var_len_prob and np.random.random() < self.var_len_prob:
            n0 = len(s)
            s = varlen_crop(s, self.min_train_len)
            # varlen_crop passes through unchanged when len(s) <= min_len
            cropped = len(s) != n0
        if self.eval_crop or self.eval_paired:
            s = eval_crop_seq(s, self.eval_crop, self.eval_paired)
        img = (seq_encode(s, self.seq_len, perm) if self.seq
               else fcgr_encode_ms(s, self.ms_luts, self.ms_canons,
                                   perm=perm, pos=self.pos)
               if self.ms_luts is not None
               else fcgr_encode(s, self.k, self.lut, perm, self.pos))
        lab = np.array([self.labels[fid].get(t, 0) for t in self.targets],
                       dtype=np.int64)
        c = self.cont.get(fid, (-1.0, -1.0))  # -1 = unlabeled/missing
        x = torch.from_numpy(img)
        if self.aux:
            x = {"x": x}
            if "hash" in self.aux:
                x["hash"] = torch.from_numpy(hash_encode(s, self.hash_canon))
            if "aa" in self.aux:
                x["aa"] = torch.from_numpy(aa_encode(s, self.aa_extra))
        if self.aa_frame:
            if not isinstance(x, dict):
                x = {"x": x}
            x["aa_frame"] = torch.from_numpy(aa_frame_encode(s))
        if self.annot is not None:
            c = c + self.annot.get(fid, self._annot_default)
        if cropped:
            # The crop shifts reading frame by (offset % 3), moves or
            # drops structural transitions, and the coding fraction was
            # measured on the full window — every window-derived target
            # (coding_frac, boundary, offset, region, frame, flags,
            # domains) is wrong for the cropped read. Mask them all to
            # the -1 unlabeled sentinel the losses already respect.
            c = (-1.0,) * len(c)
        return (x, torch.from_numpy(lab),
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


def build_flat_priors(meta: dict[str, dict], vocab: dict[str, dict],
                      ranks: list[str]) -> dict[str, torch.Tensor]:
    """Unconditional per-rank class log-priors (flat-head logit adjustment).

    lp[r][c] = log nu(c) with add-one smoothing over TRAIN rows only.
    lp[r][0] is pinned to a large negative: class 0 is the unknown mask,
    never a learned prediction, so it must never be boosted.
    During training, adding tau * lp to the logits before cross-entropy
    (Menon et al. 2021 logit-adjusted loss) makes the learned raw logit
    approximate a prior-balanced posterior, undoing majority-class
    collapse without touching the inference head.
    """
    counts = {r: np.zeros(len(vocab[r]) + 1) for r in ranks}
    for m in meta.values():
        for r in ranks:
            c = vocab[r].get(m.get(r, ""))
            if c is not None:
                counts[r][c + 1] += 1
    out: dict[str, torch.Tensor] = {}
    for r in ranks:
        n = len(vocab[r]) + 1
        lp = np.full(n, -20.0, dtype=np.float32)
        tot = counts[r][1:].sum() + (n - 1)
        for c in range(1, n):
            lp[c] = float(np.log((counts[r][c] + 1.0) / tot))
        out[r] = torch.tensor(lp, dtype=torch.float32)
    return out


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

    def marginal_logp(self, z, prev_ext=None, tau: float = 0.0):
        """Chained log-marginals for classes 1..C-1.

        prev_ext: [B, C_prev] log-marginals of the previous rank with a
        leading 0 column (log 1) so orphan classes (parent_of==0) inherit
        root mass 1.0. tau reinstates the training-time prior offset for
        MAP decisions; tau=0 gives prior-balanced conditionals instead.
        """
        cond = self.cond_logits(z, tau)
        if prev_ext is None:
            return cond
        pidx = self.parent_of[1:].expand(z.size(0), -1)
        return cond + prev_ext.gather(1, pidx)


class OmniPhyl(nn.Module):
    def __init__(self, n_classes: dict[str, int], width: int = 64,
                 cin: int = 1, backbone: str = "resnet", side: int = 128,
                 targets: list[str] | None = None, dropout: float = 0.0,
                 stem_stride: int = 2, head: str = "flat", tree=None,
                 aux: set | None = None,
                 aa_mode: str = "plain", aa_drop: float = 0.0,
                 region_head: bool = False, struct_head: bool = False,
                 domain_k: int = 0):
        super().__init__()
        if backbone == "vit":
            self.encoder = ViTEncoder(cin, side, dim=width * 4)
        elif backbone == "seq":
            self.encoder = SeqEncoder(width)
        elif backbone == "msfcgr":
            self.encoder = ResNetEncoder(cin, width, qt=True)
        else:
            self.encoder = ResNetEncoder(cin, width, stem_stride)
        self.aux = aux or set()
        aux_dim = 128
        self.hash_bag = (nn.EmbeddingBag(HASH_BUCKETS + 1, aux_dim,
                                         mode="mean",
                                         padding_idx=HASH_PAD)
                         if "hash" in self.aux else None)
        self.aa_mode = aa_mode if "aa" in self.aux else "plain"
        self.aa_drop = aa_drop
        aa_in = AA_DIM + (AA_EXTRA_DIM if self.aa_mode == "gated" else 0)
        if "aa" not in self.aux:
            self.aa_proj = None
        elif self.aa_mode == "bottleneck":
            # force aa evidence through a 64-d bottleneck so it cannot
            # memorise genome identity and starve msfcgr (A3f pathology)
            self.aa_proj = nn.Sequential(nn.Linear(aa_in, 64),
                                         nn.LayerNorm(64), nn.GELU(),
                                         nn.Linear(64, aux_dim),
                                         nn.LayerNorm(aux_dim))
        else:
            self.aa_proj = nn.Sequential(nn.Linear(aa_in, aux_dim),
                                         nn.LayerNorm(aux_dim), nn.ReLU())
        # gated fusion: scalar gate per aux dim driven by z_dna + a
        # detached coding-fraction estimate + the 18-d frame summary.
        # coding_frac is predicted from z_dna alone so coding detection
        # never co-depends on the aa branch it gates.
        gated = self.aa_mode == "gated"
        self.coding_head_dna = (nn.Linear(self.encoder.latent, 1)
                                if gated else None)
        self.aa_gate = (nn.Linear(
            self.encoder.latent + 1 + AA_EXTRA_DIM, aux_dim)
            if gated else None)
        self.latent = self.encoder.latent + aux_dim * len(self.aux)
        self.infer_tau = 0.0   # prior offset at inference (MAP if >0)
        self.drop = nn.Dropout(dropout)
        keys = targets if targets is not None else list(n_classes.keys())
        if gated:              # coding_frac comes from z_dna, not z
            keys = [t for t in keys if t != "coding_frac"]
        tree_po = (tree or {}).get("parent_of", {}) if head == "tree" else {}
        self.heads = nn.ModuleDict(
            {t: (TreeRankHead(self.latent, n_classes[t],
                              tree["parent_of"][t], tree["log_prior"][t])
                 if t in tree_po else nn.Linear(self.latent, n_classes[t]))
             for t in keys})
        # P3 annotation heads (E3 region composition / E5 domain multi-
        # label) are plain linears on z; they ride the heads dict since
        # their input is the fused latent. The struct pair (E4) needs
        # cat([z, aa_frame]) so it lives outside heads and is applied
        # explicitly in forward.
        if region_head:
            self.heads["region_fracs"] = nn.Linear(self.latent, 4)
        if domain_k:
            self.heads["domains"] = nn.Linear(self.latent, domain_k)
        self.struct_head = struct_head
        if struct_head:
            self.head_frame = nn.Linear(self.latent + AA_FRAME_DIM, 8)
            self.head_flags = nn.Linear(self.latent + AA_FRAME_DIM, 4)

    def cond_logps(self, z, tau: float = 0.0):
        """Per-rank sibling-conditional log-probs (tree heads only)."""
        return {t: h.cond_logits(z, tau) for t, h in self.heads.items()
                if isinstance(h, TreeRankHead)}

    def forward(self, x):
        lcf = None
        if isinstance(x, dict):
            z_dna = self.encoder(x["x"])
            parts = [z_dna]
            if self.hash_bag is not None:
                h = self.hash_bag(x["hash"])
                parts.append(F.layer_norm(h, (h.size(1),)))
            if self.coding_head_dna is not None:
                lcf = self.coding_head_dna(z_dna)
            if self.aa_proj is not None:
                za = self.aa_proj(x["aa"])
                if self.aa_gate is not None:
                    gi = torch.cat(
                        [z_dna, lcf.detach(),
                         x["aa"][:, -AA_EXTRA_DIM:]], 1)
                    g = torch.sigmoid(self.aa_gate(gi))
                    if self.training and self.aa_drop > 0:
                        keep = torch.bernoulli(torch.full(
                            (za.size(0), 1), 1.0 - self.aa_drop,
                            device=za.device))
                        g = g * keep
                    za = g * za
                parts.append(za)
            z = self.drop(torch.cat(parts, 1))
        else:
            z = self.drop(self.encoder(x))
        logits = {}
        prev_ext = None  # [B, C_prev] chained log-marginals, col 0 = log 1
        for t, h in self.heads.items():
            if isinstance(h, TreeRankHead):
                logm = h.marginal_logp(z, prev_ext, self.infer_tau)
                prev_ext = torch.cat(
                    [logm.new_zeros(logm.size(0), 1), logm], 1)
                # col 0 (unknown class) gets a large negative so argmax
                # never lands there; cols 1.. are log-marginals
                logits[t] = torch.cat(
                    [logm.new_full((logm.size(0), 1), -1e4), logm], 1)
            else:
                logits[t] = h(z)
        if lcf is not None:
            logits["coding_frac"] = lcf
        if self.struct_head:
            xf = (x["aa_frame"] if isinstance(x, dict)
                  and "aa_frame" in x
                  else torch.zeros(z.size(0), AA_FRAME_DIM,
                                   device=z.device))
            zs = torch.cat([z, xf], 1)
            logits["active_frame"] = self.head_frame(zs)
            logits["struct_flags"] = self.head_flags(zs)
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


ANNOT_BASE = 9   # region_fracs(4) + active_frame(1) + struct_flags(4)


def encode_annot_labels(rows: dict[str, dict],
                        domain_vocab: dict | None = None
                        ) -> dict[str, tuple]:
    """frag_id -> (r_coding, r_intron, r_rna, r_intergenic,
                   active_frame, f_start, f_stop, f_donor, f_acceptor,
                   dom_0..K-1, dom_mask)  for the P3 annotation heads.

    -1.0 marks unlabeled dims (masked out of each loss).  domain_ids
    uses a trailing mask column because the column distinguishes three
    states: 'NA'/absent = no annotation (mask off), '' = annotated, no
    domains (all-negative label), 'IPR..,PF..' = multi-hot positive.
    out["_k"] carries K so callers can build a matching default row.
    """
    domain_vocab = domain_vocab or {}
    K = len(domain_vocab)
    out: dict[str, tuple] = {}
    for fid, m in rows.items():
        vals, ok = [], True
        for col in ("coding_frac", "intron_frac", "rna_frac"):
            v = m.get(col, "")
            if v in ("", "NA", None):
                ok = False
                break
            vals.append(float(v))
        region = ([vals[0], vals[1], vals[2],
                   max(0.0, 1.0 - sum(vals))] if ok else [-1.0] * 4)
        fv = m.get("active_frame", "")
        frame = float(int(fv)) if fv not in ("", "NA", None) else -1.0
        sv = m.get("struct_flags", "")
        flags = ([float((int(sv) >> b) & 1) for b in range(4)]
                 if sv not in ("", "NA", None) else [-1.0] * 4)
        doms: list[float] = []
        dm: list[float] = []
        if K:
            dv = m.get("domain_ids")
            if dv is None or dv == "NA":
                doms, dm = [0.0] * K, [-1.0]
            else:
                dm = [1.0]
                pos = {t for t in dv.split(",") if t}
                doms = [1.0 if i in pos else 0.0 for i in domain_vocab]
        out[fid] = tuple(region + [frame] + flags + doms + dm)
    out["_k"] = K
    return out


def build_split(meta: dict[str, dict], val_frac: float, genus_holdout: float,
                seed: int, frag_val: float = 0.0):
    """(val_accs, holdout_genera, val_frags).

    frag_val > 0 withholds a CONTIGUOUS span of ~frag_val fragments from
    every train-side accession as within-genome validation. Contiguous,
    not random: interleaved holdout leaks through near-identical
    neighbouring windows, whereas a held-out region tests whether the
    class signature persists in unsampled sequence context. This is what
    gives singleton-species genomes (one assembly, so genome-level val
    can never pair them) species-level val coverage."""
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

    val_frags: set = set()
    if frag_val > 0:
        acc_genus: dict[str, str] = {}
        by_acc_seq: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list))
        for fid, m in meta.items():
            acc = m["accession"]
            acc_genus.setdefault(acc, m.get("genus", ""))
            by_acc_seq[acc][m.get("seqid", "")].append(
                (fid, int(m.get("start") or 0)))
        for acc, seqs in by_acc_seq.items():
            if acc in val_accs or acc_genus.get(acc) in holdout_genera:
                continue
            big = max(seqs.values(), key=len)
            big.sort(key=lambda t: t[1])
            w = int(round(len(big) * frag_val))
            if w < 1 or len(big) <= w:
                continue  # too few windows to spare
            i0 = rng.randint(0, len(big) - w)
            val_frags.update(fid for fid, _ in big[i0:i0 + w])
    return val_accs, holdout_genera, val_frags


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
    ap.add_argument("--frag-val", type=float, default=0.0,
                    help="fraction of each train genome's windows withheld "
                         "as one contiguous span for within-genome val "
                         "(species coverage for singleton-species genomes)")
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
    ap.add_argument("--aux", type=str, default="",
                    help="comma list of extra input branches fused into "
                         "the latent (A3): hash=canonical 11-mer "
                         "EmbeddingBag, aa=6-frame AA 3-mer spectrum")
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
    ap.add_argument("--flat-prior-tau", type=float, default=0.0,
                    help="tau for unconditional prior logit adjustment on "
                         "FLAT heads: adds tau*log(nu) to logits inside "
                         "the training cross-entropy only (Menon et al. "
                         "2021); raw logits stay prior-balanced at eval")
    ap.add_argument("--aa-mode", choices=["plain", "bottleneck", "gated"],
                    default="plain",
                    help="aa aux fusion: plain concat; bottleneck forces a "
                         "64-d compress; gated adds a z_dna-driven scalar "
                         "gate + 18-d frame/stop features and moves "
                         "coding_frac onto a z_dna-only head")
    ap.add_argument("--aa-drop", type=float, default=0.0,
                    help="per-sample Bernoulli prob of zeroing the gated "
                         "aa branch during training (branch dropout)")
    ap.add_argument("--min-train-len", type=int, default=500,
                    help="E2: shortest stochastic sub-window when "
                         "--var-len-prob > 0")
    ap.add_argument("--var-len-prob", type=float, default=0.0,
                    help="E2: per-fragment prob of applying a variable "
                         "length crop (70% contiguous sub-window, 30% "
                         "unmerged paired-end) during training")
    ap.add_argument("--region-head", action="store_true",
                    help="E3: add region_fracs 4-way soft head "
                         "[coding, intron, rna, intergenic] with "
                         "inverse-prior weight on non-coding windows")
    ap.add_argument("--struct-head", action="store_true",
                    help="E4: add active_frame (8-way) + struct_flags "
                         "(4-bit) heads on cat([z, 144-d per-frame aa "
                         "summary]); emits x['aa_frame']")
    ap.add_argument("--domain-head-k", type=int, default=0,
                    help="E5: add a K-way multi-label domains head over "
                         "the top-K IPR/PF ids seen on >=20 train windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = load_labels(args.labels)
    val_accs, holdout_genera, val_frags = build_split(
        meta, args.val_frac, args.holdout_genus_frac, args.seed,
        frag_val=args.frag_val)
    print(f"{len(val_accs)} val genomes, "
          f"{len(holdout_genera)} holdout genera, "
          f"{len(val_frags)} frag-val windows")

    def is_val(fid, m):
        return (m["accession"] in val_accs
                or m["genus"] in holdout_genera
                or fid in val_frags)

    train_meta = {k: v for k, v in meta.items() if not is_val(k, v)}
    val_meta = {k: v for k, v in meta.items() if is_val(k, v)}
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
    # build_tree is computed for every head: the sibling priors drive
    # TreeRankHead, while parent_of is saved into the checkpoint so the
    # evaluator can reconcile flat-head calls along the taxonomy tree
    # at inference without retraining.
    tree = build_tree(train_meta, vocab, cls_targets)
    flat_priors = build_flat_priors(train_meta, vocab, cls_targets)

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

    # P3 annotation labels. domain_vocab comes from TRAIN rows only and
    # requires >=20 windows per domain so the multi-label head is not
    # fitted to singleton noise. The annot tuple is appended to yc when
    # any of the three heads is enabled.
    annot_any = (args.region_head or args.struct_head
                 or args.domain_head_k > 0)
    domain_vocab: dict[str, int] = {}
    annot_train = annot_val = None
    if annot_any:
        if args.domain_head_k:
            from collections import Counter
            cnt: Counter = Counter()
            for m in train_meta.values():
                dv = m.get("domain_ids")
                if dv and dv != "NA":
                    cnt.update(t for t in dv.split(",") if t)
            domain_vocab = {
                d: i for i, (d, c) in enumerate(cnt.most_common())
                if c >= 20 and i < args.domain_head_k}
            print(f"domain head: {len(domain_vocab)} of "
                  f"{args.domain_head_k} slots filled")
        annot_train = encode_annot_labels(train_meta, domain_vocab)
        annot_val = encode_annot_labels(val_meta, domain_vocab)

    use_seq = args.backbone == "seq"
    ms_maps = ({k: build_cell_lut(k) for k in MS_KS},
               {k: rc_index_map(k) for k in MS_KS}) \
        if args.backbone == "msfcgr" else (None, None)
    aux = {a for a in args.aux.split(",") if a}
    hash_canon = rc_index_map(HASH_K) if "hash" in aux else None
    aa_extra = args.aa_mode == "gated"
    train_ds = FragmentDataset(args.fragments, encode_labels(train_meta),
                               args.k, lut,
                               augment=args.vertex_permute_augment,
                               pos=args.pos_channels,
                               targets=cls_targets, cont=cont_train,
                               seq=use_seq, seq_len=args.seq_len,
                               ms_luts=ms_maps[0], ms_canons=ms_maps[1],
                               aux=aux, hash_canon=hash_canon,
                               aa_extra=aa_extra,
                               var_len_prob=args.var_len_prob,
                               min_train_len=args.min_train_len,
                               aa_frame=args.struct_head,
                               annot=annot_train)
    val_ds = FragmentDataset(args.fragments, encode_labels(val_meta),
                             args.k, lut, pos=args.pos_channels,
                             targets=cls_targets, cont=cont_val,
                             seq=use_seq, seq_len=args.seq_len,
                             ms_luts=ms_maps[0], ms_canons=ms_maps[1],
                             aux=aux, hash_canon=hash_canon,
                             aa_extra=aa_extra,
                             aa_frame=args.struct_head,
                             annot=annot_val)
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
                     head=args.head, tree=tree if args.head == "tree" else None,
                     aux=aux, aa_mode=args.aa_mode,
                     aa_drop=args.aa_drop,
                     region_head=args.region_head,
                     struct_head=args.struct_head,
                     domain_k=len(domain_vocab)).to(device)
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
            if isinstance(x, dict):
                x = {kk: v.to(device) for kk, v in x.items()}
            else:
                x = x.to(device)
            y, yc = y.to(device), yc.to(device)
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
                    terms = []
                    for i, t in enumerate(cls_targets):
                        if not (y[:, i] != 0).any():
                            continue
                        lg = logits[t]
                        if train and args.flat_prior_tau > 0:
                            lg = lg + args.flat_prior_tau * \
                                flat_priors[t].to(device)
                        terms.append(F.cross_entropy(
                            lg, y[:, i], ignore_index=0,
                            label_smoothing=args.label_smoothing))
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
                if yc.size(1) > 2:          # P3 annotation columns
                    ya = yc[:, 2:]
                    if "region_fracs" in logits:
                        rm = ya[:, 0] >= 0
                        if rm.any():
                            # soft-CE on the composition simplex; pure
                            # non-coding windows are inverse-prior
                            # upweighted (91% coding imbalance)
                            pr = F.log_softmax(
                                logits["region_fracs"][rm], 1)
                            w = torch.where(
                                ya[rm, 0] < 0.1, 3.0, 1.0)
                            terms.append(-(w[:, None] * ya[rm, :4]
                                           * pr).sum(1).mean())
                    if "active_frame" in logits:
                        fm2 = ya[:, 4] >= 0
                        if fm2.any():
                            terms.append(F.cross_entropy(
                                logits["active_frame"][fm2],
                                ya[fm2, 4].long()))
                    if "struct_flags" in logits:
                        sm = ya[:, 5] >= 0
                        if sm.any():
                            terms.append(
                                F.binary_cross_entropy_with_logits(
                                    logits["struct_flags"][sm],
                                    ya[sm, 5:9],
                                    pos_weight=torch.tensor(
                                        [5., 5., 8., 8.],
                                        device=device)))
                    if "domains" in logits:
                        dm = ya[:, -1] >= 0
                        if dm.any():
                            K = logits["domains"].size(1)
                            terms.append(
                                F.binary_cross_entropy_with_logits(
                                    logits["domains"][dm],
                                    ya[dm, ANNOT_BASE:
                                       ANNOT_BASE + K]))
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
            # parent_of + flat priors are saved for every head so
            # eval_pilot can do cross-rank tree reconciliation and
            # prior diagnostics without rebuilding them from labels.tsv
            ckpt["tree"] = {
                "parent_of": {r: tree["parent_of"][r].tolist()
                              for r in cls_targets},
                "log_prior": {r: tree["log_prior"][r].tolist()
                              for r in cls_targets}}
            ckpt["flat_log_prior"] = {
                r: flat_priors[r].tolist() for r in cls_targets}
            if domain_vocab:
                ckpt["domain_vocab"] = domain_vocab
            torch.save(ckpt, args.out / "best.pt")
    (args.out / "vocab.json").write_text(json.dumps(vocab, indent=1))
    print(f"done. best mean val acc {best:.3f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
