"""OmniPhyl sequence encoders — single source of truth.

Extracted verbatim from ``scripts/train_omniphyl.py`` so the training
script, the evaluator, and the inference package share one canonical
implementation. ``scripts/train_omniphyl.py`` re-exports these names.

Also hosts the strand-exact codon-boundary scanner ``scan_transition``
(verbatim from ``scripts/eval_pilot.py``) and the ``FeatureEncoder``
helper used at inference time.
"""
from __future__ import annotations

import numpy as np

# A=0 C=1 G=2 T=3; every other base (N, ambiguous, lowercase->upper) = -1
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
    # bincount over repeated cell ids is ~5x faster than np.add.at
    # scatter-add on 500-elem windows (runs in every dataloader worker).
    freq += np.bincount(cells, minlength=side * side).astype(np.float32)
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
        sx += np.bincount(cells, weights=px[ends],
                          minlength=side * side).astype(np.float32)
        sy += np.bincount(cells, weights=py[ends],
                          minlength=side * side).astype(np.float32)
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
AA_EXTRA_V2_DIM = 30           # ToL: + per-frame (TGA-only ORF frac,
                               #   AGA/AGG density) -- ciliate (code 6) and
                               #   vertebrate-mito (code 2) signatures

# ------------------------------------------------------ genetic codes ---
# Standard NCBI translation table (code 1); letters use the AA_LETTERS
# alphabet with "*" = stop. Codes that reassign sense codons or stops are
# expressed as overrides on this table (NCBI gc.prt).
_STANDARD_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Codon reassignments per NCBI genetic code (gc.prt), relative to code 1.
# Only differences from the standard table are listed; code 11 is
# amino-acid-identical to 1 (it differs in the START set only).
_GCODE_OVERRIDES: dict[int, dict[str, str]] = {
    # 2  Vertebrate mitochondrial: TGA->Trp, ATA->Met, AGA/AGG->stop
    2: {"TGA": "W", "ATA": "M", "AGA": "*", "AGG": "*"},
    # 4  Mold/protozoan/coelenterate mt + Mycoplasma/Spiroplasma: TGA->Trp
    4: {"TGA": "W"},
    # 5  Invertebrate mitochondrial: TGA->Trp, ATA->Met, AGA/AGG->Ser
    5: {"TGA": "W", "ATA": "M", "AGA": "S", "AGG": "S"},
    # 6  Ciliate/Dasycladacean/Hexamita nuclear: TAA/TAG->Gln (TGA only stop)
    6: {"TAA": "Q", "TAG": "Q"},
    # 11 Bacterial/Archaeal/Plastid: sense codons identical to code 1
    11: {},
    # 12 Alternative yeast nuclear (Candida CTG clade): CTG->Ser
    12: {"CTG": "S"},
    # 16 Chlorophycean mitochondrial: TAG->Leu
    16: {"TAG": "L"},
    # 22 Scenedesmus obliquus mitochondrial: TAG->Leu, TCA->stop
    22: {"TAG": "L", "TCA": "*"},
    # 23 Thraustochytrium mitochondrial: TTA->stop
    23: {"TTA": "*"},
    # 25 Candidate Division SR1 / Gracilibacteria: TGA->Gly
    25: {"TGA": "G"},
}

_BASES = "ACGT"
# base-4 codon index -> codon string (d0 most significant)
_CODON_STRS = [_BASES[i // 16] + _BASES[(i // 4) % 4] + _BASES[i % 4]
               for i in range(64)]


def _build_aa_lut(overrides: dict[str, str] | None = None) -> np.ndarray:
    """codon base-4 index (A,C,G,T digit order) -> AA letter idx; 20=stop.

    overrides: codon -> letter reassignments on top of the standard code
    (code 1); ``overrides=None`` reproduces the legacy standard LUT
    bit-exactly.
    """
    table = _STANDARD_CODE if not overrides else {**_STANDARD_CODE,
                                                  **overrides}
    lut = np.full(64, -1, dtype=np.int64)
    for i in range(64):
        aa = table[_CODON_STRS[i]]
        lut[i] = AA_ALPHA - 1 if aa == "*" else AA_LETTERS.index(aa)
    return lut


GCODE_LUTS: dict[int, np.ndarray] = {
    g: _build_aa_lut(_GCODE_OVERRIDES.get(g))
    for g in (1, 2, 4, 5, 6, 11, 12, 16, 22, 23, 25)}
_AA_LUT = GCODE_LUTS[1]      # standard code; fixed Stage-1 basis

GCODE_STOPS: dict[int, set] = {
    g: {_CODON_STRS[i] for i in range(64) if lut[i] == AA_ALPHA - 1}
    for g, lut in GCODE_LUTS.items()}

# Start codons per code (gc.prt). Code 1 here uses the prokaryotic-style
# ATG/GTG/TTG set historically named STARTS (euk nuclear callers pass
# ("ATG",) explicitly, as before).
GCODE_STARTS: dict[int, set] = {
    1: {"ATG", "GTG", "TTG"},
    2: {"ATG", "ATA", "ATT", "ATC", "GTG"},
    4: {"TTA", "TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"},
    5: {"TTG", "ATT", "ATC", "ATA", "ATG", "GTG"},
    6: {"ATG"},
    11: {"ATG", "GTG", "TTG", "CTG", "ATT", "ATC", "ATA"},
    12: {"ATG", "CTG"},
    16: {"ATG"},
    22: {"ATG"},
    23: {"ATT", "ATG", "GTG"},
    25: {"TTG", "ATG", "GTG"},
}

# Codes whose stop set differs from the standard code and can therefore
# be distinguished by in-frame codon-usage likelihood. Standard-stop
# codes {1, 11, 12} share identical stops and are resolved from
# replicon/taxon context instead (see annotate._infer_gcode).
GCODE_STOP_VARIANTS = (1, 2, 4, 5, 6, 16, 22, 23, 25)


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


def aa_encode(seq: str, extra: bool = False,
              v2: bool = False) -> np.ndarray:
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

    v2=True (aa_extra_v2) appends two more per-frame features, widening
    the tail to AA_EXTRA_V2_DIM=30:
      - tga_only_orf_frac: longest ORF fraction when only TGA counts as
        a stop. On ciliate nuclear genomes (code 6, TAA/TAG->Gln) the
        standard-code ORF measure collapses while this one stays high --
        the signature for the reassignments Stage-1 must recognize.
      - aga_agg_density: AGA+AGG codon share of the frame. Vertebrate
        mitochondria (code 2) show near-zero in-frame AGA/AGG (they are
        stops); standard-code genomes keep ~Arg background density.
    """
    codes = BASE2DIGIT[np.frombuffer(seq.encode("ascii", "replace"),
                                     dtype=np.uint8)]
    vec = np.zeros(AA_DIM, dtype=np.float32)
    feats: list[float] = []
    nf = 5 if (extra and v2) else 3
    for c in (codes, 3 - codes[::-1]):      # forward + reverse-complement
        for off in range(3):
            n = (len(c) - off) // 3 * 3
            if n <= 0:
                if extra:
                    feats.extend((0.0,) * nf)
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
                ccod = cidx[ok]            # codon ids aligned with aa
                # TGA codon id = 3*16 + 2*4 + 0 = 56; AGA=8, AGG=10
                tga = int((ccod[is_stop] == 56).sum())
                feats.extend((n_st / max(n_cod, 1),
                              float(runs.max()) / max(n_cod, 1)
                              if len(runs) else 0.0,
                              tga / max(n_st, 1)))
                if v2:
                    is_tga = is_stop & (ccod == 56)
                    sp2 = np.where(is_tga)[0]
                    runs2 = np.diff(
                        np.concatenate(([-1], sp2, [n_cod]))) - 1
                    aga_agg = int((((ccod == 8) | (ccod == 10))
                                   & (aa >= 0)).sum())
                    feats.extend(
                        (float(runs2.max()) / max(n_cod, 1)
                         if len(runs2) else 0.0,
                         aga_agg / max(n_cod, 1)))
            if len(aa) < 3:
                continue
            w = np.lib.stride_tricks.sliding_window_view(aa, 3)
            idx = (w * np.array([AA_ALPHA ** 2, AA_ALPHA, 1])).sum(axis=1)
            vec += np.bincount(idx, minlength=AA_DIM).astype(np.float32)
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


# -------------------------------------- repeat/complexity scalars ----

DNA_STATS_DIM = 8


def dna_stats_encode(seq: str) -> np.ndarray:
    """sequence -> [DNA_STATS_DIM] float32 window-level composition /
    repeat-complexity scalars (ToL: vertebrate + plant corpora are
    40-85% transposable elements and tandem repeats).

    Kept as scalars fused into z_dna rather than extra conv channels so
    backbone weight shapes are untouched. Features, in order:
      0  shannon_k1  -- base-composition entropy / log2(4)
      1  shannon_k3  -- 3-mer entropy / log2(64); low on repeats
      2  gc          -- GC fraction
      3  cpg_oe      -- CpG observed/expected (diploid promoters /
                        5' exons in vertebrates are CpG-rich)
      4  homo_frac   -- longest homopolymer run / len (poly-A/T,
                        simple-sequence expansion)
      5  tandem2     -- max fraction covered by a tandem dinucleotide
                        repeat (e.g. (CA)n microsats)
      6  tandem3     -- same for trinucleotides (CAG expansions etc.)
      7  n_frac      -- ambiguous-base fraction
    """
    codes = BASE2DIGIT[np.frombuffer(seq.encode("ascii", "replace"),
                                     dtype=np.uint8)]
    n = len(codes)
    out = np.zeros(DNA_STATS_DIM, dtype=np.float32)
    if n == 0:
        return out
    ok = codes >= 0
    nv = int(ok.sum())
    out[7] = 1.0 - nv / n
    if nv == 0:
        return out
    c = codes[ok]
    # k=1 and k=3 Shannon entropy (normalized to [0,1] by max entropy)
    cnt1 = np.bincount(c, minlength=4).astype(np.float64) / nv
    out[0] = float(-(cnt1[cnt1 > 0] * np.log2(cnt1[cnt1 > 0])).sum() / 2.0)
    if nv >= 3:
        trip = c[:nv - (nv % 3)].reshape(-1, 3)
        # contiguous 3-mers only within the filtered valid stretch
        if len(trip) >= 1:
            tidx = trip[:, 0] * 16 + trip[:, 1] * 4 + trip[:, 2]
            cnt3 = np.bincount(tidx, minlength=64).astype(np.float64)
            cnt3 /= cnt3.sum()
            out[1] = float(-(cnt3[cnt3 > 0]
                             * np.log2(cnt3[cnt3 > 0])).sum() / 6.0)
    # GC + CpG O/E on valid dinucleotide positions
    gc = float(((c == 1) | (c == 2)).sum()) / nv
    out[2] = gc
    pc, pg = cnt1[1], cnt1[2]
    if nv >= 2 and pc > 0 and pg > 0:
        di = c[:-1] * 4 + c[1:]
        # dinucleotide index C->G is 1*4+2 = 6
        cpg = float((di == 6).sum()) / (nv - 1)
        out[3] = min(cpg / max(pc * pg, 1e-9), 4.0) / 4.0
    # longest homopolymer run fraction (on raw incl. N-break boundaries)
    if nv:
        changes = np.concatenate(([True], c[1:] != c[:-1]))
        runs = np.diff(np.concatenate(
            (np.flatnonzero(changes), [len(c)])))
        out[4] = float(runs.max()) / nv
    # tandem di-/tri-nucleotide repeat coverage: scan periods 2,3
    for k, out_i in ((2, 5), (3, 6)):
        best = 0
        for off in range(k):
            s = c[off:]
            w = len(s) // k
            if w < 2:
                continue
            pat = s[:w * k].reshape(w, k)
            match = (pat == pat[0][None, :]).all(1)
            # longest run of matching units
            i = 0
            while i < len(match):
                if match[i]:
                    j = i
                    while j + 1 < len(match) and match[j + 1]:
                        j += 1
                    best = max(best, (j - i + 1) * k)
                    i = j + 1
                else:
                    i += 1
        out[out_i] = best / nv
    return out


# -------------------------------------------- stage-2 boundary scan ---

STOPS = GCODE_STOPS[1]          # TAA, TAG, TGA (standard code)
STARTS = GCODE_STARTS[1]        # ~15-20% of prokaryotic starts are non-ATG


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
    cands = STARTS if kind == "start" else STOPS
    best, bd = None, 1e18
    for i in range(off, L - 2, 3):
        if s2[i:i + 3] not in cands:
            continue
        edge = (L - i if kind == "start" else L - i - 3) if rev \
            else (i if kind == "start" else i + 3)
        if abs(edge - pred_pos) < bd:
            bd, best = abs(edge - pred_pos), edge
    return best


# --------------------------------------------- inference-time batching ---


class FeatureEncoder:
    """Checkpoint-configured batch encoder for inference.

    Caches the quadtree LUTs / canonical RC maps once, then encodes a
    list of sequences into the tensor dict ``OmniPhyl.forward`` expects.

    ``virtual_joined=True`` marks 'N'-joined virtual fragments (pre-contig
    read packing): when the checkpoint used ``pos_channels``, the two
    mean-CGR positional channels are zeroed so arbitrary read order cannot
    inject artificial positional skew (docs/inference-plan.md §4.2).
    """

    def __init__(self, k: int, train_args: dict):
        self.k = int(k)
        self.backbone = train_args.get("backbone", "resnet")
        self.pos = bool(train_args.get("pos_channels", False))
        self.seq_len = int(train_args.get("seq_len", 500))
        self.aux = {a for a in (train_args.get("aux") or "").split(",")
                    if a}
        self.aa_mode = train_args.get("aa_mode", "plain")
        self.aa_extra_v2 = bool(train_args.get("aa_extra_v2", False))
        self.struct_head = bool(train_args.get("struct_head"))
        self.dna_stats = bool(train_args.get("dna_stats", False))
        self.lut = build_cell_lut(self.k)
        if self.backbone == "msfcgr":
            self.ms_luts = {kk: build_cell_lut(kk) for kk in MS_KS}
            self.ms_canons = {kk: rc_index_map(kk) for kk in MS_KS}
        else:
            self.ms_luts = self.ms_canons = None
        self.hash_canon = (rc_index_map(HASH_K) if "hash" in self.aux
                           else None)

    def encode_one(self, seq: str,
                   virtual_joined: bool = False) -> dict:
        """seq -> dict of numpy arrays for one input."""
        out: dict = {}
        if self.backbone == "seq":
            img = seq_encode(seq, self.seq_len)
        elif self.backbone == "msfcgr":
            img = fcgr_encode_ms(seq, self.ms_luts, self.ms_canons,
                                 pos=self.pos)
        else:
            img = fcgr_encode(seq, self.k, self.lut, pos=self.pos)
        if virtual_joined and self.pos and img.shape[0] >= 2:
            # 'N'-joined reads have no meaningful left/right positioning;
            # zero the two mean-CGR channels (their expected value on a
            # stationary sequence is ~uniform anyway).
            img[-2] = 0.0
            img[-1] = 0.0
        out["x"] = img
        if "hash" in self.aux:
            out["hash"] = hash_encode(seq, self.hash_canon)
        if "aa" in self.aux:
            out["aa"] = aa_encode(seq, extra=(self.aa_mode == "gated"),
                                  v2=self.aa_extra_v2)
        if self.struct_head:
            out["aa_frame"] = aa_frame_encode(seq)
        if self.dna_stats:
            out["dna_stats"] = dna_stats_encode(seq)
        return out

    def encode_batch(self, seqs: list[str],
                     virtual_joined: bool = False) -> dict:
        """list[seq] -> dict[str, np.ndarray] stacked along axis 0."""
        keys = ("x", *sorted(self.aux),
                *(("aa_frame",) if self.struct_head else ()),
                *(("dna_stats",) if self.dna_stats else ()))
        stacks: dict[str, list] = {k: [] for k in keys}
        for s in seqs:
            enc = self.encode_one(s, virtual_joined=virtual_joined)
            for k in keys:
                stacks[k].append(enc[k])
        return {k: np.stack(v) for k, v in stacks.items()}
