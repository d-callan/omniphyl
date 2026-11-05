"""Streaming sequence I/O, LexicMap evidence loading, and output writers.

No third-party dependencies beyond numpy. FASTQ records are parsed by
length (sequence lines until '+', then quality lines until
len(qual) == len(seq)) so wrapped records and quality strings starting
with '@' never corrupt the stream.
"""
from __future__ import annotations

import gzip
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

_COV_RE = re.compile(r"(?:cov_|multi=)(\d+\.?\d*)")
_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def revcomp(seq: str) -> str:
    return seq.translate(_COMP)[::-1]


def _open(path: Path):
    """Plain-text or gzip reader by magic bytes / .gz suffix."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b" or path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path)


def _coverage(header: str) -> float:
    m = _COV_RE.search(header)
    return float(m.group(1)) if m else 1.0


def iter_sequences(path: Path) -> Iterator[tuple[str, str, float]]:
    """(record_id, sequence, coverage) over FASTA or FASTQ, .gz-aware.

    record_id is the first whitespace-delimited token of the header.
    coverage comes from 'cov_X' / 'multi=X' assembler annotations
    (default 1.0) so per-contig read depth can weight abundance.
    """
    with _open(path) as f:
        first = ""
        for line in f:
            if line.strip():
                first = line
                break
        if not first:
            return
        is_fastq = first.startswith("@")
        if is_fastq:
            yield from _iter_fastq(f, first)
        else:
            yield from _iter_fasta(f, first)


def _iter_fasta(f, first: str):
    header, chunks = None, []
    for line in list((first,)) + list(f):
        if line.startswith(">"):
            if header is not None:
                yield header.split()[0], "".join(chunks), _coverage(header)
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line.strip())
    if header is not None:
        yield header.split()[0], "".join(chunks), _coverage(header)


def _iter_fastq(f, first: str):
    header = first[1:].strip() if first.startswith("@") else None
    seq, in_qual, qlen = [], False, 0
    for line in f:
        s = line.rstrip("\n")
        if in_qual:
            qlen += len(s)
            seq_len = sum(len(c) for c in seq)
            if qlen >= seq_len:
                yield header.split()[0], "".join(seq), _coverage(header)
                header, seq, in_qual, qlen = None, [], False, 0
        elif header is not None and s.startswith("+"):
            in_qual = True
        elif header is None and s.startswith("@"):
            header = s[1:].strip()
        else:
            seq.append(s)


def iter_paired_sequences(r1: Path, r2: Path
                          ) -> Iterator[tuple[str, str, float]]:
    """Lockstep mates -> (id, 'seq1' + 'N' + revcomp(seq2), cov).

    The 'N' junction resets k-mer windows exactly like the training-time
    eval_crop_seq augmentation, so no false cross-insert k-mers form.
    """
    for (i1, s1, c1), (i2, s2, _c2) in zip(iter_sequences(r1),
                                          iter_sequences(r2)):
        yield i1, f"{s1}N{revcomp(s2)}", c1


# ------------------------------------------------------- LexicMap -------


@dataclass
class Hsp:
    start: int          # 0-based half-open query coordinates
    end: int
    sgenome: str        # SRA run / accession (name before first '|')
    sseqid: str         # subject unitig id (for split-alignment linking)
    score: float
    strand: str


@dataclass
class ContigEvidence:
    """Filtered LexicMap evidence for one contig on a fixed bin grid."""
    hsps: list = field(default_factory=list)              # list[Hsp]
    density: np.ndarray | None = None                     # [nbins]
    endpoints: np.ndarray | None = None                   # [nbins]
    intron_candidates: list = field(default_factory=list)  # [(s, e)]
    sra_runs: set = field(default_factory=set)


def _dinuc_entropy(subseq: str) -> float:
    """Shannon entropy of the dinucleotide histogram, in bits (max 4)."""
    codes = np.frombuffer(subseq.upper().encode("ascii", "replace"),
                          dtype=np.uint8)
    lut = np.full(256, -1, np.int8)
    for i, b in enumerate(b"ACGT"):
        lut[b] = i
    d = lut[codes]
    n = len(d) - 1
    if n <= 0:
        return 0.0
    idx = d[:-1] * 4 + d[1:]
    idx = idx[(d[:-1] >= 0) & (d[1:] >= 0)]
    if len(idx) == 0:
        return 0.0
    p = np.bincount(idx, minlength=16) / len(idx)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# Row tuple: (contig, start, end, sgenome, sseqid, score, strand).
# Streaming iterators keep peak memory bounded to one unitig's rows plus
# the per-contig accumulators, so multi-hundred-MB BED/TSV files no
# longer materialize millions of Hsp objects up front.
BedRow = tuple


def _iter_bed(path: Path):
    """LexicMap BED6: chrom start end sgenome|sseqid|hspN score strand."""
    with _open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                c = line.split()
            if len(c) < 6:
                continue
            name = c[3]
            parts = name.split("|")
            sgenome = parts[0] if parts else name
            sseqid = "|".join(parts[:2]) if len(parts) > 1 else name
            yield (c[0], int(c[1]), int(c[2]), sgenome, sseqid,
                   float(c[4]), c[5])


def _iter_bedgraph(path: Path):
    """bedGraph rows -> pseudo-hits (density rows; no per-hit metadata)."""
    with _open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            c = line.split()
            if len(c) < 4:
                continue
            yield (c[0], int(c[1]), int(c[2]), "bedgraph", "bedgraph",
                   float(c[3]), "+")


def _iter_tsv(path: Path):
    """Raw LexicMap TSV (1-based inclusive qstart..qend -> 0-based)."""
    with _open(path) as f:
        header = None
        for line in f:
            if not line.strip():
                continue
            c = line.rstrip("\n").split("\t")
            if header is None:
                header = [h.lower() for h in c]
                continue
            col = dict(zip(header, c))
            try:
                qs, qe = int(col["qstart"]), int(col["qend"])
            except (KeyError, ValueError):
                continue
            sgenome = col.get("sgenome", "")
            sseqid = col.get("sseqid", "")
            hsp = col.get("hsp", "")
            yield (col.get("query", ""), min(qs, qe) - 1, max(qs, qe),
                   sgenome, f"{sgenome}|{sseqid}|hsp{hsp}",
                   float(col.get("bitscore", 0) or 0),
                   col.get("sstr", "+"))


class _ContigAcc:
    """Bounded-memory per-contig accumulator for streaming LexicMap rows.

    Only KEPT intervals are retained: per-SRA-run interval lists (for the
    capped coverage track), endpoint bin counts, split-alignment support
    counts, and the SRA run set. Rows themselves are discarded after
    their unitig block is processed, so peak memory scales with kept
    evidence rather than raw file size.
    """

    __slots__ = ("by_run", "ep_cnt", "support", "sra_runs", "bg_ivs",
                 "hsps_kept", "max_end")

    def __init__(self):
        self.by_run: dict[str, list] = {}   # sgenome -> [(s, e), ...]
        self.ep_cnt: Counter = Counter()    # bin -> endpoint impulses
        self.support: Counter = Counter()   # (d, acc) -> split count
        self.sra_runs: set = set()
        self.bg_ivs: list = []              # (s, e, val) bedGraph rows
        self.hsps_kept: list = []
        self.max_end = 0


def _hsp_seq_ok(seq: str, s: int, e: int, min_entropy: float) -> bool:
    """Sequence-quality screen for one HSP interval.

    Softmasked lowercase bases are NOT ambiguous: `b.upper() not in
    "ACGT"` only counts true N/ambiguous symbols, so a softmasked
    (lowercase) repeat does not automatically discard the hit. The
    dinucleotide-entropy floor still rejects homopolymer / microsatellite
    sequence itself (AT-rich exons of e.g. P. falciparum sit ~1.5-2.2
    bits; simple repeats fall below).
    """
    sub = seq[s:e]
    if not sub:
        return True
    ambig = sum(1 for b in sub if b.upper() not in "ACGT")
    if ambig > 0.2 * len(sub):
        return False
    return _dinuc_entropy(sub) >= min_entropy


def load_lexicmap_evidence(bed_path=None, bedgraph_path=None,
                           tsv_path=None, source: str = "rnaseq",
                           min_score: float = 150.0,
                           min_len: int = 60,
                           min_single_score: float = 350.0,
                           min_intron_support: int = 1,
                           min_entropy: float = 1.5,
                           max_per_run: int = 2,
                           max_introns: int = 500,
                           bin_width: int = 50,
                           contig_seqs: dict | None = None,
                           keep_hsps: bool = False,
                           ) -> dict[str, ContigEvidence]:
    """Parse + filter LexicMap hits into per-contig evidence tracks.

    source:
      "rnaseq"  - transcriptomic index: coverage supports exons/UTRs and
                  internal HSP endpoints mark candidate splice junctions.
      "genomic" - whole-genome index: coverage is weak conservation only;
                  endpoint/intron evidence is dropped entirely.

    Two-tier unitig-aware filtering (.bed/.tsv only):
      * Chained unitigs -- a (sseqid, strand) group with >=2 HSPs
        spanning a plausible intron gap (35-5000 bp) AND an anchor HSP
        with score >= min_single_score -- keep partner HSPs down to
        score >= min_score and length >= min(min_len, 45), so a short
        exon paired with a strong anchor on the same transcript unitig
        survives a global bitscore cutoff that would otherwise erase
        split alignments.
      * All other groups are held to a stricter singleton floor
        (score >= min_single_score, length >= max(min_len, 90)) so
        low-scoring hits from distant relatives or repetitive regions
        contribute only where they form a coherent split alignment.
      * Softmasked (lowercase) sequence is NOT ambiguous; only non-ACGT
        symbols count toward the 20% ambiguity limit, and dinucleotide
        entropy must be >= min_entropy (default 1.5).
      * Rows of one unitig are expected adjacent in the file (true for
        lexicmap_to_ucsc output); an interleaved unitig is processed as
        multiple groups, which only weakens (never fabricates) its
        chain status.
      * Per-SRA-run depth is capped at max_per_run per bin so d(x)
        measures independent sample support; intron candidates require
        >= min_intron_support corroborating splits and are capped at the
        top max_introns by support.
      * ev.hsps is [] unless keep_hsps=True -- nothing downstream reads
        it, and retaining millions of HSP objects is the dominant RAM
        cost on full SRA-index output.
    """
    if tsv_path:
        stream = _iter_tsv(Path(tsv_path))
        per_hit_meta = True
    elif bed_path:
        stream = _iter_bed(Path(bed_path))
        per_hit_meta = True
    elif bedgraph_path:
        stream = _iter_bedgraph(Path(bedgraph_path))
        per_hit_meta = False
    else:
        return {}

    seqs = contig_seqs or {}
    accs: dict[str, _ContigAcc] = {}

    def _flush_unitig(rows, st: _ContigAcc, seq, clen):
        """Two-tier keep decision + accumulation for one unitig block."""
        if not rows:
            return
        rows = sorted(rows, key=lambda r: r[1])
        has_link = any(35 <= b[1] - a[2] <= 5000
                       for a, b in zip(rows, rows[1:]))
        anchor = any(r[5] >= min_single_score for r in rows)
        if has_link and anchor:
            sel = [r for r in rows
                   if r[5] >= min_score
                   and r[2] - r[1] >= min(min_len, 45)]
        else:
            sel = [r for r in rows
                   if r[5] >= max(min_score, min_single_score)
                   and r[2] - r[1] >= max(min_len, 90)]
        if not sel:
            return
        kept = []
        for r in sel:
            s, e = r[1], r[2]
            if seq is not None and not _hsp_seq_ok(
                    seq, max(0, s), min(clen, e), min_entropy):
                continue
            kept.append(r)
            st.by_run.setdefault(r[3], []).append((s, e))
            st.sra_runs.add(r[3])
            st.max_end = max(st.max_end, e)
        for a, b in zip(kept, kept[1:]):
            gap = b[1] - a[2]
            if 35 <= gap <= 5000:
                st.support[(a[2], b[1])] += 1
        if source == "rnaseq":
            for r in kept:
                for x in (r[1], r[2]):
                    if x < 25 or x > clen - 25:
                        continue
                    if (seq is not None
                            and "N" in seq[x - 5:x + 5].upper()):
                        continue
                    st.ep_cnt[x // bin_width] += 1
        if keep_hsps:
            st.hsps_kept.extend(Hsp(r[1], r[2], r[3], r[4], r[5], r[6])
                                for r in kept)

    cur_key = None
    cur_rows: list = []
    for rec in stream:
        contig, s, e, sgenome, sseqid, score, strand = rec
        st = accs.setdefault(contig, _ContigAcc())
        if not per_hit_meta:
            st.bg_ivs.append((s, e, score))
            st.max_end = max(st.max_end, e)
            continue
        ukey = (contig, sseqid, strand)
        if ukey != cur_key:
            if cur_rows:
                pc = cur_rows[0][0]
                _flush_unitig(cur_rows, accs[pc], seqs.get(pc),
                              len(seqs[pc]) if pc in seqs
                              else accs[pc].max_end)
            cur_key, cur_rows = ukey, []
        cur_rows.append(rec)
    if cur_rows:
        pc = cur_rows[0][0]
        _flush_unitig(cur_rows, accs[pc], seqs.get(pc),
                      len(seqs[pc]) if pc in seqs else accs[pc].max_end)

    out: dict[str, ContigEvidence] = {}
    for contig, st in accs.items():
        seq = seqs.get(contig)
        clen = len(seq) if seq else st.max_end
        if clen <= 0:
            continue
        nbins = max(1, math.ceil(clen / bin_width))
        ev = ContigEvidence()
        ev.hsps = st.hsps_kept
        ev.sra_runs = st.sra_runs
        if not per_hit_meta:
            ev.density = np.zeros(nbins, np.float32)
            for s, e, v in st.bg_ivs:
                b0 = s // bin_width
                b1 = (e - 1) // bin_width
                ev.density[b0:b1 + 1] = np.maximum(
                    ev.density[b0:b1 + 1], v)
        else:
            # per-SRA-run capped depth per bin: d[b] = sum_r
            # min(cnt_r, cap), computed as raw coverage minus each
            # over-cap run's excess; runs with <= cap HSPs can never
            # exceed the cap anywhere.
            diff = np.zeros(nbins + 1, np.float64)
            for ivs in st.by_run.values():
                for s, e in ivs:
                    if s // bin_width >= nbins:
                        continue
                    diff[s // bin_width] += 1
                    diff[min((e - 1) // bin_width, nbins - 1) + 1] -= 1
            density = np.cumsum(diff)[:nbins]
            for ivs in st.by_run.values():
                if len(ivs) <= max_per_run:
                    continue
                diff_r = np.zeros(nbins + 1, np.float64)
                for s, e in ivs:
                    if s // bin_width >= nbins:
                        continue
                    diff_r[s // bin_width] += 1
                    diff_r[min((e - 1) // bin_width, nbins - 1)
                           + 1] -= 1
                cnt_r = np.cumsum(diff_r)[:nbins]
                density -= np.maximum(0.0, cnt_r - max_per_run)
            ev.density = density.astype(np.float32)
            if source == "rnaseq":
                ev.endpoints = np.zeros(nbins, np.float32)
                for b, c in st.ep_cnt.items():
                    if b < nbins:
                        ev.endpoints[b] = c
                # p95 (not max) normalization: one hyper-expressed
                # locus must not squash every other junction's impulse
                nz = ev.endpoints[ev.endpoints > 0]
                if len(nz):
                    scale = max(float(np.percentile(nz, 95)), 1.0)
                    ev.endpoints = np.minimum(1.0,
                                              ev.endpoints / scale)
                # deduplicated split-alignment candidates ordered by
                # descending support so high-depth junctions win the
                # greedy non-overlap chain in _splice_candidates
                ev.intron_candidates = [
                    iv for iv, c in sorted(
                        st.support.items(),
                        key=lambda kv: (-kv[1], kv[0]))
                    if c >= min_intron_support][:max_introns]
        out[contig] = ev
    return out


# ---------------------------------------------------------- writers -----

_GFF_SAFE = re.compile(r"[^A-Za-z0-9.:^*$@!+_?|-]")


def _gff_attr_escape(v: str) -> str:
    return _GFF_SAFE.sub(lambda m: f"%{ord(m.group()):02X}", v)


def write_gff3(path, features: Iterator[dict]):
    """Write feature dicts as GFF3.

    Each feature: seqid, type, start, end (1-based inclusive), strand,
    phase, score, attributes (dict). source defaults to 'omniphyl'.
    """
    with open(path, "w") as f:
        f.write("##gff-version 3\n")
        for ft in features:
            attrs = ";".join(
                f"{k}={_gff_attr_escape(str(v))}"
                for k, v in ft.get("attributes", {}).items())
            f.write("\t".join([
                str(ft["seqid"]), ft.get("source", "omniphyl"),
                ft["type"], str(int(ft["start"])), str(int(ft["end"])),
                str(ft.get("score", ".")),
                ft.get("strand", "."),
                str(ft.get("phase", ".")), attrs or "."]) + "\n")


def write_bedgraph(path, seqid: str, values: np.ndarray,
                   bin_width: int, digits: int = 3):
    """Per-bin float track -> bedGraph (runs of equal value merged)."""
    with open(path, "w") as f:
        f.write(f"track type=bedGraph name={seqid}\n")
        i = 0
        n = len(values)
        while i < n:
            v = float(values[i])
            j = i + 1
            while j < n and float(values[j]) == v:
                j += 1
            f.write(f"{seqid}\t{i * bin_width}\t{j * bin_width}\t"
                    f"{v:.{digits}f}\n")
            i = j
