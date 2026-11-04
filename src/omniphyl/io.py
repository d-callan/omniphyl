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


def _parse_bed(path: Path) -> dict[str, list[Hsp]]:
    """LexicMap BED6: chrom start end sgenome|sseqid|hspN score strand."""
    out: dict[str, list[Hsp]] = {}
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
            out.setdefault(c[0], []).append(Hsp(
                int(c[1]), int(c[2]), sgenome, sseqid,
                float(c[4]), c[5]))
    return out


def _parse_bedgraph(path: Path) -> dict[str, list[Hsp]]:
    """bedGraph -> pseudo-HSPs (density rows; no per-hit metadata)."""
    out: dict[str, list[Hsp]] = {}
    with _open(path) as f:
        for line in f:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            c = line.split()
            if len(c) < 4:
                continue
            out.setdefault(c[0], []).append(Hsp(
                int(c[1]), int(c[2]), "bedgraph", "bedgraph",
                float(c[3]), "+"))
    return out


def _parse_tsv(path: Path) -> dict[str, list[Hsp]]:
    """Raw LexicMap TSV (1-based inclusive qstart..qend -> 0-based)."""
    out: dict[str, list[Hsp]] = {}
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
            out.setdefault(col.get("query", ""), []).append(Hsp(
                min(qs, qe) - 1, max(qs, qe),
                sgenome, f"{sgenome}|{sseqid}|hsp{hsp}",
                float(col.get("bitscore", 0) or 0),
                col.get("sstr", "+")))
    return out


def load_lexicmap_evidence(bed_path=None, bedgraph_path=None,
                           tsv_path=None, source: str = "rnaseq",
                           min_score: float = 150.0,
                           min_len: int = 60, max_per_run: int = 2,
                           bin_width: int = 50,
                           contig_seqs: dict | None = None,
                           ) -> dict[str, ContigEvidence]:
    """Parse + filter LexicMap hits into per-contig evidence tracks.

    source:
      "rnaseq"  - transcriptomic index: coverage supports exons/UTRs and
                  internal HSP endpoints mark candidate splice junctions.
      "genomic" - whole-genome index: coverage is weak conservation only;
                  endpoint/intron evidence is dropped entirely.

    Filters (applied to .bed/.tsv; .bedGraph rows arrive pre-aggregated):
      score >= min_score, length >= min_len, dinucleotide entropy >= 1.8,
      <=20% ambiguous bases; then per-SRA-run depth is capped at
      max_per_run per bin so d(x) measures independent sample support.
    """
    if tsv_path:
        raw = _parse_tsv(Path(tsv_path))
    elif bed_path:
        raw = _parse_bed(Path(bed_path))
    elif bedgraph_path:
        raw = _parse_bedgraph(Path(bedgraph_path))
    else:
        return {}
    per_hit_meta = not bedgraph_path  # bedGraph has no run/hit identity

    out: dict[str, ContigEvidence] = {}
    for contig, hsps in raw.items():
        seq = (contig_seqs or {}).get(contig)
        clen = len(seq) if seq else max(h.end for h in hsps)
        nbins = max(1, math.ceil(clen / bin_width))
        ev = ContigEvidence()
        ev.density = np.zeros(nbins, np.float32)
        ev.endpoints = np.zeros(nbins, np.float32)

        kept: list[Hsp] = []
        for h in hsps:
            if per_hit_meta:
                if h.score < min_score or h.end - h.start < min_len:
                    continue
                if seq is not None:
                    sub = seq[max(0, h.start):min(clen, h.end)]
                    if sub:
                        ambig = sum(1 for b in sub
                                    if b.upper() not in "ACGT") + sum(
                            1 for b in sub if b.islower())
                        if ambig > 0.2 * len(sub):
                            continue
                        if _dinuc_entropy(sub) < 1.8:
                            continue
            kept.append(h)
        ev.hsps = kept
        ev.sra_runs = {h.sgenome for h in kept if per_hit_meta}

        if not per_hit_meta:
            # bedGraph rows are already per-interval densities; copy the
            # row value into covered bins (max wins on overlaps)
            for h in kept:
                b0 = h.start // bin_width
                b1 = (h.end - 1) // bin_width
                ev.density[b0:b1 + 1] = np.maximum(
                    ev.density[b0:b1 + 1], h.score)
        else:
            # per-SRA-run capped depth per bin: d[b] = Σ_r min(cnt_r, cap)
            # computed as raw coverage minus per-run excess; runs with
            # <= cap total HSPs can never exceed the cap anywhere.
            diff = np.zeros(nbins + 1, np.float64)
            for h in kept:
                diff[h.start // bin_width] += 1
                diff[(h.end - 1) // bin_width + 1] -= 1
            ev.density = np.cumsum(diff)[:nbins]
            by_run: dict[str, list] = {}
            for h in kept:
                by_run.setdefault(h.sgenome, []).append(h)
            for hsps_r in by_run.values():
                if len(hsps_r) <= max_per_run:
                    continue
                diff_r = np.zeros(nbins + 1, np.float64)
                for h in hsps_r:
                    diff_r[h.start // bin_width] += 1
                    diff_r[(h.end - 1) // bin_width + 1] -= 1
                cnt_r = np.cumsum(diff_r)[:nbins]
                ev.density -= np.maximum(0.0, cnt_r - max_per_run)
            ev.density = ev.density.astype(np.float32)

        if source == "rnaseq" and per_hit_meta:
            # endpoint impulses (splice-junction candidates), skipping
            # edges within 25 bp of contig ends / N-gaps
            for h in kept:
                for x in (h.start, h.end):
                    if x < 25 or x > clen - 25:
                        continue
                    if seq is not None and "N" in seq[x - 5:x + 5]:
                        continue
                    ev.endpoints[min(x // bin_width, nbins - 1)] += 1
            if ev.endpoints.max() > 0:
                ev.endpoints /= ev.endpoints.max()
            # split-alignment intron candidates: same unitig, same
            # orientation, genomic gap in [35, 5000]
            by_unitig: dict[tuple, list] = {}
            for h in kept:
                by_unitig.setdefault((h.sseqid, h.strand), []).append(h)
            for group in by_unitig.values():
                group.sort(key=lambda h: h.start)
                for a, b in zip(group, group[1:]):
                    gap = b.start - a.end
                    if 35 <= gap <= 5000:
                        ev.intron_candidates.append((a.end, b.start))
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
