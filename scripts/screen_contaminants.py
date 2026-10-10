#!/usr/bin/env python3
"""Sequence-based contaminant screen for OmniPhyl pilot fragments.

Three layers, cheapest first (no external dependencies — pure Python + stdlib):

  1. Ambiguity gate: fragments with >MAX_AMB_FRAC non-ACGT bases are rejected.
     (Already applied in fetch_refseq_corpus.py at 5%; this step re-checks and
     allows tuning independently of the fetch step.)

  2. k-mer self-consistency: for each assembly, build a reference 21-mer sketch
     from all sequences in the downloaded FASTA.  Compute per-fragment Jaccard
     containment of its 21-mers in that sketch.  Fragments below
     --min-self-containment are flagged as inconsistent with their source
     assembly (potential inserted contamination or mis-assembly).

     Threshold note: the distribution of containment across 500 bp fragments
     from real RefSeq assemblies is strongly bimodal — most fragments cluster
     near 1.0, with a low tail at <0.3 from repetitive/low-complexity regions
     that share few unique k-mers with the rest of the assembly.  The default
     of 0.5 cuts the fat tail without touching the main distribution.
     Inspect the containment histogram in the rejection log before changing it.

  3. Known-contaminant screen: reject fragments with containment > MAX_CONT_HIT
     in any of: human GRCh38 (Alu/repeat source), phiX174 (sequencing spike-in),
     UniVec core (vector/adapter).  Sketches are built on first run and cached.
     The human sketch is large (~1 GB of 21-mers for the full genome); use
     --human-fasta to point at a local GRCh38 FASTA, or omit to skip this layer.

Outputs in --outdir:
  fragments_screened.fna     passing fragments
  labels_screened.tsv        labels for passing fragments
  fragments_rejected.tsv     rejected fragment IDs + reason + containment scores

Usage:
  python screen_contaminants.py \\
      --fragments data/pilot/fragments.fna \\
      --labels    data/pilot/labels.tsv \\
      --outdir    data/pilot
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# Known contaminant URLs (small; fetched automatically if not cached)
PHIX_URL = ("https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/819/615/"
            "GCF_000819615.1_ViralProj14015/"
            "GCF_000819615.1_ViralProj14015_genomic.fna.gz")
UNIVEC_URL = "https://ftp.ncbi.nlm.nih.gov/pub/UniVec/UniVec_Core"

KMER_K = 21
MAX_AMB_FRAC = 0.05
MIN_SELF_CONTAINMENT = 0.5
MAX_CONT_HIT = 0.10


# ---------------------------------------------------------------------------
# k-mer utilities
# ---------------------------------------------------------------------------

def canonical_kmer(s: str) -> str:
    """Return lexicographically smaller of kmer and its reverse complement."""
    rc = s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return s if s <= rc else rc


def kmers(seq: str, k: int = KMER_K) -> set[str]:
    """Return set of canonical k-mers from an ACGT-only (uppercased) sequence."""
    seq = seq.upper()
    out = set()
    for i in range(len(seq) - k + 1):
        w = seq[i:i + k]
        if all(c in "ACGT" for c in w):
            out.add(canonical_kmer(w))
    return out


def containment(query_kmers: set[str], ref_kmers: set[str]) -> float:
    """Jaccard containment: |query ∩ ref| / |query|."""
    if not query_kmers:
        return 0.0
    return len(query_kmers & ref_kmers) / len(query_kmers)


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

def iter_fasta(path: Path):
    """Yield (header_line, sequence) from plain or gzipped FASTA."""
    opener = gzip.open if str(path).endswith(".gz") else open
    name, chunks = None, []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        yield name, "".join(chunks)


def iter_fasta_index(path: Path) -> dict[str, int]:
    """Return dict of frag_id -> byte offset in a plain (non-gzipped) FASTA."""
    offsets = {}
    with open(path, "rb") as f:
        pos = f.tell()
        line = f.readline()
        while line:
            if line.startswith(b">"):
                fid = line[1:].strip().decode()
                offsets[fid] = pos
            pos = f.tell()
            line = f.readline()
    return offsets


def read_seq_at(path: Path, offset: int) -> str:
    with open(path, "rb") as f:
        f.seek(offset)
        f.readline()  # skip header
        chunks = []
        for line in f:
            if line.startswith(b">"):
                break
            chunks.append(line.strip())
    return b"".join(chunks).decode("ascii").upper()


# ---------------------------------------------------------------------------
# Assembly sketch building
# ---------------------------------------------------------------------------

def sketch_fasta(fasta_path: Path) -> set[str]:
    """Build a k-mer sketch (set of canonical 21-mers) from a FASTA file."""
    sketch: set[str] = set()
    for _, seq in iter_fasta(fasta_path):
        sketch |= kmers(seq.upper())
    return sketch


def load_or_build_sketch(fasta_path: Path, cache_dir: Path) -> set[str]:
    """Build + cache assembly sketch as a newline-separated text file."""
    key = hashlib.md5(str(fasta_path).encode()).hexdigest()[:12]
    cache = cache_dir / f"sketch_{key}.txt"
    if cache.exists():
        return set(cache.read_text().splitlines())
    print(f"  building sketch for {fasta_path.name} ...", file=sys.stderr)
    sk = sketch_fasta(fasta_path)
    cache.write_text("\n".join(sorted(sk)))
    return sk


# ---------------------------------------------------------------------------
# Known contaminant fetch + sketch
# ---------------------------------------------------------------------------

def _fetch(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"  fetching {url} ...", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dest


def load_contaminant_sketches(cache_dir: Path,
                               human_fasta: Path | None) -> dict[str, set[str]]:
    """Return {name: kmer_set} for each enabled contaminant source."""
    sketches: dict[str, set[str]] = {}

    # phiX174
    phix = _fetch(PHIX_URL, cache_dir / "phiX174_genomic.fna.gz")
    sketches["phiX174"] = load_or_build_sketch(phix, cache_dir)

    # UniVec_Core (plain FASTA, no gz)
    uv = _fetch(UNIVEC_URL, cache_dir / "UniVec_Core.fna")
    sketches["UniVec_Core"] = load_or_build_sketch(uv, cache_dir)

    # Human (optional — large)
    if human_fasta is not None:
        if not human_fasta.exists():
            print(f"WARNING: --human-fasta {human_fasta} not found; "
                  f"skipping human screen", file=sys.stderr)
        else:
            sketches["human_GRCh38"] = load_or_build_sketch(
                human_fasta, cache_dir)

    return sketches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fragments", required=True, type=Path,
                    help="fragments.fna from fetch_refseq_corpus.py")
    ap.add_argument("--labels", required=True, type=Path,
                    help="labels.tsv from fetch_refseq_corpus.py")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--genomes-dir", type=Path, default=None,
                    help="directory containing <accession>.fna.gz files "
                         "(default: --outdir/genomes)")
    ap.add_argument("--human-fasta", type=Path, default=None,
                    help="path to local GRCh38 FASTA for human contaminant "
                         "screen (optional; layer skipped if absent)")
    ap.add_argument("--max-amb-frac", type=float, default=MAX_AMB_FRAC)
    ap.add_argument("--min-self-containment", type=float,
                    default=MIN_SELF_CONTAINMENT,
                    help="minimum Jaccard containment of fragment 21-mers "
                         "within source assembly sketch")
    ap.add_argument("--max-cont-hit", type=float, default=MAX_CONT_HIT,
                    help="maximum containment in any known-contaminant sketch "
                         "before rejection")
    ap.add_argument("--skip-self-check", action="store_true",
                    help="skip layer 2 (self-consistency); useful for quick "
                         "contaminant-only runs")
    ap.add_argument("--skip-contaminant-check", action="store_true",
                    help="skip layer 3 (known contaminants)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "_screen_cache"
    cache_dir.mkdir(exist_ok=True)
    genomes_dir = args.genomes_dir or (args.outdir / "genomes")

    # Load labels
    labels: dict[str, dict] = {}
    with open(args.labels, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            labels[row["frag_id"]] = row
    print(f"loaded {len(labels)} fragment labels")

    # Index fragments file for random-access reads
    print("indexing fragments ...")
    offsets = iter_fasta_index(args.fragments)

    # Build per-assembly self-consistency sketches (layer 2)
    asm_sketches: dict[str, set[str]] = {}
    if not args.skip_self_check:
        accs = sorted({r["accession"] for r in labels.values()})
        print(f"building assembly sketches for {len(accs)} assemblies ...")
        for acc in accs:
            fna = genomes_dir / f"{acc}.fna.gz"
            if not fna.exists():
                print(f"  WARNING: {fna} not found; self-check skipped for {acc}",
                      file=sys.stderr)
                continue
            asm_sketches[acc] = load_or_build_sketch(fna, cache_dir)

    # Load known-contaminant sketches (layer 3)
    cont_sketches: dict[str, set[str]] = {}
    if not args.skip_contaminant_check:
        print("loading known-contaminant sketches ...")
        cont_sketches = load_contaminant_sketches(cache_dir, args.human_fasta)
        print(f"  contaminant sources: {list(cont_sketches.keys())}")

    # Screen each fragment
    pass_ids: list[str] = []
    rejected: list[dict] = []

    for fid, meta in labels.items():
        seq = read_seq_at(args.fragments, offsets[fid])
        seq_upper = seq.upper()

        # Layer 1: ambiguity gate
        amb = sum(1 for c in seq_upper if c not in "ACGT") / max(len(seq_upper), 1)
        if amb > args.max_amb_frac:
            rejected.append({"frag_id": fid, "reason": "high_ambiguity",
                             "detail": f"amb_frac={amb:.3f}"})
            continue

        frag_k = kmers(seq_upper)

        # Layer 2: self-consistency
        if not args.skip_self_check:
            acc = meta["accession"]
            asm_sk = asm_sketches.get(acc)
            if asm_sk is not None:
                sc = containment(frag_k, asm_sk)
                if sc < args.min_self_containment:
                    rejected.append({"frag_id": fid,
                                     "reason": "low_self_containment",
                                     "detail": f"self_containment={sc:.3f}"})
                    continue

        # Layer 3: known contaminants
        if not args.skip_contaminant_check:
            hit = None
            hit_score = 0.0
            for cname, csk in cont_sketches.items():
                sc = containment(frag_k, csk)
                if sc > args.max_cont_hit and sc > hit_score:
                    hit, hit_score = cname, sc
            if hit is not None:
                rejected.append({"frag_id": fid, "reason": "contaminant_hit",
                                 "detail": f"source={hit} containment={hit_score:.3f}"})
                continue

        pass_ids.append(fid)

    print(f"passed: {len(pass_ids)}  rejected: {len(rejected)}")

    # Emit passing fragments FASTA
    out_fna = args.outdir / "fragments_screened.fna"
    pass_set = set(pass_ids)
    with open(out_fna, "w") as ff:
        for fid, seq in ((fid, read_seq_at(args.fragments, offsets[fid]))
                         for fid in pass_ids):
            ff.write(f">{fid}\n")
            for i in range(0, len(seq), 70):
                ff.write(seq[i:i + 70] + "\n")
    print(f"wrote {out_fna}")

    # Emit passing labels TSV
    out_lab = args.outdir / "labels_screened.tsv"
    with open(args.labels, newline="") as src, \
         open(out_lab, "w", newline="") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames,
                                delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row["frag_id"] in pass_set:
                writer.writerow(row)
    print(f"wrote {out_lab}")

    # Emit rejection log
    out_rej = args.outdir / "fragments_rejected.tsv"
    with open(out_rej, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frag_id", "reason", "detail"],
                           delimiter="\t")
        w.writeheader()
        w.writerows(rejected)
    print(f"wrote {out_rej}")

    # Summary stats by reason
    reasons: dict[str, int] = {}
    for r in rejected:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    manifest = {
        "n_input": len(labels),
        "n_passed": len(pass_ids),
        "n_rejected": len(rejected),
        "rejection_reasons": reasons,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.outdir / "screen_manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print("rejection breakdown:", reasons)


if __name__ == "__main__":
    main()
