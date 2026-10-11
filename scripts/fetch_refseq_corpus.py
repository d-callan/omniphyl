#!/usr/bin/env python3
"""Fetch RefSeq genomes and slice simulated fragments (OmniPhyl pilot corpus).

Zero-dependency (stdlib only). Steps:
  1. Download per-group assembly summaries from NCBI RefSeq FTP.
  2. Filter to latest, full-representation assemblies with GFF annotation
     (HTTP HEAD check; assemblies without GFF are dropped and logged).
  3. Sample one assembly per phylum per group (--one-per-rank phylum)
     to maximise phylogenetic diversity without flat-count bias.
  4. Download genomic FASTA + GFF, slice fragments by nucleotide budget
     (--nt-budget bytes per assembly, size-normalised across genomes).
  5. Label each fragment: taxonomy ranks via new_taxdump rankedlineage.dmp
     + CDS annotation via GFF3 interval intersection, emitted as
     composition fractions (coding_frac in [0,1]) — mixture-aware labels
     so mixed windows keep their boundary signal (see
     docs/annotation-cascade.md label-schema note). `coding` binary
     (>=50% overlap) retained for backward compatibility.
  6. With --veupath, prefer VEuPathDB gene models where available: the
     UCSC GenArk contrib track `VEuPathDBGeneModels` (~440 assemblies,
     VEuPathDB release 68) is pulled via the UCSC JSON API and seqids are
     bridged with the hub's chromAlias.txt. Per-assembly fallback to
     RefSeq GFF. `label_source` records which tier produced each row.

Outputs in --outdir:
  genomes/<accession>.fna.gz     downloaded assemblies
  genomes/<accession>.gff.gz     downloaded GFF annotations
  fragments.fna                  simulated fragments
  labels.tsv                     frag_id, accession, seqid, start, end,
                                 length, taxid, group, coding,
                                 label_source,
                                 coding_frac, boundary, n_boundaries,
                                 boundary_offset,
                                 domain, phylum, class, order, family,
                                 genus, species
  skipped.tsv                    accessions dropped + reason
  manifest.json                  provenance (args, counts, timestamp)

Label column semantics (NA when the seqid has no GFF coverage):
  coding_frac      fraction of fragment overlapped by CDS intervals
  boundary         1 if 0 < coding_frac < 1 (window contains a CDS edge
                   or partial coverage), else 0 — the candidacy label
  n_boundaries     unique CDS interval edges strictly inside the window
  boundary_offset  position [0,1] within the window of the CDS edge
                   nearest the window centre (offset-head target)

Usage:
  python fetch_refseq_corpus.py --outdir ../data/pilot \
      --groups archaea bacteria fungi protozoa invertebrate plant \
                vertebrate_mammalian vertebrate_other viral \
      --one-per-rank phylum \
      --frag-len-min 500 --frag-len-max 500 \
      --nt-budget 100000
"""
import argparse
import csv
import gzip
import io
import json
import random
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NCBI_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq"
TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"
UCSC_API = "https://api.genome.ucsc.edu"
UCSC_HUBS = "https://hgdownload.soe.ucsc.edu/hubs"
VEUPATH_TRACK = "VEuPathDBGeneModels"  # UCSC GenArk contrib track (~440 assemblies, release 68)

GROUPS = ["archaea", "bacteria", "fungi", "invertebrate", "plant",
          "protozoa", "vertebrate_mammalian", "vertebrate_other", "viral"]

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
# rankedlineage.dmp column order (after taxid, name):
RL_COLS = ["species", "genus", "family", "order", "class", "phylum",
           "kingdom", "superkingdom"]


def download(url: str, dest: Path, retries: int = 3, timeout: int = 300) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            part = dest.parent / (dest.name + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as r, \
                    open(part, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            part.replace(dest)  # atomic: never leave a truncated cache
            return dest
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  retry {attempt+1} for {url} ({e}); wait {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    return dest


def load_summary(group: str, cache: Path) -> list[dict]:
    """Download + parse assembly_summary.txt for one RefSeq group."""
    p = cache / f"assembly_summary_{group}.txt"
    if not p.exists():
        download(f"{NCBI_BASE}/{group}/assembly_summary.txt", p)
    rows = []
    with open(p, encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            if not line.startswith("#"):
                break
            h = line.lstrip("# ")
            if h.startswith("assembly_accession"):
                header = h.rstrip("\n").split("\t")
        reader = csv.DictReader(f, fieldnames=header, delimiter="\t")
        for row in reader:
            row["group"] = group
            rows.append(row)
    return rows


def has_gff(row: dict, timeout: int = 10) -> bool:
    """HEAD-check whether the GFF file exists on the FTP mirror."""
    ftp_dir = row["ftp_path"].replace("ftp://", "https://").rstrip("/")
    name = ftp_dir.rsplit("/", 1)[-1]
    url = f"{ftp_dir}/{name}_genomic.gff.gz"
    for a in range(3):
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout):
                return True
        except Exception:
            # NCBI throttles concurrent HEAD bursts; back off and retry
            time.sleep(2 * (a + 1))
    return False


def select_assemblies(rows: list[dict], levels: set[str],
                      one_per_rank: str | None,
                      rng: random.Random,
                      skipped: list[dict],
                      workers: int,
                      tax_resolver=None,
                      cache: Path | None = None) -> list[dict]:
    """Filter to eligible assemblies, HEAD-check for GFF, then sample.

    With one_per_rank set (e.g. 'phylum'): resolve each candidate's
    species_taxid through rankedlineage.dmp (downloaded once into cache)
    and keep one assembly per distinct rank value. Taxa without a value
    for that rank each form their own group so nothing is silently
    dropped; the label columns still come from the same taxdump.
    """
    keep = [r for r in rows
            if r.get("assembly_level") in levels
            and r.get("version_status") == "latest"
            and r.get("genome_rep") == "Full"
            and r.get("excluded_from_refseq", "na") in ("", "na")
            and r.get("ftp_path", "").startswith(("ftp://", "https://"))]
    rng.shuffle(keep)

    # GFF HEAD checks in parallel
    print(f"  checking GFF availability for {len(keep)} candidates ...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        gff_flags = list(ex.map(has_gff, keep))
    survivors = []
    for r, ok in zip(keep, gff_flags):
        if ok:
            survivors.append(r)
        else:
            skipped.append({"accession": r["assembly_accession"],
                            "group": r["group"], "reason": "no_gff"})
    keep = survivors

    if one_per_rank:
        taxids = {r.get("species_taxid") for r in keep
                  if r.get("species_taxid")}
        tax_map = tax_resolver(taxids, cache) if tax_resolver else {}
        groups: dict[str, list] = defaultdict(list)
        for r in keep:
            tid = r.get("species_taxid", "")
            rank_val = tax_map.get(tid, {}).get(one_per_rank)
            # missing rank value -> unique group so the taxon still samples
            groups[rank_val or f"_unresolved_{tid}"].append(r)
        keep = [grp[0] for grp in groups.values()]
        rng.shuffle(keep)

    return keep


def fetch_fasta_and_gff(row: dict, dest_dir: Path) -> tuple[Path, Path]:
    ftp_dir = row["ftp_path"].replace("ftp://", "https://").rstrip("/")
    name = ftp_dir.rsplit("/", 1)[-1]
    acc = row["assembly_accession"]
    fna = dest_dir / f"{acc}.fna.gz"
    gff = dest_dir / f"{acc}.gff.gz"
    if not fna.exists():
        download(f"{ftp_dir}/{name}_genomic.fna.gz", fna)
    if not gff.exists():
        download(f"{ftp_dir}/{name}_genomic.gff.gz", gff)
    return fna, gff


def load_gff_cds(gff_path: Path) -> dict[str, list[tuple[int, int]]]:
    """seqid -> sorted list of (start, end) 0-based half-open CDS intervals."""
    cds: dict[str, list] = defaultdict(list)
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            seqid = parts[0]
            # GFF3 coords are 1-based inclusive; convert to 0-based half-open
            start = int(parts[3]) - 1
            end = int(parts[4])
            cds[seqid].append((start, end))
    for intervals in cds.values():
        intervals.sort()
    return cds


def coding_stats(seqid: str, frag_start: int, frag_end: int,
                 cds: dict[str, list[tuple[int, int]]]):
    """Return (cds_frac, n_boundaries, boundary_offset) or None.

    cds_frac: fraction of the window covered by CDS intervals.
    n_boundaries: unique CDS edges strictly inside the window.
    boundary_offset: normalized [0,1] position of the edge nearest the
    window centre, or None when no edge is strictly inside. None is
    returned when the seqid has no coverage.
    """
    intervals = cds.get(seqid)
    if intervals is None:
        return None
    return coding_stats_iv(intervals, frag_start, frag_end)


def coding_stats_iv(intervals: list[tuple[int, int]],
                    frag_start: int, frag_end: int):
    """coding_stats over a bare interval list."""
    flen = frag_end - frag_start
    mid = frag_start + flen / 2
    overlap = 0
    # intervals are sorted by start; cursor marks the rightmost covered base
    # so overlapping entries (alt transcripts, overlapping genes) count once
    covered_until = frag_start
    edges: set[int] = set()
    for s, e in intervals:
        if e <= frag_start:
            continue
        if s >= frag_end:
            break
        s2, e2 = max(s, frag_start), min(e, frag_end)
        if e2 > covered_until:
            overlap += e2 - max(s2, covered_until)
            covered_until = e2
        if frag_start < s < frag_end:
            edges.add(s)
        if frag_start < e < frag_end:
            edges.add(e)
    frac = overlap / flen
    boff = ((min(edges, key=lambda p: abs(p - mid)) - frag_start) / flen
            if edges else None)
    return frac, len(edges), boff


# --- VEuPathDB label tier (UCSC GenArk contrib tracks) -------------------
#
# UCSC mirrors VEuPathDB gene models as `contrib/VEuPathDB/veupathGenes.bb`
# (track name VEuPathDBGeneModels, bigBed 12, VEuPathDB release 68) on
# ~440 GenArk assembly hubs. bigBed is binary, so data comes through the
# UCSC JSON API (stdlib-parseable); sequence-name bridging uses the hub's
# chromAlias.txt (VEuPathDB / GenBank / RefSeq names differ).

def hub_url(acc: str) -> str:
    """GenArk hub directory URL for an assembly accession."""
    prefix, rest = acc.split("_", 1)
    num = rest.split(".")[0]
    chunks = "/".join(num[i:i + 3] for i in range(0, len(num), 3))
    return f"{UCSC_HUBS}/{prefix}/{chunks}/{acc}"


def http_json(url: str, timeout: int = 60):
    req = urllib.request.Request(
        url, headers={"User-Agent": "omniphyl-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _collect_bed_items(data) -> list[dict]:
    """Walk nested UCSC API JSON; collect bigBed item dicts."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            if "chromStart" in x or ("chrom" in x and "chromEnd" in x):
                out.append(x)
                return
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return out


def _ucsc_gtf_url(acc: str):
    """veupathGenes.gtf.gz URL via list/files, or None.

    Mirrors the brc-analytics GTFStep discovery pattern
    (api.genome.ucsc.edu/list/files -> urlList -> .gtf.gz filter); the
    file lives under contrib/VEuPathDB/ or genes/ depending on the hub.
    """
    try:
        data = http_json(f"{UCSC_API}/list/files?genome={acc}")
    except Exception:
        return None
    for entry in data.get("urlList", []):
        url = entry.get("url", "")
        if "veupathGenes.gtf.gz" in url:
            if not url.startswith("http"):
                url = f"https://hgdownload.soe.ucsc.edu/{url}"
            return url
    return None


def fetch_veupathdb_cds(acc: str, cache: Path):
    """{track_chrom: [(s,e),...]} from VEuPathDBGeneModels, or None.

    Two sources, in order: (a) the veupathGenes.gtf.gz file when the hub
    ships one (parsed with the same CDS-line logic as RefSeq GFF);
    (b) the bigBed track items via getData/track — CDS = blocks clipped
    to thick bounds (drops UTR). Returns None when the assembly hub has
    neither; caller falls back to RefSeq GFF.
    """
    gtf_url = _ucsc_gtf_url(acc)
    if gtf_url:
        gtf = cache / f"{acc}.veupathGenes.gtf.gz"
        try:
            if not gtf.exists():
                download(gtf_url, gtf)
            cds = load_gff_cds(gtf)
            if cds:
                return cds
        except Exception:
            pass
    p = cache / f"{acc}.veupath_genes.json"
    if p.exists():
        data = json.loads(p.read_text())
        if not data:
            return None
    else:
        try:
            tracks = http_json(f"{UCSC_API}/list/tracks?genome={acc}")
        except Exception:
            return None
        genome_tracks = tracks.get(acc, tracks)
        if VEUPATH_TRACK not in genome_tracks:
            return None
        try:
            data = http_json(
                f"{UCSC_API}/getData/track?genome={acc}"
                f";track={VEUPATH_TRACK};maxItemsOutput=1000000")
        except Exception:
            return None
        p.write_text(json.dumps(data))
    cds: dict[str, list] = defaultdict(list)
    for it in _collect_bed_items(data):
        chrom = it.get("chrom")
        ts = int(it.get("thickStart", it.get("chromStart", 0)))
        te = int(it.get("thickEnd", it.get("chromEnd", 0)))
        base = int(it.get("chromStart", 0))
        sizes = [int(x) for x in
                 str(it.get("blockSizes", "")).strip(",").split(",") if x]
        starts = [int(x) for x in
                  str(it.get("chromStarts", "")).strip(",").split(",") if x]
        for st, sz in zip(starts, sizes):
            s, e = max(base + st, ts), min(base + st + sz, te)
            if e > s:
                cds[chrom].append((s, e))
    for v in cds.values():
        v.sort()
    return dict(cds) if cds else None


def load_chrom_alias(acc: str, cache: Path) -> dict[str, frozenset]:
    """any alias -> frozenset of aliases on its chromAlias row."""
    p = cache / f"{acc}.chromAlias.txt"
    if not p.exists():
        try:
            download(f"{hub_url(acc)}/{acc}.chromAlias.txt", p)
        except Exception:
            return {}
    groups: dict[str, frozenset] = {}
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = [c for c in line.rstrip("\n").split("\t") if c]
            if len(cols) < 2:
                continue
            grp = frozenset(cols)
            for a in cols:
                groups[a] = grp
    return groups


def veupath_intervals(veu: dict, alias_groups: dict, seqid: str):
    """CDS interval list for a FASTA seqid via chromAlias bridging."""
    for a in {seqid} | alias_groups.get(seqid, frozenset()):
        if a in veu:
            return veu[a]
    return None


def coding_label(seqid: str, frag_start: int, frag_end: int,
                 cds: dict[str, list[tuple[int, int]]],
                 min_overlap_frac: float = 0.5) -> str:
    """Return 'coding', 'non-coding', or 'NA' (seqid not in GFF)."""
    stats = coding_stats(seqid, frag_start, frag_end, cds)
    if stats is None:
        return "NA"
    return "coding" if stats[0] >= min_overlap_frac else "non-coding"


def iter_fasta_gz(path: Path):
    name, seq = None, []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    yield name, "".join(seq)
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line)
    if name:
        yield name, "".join(seq)


def slice_fragments(seq: str, lo: int, hi: int,
                    nt_budget: int, max_amb_frac: float,
                    rng: random.Random):
    """Yield (start, end, subseq) windows up to nt_budget total nucleotides.

    Windows are sampled uniformly weighted by position across the sequence
    so every assembly contributes the same total nucleotides regardless of
    genome size (size-normalisation fix for genome-size bias).
    """
    L = len(seq)
    if L < lo:
        return
    emitted = 0
    attempts = 0
    max_attempts = (nt_budget // lo) * 4  # generous oversample budget
    while emitted < nt_budget and attempts < max_attempts:
        attempts += 1
        flen = rng.randint(lo, min(hi, L))
        s = rng.randint(0, L - flen)
        frag = seq[s:s + flen].upper()
        amb = sum(1 for c in frag if c not in "ACGT") / flen
        if amb <= max_amb_frac:
            yield s, s + flen, frag
            emitted += flen
            if emitted >= nt_budget:
                return


def resolve_taxonomy(taxids: set[str], cache: Path) -> dict:
    """taxid -> {rank: name} via new_taxdump rankedlineage.dmp."""
    tar_path = cache / "new_taxdump.tar.gz"
    if not tar_path.exists():
        print("downloading new_taxdump (~60MB)...")
        download(TAXDUMP_URL, tar_path)
    out = {}
    with tarfile.open(tar_path, "r:gz") as tf:
        member = next(m for m in tf if m.name.endswith("rankedlineage.dmp"))
        with tf.extractfile(member) as f:
            for raw in io.TextIOWrapper(f, encoding="utf-8",
                                        errors="replace"):
                cols = [c.strip() for c in raw.split("|")]
                tid = cols[0]
                if tid not in taxids:
                    continue
                rec = {}
                for rname, val in zip(RL_COLS, cols[2:10]):
                    if rname == "superkingdom":
                        rec["domain"] = val or None
                    elif rname != "kingdom":
                        rec[rname] = val or None
                rec["species"] = rec.get("species") or cols[1]
                out[tid] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--groups", nargs="+", default=["bacteria", "viral"],
                    choices=GROUPS)
    ap.add_argument("--levels", default="Complete Genome,Chromosome")
    ap.add_argument("--one-per-rank", metavar="RANK", default=None,
                    help="sample one assembly per distinct value of this rank "
                         "(e.g. 'phylum') to maximise phylogenetic diversity")
    ap.add_argument("--nt-budget", type=int, default=100_000,
                    help="nucleotide budget per assembly (total bases sampled); "
                         "size-normalises fragment counts across genome sizes")
    ap.add_argument("--frag-len-min", type=int, default=500)
    ap.add_argument("--frag-len-max", type=int, default=500)
    ap.add_argument("--max-amb-frac", type=float, default=0.05)
    ap.add_argument("--boundary-frac", type=float, default=0.0,
                    help="target fraction of emitted fragments containing a "
                         "CDS boundary (mixed windows); 0 = uniform sampling. "
                         "Boundary windows are rare under uniform sampling, "
                         "which starves the candidacy head of positives")
    ap.add_argument("--veupath", action="store_true",
                    help="prefer VEuPathDB gene models (UCSC GenArk contrib "
                         "track VEuPathDBGeneModels) over RefSeq GFF when "
                         "available; falls back per-assembly")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache = args.outdir / "_ncbi_cache"
    gdir = args.outdir / "genomes"
    levels = set(args.levels.split(","))
    skipped: list[dict] = []

    # 1-2. summaries + selection (with GFF HEAD-check pre-filter)
    selected = []
    for g in args.groups:
        rows = load_summary(g, cache)
        sel = select_assemblies(rows, levels, args.one_per_rank,
                                rng, skipped, args.workers,
                                tax_resolver=resolve_taxonomy,
                                cache=cache)
        print(f"{g}: {len(rows)} assemblies -> {len(sel)} selected "
              f"(after GFF filter)")
        selected.extend(sel)

    # 3. download FASTAs + GFFs (parallel)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_fasta_and_gff, r, gdir): r for r in selected}
        ok = []
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                row["_fasta"], row["_gff"] = fut.result()
                ok.append(row)
            except Exception as e:
                print(f"  FAILED {row['assembly_accession']}: {e}",
                      file=sys.stderr)
                skipped.append({"accession": row["assembly_accession"],
                                "group": row["group"],
                                "reason": f"download_error: {e}"})
            if i % 50 == 0:
                print(f"  downloaded {i}/{len(selected)}")
    print(f"downloaded {len(ok)}/{len(selected)} genomes")

    # 4. taxonomy labels
    taxids = {r["species_taxid"] for r in ok if r.get("species_taxid")}
    tax_map = resolve_taxonomy(taxids, cache)

    # 5. fragment + emit
    frag_path = args.outdir / "fragments.fna"
    lab_path = args.outdir / "labels.tsv"
    skip_path = args.outdir / "skipped.tsv"
    n_frags = 0
    with open(frag_path, "w") as ff, open(lab_path, "w", newline="") as lf:
        w = csv.writer(lf, delimiter="\t")
        w.writerow(["frag_id", "accession", "seqid", "start", "end",
                    "length", "taxid", "group", "coding", "label_source",
                    "coding_frac", "boundary", "n_boundaries",
                    "boundary_offset"] + RANKS)
        for row in ok:
            tax = tax_map.get(row.get("species_taxid", ""), {})
            veu = (fetch_veupathdb_cds(row["assembly_accession"], cache)
                   if args.veupath else None)
            alias_groups = (load_chrom_alias(row["assembly_accession"], cache)
                            if veu else {})
            label_source = "veupathdb" if veu else "refseq_gff"
            cds = load_gff_cds(row["_gff"])
            n_veu_unmatched = 0
            for seqid, seq in iter_fasta_gz(row["_fasta"]):
                if veu:
                    iv = veupath_intervals(veu, alias_groups, seqid)
                    if iv is None:
                        n_veu_unmatched += 1
                        stats_of = lambda s, e: None
                    else:
                        stats_of = lambda s, e: coding_stats_iv(iv, s, e)
                else:
                    stats_of = lambda s, e: coding_stats(seqid, s, e, cds)
                cands = [(s, e, frag, stats_of(s, e))
                         for s, e, frag in slice_fragments(
                             seq, args.frag_len_min, args.frag_len_max,
                             args.nt_budget, args.max_amb_frac, rng)]
                # Boundary stratification: mixed windows (0 < frac < 1)
                # are rare under uniform sampling but carry the candidacy
                # signal; subsample each bucket to hit --boundary-frac.
                if args.boundary_frac > 0:
                    mixed = [c for c in cands
                             if c[3] is not None and 0 < c[3][0] < 1]
                    pure = [c for c in cands
                            if c[3] is None or not 0 < c[3][0] < 1]
                    n_b = min(len(mixed),
                              round(args.boundary_frac * len(cands)))
                    n_nb = min(len(pure), len(cands) - n_b)
                    cands = rng.sample(mixed, n_b) + rng.sample(pure, n_nb)
                for s, e, frag, stats in cands:
                    fid = f"{row['assembly_accession']}|{seqid}|{s}-{e}"
                    ff.write(f">{fid}\n")
                    for i in range(0, len(frag), 70):
                        ff.write(frag[i:i + 70] + "\n")
                    if stats is None:
                        frac_s, bnd_s, nb_s, boff_s = "", "", "", ""
                    else:
                        frac, n_bnd, boff = stats
                        frac_s = f"{frac:.4f}"
                        bnd_s = "1" if 0 < frac < 1 else "0"
                        nb_s = str(n_bnd)
                        boff_s = f"{boff:.4f}" if boff is not None else ""
                    cod = ("NA" if stats is None else
                           "coding" if stats[0] >= 0.5 else "non-coding")
                    w.writerow([fid, row["assembly_accession"], seqid, s, e,
                                len(frag), row.get("species_taxid", ""),
                                row["group"], cod, label_source,
                                frac_s, bnd_s, nb_s, boff_s] +
                               [tax.get(r, "") or "" for r in RANKS])
                    n_frags += 1
            if veu and n_veu_unmatched:
                print(f"  {row['assembly_accession']}: "
                      f"{n_veu_unmatched} seqids unmatched to VEuPathDB "
                      f"chrom names (NA labels)", file=sys.stderr)
    print(f"wrote {n_frags} fragments -> {frag_path}")

    with open(skip_path, "w", newline="") as sf:
        sw = csv.DictWriter(sf, fieldnames=["accession", "group", "reason"],
                            delimiter="\t")
        sw.writeheader()
        sw.writerows(skipped)
    print(f"skipped {len(skipped)} assemblies -> {skip_path}")

    manifest = {
        "args": vars(args) | {"outdir": str(args.outdir)},
        "groups": args.groups,
        "genomes_downloaded": len(ok),
        "genomes_skipped": len(skipped),
        "fragments": n_frags,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "sources": {
            "summaries": f"{NCBI_BASE}/<group>/assembly_summary.txt",
            "taxonomy": TAXDUMP_URL + " (rankedlineage.dmp)",
        },
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
