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
  genomes/<accession>.fna.gz     transient: deleted after slicing
  genomes/<accession>.gff.gz     transient: deleted after slicing
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
import errno
import gzip
import io
import json
import random
import re
import sys
import tarfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

NCBI_BASE = {
    "refseq": "https://ftp.ncbi.nlm.nih.gov/genomes/refseq",
    "genbank": "https://ftp.ncbi.nlm.nih.gov/genomes/genbank",
}
TAXDUMP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"
UCSC_API = "https://api.genome.ucsc.edu"
UCSC_HUBS = "https://hgdownload.soe.ucsc.edu/hubs"
VEUPATH_TRACK = "VEuPathDBGeneModels"  # UCSC GenArk contrib track (~440 assemblies, release 68)

GROUPS = ["archaea", "bacteria", "fungi", "invertebrate", "plant",
          "protozoa", "vertebrate_mammalian", "vertebrate_other", "viral"]

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]

# Within a rank group, picks prefer more complete assemblies; Scaffold
# (and Contig) members only fill slots after better levels exhaust —
# completeness is nice-to-have for fragment slicing, not a hard gate.
LEVEL_PREF = {"Complete Genome": 0, "Chromosome": 1,
              "Scaffold": 2, "Contig": 3}
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
            # a full disk can't be fixed by retrying; fail fast
            if getattr(e, "errno", None) == errno.ENOSPC:
                raise
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt * 5
            print(f"  retry {attempt+1} for {url} ({e}); wait {wait}s",
                  file=sys.stderr)
            time.sleep(wait)
    return dest


def load_summary(group: str, cache: Path,
                 source: str = "refseq") -> list[dict]:
    """Download + parse assembly_summary.txt for one group.

    source selects the RefSeq or GenBank FTP tree (identical file
    format; GenBank rows carry gbrs_paired_asm pointing at any RefSeq
    twin, which main() uses to keep only genuinely new assemblies).
    """
    p = cache / f"assembly_summary_{source}_{group}.txt"
    if not p.exists():
        download(f"{NCBI_BASE[source]}/{group}/assembly_summary.txt", p)
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
                      cache: Path | None = None,
                      per_rank_n: int = 1,
                      require_gff: bool = True,
                      covered: set | None = None,
                      covered_n: int = 0,
                      mark_unresolved: bool = True) -> list[dict]:
    """Filter to eligible assemblies, then sample.

    covered/covered_n implement gap-filling from a second source: rank
    keys already picked by a previous call (each kept row is tagged
    _rank_key) get at most covered_n picks instead of per_rank_n, so a
    GenBank pass fills only values RefSeq missed (covered_n=0) or adds
    a bounded supplement. mark_unresolved=False keeps supplementary
    picks train-only: they never get the _rank_unresolved test-only
    flag.

    require_gff gates a HEAD sweep dropping picks without a RefSeq GFF;
    with it off, picks are taken directly and GFF-missing genomes train
    with annotation labels masked (taxonomy labels unaffected).

    With one_per_rank set (e.g. 'phylum'): resolve each candidate's
    species_taxid through rankedlineage.dmp (downloaded once into cache)
    and keep up to per_rank_n assemblies per distinct rank value. N>1
    keeps rank vocab shared between train and held-out genomes, so
    generalization is measurable at every rank.    Taxa without a value
    for that rank share a single fallback group — so groups like viral
    (where most taxids lack a phylum in NCBI taxonomy) don't explode
    into thousands of singletons. Fallback picks are flagged
    _rank_unresolved: they stay in the corpus as test-only genomes
    (train_omniphyl.build_split forces them into val).
    Grouping happens BEFORE the GFF HEAD sweep so only the picks (~one
    per rank value) get checked, not every candidate (bacteria alone
    has ~75k); a failed pick walks its group for an alternate. Each
    group is ordered by assembly level so Scaffold members fill slots
    only after Complete/Chromosome members are exhausted.
    """
    keep = [r for r in rows
            if r.get("assembly_level") in levels
            and r.get("version_status") == "latest"
            and r.get("genome_rep") == "Full"
            and r.get("excluded_from_refseq", "na") in ("", "na")
            and r.get("ftp_path", "").startswith(("ftp://", "https://"))]
    rng.shuffle(keep)

    if one_per_rank:
        taxids = {r.get("species_taxid") for r in keep
                  if r.get("species_taxid")}
        tax_map = tax_resolver(taxids, cache) if tax_resolver else {}
        # A taxid missing the requested rank is either a GAP on an
        # otherwise-resolved lineage (protists without a phylum) or a
        # genuinely UNPLACED taxon (new virus with a species name and
        # almost nothing above it). Gaps train normally (their missing
        # label is masked) and group by the nearest resolved neighbour
        # rank for diversity; unplaced taxa (<=2 of the 6 lineage ranks
        # resolved) pool under _unresolved and are flagged test-only.
        rk_order = ["species", "genus", "family", "order", "class",
                    "phylum", "domain"]
        i_req = rk_order.index(one_per_rank) if one_per_rank in rk_order else -1
        lin_ranks = [rk for rk in rk_order[1:] if rk != "species"]
        scan_order = sorted((rk for rk in lin_ranks if rk != one_per_rank),
                            key=lambda rk: abs(rk_order.index(rk) - i_req))
        groups: dict[str, list] = defaultdict(list)
        for r in keep:
            tid = r.get("species_taxid", "")
            lin = tax_map.get(tid, {})
            rank_val = lin.get(one_per_rank)
            if not rank_val:
                resolved = sum(1 for rk in lin_ranks if lin.get(rk))
                if resolved <= 2:
                    groups["_unresolved"].append(r)
                    continue
                near = next((lin[rk] for rk in scan_order if lin.get(rk)),
                            None)
                groups[f"_gap:{near}" if near else "_unresolved"].append(r)
                continue
            groups[rank_val].append(r)
        # prefer better assemblies within each rank value: the pre-group
        # shuffle already randomised order, so a stable sort by level
        # keeps randomness inside a level while preferring completeness
        candidates = []
        for key, grp in groups.items():
            grp.sort(key=lambda r: LEVEL_PREF.get(
                r.get("assembly_level", ""), 9))
            candidates.append((key, grp))
    else:
        candidates = [(None, [r]) for r in keep]

    covered = covered or set()

    def quota(key):
        return covered_n if key in covered else per_rank_n

    def members_with_gff(grp, q):
        # up to q GFF-available members, scanning a couple of
        # alternates so dead FTP dirs don't starve the rank value
        got = []
        for r in grp[:q + 2]:
            if has_gff(r):
                got.append(r)
                if len(got) == q:
                    break
        return got

    # drop rank keys whose gap-fill quota is 0 before the GFF sweep
    candidates = [c for c in candidates if quota(c[0]) > 0]

    if require_gff:
        # GFF HEAD checks in parallel over picks (or all rows ungrouped)
        print(f"  checking GFF availability for {len(candidates)} picks ...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            picks = list(ex.map(lambda c: members_with_gff(c[1],
                                                         quota(c[0])),
                                candidates))
    else:
        picks = [grp[:quota(key)] for key, grp in candidates]
    keep = []
    for (key, grp), got in zip(candidates, picks):
        if not got:
            skipped.append({"accession": grp[0]["assembly_accession"],
                            "group": grp[0]["group"], "reason": "no_gff"})
            continue
        for r in got:
            r["_rank_key"] = key
        if key == "_unresolved" and mark_unresolved:
            for r in got:
                r["_rank_unresolved"] = "1"
        keep.extend(got)

    return keep


def fetch_fasta_and_gff(row: dict, dest_dir: Path,
                        require_gff: bool = True):
    ftp_dir = row["ftp_path"].replace("ftp://", "https://").rstrip("/")
    name = ftp_dir.rsplit("/", 1)[-1]
    acc = row["assembly_accession"]
    fna = dest_dir / f"{acc}.fna.gz"
    gff = dest_dir / f"{acc}.gff.gz"
    if not fna.exists():
        download(f"{ftp_dir}/{name}_genomic.fna.gz", fna)
    if not gff.exists():
        try:
            download(f"{ftp_dir}/{name}_genomic.gff.gz", gff)
        except Exception:
            if require_gff:
                raise
            gff = None
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


# --- rich GFF features (P3/E3-E5) ---------------------------------------
# Single-pass parse retaining CDS strand/phase/Parent/Dbxref so windows
# can be labelled with region composition, active reading frame,
# structural-transition flags (start/stop/splice sites) and Pfam/InterPro
# domains. All intervals are 0-based half-open.

FEAT_TYPES = {"CDS": "cds", "exon": "exon",
              "mRNA": "mrna", "transcript": "mrna",
              "rRNA": "rna", "tRNA": "rna"}
DOMAIN_RE = re.compile(r"(?:InterPro:IPR\d+|(?:PFAM|Pfam):PF\d+)")


def load_gff_features(gff_path: Path) -> dict[str, dict]:
    """seqid -> {cds, exon, mrna, rna interval lists; cds_recs}.

    cds_recs entries are (start, end, strand, phase, parent, [domains]).
    Superset of load_gff_cds; the CDS interval list is derivable via
    feats[seqid]["cds"].
    """
    feats: dict[str, dict] = defaultdict(
        lambda: {"cds": [], "exon": [], "mrna": [], "rna": [],
                 "cds_recs": []})
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in FEAT_TYPES:
                continue
            seqid, ftype = parts[0], FEAT_TYPES[parts[2]]
            start, end = int(parts[3]) - 1, int(parts[4])
            feats[seqid][ftype].append((start, end))
            if ftype == "cds":
                strand, phase = parts[6], parts[7]
                attrs = parts[8]
                parent, domains = "", []
                for tok in attrs.split(";"):
                    if tok.startswith(("Parent=", "protein_id=")):
                        parent = tok.split("=", 1)[1]
                    elif tok.startswith("Dbxref="):
                        domains += DOMAIN_RE.findall(tok)
                try:
                    phase = int(phase)
                except ValueError:
                    phase = 0
                feats[seqid]["cds_recs"].append(
                    (start, end, strand, phase, parent, domains))
    for f in feats.values():
        for k in ("cds", "exon", "mrna", "rna"):
            f[k].sort()
    return feats


def overlap_bases(intervals: list[tuple[int, int]],
                  lo: int, hi: int) -> int:
    """Union-coverage of sorted intervals within [lo, hi)."""
    cov, cur = 0, lo
    for s, e in intervals:
        if e <= lo:
            continue
        if s >= hi:
            break
        s2, e2 = max(s, lo), min(e, hi)
        if e2 > cur:
            cov += e2 - max(s2, cur)
            cur = e2
    return cov


def subtract_intervals(outer: list[tuple[int, int]],
                       inner: list[tuple[int, int]]):
    """outer minus merged(inner); both sorted, 0-based half-open."""
    out = []
    i, cur_inner_end = 0, -1
    merged = []
    for s, e in inner:
        if s > cur_inner_end:
            merged.append([s, e])
            cur_inner_end = e
        else:
            merged[-1][1] = cur_inner_end = max(cur_inner_end, e)
    for s, e in outer:
        cursor = s
        while i < len(merged) and merged[i][1] <= cursor:
            i += 1
        j = i
        while j < len(merged) and merged[j][0] < e:
            ms, me = merged[j]
            if ms > cursor:
                out.append((cursor, min(ms, e)))
            cursor = max(cursor, me)
            if cursor >= e:
                break
            j += 1
        if cursor < e:
            out.append((cursor, e))
    return out


def cds_frame(rec_start: int, rec_end: int, strand: str, phase: int,
              frag_start: int, frag_end: int) -> int:
    """0-5 frame index of a CDS record relative to window [fs, fe),
    aligned with aa_encode's 6 frames (0-2 forward offsets, 3-5 reverse).
    """
    if strand == "-":
        codon0 = rec_end - ((3 - phase) % 3)
        return 3 + (frag_end - codon0) % 3
    codon0 = rec_start + ((3 - phase) % 3)
    return (codon0 - frag_start) % 3


def feature_stats(feats: dict, seqid: str, frag_start: int,
                  frag_end: int):
    """Rich label tuple for a window, or None if seqid is unannotated.

    Returns dict: frac, n_bnd, boff (as coding_stats), intron, rna,
    frame (0-5 dominant frame, 6 multi, 7 non-coding), flags (4-bit
    start/stop/splice-donor/splice-acceptor mask), domains (IPR/PF ids
    on CDS overlapping the window by >=90bp).
    """
    f = feats.get(seqid)
    if f is None:
        return None
    flen = frag_end - frag_start
    frac, n_bnd, boff = coding_stats_iv(f["cds"], frag_start, frag_end)
    intron_iv = subtract_intervals(f["mrna"], f["exon"]) if f["mrna"] else []
    intron = (overlap_bases(intron_iv, frag_start, frag_end) / flen
              if intron_iv else 0.0)
    rna = overlap_bases(f["rna"], frag_start, frag_end) / flen

    # active frame: coding coverage per 6-frame index. Parent->exon
    # groups span ALL CDS records (not just window-overlapping ones):
    # a multi-exon CDS's true start/stop is the outermost edge of the
    # whole transcript, and splice sites are between consecutive exons —
    # restricting the group to visible segments would fabricate
    # start/stop flags at internal CDS boundaries.
    cov = [0] * 6
    flags = 0
    domains: set[str] = set()
    by_parent: dict[str, list] = defaultdict(list)
    for s, e, strand, phase, parent, doms in f["cds_recs"]:
        by_parent[(parent or f"{s}-{e}-{strand}", strand)].append((s, e))
        if e <= frag_start or s >= frag_end:
            continue
        clip = min(e, frag_end) - max(s, frag_start)
        cov[cds_frame(s, e, strand, phase, frag_start, frag_end)] += clip
        if clip >= 90:
            domains.update(doms)
    coding_bases = sum(cov)
    if frac < 0.1:
        frame = 7
    else:
        top = max(range(6), key=lambda i: cov[i])
        frame = top if cov[top] >= 0.5 * coding_bases else 6

    # structural transitions: group CDS exons by (parent, strand);
    # transcript order is reversed on the - strand
    for (parent, strand), exs in by_parent.items():
        exs.sort()
        exons = exs if strand != "-" else exs[::-1]
        start_pos = exons[0][0] if strand != "-" else exons[0][1]
        stop_pos = exons[-1][1] if strand != "-" else exons[-1][0]
        if frag_start < start_pos < frag_end:
            flags |= 1
        if frag_start < stop_pos < frag_end:
            flags |= 2
        for i in range(len(exons) - 1):
            donor = exons[i][1] if strand != "-" else exons[i][0]
            acceptor = exons[i + 1][0] if strand != "-" else exons[i + 1][1]
            if frag_start < donor < frag_end:
                flags |= 4
            if frag_start < acceptor < frag_end:
                flags |= 8
    return {"frac": frac, "n_bnd": n_bnd, "boff": boff,
            "intron": intron, "rna": rna, "frame": frame,
            "flags": flags, "domains": sorted(domains)}


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


def slice_assembly(row: dict, gdir: Path, args, cache: Path):
    """Fetch one assembly, slice its nt-budgeted fragments, delete sources.

    Genome FASTAs/GFFs are transient per accession: nothing downstream
    reads genomes/, and hoarding thousands of assemblies overflows the
    ~20GB /kaggle/working quota.  Returns
    (row, label_source, [(seqid, s, e, frag, stats)], n_veu_unmatched).
    Each accession seeds its own rng (f"{seed}:{acc}") so fragment draws
    are independent of selection order and worker scheduling.
    """
    acc = row["assembly_accession"]
    try:
        fna, gff = fetch_fasta_and_gff(row, gdir, args.require_gff)
        veu = (fetch_veupathdb_cds(acc, cache) if args.veupath else None)
        alias_groups = load_chrom_alias(acc, cache) if veu else {}
        label_source = ("veupathdb" if veu else
                        "refseq_gff" if gff else "none")
        feats = load_gff_features(gff) if gff else {}
        rng = random.Random(f"{args.seed}:{acc}")
        acc_budget = row.get("_nt_budget", args.nt_budget)
        # --nt-budget is per ACCESSION; each seqid draws a
        # length-proportional share so every nucleotide position is
        # equally likely to be sampled regardless of contig count.
        seq_lens = {s: len(q) for s, q in iter_fasta_gz(fna)}
        total_len = sum(seq_lens.values())
        frags = []
        n_veu_unmatched = 0
        for seqid, seq in iter_fasta_gz(fna):
            seq_budget = round(acc_budget * len(seq) / total_len)
            if veu:
                iv = veupath_intervals(veu, alias_groups, seqid)
                if iv is None:
                    n_veu_unmatched += 1
                    stats_of = lambda s, e: None
                else:
                    stats_of = lambda s, e: coding_stats_iv(iv, s, e)
            else:
                stats_of = lambda s, e: feature_stats(feats, seqid, s, e)
            cands = [(s, e, frag, stats_of(s, e))
                     for s, e, frag in slice_fragments(
                         seq, args.frag_len_min, args.frag_len_max,
                         seq_budget, args.max_amb_frac, rng)]
            # Boundary stratification: mixed windows (0 < frac < 1) are
            # rare under uniform sampling but carry the candidacy signal;
            # subsample each bucket to hit --boundary-frac. stats may be a
            # feature_stats dict (refseq tier) or a (frac, n_bnd, boff)
            # tuple (veupath tier).
            if args.boundary_frac > 0:
                def _frac(st):
                    return None if st is None else \
                        (st["frac"] if isinstance(st, dict) else st[0])
                mixed = [c for c in cands
                         if _frac(c[3]) is not None and
                         0 < _frac(c[3]) < 1]
                pure = [c for c in cands
                        if _frac(c[3]) is None or
                        not 0 < _frac(c[3]) < 1]
                n_b = min(len(mixed),
                          round(args.boundary_frac * len(cands)))
                n_nb = min(len(pure), len(cands) - n_b)
                cands = rng.sample(mixed, n_b) + rng.sample(pure, n_nb)
            frags.extend((seqid, s, e, frag, stats)
                         for s, e, frag, stats in cands)
        return row, label_source, frags, n_veu_unmatched
    finally:
        for name in (f"{acc}.fna.gz", f"{acc}.fna.gz.part",
                     f"{acc}.gff.gz", f"{acc}.gff.gz.part"):
            (gdir / name).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--groups", nargs="+", default=["bacteria", "viral"],
                    choices=GROUPS)
    ap.add_argument("--levels", default="Complete Genome,Chromosome")
    ap.add_argument("--one-per-rank", metavar="RANK", default=None,
                    help="sample assemblies per distinct value of this rank "
                         "(e.g. 'phylum') to diversify phylogenetic coverage")
    ap.add_argument("--per-rank-n", type=int, default=1,
                    help="assemblies to keep per rank value (default 1); "
                         "N>1 lets held-out genomes share rank vocabulary")
    ap.add_argument("--require-gff", action="store_true",
                    help="drop assemblies lacking a RefSeq GFF; default keeps "
                         "them with annotation labels masked")
    ap.add_argument("--nt-budget", type=int, default=100_000,
                    help="nucleotide budget per assembly (total bases sampled); "
                         "size-normalises fragment counts across genome sizes")
    ap.add_argument("--group-nt-total", type=int, default=0,
                    help="if >0, overrides --nt-budget: this total nucleotide "
                         "budget is split EVENLY across each group's selected "
                         "assemblies, so every group contributes equal bases "
                         "(sparse groups get deep per-genome coverage, "
                         "dominant groups thin)")
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
    ap.add_argument("--genbank-groups", nargs="+", default=[],
                    choices=GROUPS, metavar="GROUP",
                    help="groups that additionally draw picks from GenBank "
                         "assemblies with no RefSeq twin (gbrs_paired_asm=na). "
                         "Use for underrepresented groups; picks are marked "
                         "seq_source=genbank in labels.tsv and are train-only "
                         "(build_split keeps val purely RefSeq, so label "
                         "noise in submitter-declared taxonomy can't leak "
                         "into the benchmark)")
    ap.add_argument("--genbank-extra", type=int, default=0,
                    help="genbank picks allowed for rank values already "
                         "covered by refseq (default 0 = pure gap-fill: "
                         "genbank only adds rank values refseq missed)")
    ap.add_argument("--genbank-levels",
                    default="Complete Genome,Chromosome,Scaffold,Contig",
                    help="assembly levels eligible for genbank picks; "
                         "broader than --levels since genbank is mostly "
                         "draft assemblies")
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
    gb_groups = set(args.genbank_groups)
    gb_levels = set(args.genbank_levels.split(","))
    selected = []
    for g in args.groups:
        rows = load_summary(g, cache)
        sel = select_assemblies(rows, levels, args.one_per_rank,
                                rng, skipped, args.workers,
                                tax_resolver=resolve_taxonomy,
                                cache=cache, per_rank_n=args.per_rank_n,
                                require_gff=args.require_gff)
        if args.group_nt_total and sel:
            for r in sel:
                r["_nt_budget"] = args.group_nt_total // len(sel)
        n_refseq = len(sel)
        n_gb = 0
        if g in gb_groups:
            # GenBank pass: keep only assemblies with no RefSeq twin,
            # then gap-fill rank values the RefSeq pass missed
            # (--genbank-extra relaxes this to a bounded supplement).
            # GFF stays optional (genbank annotation is sparse) and picks
            # are never flagged test-only — genbank rows are train-only
            # via seq_source.
            covered = {r["_rank_key"] for r in sel if "_rank_key" in r}
            gb_rows = [r for r in load_summary(g, cache, source="genbank")
                       if (r.get("gbrs_paired_asm") or "na").strip()
                       in ("", "na")]
            gb_sel = select_assemblies(
                gb_rows, gb_levels, args.one_per_rank, rng, skipped,
                args.workers, tax_resolver=resolve_taxonomy, cache=cache,
                per_rank_n=args.per_rank_n, require_gff=False,
                covered=covered, covered_n=args.genbank_extra,
                mark_unresolved=False)
            for r in gb_sel:
                r["_seq_source"] = "genbank"
            n_gb = len(gb_sel)
            sel.extend(gb_sel)
        tag = "GFF required" if args.require_gff else "GFF optional"
        print(f"{g}: {len(rows)} assemblies -> {n_refseq} selected"
              + (f" (+{n_gb} genbank)" if g in gb_groups else "")
              + f" ({tag})")
        selected.extend(sel)

    # 3. taxonomy labels (resolved for all selected upfront so the
    #    per-accession workers return fully-labelled records)
    taxids = {r["species_taxid"] for r in selected
              if r.get("species_taxid")}
    tax_map = resolve_taxonomy(taxids, cache)

    # 4. fetch + fragment (parallel); results buffered by selection index
    #    so fragments/labels are written in deterministic order
    results = {}  # i -> (row, label_source, frags, n_veu_unmatched)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(slice_assembly, r, gdir, args, cache): i
                for i, r in enumerate(selected)}
        for n_done, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            row = selected[i]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"  FAILED {row['assembly_accession']}: {e}",
                      file=sys.stderr)
                skipped.append({"accession": row["assembly_accession"],
                                "group": row["group"],
                                "reason": f"fetch_error: {e}"})
            if n_done % max(1, len(selected) // 20) == 0:
                print(f"  processed {n_done}/{len(selected)}")
    ok = [selected[i] for i in sorted(results)]
    print(f"downloaded+sliced {len(ok)}/{len(selected)} genomes")

    # 5. emit
    frag_path = args.outdir / "fragments.fna"
    lab_path = args.outdir / "labels.tsv"
    skip_path = args.outdir / "skipped.tsv"
    n_frags = 0
    with open(frag_path, "w") as ff, open(lab_path, "w", newline="") as lf:
        w = csv.writer(lf, delimiter="\t")
        w.writerow(["frag_id", "accession", "seqid", "start", "end",
                    "length", "taxid", "group", "rank_unresolved",
                    "seq_source", "coding", "label_source", "coding_frac",
                    "boundary", "n_boundaries", "boundary_offset"] +
                   RANKS +
                   ["intron_frac", "rna_frac", "active_frame",
                    "struct_flags", "domain_ids"])
        for i in sorted(results):
            row, label_source, frags, n_veu_unmatched = results[i]
            tax = tax_map.get(row.get("species_taxid", ""), {})
            for seqid, s, e, frag, stats in frags:
                fid = f"{row['assembly_accession']}|{seqid}|{s}-{e}"
                ff.write(f">{fid}\n")
                for i2 in range(0, len(frag), 70):
                    ff.write(frag[i2:i2 + 70] + "\n")
                if stats is None:
                    frac_s = bnd_s = nb_s = boff_s = ""
                    intr_s = rna_s = frame_s = flag_s = dom_s = "NA"
                    frac = None
                elif isinstance(stats, dict):  # refseq_gff rich tier
                    frac = stats["frac"]
                    frac_s = f"{frac:.4f}"
                    bnd_s = "1" if 0 < frac < 1 else "0"
                    nb_s = str(stats["n_bnd"])
                    boff_s = (f"{stats['boff']:.4f}"
                              if stats["boff"] is not None else "")
                    intr_s = f"{stats['intron']:.4f}"
                    rna_s = f"{stats['rna']:.4f}"
                    frame_s = str(stats["frame"])
                    flag_s = str(stats["flags"])
                    # "" = annotated, no domains; "NA" = unannotated
                    dom_s = ",".join(stats["domains"])
                else:                        # veupath tuple tier
                    frac, n_bnd, boff = stats
                    frac_s = f"{frac:.4f}"
                    bnd_s = "1" if 0 < frac < 1 else "0"
                    nb_s = str(n_bnd)
                    boff_s = f"{boff:.4f}" if boff is not None else ""
                    intr_s = rna_s = frame_s = flag_s = dom_s = "NA"
                cod = ("NA" if stats is None else
                       "coding" if frac >= 0.5 else "non-coding")
                w.writerow([fid, row["assembly_accession"], seqid, s, e,
                            len(frag), row.get("species_taxid", ""),
                            row["group"], row.get("_rank_unresolved", "0"),
                            row.get("_seq_source", "refseq"),
                            cod, label_source,
                            frac_s, bnd_s, nb_s, boff_s] +
                           [tax.get(r, "") or "" for r in RANKS] +
                           [intr_s, rna_s, frame_s, flag_s, dom_s])
                n_frags += 1
            if n_veu_unmatched:
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
