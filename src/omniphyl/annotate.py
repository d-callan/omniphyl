"""omniphyl annotate — sliding-window genome annotation.

Implements docs/inference-plan.md §3: strided windows -> raised-cosine
per-bin aggregation (with the contig-global frame coordinate transform)
-> optional LexicMap RNA-seq evidence fusion -> Stage-2 base-exact
ORF / splice-junction decoding -> GFF3 + bedGraph tracks + a whole
genome report.
"""
from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encode import STOPS, STARTS
from .io import (ContigEvidence, iter_sequences, load_lexicmap_evidence,
                 revcomp, write_gff3)
from .model import OmniPhylPredictor, RANKS

REGION_NAMES = ("coding", "intron", "rna", "intergenic")
FLAG_NAMES = ("start", "stop", "donor", "acceptor")

_CODE4_STOPS = {"TAA", "TAG"}  # TGA->Trp in genetic code 4


@dataclass
class ContigResult:
    contig: str
    length: int
    n_bins: int
    bin_width: int
    p_region: np.ndarray          # [nbins, 4]
    p_frame: np.ndarray           # [nbins, 8]
    b_tracks: np.ndarray          # [nbins, 4]  unstranded boundary impulse
    p_boundary: np.ndarray        # [nbins]
    p_coding: np.ndarray          # [nbins]
    dtilde: np.ndarray            # [nbins] fused evidence coverage
    e_lex: np.ndarray             # [nbins] evidence endpoint impulses
    taxon: object = None          # TaxonCall
    n_windows: int = 0
    genes: list = field(default_factory=list)


def _window_starts(L: int, W: int, S: int) -> list[int]:
    if L <= W:
        return [0]
    starts = list(range(0, L - W + 1, S))
    if starts[-1] != L - W:
        starts.append(L - W)
    return starts


def _gcode_stops(gcode: int) -> set:
    return _CODE4_STOPS if gcode == 4 else STOPS


def _n_frac(seq: str) -> float:
    return sum(1 for b in seq if b.upper() not in "ACGT") / max(len(seq), 1)


def run_annotate(model_path, fasta_path, out_prefix: Path,
                 window: int = 500, stride: int = 100,
                 bin_width: int = 0,
                 batch_size: int = 256, device: str = "auto",
                 flat_prior_tau: float | None = None,
                 lexicmap_bed=None, lexicmap_bedgraph=None,
                 lexicmap_tsv=None, lexicmap_source: str = "rnaseq",
                 lexicmap_min_score: float = 150.0,
                 lexicmap_min_len: int = 60,
                 lexicmap_min_single_score: float = 350.0,
                 lexicmap_min_intron_support: int = 1,
                 lexicmap_max_per_run: int = 2,
                 evidence_weight: float = 0.35,
                 kingdom: str = "auto", genetic_code: str = "auto",
                 min_cds_len: int = 90, coding_threshold: float = 0.45,
                 boundary_threshold: float = 0.25,
                 domain_threshold: float = 0.5,
                 conf_tau: float = 0.5, anc_tau: float = 0.2,
                 min_intron: int = 35, max_intron: int = 5000,
                 verbose: bool = True) -> dict:
    """Full annotate pipeline; writes all outputs, returns report dict."""
    out_prefix = Path(out_prefix)
    pred = OmniPhylPredictor(model_path, device=device,
                             flat_prior_tau=flat_prior_tau,
                             verbose=verbose)
    W, S = int(window), int(stride)
    # bin width: 50 bp at defaults; finer B upsamples the same smooth
    # aggregate (per-window votes are spread over the ~W they cover) but
    # resolves boundary impulses and run edges more sharply.
    B = int(bin_width) if bin_width else math.gcd(S, 50)
    w_ev = float(evidence_weight)

    contigs = [(i, s, c) for i, s, c in iter_sequences(Path(fasta_path))]
    if verbose:
        print(f"omniphyl annotate: {len(contigs)} contigs, "
              f"{sum(len(s) for _, s, _ in contigs):,} bp")
    contig_seqs = {i: s for i, s, _ in contigs}

    # ---------------- Step 1: windows across ALL contigs, batched ---------
    win_meta: list[tuple] = []     # (contig_idx, a_w, win_len)
    win_seqs: list[str] = []
    for ci, (cid, seq, _cov) in enumerate(contigs):
        for a in _window_starts(len(seq), W, S):
            w = seq[a:a + W]
            if _n_frac(w) > 0.5:
                continue
            win_meta.append((ci, a, len(w)))
            win_seqs.append(w)
    if verbose:
        print(f"omniphyl annotate: {len(win_seqs)} windows "
              f"(W={W}, S={S}, batch={batch_size})")
    bp = pred.predict_sequences(win_seqs, batch_size=batch_size)

    # ---------------- Step 3 (early load): LexicMap evidence ---------------
    evidence: dict[str, ContigEvidence] = {}
    if lexicmap_bed or lexicmap_bedgraph or lexicmap_tsv:
        evidence = load_lexicmap_evidence(
            bed_path=lexicmap_bed, bedgraph_path=lexicmap_bedgraph,
            tsv_path=lexicmap_tsv, source=lexicmap_source,
            min_score=lexicmap_min_score, min_len=lexicmap_min_len,
            min_single_score=lexicmap_min_single_score,
            min_intron_support=lexicmap_min_intron_support,
            max_per_run=lexicmap_max_per_run, bin_width=B,
            contig_seqs=contig_seqs)
        if verbose:
            print(f"omniphyl annotate: lexicmap evidence ({lexicmap_source}) "
                  f"on {len(evidence)} contigs")

    # ---------------- Step 2: raised-cosine aggregation --------------------
    results: list[ContigResult] = []
    logp_by_contig: list[list[int]] = [[] for _ in contigs]
    for wi, (ci, a, wl) in enumerate(win_meta):
        logp_by_contig[ci].append(wi)

    for ci, (cid, seq, cov) in enumerate(contigs):
        L = len(seq)
        nbins = max(1, math.ceil(L / B))
        wsum = np.full(nbins, 1e-9, np.float64)
        p_region = np.zeros((nbins, 4), np.float64)
        p_frame = np.zeros((nbins, 8), np.float64)
        p_coding = np.zeros(nbins, np.float64)
        p_boundary = np.zeros(nbins, np.float64)
        b_tracks = np.zeros((nbins, 4), np.float64)
        for wi in logp_by_contig[ci]:
            _ci, a, wl = win_meta[wi]
            x = (np.arange(nbins) * B + B / 2 - a) / wl
            u = 0.1 + 0.9 * np.sin(np.pi * np.clip(x, 0, 1)) ** 2
            u = np.where((x >= 0) & (x <= 1), u, 0.0)
            if not u.any():
                continue
            wsum += u
            p_coding += u * bp.coding_frac[wi]
            p_boundary += u * bp.boundary[wi]
            if bp.region_fracs is not None:
                p_region += u[:, None] * bp.region_fracs[wi]
            else:
                p_region[:, 0] += u * bp.coding_frac[wi]
                p_region[:, 3] += u * (1 - bp.coding_frac[wi])
            if bp.active_frame is not None:
                # window-local frames -> contig-global frame ids.
                # fwd f: g = (a + f) % 3
                # rev f: g = 3 + ((a + wl - (f-3)) % 3)
                row = bp.active_frame[wi]
                gmap = [(a + f) % 3 for f in range(3)] + [
                    3 + ((a + wl - r) % 3) for r in range(3)]
                for f, g in enumerate(gmap):
                    p_frame[:, g] += u * row[f]
                p_frame[:, 6] += u * row[6]
                p_frame[:, 7] += u * row[7]
                if bp.struct_flags is not None:
                    fl = bp.struct_flags[wi]
                    xhat = a + bp.offset[wi] * wl
                    # gaussian impulse, sigma=35 bp. Strand-awareness lives
                    # at decode time: track idx 2 is the splice donor on +
                    # and the acceptor on - (and vice-versa for idx 3).
                    dist = np.arange(nbins) * B + B / 2 - xhat
                    imp = (np.exp(-0.5 * (dist / 35.0) ** 2)
                           * bp.boundary[wi])
                    b_tracks += imp[:, None] * fl
        p_region /= wsum[:, None]
        p_frame /= wsum[:, None]
        p_coding /= wsum
        p_boundary /= wsum
        # evidence fusion
        ev = evidence.get(cid)
        dtilde = np.zeros(nbins, np.float64)
        e_lex = np.zeros(nbins, np.float64)
        if ev is not None:
            d = ev.density[:nbins]
            if len(d) < nbins:
                d = np.pad(d, (0, nbins - len(d)))
            l1p = np.log1p(d)
            p90 = np.percentile(l1p[l1p > 0], 90) if (l1p > 0).any() else 1
            dtilde = np.minimum(1.0, l1p / max(p90, 1.0))
            # repeat damper: high depth + low model evidence -> cap
            med = np.median(d[d > 0]) if (d > 0).any() else 0
            if med > 0:
                p_crna = (p_region[:, 0]
                          + (p_region[:, 2] if bp.region_fracs is not None
                             else 0.0))
                damp = (d > 10 * med) & (p_crna < 0.35)
                dtilde = np.where(damp, np.minimum(dtilde, 0.15), dtilde)
            if lexicmap_source == "rnaseq" and ev.endpoints is not None:
                e_lex = ev.endpoints[:nbins]
                if len(e_lex) < nbins:
                    e_lex = np.pad(e_lex, (0, nbins - len(e_lex)))
                # transcript endpoints are junction candidates on both
                # splice tracks; motif+phase validation sorts out which
                # side is donor vs acceptor
                b_tracks[:, 2] += e_lex
                b_tracks[:, 3] += e_lex
        results.append(ContigResult(
            contig=cid, length=L, n_bins=nbins, bin_width=B,
            p_region=p_region, p_frame=p_frame,
            b_tracks=b_tracks, p_boundary=p_boundary,
            p_coding=p_coding, dtilde=dtilde, e_lex=e_lex,
            n_windows=len(logp_by_contig[ci])))

    # ---------------- Step 4a: taxonomy calls -------------------------------
    def _pooled_call(idxs):
        if not idxs:
            return None
        # mean of log-probs must be renormalized: arithmetic averaging
        # drops total mass (mean log != log mean), deflating confidence
        # for every rank. log-mean-exp keeps pooled a proper log-softmax.
        pooled = {}
        for t in pred.cls_targets:
            lp = bp.logp[t][idxs]
            m = lp.max(0, keepdims=True)
            pooled[t] = m + np.log(
                np.exp(lp - m).mean(0, keepdims=True))
        pod = {t: bp.p_ood[t][idxs].mean(0, keepdims=False)
               for t in pred.cls_targets}
        calls = pred.call(pooled, conf_tau=conf_tau, anc_tau=anc_tau,
                          p_ood={t: v[None] for t, v in pod.items()})
        return calls[0]

    genome_call = _pooled_call(list(range(len(win_seqs))))
    for ci, res in enumerate(results):
        res.taxon = _pooled_call(logp_by_contig[ci])

    # ---------------- Step 4b: genetic-code inference -----------------------
    gcode = 1
    if genetic_code not in ("auto", "1"):
        gcode = int(genetic_code)
    else:
        n_cod = {"TAA": 0, "TAG": 0, "TGA": 0}
        total = 0
        for ci, res in enumerate(results):
            if bp.active_frame is None:
                break
            seq = contig_seqs[res.contig]
            gstar = res.p_frame[:, :6].argmax(1)
            for b in range(res.n_bins):
                if res.p_coding[b] <= 0.8:
                    continue
                g = int(gstar[b])
                off = g if g < 3 else g - 3
                for x in range(b * B + off,
                               min((b + 1) * B, res.length) - 2, 3):
                    # minus frame: codon occupies genomic [x, x+3) with
                    # (x+3) ≡ (g-3) mod 3 → x ≡ off; read the triplet on
                    # the transcript strand
                    tri = (seq[x:x + 3] if g < 3
                           else revcomp(seq[x:x + 3])).upper()
                    if tri in STOPS:
                        n_cod[tri] += 1
                        total += 1
                    elif set(tri) <= set("ACGT"):
                        total += 1
        if total >= 1000 and (n_cod["TAA"] + n_cod["TAG"]) / total < 0.002 \
                and n_cod["TGA"] / total > 0.02:
            gcode = 4
    if verbose:
        print(f"omniphyl annotate: genetic code = {gcode}")

    # ---------------- Step 4c: Stage-2 gene decoding -------------------------
    stops = _gcode_stops(gcode)
    for ci, res in enumerate(results):
        seq = contig_seqs[res.contig]
        dom = (res.taxon.names.get("domain", "") if res.taxon else "")
        euk = (kingdom == "eukaryote"
               or (kingdom == "auto" and dom == "Eukaryota"))
        prok = (kingdom == "prokaryote" or kingdom == "viral"
                or (kingdom == "auto" and dom in
                    ("Bacteria", "Archaea", "Viruses")))
        res.genes = _decode_genes(
            seq, res, stops=stops, euk=euk and not prok,
            w_ev=w_ev, min_cds_len=min_cds_len,
            coding_threshold=coding_threshold,
            boundary_threshold=boundary_threshold,
            min_intron=min_intron, max_intron=max_intron,
            intron_hints=(evidence.get(res.contig).intron_candidates
                          if res.contig in evidence else []))
        # per-gene Pfam assignment
        if bp.domains is not None and pred.domain_vocab:
            id2dom = {i: d for d, i in pred.domain_vocab.items()}
            for g in res.genes:
                s, e, strand = g["start"], g["end"], g["strand"]
                pool = [bp.domains[wi] for wi, (_ci, a, wl)
                        in enumerate(win_meta)
                        if _ci == ci and a < e and a + wl > s]
                if pool:
                    mean_p = np.mean(pool, axis=0)
                    g["domains"] = [id2dom[i] for i in np.argsort(-mean_p)
                                    if mean_p[i] >= domain_threshold]
                else:
                    g["domains"] = []

    # ---------------- Step 4d: chimeric contig flagging ----------------------
    consensus_deep = {}
    if genome_call and genome_call.deepest_rank:
        for r in ("phylum", "class"):
            if RANKS.index(r) <= RANKS.index(genome_call.deepest_rank):
                consensus_deep[r] = genome_call.names.get(r)
    for res in results:
        res.chimeric = False
        if (res.taxon and res.length >= 1500
                and res.taxon.deepest_rank
                and RANKS.index(res.taxon.deepest_rank)
                >= RANKS.index("class")
                and res.taxon.path_conf.get(res.taxon.deepest_rank, 0)
                >= 0.6):
            for r in ("phylum", "class"):
                if (r in consensus_deep
                        and res.taxon.names.get(r) not in
                        (None, "", "<unknown>")
                        and res.taxon.names.get(r) != consensus_deep[r]):
                    res.chimeric = True

    # ---------------- outputs ----------------------------------------------
    # Opt-in dump of the aggregated per-contig tracks + sequences; used
    # to iterate on Stage-2 decode parameters without re-running window
    # inference (decode-only benchmarks, debugging).
    if os.environ.get("OMNIPHYL_DUMP_RES"):
        with open(str(out_prefix) + ".res.pkl", "wb") as fh:
            pickle.dump({"results": results, "contig_seqs": contig_seqs,
                         "stops": stops, "gcode": gcode}, fh)
    report = _write_outputs(out_prefix, results, contigs, genome_call,
                            gcode, w_ev, lexicmap_source, verbose)
    return report


# ---------------------------------------------------------- Stage-2 ------


def _scan_codons(seq: str, frame: int, kind: str, lo: int, hi: int,
                 stops: set) -> list[int]:
    """All canonical triplet edges of `kind` in [lo, hi) for global frame.

    frame 0..2 = forward (codon0 = i, frame = i%3);
    frame 3..5 = reverse (codon0 = i+3 exclusive end; frame = (i+3)%3+3).
    kind start/stop matches scan_transition edge conventions.

    The contig-global grid keys a reverse codon [x, x+3) to
    3 + x % 3 (genomic start residue), matching the window->global
    gmap and gcode inference. In revcomp coordinates a codon at index
    i occupies genomic [L-i-3, L-i), so x = L-i-3 and the required
    residue is i ~ L - off, NOT i ~ off -- iterating revcomp residue
    `off` directly would apply the per-fragment label convention
    globally and scan the wrong codon positions (labels 4/5 swap when
    L % 3 == 0, all three shift otherwise).
    """
    cands = STARTS if kind == "start" else stops
    rev = frame >= 3
    s2 = revcomp(seq) if rev else seq
    L = len(s2)
    off = (frame - 3) if rev else frame
    out = []
    i0 = (L - off) % 3 if rev else off
    for i in range(i0, L - 2, 3):
        if s2[i:i + 3] not in cands:
            continue
        edge = (L - i if kind == "start" else L - i - 3) if rev \
            else (i if kind == "start" else i + 3)
        if lo <= edge < hi:
            out.append(edge)
    return sorted(out)


def _orf_extends(seq: str, frame: int, edge: int, direction: int,
                 stops: set, max_scan: int = 30000):
    """Walk codons from a start edge until an in-frame stop; return stop edge."""
    rev = frame >= 3
    s2 = revcomp(seq) if rev else seq
    L = len(s2)
    if rev:
        # edge is genomic coord of transcript start (stop side = lower coord)
        i2 = L - edge
        rng = range(i2, min(i2 + max_scan, L - 2), 3)
    else:
        i2 = edge
        rng = range(i2, min(i2 + max_scan, L - 2), 3)
    for i in rng:
        if s2[i:i + 3] in stops:
            return (L - i - 3) if rev else (i + 3)
    return None


def _region_of(p_row) -> int:
    return int(np.argmax(p_row))


def _decode_genes(seq: str, res: ContigResult, stops: set, euk: bool,
                  w_ev: float, min_cds_len: int,
                  coding_threshold: float, boundary_threshold: float,
                  min_intron: int, max_intron: int,
                  intron_hints: list) -> list:
    """Segment bins -> ORF/splice-snapped gene models on both strands.

    Runs of high fused-coding-score bins define candidate regions; inside
    each region the dominant global frame per strand is decoded by
    start->stop ORF scanning, and (euk mode) interior dips linked by
    splice motifs become introns.
    """
    # softmasked (lowercase) sequence is biologically identical to its
    # uppercase form -- all codon/motif comparisons below are exact-
    # string matches, so normalize once here instead of sprinkling
    # .upper() through every scanner
    seq = seq.upper()
    L = res.length
    B = res.bin_width
    genes = []
    coding = res.p_region[:, 0] if res.p_region[:, 0].sum() > 0 \
        else res.p_coding
    utr_gate = np.minimum(1.0, coding / 0.25)
    # fused evidence: log p_coding + w_ev * UTR_gate * d~(x)  (plan §3.2)
    cscore = np.exp(np.log(np.clip(coding, 1e-6, 1.0))
                    + w_ev * utr_gate * res.dtilde)
    have_frames = res.p_frame[:, :6].sum() > 0
    in_coding = cscore >= coding_threshold
    runs = []
    b = 0
    while b < res.n_bins:
        if not in_coding[b]:
            b += 1
            continue
        b2 = b
        while b2 + 1 < res.n_bins and in_coding[b2 + 1]:
            b2 += 1
        runs.append((b, b2))
        b = b2 + 1
    # decode each run independently AND (euk) fused with adjacent
    # same-dominant-strand runs across gaps <= max_intron: an intron
    # whose bins all dip below the coding threshold splits one gene
    # into separate runs, which per-run decoding can never re-join
    # (chr1 benchmark: 15 of 208 RefSeq introns do this). The fused
    # span is an ADDITIONAL decode -- each interior gap is seeded into
    # _splice_candidates as an intron proposal and _score_orf /
    # _better_shared_stop arbitrate merged vs split calls, so close
    # tandem same-strand genes only fuse when a valid phased ORF
    # actually reads through.
    hint_xy = [(iv[0], iv[1]) for iv in intron_hints if len(iv) == 2]
    # a run that crosses the coding threshold ONLY through fused
    # evidence (raw model coding < threshold on every bin) is usually
    # UTR/ncRNA coverage masquerading as a CDS: decode it standalone
    # only when it also carries concentrated single-frame support. It
    # still participates in chained / hint-linked super-runs below.
    if have_frames:
        def _standalone_ok(b0, b1):
            return bool(coding[b0:b1 + 1].max() >= coding_threshold
                        or res.p_frame[b0:b1 + 1, :6].max(1).mean()
                        >= 0.40)
    else:
        def _standalone_ok(b0, b1):
            return True
    spans = [(b0 * B, min((b1 + 1) * B, L), [])
             for b0, b1 in runs if _standalone_ok(b0, b1)]
    if euk and have_frames and len(runs) > 1:
        def _run_strand(rr):
            b0, b1 = rr
            sp = float(res.p_frame[b0:b1 + 1, :3].sum())
            sn = float(res.p_frame[b0:b1 + 1, 3:6].sum())
            return "+" if sp >= sn else "-"
        i = 0
        while i < len(runs):
            j = i
            seeds = []
            while (j + 1 < len(runs)
                   and _run_strand(runs[j]) == _run_strand(runs[j + 1])
                   and 0 < (runs[j + 1][0] - runs[j][1] - 1) * B
                   <= max_intron):
                seeds.append(((runs[j][1] + 1) * B,
                              runs[j + 1][0] * B,
                              min(150, 3 * B)))
                j += 1
            if j > i:
                spans.append((runs[i][0] * B,
                              min((runs[j][1] + 1) * B, L), seeds))
            i = j + 1
    # hint-linked expansion: a split-alignment intron with exactly one
    # junction inside a coding run pins a partner exon beyond the local
    # search margin -- widen the span so the splice scan reaches it.
    # Junctions bridging two runs merge them even when their gap does
    # not satisfy the generic chain conditions; a hint fully outside
    # all runs has no anchor and cannot be decoded anyway.
    if euk and hint_xy:
        def _run_of(x):
            for i, (b0, b1) in enumerate(runs):
                if b0 * B <= x < min((b1 + 1) * B, L):
                    return i
            return -1
        for hd, ha in hint_xy:
            i, j = _run_of(hd), _run_of(ha)
            if i == j:
                continue
            if 0 <= i < j:
                if ha - hd <= max_intron:
                    spans.append((runs[i][0] * B,
                                  min((runs[j][1] + 1) * B, L), []))
            elif i >= 0 and ha - min((runs[i][1] + 1) * B, L) <= max_intron:
                spans.append((runs[i][0] * B, min(L, ha + 300), []))
            elif j >= 0 and runs[j][0] * B - hd <= max_intron:
                spans.append((max(0, hd - 300),
                              min((runs[j][1] + 1) * B, L), []))
    # a wider span (chained super-run or hint-linked expansion) fully
    # contains the standalone spans inside it. Decoding a contained
    # span separately would only re-emit sub-fragments that compete
    # with -- and can outscore and suppress -- the full-length spliced
    # call arbitration produces in the wider context; every candidate
    # of the narrow span is re-scanned inside the wider one anyway.
    # Identical bounds from chaining vs hint-merging collapse too.
    spans = [(a, b, tuple(s)) for a, b, s in dict.fromkeys(
        (lo, hi, tuple(gs)) for lo, hi, gs in spans)]
    spans = [(lo, hi, list(gs)) for lo, hi, gs in spans
             if not any(olo <= lo and hi <= ohi
                        and (olo, ohi) != (lo, hi)
                        for olo, ohi, _ in spans)]
    for lo, hi, gap_seeds in spans:
        # Both strands are always decoded: strand dominance is enforced
        # per-ORF inside _score_orf (over the candidate's own exonic
        # bins) rather than across the whole run -- a whole-run gate
        # would suppress a short gene adjacent to a long opposite-
        # strand gene merged into the same coding run.
        for strand in ("+", "-"):
            g_lo, g_hi = (0, 3) if strand == "+" else (3, 6)
            if have_frames:
                # decode every frame dominant in >=2 bins of the run:
                # adjacent genes (and exons across non-multiple-of-3
                # introns) can occupy different genomic frames within
                # a single merged coding run
                b, b2 = lo // B, (hi - 1) // B
                dom = res.p_frame[b:b2 + 1, g_lo:g_hi].argmax(1)
                cnt = np.bincount(dom, minlength=3)
                frames = [g_lo + i for i in range(3) if cnt[i] >= 2]
                if not frames:
                    fm = res.p_frame[b:b2 + 1, g_lo:g_hi].mean(0)
                    frames = [g_lo + int(fm.argmax())]
            else:
                frames = list(range(g_lo, g_hi))
            # spliced ORF scan is frame-free; run it once per strand
            for g in frames[:1] if euk else frames:
                genes += _emit_orfs(
                    seq, g, lo, hi, res, stops, strand,
                    min_cds_len, euk, min_intron, max_intron,
                    intron_hints + gap_seeds, in_coding,
                    boundary_threshold, cscore, coding_threshold)
    # dedupe identical CDS intervals (different frame bins and chained
    # vs per-run decodes can collide); score-first ordering keeps the
    # better-supported variant, e.g. a spliced call over an identical
    # unspliced span
    seen, uniq = set(), []
    for g in sorted(genes, key=lambda x: -x.get("score", 0.0)):
        key = (g["start"], g["end"], g["strand"])
        if key not in seen:
            seen.add(key)
            uniq.append(g)
    # Even with frame evidence, bin-runs decoded per-strand can emit
    # overlapping ORFs (adjacent runs, snapping drift, shadow ORFs on
    # the opposite strand). Greedy best-score keep: drop ORFs
    # overlapping a kept one by >50% of their span -- same strand
    # unconditionally, opposite strand unless the weaker call carries
    # independent strand evidence of its own.
    uniq = _nonoverlap_select(uniq)
    return sorted(uniq, key=lambda g: (g["start"], g["end"]))


def _nonoverlap_select(genes, max_overlap_frac: float = 0.5) -> list:
    kept = []
    for g in sorted(genes,
                    key=lambda x: -x.get("score", x["end"] - x["start"])):
        span = g["end"] - g["start"]
        drop = False
        for k in kept:
            ov = min(g["end"], k["end"]) - max(g["start"], k["start"])
            if ov <= max_overlap_frac * span:
                continue
            if k["strand"] == g["strand"]:
                drop = True
                break
            # cross-strand overlap: a stop-poor opposite-strand ORF can
            # shadow a real gene's entire span. Suppress it unless it
            # carries independent strand evidence (strand_share = the
            # fraction of frame mass on ITS exonic bins that agrees
            # with its strand) and a competitive score -- those are the
            # signatures of a real nested antisense gene.
            if not (g.get("strand_share", 0.0) >= 0.45
                    and g.get("score", 0.0)
                    >= 0.8 * k.get("score", 0.0)):
                drop = True
                break
        if not drop:
            kept.append(g)
    return kept


def _exon_frames(g, res):
    """Global frame id (0..5) implied for each exon of a gene dict.

    An intron whose length is not a multiple of 3 shifts the genomic
    frame of downstream exons, so each exon's frame is derived from the
    cumulative transcript length of the exons before it rather than
    taken from a single run-dominant frame. For '-' strand, exons are
    walked in transcript order (descending genomic) and a codon's
    global frame is keyed on its transcript-exclusive end coordinate,
    matching the convention used by _scan_codons/gcode inference.
    Returns a list of (g_ex, b0, b1) per exon: frame id and covered
    bins [b0, b1).
    """
    B = res.bin_width
    strand = g["strand"]
    exons = sorted(_exon_bounds(g), reverse=(strand == "-"))
    prev_len = 0
    out = []
    for es, ee in exons:
        phase = (3 - prev_len % 3) % 3
        if strand == "+":
            g_ex = (es + phase) % 3
        else:
            g_ex = 3 + ((ee + 1 - phase) % 3)
        b0 = es // B
        b1 = min(res.n_bins, ee // B + 1)
        out.append((g_ex, b0, b1))
        prev_len += ee - es + 1
    return out


def _score_orf(seq, res, cscore, have_frames, g):
    """Model-posterior support score for one candidate gene dict.

    Replaces length-only arbitration (longest-ORF and longest-first
    overlap selection), which systematically over-extends 5' ends
    through stop-poor non-coding sequence and lets spurious readthrough
    ORFs suppress real shorter genes.

    Components:
      * body  - mean support over EXONIC bins. With a frame head this
        is p_frame at each exon's IMPLIED genomic frame (which shifts
        across non-multiple-of-3 introns), so a real multi-exon gene
        scores well while a false splice join yields downstream exons
        whose implied frame mismatches the model's per-bin frame and
        is penalised. Without frames, the fused coding score is used.
      * start - start-codon impulse at the start edge plus the coding
        step-up across it (a true start sits at the 5' edge of the
        coding signal; a spurious upstream ATG inside non-coding
        sequence has ~zero step).
      * prior - 0 for ATG, -0.15 for GTG/TTG so alternative starts
        need supporting signal to win (prokaryotic only; euk spliced
        scans emit ATG only).

    Local strand dominance is enforced per-ORF here (>=25% of frame
    mass on exonic bins must lie on the gene's strand) instead of
    gating the whole bin-run: a short gene adjacent to a long
    opposite-strand gene in one merged coding run would otherwise be
    suppressed across its entire span.

    Returns the score, or -1.0 to reject outright.
    """
    B = res.bin_width
    a, bnd, strand = g["start"], g["end"], g["strand"]
    ex_bins = _exon_frames(g, res)
    n_bins_tot = sum(b1 - b0 for _g, b0, b1 in ex_bins)
    if n_bins_tot <= 0:
        return -1.0
    mask = np.zeros(res.n_bins, dtype=bool)
    for _g, b0, b1 in ex_bins:
        mask[b0:b1] = True
    if have_frames:
        raw_body = float(sum(res.p_frame[b0:b1, gx].sum()
                             for gx, b0, b1 in ex_bins)) / n_bins_tot
        s_pos = float(res.p_frame[mask, :3].sum())
        s_neg = float(res.p_frame[mask, 3:6].sum())
        own = s_pos if strand == "+" else s_neg
        if own < 0.25 * (s_pos + s_neg):
            return -1.0
        # fraction of exonic-bin frame mass on this strand; recorded
        # for cross-strand overlap arbitration in _nonoverlap_select
        g["strand_share"] = own / max(1e-9, s_pos + s_neg)
        # RNA-seq coverage corroborates an exon only in proportion to
        # its own-strand frame mass: unstranded depth cannot rescue
        # the minority strand, and a weak-exon frame vote backed by
        # coverage is worth more than the same vote in empty coverage
        dt = res.dtilde
        ev_cov = (float(dt[mask].mean())
                  if dt is not None and len(dt) >= res.n_bins else 0.0)
        body = raw_body + 0.15 * g["strand_share"] * ev_cov
    else:
        body = float(cscore[mask].mean())
        g["strand_share"] = 1.0
    se = a if strand == "+" else bnd
    edge_bin = se // B if strand == "+" else (se - 1) // B
    edge_bin = max(0, min(res.n_bins - 1, edge_bin))
    imp = float(res.b_tracks[edge_bin, 0])
    if strand == "+":
        inside = cscore[edge_bin:min(res.n_bins, edge_bin + 2)]
        outside = cscore[max(0, edge_bin - 2):edge_bin]
    else:
        inside = cscore[max(0, edge_bin - 1):edge_bin + 1]
        outside = cscore[edge_bin + 1:min(res.n_bins, edge_bin + 3)]
    step = (float(inside.mean()) - float(outside.mean())
            if len(inside) and len(outside) else 0.0)
    codon = (seq[a:a + 3].upper() if strand == "+"
             else revcomp(seq[max(0, bnd - 3):bnd]).upper())
    prior = 0.0 if codon == "ATG" else -0.15
    cds_len = sum(ee - es + 1 for es, ee in _exon_bounds(g))
    # short unspliced ORFs need clearly-above-noise support: in
    # stop-poor sequence a sub-200 bp intergenic ORF can pass the
    # 50%-coding-bin gate on a handful of weak bins. The 0.35 body
    # floor sits above the non-coding per-bin frame-mass baseline
    # (~0.2 on the chr1 benchmark) so such calls must carry real
    # signal. Spliced calls are exempt -- a short gene with a
    # motif-validated intron already passed structural scrutiny.
    if (have_frames and not g.get("introns") and cds_len < 180
            and body < 0.35):
        return -1.0
    g["body"] = body
    # each intron corroborated by a split-alignment hint is a direct
    # structural vote for the full-length spliced call over its own
    # sub-fragments
    g["score"] = (body * math.log1p(cds_len)
                  + 0.5 * (imp + step) + prior
                  + 0.4 * g.get("n_hint_introns", 0))
    return g["score"]


def _better_shared_stop(a, b, res, cscore, have_frames,
                        coding_threshold):
    """Choose between two ORFs sharing a stop edge (alt. starts).

    Whether the longer ORF is real reduces to one local question: is
    its 5' extension coding *on the ORF's own reading frame*? Support
    is measured at the extension's implied frame (same global frame as
    the ORF -- extension codons continue its phase) against half the
    shared gene body's own-frame support, so the rule is scale-free
    and strand-aware: sequence coding only in another frame or on the
    opposite strand does not justify extending the CDS. Without a
    frame head the fused coding score is used against the same
    relative threshold. Bins fully inside the longer ORF's introns are
    excluded -- introns are expected to be non-coding.
    """
    if a["strand"] == "+":
        lng, sht = (a, b) if a["start"] <= b["start"] else (b, a)
        ext_lo, ext_hi = lng["start"], sht["start"]
    else:
        lng, sht = (a, b) if a["end"] >= b["end"] else (b, a)
        ext_lo, ext_hi = sht["end"], lng["end"]
    B = res.bin_width
    ext_bins = [x for x in range(ext_lo // B,
                                 min(res.n_bins, (ext_hi - 1) // B + 1))
                if not any(d <= x * B and (x + 1) * B <= acc
                           for d, acc in lng["introns"])]
    if not ext_bins or ext_lo == ext_hi:
        return lng if lng["score"] >= sht["score"] else sht
    if have_frames:
        if a["strand"] == "+":
            g_orf = lng.get("frame")
            if g_orf is None:
                g_orf = ext_lo % 3
        else:
            g_orf = lng.get("frame")
            if g_orf is None:
                g_orf = 3 + (lng["end"] % 3)
        sup = float(res.p_frame[ext_bins, g_orf].mean())
        # a split-alignment intron inside the longer ORF's extension
        # corroborates a short partner exon whose window-averaged
        # frame mass alone falls below the extension gate
        if (lng.get("n_hint_introns", 0) > sht.get("n_hint_introns", 0)
                and res.dtilde is not None
                and len(res.dtilde) >= res.n_bins):
            sup += 0.25 * float(res.dtilde[ext_bins].mean())
    else:
        sup = float(cscore[ext_bins].mean())
    ref = sht.get("body", 0.0) if have_frames else \
        float(cscore[sht["start"] // B:(sht["end"] - 1) // B + 1].mean())
    return lng if sup >= 0.5 * ref else sht


def _emit_orfs(seq, g, lo, hi, res, stops, strand, min_cds_len,
               euk, min_intron, max_intron, intron_hints, in_coding,
               boundary_threshold, cscore, coding_threshold):
    """Within one coding bin-run, emit ORFs that overlap the run;
    alternative starts sharing a stop are collapsed to the longest.

    in_coding is the fused per-bin coding mask; an ORF must have >=50%
    of its span inside coding bins, not merely clip a run's edge.

    euk mode: ORFs are additionally scanned on spliced transcripts
    (candidate introns excised) so an intronic in-frame stop cannot
    truncate a gene; junction codons/phase are then literal transcript
    bases. Each candidate intron is also tried alone and in adjacent
    pairs -- excising the whole candidate set at once would let a
    single false intron mis-frame the entire downstream transcript.

    All arbitration is by _score_orf (model-posterior support), not
    span length: candidates sharing a stop edge keep the best score.
    """
    out = []
    L = len(seq)
    B = res.bin_width
    have_frames = res.p_frame[:, :6].sum() > 0
    hint_xy = [iv for iv in intron_hints if len(iv) == 2]
    # euk gets a wider transcript-5' margin: short terminal exons can
    # sit an intron away from the first bin-run that crosses the
    # coding threshold, so the splice scan must reach further on the
    # side where a leader exon could be hiding.
    m5 = min(max_intron, 600) if euk else 300
    s_lo = max(0, lo - (m5 if strand == "+" else 300))
    s_hi = min(L, hi + (m5 if strand == "-" else 300))
    by_stop: dict = {}
    for start_edge in _scan_codons(seq, g, "start", s_lo, s_hi, stops):
        stop_edge = _orf_extends(seq, g, start_edge, +1, stops)
        if stop_edge is None:
            continue
        if strand == "+":
            a, bnd = start_edge, stop_edge
        else:
            a, bnd = stop_edge, start_edge
        if bnd - a < min_cds_len or a >= hi or bnd <= lo:
            continue
        # ORF-body support: >= half of covered bins must be coding
        b0, b1 = a // B, min(res.n_bins, (bnd - 1) // B + 1)
        if b1 > b0 and in_coding[b0:b1].mean() < 0.5:
            continue
        cand = {"start": a, "end": bnd, "strand": strand, "frame": g,
                "introns": [], "domains": []}
        # Key on the shared stop edge (upper coord on +, lower on -),
        # NOT `bnd`: on the minus strand bnd is the start edge, so each
        # alternative start keyed a distinct entry and never collapsed.
        if _score_orf(seq, res, cscore, have_frames, cand) < 0:
            continue
        prev = by_stop.get(stop_edge)
        if prev is None:
            by_stop[stop_edge] = cand
        else:
            by_stop[stop_edge] = _better_shared_stop(
                prev, cand, res, cscore, have_frames, coding_threshold)
    if euk:
        cands = _splice_candidates(
            seq, res, s_lo, s_hi, strand, min_intron, max_intron,
            intron_hints, boundary_threshold, in_coding, cscore,
            have_frames, coding_threshold)
        # [] = unspliced frame-free scan: needed even when introns are
        # proposed (ORFs in non-dominant frames), and it is the only
        # frame-free coverage when no candidates exist.
        subsets = [[]] + ([cands] if cands else []) + \
            [[iv] for iv in cands] + \
            [cands[i:i + 2] for i in range(len(cands) - 1)]
        for sub in subsets:
            seq_s, pos = _spliced(seq, s_lo, s_hi, sub)
            for o in _emit_spliced_orfs(
                    seq_s, pos, lo, hi, res, stops, strand,
                    min_cds_len, sub, in_coding, seq, cscore,
                    have_frames, coding_threshold, hint_xy):
                if o["score"] < 0:
                    continue
                se = o["end"] if strand == "+" else o["start"]
                prev = by_stop.get(se)
                if prev is None:
                    o["frame"] = g
                    by_stop[se] = o
                else:
                    by_stop[se] = _better_shared_stop(
                        prev, o, res, cscore, have_frames,
                        coding_threshold)
    out.extend(by_stop.values())
    return out


def _splice_candidates(seq, res, lo, hi, strand, min_intron,
                       max_intron, hints, boundary_threshold, in_coding,
                       cscore, have_frames, coding_threshold):
    """Motif-validated intron candidates inside [lo, hi), one strand.

    Proposal sources:
      * evidence split-alignment hints (exact coords, snapped +-12 bp);
      * model dips: a coding sag, an intron-head bin
        (p_region[:,1] >= 0.15), or a local coding minimum, linked to
        donor/acceptor impulse mass in neighbouring bins;
      * frame-shift transitions: consecutive coding bins on this
        strand whose dominant same-strand frame differs -- a short
        intron that never dips a whole bin below the coding threshold
        still shifts the downstream reading frame. This is the primary
        detector for introns shorter than ~1-2 bins.

    Model proposals carry bin-center quantization error of up to B/2,
    so they snap to canonical splice motifs within +-B; evidence hints
    snap within +-12 bp. Candidate pairs are scored by motif class
    (canonical GT-AG / CT-AC preferred over GC-AG / AC-GC) plus
    donor/acceptor impulse and intron-head mass, then filtered to a
    greedy non-overlapping chain in score order. Fully-interior bins
    that still look coding (candidate carved out of an exon) or that
    carry a gene-boundary impulse (candidate fusing two genes across
    an intergenic gap) are rejected. Junction codon/phase checks are
    deliberately absent -- they are implicit once the accepted set is
    excised and ORFs are scanned on the spliced transcript.
    Returns an ordered list of (donor_end, acceptor_start) pairs.
    """
    B = res.bin_width
    coding = res.p_region[:, 0]
    d_track = res.b_tracks[:, 2] if strand == "+" else res.b_tracks[:, 3]
    a_track = res.b_tracks[:, 3] if strand == "+" else res.b_tracks[:, 2]
    # (d0, a0, snap_radius): 2-tuple hints are evidence proposals and
    # snap +-12 bp; 3-tuples carry their own radius (chained-run gap
    # seeds and other structural proposals).
    cand = [(iv[0], iv[1], iv[2] if len(iv) > 2 else 12)
            for iv in hints
            if lo < iv[0] < iv[1] < hi]
    hint_xy = [(iv[0], iv[1]) for iv in hints if len(iv) == 2]
    b_lo = max(1, lo // B)
    b_hi = min(res.n_bins - 1, (hi - 1) // B + 1)
    for b in range(b_lo, b_hi):
        dip = (coding[b] <= 0.35
               and coding[b - 1] >= 0.5 and coding[b + 1] >= 0.5)
        intron_head = res.p_region[b, 1] >= 0.15
        local_min = (coding[b] < coding[b - 1] - 0.1
                     and coding[b] < coding[b + 1] - 0.1)
        if not (dip or intron_head or local_min):
            continue
        d_lo = max(0, b - 3)
        a_hi = min(res.n_bins - 1, b + 3)
        dpos = int(np.argmax(d_track[d_lo:b + 1]) + d_lo)
        apos = int(np.argmax(a_track[b:a_hi + 1]) + b)
        cand.append((dpos * B + B // 2, apos * B + B // 2, B))
    domf = None
    if have_frames:
        g_lo = 0 if strand == "+" else 3
        domf = res.p_frame[:, g_lo:g_lo + 3].argmax(1)
        strong = np.nonzero(cscore >= coding_threshold)[0]
        strong = strong[(strong >= b_lo) & (strong < b_hi)]
        for bi, bj in zip(strong, strong[1:]):
            if (bj - bi) * B > max_intron:
                continue
            if domf[bi] != domf[bj]:
                cand.append((bi * B + B // 2, bj * B + B // 2, B))
        # weak-signal terminal-exon seeds: a non-coding bin in the
        # transcript-5' margin carrying real same-strand frame mass can
        # be a short terminal exon that never reached the coding
        # threshold. 0.35 sits between the non-coding baseline (median
        # ~0.23 on the chr1 benchmark) and genuine exonic signal; a
        # lower cutoff proposes from pure noise, and purely structural
        # proposals cannot win arbitration anyway because
        # _better_shared_stop requires extension support on the ORF's
        # own frame.
        sm = res.p_frame[:, g_lo:g_lo + 3].sum(1)
        for bm in range(b_lo, b_hi):
            if cscore[bm] >= coding_threshold or sm[bm] < 0.35:
                continue
            if strand == "+":
                right = strong[strong > bm]
                if len(right):
                    cand.append((bm * B + B // 2,
                                 int(right[0]) * B + B // 2, B))
            else:
                left = strong[strong < bm]
                if len(left):
                    cand.append((int(left[-1]) * B + B // 2,
                                 bm * B + B // 2, B))
    hint_set = {(iv[0], iv[1]) for iv in hints if len(iv) == 2}
    scored = []
    for d0, a0, rad in cand:
        # a literal split-alignment hint outranks motif/impulse
        # heuristics: it gets a score bonus, a tighter snap radius, and
        # a stronger pull back toward the observed junction
        is_hint = (d0, a0) in hint_set
        best = None
        for d in range(max(lo + 3, d0 - rad),
                       min(hi - min_intron, d0 + rad + 1)):
            if strand == "+":
                d_canon = seq[d:d + 2] == "GT"
                if seq[d:d + 2] not in ("GT", "GC"):
                    continue
            else:
                # minus strand: transcript acceptor AG -> genomic "CT"
                # at the intron's LEFT (5'-genomic) end
                d_canon = seq[d:d + 2] == "CT"
                if seq[d:d + 2] != "CT":
                    continue
            for acc in range(max(a0 - rad, d + min_intron),
                             min(a0 + rad, d + max_intron, hi) + 1):
                if strand == "+":
                    if seq[acc - 2:acc] != "AG":
                        continue
                    a_canon = True
                else:
                    # minus strand: transcript donor GT -> genomic "AC"
                    # at the RIGHT (3'-genomic) end; the non-canonical
                    # GC donor appears as "GC"
                    a_canon = seq[acc - 2:acc] == "AC"
                    if seq[acc - 2:acc] not in ("AC", "GC"):
                        continue
                # canonicality is per-PAIR (donor AND acceptor); the
                # donor flag must not be folded into a loop-mutated
                # variable that leaks between acceptor iterations
                sc = (2.0 if (d_canon and a_canon) else 1.0)
                sc += 1.5 if is_hint else 0.0
                sc += float(d_track[min(res.n_bins - 1, d // B)])
                sc += float(a_track[min(res.n_bins - 1, acc // B)])
                ib0, ib1 = (d + B - 1) // B, acc // B
                if ib1 > ib0:
                    sc += float(res.p_region[ib0:ib1, 1].mean())
                # break ties toward the un-snapped proposal; hints pull
                # harder so every valid acceptor in the snap window is
                # compared, not just the leftmost
                sc -= (0.02 if is_hint else 0.001) * (
                    abs(d - d0) + abs(acc - a0))
                if best is None or sc > best[0]:
                    best = (sc, d, acc)
        if best:
            scored.append(best)
    # greedy non-overlapping chain in quality order: a higher-scored
    # junction outranks a conflicting neighbour
    scored.sort(key=lambda x: -x[0])
    chain = []
    for _sc, d, acc in scored:
        if not all(acc <= od or d >= oa for od, oa in chain):
            continue
        ib0, ib1 = (d + B - 1) // B, acc // B
        if ib1 > ib0:
            # A real intron's interior codes WEAKER than its flanking
            # exonic bins; a candidate whose interior codes as
            # strongly as its flanks is carving a continuous exon in
            # two -- reject unless evidence says otherwise. (A hard
            # in_coding vote cannot work: candidates are scanned
            # inside coding runs, so their interior bins are in_coding
            # by construction and the old check rejected everything.)
            has_hint = any(abs(d - hd) <= 12 and abs(acc - ha) <= 12
                           for hd, ha in hint_xy)
            f_shift = (domf is not None
                       and int(domf[min(res.n_bins - 1, d // B)])
                       != int(domf[min(res.n_bins - 1, acc // B)]))
            if not (has_hint or f_shift):
                int_c = float(cscore[ib0:ib1].mean())
                flank = 0.5 * (float(cscore[min(res.n_bins - 1,
                                               d // B)])
                               + float(cscore[min(res.n_bins - 1,
                                                  acc // B)]))
                if int_c > max(0.75, flank - 0.05):
                    continue
            # a strong start/stop impulse inside the candidate marks a
            # gene boundary, not a splice junction. Splice sites
            # themselves fire p_boundary (any CDS/non-CDS transition),
            # so p_boundary cannot discriminate -- compare structural
            # head mass in the fully-interior bins directly. A split-
            # alignment hint overrides this veto: observed RNA-seq
            # reads spanning the junction are direct evidence the two
            # sides join, whatever the impulse tracks say.
            if (not has_hint
                    and res.b_tracks[ib0:ib1, :2].max()
                    > res.b_tracks[ib0:ib1, 2:4].max()):
                continue
        chain.append((d, acc))
        chain.sort()
    return chain


def _spliced(seq, lo, hi, introns):
    """Excise intron interiors [d, acc) from seq[lo:hi).

    Returns (spliced_seq, pos) where pos[t] is the genomic coordinate
    of spliced base t (absolute, 0-based).
    """
    parts, pos = [], []
    cur = lo
    for d, acc in introns:
        parts.append(seq[cur:d])
        pos.extend(range(cur, d))
        cur = acc
    parts.append(seq[cur:hi])
    pos.extend(range(cur, hi))
    return "".join(parts), pos


def _emit_spliced_orfs(seq_s, pos, lo, hi, res, stops, strand,
                       min_cds_len, cands, in_coding, seq, cscore,
                       have_frames, coding_threshold, hint_xy):
    """Start->stop ORFs on the spliced transcript, mapped to genomic.

    Exon-chain phase and junction codons need no explicit check: a
    wrong intron choice produces an in-frame stop and the ORF simply
    truncates. For the minus strand the transcript is revcomp(seq_s)
    with pos reversed, so tpos[t] is the genomic coordinate of
    transcript base t on that strand's reading direction.

    Each emitted dict carries "score" from _score_orf; because the
    score sums per-exon support at each exon's IMPLIED genomic frame,
    a false splice join (mis-phased downstream exons) scores below the
    intact genomic ORF in the caller's shared-stop arbitration.
    """
    B = res.bin_width
    if strand == "+":
        tseq, tpos = seq_s, pos
    else:
        tseq, tpos = revcomp(seq_s), pos[::-1]
    n = len(tseq)
    by_stop: dict = {}
    for t in range(n - 2):
        if tseq[t:t + 3] != "ATG":
            continue
        u = t + 3
        while u + 3 <= n and tseq[u:u + 3] not in stops:
            u += 3
        if u + 3 > n:
            break  # no stop downstream for this or any later start
        if strand == "+":
            a, bnd = tpos[t], tpos[u + 2] + 1
        else:
            a, bnd = tpos[u + 2], tpos[t] + 1
        if bnd - a < min_cds_len or a >= hi or bnd <= lo:
            continue
        ivs = [iv for iv in cands if a < iv[0] and iv[1] <= bnd]
        # ORF-body support over EXONIC bins only: a bin fully inside an
        # intron is transcribed but not coding, so it must not count
        # against the >=50% coding-support gate
        exonic = [b for b in range(a // B,
                                   min(res.n_bins, (bnd - 1) // B + 1))
                  if not any(d <= b * B and (b + 1) * B <= acc
                             for d, acc in ivs)]
        if exonic and in_coding[exonic].mean() < 0.5:
            continue
        n_hint = sum(1 for d, acc in ivs
                     if any(abs(d - hd) <= 12 and abs(acc - ha) <= 12
                            for hd, ha in hint_xy))
        cand = {"start": a, "end": bnd, "strand": strand, "frame": None,
                "introns": ivs, "n_hint_introns": n_hint, "domains": []}
        if _score_orf(seq, res, cscore, have_frames, cand) < 0:
            continue
        stop_edge = bnd if strand == "+" else a
        prev = by_stop.get(stop_edge)
        if prev is None:
            by_stop[stop_edge] = cand
        else:
            by_stop[stop_edge] = _better_shared_stop(
                prev, cand, res, cscore, have_frames, coding_threshold)
    return list(by_stop.values())


# ---------------------------------------------------------- outputs ------


def _lineage_str(call) -> str:
    prefixes = {"domain": "d", "phylum": "p", "class": "c", "order": "o",
                "family": "f", "genus": "g", "species": "s"}
    parts = []
    for r in RANKS:
        nm = call.names.get(r, "") if call else ""
        if not nm or nm in ("<unknown>", "unclassified"):
            nm = ""
        parts.append(f"{prefixes[r]}__{nm}")
    return ";".join(parts)


def _write_outputs(out_prefix: Path, results: list[ContigResult],
                   contigs, genome_call, gcode, w_ev,
                   lexicmap_source, verbose) -> dict:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    gid = 0
    features = []
    contig_rows = []
    n_genes = n_multi = 0
    dom_count: dict[str, int] = {}
    for res in results:
        feats = []
        for g in res.genes:
            gid += 1
            n_genes += 1
            introns = g.get("introns", [])
            n_ex = 1 + len(introns)
            if n_ex > 1:
                n_multi += 1
            exons = sorted(_exon_bounds(g))
            gene_id = f"g{gid}"
            strand = g["strand"]
            attrs = {"ID": gene_id,
                     "frame_global": g["frame"],
                     "domains": ",".join(g.get("domains", [])) or "."}
            feats.append({"seqid": res.contig, "type": "gene",
                          "start": g["start"] + 1, "end": g["end"],
                          "strand": strand, "score": ".",
                          "attributes": dict(attrs)})
            feats.append({"seqid": res.contig, "type": "mRNA",
                          "start": g["start"] + 1, "end": g["end"],
                          "strand": strand, "score": ".",
                          "attributes": {"ID": f"{gene_id}.t1",
                                         "Parent": gene_id}})
            # CDS phase: bases to skip from feature start to the next
            # codon boundary, measured in transcript order. For minus
            # strand the feature's transcript-5' end is its genomic end.
            ex_sorted = (sorted(exons) if strand == "+"
                         else sorted(exons, reverse=True))
            for ei, (es, ee) in enumerate(ex_sorted):
                prev_len = sum(x2 - x1 + 1 for x1, x2 in ex_sorted[:ei])
                cds_phase = (3 - prev_len % 3) % 3 if ei else 0
                feats.append({"seqid": res.contig, "type": "exon",
                              "start": es + 1, "end": ee, "strand": strand,
                              "score": ".",
                              "attributes": {"ID": f"{gene_id}.e{ei+1}",
                                             "Parent": f"{gene_id}.t1"}})
                feats.append({"seqid": res.contig, "type": "CDS",
                              "start": es + 1, "end": ee, "strand": strand,
                              "score": ".", "phase": cds_phase,
                              "attributes": {"ID": f"{gene_id}.c{ei+1}",
                                             "Parent": f"{gene_id}.t1"}})
            for d in g.get("domains", []):
                dom_count[d] = dom_count.get(d, 0) + 1
        features.extend(feats)
        tn = res.taxon
        contig_rows.append({
            "contig": res.contig, "length": res.length,
            "n_windows": res.n_windows, "n_genes": len(res.genes),
            "coding_density": round(
                float((res.p_region[:, 0] > 0.5).mean()), 4),
            "deepest_rank": tn.deepest_rank if tn else None,
            "lineage": _lineage_str(tn) if tn else "",
            "chimeric_flag": bool(getattr(res, "chimeric", False))})
    write_gff3(str(out_prefix) + ".gff3", iter(features))

    # bedGraph tracks (concatenated across contigs, one track section each)
    with open(str(out_prefix) + ".coding.bedGraph", "w") as f:
        f.write("track type=bedGraph name=omniphyl_coding\n")
        for res in results:
            _write_bg(f, res.contig, res.p_region[:, 0]
                      if res.p_region[:, 0].sum() > 0 else res.p_coding,
                      res.bin_width)
    with open(str(out_prefix) + ".boundary.bedGraph", "w") as f:
        f.write("track type=bedGraph name=omniphyl_boundary\n")
        for res in results:
            _write_bg(f, res.contig, res.p_boundary, res.bin_width)

    # contig summary
    with open(str(out_prefix) + ".contig_summary.tsv", "w") as f:
        f.write("contig\tlength\tn_windows\tn_genes\tcoding_density\t"
                "deepest_rank\tlineage\tchimeric_flag\n")
        for r in contig_rows:
            f.write("\t".join(str(r[k]) for k in
                              ("contig", "length", "n_windows", "n_genes",
                               "coding_density", "deepest_rank", "lineage",
                               "chimeric_flag")) + "\n")

    # whole-genome report
    total_bp = sum(len(s) for _, s, _ in contigs)
    lens = sorted((len(s) for _, s, _ in contigs), reverse=True)
    acc = 0
    n50 = 0
    for L in lens:
        acc += L
        if acc >= total_bp / 2:
            n50 = L
            break
    gc = (sum(s.upper().count("G") + s.upper().count("C")
              for _, s, _ in contigs) / max(total_bp, 1))
    comp = np.zeros(4)
    for res in results:
        comp += res.p_region.sum(0) * res.bin_width
    comp /= max(comp.sum(), 1e-9)
    n_chimeras = sum(bool(r.get("chimeric_flag")) for r in contig_rows)
    top_doms = sorted(dom_count.items(), key=lambda x: -x[1])[:20]
    report = {
        "taxonomy": {
            "deepest_rank": genome_call.deepest_rank if genome_call else None,
            "lineage": _lineage_str(genome_call) if genome_call else "",
            "per_rank_confidence": (genome_call.path_conf
                                    if genome_call else {}),
            "p_ood": (genome_call.p_ood if genome_call else {}),
        },
        "assembly": {
            "total_bp": total_bp, "n_contigs": len(contigs), "n50": n50,
            "gc": round(gc, 4), "genetic_code": gcode,
            "composition": dict(zip(REGION_NAMES, comp.round(4).tolist())),
        },
        "features": {
            "n_genes": n_genes, "n_multi_exon": n_multi,
            "n_single_exon": n_genes - n_multi,
            "n_chimeric_contigs": n_chimeras,
            "top_domains": top_doms,
        },
        "lexicmap": {"source": lexicmap_source, "weight": w_ev},
    }
    (Path(str(out_prefix) + ".report.json")).write_text(
        json.dumps(report, indent=1))
    md = _report_md(report, contig_rows)
    (Path(str(out_prefix) + ".report.md")).write_text(md)
    if verbose:
        print(md)
        print(f"omniphyl annotate: wrote {out_prefix}.{{gff3,report.md,"
              f"report.json,coding.bedGraph,boundary.bedGraph,"
              f"contig_summary.tsv}}")
    return report


def _exon_bounds(g):
    """Exon (start, end-inclusive) pairs for a gene's intron chain.

    _find_introns already guarantees an ordered, non-overlapping chain;
    non-positive exons are skipped defensively rather than emitted as
    inverted intervals.
    """
    a, bnd = g["start"], g["end"]
    exons, cur = [], a
    for d, acc in sorted(g.get("introns", [])):
        if d - 1 >= cur:
            exons.append((cur, d - 1))
        cur = max(cur, acc)
    if bnd - 1 >= cur:
        exons.append((cur, bnd - 1))
    return exons


def _write_bg(f, seqid, values, bin_width):
    i, n = 0, len(values)
    while i < n:
        v = float(values[i])
        j = i + 1
        while j < n and float(values[j]) == v:
            j += 1
        f.write(f"{seqid}\t{i * bin_width}\t{j * bin_width}\t{v:.3f}\n")
        i = j


def _report_md(report, contig_rows) -> str:
    tx, asm, ft = report["taxonomy"], report["assembly"], report["features"]
    lines = [
        "# OmniPhyl annotation report", "",
        "## Whole-genome taxonomic identity",
        f"- **Deepest confident rank:** {tx['deepest_rank'] or 'none'}",
        f"- **Lineage:** `{tx['lineage']}`",
        "- **Per-rank path confidence:** " + ", ".join(
            f"{r}={v:.3f}" for r, v in tx["per_rank_confidence"].items()),
        "", "## Assembly",
        f"- **Total bp:** {asm['total_bp']:,}",
        f"- **Contigs:** {asm['n_contigs']}  (N50 = {asm['n50']:,} bp)",
        f"- **GC:** {asm['gc']:.1%}   **Genetic code:** {asm['genetic_code']}",
        "- **Composition:** " + ", ".join(
            f"{k} {v:.1%}" for k, v in asm["composition"].items()),
        "", "## Features",
        f"- **Genes:** {ft['n_genes']} "
        f"({ft['n_single_exon']} single-exon / {ft['n_multi_exon']} multi-exon)",
        f"- **Chimeric/discordant contigs flagged:** {ft['n_chimeric_contigs']}",
    ]
    if ft["top_domains"]:
        lines += ["- **Top Pfam domains:** " + ", ".join(
            f"{d} (n={c})" for d, c in ft["top_domains"])]
    disc = [r for r in contig_rows if r["chimeric_flag"]]
    if disc:
        lines += ["", "## Discordant contigs", "",
                  "| contig | length | lineage |", "|---|---|---|"]
        for r in disc:
            lines.append(f"| {r['contig']} | {r['length']:,} | "
                         f"{r['lineage']} |")
    return "\n".join(lines) + "\n"
