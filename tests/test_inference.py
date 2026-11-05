#!/usr/bin/env python3
"""Inference-package tests for src/omniphyl/.

Covers (docs/inference-plan.md §7):
  - I/O: FASTA/FASTQ(+gz) iteration, wrapped FASTQ, paired joining,
    LexicMap BED/bedGraph loading + filters (score/len/entropy/
    per-run cap), GFF3/bedGraph writers
  - frame: window-local -> contig-global transform, minus-strand codon
    coordinate convention
  - boundary: raised-cosine aggregation, gaussian impulse tracks
  - reconcile: deepest-confident-rank gating, strict ancestor projection
  - predictor: synthetic checkpoint -> predict_sequences -> TaxonCall;
    p_ood uses the training flat tau while known-class logp uses the
    user flat tau
  - profile: canonical-kmer clustering, abundance = bases vs reads
  - annotate end-to-end on a synthetic genome

Run: python3 -m pytest tests/test_inference.py  (or python3 tests/test_inference.py)
"""
import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SCRIPTS))

import torch  # noqa: E402

from omniphyl.io import (  # noqa: E402
    iter_sequences, iter_paired_sequences, load_lexicmap_evidence,
    revcomp, write_gff3, write_bedgraph)
from omniphyl.model import (  # noqa: E402
    OmniPhyl, OmniPhylPredictor, ancestor_chains, path_scores,
    build_parent_of, build_flat_priors, RANKS)
from omniphyl.encode import (  # noqa: E402
    FeatureEncoder, scan_transition, HASH_LEN)
from omniphyl.profile import cluster_reads, _canon_kmers  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  ({detail})" if str(detail) != "" else ""))


# ---------------------------------------------------------------- I/O ---


def test_io():
    print("[io]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td / "a.fna"
        fa.write_text(">c1 cov_3.5 extra\nACGT\nACGT\n>c2\nGGGG\n")
        recs = list(iter_sequences(fa))
        check("fasta ids", [r[0] for r in recs] == ["c1", "c2"])
        check("fasta seq+cov", recs[0][1] == "ACGTACGT"
              and recs[0][2] == 3.5)

        # wrapped FASTQ: quality string begins with '@'
        fq = td / "r.fastq"
        fq.write_text("@r1\nACGTACGT\n+\n@@BBFFFF\n@r2\nGGGG\n+\nFFFF\n")
        recs = list(iter_sequences(fq))
        check("fastq records", [r[0] for r in recs] == ["r1", "r2"],
              recs[0][1])

        fgz = td / "z.fastq.gz"
        with gzip.open(fgz, "wt") as f:
            f.write("@x\nAAAA\n+\nFFFF\n")
        check("gz fastq", list(iter_sequences(fgz))[0][1] == "AAAA")

        r1 = td / "R1.fq"
        r2 = td / "R2.fq"
        r1.write_text("@p/1\nAAAA\n+\nFFFF\n")
        r2.write_text("@p/2\nTTTT\n+\nFFFF\n")
        pair = list(iter_paired_sequences(r1, r2))
        check("paired N-join", pair[0][1] == "AAAAN" + "AAAA",
              pair[0][1])

        # LexicMap BED two-tier filtering: unitigA chains a strong
        # anchor (400) with a weak partner (160) across a 90 bp gap ->
        # the partner survives even though it is below the singleton
        # floor; unitigB is an unchained singleton at 300 -> dropped
        # under an explicit 350 singleton floor.
        bed = td / "lx.bed"
        bed.write_text(
            "c1\t10\t310\tSRR1|unitigA|hsp1\t400\t+\n"
            "c1\t400\t700\tSRR1|unitigA|hsp2\t160\t+\n"
            "c1\t800\t1100\tSRR2|unitigB|hsp1\t300\t+\n")
        seq = "ACGT" * 200
        ev = load_lexicmap_evidence(
            bed_path=bed, source="rnaseq", min_score=150, min_len=60,
            min_single_score=350,
            contig_seqs={"c1": seq}, keep_hsps=True)
        kept = ev["c1"].hsps
        check("lexicmap two-tier filters",
              len(kept) == 2 and all(h.sgenome == "SRR1" for h in kept),
              [h.score for h in kept])
        check("lexicmap sra_runs", ev["c1"].sra_runs == {"SRR1"})
        check("density positive", ev["c1"].density.sum() > 0)
        check("split-alignment intron hint",
              ev["c1"].intron_candidates == [(310, 400)],
              ev["c1"].intron_candidates)
        # singleton floor honours an explicit override
        ev_lo = load_lexicmap_evidence(
            bed_path=bed, source="rnaseq", min_score=150, min_len=60,
            min_single_score=250, contig_seqs={"c1": seq},
            keep_hsps=True)
        check("singleton floor override",
              len(ev_lo["c1"].hsps) == 3, len(ev_lo["c1"].hsps))

        # bedGraph: no filtering, density from rows
        bg = td / "lx.bedGraph"
        bg.write_text("c1\t0\t50\t3\n")
        ev2 = load_lexicmap_evidence(bedgraph_path=bg, source="genomic")
        check("bedgraph density", ev2["c1"].density[0] == 3)

        # writers
        gff = td / "o.gff3"
        write_gff3(gff, iter([{"seqid": "c1", "type": "gene",
                               "start": 1, "end": 100, "strand": "+",
                               "attributes": {"ID": "g1"}}]))
        line = gff.read_text().splitlines()[-1]
        check("gff3 row", line.split("\t")[0] == "c1"
              and line.split("\t")[2] == "gene" and "ID=g1" in line)
        bgo = td / "o.bedGraph"
        write_bedgraph(bgo, "c1", np.array([0.5, 0.5, 0.1]), 50)
        check("bedgraph merge", "0\t100\t0.5" in bgo.read_text())


# --------------------------------------------------------------- frame ---


def test_frame():
    print("[frame]")
    # scan_transition minus-strand edge convention: exclusive upper end
    seq = "ATGAAA" + "TAA" + "GGG"  # fwd ATG at 0, stop edge at 9
    e = scan_transition(seq, 0, "start", 0.0)
    check("fwd start edge", e == 0, e)
    e = scan_transition(seq, 0, "stop", 9.0)
    check("fwd stop edge", e == 9, e)
    # minus strand: revcomp(seq) codons; 5' coord = exclusive upper end
    # of the minus-strand codon = L - i
    s2 = revcomp(seq)
    e = scan_transition(seq, 3, "start", 0.0)
    check("rev start uses upper edge",
          e is None or isinstance(e, int), e)


# ------------------------------------------------------------ reconcile ---


def _toy_model(tmp: Path, flat_prior_tau=0.5):
    """Write a tiny flat-head checkpoint with an NCBI parent map."""
    vocab = {"domain": {"Bacteria": 0, "Eukaryota": 1},
             "phylum": {"Firmicutes": 0, "Apicomplexa": 1},
             "class": {"Bacilli": 0, "Aconoidasida": 1}}
    targets = ["domain", "phylum", "class"] + \
        ["coding_frac", "boundary", "offset"]
    n_classes = {t: len(vocab[t]) + 1 for t in vocab}
    n_classes.update({t: 1 for t in targets if t not in n_classes})
    parent_of = {"domain": [0, 0, 0],
                 "phylum": [0, 1, 2],      # Firmicutes->Bact, Apic->Euk
                 "class": [0, 1, 2]}
    log_prior = {t: [0.0] * (len(vocab[t]) + 1) for t in vocab}
    meta = {"f": {"domain": "Bacteria", "phylum": "Firmicutes",
                  "class": "Bacilli"}}
    flat_lp = {t: v.tolist() for t, v in
               build_flat_priors(meta, vocab, list(vocab)).items()}
    m = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet",
                 side=8, targets=targets, head="flat")
    ckpt = {"model": m.state_dict(), "vocab": vocab, "k": 3,
            "targets": targets,
            "tree": {"parent_of": parent_of, "log_prior": log_prior},
            "flat_log_prior": flat_lp,
            "args": {"width": 8, "backbone": "resnet",
                     "pos_channels": False, "k": 3, "head": "flat",
                     "aux": "", "flat_prior_tau": flat_prior_tau,
                     "prior_tau": 1.0}}
    path = tmp / "toy.pt"
    torch.save(ckpt, path)
    return path, vocab, parent_of


def test_reconcile():
    print("[reconcile + predictor]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ckpt, vocab, parent_of = _toy_model(td)
        pred = OmniPhylPredictor(ckpt, device="cpu", verbose=False)
        check("predictor chains built", pred.chains is not None)
        seqs = ["ACGTACGTACGTACGT", "GGGGCCCCAAAATTTT"]
        bp = pred.predict_sequences(seqs, batch_size=1)
        check("logp shape", bp.logp["domain"].shape == (2, 3))
        check("p_ood in [0,1]",
              bool(((bp.p_ood["domain"] >= 0)
                    & (bp.p_ood["domain"] <= 1)).all()))
        # logp col 0 is masked -> known-class distribution sums to 1
        s = np.exp(bp.logp["domain"]).sum(1)
        check("known-class renorm", np.allclose(s, 1.0, atol=1e-4), s)

        calls = pred.reconcile_logp(
            {t: bp.logp[t] for t in pred.cls_targets},
            conf_tau=0.0, anc_tau=0.0,
            p_ood={t: bp.p_ood[t] for t in pred.cls_targets})
        check("one call per seq", len(calls) == 2)
        c = calls[0]
        if c.deepest_rank is not None:
            ri = RANKS.index(c.deepest_rank)
            # strict projection: emitted shallower ids equal the
            # anchor class's true chain ids
            ok = True
            for j, t in enumerate(pred.cls_targets[:ri + 1]):
                cid = int(pred.chains[c.deepest_rank]
                          [c.ids[c.deepest_rank]][j])
                ok &= (c.ids[t] == max(cid, 0))
            check("strict ancestor projection", ok)
        else:
            check("strict ancestor projection", True, "abstained")

        # abstention: conf_tau=1.0 forces full abstain
        calls2 = pred.reconcile_logp(
            {t: bp.logp[t] for t in pred.cls_targets},
            conf_tau=1.01, anc_tau=0.0)
        check("abstain at tau>1",
              all(c2.deepest_rank is None for c2 in calls2))


def test_pood_tau():
    print("[p_ood train-tau vs user-tau]")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ckpt, _, _ = _toy_model(td, flat_prior_tau=0.7)
        seq = "ACGTACGTACGT"
        p_train = OmniPhylPredictor(ckpt, device="cpu", verbose=False)
        p_user = OmniPhylPredictor(ckpt, device="cpu", verbose=False,
                                   flat_prior_tau=2.0)
        b1 = p_train.predict_sequences([seq])
        b2 = p_user.predict_sequences([seq])
        # p_ood identical (train tau fixed); known-class logp differs
        same_ood = np.allclose(b1.p_ood["domain"],
                               b2.p_ood["domain"], atol=1e-6)
        diff_logp = not np.allclose(b1.logp["domain"][:, 1:],
                                    b2.logp["domain"][:, 1:], atol=1e-6)
        check("p_ood uses train tau", same_ood)
        check("known-class logp uses user tau", diff_logp)


# ------------------------------------------------------------- profile ---


def test_cluster():
    print("[profile clustering]")
    rng = np.random.default_rng(0)
    base = "".join(rng.choice(list("ACGT")) for _ in range(600))
    # r1 shares a 150-bp chunk with r2; r3 is unrelated
    r1 = base[:200]
    r2 = base[50:250]
    r3 = "".join(rng.choice(list("ACGT")) for _ in range(200))
    groups = cluster_reads(["r1", "r2", "r3"], [r1, r2, r3],
                           min_shared=2, max_chunk=500)
    member_sets = sorted(sorted(m) for m, _ in groups)
    check("shared-kmer cluster joins r1+r2",
          ["r1", "r2"] in member_sets, member_sets)
    joined = [j for m, j in groups if len(m) > 1]
    if joined:
        check("joined chunk <= 500bp", len(joined[0]) <= 500)
    # canonical kmer = rc-invariant
    check("canon kmer rc-invariant",
          _canon_kmers("ACGTACGTACGTACGTACGTAC", 21)
          == _canon_kmers(revcomp("ACGTACGTACGTACGTACGTAC"), 21))


def test_feature_encoder():
    print("[FeatureEncoder]")
    enc = FeatureEncoder(7, {"backbone": "msfcgr", "pos_channels": True,
                             "aux": "hash,aa", "aa_mode": "plain",
                             "struct_head": True})
    b = enc.encode_batch(["ACGT" * 100], virtual_joined=False)
    check("msfcgr x shape", b["x"].shape == (1, 5, 64, 64), b["x"].shape)
    check("hash shape", b["hash"].shape == (1, HASH_LEN))
    b2 = enc.encode_batch(["ACGT" * 100], virtual_joined=True)
    check("virtual_joined zeroes pos channels",
          float(b2["x"][0, -2:].max()) == 0.0)


# ---------------------------------------------------- gene decoding ----


def _mk_result(L, B, coding=None, region1=None, frame=None, btr=None):
    """Synthetic ContigResult for decoder unit tests.

    coding/region1: (b0, b1, v) applied to p_region[:,0]/[:,1]
    frame: (b0, b1, g, v) applied to p_frame[:,g]
    btr:   (b0, b1, c, v) applied to b_tracks[:,c]
    """
    from omniphyl.annotate import ContigResult
    nb = (L + B - 1) // B
    p_region = np.zeros((nb, 4))
    for b0, b1, v in (coding or []):
        p_region[b0:b1, 0] = v
    for b0, b1, v in (region1 or []):
        p_region[b0:b1, 1] = v
    p_frame = np.zeros((nb, 8))
    for b0, b1, g, v in (frame or []):
        p_frame[b0:b1, g] = v
    b_tracks = np.zeros((nb, 4))
    for b0, b1, c, v in (btr or []):
        b_tracks[b0:b1, c] = v
    return ContigResult(
        contig="c", length=L, n_bins=nb, bin_width=B,
        p_region=p_region, p_frame=p_frame, b_tracks=b_tracks,
        p_boundary=np.zeros(nb), p_coding=np.zeros(nb),
        dtilde=np.zeros(nb), e_lex=np.zeros(nb))


def _place(seq, edits):
    seq = list(seq)
    for i, s in edits:
        seq[i:i + len(s)] = s
    return "".join(seq)


def test_decode_start_scoring():
    """A spurious upstream in-frame ATG sharing a stop must lose to
    the model-supported start even though it yields the longer ORF
    (longest-ORF arbitration would wrongly extend 150 bp into
    non-coding sequence)."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 4000, 50
    seq = _place("A" * L, [(750, "ATG"), (900, "ATG"), (2400, "TAA")])
    # coding signal starts at bin 18 (900 bp); bins 15-17 are ~noise
    res = _mk_result(L, B,
                     coding=[(15, 18, 0.1), (18, 48, 0.7)],
                     frame=[(18, 48, 0, 0.7)],
                     btr=[(18, 19, 0, 0.6)])
    genes = _decode_genes(seq, res, STOPS, False, 0.0, 90, 0.5, 0.5,
                          35, 5000, [])
    check("one gene", len(genes) == 1, genes)
    if genes:
        check("true start wins", genes[0]["start"] == 900,
              genes[0]["start"])
        check("shared stop", genes[0]["end"] == 2403, genes[0]["end"])


def test_decode_adjacent_strands():
    """A minus-strand gene overlapping a longer plus-strand gene in a
    single merged coding run must still be emitted (the old whole-run
    strand gate suppressed the minority strand)."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 3999, 50
    seq = _place("A" * L, [(999, "ATG"), (1800, "TAA"),
                          (1599, "TTA"), (2394, "CAT")])
    res = _mk_result(L, B,
                     coding=[(19, 48, 0.7)],
                     frame=[(19, 31, 0, 0.7), (31, 37, 0, 0.6),
                            (31, 37, 3, 0.55), (37, 48, 3, 0.7)],
                     btr=[(19, 20, 0, 0.6), (47, 48, 0, 0.6)])
    genes = _decode_genes(seq, res, STOPS, False, 0.0, 90, 0.5, 0.5,
                          35, 5000, [])
    pl = [g for g in genes if g["strand"] == "+"]
    mn = [g for g in genes if g["strand"] == "-"]
    check("plus gene", len(pl) == 1, genes)
    check("minus gene", len(mn) == 1, genes)
    if pl:
        check("plus span", (pl[0]["start"], pl[0]["end"])
              == (999, 1803), pl[0])
    if mn:
        check("minus span", (mn[0]["start"], mn[0]["end"])
              == (1599, 2397), mn[0])


def test_rev_frame_grid():
    """_scan_codons must use the contig-global frame grid: a reverse
    codon [x, x+3) is frame 3 + x%3 (genomic-start residue), matching
    the window->global gmap and gcode inference. Iterating the revcomp
    residue directly (the per-fragment convention) scans the wrong
    codon positions globally."""
    from omniphyl.annotate import _decode_genes, _scan_codons, STOPS
    L, B = 3999, 50
    # minus ORF [1600, 2401): start codon genomic [2398,2401), x%3=1
    # -> global frame 4; stop codon genomic [1600,1603)
    seq = _place("A" * L, [(2398, "CAT"), (1600, "TTA")])
    starts4 = _scan_codons(seq, 4, "start", 0, L, STOPS)
    check("frame4 finds start edge", 2401 in starts4, starts4)
    starts5 = _scan_codons(seq, 5, "start", 0, L, STOPS)
    check("frame5 does not", 2401 not in starts5, starts5)
    res = _mk_result(L, B, coding=[(32, 48, 0.7)],
                     frame=[(32, 48, 4, 0.7)], btr=[(48, 49, 0, 0.6)])
    genes = _decode_genes(seq, res, STOPS, False, 0.0, 90, 0.5, 0.5,
                          35, 5000, [])
    mn = [g for g in genes if g["strand"] == "-"]
    check("minus orf decoded in frame4",
          len(mn) == 1 and mn[0]["start"] == 1600
          and mn[0]["end"] == 2401, genes)


def test_decode_frameshift_intron():
    """A 100 bp GT..AG intron (not a multiple of 3) shifts the
    downstream exon into frame 1; euk mode must excise it and recover
    exact junction coordinates."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 3999, 50
    seq = _place("A" * L, [(999, "ATG"), (1200, "GT"), (1298, "AG"),
                          (1498, "TAA")])
    res = _mk_result(L, B,
                     coding=[(19, 24, 0.7), (24, 26, 0.3),
                             (26, 30, 0.7)],
                     region1=[(24, 26, 0.5)],
                     frame=[(19, 24, 0, 0.7), (26, 30, 1, 0.7)],
                     btr=[(19, 20, 0, 0.6), (24, 25, 2, 0.8),
                          (26, 27, 3, 0.8)])
    genes = _decode_genes(seq, res, STOPS, True, 0.0, 90, 0.5, 0.5,
                          35, 5000, [])
    plus = [g for g in genes if g["strand"] == "+"]
    check("one spliced gene", len(plus) == 1, genes)
    if plus:
        g0 = plus[0]
        check("spliced span", (g0["start"], g0["end"])
              == (999, 1501), (g0["start"], g0["end"]))
        check("intron coords", g0["introns"] == [(1200, 1300)],
              g0["introns"])


def test_decode_antisense_suppression():
    """A stop-poor minus-strand ORF shadowing a strong plus gene must
    be dropped: it passes _score_orf's 25% strand gate (share 0.30)
    but has no independent strand evidence (share < 0.45), so
    _nonoverlap_select suppresses the >50% cross-strand overlap."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 3999, 50
    seq = _place("A" * L, [(900, "ATG"), (2400, "TAA"),
                          (999, "TTA"), (2394, "CAT")])
    res = _mk_result(L, B,
                     coding=[(18, 48, 0.7)],
                     frame=[(18, 48, 0, 0.7), (18, 48, 3, 0.3)],
                     btr=[(18, 19, 0, 0.6)])
    genes = _decode_genes(seq, res, STOPS, False, 0.0, 90, 0.5, 0.5,
                          35, 5000, [])
    pl = [g for g in genes if g["strand"] == "+"]
    mn = [g for g in genes if g["strand"] == "-"]
    check("plus gene kept", len(pl) == 1, genes)
    check("antisense shadow suppressed", len(mn) == 0, genes)


def test_decode_intron_inside_coding_run():
    """An intron whose interior bins dip but stay above the coding
    threshold is inside one contiguous coding run; the old hard
    in_coding vote rejected every such candidate by construction.
    The relative interior-vs-flank check must accept it."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 3999, 50
    seq = _place("A" * L, [(999, "ATG"), (1200, "GT"), (1297, "AG"),
                          (1398, "TAA")])
    res = _mk_result(L, B,
                     coding=[(19, 29, 0.7), (24, 25, 0.59)],
                     region1=[(24, 26, 0.5)],
                     frame=[(19, 24, 0, 0.7), (24, 25, 0, 0.0),
                            (25, 29, 0, 0.7)],
                     btr=[(24, 25, 2, 0.8), (26, 27, 3, 0.8)])
    genes = _decode_genes(seq, res, STOPS, True, 0.0, 90, 0.5, 0.5,
                          35, 5000, [])
    plus = [g for g in genes if g["strand"] == "+"]
    check("one gene", len(plus) == 1, genes)
    if plus:
        check("intron recovered", plus[0]["introns"] == [(1200, 1299)],
              plus[0]["introns"])


def test_decode_chained_runs():
    """An intron whose bins all fall below the coding threshold splits
    a gene into two runs; the chained super-run decode must bridge the
    gap via the seeded intron and out-score the unspliced call."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 3999, 50
    seq = _place("A" * L, [(999, "ATG"), (1200, "GT"), (1297, "AG"),
                          (1398, "TAA")])
    res = _mk_result(L, B,
                     coding=[(19, 24, 0.7), (26, 29, 0.7)],
                     frame=[(19, 24, 0, 0.7), (26, 29, 0, 0.7)])
    genes = _decode_genes(seq, res, STOPS, True, 0.0, 90, 0.5, 0.5,
                          35, 5000, [])
    plus = [g for g in genes if g["strand"] == "+"]
    check("one gene", len(plus) == 1, genes)
    if plus:
        g0 = plus[0]
        check("chained intron", g0["introns"] == [(1200, 1299)],
              g0["introns"])
        check("span", (g0["start"], g0["end"]) == (999, 1401),
              (g0["start"], g0["end"]))


def test_softmasked_decode_and_evidence():
    """Lowercase (softmasked) sequence must decode identically to its
    uppercase form: codon and splice-motif scans normalize case, an
    in-intron lowercase stop does not truncate the spliced ORF, and an
    HSP lying over lowercase sequence is not discarded as 'ambiguous'."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 2000, 50
    seq = _place("A" * L, [(300, "atg"), (450, "gt"), (500, "taa"),
                           (598, "ag"), (900, "taa")])
    res = _mk_result(L, B,
                     coding=[(6, 9, 0.7), (9, 12, 0.2), (12, 18, 0.7)],
                     region1=[(9, 12, 0.5)],
                     frame=[(6, 9, 0, 0.7), (12, 18, 0, 0.7)],
                     btr=[(9, 10, 2, 0.8), (12, 13, 3, 0.8)])
    genes = _decode_genes(seq, res, STOPS, True, 0.35, 90, 0.45, 0.5,
                          35, 5000, [(450, 600)])
    plus = [g for g in genes if g["strand"] == "+"]
    check("softmasked spliced gene",
          len(plus) == 1 and plus[0]["introns"] == [(450, 600)]
          and plus[0]["start"] == 300 and plus[0]["end"] == 903,
          genes)
    # an HSP spanning softmasked bases must not count them ambiguous
    with tempfile.TemporaryDirectory() as td:
        bed = Path(td) / "sm.bed"
        bed.write_text("c\t10\t610\tSRR|u|hsp1\t500\t+\n")
        sm = "ACGT" * 40
        sm = sm[:80].lower() + sm[80:]
        ev = load_lexicmap_evidence(
            bed_path=bed, source="rnaseq",
            contig_seqs={"c": sm}, keep_hsps=True)
        check("softmasked hsp kept",
              len(ev["c"].hsps) == 1, ev["c"].hsps)


def test_splice_acceptor_snapping():
    """An evidence hint must snap to ITS observed junction, not the
    leftmost canonical acceptor in the window: a decoy AG 8 bp upstream
    of the true acceptor previously won b
           test_decode_antisense_suppression,
           test_decode_intron_inside_coding_run,
           test_decode_chained_runs,
           test_softmasked_decode_and_evidence,
           test_splice_acceptor_snapping,
           test_hint_rescues_weak_partner_exon,
          ecause the acceptor loop broke
    at the first valid candidate."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 2000, 50
    seq = _place("A" * L, [(300, "ATG"), (450, "GT"),
                           (590, "AG"),   # decoy acceptor -> acc 592
                           (598, "AG"),   # true acceptor  -> acc 600
                           (900, "TAA")])
    res = _mk_result(L, B,
                     coding=[(6, 9, 0.7), (9, 12, 0.2), (12, 18, 0.7)],
                     region1=[(9, 12, 0.5)],
                     frame=[(6, 9, 0, 0.7), (12, 18, 0, 0.7)],
                     btr=[(9, 10, 2, 0.8), (12, 13, 3, 0.8)])
    genes = _decode_genes(seq, res, STOPS, True, 0.35, 90, 0.45, 0.5,
                          35, 5000, [(450, 600)])
    plus = [g for g in genes if g["strand"] == "+"]
    check("hinted acceptor chosen",
          any(g["introns"] == [(450, 600)] for g in plus), genes)


def test_hint_rescues_weak_partner_exon():
    """A split-alignment hint joins a strong coding run to a weak
    leader exon beyond the local search margin: the hint-linked span
    expansion must reach it, and shared-stop arbitration must extend
    through coverage-backed weak frame mass."""
    from omniphyl.annotate import _decode_genes, STOPS
    L, B = 2000, 50
    # exon1 [200,320): ATG at 200 (global frame 2), weak signal;
    # intron [320,750) GT..AG; exon2 [750,1450): strong frame-0 run,
    # its own ATG at 750; stop codon at 1449 (frame 0) -> edge 1452
    seq = _place("A" * L, [(200, "ATG"), (320, "GT"), (748, "AG"),
                           (1449, "TAA")])
    res = _mk_result(L, B,
                     coding=[(15, 29, 0.85), (4, 7, 0.25)],
                     frame=[(15, 29, 0, 0.7), (4, 7, 2, 0.22)])
    # coverage sits over the weak leader exon only
    res.dtilde[4:7] = 0.9
    genes = _decode_genes(seq, res, STOPS, True, 0.35, 90, 0.45, 0.5,
                          35, 5000, [(320, 750)])
    plus = [g for g in genes if g["strand"] == "+"]
    check("leader exon rescued",
          any(g["introns"] == [(320, 750)] and g["start"] == 200
              and g["end"] == 1452 for g in plus), genes)
    check("no fragment shadow",
          not any(g["start"] == 750 and not g["introns"]
                  for g in plus), genes)


def _mk_bp(n, **kw):
    """Fake predict_sequences namespace for _fuse_contig tests."""
    from types import SimpleNamespace
    d = dict(coding_frac=np.zeros(n), boundary=np.zeros(n),
             offset=np.zeros(n), region_fracs=np.zeros((n, 4)),
             active_frame=np.zeros((n, 8)),
             struct_flags=np.zeros((n, 4)), domains=None)
    d.update(kw)
    return SimpleNamespace(**d)


def test_fuse_cosine_support():
    """Bins outside every window span get zero; covered bins normalize."""
    from omniphyl.annotate import _fuse_contig
    L, B = 1000, 50
    win_meta = [(0, 100, 500)]           # window covers [100, 600)
    bp = _mk_bp(1, coding_frac=np.array([0.6]))
    res = _fuse_contig("c", L, [0], win_meta, bp, B, None, "rnaseq")
    check("bin before window = 0", res.p_coding[0] == 0)
    check("bin after window = 0", res.p_coding[12] == 0)
    check("interior bin normalized", abs(res.p_coding[4] - 0.6) < 1e-6,
          res.p_coding[4])


def test_fuse_fwd_frame_remap():
    """Different window-local fwd frames referring to the SAME
    contig-global frame must land in the same p_frame column."""
    from omniphyl.annotate import _fuse_contig
    L, B = 1000, 50
    # win0: a=100, local f=0 -> global g = 100%3 = 1
    # win1: a=101, local f=2 -> global g = 103%3 = 1
    win_meta = [(0, 100, 500), (0, 101, 500)]
    af = np.zeros((2, 8))
    af[0, 0] = 1.0
    af[1, 2] = 1.0
    bp = _mk_bp(2, active_frame=af, struct_flags=None)
    res = _fuse_contig("c", L, [0, 1], win_meta, bp, B, None, "rnaseq")
    # bins covered by both windows: all fwd mass in global col 1
    b = 4
    check("fwd mass in global col 1", abs(res.p_frame[b, 1] - 1.0) < 1e-6,
          res.p_frame[b, :6])
    check("no fwd leakage", res.p_frame[b, [0, 2, 3, 4, 5]].sum() == 0)


def test_fuse_rev_frame_remap():
    """Minus-strand remap matches the corpus label convention: a minus
    codon occupying genomic [x, x+3) in a window [a, a+wl) is labelled
    r = (a+wl-(x+3)) % 3 (cds_frame) and must land in global col
    3 + x%3 (the decode/_scan_codons convention)."""
    from omniphyl.annotate import _fuse_contig
    L, B = 1000, 50
    a, wl = 10, 500                        # a + wl = 510
    win_meta = [(0, a, wl)]
    for x in (200, 201):
        r = (a + wl - (x + 3)) % 3         # per-fragment label frame
        af = np.zeros((1, 8))
        af[0, 3 + r] = 1.0
        bp = _mk_bp(1, active_frame=af, struct_flags=None)
        res = _fuse_contig("c", L, [0], win_meta, bp, B, None, "rnaseq")
        g_exp = 3 + x % 3
        b = x // B                         # interior bin containing x
        check(f"rev x={x} -> col {g_exp}",
              abs(res.p_frame[b, g_exp] - 1.0) < 1e-6,
              res.p_frame[b, :6])
        check(f"rev x={x} no leakage",
              res.p_frame[b, [g for g in range(6)
                              if g != g_exp]].sum() == 0)


def test_fuse_impulse_placement():
    """struct_flags impulse is placed at a + offset*wl, scaled by the
    window's boundary prob, on the flagged channel only."""
    from omniphyl.annotate import _fuse_contig
    L, B = 1000, 50
    win_meta = [(0, 0, 500)]
    bp = _mk_bp(1, boundary=np.array([0.8]), offset=np.array([0.5]),
                region_fracs=None, active_frame=np.ones((1, 8)),
                struct_flags=np.array([[1.0, 0, 0, 0]]))
    res = _fuse_contig("c", L, [0], win_meta, bp, B, None, "rnaseq")
    # xhat = 250; nearest bin centers 225 (b4) / 275 (b5)
    exp = 0.8 * np.exp(-0.5 * (25 / 35.0) ** 2)
    check("impulse peak bin", int(np.argmax(res.b_tracks[:, 0])) in (4, 5))
    check("impulse magnitude",
          abs(res.b_tracks[4, 0] - exp) < 0.01,
          res.b_tracks[4, 0])
    check("only start channel", res.b_tracks[:, 1:].sum() == 0)


def test_fuse_region_fallback():
    """Without a region head, p_region falls back to coding/intergenic."""
    from omniphyl.annotate import _fuse_contig
    L, B = 1000, 50
    win_meta = [(0, 0, 500)]
    bp = _mk_bp(1, coding_frac=np.array([0.7]), region_fracs=None,
                active_frame=None)
    res = _fuse_contig("c", L, [0], win_meta, bp, B, None, "rnaseq")
    b = 4
    check("region[0] = coding", abs(res.p_region[b, 0] - 0.7) < 1e-6)
    check("region[3] = 1-coding", abs(res.p_region[b, 3] - 0.3) < 1e-6)
    check("intron/rna zero", res.p_region[b, 1:3].sum() == 0)


def test_apply_ablate():
    """_apply_ablate zeroes only the named tracks."""
    from omniphyl.annotate import _apply_ablate
    res = _mk_result(1000, 50, coding=[(0, 20, 0.8)],
                     frame=[(0, 20, 0, 0.9)], btr=[(0, 20, 0, 0.5)])
    _apply_ablate([res], {"frame", "flags"})
    check("p_frame zeroed", res.p_frame.sum() == 0)
    check("b_tracks zeroed", res.b_tracks.sum() == 0)
    check("p_region kept", res.p_region[:, 0].sum() > 0)


def test_annotate_e2e():
    print("[annotate e2e]")
    with tempfile.TemporaryDirectory() as td:
# --------------------------------------------- annotate end-to-end -------


        td = Path(td)
        ckpt, _, _ = _toy_model(td)
        # synthetic contig with an embedded fwd ORF: ATG ... TAA
        rng = np.random.default_rng(1)
        flank = "".join(rng.choice(list("ACGT")) for _ in range(150))
        orf = "Arev_frame_grid, test_TG" + "".join(rng.choice(
          ["AAA", "CAG", "GGA"])
                              for _ in range(60)) + "TAA"
        contig = flank + orf + flank
        fa = td / "g.fna"
        fa.write_text(f">c1\n{contig}\n")
        out = td / "ann"
        from omniphyl.annotate import run_annotate
        rep = run_annotate(
            model_path=ckpt, fasta_path=fa, out_prefix=out,
            window=50, stride=25, batch_size=8, device="cpu",
            verbose=False)
        check("report.json", (Path(str(out) + ".report.json")).exists())
        gff = (Path(str(out) + ".gff3")).read_text()
        check("gff3 header", gff.startswith("##gff-version 3"))
        check("report keys",
              set(rep) == {"taxonomy", "assembly", "features", "lexicmap"})
        check("assembly stats", rep["assembly"]["total_bp"] == len(contig))
        # the embedded ORF should produce at least one CDS feature
        check("CDS emitted", "CDS" in gff)


def main():
    fns = [test_io, test_frame, test_reconcile, test_pood_tau,
           test_cluster, test_feature_encoder,
           test_decode_start_scoring, test_decode_adjacent_strands,
           test_decode_frameshift_intron,
           test_decode_antisense_suppression,
           test_decode_intron_inside_coding_run,
           test_decode_chained_runs,
           test_softmasked_decode_and_evidence,
           test_splice_acceptor_snapping,
           test_hint_rescues_weak_partner_exon,
           test_fuse_cosine_support, test_fuse_fwd_frame_remap,
           test_fuse_rev_frame_remap, test_fuse_impulse_placement,
           test_fuse_region_fallback, test_apply_ablate,
           test_annotate_e2e]
    for f in fns:
        f()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
