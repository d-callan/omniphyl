#!/usr/bin/env python3
"""Smoke tests for the OmniPhyl phase-1 machinery.

Covers, without network access:
  - label math (coding_stats / boundary / offset) on synthetic intervals
  - VEuPathDB-tier helpers (hub URL, chromAlias bridging, bigBed item walk)
  - fragment slicing + label-column encoding
  - end-to-end: synthetic corpus -> train_omniphyl (2 epochs, CPU)
                -> eval_pilot -> metrics JSON

Run:  python3 tests/test_smoke.py
"""

import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from fetch_refseq_corpus import (  # noqa: E402
    coding_stats, coding_stats_iv, slice_fragments, hub_url,
    veupath_intervals, _collect_bed_items,
    load_gff_features, feature_stats, cds_frame, subtract_intervals,
    DOMAIN_RE, parse_gtdb_lineage, attach_gtdb_lineage,
    select_assemblies, translate_frame, extract_orfs_for_fragment,
    annotate_fragments_pfam, _excl_ok, parse_group_map)
from train_omniphyl import (  # noqa: E402
    encode_cont_labels, encode_annot_labels, CONT_TARGETS, RANKS,
    FragmentDataset, build_cell_lut, load_labels, build_tree,
    build_flat_priors, OmniPhyl,
    fcgr_encode_ms, rc_index_map, MS_KS,
    hash_encode, aa_encode, aa_frame_encode, AA_FRAME_DIM,
    _AA_LUT, AA_LETTERS, AA_ALPHA,
    HASH_K, HASH_BUCKETS, HASH_PAD, HASH_LEN,
    varlen_crop, eval_crop_seq, PE_LEN)
from eval_pilot import (  # noqa: E402
    ancestor_chains, build_parent_of, path_scores, selective_metrics,
    scan_transition, auroc, auprc,
    FIXED_TAUS)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- units ---

def test_label_math():
    print("[label math]")
    cds = {"chr1": [(100, 200), (300, 400)], "chr2": [(0, 50)]}

    # Pure non-coding window, pure coding window, boundary window
    s = coding_stats("chr1", 400, 500, cds)
    check("noncoding window frac=0", s == (0.0, 0, None), s)
    s = coding_stats("chr1", 300, 400, cds)
    check("coding window frac=1", s[0] == 1.0 and s[1] == 0, s)
    s = coding_stats("chr1", 150, 350, cds)
    check("boundary window 0<frac<1", 0 < s[0] < 1, s)
    check("two interior edges counted", s[1] == 2, s[1])
    check("offset in (0,1)", s[2] is not None and 0 < s[2] < 1, s[2])
    check("uncovered seqid -> None",
          coding_stats("chrX", 0, 100, cds) is None)
    # Edges exactly at window bounds don't count; the interior end of
    # the first CDS (200) does. CDS starting at frag_end is excluded,
    # so frac=0.5 here.
    s = coding_stats("chr1", 100, 300, cds)
    check("flush edges excluded, interior kept", s == (0.5, 1, 0.5), s)


def test_veupath_helpers():
    print("[veupath tier]")
    check("hub_url chunking",
          hub_url("GCA_000256705.2") ==
          "https://hgdownload.soe.ucsc.edu/hubs/GCA/000/256/705/"
          "GCA_000256705.2",
          hub_url("GCA_000256705.2"))
    veu = {"Pf3D7_01_v3": [(10, 20)]}
    alias = {"NC_000001": frozenset({"NC_000001", "Pf3D7_01_v3"})}
    check("alias bridging resolves",
          veupath_intervals(veu, alias, "NC_000001") == [(10, 20)])
    check("unbridged seqid -> None",
          veupath_intervals(veu, {}, "NC_000002") is None)
    items = _collect_bed_items(
        {"track": "t", "genome": {"chr1": [
            {"chrom": "chr1", "chromStart": 0, "chromEnd": 9}],
            "meta": {"note": "skip me"}}})
    check("bed item walk finds items, skips metadata",
          len(items) == 1 and items[0]["chromEnd"] == 9)


def test_slicing_and_encoding():
    print("[slicing + encoding]")
    rng = random.Random(0)
    seq = "ACGT" * 300  # 1200 bp
    frags = list(slice_fragments(seq, 500, 500, 1000, 0.05, rng))
    check("nt-budget caps emitted bases",
          sum(len(f) for _, _, f in frags) <= 1000, len(frags))
    check("frag length honoured",
          all(len(f) == 500 for _, _, f in frags))
    check("coords sorted, inside seq",
          all(0 <= s < e <= len(seq) for s, e, _ in frags))

    rows = {"f1": {"coding_frac": "0.5000", "boundary_offset": "0.25"},
            "f2": {"coding_frac": "", "boundary_offset": ""},
            "f3": {"coding_frac": "0.0000", "boundary_offset": "NA"}}
    enc = encode_cont_labels(rows)
    check("encoded frac+offset", enc["f1"] == (0.5, 0.25), enc["f1"])
    check("missing -> -1 sentinel", enc["f2"] == (-1.0, -1.0))
    check("non-numeric offset -> -1", enc["f3"] == (0.0, -1.0))


def test_tree_head():
    print("[tree head / A1]")
    import torch
    rng = random.Random(2)
    # 2 phyla, each with 2 families; species nested under families
    rows = {}
    vocab = {r: {} for r in RANKS}
    lin = [("Bacteria", "P1", "C1", "O1", "F1", "G1", "S1"),
           ("Bacteria", "P1", "C1", "O1", "F2", "G2", "S2"),
           ("Bacteria", "P2", "C2", "O2", "F3", "G3", "S3")]
    for i, lin_row in enumerate(lin):
        rows[f"f{i}"] = {r: v for r, v in zip(RANKS, lin_row)}
    for m in rows.values():
        for r in RANKS:
            if m[r] not in vocab[r]:
                vocab[r][m[r]] = len(vocab[r])
    tree = build_tree(rows, vocab, RANKS)
    check("parent_of covers ranks", set(tree["parent_of"]) == set(RANKS))
    po = tree["parent_of"]["family"]  # [0, F1, F2, F3] -> parents O1,O1,O2
    fam_ids = [vocab["family"][n] + 1 for n in ("F1", "F2", "F3")]
    check("families link to their orders",
          [po[c].item() for c in fam_ids]
          == [vocab["order"][o] + 1 for o in ("O1", "O1", "O2")],
          po.tolist())
    # priors normalized within sibling groups
    lp = tree["log_prior"]["family"]
    g = {po[c].item() for c in fam_ids}
    check("one entry per parent group", len(g) == 2, g)

    n_classes = {r: len(vocab[r]) + 1 for r in RANKS}
    n_classes.update({t: 1 for t in CONT_TARGETS})
    model = OmniPhyl(n_classes, width=16, cin=1, side=8,
                     targets=RANKS + CONT_TARGETS, head="tree",
                     tree=tree)
    model.eval()
    with torch.no_grad():
        _, logits = model(torch.randn(4, 1, 8, 8))
        # marginal probs must be non-increasing down the tree for the
        # argmax lineage (tree coherence guarantee)
        marg = {t: logits[t][:, 1:].exp() for t in RANKS}
        cond = model.cond_logps(model.drop(torch.randn(4, 16 * 8)), tau=0.0)
    # marg(c) <= marg(parent(c)) for every class — the tree invariant;
    # root/orphan groups (parent 0) inherit mass 1 and are skipped
    mono = True
    for i in range(1, len(RANKS)):
        po = tree["parent_of"][RANKS[i]][1:]
        linked = po > 0
        if not bool(linked.any()):
            continue
        child = marg[RANKS[i]][:, linked]
        par = marg[RANKS[i - 1]][:, po[linked] - 1]
        mono &= bool((child <= par + 1e-5).all())
    check("marginals non-increasing down ranks", mono)
    # conditional probs sum to 1 within each sibling group
    ok = True
    for i in range(1, len(RANKS)):
        po = tree["parent_of"][RANKS[i]][1:]
        pc = cond[RANKS[i]].exp()
        for g in po.unique().tolist():
            s = pc[:, po == g].sum(1)
            ok &= bool(((s - 1).abs() < 1e-4).all())
    check("sibling conditionals sum to 1", ok)
    check("logit col0 pinned to -inf",
          bool((logits["species"][:, 0] < -1e3).all()))


def test_p0_eval_tools():
    """P0: flat priors, parent-of fallback, ancestor chains, path scores,
    and the FP-aware selective metrics."""
    print("[P0 eval tools]")
    import numpy as np
    import torch

    ranks = ["domain", "phylum", "class_"]
    # use a small synthetic vocab: D0={Bac}; P={Firm,Actino}; C={A,B,X}
    vocab = {"domain": {"Bac": 0},
             "phylum": {"Firm": 0, "Actino": 1},
             "class_": {"A": 0, "B": 1, "X": 2}}
    rows = []
    for i in range(10):   # Firm>A x5, Firm>B x4, Actino>X x1
        d, p, c = (("Bac", "Firm", "A") if i < 5 else
                   ("Bac", "Firm", "B") if i < 9 else
                   ("Bac", "Actino", "X"))
        rows.append({"domain": d, "phylum": p, "class_": c})

    # --- build_flat_priors: unconditional priors over train rows -------
    fp = build_flat_priors({str(i): m for i, m in enumerate(rows)},
                           vocab, ranks)
    check("flat prior shape", fp["class_"].shape == (4,), fp["class_"].shape)
    check("flat prior col0 pinned -20",
          float(fp["class_"][0]) == -20.0, fp["class_"].tolist())
    check("majority class gets largest prior",
          float(fp["class_"][1]) > float(fp["class_"][3]),
          fp["class_"].tolist())
    lp1 = fp["phylum"][1:].exp()
    check("priors sum ~1 over real classes",
          abs(float(lp1.sum()) - 1) < 0.02, float(lp1.sum()))

    # --- build_parent_of fallback matches build_tree parent votes ------
    po = build_parent_of(iter(rows), vocab, ranks)
    check("fallback parent_of class ids",
          po["class_"].tolist() == [0, 1, 1, 2], po["class_"].tolist())
    check("fallback parent_of phylum -> domain",
          po["phylum"].tolist() == [0, 1, 1], po["phylum"].tolist())

    # --- ancestor chains ------------------------------------------------
    chains = ancestor_chains(po, ranks)
    # class A (id 1): path [domain 1, phylum 1(Firm), self 1]
    check("chain for class A",
          chains["class_"][1].tolist() == [1, 1, 1],
          chains["class_"][1].tolist())
    check("chain for class X (Actino path)",
          chains["class_"][2 + 1].tolist() == [1, 2, 3],
          chains["class_"][3].tolist())

    # --- path_scores: consistent path wins over raw argmax flip -------
    # logits favour class B at class_ rank but its phylum evidence is
    # weak; class X is less favoured locally yet path-consistent.
    # (col 0 = unknown mask class; real logit tensors always include it)
    logp = {
        "domain": torch.log_softmax(torch.tensor([[-9.0, 8.0]]), 1),
        "phylum": torch.log_softmax(torch.tensor([[-9.0, 0.1, 4.0]]), 1),
        "class_": torch.log_softmax(torch.tensor([[-9.0, 2.0, 3.0, 1.0]]), 1)
    }
    s = path_scores(logp, chains["class_"], ranks)
    check("path score picks path-consistent class X",
          int(s.argmax(1)[0]) == 3, s.tolist())

    # --- selective_metrics: unseen-clade calls count as FP --------------
    preds = {"phylum": [1, 1, 2, 2]}      # 4 frags
    confs = {"phylum": [0.9, 0.9, 0.95, 0.3]}
    trues = {"phylum": [1, 2, 0, 0]}      # 2 correct, 2 OOV (genus-holdout)
    sm = selective_metrics([0, 1, 2, 3], preds, confs, trues, ["phylum"])
    r = sm["phylum"]["tau_0.9"]
    # at tau=0.9: frags 0,1,2 called; frag2 is OOV -> FP; frag1 wrong -> FP
    check("tau0.9 precision counts OOV as FP",
          r["n_called"] == 3 and r["n_fp"] == 2 and
          abs(r["precision"] - 1 / 3) < 0.01, r)
    check("cov_at_p90 zero when impossible",
          sm["phylum"]["cov_at_p90"] < 0.3, sm["phylum"]["cov_at_p90"])
    sm2 = selective_metrics([0, 1], preds, confs, trues, ["phylum"],
                            min_anc={"phylum": [0.9, 0.05, 0.9, 0.9]})
    av = sm2["phylum"]["ancestor_veto"]["t0.9_a0.2"]
    # frag0 emitted (conf .9, anc .9); frag1 vetoed (anc .05 < .2)
    check("ancestor veto suppresses weak-ancestor call",
          av["coverage"] == 0.5 and av["precision"] == 1.0, av)


# ----------------------------------------------------------- end to end ---

def test_msfcgr():
    print("[msfcgr / A2]")
    import numpy as np
    import torch
    rng = random.Random(3)
    seq = "".join(rng.choice("ACGT") for _ in range(500))
    rc = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    luts = {k: build_cell_lut(k) for k in MS_KS}
    canons = {k: rc_index_map(k) for k in MS_KS}

    # the core invariant: a fragment and its reverse complement must
    # produce identical inputs
    a = fcgr_encode_ms(seq, luts, canons)
    b = fcgr_encode_ms(rc, luts, canons)
    check("RC-invariant encoding", bool((a == b).all()))

    check("pyramid shape", a.shape == (len(MS_KS), 64, 64), a.shape)
    ap = fcgr_encode_ms(seq, luts, canons, pos=True)
    check("pos adds 2 channels", ap.shape == (len(MS_KS) + 2, 64, 64),
          ap.shape)

    # canonical palindromes collapse: each canonical index has <=2 raw
    # preimages, so the map is a valid idempotent projection
    for k in MS_KS:
        c = canons[k]
        check(f"k={k} canon idempotent", bool((c[c] == c).all()))

    # quadtree encoder: spatial halving is exact 2x2 at every stage
    n_classes = {r: 3 for r in RANKS}
    model = OmniPhyl(n_classes, width=8, cin=len(MS_KS),
                     backbone="msfcgr", targets=RANKS)
    model.eval()
    with torch.no_grad():
        _, logits = model(torch.randn(2, len(MS_KS), 64, 64))
    check("msfcgr forward -> rank logits",
          all(logits[r].shape == (2, 3) for r in RANKS))

    # E2 variable-length helpers
    np.random.seed(7)
    got_pe = got_sub = False
    for _ in range(60):
        c = varlen_crop(seq, 150)
        if c == seq[:PE_LEN] + "N" + seq[-PE_LEN:]:
            got_pe = True
        else:
            got_sub |= 150 <= len(c) <= 500 and c in seq
    check("varlen emits both sub-window and PE modes", got_pe and got_sub)
    # sequences at/below min_len must not hit randint(min_len, len)
    # with an empty range — they pass through unchanged
    check("varlen passes through short seqs",
          varlen_crop("ACGT" * 30, 150) == "ACGT" * 30 and
          varlen_crop("A" * 150, 150) == "A" * 150)
    check("eval_crop truncates to prefix",
          eval_crop_seq(seq, crop=250) == seq[:250] and
          len(eval_crop_seq(seq, crop=250)) == 250)
    check("eval_paired joins mates on N",
          eval_crop_seq(seq, paired=150) == seq[:150] + "N" + seq[-150:])
    check("crop no-ops on short seq",
          eval_crop_seq("ACGT" * 20, crop=250) == "ACGT" * 20 and
          eval_crop_seq("ACGT" * 20, paired=150) == "ACGT" * 20)


def test_aux_branches():
    print("[aux branches / A3]")
    import numpy as np
    import torch
    rng = random.Random(4)
    seq = "".join(rng.choice("ACGT") for _ in range(500))
    rc = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    # codon table: ATG->M (index of 'M'), TAA/TAG/TGA -> stop (20)
    import train_omniphyl as tr
    atg = 0 * 16 + 3 * 4 + 2          # A=0,T=3,G=2 digit order
    check("ATG -> M", _AA_LUT[atg] == AA_LETTERS.index("M"),
          int(_AA_LUT[atg]))
    check("stops -> 20",
          all(_AA_LUT[i] == AA_ALPHA - 1 for i in
              (3 * 16 + 0 * 4 + 0,   # TAA
               3 * 16 + 0 * 4 + 2,   # TAG
               3 * 16 + 2 * 4 + 0)))  # TGA

    canon11 = rc_index_map(HASH_K)
    a = hash_encode(seq, canon11)
    check("hash bag shape+dtype", a.shape == (HASH_LEN,) and
          a.dtype == np.int64, (a.shape, a.dtype))
    used = a[a != HASH_PAD]
    check("buckets in range",
          bool(((used >= 0) & (used < HASH_BUCKETS)).all()) and
          len(used) == len(seq) - HASH_K + 1, len(used))
    b = hash_encode(rc, canon11)
    check("hash RC-invariant multiset",
          sorted(used.tolist()) == sorted(b[b != HASH_PAD].tolist()))

    av = aa_encode(seq)
    check("aa spectrum shape", av.shape == (AA_ALPHA ** 3,),
          av.shape)
    check("aa RC-invariant multiset",
          sorted(av.tolist()) ==
          sorted(aa_encode(rc).tolist()))
    avx = aa_encode(seq, extra=True)
    check("aa extra dims", avx.shape == (AA_ALPHA ** 3 + 18,), avx.shape)
    check("aa extra feats in [0,1]",
          bool(((avx[-18:] >= 0) & (avx[-18:] <= 1)).all()),
          avx[-18:].tolist())

    # gated fusion: aa branch gated by z_dna, coding_frac on z_dna only
    n_classes = {r: 3 for r in RANKS}
    n_classes["coding_frac"] = 1
    gmodel = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet",
                      side=8, targets=RANKS + ["coding_frac"],
                      aux={"aa"}, aa_mode="gated", aa_drop=0.5)
    check("gated ckpt params exist",
          gmodel.aa_gate is not None and
          gmodel.coding_head_dna is not None and
          "coding_frac" not in gmodel.heads)
    gmodel.train()          # exercise branch-dropout path
    xg = {"x": torch.randn(2, 1, 8, 8),
          "aa": torch.rand(2, AA_ALPHA ** 3 + 18)}
    _, lg = gmodel(xg)
    check("gated forward emits all targets incl coding_frac",
          all(lg[t].shape[0] == 2 for t in RANKS + ["coding_frac"]) and
          lg["coding_frac"].shape == (2, 1), lg["coding_frac"].shape)

    bmodel = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet",
                      side=8, targets=RANKS, aux={"aa"},
                      aa_mode="bottleneck")
    bmodel.eval()
    _, lb = bmodel({"x": torch.randn(2, 1, 8, 8),
                    "aa": torch.rand(2, AA_ALPHA ** 3)})
    check("bottleneck forward ok",
          all(lb[t].shape == (2, 3) for t in RANKS))

    # dict-input forward through a fused model
    n_classes = {r: 3 for r in RANKS}
    model = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet",
                     side=8, targets=RANKS, aux={"hash", "aa"})
    check("aux widens latent", model.latent == 8 * 8 + 2 * 128,
          model.latent)
    model.eval()
    xb = {"x": torch.randn(2, 1, 8, 8),
          "hash": torch.randint(0, HASH_BUCKETS, (2, HASH_LEN)),
          "aa": torch.rand(2, AA_ALPHA ** 3)}
    with torch.no_grad():
        _, logits = model(xb)
    check("aux forward -> rank logits",
          all(logits[r].shape == (2, 3) for r in RANKS))


def make_corpus(tmp: Path):
    """4 accessions x 24 frags in 2 two-genome genera; mixture of
    pure/boundary/no-label rows. Two genera with 2 genomes each lets the
    e2e test exercise genus holdout + genome val + a non-empty train set."""
    rng = random.Random(1)
    fna, rows = [], []
    accs = [("GCA_000001.1", "Alpha proto", "Pseudomonada", "testus"),
            ("GCA_000002.1", "Alpha proto", "Pseudomonada", "beta"),
            ("GCA_000003.1", "Beta proto", "Cryptosporida", "testus"),
            ("GCA_000004.1", "Beta proto", "Cryptosporida", "gamma")]
    n = 0
    for acc, genus, family, epithet in accs:
        for j in range(24):
            n += 1
            seqid, start, end = "chr1", j * 500, j * 500 + 500
            fid = f"{acc}|{seqid}|{start}-{end}"
            seq = "".join(rng.choice("ACGT") for _ in range(500))
            fna.append(f">{fid}\n{seq}\n")
            # label pattern: thirds pure-coding / mixed / non-coding
            mode = j % 3
            frac = {0: "1.0000", 1: "0.5000", 2: "0.0000"}[mode]
            boff = "0.2500" if mode == 1 else ""
            rows.append({
                "frag_id": fid, "accession": acc, "seqid": seqid,
                "start": str(start), "end": str(end), "length": "500",
                "taxid": "1", "group": "protozoa",
                "coding": {"0": "coding", "1": "non-coding",
                           "2": "non-coding"}[str(mode)],
                "label_source": "veupathdb",
                "coding_frac": frac,
                "boundary": "1" if mode == 1 else "0",
                "n_boundaries": "1" if mode == 1 else "0",
                "boundary_offset": boff,
                "domain": "Eukaryota", "phylum": "Apicomplexa",
                "class": "Aconoidasida", "order": "Eucoccidiorida",
                "family": family, "genus": genus,
                "species": f"{genus} {epithet}",
                # P3 annot columns: introns on boundary windows, rna
                # nowhere, frame mirrors the coding mode, start flag on
                # boundary rows; IPR000001 on every row keeps the domain
                # vocab above the >=20-frag floor, PF00002 on a minority
                # stays below it (tests the threshold).
                "intron_frac": "0.2500" if mode == 1 else "0.0000",
                "rna_frac": "0.0000",
                "active_frame": {"0": "0", "1": "6",
                                 "2": "7"}[str(mode)],
                "struct_flags": "1" if mode == 1 else "0",
                "domain_ids": "IPR000001" +
                              (",PF00002" if mode == 0 else "")})
    fna_path = tmp / "fragments.fna"
    fna_path.write_text("".join(fna))
    lab_path = tmp / "labels.tsv"
    cols = ["frag_id", "accession", "seqid", "start", "end", "length",
            "taxid", "group", "coding", "label_source", "coding_frac",
            "boundary", "n_boundaries", "boundary_offset"] + RANKS + \
           ["intron_frac", "rna_frac", "active_frame",
            "struct_flags", "domain_ids"]
    with open(lab_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return fna_path, lab_path


def test_p3_annot():
    """P3: rich GFF features -> per-frag labels, annot label encoding,
    144-d frame summary, annotation heads forward, frame scanner."""
    print("[P3 annot heads]")
    import numpy as np
    import torch

    # --- fetcher: load_gff_features + feature_stats -------------------
    tmp = Path(tempfile.mkdtemp(prefix="omniphyl_gff_"))
    gff = tmp / "t.gff"
    gff.write_text(
        "##gff-version 3\n"
        "chr1\tref\tmRNA\t1\t900\t.\t+\t.\tID=tx1;Parent=g1\n"
        "chr1\tref\texon\t1\t400\t.\t+\t.\tID=e1;Parent=tx1\n"
        "chr1\tref\texon\t700\t900\t.\t+\t.\tID=e2;Parent=tx1\n"
        "chr1\tref\tCDS\t100\t400\t.\t+\t0\tID=cds1a;Parent=tx1\n"
        "chr1\tref\tCDS\t700\t850\t.\t+\t2\tID=cds1b;Parent=tx1;"
        "Dbxref=InterPro:IPR000001,PFAM:PF00001\n"
        "chr1\tref\trRNA\t1000\t1100\t.\t+\t.\tID=rrna1\n")
    feats = load_gff_features(gff)
    check("gff features parsed",
          feats["chr1"]["cds"] == [(99, 400), (699, 850)] and
          feats["chr1"]["exon"] == [(0, 400), (699, 900)] and
          len(feats["chr1"]["cds_recs"]) == 2,
          feats["chr1"])

    st = feature_stats(feats, "chr1", 150, 650)
    check("feature_stats coding frac", abs(st["frac"] - 0.5) < 1e-6,
          st["frac"])
    check("feature_stats intron frac",
          abs(st["intron"] - 0.5) < 1e-6, st["intron"])
    check("feature_stats rna zero", st["rna"] == 0.0)
    # CDS1a (+, phase 0, start 99): codon0 = 99 -> (99-150)%3 = 0
    check("feature_stats active frame", st["frame"] == 0, st["frame"])
    # splice donor at exon end 400 inside (150,650); start@99 and
    # acceptor@699 are outside
    check("feature_stats splice-donor bit", st["flags"] == 4,
          bin(st["flags"]))
    check("feature_stats no domains in window", st["domains"] == [],
          st["domains"])

    st2 = feature_stats(feats, "chr1", 600, 900)
    # CDS1b clip = 900-699 = 201 >= 90 -> domains present; phase 2 ->
    # codon0 = 699 + (3-2)%3 = 700 -> (700-600)%3 = 1
    check("feature_stats domains >=90bp clip",
          st2["domains"] == ["InterPro:IPR000001", "PFAM:PF00001"],
          st2["domains"])
    check("feature_stats phase-aware frame", st2["frame"] == 1,
          st2["frame"])
    # start@99 outside; stop@850 inside -> bit1; donor@400 outside;
    # acceptor@699 inside -> bit3  => 0b1010
    check("feature_stats stop+acceptor bits", st2["flags"] == 0b1010,
          bin(st2["flags"]))
    check("unannotated seqid -> None",
          feature_stats(feats, "chrX", 0, 500) is None)

    # interval subtraction: mrna (0,900) - exons -> intron (400,699)
    check("subtract_intervals intron",
          subtract_intervals([(0, 900)], [(0, 400), (699, 900)])
          == [(400, 699)])

    # precomputed per-seqid structures: the full-GFF case derives the
    # intron as mRNA minus (exon ∪ CDS), and the transcript group
    # carries transcript-oriented start/stop/donor/acceptor coords
    check("precomputed intron interval",
          feats["chr1"]["intron"] == [(400, 699)],
          feats["chr1"]["intron"])
    tr = feats["chr1"]["trans"]
    check("precomputed transcript transitions",
          len(tr) == 1 and tr[0][0] == 99 and tr[0][1] == 850 and
          tr[0][2] == [400] and tr[0][3] == [699],
          tr)

    # Parent= must win over protein_id=: two CDSs sharing a Parent but
    # carrying DIFFERENT protein ids (split-CDS annotation artefact)
    # must still group into ONE transcript, while a protein_id-only
    # GFF (no Parent= at all) still groups correctly.
    gff2 = tmp / "t2.gff"
    gff2.write_text(
        "##gff-version 3\n"
        # tx2: two CDSs, same Parent, divergent protein_ids -> one group,
        # one internal gap -> intron (400,500)
        "chrA\tref\tCDS\t100\t400\t.\t+\t0\tParent=tx2;"
        "protein_id=prot_a\n"
        "chrA\tref\tCDS\t500\t700\t.\t+\t0\tParent=tx2;"
        "protein_id=prot_b\n"
        # tx3: protein_id-only grouping on the minus strand
        "chrA\tref\tCDS\t900\t1000\t.\t-\t0\tprotein_id=prot_c\n"
        "chrA\tref\tCDS\t1100\t1200\t.\t-\t0\tprotein_id=prot_c;"
        "Note=InterPro:IPR999999\n")
    feats2 = load_gff_features(gff2)
    check("Parent groups split-CDS transcript",
          [(t[0], t[1]) for t in feats2["chrA"]["trans"]] ==
          [(99, 700), (1200, 899)],
          feats2["chrA"]["trans"])
    check("inter-CDS gap -> intron (no exon rows)",
          feats2["chrA"]["intron"] == [(400, 499), (1000, 1099)],
          feats2["chrA"]["intron"])
    check("minus-strand donor/acceptor orientation",
          feats2["chrA"]["trans"][1][2] == [1099] and
          feats2["chrA"]["trans"][1][3] == [1000],
          feats2["chrA"]["trans"])
    # DOMAIN_RE must find ids anywhere in the attributes column, not
    # only inside Dbxref= tokens
    cds_with_note = [r for r in feats2["chrA"]["cds_recs"]
                     if r[5] == ["InterPro:IPR999999"]]
    check("domain id found outside Dbxref attr",
          len(cds_with_note) == 1,
          [r[5] for r in feats2["chrA"]["cds_recs"]])
    check("widened DOMAIN_RE matches CDD/COG/TIGRFAM/PROSITE",
          DOMAIN_RE.findall(
              "Dbxref=CDD:23456,COG:COG0001;inference=protein motif:"
              "TIGRFAM:TIGR00123,PROSITE:PS01008") == [
              "CDD:23456", "COG:COG0001",
              "TIGRFAM:TIGR00123", "PROSITE:PS01008"])
    # GTDB lineage parser + species-interleaved genus selection
    glin = parse_gtdb_lineage(
        "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;"
        "o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;"
        "s__Escherichia coli")
    check("parse_gtdb_lineage 7 ranks",
          glin.get("domain") == "Bacteria" and
          glin.get("genus") == "Escherichia" and
          glin.get("species") == "Escherichia coli", glin)
    mock_rows = [
        {"assembly_accession": "GCF_1.1", "assembly_level": "Complete Genome",
         "version_status": "latest", "genome_rep": "Full",
         "excluded_from_refseq": "na", "ftp_path": "https://x/GCF_1",
         "species_taxid": "10", "group": "bacteria"},
        {"assembly_accession": "GCF_2.1", "assembly_level": "Complete Genome",
         "version_status": "latest", "genome_rep": "Full",
         "excluded_from_refseq": "na", "ftp_path": "https://x/GCF_2",
         "species_taxid": "10", "group": "bacteria"},
        {"assembly_accession": "GCF_3.1", "assembly_level": "Complete Genome",
         "version_status": "latest", "genome_rep": "Full",
         "excluded_from_refseq": "na", "ftp_path": "https://x/GCF_3",
         "species_taxid": "20", "group": "bacteria"},
    ]
    gmap = {
        "GCF_1": {**glin, "species": "Escherichia coli"},
        "GCF_2": {**glin, "species": "Escherichia coli"},
        "GCF_3": {**glin, "species": "Escherichia albertii"},
    }
    attach_gtdb_lineage(mock_rows, gmap)
    sel = select_assemblies(
        mock_rows, {"Complete Genome"}, "genus", random.Random(0),
        [], workers=1, per_rank_n=2, require_gff=False)
    sel_sp = {r["_gtdb_lin"]["species"] for r in sel}
    check("select_assemblies interleaves distinct species within genus",
          sel_sp == {"Escherichia coli", "Escherichia albertii"},
          sel_sp)
    mag_row = [{"assembly_accession": "GCA_9.1", "assembly_level": "Scaffold",
                "version_status": "latest", "genome_rep": "Full",
                "excluded_from_refseq": "derived from metagenome",
                "ftp_path": "https://x/GCA_9", "species_taxid": "99",
                "group": "archaea", "_gtdb_lin": glin}]
    sel_mag = select_assemblies(
        mag_row, {"Scaffold"}, "genus", random.Random(0),
        [], workers=1, per_rank_n=1, require_gff=False)
    check("GTDB-curated GenBank MAG passes excluded_from_refseq filter",
          len(sel_mag) == 1, len(sel_mag))
    euk_mag_rows = [
        {"assembly_accession": "GCA_10.1", "assembly_level": "Scaffold",
         "version_status": "latest", "genome_rep": "Full",
         "excluded_from_refseq": "derived from metagenome; genus undefined",
         "ftp_path": "https://x/GCA_10", "species_taxid": "10",
         "group": "protozoa"},
        {"assembly_accession": "GCA_11.1", "assembly_level": "Scaffold",
         "version_status": "latest", "genome_rep": "Full",
         "excluded_from_refseq": "derived from metagenome; contaminated",
         "ftp_path": "https://x/GCA_11", "species_taxid": "10",
         "group": "protozoa"},
    ]
    sel_euk_mag = select_assemblies(
        euk_mag_rows, {"Scaffold"}, None, random.Random(0),
        [], workers=1, per_rank_n=2, require_gff=False, include_mags=True)
    check("include_mags keeps eukaryotic MAG but blocks contaminated",
          [r["assembly_accession"] for r in sel_euk_mag] == ["GCA_10.1"],
          sel_euk_mag)
    # Deficit-aware covered dict + exclude_accs (incremental base_corpus)
    sel_def = select_assemblies(
        mock_rows, {"Complete Genome"}, "genus", random.Random(0),
        [], workers=1, per_rank_n=3, require_gff=False,
        covered={"Escherichia": 2}, covered_n=0,
        exclude_accs={"GCF_1.1", "GCF_3.1"})
    check("deficit-aware covered dict fills remaining slot & skips base accs",
          [r["assembly_accession"] for r in sel_def] == ["GCF_2.1"],
          [r["assembly_accession"] for r in sel_def])
    check("parse_group_map parses typed overrides",
          parse_group_map("protozoa=50000,fungi=35000", int)
          == {"protozoa": 50000, "fungi": 35000})
    # ORF extraction for pyhmmer Pfam scanner
    coding_dna = "ATG" + "GCT" * 35 + "TAA"
    orfs_f0 = extract_orfs_for_fragment(coding_dna, "0", min_aa=30)
    orfs_nc = extract_orfs_for_fragment(coding_dna, "7", min_aa=30)
    orfs_na = extract_orfs_for_fragment("GCT" * 160, "NA", min_aa=30)
    check("extract_orfs_for_fragment honours active_frame",
          len(orfs_f0) == 1 and len(orfs_f0[0][1]) == 36
          and orfs_nc == [] and len(orfs_na) == 2,
          (len(orfs_f0), len(orfs_nc), len(orfs_na)))
    try:
        import gzip
        import pyhmmer
        alpha = pyhmmer.easel.Alphabet.amino()
        hc_dna = ("ATG" + "GCTGACGAATTCGGCCACATCAAACTGTTCCCG" * 4 + "TAA")
        pep = extract_orfs_for_fragment(hc_dna, "0", min_aa=30)[0][1]
        msa_seqs = [pyhmmer.easel.TextSequence(
            name=f"s{i}".encode(), sequence=pep).digitize(alpha)
            for i in range(4)]
        msa = pyhmmer.easel.DigitalMSA(
            alpha, name=b"test", accession=b"PF99999.1", sequences=msa_seqs)
        hmm, _, _ = pyhmmer.plan7.Builder(alpha).build_msa(
            msa, pyhmmer.plan7.Background(alpha))
        pf_dir = tmp / "pfam_test"
        pf_dir.mkdir(exist_ok=True)
        with gzip.open(pf_dir / "Pfam-A.hmm.gz", "wb") as gz_out:
            hmm.write(gz_out)
        pf_fna = pf_dir / "fragments.fna"
        pf_fna.write_text(f">f1\n{hc_dna}\n>f2\n{hc_dna}\n")
        pf_lab = pf_dir / "labels.tsv"
        pf_lab.write_text(
            "frag_id\tactive_frame\tdomain_ids\n"
            "f1\t0\tNA\n"
            "f2\t7\tNA\n")
        n_added = annotate_fragments_pfam(
            pf_fna, pf_lab, pf_dir, workers=1, evalue=1e-3, hmm_path=None)
        rows_after = list(csv.DictReader(open(pf_lab), delimiter="\t"))
        check("annotate_fragments_pfam decompresses gz and merges Pfam hit",
              n_added == 1 and rows_after[0]["domain_ids"] == "Pfam:PF99999"
              and rows_after[1]["domain_ids"] == "",
              rows_after)
    except ImportError:
        pass

    # mRNA+CDS but NO exon rows: intron = mRNA minus CDS coverage — the
    # CDS portion of the transcript must NOT be labelled intronic.
    gff3 = tmp / "t3.gff"
    gff3.write_text(
        "##gff-version 3\n"
        "chrB\tref\tmRNA\t1\t900\t.\t+\t.\tID=tx4\n"
        "chrB\tref\tCDS\t100\t400\t.\t+\t0\tParent=tx4\n"
        "chrB\tref\tCDS\t700\t850\t.\t+\t2\tParent=tx4\n")
    feats3 = load_gff_features(gff3)
    check("no-exon GFF intron excludes CDS",
          feats3["chrB"]["intron"] == [(0, 99), (400, 699), (850, 900)],
          feats3["chrB"]["intron"])

    # --- annot label encoding -----------------------------------------
    rows = {
        "f1": {"coding_frac": "0.5", "intron_frac": "0.25",
               "rna_frac": "0.0", "active_frame": "1",
               "struct_flags": "5", "domain_ids": "IPR000001"},
        "f2": {"coding_frac": "1.0", "intron_frac": "0",
               "rna_frac": "0", "active_frame": "0",
               "struct_flags": "0", "domain_ids": ""},
        "f3": {"coding_frac": "", "intron_frac": "NA",
               "rna_frac": "NA", "active_frame": "NA",
               "struct_flags": "NA", "domain_ids": "NA"},
    }
    enc = encode_annot_labels(rows, {"IPR000001": 0, "PF9": 1})
    check("annot tuple width", len(enc["f1"]) == 9 + 2 + 1,
          len(enc["f1"]))
    check("region fracs + intergenic residual",
          enc["f1"][:4] == (0.5, 0.25, 0.0, 0.25), enc["f1"][:4])
    check("frame + flag bits decoded",
          enc["f1"][4] == 1.0 and enc["f1"][5:9] == (1.0, 0, 1.0, 0),
          enc["f1"][4:9])
    check("domain multi-hot + mask",
          enc["f1"][9:] == (1.0, 0.0, 1.0), enc["f1"][9:])
    check("empty domain_ids = annotated all-negative",
          enc["f2"][9:] == (0.0, 0.0, 1.0), enc["f2"][9:])
    check("NA labels masked",
          enc["f3"][:9] == (-1.0,) * 9 and enc["f3"][-1] == -1.0,
          enc["f3"])

    # --- aa_frame_encode -----------------------------------------------
    v = aa_frame_encode("ATG" * 167)
    check("aa_frame dim", v.shape == (AA_FRAME_DIM,), v.shape)
    check("aa_frame bounded",
          float(v.min()) >= 0 and float(v.max()) <= 1.0,
          (float(v.min()), float(v.max())))
    # pure-ATG forward frame 0 is ORF-clean (no stops); frame 1+ hit
    # stops. TGA-free seq -> stop_density 0 on F0.
    check("orf-clean forward frame", v[21] == 0.0 and v[22] == 1.0,
          (v[21], v[22]))

    # --- model heads ---------------------------------------------------
    n_classes = {r: 3 for r in RANKS}
    m = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet", side=8,
                 targets=RANKS, region_head=True, struct_head=True,
                 domain_k=4)
    m.eval()
    with torch.no_grad():
        _, lg = m({"x": torch.randn(2, 1, 8, 8),
                   "aa_frame": torch.rand(2, AA_FRAME_DIM)})
    check("annot head shapes",
          lg["region_fracs"].shape == (2, 4) and
          lg["active_frame"].shape == (2, 8) and
          lg["struct_flags"].shape == (2, 4) and
          lg["domains"].shape == (2, 4),
          {k: tuple(v.shape) for k, v in lg.items()
           if k in ("region_fracs", "active_frame", "struct_flags",
                    "domains")})
    # heads absent by default for back-compat
    m0 = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet", side=8,
                  targets=RANKS)
    _, lg0 = m0(torch.randn(2, 1, 8, 8))
    check("annot heads off by default",
          "region_fracs" not in lg0 and "active_frame" not in lg0)

    # --- frame scanner --------------------------------------------------
    seq = "GGT" * 33 + "ATG" + "GGT" * 32 + "TAG" + "GGT" * 32
    edge = scan_transition(seq, 0, "start", 90)
    check("scan start edge", edge == 99, edge)
    edge = scan_transition(seq, 0, "stop", 200)
    # TAG starts at 99+3+96=198 -> edge at 198+3=201
    check("scan stop edge", edge == 201, edge)
    # minus strand: CAT at fwd 90..93 -> ATG at rc offset L-93; edge=93
    seq_rc = "GGT" * 30 + "CAT" + "AAA" * 30
    edge = scan_transition(seq_rc, 3, "start", 90)
    check("scan minus-strand start edge", edge == 93, edge)
    check("scan misses canonical triplet -> None",
          scan_transition("AAA" * 100, 0, "start", 50) is None)
    check("auroc/auprc sanity",
          auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.9]) == 0.75 and
          auroc([0, 0, 0], [0.1, 0.2, 0.3]) is None and
          auprc([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) is not None)


def test_end_to_end():
    print("[end-to-end train+eval]")
    tmp = Path(tempfile.mkdtemp(prefix="omniphyl_smoke_"))
    fna, lab = make_corpus(tmp)

    # dataset shape check first (fast fail before subprocess)
    meta = load_labels(lab)
    lut = build_cell_lut(3)
    ds = FragmentDataset(fna, {fid: {} for fid in meta}, 3, lut,
                         pos=True, targets=[], cont=encode_cont_labels(meta))
    x, y, yc = ds[0]
    check("PFCGR image 3x8x8",
          tuple(x.shape) == (3, 8, 8), tuple(x.shape))
    check("cont label tuple floats",
          tuple(yc.shape) == (2,) and yc.is_floating_point(),
          yc.tolist())

    # var-len crop: window-derived cont targets describe the full 500bp
    # window, so a stochastic crop must mask them to -1 (unlabeled).
    # var_len_prob=1.0 forces a crop on every 500bp frag.
    import numpy as np
    np.random.seed(0)
    ds_v = FragmentDataset(fna, {fid: {} for fid in meta}, 3, lut,
                           pos=True, targets=[],
                           cont=encode_cont_labels(meta),
                           var_len_prob=1.0, min_train_len=150)
    _, _, yc_v = ds_v[0]
    check("cropped frag masks cont labels",
          bool((yc_v == -1.0).all()), yc_v.tolist())

    # vertex-permute augment must feed EVERY encoder the same
    # permuted molecule — previously only the FCGR image was remapped
    # while hash/aa/frame branches still saw the raw sequence.
    np.random.seed(3)
    ds_p = FragmentDataset(fna, {fid: {} for fid in meta}, 3, lut,
                           pos=True, targets=[], augment=True,
                           aux={"aa"})
    xp, _, _ = ds_p[0]
    np.random.seed(3)
    perm = np.random.permutation(4)   # first rng draw inside __getitem__
    tbl = str.maketrans("ACGTacgt",
                        "".join("ACGT"[d] for d in perm) * 2)
    s0 = ds_p._read_seq(ds_p.ids[0])
    check("permute reaches aa branch",
          np.allclose(xp["aa"].numpy(), aa_encode(s0.translate(tbl))),
          xp["aa"].shape)

    # in-RAM sequence cache: every frag_id resolves through the blob and
    # yields the exact same sequence as the seek-based path
    ds_c = FragmentDataset(fna, {fid: {} for fid in meta}, 3, lut,
                           pos=True, targets=[],
                           cont=encode_cont_labels(meta),
                           cache_seqs=True)
    check("cache_seqs: blob covers all ids",
          set(ds_c._seq_pos) == set(meta), len(ds_c._seq_pos))
    check("cache_seqs: sequences identical to seek path",
          all(ds_c._read_seq(f) == ds._read_seq(f) for f in
              list(meta)[:20]))

    out = tmp / "ckpt"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "train_omniphyl.py"),
         "--fragments", str(fna), "--labels", str(lab),
         "--out", str(out), "--k", "3", "--width", "16",
         "--epochs", "2", "--batch-size", "16", "--val-frac", "0.5",
         "--flat-prior-tau", "1.0", "--holdout-genus-frac", "0.5",
         "--num-workers", "0", "--pos-channels", "--cpu"],
        capture_output=True, text=True, timeout=600)
    check("train exits 0", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode else "")
    ckpts = list(out.glob("*.pt")) + list(out.glob("*.pth"))
    check("checkpoint written", len(ckpts) == 1,
          [p.name for p in ckpts])
    if not ckpts:
        print(r.stdout[-2000:], r.stderr[-2000:])
        return
    import torch
    ck = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    check("checkpoint has all targets",
          set(ck["targets"]) == set(RANKS) | set(CONT_TARGETS),
          ck["targets"])
    check("metric lines emitted",
          "frac_mae" in r.stdout and "boundary_acc" in r.stdout)
    check("flat ckpt saves parent map + flat priors",
          "tree" in ck and "parent_of" in ck["tree"] and
          "flat_log_prior" in ck and "species" in ck["flat_log_prior"],
          list(ck.get("tree", {})) + list(ck.get("flat_log_prior", {})))

    ev = tmp / "eval"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "eval_pilot.py"),
         "--model", str(ckpts[0]), "--labels", str(lab),
         "--fragments", str(fna), "--outdir", str(ev),
         "--num-workers", "0", "--cpu"],
        capture_output=True, text=True, timeout=600)
    check("eval exits 0", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode else "")
    res_files = list(ev.glob("*.json"))
    check("eval results json written", len(res_files) >= 1)
    if res_files:
        res = json.loads(res_files[0].read_text())
        check("annotation_heads section present",
              "annotation_heads" in res and
              "coding_frac" in res["annotation_heads"] and
              "boundary" in res["annotation_heads"] and
              "offset" in res["annotation_heads"],
              list(res.get("annotation_heads", {})))
        check("boundary recall@flag-rate emitted",
              "recall_at_flag_rate" in
              res.get("annotation_heads", {}).get("boundary", {}))
        check("predicted_class_entropy section present",
              "predicted_class_entropy" in res and
              "phylum" in res["predicted_class_entropy"],
              list(res.get("predicted_class_entropy", {})))
        check("selective precision section + tsv",
              "selective" in res and "val" in res["selective"] and
              (ev / "selective_precision.tsv").exists(),
              list(res.get("selective", {})))
        check("reconciled block emitted",
              "reconciled" in res and
              "selective_val" in res["reconciled"],
              list(res.get("reconciled", {})))
        # regression: s.max(1) returns (values, indices) — swapped
        # unpacking stored float log-probs as predicted class ids, so
        # reconciled accuracy was ~0. Every synthetic row is Eukaryota,
        # so domain reconciliation must be perfect post-fix.
        check("reconciled argmax returns class ids",
              res.get("reconciled", {}).get("per_target_val", {})
                 .get("domain", {}).get("accuracy", 0.0) > 0.99,
              res.get("reconciled", {}).get("per_target_val", {})
                 .get("domain"))

    # annot arm: region+struct+domain heads -> ckpt -> eval annot_heads
    out_a = tmp / "ckpt_annot"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "train_omniphyl.py"),
         "--fragments", str(fna), "--labels", str(lab),
         "--out", str(out_a), "--k", "3", "--width", "16",
         "--epochs", "2", "--batch-size", "16", "--val-frac", "0.5",
         "--region-head", "--struct-head", "--domain-head-k", "8",
         "--num-workers", "0", "--pos-channels", "--cpu"],
        capture_output=True, text=True, timeout=600)
    check("annot train exits 0", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode else "")
    ckpts_a = list(out_a.glob("*.pt"))
    if ckpts_a and r.returncode == 0:
        import torch
        cka = torch.load(ckpts_a[0], map_location="cpu",
                         weights_only=False)
        check("annot ckpt has the three head groups",
              "heads.region_fracs.weight" in cka["model"] and
              "head_frame.weight" in cka["model"] and
              "heads.domains.weight" in cka["model"],
              sorted(k for k in cka["model"] if "region" in k
                     or "frame" in k or "domains" in k))
        check("domain vocab respects >=20 floor",
              cka.get("domain_vocab") == {"IPR000001": 0},
              cka.get("domain_vocab"))
        ev_a = tmp / "eval_annot"
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "eval_pilot.py"),
             "--model", str(ckpts_a[0]), "--labels", str(lab),
             "--fragments", str(fna), "--outdir", str(ev_a),
             "--num-workers", "0", "--cpu"],
            capture_output=True, text=True, timeout=600)
        check("annot eval exits 0", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
        res_a = list(ev_a.glob("*.json"))
        if res_a:
            res = json.loads(res_a[0].read_text())
            ah = res.get("annot_heads", {})
            check("annot_heads sections present",
                  "region_fracs" in ah and "active_frame" in ah
                  and "struct_flags" in ah and "domains" in ah,
                  list(ah))
            fs = res.get("split_metrics", {}).get(
                "val", {}).get("annot_heads", {}).get("frame_scan", {})
            check("annot frame-scan gate ran",
                  bool(fs), list(res.get("split_metrics", {})
                                 .get("val", {})
                                 .get("annot_heads", {})))
            check("frame-scan emits unconditional + flagged rates",
                  not fs or all(
                      "unconditional_hit_rate_3bp" in fs.get(k, {}) and
                      "flagged_hit_rate_3bp" in fs.get(k, {})
                      for k in ("start", "stop")),
                  fs)
            rf = res.get("split_metrics", {}).get(
                "val", {}).get("annot_heads", {}).get("region_fracs", {})
            check("intron-vs-intergenic AUROC emitted",
                  not rf or "intron_vs_intergenic_auroc" in rf, rf)
            check("reconciled genome consensus emitted",
                  "genome_consensus" in res.get("reconciled", {}),
                  list(res.get("reconciled", {})))

    # gated aa arm: exercises aa_extra encoding -> gated ckpt ->
    # eval_pilot reconstruction (aa_mode/aa_extra round-trip)
    out_g = tmp / "ckpt_gated"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "train_omniphyl.py"),
         "--fragments", str(fna), "--labels", str(lab),
         "--out", str(out_g), "--k", "3", "--width", "16",
         "--epochs", "2", "--batch-size", "16", "--val-frac", "0.5",
         "--aux", "aa", "--aa-mode", "gated", "--aa-drop", "0.2",
         "--num-workers", "0", "--pos-channels", "--cpu"],
        capture_output=True, text=True, timeout=600)
    check("gated train exits 0", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode else "")
    ckpts_g = list(out_g.glob("*.pt"))
    if ckpts_g and r.returncode == 0:
        import torch
        ckg = torch.load(ckpts_g[0], map_location="cpu",
                         weights_only=False)
        check("gated ckpt has gate + dna coding head params",
              "aa_gate.weight" in ckg["model"] and
              "coding_head_dna.weight" in ckg["model"] and
              "heads.coding_frac.weight" not in ckg["model"],
              sorted(k for k in ckg["model"] if "aa_" in k or
                     "coding" in k))
        ev_g = tmp / "eval_gated"
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "eval_pilot.py"),
             "--model", str(ckpts_g[0]), "--labels", str(lab),
             "--fragments", str(fna), "--outdir", str(ev_g),
             "--num-workers", "0", "--cpu"],
            capture_output=True, text=True, timeout=600)
        check("gated eval exits 0", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")

    # annot_ctx arm: detached tax->annot posterior conditioning moves
    # the optional heads out of heads{} under new names with wider
    # inputs; eval_pilot must rebuild the same layout from ckpt args.
    out_c = tmp / "ckpt_annot_ctx"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "train_omniphyl.py"),
         "--fragments", str(fna), "--labels", str(lab),
         "--out", str(out_c), "--k", "3", "--width", "16",
         "--epochs", "2", "--batch-size", "16", "--val-frac", "0.5",
         "--region-head", "--struct-head", "--domain-head-k", "8",
         "--annot-ctx", "--annot-weight", "0.5",
         "--num-workers", "0", "--pos-channels", "--cpu"],
        capture_output=True, text=True, timeout=600)
    check("annot-ctx train exits 0", r.returncode == 0,
          r.stderr.strip().splitlines()[-1] if r.returncode else "")
    ckpts_c = list(out_c.glob("*.pt"))
    if ckpts_c and r.returncode == 0:
        import torch
        ckc = torch.load(ckpts_c[0], map_location="cpu",
                         weights_only=False)
        check("annot-ctx ckpt uses detached-ctx head modules",
              "head_region.weight" in ckc["model"] and
              "head_domains.weight" in ckc["model"] and
              "heads.region_fracs.weight" not in ckc["model"],
              sorted(k for k in ckc["model"]
                     if "region" in k or "domains" in k))
        check("annot_ctx flag round-trips in ckpt args",
              ckc.get("args", {}).get("annot_ctx") is True)
        ev_c = tmp / "eval_annot_ctx"
        r = subprocess.run(
            [sys.executable, str(SCRIPTS / "eval_pilot.py"),
             "--model", str(ckpts_c[0]), "--labels", str(lab),
             "--fragments", str(fna), "--outdir", str(ev_c),
             "--num-workers", "0", "--cpu"],
            capture_output=True, text=True, timeout=600)
        check("annot-ctx eval exits 0", r.returncode == 0,
              r.stderr.strip().splitlines()[-1] if r.returncode else "")
        res_c = list(ev_c.glob("*.json"))
        if res_c:
            res = json.loads(res_c[0].read_text())
            check("annot-ctx eval emits annot_heads sections",
                  "region_fracs" in res.get("annot_heads", {}) and
                  "domains" in res.get("annot_heads", {}),
                  list(res.get("annot_heads", {})))


def test_contaminant_screen():
    """Layer-3 accession exemption: an accession that is itself a contaminant
    species keeps its fragments; isolated hits are still rejected."""
    import screen_contaminants as sc

    tmp = Path(tempfile.mkdtemp())
    fna, lab, out = tmp / "fragments.fna", tmp / "labels.tsv", tmp / "out"
    contam_seq = "A" * 200          # every 21-mer = poly-A canonical k-mer
    clean_seq = "ACGT" * 50         # real-looking, no poly-A k-mers
    # GCF_X: all frags contaminant-like -> accession exempted wholesale
    # GCF_Y: 1 of 4 frags contaminated -> just that frag rejected
    rows = ([("X|0", "GCF_X.1", contam_seq), ("X|1", "GCF_X.1", contam_seq),
             ("Y|0", "GCF_Y.1", contam_seq)] +
            [(f"Y|{i}", "GCF_Y.1", clean_seq) for i in (1, 2, 3)])
    with open(fna, "w") as f:
        for fid, _, seq in rows:
            f.write(f">{fid}\n{seq}\n")
    with open(lab, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["frag_id", "accession"])
        for fid, acc, _ in rows:
            w.writerow([fid, acc])

    old_argv, old_loader = sys.argv, sc.load_contaminant_sketches
    sys.argv = ["screen_contaminants.py", "--fragments", str(fna),
                "--labels", str(lab), "--outdir", str(out)]
    sc.load_contaminant_sketches = lambda cd, hf: {"fake": sc.kmers(contam_seq)}
    try:
        sc.main()
        ran = True
    except Exception as e:
        print(f"    screen raised: {e!r}")
        ran = False
    finally:
        sys.argv, sc.load_contaminant_sketches = old_argv, old_loader
    check("screen runs on synthetic corpus", ran)
    if not ran:
        return

    man = json.loads((out / "screen_manifest.json").read_text())
    check("contaminant-species member exempted",
          man["contaminant_member_accessions"] == ["GCF_X.1"],
          str(man["contaminant_member_accessions"]))
    check("isolated hit still rejected",
          man["n_rejected"] == 1 and man["n_passed"] == 5,
          f"passed={man['n_passed']} rejected={man['n_rejected']}")
    with open(out / "labels_screened.tsv") as f:
        passed = {r["frag_id"]
                  for r in csv.DictReader(f, delimiter="\t")}
    check("member frags + clean frags kept",
          passed == {"X|0", "X|1", "Y|1", "Y|2", "Y|3"},
          str(sorted(passed)))
    with open(out / "fragments_rejected.tsv") as f:
        rej = list(csv.DictReader(f, delimiter="\t"))
    check("rejected row is the contamination island",
          len(rej) == 1 and rej[0]["frag_id"] == "Y|0" and
          rej[0]["reason"] == "contaminant_hit",
          str(rej))


def main():
    for t in (test_label_math, test_veupath_helpers,
              test_slicing_and_encoding, test_tree_head, test_msfcgr,
              test_aux_branches, test_p0_eval_tools, test_p3_annot,
              test_contaminant_screen, test_end_to_end):
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", *FAIL, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
