#!/usr/bin/env python3
"""Tree-of-Life architecture invariant suite (docs/tol-arch-plan.md §4.1).

CPU-only, zero network. Covers:
  1. pack_corpus.py mmap store vs raw-FASTA FragmentDataset parity
  2. gap-bridging ancestor_of (-2 rankless sentinel) reconciliation
  3. genetic-code tables + per-contig infer_gcode + alt-code decode
  4. OmniPhylRefiner v2 geometry, strand-aware _phase_loss,
     canonical-isoform / ST_RNA / ST_DUAL labels, long-intron decode
  5. backward compat: v1 refiner ckpt (8192 chunk) + v6 Stage-1 ckpt
  6. checkpoint portability: no DDP/compile prefixes, CPU float32
Plus the profile.py clustering/host-filter invariants.

Run: python3 -m pytest tests/test_tol_arch.py  (or python3 tests/test_tol_arch.py)
"""
import tempfile
import types
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

from omniphyl.encode import (  # noqa: E402
    GCODE_LUTS, GCODE_STOPS, GCODE_STARTS, STOPS, STARTS,
    AA_ALPHA, build_cell_lut)
from omniphyl.model import (  # noqa: E402
    OmniPhyl, OmniPhylPredictor, GAP_SENTINEL)
from omniphyl.annotate import infer_gcode  # noqa: E402
from omniphyl.profile import (  # noqa: E402
    _kmer_ids, _kmer_base_entropy, _host_masked, cluster_reads,
    KMER_MIN_ENTROPY, MIN_FREQ_CAP)
from omniphyl.refiner import (  # noqa: E402
    OmniPhylRefiner, decode_with_refiner, load_refiner, save_refiner,
    clade_max_intron,
    ST_INTERGENIC, ST_CDS_P, ST_INT_P, ST_CDS_M, ST_INT_M,
    ST_RNA, ST_DUAL,
    S_START_P, S_STOP_P, S_DON_P, S_ACC_P,
    S_START_M, S_STOP_M, S_DON_M, S_ACC_M,
    N_STATES, N_SITES)
from pack_corpus import pack  # noqa: E402
from train_omniphyl import FragmentDataset  # noqa: E402
import train_refiner as tr  # noqa: E402

FAILS: list[str] = []


def check(name, ok, extra=""):
    tag = "ok" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  ({extra})" if extra and not ok else ""))
    if not ok:
        FAILS.append(name)


# -------------------------------------------------------- 1. packed parity ---


def _fake_corpus(tmp: Path):
    rng = np.random.default_rng(0)
    fna = tmp / "frags.fna"
    seqs = {}
    with fna.open("w") as f:
        for i in range(12):
            fid = f"f{i:03d}"
            s = "".join(rng.choice(list("ACGTacgt"), 500 + i * 7))
            seqs[fid] = s.upper()
            f.write(f">{fid}\n{s}\n")
    lbl = tmp / "labels.tsv"
    lbl.write_text("frag_id\tdomain\n" + "".join(
        f"{fid}\tBacteria\n" for fid in seqs))
    return fna, lbl, seqs


def _flatten(out):
    """Flatten a FragmentDataset item to a list of tensors."""
    flat = []
    if isinstance(out, dict):
        for k in sorted(out):
            flat += _flatten(out[k])
    elif isinstance(out, (tuple, list)):
        for v in out:
            flat += _flatten(v)
    else:
        flat.append(torch.as_tensor(np.asarray(out)))
    return flat


def test_packed_parity():
    print("[1] packed corpus parity]")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        fna, lbl, seqs = _fake_corpus(tmp)
        out = tmp / "corpus.packed"
        pack(fna, lbl, out)
        check("pack artifacts",
              all((out / f).exists() for f in
                  ("seqs.u8", "seq_offsets.npy", "seq_ids.txt",
                   "labels.tsv", "meta.json")))
        labels = {fid: {} for fid in seqs}
        lut = build_cell_lut(3)
        kw = dict(targets=["domain"], aux={"aa"}, dna_stats=True)
        d_raw = FragmentDataset(fna, labels, 3, lut, **kw)
        d_pk = FragmentDataset(None, labels, 3, lut, packed=out, **kw)
        check("same ids", d_raw.ids == d_pk.ids)
        seq_ok = all(d_pk._read_seq(f) == seqs[f] for f in seqs)
        check("mmap sequences identical", seq_ok)
        torch.manual_seed(0)
        np.random.seed(0)
        item_ok = True
        for i in range(len(d_raw)):
            a, b = d_raw[i], d_pk[i]
            fa, fb = _flatten(a), _flatten(b)
            item_ok &= len(fa) == len(fb) and all(
                torch.equal(x, y) for x, y in zip(fa, fb))
        check("encoded items bit-identical", item_ok)


# ---------------------------------------------------- 2. gap reconciliation ---


def test_gap_reconcile():
    print("[2] gap-aware ancestor reconciliation")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ranks = ["domain", "phylum", "class", "order"]
        vocab = {"domain": {"Eukaryota": 0},
                 "phylum": {"Euglenozoa": 0},
                 "class": {"Kinetoplastea": 0},
                 "order": {"Trypanosomatida": 0}}
        targets = ranks + ["coding_frac", "boundary", "offset"]
        n_classes = {t: len(vocab[t]) + 1 for t in vocab}
        n_classes.update({t: 1 for t in targets if t not in n_classes})
        # ancestor_of[r] [n_classes, ri+1]: Trypanosomatida sits under
        # Eukaryota>Euglenozoa with NO named NCBI class -> -2 gap, not 0.
        anc = {
            "domain": [[0], [1]],
            "phylum": [[0, 0], [1, 1]],
            "class": [[0, 0, 0], [1, 1, 1]],
            "order": [[0, 0, 0, 0],
                      [1, 1, GAP_SENTINEL, 1]],
        }
        m = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet",
                     side=8, targets=targets, head="flat")
        ckpt = {"model": m.state_dict(), "vocab": vocab, "k": 3,
                "targets": targets,
                "tree": {"ancestor_of": anc},
                "flat_log_prior": {t: [0.0, 0.0] for t in vocab},
                "args": {"width": 8, "backbone": "resnet",
                         "pos_channels": False, "k": 3, "head": "flat",
                         "aux": "", "flat_prior_tau": 0.5,
                         "prior_tau": 1.0}}
        p = tmp / "gap.pt"
        torch.save(ckpt, p)
        pred = OmniPhylPredictor(p, device="cpu", verbose=False)
        logp = {t: np.array([[-8.0, 8.0]], np.float32) for t in ranks}
        pood = {t: np.array([0.0], np.float32) for t in ranks}
        calls = pred.reconcile_logp(logp, conf_tau=0.0, anc_tau=0.0,
                                    p_ood=pood)
        c = calls[0]
        check("deepest rank = order", c.deepest_rank == "order",
              str(c.deepest_rank))
        check("no <unknown> in lineage",
              "<unknown>" not in ";".join(c.names.values()),
              str(c.names))
        check("gap rank -> incertae_sedis",
              c.names.get("class") == "incertae_sedis",
              str(c.names.get("class")))
        check("anchored ids",
              c.ids.get("domain") == 1 and c.ids.get("order") == 1,
              str(c.ids))


# --------------------------------------------------------- 3. genetic code ---


def _translate_lut(seq: str, g: int) -> list:
    lut = GCODE_LUTS[g]
    bases = {b: i for i, b in enumerate("ACGT")}
    out = []
    for i in range(0, len(seq) - 2, 3):
        c = seq[i:i + 3].upper()
        try:
            idx = (bases[c[0]] * 16 + bases[c[1]] * 4 + bases[c[2]])
            out.append(int(lut[idx]))
        except KeyError:
            out.append(-1)
    return out


def test_gcode():
    print("[3] genetic code tables + per-contig inference")
    need = {1, 2, 4, 5, 6, 11, 12, 16, 22, 23, 25}
    check("all LUTs present", need <= set(GCODE_LUTS), str(set(GCODE_LUTS)))
    check("code1 stops", GCODE_STOPS[1] == {"TAA", "TAG", "TGA"},
          str(GCODE_STOPS[1]))
    # vertebrate mt (2) & invertebrate mt (5) + mycoplasma (4) +
    # SR1 (25) all reassign TGA to Trp; ciliate (6) reassigns TAA/TAG
    for g in (2, 4, 5, 25):
        check(f"code{g} TGA is sense", "TGA" not in GCODE_STOPS[g],
              str(GCODE_STOPS[g]))
    check("code2 AGA/AGG stop",
          {"AGA", "AGG"} <= GCODE_STOPS[2], str(GCODE_STOPS[2]))
    check("code6 only TGA stops",
          "TAA" not in GCODE_STOPS[6] and "TAG" not in GCODE_STOPS[6]
          and "TGA" in GCODE_STOPS[6], str(GCODE_STOPS[6]))
    check("code11 standard stops", GCODE_STOPS[11] == GCODE_STOPS[1])
    # every code still starts at ATG
    check("ATG start universal",
          all("ATG" in GCODE_STARTS[g] for g in need))
    # LUT agrees with derived stop sets on a coding fragment
    orf = "ATG" + "GCT" * 4 + "TGA" + "AAA" + "TAA"
    for g in (1, 2, 4, 6):
        aas = _translate_lut(orf, g)
        n_stop = sum(1 for a in aas if a == AA_ALPHA - 1)
        expect = {1: 2, 2: 1, 4: 1, 6: 1}[g]
        check(f"code{g} translation stop count", n_stop == expect,
              f"{n_stop} vs {expect}")

    # --- per-contig inference ----------------------------------------
    # chrM in a chordate assembly: TGA at sense-codon rate -> code 2
    mt_counts = Counter({"TGA": 2600, "AAA": 500, "GCT": 500,
                         "CGT": 400})
    g = infer_gcode({"domain": "Eukaryota", "phylum": "Chordata"},
                    "mitochondrion", mt_counts, sum(mt_counts.values()))
    check("chrM -> code 2", g == 2, str(g))
    g = infer_gcode({"domain": "Eukaryota"}, "plastid",
                    Counter(), 0)
    check("plastid -> code 11", g == 11, str(g))
    g = infer_gcode({"domain": "Bacteria", "genus": "Escherichia"},
                    "nuclear_or_chrom", Counter(), 0)
    check("bacterial contig -> code 11", g == 11, str(g))
    g = infer_gcode({"domain": "Eukaryota", "genus": "Paramecium"},
                    "nuclear_or_chrom",
                    Counter({"TAA": 900, "TAG": 700, "AAA": 900,
                             "GCT": 900, "TGA": 60}),
                    3460)
    check("ciliate TAA/TAG-sense -> code 6", g == 6, str(g))

    # --- decode through an alt stop set -------------------------------
    # ORF: ATG @30 ... TGA @90 (sense under code 2) ... TAA @240
    L = 400
    seq = list("A" * L)
    seq[30:33] = "ATG"
    seq[90:93] = "TGA"
    seq[240:243] = "TAA"
    seq = "".join(seq)
    p_state = np.zeros((L, N_STATES), np.float32)
    p_state[:, ST_INTERGENIC] = 0.9
    p_state[30:243] = 0.0
    p_state[30:243, ST_CDS_P] = 0.9
    p_site = np.zeros((L, N_SITES), np.float32)
    p_site[30, S_START_P] = 0.95
    p_site[240, S_STOP_P] = 0.95
    nbins = (L + 499) // 500
    res = types.SimpleNamespace(
        length=L, bin_width=500,
        p_frame=np.zeros((nbins, 8), np.float32),
        taxon=types.SimpleNamespace(names={"domain": "Eukaryota"}))
    genes = decode_with_refiner(
        None, seq, res, stops=GCODE_STOPS[2], euk=False,
        p_state=p_state, p_site=p_site, max_intron=5000)
    check("code2 decode spans TGA", bool(genes) and
          genes[0]["end"] >= 243, str(genes[:1]))
    genes1 = decode_with_refiner(
        None, seq, res, stops=STOPS, euk=False,
        p_state=p_state, p_site=p_site, max_intron=5000)
    short = (not genes1) or genes1[0]["end"] < 100
    check("code1 decode truncates at TGA", short, str(genes1[:1]))


# ------------------------------------------------------- 4. refiner v2 core ---


def test_refiner_v2():
    print("[4] OmniPhylRefiner v2 forward / phase loss / decode")
    torch.manual_seed(0)
    m = OmniPhylRefiner(hidden=96, n_phase=6,
                        dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                        rnn_pool=8, tfm_pool=64, tfm_layers=2)
    L = 32768
    x = torch.randn(1, m.stem[0].in_channels, L)
    cond = torch.randn(1, m.blocks[0].film.in_features)
    sl, tl, pl = m(x, cond)
    check("v2 state shape", tuple(sl.shape[:2]) == (1, N_STATES)
          and sl.shape[-1] == L, str(tuple(sl.shape)))
    check("v2 site shape", tuple(tl.shape[:2]) == (1, N_SITES))
    check("v2 phase head", pl is not None and pl.shape[1] == 6)
    loss = sl.mean() + tl.mean() + pl.mean()
    loss.backward()
    check("v2 backward", all(p.grad is not None for p in
                             (m.stem[0].weight, m.rnn.weight_ih_l0)))

    # --- strand-aware phase loss -------------------------------------
    tlz = torch.zeros(1, N_SITES, L)
    yz = torch.zeros(1, N_SITES, L)
    # plus donor at 5000, true acceptor at 27000 (22 kb intron)
    yz[0, S_DON_P, 5000] = 1.0
    tlz[0, S_ACC_P, 27000] = 10.0
    lp = tr._phase_loss(tlz, yz, 35, 50000)
    check("+donor sees 22kb-away acceptor", float(lp) < 0.75, str(lp))
    # minus donor at 30000 (right edge of intron in fwd coords),
    # true acceptor at 10000 -> only a BACKWARD window finds it.
    yz2 = torch.zeros(1, N_SITES, L)
    tlz2 = torch.zeros(1, N_SITES, L)
    yz2[0, S_DON_M, 30000] = 1.0
    tlz2[0, S_ACC_M, 10000] = 10.0
    lm = tr._phase_loss(tlz2, yz2, 35, 50000)
    check("-donor uses backward window", float(lm) < 0.75, str(lm))
    # minus donor at < min_intron: partner window is entirely
    # before position 0 -> empty -> term = 1 - pd*0 (full penalty)
    yz3 = torch.zeros(1, N_SITES, L)
    tlz3 = torch.zeros(1, N_SITES, L)
    yz3[0, S_DON_M, 20] = 1.0
    lo = tr._phase_loss(tlz3, yz3, 35, 50000)
    check("orphan donor penalized", float(lo) > 0.9, str(lo))

    # --- canonical isoform + label states ------------------------------
    cds = {"tx_long": {"ivs": [(0, 90), (200, 290)]},
           "tx_short": {"ivs": [(0, 30)]},
           "tx_mane": {"ivs": [(0, 60)]}}
    tx2gene = {t: "g1" for t in cds}
    keep = tr.canonical_txs(cds, tx2gene, set())
    check("canonical = longest CDS", keep == {"tx_long"}, str(keep))
    keep = tr.canonical_txs(cds, tx2gene, {"tx_mane"})
    check("MANE overrides length", keep == {"tx_mane"}, str(keep))
    # a+ has an intron [200,300); b- CDS [150,250) overlaps the
    # first exon (dual) and part of the intron (stays CDS_M)
    rows = [("a", "+", (100, 200)), ("a", "+", (300, 400)),
            ("b", "-", (150, 250))]
    st, site, phase = tr.build_refiner_labels(rows, 600,
                                              rna_ivs=[(500, 550)])
    check("ST_DUAL on antisense overlap",
          st[160] == ST_DUAL, f"state={st[160]}")
    check("ST_RNA off-CDS", st[520] == ST_RNA, f"state={st[520]}")
    check("minus CDS inside + intron", st[220] == ST_CDS_M,
          f"state={st[220]}")
    check("ST_INT_P in gap", st[260] == ST_INT_P, f"state={st[260]}")
    check("+ phase wins dual region",
          set(np.unique(phase[150:200])) <= {0, 1, 2},
          str(np.unique(phase[150:200])))
    check("phase classes on - CDS",
          set(np.unique(phase[200:250])) <= {3, 4, 5},
          str(np.unique(phase[200:250])))
    # minus-strand site conventions: DONOR at intron_end-1 (right edge)
    rows_m = [("m", "-", (1000, 1200)), ("m", "-", (1400, 1600))]
    _, site_m, _ = tr.build_refiner_labels(rows_m, 2000)
    check("-donor at intron right edge",
          site_m[S_DON_M, 1399] == 1.0 and
          site_m[S_ACC_M, 1200] == 1.0,
          f"don={site_m[S_DON_M].argmax()} acc={site_m[S_ACC_M].argmax()}")

    # --- long-intron decode: 22 kb intron + 45 bp micro-exon -----------
    G = 24000
    s = list("A" * G)
    s[1000:1003] = "ATG"          # exon1 [1000, 1099)
    s[1099:1101] = "GT"           # donor, intron1 -> 23099
    s[23097:23099] = "AG"         # acceptor -> micro-exon [23099, 23144)
    s[23144:23146] = "GT"         # donor, intron2 -> 23544
    s[23542:23544] = "AG"         # acceptor -> exon3 [23544, 23637)
    s[23634:23637] = "TAA"        # stop (tx pos 234)
    seq = "".join(s)
    p_state = np.zeros((G, N_STATES), np.float32)
    p_state[:, ST_INTERGENIC] = 0.9
    for s0, e0, stc in ((1000, 1099, ST_CDS_P), (23099, 23144, ST_CDS_P),
                       (23544, 23637, ST_CDS_P),
                       (1099, 23099, ST_INT_P), (23144, 23544, ST_INT_P)):
        p_state[s0:e0] = 0.0
        p_state[s0:e0, stc] = 0.9
    p_site = np.zeros((G, N_SITES), np.float32)
    for ch, pos in ((S_START_P, 1000), (S_STOP_P, 23634),
                    (S_DON_P, 1099), (S_ACC_P, 23098),
                    (S_DON_P, 23144), (S_ACC_P, 23543)):
        p_site[pos, ch] = 0.95
    nbins = (G + 499) // 500
    res = types.SimpleNamespace(
        length=G, bin_width=500,
        p_frame=np.zeros((nbins, 8), np.float32),
        taxon=types.SimpleNamespace(
            names={"domain": "Eukaryota", "phylum": "Arthropoda"}))
    genes = decode_with_refiner(
        None, seq, res, euk=True, p_state=p_state, p_site=p_site)
    ok = bool(genes)
    g = genes[0] if ok else {}
    check("metazoan 22kb-intron gene recovered",
          ok and g["start"] <= 1010 and g["end"] >= 23630,
          str([(gg["start"], gg["end"], gg.get("introns"))
               for gg in genes[:3]]))
    check("micro-exon intron pair kept",
          ok and len(g.get("introns") or ()) == 2,
          str(g.get("introns")))
    # clade caps
    bac = types.SimpleNamespace(
        taxon=types.SimpleNamespace(names={"domain": "Bacteria"}))
    check("bacterial cap 5k", clade_max_intron(bac) == 5000)
    check("metazoan cap 50k", clade_max_intron(res) == 50000)
    unk = types.SimpleNamespace(taxon=None)
    check("default cap 15k", clade_max_intron(unk) == 15000)


# ------------------------------------------------ 5-6. compat + portability ---


def _toy_stage1(tmp: Path, arch_args: dict, tree: dict = None):
    vocab = {"domain": {"Bacteria": 0, "Eukaryota": 1},
             "phylum": {"Firmicutes": 0, "Apicomplexa": 1}}
    targets = ["domain", "phylum", "coding_frac", "boundary", "offset"]
    n_classes = {t: len(vocab[t]) + 1 for t in vocab}
    n_classes.update({t: 1 for t in targets if t not in n_classes})
    m = OmniPhyl(n_classes, width=8, cin=1, backbone="resnet",
                 side=8, targets=targets, head="flat")
    ckpt = {"model": m.state_dict(), "vocab": vocab, "k": 3,
            "targets": targets, "tree": tree or {},
            "flat_log_prior": {t: [0.0] * (len(vocab[t]) + 1)
                               for t in vocab},
            "args": {"width": 8, "backbone": "resnet",
                     "pos_channels": False, "k": 3, "head": "flat",
                     "aux": "", "flat_prior_tau": 0.5,
                     "prior_tau": 1.0, **arch_args}}
    p = tmp / "s1.pt"
    torch.save(ckpt, p)
    return p


def test_compat_portable():
    print("[5-6] backward compat + checkpoint portability")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # v1 refiner ckpt: no phase head, no transformer, 8192 chunk
        v1 = OmniPhylRefiner(hidden=32)
        cfg = {"format": "omniphyl-refiner-v1",
               "model": v1.state_dict(),
               "config": {"c_in": v1.stem[0].in_channels,
                          "cond_dim": v1.blocks[0].film.in_features,
                          "hidden": 32, "n_states": N_STATES,
                          "n_sites": N_SITES,
                          "dilations": list(v1.dilations),
                          "rnn_pool": v1.rnn_pool,
                          "chunk": 8192, "chunk_stride": 6144}}
        p1 = tmp / "ref_v1.pt"
        torch.save(cfg, p1)
        r1 = load_refiner(p1)
        check("v1 chunk geometry", r1.chunk == 8192
              and r1.chunk_stride == 6144)
        check("v1 no phase head", getattr(r1, "head_phase", None) is None)
        sl, tl, pl = r1(torch.randn(1, v1.stem[0].in_channels, 512),
                        torch.randn(1, v1.blocks[0].film.in_features))
        check("v1 forward 3rd out None", pl is None)

        # v2 refiner via save_refiner -> reload identical geometry
        v2 = OmniPhylRefiner(hidden=32, n_phase=6,
                             dilations=(1, 2, 4, 8, 16, 32, 64, 128),
                             rnn_pool=8, tfm_pool=64, tfm_layers=2)
        p2 = tmp / "ref_v2.pt"
        save_refiner(v2, p2, extra={"chunk": 32768,
                                    "chunk_stride": 24576})
        r2 = load_refiner(p2)
        check("v2 reload geometry", r2.chunk == 32768
              and getattr(r2, "head_phase", None) is not None
              and len(r2.blocks) == 8)

        # v6-style Stage-1: no bottleneck/replicon/dna_stats args
        s1 = _toy_stage1(tmp, {})
        pred = OmniPhylPredictor(s1, device="cpu", verbose=False)
        bp = pred.predict_sequences(["ACGTACGTACGT" * 10])
        check("v6 ckpt predicts", bp.logp["domain"].shape == (1, 3))
        # bottlenecked Stage-1: many-class head survives round-trip
        s1b = _toy_stage1(tmp, {"head_bottleneck": 256})
        pred2 = OmniPhylPredictor(s1b, device="cpu", verbose=False)
        check("bottleneck arg honored",
              pred2.train_args["head_bottleneck"] == 256)

        # portability: no DDP/compile prefixes, CPU float32 everywhere
        for pth in (p1, p2, s1):
            ck = torch.load(pth, map_location="cpu",
                            weights_only=False)
            sd = ck["model"]
            bad_pref = [k for k in sd
                        if k.startswith(("module.", "_orig_mod."))]
            bad_t = [(k, v.dtype, str(v.device)) for k, v in sd.items()
                     if torch.is_tensor(v) and v.is_floating_point()
                     and (v.device.type != "cpu"
                          or v.dtype != torch.float32)]
            check(f"{pth.name} clean prefixes", not bad_pref,
                  str(bad_pref[:3]))
            check(f"{pth.name} cpu/float32", not bad_t, str(bad_t[:3]))


# --------------------------------------------------- profile invariants ---


def test_profile_invariants():
    print("[+] profile cluster cap/entropy + host filter")
    # low-complexity k-mers cannot bridge reads
    hom = _kmer_ids("A" * 40, 21)
    var = _kmer_ids("ACGTGCAATCGGATCCGATCGTACGATGC", 21)
    check("homopolymer entropy below floor",
          _kmer_base_entropy(int(hom[0])) < KMER_MIN_ENTROPY)
    check("diverse kmer entropy above floor",
          _kmer_base_entropy(int(var[0])) > KMER_MIN_ENTROPY)
    # small-n clustering: floor keeps cap from degenerating
    rng = np.random.default_rng(2)
    locus = "".join(rng.choice(list("ACGT"), 300))
    reads = [locus[i:i + 150] for i in range(0, 150, 30)]
    groups = cluster_reads([f"r{i}" for i in range(len(reads))], reads,
                           min_shared=2, assemble=False)
    check("small-n reads cluster",
          max(len(mm) for mm, _ in groups) == len(reads),
          str([len(mm) for mm, _ in groups]))
    # host filter: host-derived unit dropped, foreign kept
    host = "GGCTA" * 100
    sk = set(_kmer_ids(host).tolist())
    ids, _, nd = _host_masked(["r1", "r2"],
                              [host, "ACGT" * 100], sk)
    check("host unit dropped", ids == ["r2"] and nd == 1, str(ids))


# ------------------------------------------------------------------------ main

ALL = [test_packed_parity, test_gap_reconcile, test_gcode,
       test_refiner_v2, test_compat_portable, test_profile_invariants]


def main() -> int:
    for fn in ALL:
        fn()
    if FAILS:
        print(f"\n{len(FAILS)} FAILURES: {FAILS}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
