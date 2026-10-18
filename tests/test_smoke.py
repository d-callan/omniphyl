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
    veupath_intervals, _collect_bed_items)
from train_omniphyl import (  # noqa: E402
    encode_cont_labels, CONT_TARGETS, RANKS, FragmentDataset,
    build_cell_lut, load_labels, build_tree, OmniPhyl)

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


# ----------------------------------------------------------- end to end ---

def make_corpus(tmp: Path):
    """2 accessions x 24 frags, mixture of pure/boundary/no-label rows."""
    rng = random.Random(1)
    fna, rows = [], []
    accs = [("GCA_000001.1", "Alpha proto", "Pseudomonada"),
            ("GCA_000002.1", "Beta proto", "Cryptosporida")]
    n = 0
    for acc, genus, family in accs:
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
                "species": f"{genus} testus"})
    fna_path = tmp / "fragments.fna"
    fna_path.write_text("".join(fna))
    lab_path = tmp / "labels.tsv"
    cols = ["frag_id", "accession", "seqid", "start", "end", "length",
            "taxid", "group", "coding", "label_source", "coding_frac",
            "boundary", "n_boundaries", "boundary_offset"] + RANKS
    with open(lab_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    return fna_path, lab_path


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

    out = tmp / "ckpt"
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "train_omniphyl.py"),
         "--fragments", str(fna), "--labels", str(lab),
         "--out", str(out), "--k", "3", "--width", "16",
         "--epochs", "2", "--batch-size", "16", "--val-frac", "0.5",
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
              test_slicing_and_encoding, test_tree_head,
              test_contaminant_screen, test_end_to_end):
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", *FAIL, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
