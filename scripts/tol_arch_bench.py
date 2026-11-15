#!/usr/bin/env python3
"""Tree-of-Life mini benchmark harness (docs/tol-arch-plan.md §4.2).

Subcommands
-----------
stage1      Train baseline-v6 vs ToL Stage-1 arms on the packed mini-ToL
            corpus, eval both with eval_pilot, and emit
            ``stage1_gate_report.json`` covering gates S1-A (selective
            FDR + mammalian/plant species accuracy), S1-B (synthetic
            C_species bottleneck VRAM probe), S1-C (gap-lineage
            reconciliation consistency). Copies the ToL best checkpoint
            to ``<out>/stage1_mini_best.pt`` for the refiner bench.
bottleneck  Standalone S1-B probe: build OmniPhyl with a synthetic
            120k-class species head, run fwd+bwd steps, report peak
            VRAM / step time with and without ``head_bottleneck``.
refiner     Stage-1-cache the refiner panel genomes, train
            OmniPhylRefiner v2 and v1, run eval_refiner_panel on the
            held-out species, and emit ``refiner_gate_report.json``
            covering S2-A (microbial non-regression), S2-B (plant &
            metazoan multi-exon recovery + 5-30kb intron recall),
            S2-C (organelle/alt-code gcode detection + nuc F1).

Panel layout (produced by kaggle/tol-mini-corpus):
    refiner_panel/train/<name>.fna + <name>.gff3
    refiner_panel/heldout/<name>.fna + <name>.gff3
    refiner_panel/panel.json   {"<name>": {"clade": ..., "gcode": int}}

All training/eval runs via subprocess so failures surface with rc and
peak RSS stays bounded per step.
"""
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SCRIPTS = ROOT / "scripts"


def sh(argv: list, **kw) -> subprocess.CompletedProcess:
    """Run a repo script, streaming output; raise on nonzero exit."""
    print("+", " ".join(str(a) for a in argv), flush=True)
    r = subprocess.run([str(a) for a in argv], **kw)
    if r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {argv}")
    return r


def _device(req: str) -> str:
    if req != "auto":
        return req
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------- S1-B: bottleneck ---


def _synth_n_classes(n_species: int) -> dict:
    """Minimal vocab where only the species head is huge."""
    return {"domain": 5, "phylum": 9, "class": 9, "order": 9,
            "family": 9, "genus": 9, "species": n_species + 1,
            "coding_frac": 1, "boundary": 1, "offset": 1,
            "replicon": 4}


def _first_conv_in(m) -> int:
    import torch.nn as nn
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            return mod.in_channels
    raise SystemExit("no Conv2d in encoder")


def bottleneck_probe(n_classes: int, bottleneck: int, batch_size: int,
                     steps: int, device: str, width: int,
                     amp: str) -> dict:
    """fwd+bwd step probe on a synthetic wide species head.

    Returns params, peak bytes (CUDA) and ms/step. The species head is
    the only >8k-class head, so ``head_bottleneck`` engages exactly
    there (per WP1c semantics).
    """
    import torch
    from omniphyl.model import OmniPhyl

    nc = _synth_n_classes(n_classes)
    targets = list(nc)
    # plain encoder(x) path: aux / dna_stats heads would need the
    # dict-style input used by the trainer; the bottleneck question is
    # only about the species head's [B, 120k] logits, so they're off.
    m = OmniPhyl(nc, width=width, backbone="msfcgr", targets=targets,
                 head="flat", head_bottleneck=bottleneck,
                 bottleneck_min_classes=8192, replicon_head=True
                 ).to(device)
    cin = _first_conv_in(m)
    side = m.encoder.side if hasattr(m.encoder, "side") else 128
    x = torch.randn(batch_size, cin, side, side, device=device)
    y = torch.randint(0, n_classes + 1, (batch_size,), device=device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    autocast = (device == "cuda" and amp != "none")
    dt = torch.bfloat16 if amp in ("bf16", "auto") else torch.float16
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(steps):
        with torch.autocast("cuda", dtype=dt,
                            enabled=autocast):
            logits = m(x)[1]["species"]
            loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt_ms = (time.time() - t0) * 1000 / steps
    n_par = sum(p.numel() for p in m.parameters())
    head_par = sum(p.numel() for n, p in m.named_parameters()
                   if "heads.species" in n)
    return {"n_classes": n_classes + 1, "bottleneck": bottleneck,
            "batch_size": batch_size, "params": n_par,
            "species_head_params": head_par,
            "ms_per_step": round(dt_ms, 1),
            "peak_vram_gb": (round(torch.cuda.max_memory_allocated()
                                   / 2**30, 2)
                             if device == "cuda" else None)}


# ------------------------------------------------------------- stage1 -------


# v6-style baseline: resnet backbone, gated-aa aux, flat heads, none of
# the ToL additions (dna_stats / aa_extra_v2 / replicon / annot-ctx /
# struct+domain heads / var-len).
BASELINE_ARGS = ["--backbone", "resnet", "--width", "64",
                 "--pos-channels", "--aux", "aa", "--aa-mode", "gated",
                 "--aa-drop", "0.2", "--head", "flat",
                 "--flat-prior-tau", "0.5"]
# ToL arch per docs/tol-arch-locked contract (frozen values; any bench
# ablation that changes one must update docs/tol-arch-locked.json).
TOL_ARGS = ["--backbone", "msfcgr", "--width", "96",
            "--pos-channels", "--aux", "aa", "--aa-mode", "gated",
            "--aa-drop", "0.2", "--aa-extra-v2", "--dna-stats",
            "--head", "flat", "--flat-prior-tau", "1.0",
            "--head-bottleneck", "256", "--bottleneck-min-classes",
            "8192", "--struct-head", "--replicon-head",
            "--domain-head-k", "512", "--annot-ctx",
            "--var-len-prob", "0.25", "--min-train-len", "150"]


def _train_arm(out_dir: Path, name: str, packed: Path, labels: Path,
               arm_args: list, epochs: int, batch: int,
               extra: list) -> Path:
    out = out_dir / f"train_{name}"
    ck = out / "best.pt"
    if ck.exists():
        print(f"{name}: {ck} exists, skipping train")
        return ck
    sh([sys.executable, SCRIPTS / "train_omniphyl.py",
        "--packed", packed, "--labels", labels, "--out", out,
        "--epochs", epochs, "--batch-size", batch,
        "--amp", "auto", "--num-workers", "4", *arm_args, *extra])
    return ck


def _eval_arm(out_dir: Path, name: str, ckpt: Path, labels: Path,
              fragments: Path) -> dict:
    ev = out_dir / f"eval_{name}"
    res_files = list(ev.glob("*.json"))
    if res_files:
        return json.loads(res_files[0].read_text())
    sh([sys.executable, SCRIPTS / "eval_pilot.py",
        "--model", ckpt, "--labels", labels,
        "--fragments", fragments, "--outdir", ev,
        "--num-workers", "4"])
    res_files = list(ev.glob("*.json"))
    return json.loads(res_files[0].read_text())


def _selective(res: dict, rank: str, tau_key: str) -> dict:
    return (res.get("reconciled", {}).get("selective_val", {})
            .get(rank, {}).get(tau_key, {}))


def _labels_tsv_rows(path: Path):
    with open(path) as f:
        rd = csv.DictReader(f, delimiter="\t")
        yield from rd


def gate_s1a(res_tol: dict, labels: Path) -> dict:
    """FDR@tau0.5 reconciled species + mammalian/plant species acc +
    Homo sapiens screen retention."""
    sel = _selective(res_tol, "species", "tau_0.5")
    vg = res_tol.get("val_per_group", {})
    grp_acc = {}
    for g in ("vertebrate_mammalian", "plant"):
        rec = (vg.get(g, {}).get("reconciled_per_target", {})
               .get("species", {}).get("accuracy"))
        grp_acc[g] = rec
    human = any("homo sapiens" in (r.get("species") or "").lower()
                for r in _labels_tsv_rows(labels))
    ok = (sel.get("fdr") is not None and sel["fdr"] <= 0.10
          and all(a is not None and a >= 0.90 for a in grp_acc.values())
          and human)
    return {"pass": bool(ok), "fdr_tau05_species": sel.get("fdr"),
            "coverage_tau05_species": sel.get("coverage"),
            "group_species_acc": grp_acc,
            "homo_sapiens_in_labels": human}


def gate_s1b(n_classes: int, batch: int, device: str,
             amp: str) -> dict:
    """VRAM probe at synthetic C_species; gate = <16 GB bottlenecked."""
    bott = bottleneck_probe(n_classes, 256, batch, 20, device, 96, amp)
    row = {"pass": (bott["peak_vram_gb"] is not None
                    and bott["peak_vram_gb"] < 16.0)
           or bott["peak_vram_gb"] is None,
           "bottlenecked": bott}
    # full-head arm for contrast (kept small: same steps, params only
    # when it would obviously blow the budget is fine — head is the
    # dominant term at 120k classes)
    full = bottleneck_probe(n_classes, 0, batch, 20, device, 96, amp)
    row["full_head"] = full
    row["delta_gb"] = (round(full["peak_vram_gb"]
                           - bott["peak_vram_gb"], 2)
                       if device == "cuda" else None)
    return row


def gate_s1c(ckpt_path: Path, fragments: Path, n_seqs: int,
             device: str) -> dict:
    """Reconcile a sample of held-out windows: zero <unknown> lineages,
    100% ancestor consistency vs the ckpt's ancestor_of maps."""
    import torch
    from omniphyl.io import iter_sequences
    from omniphyl.model import OmniPhylPredictor, GAP_SENTINEL

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    anc = (ck.get("tree") or {}).get("ancestor_of") or {}
    if not anc:
        return {"pass": False, "reason": "no ancestor_of in ckpt"}
    ranks = [t for t in ck["targets"] if t in anc]
    ridx = {r: i for i, r in enumerate(ranks)}
    pred = OmniPhylPredictor(ckpt_path, device=device, verbose=False)
    seqs, n_unknown, n_bad, n_calls = [], 0, 0, 0
    for _h, s, _c in iter_sequences(fragments):
        seqs.append(s)
        if len(seqs) >= n_seqs:
            break
    for off in range(0, len(seqs), 256):
        bp = pred.predict_sequences(seqs[off:off + 256])
        calls = pred.reconcile_logp(bp.logp, conf_tau=0.0,
                                    anc_tau=0.0, p_ood=bp.p_ood)
        for c in calls:
            n_calls += 1
            if "<unknown>" in ";".join(c.names.values()):
                n_unknown += 1
            for r in ranks[1:]:
                cid = c.ids.get(r)
                if not cid:
                    continue
                row = anc[r][cid]
                for j in range(ridx[r]):
                    emit = c.ids.get(ranks[j])
                    exp = row[j] if j < len(row) else 0
                    if exp == GAP_SENTINEL:
                        if c.names.get(ranks[j]) != "incertae_sedis":
                            n_bad += 1
                    elif exp and emit and emit != exp:
                        n_bad += 1
    return {"pass": n_unknown == 0 and n_bad == 0,
            "n_calls": n_calls, "n_unknown_names": n_unknown,
            "n_ancestor_violations": n_bad}


def cmd_stage1(a) -> None:
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    labels = a.labels or a.packed / "labels.tsv"
    fragments = a.fragments
    report = {"corpus": str(a.packed), "arms": {}}

    ck_base = a.baseline_ckpt or _train_arm(
        out, "baseline", a.packed, labels, BASELINE_ARGS,
        a.epochs, a.batch_size, a.extra_train)
    ck_tol = a.tol_ckpt or _train_arm(
        out, "tol", a.packed, labels, TOL_ARGS,
        a.epochs, a.batch_size, a.extra_train)
    best = out / "stage1_mini_best.pt"
    if ck_tol != best:
        best.write_bytes(Path(ck_tol).read_bytes())

    res_base = _eval_arm(out, "baseline", ck_base, labels, fragments)
    res_tol = _eval_arm(out, "tol", ck_tol, labels, fragments)
    report["arms"]["baseline"] = _arm_summary(res_base)
    report["arms"]["tol"] = _arm_summary(res_tol)

    dev = _device(a.device)
    report["gates"] = {
        "S1-A_fdr_and_clade_recall": gate_s1a(res_tol, labels),
        "S1-B_bottleneck_vram": (gate_s1b(a.n_classes, a.probe_batch,
                                        dev, a.amp)
                                 if a.probe else {"pass": None,
                                                  "skipped": True}),
        "S1-C_gap_reconcile": gate_s1c(ck_tol, fragments,
                                       a.n_seqs, dev),
    }
    report["all_gates_pass"] = all(
        g.get("pass") for g in report["gates"].values()
        if g.get("pass") is not None)
    (out / "stage1_gate_report.json").write_text(
        json.dumps(report, indent=1))
    print(json.dumps({k: v.get("pass") for k, v in
                      report["gates"].items()}, indent=1))
    print(f"wrote {out / 'stage1_gate_report.json'}")


def _arm_summary(res: dict) -> dict:
    return {"per_target_val": res.get("per_target_val"),
            "reconciled": res.get("reconciled", {})
                             .get("per_target_val"),
            "val_per_group": {g: v.get("domain_accuracy")
                              for g, v in
                              res.get("per_group", {}).items()}}


# ------------------------------------------------------------- refiner ------


def _panel_pairs(d: Path) -> list[tuple[str, Path, Path]]:
    out = []
    for fna in sorted(d.glob("*.fna")) + sorted(d.glob("*.fasta")):
        gff = (fna.with_suffix(".gff3") if fna.with_suffix(".gff3")
               .exists() else fna.with_suffix(".gff"))
        if gff.exists():
            out.append((fna.stem, fna, gff))
    return out


def _stage1_caches(pairs, model: Path, cache_dir: Path,
                   dev: str) -> dict:
    """res.pkl per panel genome via train_refiner's cached splitter."""
    import train_refiner as tr
    ns = types.SimpleNamespace(window=500, stride=100, bin_width=0,
                               stage1_batch=256, device=dev)
    pks = {}
    for name, fna, _gff in pairs:
        pk = cache_dir / f"{fna.stem}.res.pkl"
        if not pk.exists():
            res = tr._stage1_cache(fna, model, cache_dir, ns)
            if res is None:
                print(f"WARNING: stage-1 cache failed for {name}")
                continue
            if not pk.exists():
                with open(pk, "wb") as fh:
                    pickle.dump(res, fh)
        pks[name] = pk
    return pks


def _panel_manifest(panel_dir: Path) -> dict:
    p = panel_dir / "panel.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _inferred_gcode(name: str):
    n = name.lower()
    if "chrm" in n or "mito" in n:
        return 2
    if "tetrahymena" in n or "paramecium" in n:
        return 6
    if "mycoplasma" in n or "mollicutes" in n:
        return 4
    return None


def _panel_role(name: str, meta: dict, euk: bool) -> str:
    """microbe | euk_multiexon | organelle from manifest or fallback."""
    clade = (meta.get("clade") or "").lower()
    if meta.get("gcode") or _inferred_gcode(name):
        return "organelle"
    if clade in ("bacteria", "archaea", "fungi", "protozoa",
                 "apicomplexa"):
        return "microbe"
    if clade in ("plant", "arthropod", "invertebrate", "vertebrate"):
        return "euk_multiexon"
    return "euk_multiexon" if euk else "microbe"


def _read_panel_gcode(pk: Path) -> int | None:
    try:
        with open(pk, "rb") as fh:
            d = pickle.load(fh)
        gcodes = {getattr(r, "gcode", None) for r in d["results"]}
        gcodes.discard(None)
        return gcodes.pop() if len(gcodes) == 1 else d.get("gcode")
    except Exception:
        return None


def cmd_refiner(a) -> None:
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    cache = a.cache_dir or out / "refiner_cache"
    dev = _device(a.device)
    train = _panel_pairs(a.panel_dir / "train")
    held = _panel_pairs(a.panel_dir / "heldout")
    manifest = _panel_manifest(a.panel_dir)
    if not train or not held:
        raise SystemExit("panel dir needs train/ and heldout/ "
                         "(<name>.fna + <name>.gff3 pairs)")
    print(f"panel: {len(train)} train, {len(held)} held-out genomes")

    pks = _stage1_caches(train + held, a.stage1_model, cache, dev)
    print(f"stage-1 caches: {len(pks)}/{len(train) + len(held)}")

    def train_cmd(arm: str, extra: list) -> Path:
        ck = out / f"refiner_{arm}.pt"
        if ck.exists():
            return ck
        argv = [sys.executable, SCRIPTS / "train_refiner.py",
                "--model", a.stage1_model, "--cache-dir", cache,
                "--out", ck, "--epochs", a.epochs,
                "--batch-size", a.batch_size, "--balance-genomes",
                "--cond-mode", a.cond_mode, "--amp", a.amp,
                "--device", dev]
        for _n, fna, gff in train:
            argv += ["--data", fna, gff]
        for _n, fna, gff in held:
            argv += ["--val-data", fna, gff]
        sh(argv + extra)
        return ck

    ck_v2 = a.v2_ckpt or train_cmd("v2", [])
    ck_v1 = a.v1_ckpt or train_cmd("v1", ["--v1-arch"])

    def run_panel(ck: Path, arm: str) -> dict:
        rp = out / f"panel_{arm}.json"
        if not rp.exists():
            argv = [sys.executable, SCRIPTS / "eval_refiner_panel.py",
                    "--ckpt", ck, "--device", dev, "--out", rp]
            for name, _fna, gff in held:
                if name in pks:
                    argv += ["--panel", f"{name},{pks[name]},{gff}"]
            sh(argv)
        return json.loads(rp.read_text())

    rep_v2 = run_panel(ck_v2, "v2")
    rep_v1 = run_panel(ck_v1, "v1")

    gates = {"S2-A_microbial_nonregression": {"panels": {}, "pass": True},
             "S2-B_multiexon_recovery": {"panels": {}, "pass": True},
             "S2-C_organelle_altcode": {"panels": {}, "pass": True}}
    for name in rep_v2["panels"]:
        r2, r1 = rep_v2["panels"][name], rep_v1["panels"].get(name, {})
        meta = manifest.get(name, {})
        role = _panel_role(name, meta, bool(r2.get("euk")))
        exp_gc = meta.get("gcode") or _inferred_gcode(name)
        if role == "organelle":
            got = (r2.get("gcode")
                   or (pks.get(name) and _read_panel_gcode(pks[name])))
            ok = (got == exp_gc
                  and (r2.get("nuc_f1") or 0) >= 0.85)
            gates["S2-C_organelle_altcode"]["panels"][name] = {
                "expected_gcode": exp_gc, "got_gcode": got,
                "nuc_f1": r2.get("nuc_f1"), "pass": bool(ok)}
            gates["S2-C_organelle_altcode"]["pass"] &= bool(ok)
        elif role == "microbe":
            ok = ((r2.get("nuc_f1") or 0)
                  >= (r1.get("nuc_f1") or 0) - 0.01
                  and (r2.get("exon_f1") or 0)
                  >= (r1.get("exon_f1") or 0))
            gates["S2-A_microbial_nonregression"]["panels"][name] = {
                "nuc_f1_v1": r1.get("nuc_f1"),
                "nuc_f1_v2": r2.get("nuc_f1"),
                "exon_f1_v1": r1.get("exon_f1"),
                "exon_f1_v2": r2.get("exon_f1"), "pass": bool(ok)}
            gates["S2-A_microbial_nonregression"]["pass"] &= bool(ok)
        else:
            band = (r2.get("intron", {}).get("by_len", {})
                    .get("5000_30000", {}))
            delta = ((r2.get("exon_f1") or 0)
                     - (r1.get("exon_f1") or 0))
            rec = band.get("recall")
            ok = delta >= 0.15 or (rec is not None and rec >= 0.40)
            gates["S2-B_multiexon_recovery"]["panels"][name] = {
                "exon_f1_delta": round(delta, 4),
                "intron_recall_5_30kb": rec,
                "n_band_introns": band.get("n"), "pass": bool(ok)}
            gates["S2-B_multiexon_recovery"]["pass"] &= bool(ok)
    report = {"v2_ckpt": str(ck_v2), "v1_ckpt": str(ck_v1),
              "gates": gates,
              "all_gates_pass": all(g["pass"] for g in gates.values())}
    (out / "refiner_gate_report.json").write_text(
        json.dumps(report, indent=1))
    print(json.dumps({k: v["pass"] for k, v in gates.items()},
                     indent=1))
    print(f"wrote {out / 'refiner_gate_report.json'}")


def cmd_bottleneck(a) -> None:
    dev = _device(a.device)
    rows = [bottleneck_probe(a.n_classes, a.bottleneck, a.batch_size,
                             a.steps, dev, a.width, a.amp)]
    if a.full_too:
        rows.append(bottleneck_probe(a.n_classes, 0, a.batch_size,
                                     a.steps, dev, a.width, a.amp))
    out = {"device": dev, "probe": rows}
    txt = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stage1")
    p.add_argument("--packed", required=True, type=Path)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--fragments", required=True, type=Path,
                   help="raw screened FASTA (eval_pilot input)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", default="bf16")
    p.add_argument("--extra-train", nargs="*", default=[])
    p.add_argument("--baseline-ckpt", type=Path, default=None)
    p.add_argument("--tol-ckpt", type=Path, default=None)
    p.add_argument("--probe", action="store_true",
                   help="also run the S1-B VRAM probe in-process")
    p.add_argument("--n-classes", type=int, default=120000)
    p.add_argument("--probe-batch", type=int, default=512)
    p.add_argument("--n-seqs", type=int, default=2000,
                   help="windows reconciled for the S1-C gap gate")
    p.set_defaults(fn=cmd_stage1)

    p = sub.add_parser("refiner")
    p.add_argument("--panel-dir", required=True, type=Path)
    p.add_argument("--stage1-model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cond-mode", default="reduced")
    p.add_argument("--amp", default="fp16")
    p.add_argument("--device", default="auto")
    p.add_argument("--v1-ckpt", type=Path, default=None)
    p.add_argument("--v2-ckpt", type=Path, default=None)
    p.set_defaults(fn=cmd_refiner)

    p = sub.add_parser("bottleneck")
    p.add_argument("--n-classes", type=int, default=120000)
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--amp", default="bf16")
    p.add_argument("--device", default="auto")
    p.add_argument("--full-too", action="store_true",
                   help="also probe the unbottlenecked head")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(fn=cmd_bottleneck)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
