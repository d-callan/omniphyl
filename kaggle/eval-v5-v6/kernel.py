"""Kaggle kernel: comprehensive evaluation of v5 (w32, w64) vs v6 (w32, w64).

Runs scripts/eval_pilot.py across:
  1. Native validation splits:
     - v5_w32, v5_w64 on omniphyl-v5-20k (--legacy-split --val-only)
     - v6_w32, v6_w64 on omniphyl-v6 (--val-only)
  2. Strictly leak-free cross-corpus validation split on omniphyl-v6:
     - Computes clean_val_accs = v6_val_accs - v5_train_accs (RefSeq
       accessions held out in v6 that were NEVER in v5's training set,
       comprising both shared v5∩v6 held-out genomes and newly added
       v6-only held-out RefSeq genomes).
     - Evaluates all 4 checkpoints (v5_w32, v5_w64, v6_w32, v6_w64) on
       this identical leak-free cohort (--val-accs-file).
"""

import glob
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
os.environ.setdefault("OMP_NUM_THREADS", "4")

subprocess.run(["git", "clone", "--depth", "1", REPO, str(WORK / "omniphyl")],
               check=True)
EVAL = str(WORK / "omniphyl/scripts/eval_pilot.py")
sys.path.insert(0, str(WORK / "omniphyl/scripts"))
sys.path.insert(0, str(WORK / "omniphyl/src"))


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern} under /kaggle/input; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return hits[0]


def find_corpus(tag: str) -> tuple[str, str]:
    hits = sorted(glob.glob("/kaggle/input/**/fragments.fna", recursive=True))
    for h in hits:
        if tag in h:
            d = os.path.dirname(h)
            return h, f"{d}/labels.tsv"
    raise SystemExit(f"corpus '{tag}' not found in {hits}")


v5_fna, v5_lab = find_corpus("v5-20k")
v6_fna, v6_lab = find_corpus("omniphyl-v6")

CKPTS = {
    "v5_w32": find_one("v5_w32_best.pt"),
    "v5_w64": find_one("v5_w64_best.pt"),
    "v6_w32": find_one("w32_e15_best.pt"),
    "v6_w64": find_one("w64_e15_best.pt"),
}
print("Checkpoints:", CKPTS, flush=True)
print("v5 corpus:", v5_fna, v5_lab, flush=True)
print("v6 corpus:", v6_fna, v6_lab, flush=True)

# 1. Native validation splits (includes val_accs, val_frag, and val_genus)
NATIVE_JOBS = [
    ("native_v5_w32", CKPTS["v5_w32"], v5_fna, v5_lab, ["--legacy-split"]),
    ("native_v5_w64", CKPTS["v5_w64"], v5_fna, v5_lab, ["--legacy-split"]),
    ("native_v6_w32", CKPTS["v6_w32"], v6_fna, v6_lab, []),
    ("native_v6_w64", CKPTS["v6_w64"], v6_fna, v6_lab, []),
]

for tag, ckpt, fna, lab, extra in NATIVE_JOBS:
    outdir = WORK / tag
    print(f"\n=== Running {tag} ===", flush=True)
    subprocess.run(
        [sys.executable, EVAL,
         "--model", ckpt,
         "--fragments", fna,
         "--labels", lab,
         "--outdir", str(outdir),
         "--val-only", "--cache-seqs", "--no-predictions-tsv",
         "--batch-size", "512", "--num-workers", "2"] + extra,
        check=True)

# 2. Construct strictly leak-free cross-generation held-out accession sets
from eval_pilot import load_labels  # noqa: E402
from train_omniphyl import build_split  # noqa: E402

SPLIT_COLS = {"accession", "rank_unresolved", "genus", "species",
              "seq_source", "seqid", "start"}
v5_meta = load_labels(Path(v5_lab), cols=SPLIT_COLS)
v5_val_accs, v5_hold_gen, _ = build_split(
    v5_meta, 0.1, 0.1, 0, 0.1, legacy=True)
v5_all_accs = {m["accession"] for m in v5_meta.values()}
v5_train_accs = {
    m["accession"] for m in v5_meta.values()
    if m["accession"] not in v5_val_accs
    and m.get("genus", "") not in v5_hold_gen
}
del v5_meta

v6_meta = load_labels(Path(v6_lab), cols=SPLIT_COLS)
v6_val_accs, _, _ = build_split(
    v6_meta, 0.05, 0.02, 0, 0.05, legacy=False)
del v6_meta

clean_val_accs = sorted(v6_val_accs - v5_train_accs)
shared_val_accs = sorted(set(clean_val_accs) & v5_val_accs)
delta_val_accs = sorted(set(clean_val_accs) - v5_all_accs)

clean_file = WORK / "clean_val_accs.txt"
clean_file.write_text("\n".join(clean_val_accs) + "\n")
(WORK / "clean_val_split.json").write_text(json.dumps({
    "n_v5_all_accs": len(v5_all_accs),
    "n_v5_val_accs": len(v5_val_accs),
    "n_v5_train_accs": len(v5_train_accs),
    "n_v6_val_accs": len(v6_val_accs),
    "n_clean_val_accs": len(clean_val_accs),
    "n_shared_v5_v6_val_accs": len(shared_val_accs),
    "n_v6_new_val_accs": len(delta_val_accs),
    "clean_val_accs": clean_val_accs,
    "shared_val_accs": shared_val_accs,
    "delta_val_accs": delta_val_accs,
}, indent=2))
print(f"\nLeak-free v6 val cohort: {len(clean_val_accs)} genomes "
      f"({len(shared_val_accs)} in v5_val ∩ v6_val, "
      f"{len(delta_val_accs)} new in v6_val)", flush=True)

for name, ckpt in CKPTS.items():
    outdir = WORK / f"clean_v6val_{name}"
    print(f"\n=== Running clean_v6val_{name} ===", flush=True)
    subprocess.run(
        [sys.executable, EVAL,
         "--model", ckpt,
         "--fragments", v6_fna,
         "--labels", v6_lab,
         "--outdir", str(outdir),
         "--val-only", "--cache-seqs", "--no-predictions-tsv",
         "--val-accs-file", str(clean_file),
         "--batch-size", "512", "--num-workers", "2"],
        check=True)

shutil.rmtree(WORK / "omniphyl", ignore_errors=True)
print("\nAll evaluations complete!", flush=True)
