"""Re-evaluate e1c + e2 checkpoints on the mini corpus, CPU only.

Verifies the flat_prior inference fix (bed723e): the training loop
applied tau*log_prior only during training, so raw ckpt logits were
prior-negated and eval argmax collapsed (0.0 acc, <unknown> preds).
This kernel clones HEAD (fixed eval_pilot) and re-scores both
checkpoints with --val-only.
"""
import glob
import subprocess
import sys

WORK = "/kaggle/working"
REPO = "https://github.com/d-callan/omniphyl.git"

subprocess.run(["git", "clone", "--depth", "1", REPO, f"{WORK}/omniphyl"],
               check=True)
EVAL = f"{WORK}/omniphyl/scripts/eval_pilot.py"

INPUT = "/kaggle/input"


def find(pattern):
    hits = glob.glob(f"{INPUT}/**/{pattern}", recursive=True)
    return hits[0] if hits else None


CKPTS = {
    "e1a": find("e1a/best.pt"),
    "e1b": find("e1b/best.pt"),
    "e1c": find("e1c/best.pt"),
    "e2": find("e2/best.pt"),
}
LABELS = find("mini/labels.tsv")
FRAGS = find("mini/fragments.fna")
if not LABELS or not FRAGS:
    raise SystemExit("missing mini/labels.tsv or mini/fragments.fna "
                     "under /kaggle/input")

for name, ckpt in CKPTS.items():
    if not ckpt:
        print(f"SKIP {name}: no best.pt in mounted dataset")
        continue
    outdir = f"{WORK}/eval_{name}"
    print(f"=== {name}: {ckpt} -> {outdir}")
    subprocess.run(
        [sys.executable, EVAL,
         "--model", ckpt,
         "--labels", LABELS,
         "--fragments", FRAGS,
         "--outdir", outdir,
         "--val-only",
         "--batch-size", "512", "--num-workers", "2"],
        check=True)

print("re-eval complete")
