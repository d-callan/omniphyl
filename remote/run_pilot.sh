#!/usr/bin/env bash
# OmniPhyl pilot: clone -> smoke test -> fetch corpus -> train -> eval.
#
# Env-agnostic runner: invoked by kaggle/kernel.py on Kaggle Kernels,
# `colab run --gpu T4 remote/run_pilot.sh` on Colab, or any bash shell
# locally. All knobs are env vars (see docs/remote-runs.md):
#
#   REPO_URL        clone source (default: public GitHub URL; use
#                   https://<token>@github.com/d-callan/omniphyl.git if private)
#   REF             branch/tag/commit          (default: main)
#   WORKDIR         clone + corpus workspace   (default: $PWD/work)
#   OUTDIR          result tarball dir         (default: $PWD/output)
#   GROUPS          --groups value             (default: "bacteria viral")
#   LEVELS          --levels value             (default: "Complete Genome,Chromosome")
#   NT_BUDGET       nt per seqid               (default: 100000)
#   FRAG_LEN        fragment length            (default: 500)
#   BOUNDARY_FRAC   mixed-window fraction      (default: 0.3)
#   VEUPATH         1 = pass --veupath         (default: 1)
#   K               FCGR k-mer order           (default: 7)
#   WIDTH           ResNet width               (default: 64)
#   EPOCHS          training epochs            (default: 30)
#   BATCH           batch size                 (default: 256)
#   POS_CHANNELS    1 = --pos-channels         (default: 1)
#   VAL_FRAC        held-out genome fraction   (default: 0.1)
#   SMOKE           1 = run tests first        (default: 1)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/d-callan/omniphyl.git}"
REF="${REF:-main}"
WORKDIR="${WORKDIR:-$PWD/work}"
OUTDIR="${OUTDIR:-$PWD/output}"
GROUPS="${GROUPS:-bacteria viral}"
LEVELS="${LEVELS:-Complete Genome,Chromosome}"
NT_BUDGET="${NT_BUDGET:-100000}"
FRAG_LEN="${FRAG_LEN:-500}"
BOUNDARY_FRAC="${BOUNDARY_FRAC:-0.3}"
VEUPATH="${VEUPATH:-1}"
K="${K:-7}"
WIDTH="${WIDTH:-64}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-256}"
POS_CHANNELS="${POS_CHANNELS:-1}"
VAL_FRAC="${VAL_FRAC:-0.1}"
SMOKE="${SMOKE:-1}"

mkdir -p "$WORKDIR" "$OUTDIR"
cd "$WORKDIR"

# --- clone (or update a cached clone, e.g. on Drive) -------------------------
if [ ! -d omniphyl/.git ]; then
    git clone --depth 1 --branch "$REF" "$REPO_URL" omniphyl
else
    git -C omniphyl fetch --depth 1 origin "$REF"
    git -C omniphyl checkout "origin/$REF" || true
fi
cd omniphyl

python3 -c "import torch" 2>/dev/null || pip install torch numpy --quiet

if [ "$SMOKE" = 1 ]; then
    python3 tests/test_smoke.py
fi

# --- corpus -----------------------------------------------------------------
DATA="$WORKDIR/corpus"
VEUPATH_FLAG=""
[ "$VEUPATH" = 1 ] && VEUPATH_FLAG="--veupath"
# shellcheck disable=SC2086
python3 scripts/fetch_refseq_corpus.py --outdir "$DATA" \
    --groups $GROUPS --levels "$LEVELS" \
    --nt-budget "$NT_BUDGET" --frag-len-min "$FRAG_LEN" \
    --frag-len-max "$FRAG_LEN" --boundary-frac "$BOUNDARY_FRAC" \
    $VEUPATH_FLAG --workers 8

# --- train ------------------------------------------------------------------
RUN="$WORKDIR/run"
POS_FLAG=""
[ "$POS_CHANNELS" = 1 ] && POS_FLAG="--pos-channels"
python3 scripts/train_omniphyl.py \
    --fragments "$DATA/fragments.fna" --labels "$DATA/labels.tsv" \
    --out "$RUN" --k "$K" --width "$WIDTH" --epochs "$EPOCHS" \
    --batch-size "$BATCH" --val-frac "$VAL_FRAC" $POS_FLAG

# --- eval + bundle ----------------------------------------------------------
python3 scripts/eval_pilot.py --model "$RUN/best.pt" \
    --fragments "$DATA/fragments.fna" --labels "$DATA/labels.tsv" \
    --outdir "$RUN/eval" --num-workers 4

STAMP=$(date +%Y%m%d-%H%M%S)
tar -czf "$OUTDIR/omniphyl-pilot-$STAMP.tar.gz" \
    -C "$WORKDIR" "corpus/labels.tsv" "corpus/labels.meta.json" \
    "run/best.pt" "run/eval" \
    $(find run -name "*.log" -o -name "*.json" 2>/dev/null | xargs -r)

echo "==> results: $OUTDIR/omniphyl-pilot-$STAMP.tar.gz"
