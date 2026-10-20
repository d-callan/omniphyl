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
#   NT_BUDGET       nt per accession           (default: 100000)
#   FRAG_LEN        fragment length            (default: 500)
#   BOUNDARY_FRAC   mixed-window fraction      (default: 0.3)
#   ONE_PER_RANK    one assembly per rank val  (default: "" = no cap;
#                   e.g. "phylum" -> max phylogenetic diversity, few genomes)
#   VEUPATH         1 = pass --veupath         (default: 1)
#   K               FCGR k-mer order           (default: 7)
#   WIDTH           ResNet width               (default: 64)
#   EPOCHS          training epochs            (default: 30)
#   BATCH           batch size                 (default: 256)
#   POS_CHANNELS    1 = --pos-channels         (default: 1)
#   VAL_FRAC        held-out genome fraction   (default: 0.1)
#   HOLDOUT_GENUS_FRAC  fraction of train genera moved entirely to val
#                   (leave-clade-out probe)    (default: 0)
#   FETCH_ONLY      1 = fetch corpus, bundle, exit before training
#   SKIP_FETCH      1 = skip fetch; corpus must already exist in CORPUS_DIR
#   CORPUS_DIR      prebuilt corpus location   (default: $WORKDIR/corpus;
#                   on Kaggle point at /kaggle/input/<corpus-dataset>)
#   HEAD            flat | tree (A1 conditional-softmax cosine head)
#   PRIOR_TAU       sibling-prior offset strength for HEAD=tree (default 1)
#   GENBANK_GROUPS  groups that also gap-fill from GenBank (train-only,
#                   seq_source=genbank; default "" = RefSeq only)
#   GENBANK_EXTRA   genbank picks per already-covered rank value (default 0)
#   SCREEN_CONTAM   1 = run contaminant screen on all fragments (default 1)
#   HUMAN_FASTA     local GRCh38 fasta for the human sketch (default "" = skip)
#   SMOKE           1 = run tests first        (default: 1)

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/d-callan/omniphyl.git}"
REF="${REF:-main}"
WORKDIR="${WORKDIR:-$PWD/work}"
OUTDIR="${OUTDIR:-$PWD/output}"
GROUPS="${GROUPS:-bacteria viral}"
LEVELS="${LEVELS:-Complete Genome,Chromosome}"
NT_BUDGET="${NT_BUDGET:-100000}"
GROUP_NT_TOTAL="${GROUP_NT_TOTAL:-0}"  # >0 = equal total nt per group, overrides NT_BUDGET
FRAG_LEN="${FRAG_LEN:-500}"
BOUNDARY_FRAC="${BOUNDARY_FRAC:-0.3}"
ONE_PER_RANK="${ONE_PER_RANK:-}"
PER_RANK_N="${PER_RANK_N:-1}"  # assemblies kept per rank value
GENBANK_GROUPS="${GENBANK_GROUPS:-}"  # e.g. "archaea protozoa fungi"
GENBANK_EXTRA="${GENBANK_EXTRA:-0}"   # genbank picks on covered rank values
VEUPATH="${VEUPATH:-1}"
REQUIRE_GFF="${REQUIRE_GFF:-0}"  # 1 = drop assemblies lacking RefSeq GFF
K="${K:-7}"
WIDTH="${WIDTH:-64}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-256}"
POS_CHANNELS="${POS_CHANNELS:-1}"
VAL_FRAC="${VAL_FRAC:-0.1}"
DROPOUT="${DROPOUT:-0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0}"
WD="${WD:-1e-4}"
BACKBONE="${BACKBONE:-resnet}"      # resnet | vit | seq
STEM_STRIDE="${STEM_STRIDE:-2}"     # 1 = full-res FCGR stem
HOLDOUT_GENUS_FRAC="${HOLDOUT_GENUS_FRAC:-0}"
FRAG_VAL="${FRAG_VAL:-0}"             # within-genome frag holdout fraction (contiguous span)
FETCH_ONLY="${FETCH_ONLY:-0}"       # 1 = corpus capture run, no training
SKIP_FETCH="${SKIP_FETCH:-0}"       # 1 = use prebuilt corpus in CORPUS_DIR
CORPUS_DIR="${CORPUS_DIR:-}"        # empty = $WORKDIR/corpus
HEAD="${HEAD:-flat}"               # flat | tree (A1 head/loss arm)
PRIOR_TAU="${PRIOR_TAU:-1}"
FLAT_PRIOR_TAU="${FLAT_PRIOR_TAU:-0}"  # unconditional prior LA on flat heads
SCREEN_CONTAM="${SCREEN_CONTAM:-1}"   # 1 = screen all fragments for contaminants
HUMAN_FASTA="${HUMAN_FASTA:-}"        # optional GRCh38 fasta for human sketch
SMOKE="${SMOKE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"   # dataloader workers; Kaggle VMs have 4 vCPU
AUX="${AUX:-}"                    # comma list: hash,aa (A3 aux branches)
AA_MODE="${AA_MODE:-plain}"       # plain | bottleneck | gated (E1)
AA_DROP="${AA_DROP:-0}"           # branch dropout rate for gated mode
MIN_TRAIN_LEN="${MIN_TRAIN_LEN:-500}"  # E2: shortest var-len sub-window
VAR_LEN_PROB="${VAR_LEN_PROB:-0}"      # E2: stochastic crop probability
EVAL_CROP_LEN="${EVAL_CROP_LEN:-0}"    # E2 eval: single-end read length
EVAL_PAIRED_LEN="${EVAL_PAIRED_LEN:-0}" # E2 eval: paired-end mate length

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
DATA="${CORPUS_DIR:-$WORKDIR/corpus}"
if [ "$SKIP_FETCH" = 1 ]; then
    for f in fragments.fna labels.tsv; do
        [ -f "$DATA/$f" ] || { echo "SKIP_FETCH=1 but $DATA/$f missing"; exit 1; }
    done
    echo "==> SKIP_FETCH: using corpus at $DATA"
else
    mkdir -p "$DATA"
    VEUPATH_FLAG=""
    [ "$VEUPATH" = 1 ] && VEUPATH_FLAG="--veupath"
    RANK_FLAG=""
    [ -n "$ONE_PER_RANK" ] && RANK_FLAG="--one-per-rank $ONE_PER_RANK --per-rank-n $PER_RANK_N"
    GFF_FLAG=""
    [ "$REQUIRE_GFF" = 1 ] && GFF_FLAG="--require-gff"
    GBUDGET_FLAG=""
    [ "$GROUP_NT_TOTAL" != 0 ] && GBUDGET_FLAG="--group-nt-total $GROUP_NT_TOTAL"
    GENBANK_FLAG=""
    [ -n "$GENBANK_GROUPS" ] && GENBANK_FLAG="--genbank-groups $GENBANK_GROUPS"
    [ -n "$GENBANK_EXTRA" ] && GENBANK_FLAG+=" --genbank-extra $GENBANK_EXTRA"
    # shellcheck disable=SC2086
    python3 scripts/fetch_refseq_corpus.py --outdir "$DATA" \
        --groups $GROUPS --levels "$LEVELS" \
        --nt-budget "$NT_BUDGET" --frag-len-min "$FRAG_LEN" \
        --frag-len-max "$FRAG_LEN" --boundary-frac "$BOUNDARY_FRAC" \
        $VEUPATH_FLAG $RANK_FLAG $GFF_FLAG $GBUDGET_FLAG $GENBANK_FLAG --workers 8
fi

# --- contaminant screen (all fragments) --------------------------------------
FRAG_FILE="$DATA/fragments.fna"
LAB_FILE="$DATA/labels.tsv"
if [ "$SCREEN_CONTAM" = 1 ]; then
    if [ -f "$DATA/labels_screened.tsv" ] && \
       [ -f "$DATA/fragments_screened.fna" ]; then
        echo "==> screened corpus already present in $DATA"
        FRAG_FILE="$DATA/fragments_screened.fna"
        LAB_FILE="$DATA/labels_screened.tsv"
    else
        # mounted corpora (SKIP_FETCH) may be read-only; write screened
        # outputs under WORKDIR in that case
        SOUT="$DATA"
        [ -w "$DATA" ] || SOUT="$WORKDIR/screened"
        mkdir -p "$SOUT"
        HUMAN_FLAG=""
        [ -n "$HUMAN_FASTA" ] && HUMAN_FLAG="--human-fasta $HUMAN_FASTA"
        python3 scripts/screen_contaminants.py \
            --fragments "$DATA/fragments.fna" --labels "$DATA/labels.tsv" \
            --outdir "$SOUT" $HUMAN_FLAG
        FRAG_FILE="$SOUT/fragments_screened.fna"
        LAB_FILE="$SOUT/labels_screened.tsv"
    fi
fi

if [ "$FETCH_ONLY" = 1 ]; then
    STAMP=$(date +%Y%m%d-%H%M%S)
    tar -czf "$OUTDIR/omniphyl-corpus-$STAMP.tar.gz" -C "$DATA" \
        fragments.fna fragments.fna.idx.json labels.tsv skipped.tsv \
        fragments_screened.fna fragments_screened.fna.idx.json \
        labels_screened.tsv fragments_rejected.tsv screen_manifest.json \
        manifest.json 2>/dev/null || \
    tar -czf "$OUTDIR/omniphyl-corpus-$STAMP.tar.gz" -C "$DATA" \
        fragments.fna labels.tsv manifest.json
    echo "==> corpus bundle: $OUTDIR/omniphyl-corpus-$STAMP.tar.gz"
    exit 0
fi

# --- train ------------------------------------------------------------------
RUN="$WORKDIR/run"
POS_FLAG=""
[ "$POS_CHANNELS" = 1 ] && POS_FLAG="--pos-channels"
python3 scripts/train_omniphyl.py \
    --fragments "$FRAG_FILE" --labels "$LAB_FILE" \
    --out "$RUN" --k "$K" --width "$WIDTH" --epochs "$EPOCHS" \
    --batch-size "$BATCH" --val-frac "$VAL_FRAC" \
    --holdout-genus-frac "$HOLDOUT_GENUS_FRAC" --frag-val "$FRAG_VAL" \
    --dropout "$DROPOUT" --label-smoothing "$LABEL_SMOOTHING" --wd "$WD" \
    --backbone "$BACKBONE" --stem-stride "$STEM_STRIDE" \
    --head "$HEAD" --prior-tau "$PRIOR_TAU" \
    --flat-prior-tau "$FLAT_PRIOR_TAU" \
    --num-workers "$NUM_WORKERS" \
    --aux "$AUX" --aa-mode "$AA_MODE" --aa-drop "$AA_DROP" \
    --min-train-len "$MIN_TRAIN_LEN" --var-len-prob "$VAR_LEN_PROB" \
    --seq-len "$FRAG_LEN" \
    $POS_FLAG

# --- eval + bundle ----------------------------------------------------------
python3 scripts/eval_pilot.py --model "$RUN/best.pt" \
    --fragments "$FRAG_FILE" --labels "$LAB_FILE" \
    --outdir "$RUN/eval" --num-workers 4 \
    --eval-crop-len "$EVAL_CROP_LEN" --eval-paired-len "$EVAL_PAIRED_LEN"

STAMP=$(date +%Y%m%d-%H%M%S)
# corpus files exist under corpus/ only on a fresh fetch — with SKIP_FETCH
# the corpus is a mounted dir outside WORKDIR.  screened outputs land under
# corpus/ (fresh) or screened/ (mounted).  Bundle whatever exists.
cd "$WORKDIR"
BUNDLE_FILES="run/best.pt run/eval"
for d in corpus screened; do
    for f in fragments.fna fragments.fna.idx.json labels.tsv skipped.tsv \
             manifest.json fragments_screened.fna \
             fragments_screened.fna.idx.json labels_screened.tsv \
             fragments_rejected.tsv screen_manifest.json; do
        [ -f "$WORKDIR/$d/$f" ] && BUNDLE_FILES="$BUNDLE_FILES $d/$f"
    done
done
EXTRA=$(find run -name "*.log" -o -name "*.json" 2>/dev/null | tr '\n' ' ')
tar -czf "$OUTDIR/omniphyl-pilot-$STAMP.tar.gz" -C "$WORKDIR" \
    $BUNDLE_FILES $EXTRA

echo "==> results: $OUTDIR/omniphyl-pilot-$STAMP.tar.gz"
