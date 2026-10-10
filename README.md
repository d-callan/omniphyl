# OmniPhyl

FCGR-based genome-classification pilot, extended into a single-model
multi-task annotator: taxonomy classification per fragment plus fuzzy
structural-annotation heads (coding fraction, boundary candidacy,
boundary-offset regression).

Design documents live in `docs/`:

- `docs/omniphyl-design.md` — overall architecture, PFCGR positional
  channels, multi-task heads, corpus plan.
- `docs/annotation-cascade.md` — the two-stage annotation design
  (fuzzy candidacy -> exact boundary calls) and the VEuPathDB label tier.
- `docs/pilot-plan.md` — evaluation methodology.
- `docs/use-cases.md` — downstream applications.

## Layout

- `scripts/fetch_refseq_corpus.py` — downloads RefSeq genomes, slices
  fragments, emits `fragments.fna` + `labels.tsv` with composition
  fractions, boundary metadata, and `label_source` provenance
  (`refseq_gff` / `veupathdb`). Optional tiers: `--boundary-frac`
  stratified sampling, `--veupath` VEuPathDB gene models via UCSC
  GenArk contrib tracks (GTF-first, bigBed-API fallback, chromAlias
  bridging).
- `scripts/train_omniphyl.py` — trains the shared encoder with
  classification heads (taxonomy ranks) + continuous heads
  (`coding_frac`, `boundary`, `offset`), masked losses, cosine LR.
- `scripts/eval_pilot.py` — per-rank accuracy/F1, confusion matrices,
  deepest-confident-rank, plus annotation-head metrics (frac MAE,
  boundary recall-vs-flag-rate, offset MAE).
- `scripts/screen_contaminants.py` — contaminant screening for pilot
  fragments.
- `tests/test_smoke.py` — network-free smoke tests + end-to-end
  train/eval on a synthetic 2-genome corpus.

## Quickstart

```bash
# fetch a corpus (protozoa clades, VEuPathDB labels where available)
python3 scripts/fetch_refseq_corpus.py --outdir data/pilot \
    --groups protozoa --veupath --boundary-frac 0.3

# train
python3 scripts/train_omniphyl.py --fragments data/pilot/fragments.fna \
    --labels data/pilot/labels.tsv --out runs/pilot --pos-channels

# evaluate
python3 scripts/eval_pilot.py --model runs/pilot/best.pt \
    --fragments data/pilot/fragments.fna --labels data/pilot/labels.tsv \
    --outdir runs/pilot/eval

# test
python3 tests/test_smoke.py
```

Dependencies: Python >= 3.10, PyTorch, NumPy. Data fetch uses stdlib
`urllib` only (NCBI datasets API + UCSC GenArk).
