# OmniPhyl

Twin-model eukaryotic gene annotator and taxonomic profiler built on
FCGR positional channels:

- **Stage 1 — base model.** One shared encoder, multi-task heads:
  taxonomy classification per 500 bp window (whole-tree, calibrated
  per-rank confidence + OOD probabilities) and structural tracks
  (coding, boundary, frame, domain) binned to contig resolution.
- **Stage 2 — neural refiner.** Per-base decoder conditioned on the
  Stage-1 track stack *and* the genome's taxonomy call, so structure
  decoding is clade-aware (a *Plasmodium* genome and an *E. coli*
  genome decode differently). Trained on 8 genomes across protists,
  fungi, archaea, and bacteria.

## What it's for

`omniphyl annotate` does two jobs in a single pass over the genome:
structural annotation (the GFF3) and contaminant screening
(per-contig calls that disagree with the consensus). No separate
tools, no index builds, no reference panels.

- **Annotate a new genome and know what it is.** `omniphyl annotate`
  emits a GFF3 plus a report with a pooled taxonomy call: deepest
  confident rank, per-rank path confidence, and OOD probability —
  the model says not just *what* it thinks the genome is but *how
  sure* it is at each rank, and flags sequences that fall outside
  its training distribution (e.g., a *P. vivax*-like signal on a
  *P. falciparum* run resolves confidently to the right genus and
  goes OOD below it, instead of forcing a wrong species call).
- **Screen contamination with a confidence, not a hunch.** Per-contig
  taxonomy calls are reconciled against the pooled genome call —
  contigs that disagree flag chimeric/contaminant sequence, reported
  as `n_chimeric_contigs` + per-contig calls in `contig_summary.tsv`.

`omniphyl profile` lets you profile anything with no reference db. It
 classifies against the whole NCBI tree directly — contigs or raw reads. 
 No index build, no curated panel; same model works across clades.

## Inference quickstart

```bash
# gene annotation (base + refiner)
omniphyl annotate --model w64_e15_best.pt --refiner refiner.pt \
    --fasta genome.fa --out pred/genome
# -> pred/genome.{gff3,report.json,report.md,contig_summary.tsv,
#    coding|boundary|frame.bedGraph,coding_regions.bed}

# taxonomy profile of contigs or reads
omniphyl profile --model w64_e15_best.pt reads_R1.fastq \
    --reads2 reads_R2.fastq --out prof/sample
```

`run_annotate()` / `run_profile()` in `src/omniphyl/` expose the same
pipeline programmatically. A `res.pkl` Stage-1 cache can be dumped
(`OMNIPHYL_DUMP_RES=1`) to re-decode later without re-running the model.

## Training

- `scripts/fetch_refseq_corpus.py` — downloads RefSeq genomes, slices
  fragments, emits `fragments.fna` + `labels.tsv` with composition
  fractions, boundary metadata, and `label_source` provenance
  (`refseq_gff` / `veupathdb`).
- `scripts/train_omniphyl.py` — trains the Stage-1 shared encoder.
- `scripts/train_refiner.py` — trains the Stage-2 refiner from cached
  Stage-1 tracks + reference GFFs (`--data fasta gff` per genome;
  per-genome Stage-1 runs as an isolated subprocess and caches to
  `refiner_cache/` so retries skip finished genomes).
- `scripts/eval_pilot.py`, `scripts/eval_genes.py` — taxonomy and
  gene-structure metrics.
- `remote/run_pilot.sh`, `kaggle/` — end-to-end runners on free
  remote hardware.

## Future directions

Planned inference modes on the same twin-model stack — see
`docs/use-cases.md` for the full catalog and feasibility notes.

- **`omniphyl screen` — pathogen screening.** A dedicated script
  answering "how confident is the model that this sample contains
  evidence of a human pathogen, and which one(s)?" — the pooled
  per-rank confidence + OOD machinery is already built for exactly
  this; what remains is the pathogen panel definition, a calibrated
  aggregation over contigs, and thresholds.
- **Functional annotation.** The domain head already reports
  top-domain counts in the report; the next step is per-gene
  functional calls (Pfam domain assignments, COG-category labels)
  written onto the GFF — annotate's structural output is the
  scaffold these attach to.
- **Contig binning.** Cluster the shared-latent embeddings into
  metagenome bins (k-mer composition is the classic binning signal).
- **Mobile-element typing.** A prophage/plasmid/chromosomal head —
  doubles as a taxonomy-quality fix so prophage contigs stop
  inheriting carrier taxonomy.
- **Strain / sub-species typing.** Below-species calls for lineage
  tracking (FCGR precedent is strong on viral clades; cultivar-level
  is the hard regime).

## Docs

- `docs/omniphyl-design.md` — architecture, PFCGR channels, heads.
- `docs/neural-refiner-plan.md` — Stage-2 refiner design.
- `docs/annotation-cascade.md` — fuzzy candidacy -> exact calls.
- `docs/use-cases.md` — downstream applications catalog.
- `docs/remote-runs.md` — Kaggle/Colab/laptop runs.

Dependencies: Python >= 3.10, PyTorch, NumPy. Data fetch uses stdlib
`urllib` only (NCBI datasets API + UCSC GenArk).
