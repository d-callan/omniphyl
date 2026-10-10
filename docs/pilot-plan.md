# OmniPhyl — Pilot Plan

*A focused proof-of-concept scoped to what can be verified quickly and cleanly. Full design: [`omniphyl-design.md`](omniphyl-design.md). Last updated: 2026-10-02.*

---

## Scope

| Dimension | Pilot |
|---|---|
| Reference corpus | NCBI RefSeq; assembly count is secondary — see **corpus design** below |
| Fragment size | **500 bp** fixed (short-read / metagenome-realistic) |
| Contaminant filter | **Sequence-based** confidence screen (see below) — not metadata |
| Training targets | Multi-rank taxonomy (domain → species) + coding vs. non-coding |
| Architecture | ResNet backbone (default); ViT optional for ablation |
| Scope ceiling | No host-range, no AMR, no AA/polyflake stem, no pooled-sample profiles |

Rationale: 500 bp is the hardest case (least signal per fragment) and the regime most relevant to real short-read metagenomics. If the FCGR signal is useful at 500 bp, longer contigs only get easier.

---

## Corpus design

Three problems with a naïve "N assemblies per group" approach, and the fixes:

### 1. Genome-size bias
A 6 Gbp plant genome with `--frags-per-genome 200` contributes the same fragment count as a 2 Mbp bacterium, but dominates any sequence-space representation. The model risks learning genome-size-correlated context rather than clade-discriminative signal.

**Fix**: sample by **nucleotide budget**, not fragment count. Divide a per-assembly budget (e.g. 100 kb total sequence, = 200 × 500 bp) uniformly across the genome, so every assembly contributes the same total nucleotides to the training set regardless of genome size. Implement as sampling without replacement from all valid 500 bp windows in proportion to contig length, capped at the budget.

### 2. Clade diversity imbalance within groups
RefSeq groups are not phylogenetically equivalent: "bacteria" spans more genomic diversity (GC 13–75%, radically different k-mer landscapes across Firmicutes/Proteobacteria/Actinobacteria/CPR/DPANN-adjacent) than "vertebrate" does. A flat per-group count severely undersamples bacterial diversity.

**Fix**: stratify within groups **by phylum** (or equivalent rank), aiming for ≥1 assembly per phylum where RefSeq has representation, rather than a fixed per-group count. For bacteria this means ~30–50 phyla represented; for vertebrates, a handful of classes suffices. The total assembly count is a consequence of this strategy, not a target.

Practical for the pilot: use `fetch_refseq_corpus.py`'s existing taxonomy labels to enforce one-per-phylum sampling within each group. Add a `--one-per-rank phylum` flag.

### 3. Uncultured and culture-biased gaps
RefSeq is heavily culture-biased. CPR bacteria, DPANN archaea, most protists, and the vast majority of eukaryotic microbial diversity are either absent or represented by single, often poor-quality assemblies. This is a known limitation of any RefSeq-trained model — OmniPhyl inherits it.

**Pilot position**: this is *not a pilot problem to solve*. The pilot's goal is to show the FCGR signal works on well-represented clades. What to do:
- Flag the gap explicitly in evaluation: report which phyla have zero or single-assembly representation in the training set.
- The unknown-mass channel is the long-term answer for unrepresented lineages — the pilot should exercise it by holding out entire phyla during training (leave-phylum-out test split) rather than random assembly splits.
- In the full corpus (Phase 1), supplement RefSeq with GTDB (broader prokaryote coverage), IMGVR (viral), and MAG collections for uncultured branches.

---

## Scripts

### Implemented

| Script | Role | Pilot additions |
|---|---|---|
| `scripts/fetch_refseq_corpus.py` | Download RefSeq assemblies, slice fragments, emit labels | `--one-per-rank` (phylum-stratified sampling); `--nt-budget` (size-normalised, replaces `--frags-per-genome`); GFF HEAD-check pre-filter; downloads `_genomic.gff.gz` alongside FASTA; `coding` column via GFF3 CDS interval intersection; `skipped.tsv` audit log |
| `scripts/train_omniphyl.py` | FCGR encoding on-the-fly, ResNet/ViT backbone, multi-rank + coding heads | `coding` target head added alongside taxonomy ranks; `NA` fragments excluded from coding loss via `ignore_index=0`; checkpoint saves `targets` list |

### To write

*All scripts implemented.* See pipeline section for usage.

---

## Pipeline

```
fetch_refseq_corpus.py          # step 1: download + slice
    --groups archaea bacteria fungi protozoa invertebrate plant
             vertebrate_mammalian vertebrate_other viral
    --one-per-rank phylum       # ≥1 assembly per phylum per group
    --frag-len-min 500          # fixed 500 bp
    --frag-len-max 500
    --nt-budget 100000          # 100 kb per assembly = 200 × 500 bp, size-normalized
    --outdir data/pilot

screen_contaminants.py          # step 2: drop suspect fragments
    --fragments data/pilot/fragments.fna
    --labels    data/pilot/labels.tsv
    --outdir    data/pilot

train_omniphyl.py               # step 3: train
    --fragments data/pilot/fragments_screened.fna
    --labels    data/pilot/labels_screened.tsv
    --targets   domain phylum class order family genus species coding
    --backbone  resnet
    --k         7
    --pos-channels
    --epochs    30
    --outdir    data/pilot/model

eval_pilot.py                   # step 4: evaluate
    --model   data/pilot/model/best.pt
    --labels  data/pilot/labels_screened.tsv
    --outdir  data/pilot/eval
```

---

## Sequence-based contaminant screen (`screen_contaminants.py`)

The goal: drop fragments that are likely cross-species contamination *without relying on assembly metadata*, since metadata may itself be wrong. Layers, cheapest first:

1. **Ambiguity gate** — already in `slice_fragments`; fragments >10% non-ACGT are dropped. Tighten to 5% for the pilot.

2. **k-mer self-consistency within assembly**: compute a per-assembly reference k-mer profile (union of 21-mers from all sequences in the assembly). For each 500 bp fragment, compute Jaccard containment of its 21-mers in that profile. Flag fragments below a containment threshold (e.g. 0.5) — they share little sequence with the rest of the assembly and may be inserted contaminant sequence.
   - Implementation: `sketch_assembly()` → sorted list of hashed 21-mers; per-fragment containment as `|frag_kmers ∩ asm_kmers| / |frag_kmers|`.
   - Cheap in memory if we store the assembly sketch as a frozenset (not the full sequence).
   - Threshold is a hyperparameter — log the distribution, pick a threshold that drops the fat low-containment tail; document the cut.

3. **Known-contaminant screen**: sketch a small set of universal contaminants — human chr1–22+X+Y 21-mer sketch, phiX174, UniVec core sequences — and flag any fragment with containment > 0.1 in any contaminant sketch. These are the canonical RefSeq false-positive sources.
   - Human genome sketch can be built from the GRCh38 assembly in the same pipeline; or use a precomputed sketch file.
   - This step is the sequence-level Alu/repeat filter mentioned in design decision 8.

4. **Emit two outputs**: `fragments_screened.fna` + `labels_screened.tsv` (passing fragments); `fragments_rejected.tsv` (reason, containment scores) for audit.

Note: we are deliberately **not** using Kraken/Mash as a dependency in the pilot — the above is pure Python + stdlib so it runs anywhere. A later production version can swap in a proper k-mer index.

---

## Coding / non-coding labels

RefSeq assemblies include GFF3 annotation files alongside genomic FASTA. For each fragment, intersect its `[start, end)` interval against CDS features on its source sequence:

- **coding**: ≥50% of the fragment overlaps at least one CDS
- **non-coding**: <50% overlap

Labels are added as a `coding` column in `labels.tsv` during the `fetch_refseq_corpus.py` step. Both `_genomic.fna.gz` and `_genomic.gff.gz` are downloaded together.

**Pilot policy**: only assemblies that have a GFF are eligible. Filter at selection time — before downloading — by checking whether `<ftp_dir>/<name>_genomic.gff.gz` exists (HTTP HEAD request, cheap). Assemblies that fail the HEAD check are dropped from the candidate pool and logged. This avoids the `coding=NA` case entirely for the pilot and keeps the dataset clean. For viral RefSeq entries that lack GFF (common for small genome submissions), this means some will be dropped; that's acceptable — those without annotation aren't useful for the coding/non-coding head, and the ones that remain (well-annotated virus reference genomes) are still representative.

---

## Evaluation targets

| Target | Metric | Baseline to beat |
|---|---|---|
| Taxonomy per rank | Top-1 accuracy, macro-F1 | Random (uniform over classes) |
| Taxonomy per rank | Deepest-confident-rank accuracy (unknown-mass channel) | — |
| Coding vs. non-coding | Binary F1 | Prodigal on 500 bp fragments (known to degrade at short lengths — this is the regime where a trained FCGR signal may win) |
| Cross-group transfer | Per-group accuracy breakdown | Uniform |

Leave-out design: **leave-phylum-out** — hold out all assemblies from one phylum per group for test; train on the rest. This is the correct pilot split because it tests the unknown-mass channel (can a held-out phylum land in the unknown bucket or at least at a higher rank?) rather than just intra-species generalization. Report both within-training-phyla accuracy and cross-phylum generalization depth separately.

Also report: which phyla have zero representation in RefSeq for the pilot groups — these are the cultural gaps the full corpus will need to address.

---

## Open questions the pilot answers

1. Is the FCGR signal sufficient at 500 bp for any rank above species? (The key feasibility question.)
2. Does the coding/non-coding head converge without the domain/AA stem — i.e., is nt-FCGR texture enough at this scale?
3. What does the containment distribution look like across groups — is the k-mer self-consistency screen well-calibrated or too aggressive?
4. Does `--pos-channels` help at 500 bp (where positional context is most compressed)?
5. Does ViT beat ResNet at this scale and fragment length? (Optional `--backbone vit` ablation run.)
6. How many phyla per group does RefSeq actually cover with complete/chromosome-level assemblies — what fraction of the known tree is represented at pilot scale?
