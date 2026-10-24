# OmniPhyl 500 bp Architecture & Evaluation Recommendations

*Synthesis of pilot runs (`v2`, `v3`, `v4`, `fungi-probe`), design documents (`omniphyl-design.md`, `pilot-plan.md`, `annotation-cascade.md`), and literature on short-fragment taxonomic classification.*

---

## 1. Diagnosis of Pilot Runs: Why 500 bp Classification Stalled

Across four training iterations, separating training-genome fragments from held-out validation-genome fragments exposed three coupled bottlenecks:

| Run | Corpus / Split Regime | Train Acc (Dom / Phy / Fam / Sp) | Val Acc (Dom / Phy / Fam / Sp) | Val Macro-F1 (Phy / Fam / Sp) | Primary Diagnostic Finding |
|---|---|---|---|---|---|
| **v2** | 4 groups, 1 asm/phylum (`PER_RANK_N=1`) | 0.96 / 1.00 / 1.00 / 1.00 | *(0 shared phyla in val)* | — | **Memorization illusion**: without train/val split separation, per-genome overfitting looked like 100% accuracy. |
| **v3** | 4 groups, `PER_RANK_N=3`/phylum, 300k nt/asm, 30 ep | 0.93 / 0.75 / 0.75 / 0.66 | 0.63 / 0.31 / 0.45 / 0.65 | 0.070 / 0.038 / 0.013 | **Severe overfitting after epoch 6** + high val accuracy on head classes masking near-zero macro-F1 on tail classes. |
| **v4** | 5 groups (+archaea), `PER_RANK_N=2`/family (3,578 asms), 75k nt/asm, 10 ep + reg | 0.88 / 0.75 / 0.49 / 0.40 | 0.73 / 0.58 / 0.11 / 0.21 | 0.102 / 0.018 / 0.002 | **Prior collapse & depth starvation**: 41.7% of all phylum predictions collapse into *Uroviricota*; 8 rare phyla have 0.0% recall. |
| **Fungi** | Fungi only, `PER_RANK_N=2`/species (461 asms, 456 sp), 300k nt/asm, 10 ep | 1.00 / 0.86 / 0.33 / 0.11 | 1.00 / 0.89 / 0.31 / 0.004 | 0.318 / 0.047 / 0.000 | **Representation ceiling**: `train ≈ val` from phylum (0.86 vs 0.89) to genus (0.27 vs 0.25), but train species is 0.105. |

### Five Mechanistic Root Causes

1. **Single-scale $k=7$ FCGR sparsity at 500 bp**:
   A 500 bp fragment contains $500 - 7 + 1 = 494$ overlapping 7-mers distributed across $4^7 = 16,384$ grid cells. At least $97\%$ of cells are zero, and ~95% of non-zero cells contain a count of 1. Meanwhile, the stable compositional signature of a 500 bp window lives at $k \in \{3, 4, 5\}$ (64 to 1,024 cells, where expected counts per cell are $0.5\text{--}8$), which the ResNet only sees after four stages of lossy spatial striding.
2. **Quadtree resolution destruction in the stem**:
   In standard CGR addressing (`build_cell_lut`), each axis encodes half of the $k$-mer's base-4 digits, with the terminal nucleotide occupying the least-significant bit ($1\times 1$ pixel). `ResNetEncoder`'s initial `Conv2d(..., stride=2)` immediately pools $2\times 2$ pixel blocks, erasing the terminal base on both axes before Stage 1 begins.
3. **Lack of reverse-complement (RC) canonicalization**:
   A 500 bp shotgun fragment arrives from either strand with equal probability. Whereas Chargaff's second parity rule approximately holds over whole chromosomes, a single 500 bp window (often a single coding exon on one strand) has strong strand asymmetry. Currently, $s$ and $\text{RC}(s)$ map to transposed/inverted FCGR patterns that the CNN must learn separately, cutting effective sample efficiency in half.
4. **6-frame coding / amino-acid blindness**:
   Across whole-tree taxonomy (especially family/genus in eukaryotes and viruses), synonymous third-codon-position drift scrambles nucleotide 7-mers while amino-acid 3-mers/4-mers remain conserved. Conversely, coding codon-usage bias is a strong clade separator that is blurred when all three reading frames are summed into a frame-agnostic $k$-mer histogram.
5. **Independent flat heads + uncalibrated softmax**:
   `OmniPhyl` currently trains seven independent `nn.Linear` heads (`domain` through `species`) with unadjusted cross-entropy, and `class 0` (`<unknown>`) is masked via `ignore_index=0`. Consequently:
   - The model never learns to emit `<unknown>` (logits for class 0 receive no positive gradient).
   - Heads make mutually inconsistent calls (e.g., predicting a fungal species while predicting a bacterial phylum).
   - Under uncertainty at deep ranks, linear classifiers with standard cross-entropy provably default to the largest prior class (*Uroviricota* in `v4`, *Ascomycota* in `fungi-probe`).

---

## 2. Why Raw 1D Sequence Is the Wrong Primary Input at 500 bp

Your intuition that raw 1D nucleotide sequence is too noisy is supported by both information theory and published benchmarks (Liang et al., *DeepMicrobes*, *NAR Genom Bioinform* 2020; Mock et al., *BERTax*, *PNAS* 2022; Karamichalis et al., *BMC Bioinform* 2015):

- **Homology famine per window**: Any random 500 bp slice from a 10 Mbp fungal or protozoan genome has $<10^{-4}$ probability of overlapping the same locus in another training genome unless tens of thousands of fragments are sampled per species. A 1D sequence CNN or Transformer looking for positional motifs across unaligned random genomic windows fits locus-specific noise rather than genome-wide evolutionary signatures.
- **Empirical precedent**: *DeepMicrobes* explicitly benchmarked 1D ResNet-like CNNs and hybrid CNN-BiLSTMs on one-hot DNA for short metagenomic reads and found both accuracy and confidence collapsed compared to $k$-mer spectrum/embedding representations.
- **Order-invariance is the right inductive bias**: What is invariant across *arbitrary* 500 bp windows of the same organism is the stationary Markov transition structure (oligonucleotide usage, dinucleotide relative abundance, codon bias, and amino-acid composition)—not the 1D spatial coordinate of a motif within the 500 bp slice.

---

## 3. Proposed Architecture: Multi-Scale Canonical Spectrum + Hierarchical Prototype Network (`OmniPhyl-MSHP`)

The goal is an architecture that **maximizes per-fragment taxonomic signal extraction without raw-sequence noise**, **cannot suffer prior collapse**, **enforces tree-consistent lineages**, and **produces calibrated confidence scores**—while training in the same ~2–4 hour T4 envelope as the current ResNet.

```
500 bp DNA fragment s (and its reverse complement s_rc)
 │
 ├─► Stream 1: Multi-Resolution Canonical FCGR Pyramid (dense compositional + codon signature)
 │     • Canonical count: C_k(s) = FCGR_k(s) + FCGR_k(s_rc)
 │     • Multi-scale quadtree pyramid: k=4 (16×16), k=5 (32×32), k=6 (64×64)
 │       + frame-aware in-frame 3-mer/6-mer coding periodicity summary
 │     • Stride-1 Quadtree-Aligned ConvNeXt/ResNet Stem (preserves 2×2 CGR sibling structure)
 │
 ├─► Stream 2: Sparse High-k Canonical Hash Sketch (fine-grained clade/genus markers)
 │     • Canonical 11-mers (490 per fragment) → Multi-Hash EmbeddingBag (4×2^16 buckets, dim 128)
 │     • Captures specific lineage marker words without 4^11 dense memory explosion
 │
 └─► Stream 3: 6-Frame Translated AA 3-mer Spectrum (divergent family/order bridge)
       • 6-frame translation → 20^3 = 8,000-dim log-normalized tripeptide vector → 2-layer MLP
       • Invariant to 3rd-codon synonymous saturation across distant genera/families
 │
 ▼
 Gated Multi-Modal Fusion + L2-Normalized Latent Projection (z ∈ S^{D-1}, D=256)
 │
 ▼
 Taxonomic Tree Conditional Cosine-Prototype Head + Dirichlet Energy Abstention
   • Unit-norm class prototypes w_{r,c} ∈ S^{D-1} with learnable temperature s_r
   • Conditional brotherhood softmax: P(c | parent(c), x) + Prior Logit Adjustment (τ·log π)
   • Chained lineage probability: P(node | x) = ∏_{a ∈ ancestors(node)} P(a | parent(a), x)
   • Stopping rule: descend tree while marginal P(node | x) ≥ τ_r and sibling entropy H ≤ H_max
```

### Component Breakdown & Justification

#### A. Input Representation: Three Complementary Order-Invariant Views (Zero Raw-Sequence Noise)
1. **RC-Canonical Multi-Scale FCGR Pyramid ($k \in \{4, 5, 6\}$ instead of sparse $k=7$)**:
   - **RC Canonicalization**: Compute $k$-mer counts on $s \oplus \text{RC}(s)$ (or equivalently average $\text{FCGR}_k(s)$ with its RC-permuted matrix in $O(4^k)$ via a precomputed index map). This makes the representation strictly strand-invariant at zero runtime cost.
   - **Why $k=6$ ($64\times 64$) + pooling pyramid beats $k=7$ ($128\times 128$)**: At $k=6$ ($4,096$ cells), a 500 bp fragment + RC (990 $k$-mers) has $4\times$ higher cell occupancy than single-strand $k=7$, and its $2\times 2$ average pools yield exact $k=5$ ($32\times 32$), $k=4$ ($16\times 16$), and $k=3$ ($8\times 8$) FCGRs. Feeding $k=6$ at full $64\times 64$ resolution with a **stride-1 stem** and concatenating explicit $k=4, 5$ pooled maps at intermediate stages gives the network both dense Markov statistics ($k=3\text{--}5$, where domain/phylum/class live) and $k=6$ oligomer texture, at **one-quarter the spatial FLOPs** of stride-1 $128\times 128$.
   - **Standardization over max-normalization**: Replace `img /= img.max()` (which makes every 500 bp fragment's scaling sensitive to a single repeat $k$-mer) with standardized log-frequency `(log1p(freq) - mean) / std` or total-count normalization `freq / N_kmers` (Wang et al., *Gene* 2005; Löchel & Heider 2021).
2. **Sparse High-$k$ Hash-Embedding Bag ($k=11$ or $12$)**:
   - Why FCGR caps out at family/genus: $k \le 7$ cannot represent discriminative 10–12 bp lineage markers (DeepMicrobes showed a massive jump from $k=8$ to $k=12$).
   - Instead of a $4^{12} = 16.7\text{M}$ vocabulary table (which blows up GPU memory and overfits), hash canonical 11-mers into $B = 65,536$ buckets using 2–4 independent murmur/tabulation hashes and pass through `nn.EmbeddingBag(65536, 128, mode="sum")` followed by LayerNorm (MetaTransformer Hash-EmbedPool, *NAR Genom Bioinform* 2023). For 500 bp, this is a lookup of 490 integers per fragment—microseconds on GPU, ~32 MB parameters, and directly supplies family/genus/species marker features.
3. **Lightweight 6-Frame AA 3-mer / Reduced-Alphabet 4-mer Vector**:
   - Translate the 500 bp window in all 6 frames ($\sim 6 \times 166 = 996$ amino acids) or use the top-scoring coding frame when `coding_frac > 0.5`, count 3-mers ($20^3 = 8,000$ bins) or Murphy-10 reduced-alphabet 4-mers ($10^4 = 10,000$ bins), apply $\log(1+c)$ and a single linear projection `nn.Linear(8000, 128)`. This captures conserved protein-domain composition across genera/families where nucleotide $k$-mers have saturated.

#### B. Backbone: Quadtree-Aligned ConvNeXt-Small / ResNet with $2\times 2$ Downsampling
- In FCGR, every $2\times 2$ block aligned on even coordinates corresponds to the 4 extensions of a $(k-1)$-mer. Standard $3\times 3$ stride-2 convolutions straddle quadtree boundaries and mix unrelated $k$-mer prefixes.
- **Fix**: Use a `3×3 stride-1` stem on the $64\times 64$ ($k=6$) input, and perform all spatial downsampling between stages via **`2×2 stride-2` convolutions** (exact quadtree parent aggregation: $64 \to 32 \to 16 \to 8 \to 4$).

#### C. Output Heads & Loss: Fixing Prior Collapse, Tree Incoherence, and Abstention
1. **Cosine-Prototype Classifiers (Eliminates Majority-Class Norm Bias)**:
   - In standard `nn.Linear`, dominant classes (*Uroviricota*, *Ascomycota*) accumulate larger weight norms $\|w_c\|$ and positive bias $b_c$, pulling all uncertain embeddings into their decision region (Sharma et al. 2023; Menon et al. *ICLR* 2021).
   - Replace `nn.Linear(D, C)` with L2-normalized features and unit-norm prototypes:
     $$\text{logit}_{r,c}(z) = s_r \cdot \frac{z^\top w_{r,c}}{\|z\|_2 \|w_{r,c}\|_2}$$
     where $s_r$ is a learnable scalar scale (initialized to $16.0$). Because $\|w_{r,c}\|_2 = 1$ for every taxon, majority classes cannot dominate by norm inflation.
2. **Logit-Adjusted Conditional Tree Softmax (Enforces Lineage Coherence & Balanced Recall)**:
   - Instead of 7 independent flat softmaxes, structure the taxonomy as its actual tree $\mathcal{T}$ (using the training `vocab` parent–child map; SALT 2025 / Conditional Softmax, Redmon & Farhadi 2017 / CoNLL 2024).
   - Normalize logits **within sibling groups** $\mathcal{C}(\pi(c))$ (all classes sharing parent $\pi(c)$), with **Logit Adjustment** for sibling prior frequencies $\nu(c \mid \pi(c))$:
     $$P_{\text{train}}(c \mid \pi(c), x) = \frac{\exp(\text{logit}_c(x) + \tau \log \nu(c \mid \pi(c)))}{\sum_{c' \in \mathcal{C}(\pi(c))} \exp(\text{logit}_{c'}(x) + \tau \log \nu(c' \mid \pi(c)))}$$
   - At inference time, remove the prior offset ($\tau=0$) to obtain the **Bayes-optimal balanced conditional probability** $P(c \mid \pi(c), x)$, and chain probabilities top-down from root to leaf:
     $$P_{\text{lineage}}(c_r \mid x) = \prod_{a \in \text{Ancestors}(c_r)} P(a \mid \pi(a), x)$$
   - **Why this transforms behavior**:
     - Predictions are **guaranteed to be monotonically decreasing and tree-coherent** ($P(\text{domain}) \ge P(\text{phylum}) \ge \dots \ge P(\text{species})$).
     - A fungal fragment can never be assigned to a viral phylum or a bacterial family.
     - Rare phyla (e.g., *Rhodophyta*, *Parabasalia*) only compete against sibling phyla within *Eukaryota* on balanced terms, rather than competing against *Uroviricota* across the entire dataset.
3. **Principled Confidence & Open-Set Abstention (When to Stop Descending the Tree)**:
   - Because $P_{\text{lineage}}(c_r \mid x)$ is a chained hierarchical probability, abstention ("deepest confident rank") is no longer an arbitrary threshold on decoupled flat heads.
   - **Two-part stopping criterion at rank $r$**:
     1. **Marginal lineage probability** $P_{\text{lineage}}(\hat{c}_r \mid x) \ge \tau_r$, where $\tau_r$ is calibrated on held-out validation clades to achieve a target precision (e.g., $80\%$ precision at rank $r$).
     2. **Normalized sibling entropy** $\tilde{H}(\mathcal{C}(\hat{c}_{r-1}) \mid x) = -\sum_{c \in \mathcal{C}} P(c \mid \hat{c}_{r-1}) \log P(c \mid \hat{c}_{r-1}) / \log |\mathcal{C}| \le H_{\max}$. When a fragment belongs to an **unseen child clade** of a known parent $\hat{c}_{r-1}$ (leave-clade-out), cosine similarities to all known sibling prototypes are uniformly low, producing high sibling entropy $\tilde{H} \to 1$; the classifier cleanly stops at $\hat{c}_{r-1}$ ("confident family, unknown genus").
4. **Same-Genome Supervised Contrastive Auxiliary Loss (`SupCon`)**:
   - In every batch, pair each fragment with 1 positive fragment from the **same assembly/species** (`BatchSampler` drawing 2 fragments per genome across 128 genomes = batch size 256).
   - Add a lightweight hierarchical SupCon / prototype-pull loss on the L2-normalized latent $z$. Even when a taxon is a singleton in its family, pulling fragments of the same genome together in $z$-space teaches the encoder which spectrum features are **genome-invariant** vs. **locus-specific noise** (CGRclust, *BMC Genomics* 2024; ViTax, *Brief Bioinform* 2025).

---

## 4. What Honest Ceiling to Expect at 500 bp vs. Multi-Fragment Aggregation

We should separate two distinct deployment questions:

1. **Single isolated 500 bp fragment**:
   - **Domain / Phylum / Class**: High confidence and high accuracy ($75\text{--}95\%$ with multi-scale FCGR + logit adjustment) because global GC, dinucleotide, and codon biases stabilize within 300–500 bp.
   - **Order / Family / Genus**: Moderate accuracy ($40\text{--}70\%$ when confident, with explicit abstention to Class/Order when a fragment falls in a conserved or non-coding repeat region). Adding the canonical 11-mer hash sketch + 6-frame AA 3-mer branch is the primary lever to lift this tier.
   - **Species**: Physically bounded on a single random 500 bp window across thousands of species unless the window hits a species-specific 11-mer marker. The model *should* abstain at Genus/Family on most isolated 500 bp fragments rather than hallucinate a species.
2. **Multi-fragment / Contig / Read-Set Aggregation (Zero Retraining)**:
   - Because `OmniPhyl-MSHP` outputs calibrated log-probabilities $\log P(c \mid \pi(c), x_i)$ per 500 bp window, any sequence of length $L > 500\text{ bp}$ (or a bin of $M$ reads from the same organism) is classified by **Bayesian log-odds pooling** over $M$ windows:
     $$\log P(c \mid \pi(c), \{x_1, \dots, x_M\}) \propto \sum_{i=1}^M w_i \log P(c \mid \pi(c), x_i)$$
     where $w_i = 1 - \tilde{H}_i$ downweights uninformative/repetitive windows.
   - We can test this **immediately offline on the existing checkpoints** (`eval_pilot.py` genome-consensus metric): even when single-fragment accuracy is $25\%$, pooling $M=20$ fragments ($10\text{ kb}$) drives accuracy above $80\text{--}90\%$ as long as errors are unbiased (which Cosine Prototypes + Logit Adjustment guarantee, unlike prior-collapsed linear heads).

---

## 5. Fast Iteration & Ablation Plan (CPU / Single-GPU Friendly)

To iterate in **15–30 minutes per experiment** instead of 6–10 hour blind Kaggle runs, lock a **Fixed Micro-Benchmark (`fungi-mini` + `5group-mini`)** that caches to disk once (~150 MB) and runs in <20 minutes on a single T4 (or ~45 min locally on CPU/MPS):

### Benchmark Protocol
- **`5group-mini` (Breadth + Imbalance + OOD)**: ~500 genomes across all 5 groups (`GROUP_NT_TOTAL` equalized, `PER_RANK_N=2` per family, 20k nt/genome = 40 fragments/genome → 20k total fragments, plus 10% held-out genera for open-set abstention eval).
- **Primary Metrics** (reported on **held-out val genomes** only):
  1. **Val Macro-F1** at Phylum, Class, Order, Family, Genus (measures balanced performance across all clades, immune to majority-class gaming).
  2. **Area Under the Risk-Coverage Curve (AURC) / Precision@Coverage** across hierarchical cuts (measures whether the confidence score actually knows when it is right).
  3. **Leave-Genus-Out Parent Accuracy**: on fragments from held-out genera, how often does the model stop at the correct Family/Order vs. hallucinating a wrong known genus?
  4. **$M$-Fragment Consensus Accuracy ($M \in \{1, 5, 20\}$)**: measures how quickly per-fragment evidence compounds across a contig or genome.

### Staged Ablation Ladder (Ordered by Expected Impact / Implementation Cost)

| Step | Change | Lines of Code | Hypothesis Tested | Success Gate on Val Split |
|---|---|---|---|---|
| **0 (Eval)** | Add hierarchical calibration, AURC, per-rank macro-F1, and $M$-fragment consensus pooling to `eval_pilot.py` | ~80 LOC | Quantifies exact current calibration + whether pooling $M=5, 20$ fragments rescues family/genus on existing `v4`/`fungi` checkpoints | Baseline established offline in <2 min |
| **1 (Head/Loss)** | **Cosine Prototypes + Logit-Adjusted Conditional Tree Softmax** (replace flat `nn.Linear` + unadjusted CE) | ~70 LOC | Eliminates *Uroviricota*/*Ascomycota* prior collapse, guarantees tree-coherent predictions, and yields calibrated hierarchical abstention | Val Phylum/Family Macro-F1 $\ge 2\times$ baseline; 0-recall phyla eliminated |
| **2 (FCGR Input)** | **RC-Canonical Multi-Scale FCGR ($k=6$ full-res + $k=4,5$ pyramid, standardized log-counts, $2\times 2$ quadtree downsampling)** | ~60 LOC | Strand invariance + denser $k=4\text{--}6$ Markov statistics + preserved quadtree boundaries lift single-fragment representation ceiling | Val Class/Order/Family accuracy $+10\text{--}15$ pts at $2\times$ faster epoch time |
| **3 (High-$k$ + AA)** | **Parallel Canonical 11-mer `EmbeddingBag` Hash Sketch + 6-Frame AA 3-mer branch** fused into latent $z$ | ~90 LOC | Supplies fine-grained lineage markers (11-mers) and synonymous-drift-invariant protein composition (AA 3-mers) for Family/Genus/Species | Val Family/Genus accuracy + Leave-Genus-Out family recall lift significantly |
| **4 (Contrastive)** | **2-per-genome Paired Batch Sampler + Hierarchical SupCon Loss** on latent $z$ | ~50 LOC | Forces latent $z$ to discard locus-specific noise and retain genome-wide invariant signature | Improved $M$-fragment consensus and open-set clustering |

---

## 6. Key Literature Anchors

- **CGR / FCGR Representation & Distances**: Jeffrey (*NAR* 1990); Deschavanne et al. (*Mol Biol Evol* 1999); Wang et al. (*Gene* 2005, RC-concatenated standardized FCGR); Karamichalis et al. (*BMC Bioinform* 2015, DSSIM/descriptor distances vs. Euclidean across 6 kingdoms); Löchel & Heider (*CSBJ* 2021 review).
- **Short-Read Deep Taxonomic Classifiers**: Rizzo et al. (2016, CNN on FCGR 500 bp 16S); Liang et al. (*DeepMicrobes*, *NAR Genom Bioinform* 2020 — showed 1D one-hot CNNs underperform $k=12$ multi-mer representations on short shotgun reads); Wichmann et al. (*MetaTransformer*, *NAR Genom Bioinform* 2023 — `Hash-EmbedPool` for compact high-$k$ representations); Mock et al. (*BERTax*, *PNAS* 2022 — `all-in-one` hierarchical rank conditioning).
- **Contrastive & Foundation-Model Taxonomy**: Millán Arias et al. (*CGRclust*, *BMC Genomics* 2024 — twin contrastive learning on FCGR across 664 bp–100 kb); Camargo et al. (*geNomad*, *Nat Biotech* 2023 — alignment-free sequence + marker fusion + length-aware score calibration); *ViTax* (*Brief Bioinform* 2025) & *ICCTax* (*Bioinform Adv* 2025) — supervised prototypical contrastive learning + tree belief mapping / LCA confidence stopping for open-set taxonomy.
- **Hierarchical Softmax, Long-Tail & Calibration**: Redmon & Farhadi (*YOLO9000* conditional tree softmax, 2017); Menon et al. (*Long-Tail Learning via Logit Adjustment*, *ICLR* 2021); Plaud et al. (*Logit-Adjusted Conditional Softmax for Hierarchical Classification*, *CoNLL* 2024); Wu et al. (*Deep-RTC: Solving Long-Tailed Recognition with Deep Realistic Taxonomic Classifier*, *ECCV* 2020 — calibrated tree-cut rejection / correctly predicted bits); *Perseus* (*Bioinformatics* 2026 — multi-rank lineage-consistent confidence & backoff).

---

## 7. Current Best-Bet Architecture & Self-Contained Pre-Scale Implementation Plan (Updated 2026-10-19)

> **Self-Contained Agent Handoff Notice:** This section is written so an agent with **zero prior conversation context** can implement, test, launch on Kaggle, and evaluate every pre-scale experiment end-to-end.

---

### 7.0 Repository Map, Existing Corpora & Kaggle Datasets

#### Codebase Layout (`/home/danielle/Documents/veg/omniphyl`)
- **`scripts/fetch_refseq_corpus.py`**: Downloads NCBI RefSeq (and optional GenBank gap-fill) FASTA + GFF3 assemblies, intersects 500 bp windows with `CDS` intervals (or UCSC `VEuPathDBGeneModels`), resolves NCBI `rankedlineage.dmp` taxonomy (`domain`, `phylum`, `class`, `order`, `family`, `genus`, `species`), and writes `fragments.fna` + `labels.tsv`.
- **`scripts/screen_contaminants.py`**: Screens `fragments.fna` and emits `fragments_screened.fna` + `labels_screened.tsv`.
- **`scripts/train_omniphyl.py`**: Implements `fcgr_encode_ms` (`k=(4,5,6)` RC-canonical pyramid, lines `147–178`), `aa_encode` (6-frame 21-letter tripeptide spectrum, lines `249–276`), `FragmentDataset` (lines `326–415`), `ResNetEncoder` (`qt=True` quadtree-aligned stages, lines `441–465`), `OmniPhyl` (lines `625–690`), `build_split` (lines `721–783`), and the training loop (lines `934–1046`).
- **`scripts/eval_pilot.py`**: Reconstructs checkpoints, reproduces the exact 3-way split (`val` = held-out RefSeq genomes, `val_frag` = contiguous within-genome span holdout, `train` = remaining windows; plus `holdout_genera`), runs `M ∈ {1, 5, 20, all}` log-probability consensus pooling (lines `701–758`), and writes `results.json`.
- **`remote/run_pilot.sh`**: Environment-agnostic bash entrypoint invoked by Kaggle kernels (`kaggle/<arm>/kernel.py`). Forwards env vars to `fetch_refseq_corpus.py`, `train_omniphyl.py`, and `eval_pilot.py`.
- **`tests/test_smoke.py`**: Fast CPU unit + end-to-end smoke test (`python3 tests/test_smoke.py`). **Must pass before pushing any Kaggle kernel.**

#### Local Artifacts & Existing Kaggle Datasets
- **Local artifact cache (`/home/danielle/omniphyl-work`)**:
  - `mini_corpus/`: Unpacked 5-group mini corpus (`63,372` screened fragments, `2,534` genomes; `1` RefSeq assembly per family + GenBank train-only gap-fill for `archaea`, `protozoa`, `fungi`).
  - `a0_out/`, `a1_out/`, `a2_out/`, `a25_out/`, `a3_out/`, `a3f_out/`, `a4seq_out/`, `v4_out/`, `fungi_out/`: Completed checkpoints (`run/best.pt`) and evaluator outputs (`run/eval/results.json`).
  - `fungi_corpus/work/corpus/_ncbi_cache/`: Contains `461` cached `.gff.gz` files and `assembly_summary_*.txt` tables.
- **Mounted Kaggle Datasets (already uploaded under user `daniellecallan`)**:
  1. **`daniellecallan/omniphyl-5group-mini-corpus`**: `63,372` screened fragments (`1` assembly/family). Use for **fast shallow-rank (`domain`–`order`), coding/region-type, and structural-head ablations** (~12 min / 10 epochs on Kaggle T4). *Caveat:* Because `PER_RANK_N=1` at `family`, held-out RefSeq validation genomes have `n=0` in-vocab `family`/`genus`/`species` labels unless `--frag-val > 0` is used.
  2. **`daniellecallan/omniphyl-v4-corpus`**: `460,731` fragments across `3,571` genomes (`PER_RANK_N=2` per `family`, `NT_BUDGET=75000`). Has shared `family` vocabulary on held-out genomes (`475` families) and `19` multi-genome held-out species (`137` multi-genome species total). Use for **deep-rank (`family`/`genus`/`species`) and $M$-window consensus validation** (~55 min / 10 epochs on Kaggle T4).
- **New Kaggle Dataset Needed Only for Experiment E4/E5 (Annotated Features & Pfam/InterPro)**:
  3. **`daniellecallan/omniphyl-5group-annot-mini`** (to be generated by a 1-hour CPU fetch-only kernel `kaggle/fetch-annot-mini/`): Same 5-group mini selection (`ONE_PER_RANK=family`, `PER_RANK_N=2`, `NT_BUDGET=15000`, `REQUIRE_GFF=1`), built after extending `fetch_refseq_corpus.py` to emit the new GFF-derived columns (`region_type`, `active_frame`, `struct_flags`, `pfam_ids`, `interpro_ids`).

---

### 7.1 Current Expected Best Architecture (`OmniPhyl-MSFCGR-AA-LA`)

**One-line summary:**
`msfcgr` ($k \in \{4,5,6\}$ RC-canonical DNA quadtree + `pos_channels`) **+ coding-gated/bottlenecked `aa` aux** (6-frame 21-letter tripeptide spectrum + frame/stop summary) $\to$ **prior-logit-adjusted `flat` heads** $\to$ **NCBI-tree-reconciled $M$-window log-probability pooling + calibrated abstention at inference**.

| Component | Decision | Empirical Evidence from Ablations |
|---|---|---|
| **Primary Trunk: `msfcgr` (`[5, 64, 64]`, $3\times 3$ stride-1 stem + $2\times 2$ quadtree stages $\to 256\text{-d}$ $z_{\text{dna}}$)** | **KEEP** | `A2` (`msfcgr`) beat `A0` (`resnet` $k=7$) at every rank (`+3.9` to `+8.0` pts single-frag; `2×` deep-rank consensus) and beat `A4-seq` (1D CNN on one-hot DNA) by `+13.7` pts Phylum / `+26.4` pts Class on true held-out validation genomes. |
| **Aux Branch 1: `aa` (6-frame 21-letter tripeptide log-enrichment, `9,261-d` $\to$ bottleneck `64 → 128-d` $z_{\text{aa}}$)** | **KEEP (with bottleneck + coding gate)** | Adding `aa` in `A3f` drove the $M=20$ multi-fragment consensus jump on held-out genomes (`0.814 → 0.845` Domain, `0.620 → 0.671` Phylum, `0.548 → 0.601` Class, `0.222 → 0.242` Order) and lifted eukaryotic coding (`+15.5%` Phylum) and non-coding (`+4.5%` Phylum) classification as well as coding/non-coding AUROC (`0.811 → 0.833` overall; `0.701 → 0.757` in eukaryotes). Keeping all 6 frames + stop codon (`*` = symbol 20) beat single-frame (`min-stop`) selection (`0.370` vs `0.185` species) because of bacterial/viral operons (`20–30%` of coding windows span a frame change), high-GC antisense open frames (`16.5%`), and non-coding eukaryotic windows (`37%`). |
| **Aux Branch 2: `hash` (canonical 11-mer `EmbeddingBag`, `11.2M` params)** | **DROP** | Accounted for `72%` of total model parameters (`11.2M / 15.6M`), memorized training genomes, starved `msfcgr` via co-adaptation, and contributed `<0.01` on held-out validation when ablated from `A3f`. |
| **Output Heads: `flat` linear heads + Prior Logit Adjustment (`logits + tau * log_prior` in training CE; raw `logits` at eval)** | **KEEP `flat` + ADD Logit Adjustment; DROP `TreeRankHead`** | `TreeRankHead` (`A1`/`A2.5`/`A3`) failed at deep ranks (`Order = 0.053`) because `59%` of classes and `67%` of orders in sparse corpora are sole children of their parent (`singleton sibling groups` $\implies \log(1)=0$ loss and zero gradient), and parent misses lock out child marginals. Flat heads (`A2`/`A3f`) learn every rank independently. |
| **Multi-Fragment Consensus ($M \times 500\text{ bp}$ Chunk-Pool at inference)** | **KEEP in inference/eval** | Slicing a `1.5–10 kb` contig into $M$ windows of `500 bp` and summing prior-free log-probabilities beats encoding the entire contig as a single FCGR by `+15 to +20` pts (`0.847` vs `0.641` Domain, `0.675` vs `0.522` Phylum at `10 kb`). On `v4`, $M=20$ pooling lifts held-out species accuracy from `0.208` ($M=1$) to `0.842` (`16/19`). |

---

### 7.2 Step-by-Step Code & Experiment Implementation Plan

Below are the **6 self-contained implementation packages (`P0`–`P5`)**. Implement and smoke-test `P0` (Evaluator + Flat Prior Logit Adjustment) first so every subsequent experiment is scored with the false-positive-aware evaluator.

---

#### Package P0 — False-Positive-First Evaluator, Flat Prior Logit Adjustment & Inference Tree Reconciliation
**Goal:**
1. Upgrade `eval_pilot.py` so **Selective Precision ($1-\text{FDR}$)**, **Coverage @ Target Precision ($\text{Cov}@P_{90}, \text{Cov}@P_{95}$)**, **Unseen-Clade Over-Call (Hallucination) Rate**, and **NCBI Cross-Rank Tree Reconciliation** are computed automatically on every checkpoint (including existing `A2`, `A3f`, `v4` checkpoints without retraining).
2. Add `--flat-prior-tau` to `train_omniphyl.py` so `HEAD=flat` can apply Menon et al. (2021) prior logit adjustment during training.

**Exact Code Changes:**
1. **`scripts/train_omniphyl.py` (Flat Prior Logit Adjustment):**
   - In `build_tree(meta, vocab, ranks)` (lines `526–570`), also compute unconditional per-rank log-priors:
     ```python
     # Unconditional class prior for flat logit adjustment: log((count[c] + 1) / total)
     flat_lp = {}
     for r in ranks:
         n = len(vocab[r]) + 1
         lp = np.zeros(n, dtype=np.float32)
         lp[0] = -1e4  # unknown class 0 never predicted
         tot = counts[r][1:].sum() + (n - 1)
         for c in range(1, n):
             lp[c] = float(np.log((counts[r][c] + 1.0) / tot))
         flat_lp[r] = torch.tensor(lp, dtype=torch.float32)
     ```
   - In `main()` (around line `971`), when `args.head == "flat"` and `args.prior_tau > 0` during `train=True`, add `args.prior_tau * flat_lp[t].to(device)` to `logits[t]` **inside `F.cross_entropy` only** (keep raw `logits[t]` unshifted at `train=False` and in `eval_pilot.py` so inference is prior-balanced):
     ```python
     adj = (logits[t] + args.prior_tau * flat_lp[t].to(device)
            if (train and args.prior_tau > 0) else logits[t])
     terms = [F.cross_entropy(adj, y[:, i], ignore_index=0,
                              label_smoothing=args.label_smoothing) ...]
     ```
   - Save `"tree": {"parent_of": ..., "log_prior": ..., "flat_log_prior": ...}` in `ckpt` (`best.pt` line `1037`) **even when `args.head == "flat"`** so `eval_pilot.py` always has `parent_of` (the NCBI parent map for all in-vocab classes) available for inference-time tree reconciliation.

2. **`scripts/eval_pilot.py` (False-Positive Metrics & Inference Tree Reconciliation):**
   - **Build `parent_of` if missing from older flat checkpoints:** Call `build_tree(train_meta, vocab, cls_targets)` in `eval_pilot.py` whenever `"tree"` is not in `ckpt`.
   - **Metric 1 — Selective Precision ($1-\text{FDR}$) & Coverage @ Target Precision (`selective_precision` block):**
     - On `val_set` (held-out RefSeq genomes, **including** rows where `true[t] == 0` so confident calls on unseen clades count as false positives in the denominator):
       - For each rank $t$ and threshold $\tau \in \{0.50, 0.70, 0.80, 0.90, 0.95\}$:
         - `called = conf >= tau`
         - `tp = called & (true > 0) & (pred == true)`
         - `fp = called & ((true == 0) | (pred != true))`
         - Report `coverage = called.mean()`, `precision = tp.sum() / max(called.sum(), 1)`, `fdr = fp.sum() / max(called.sum(), 1)`.
       - Sweep $\tau \in [0.01, 0.99]$ on the validation set to report **`cov_at_p90`** and **`cov_at_p95`** (maximum coverage where $\text{Precision}_t(\tau) \ge 0.90$ or $\ge 0.95$).
   - **Metric 2 — Unseen-Clade Over-Call (Hallucination) vs. Safe Ancestor Stop (`unseen_clade_safety` block):**
     - Subset to fragments in `holdout_genera` (where `true["family"] > 0` and `true["genus"] == 0`).
     - At $\tau \in \{0.5, 0.8, 0.9\}$ (and at the calibrated $P_{90}$ threshold):
       - `overcall_rate`: fraction of unseen-genus fragments where `conf["genus"] >= tau` or `conf["species"] >= tau` (false positive hallucination of a known genus/species).
       - `safe_ancestor_stop_rate`: fraction where `conf["genus"] < tau` AND `conf["family"] >= tau` AND `pred["family"] == true["family"]`.
   - **Metric 3 — Cross-Rank NCBI Tree Reconciliation (`--reconcile-tree` mode in `eval_pilot.py`):**
     - Given flat per-rank log-probabilities $\log p_r(c \mid x)$ (`c = 1..C_r-1`) and `parent_of[r][c]`:
     - For every deepest rank $R \in \text{RANKS}$ and candidate taxon $c_R \in \{1..C_R-1\}$, trace its deterministic ancestor chain $a_{R-1} = \text{parent\_of}[R][c_R], \dots, a_0 = \text{parent\_of}[1][a_1]$.
     - Compute the **path-consistent score**:
       $$S_R(c_R \mid x) = \frac{1}{R+1} \sum_{k=0}^{R} \log p_k(a_k \mid x)$$
     - Also compute the **Cross-Rank Contradiction Veto**: if $\arg\max_c p_R(c \mid x)$ has an ancestor $a_k$ at any shallower rank $k < R$ whose rank-$k$ probability $p_k(a_k \mid x) < \tau_{\text{anc}}$ (default `0.20`), veto the rank-$R$ call and back off to the deepest consistent ancestor rank.
     - Report both raw flat accuracy/FDR and tree-reconciled accuracy/FDR at $M=1$ and $M=20$ in `results["tree_reconciled"]`.

**How to Run & Verify `P0`:**
- Local unit test: Add `test_flat_prior_la_and_reconcile()` to `tests/test_smoke.py` and run `python3 tests/test_smoke.py`.
- Zero-retraining baseline run on existing local checkpoints (`A2`, `A3f`, `v4`):
  ```bash
  python3 scripts/eval_pilot.py --model /home/danielle/omniphyl-work/a3f_out/run/best.pt \
      --fragments /home/danielle/omniphyl-work/mini_corpus/fragments_screened.fna \
      --labels /home/danielle/omniphyl-work/mini_corpus/labels_screened.tsv \
      --outdir /home/danielle/omniphyl-work/a3f_out/eval_fp --val-only
  ```

---

#### Package P1 (Experiment E1) — Bottlenecked & Coding-Gated `aa` Fusion (Fixing Co-Adaptation)
**Goal:** Stop `aa_proj` from starving `msfcgr` during training while preserving or improving `aa`'s $+4\text{–}15\%$ lift on coding windows and $M=20$ consensus.

**Exact Code Changes in `scripts/train_omniphyl.py` & `remote/run_pilot.sh`:**
1. **Expand `aa_encode(seq)` (`scripts/train_omniphyl.py:249–276`) with an `18-d` Frame/Stop Summary (`AA_EXTRA_DIM = 18`, total `AA_DIM = 9279` when `--aa-mode gated`):**
   - For each of the 6 reading frames (`3` forward, `3` reverse-complement), compute 3 scalar features:
     1. `stop_density = n_stops / max(n_codons, 1)`
     2. `longest_orf_frac = max_run_without_stop / max(n_codons, 1)` (computed via `np.diff(np.where(is_stop)[0])`)
     3. `tga_ratio = n_TGA / max(n_stops, 1)` (distinguishes NCBI Code 4 *Mycoplasma*/*Spiroplasma* `TGA→Trp` from standard Code 1/11).
   - Concatenate these `18` floats to the `9,261-d` log-enrichment tripeptide vector (`9261 + 18 = 9279`).
2. **Add `--aa-mode` (`plain` | `bottleneck` | `gated`) and `--aa-drop` (`float`, default `0.0`) to `OmniPhyl.__init__` and `forward` (`scripts/train_omniphyl.py:625–689`):**
   - `bottleneck`:
     ```python
     self.aa_proj = nn.Sequential(
         nn.Linear(aa_in_dim, 64),
         nn.LayerNorm(64),
         nn.GELU(),
         nn.Linear(64, aux_dim),
         nn.LayerNorm(aux_dim),
     )
     ```
   - `gated`:
     - Compute `coding_frac` logit directly from `z_dna` (`self.coding_head_dna = nn.Linear(self.encoder.latent, 1)`) so coding detection never depends on `aa`.
     - Compute gate $g = \sigma(\text{self.aa\_gate}(\text{torch.cat}([z_{\text{dna}}, l_{\text{cf}}.\text{detach}(), x[\text{"aa"}][:, -18:]], \text{dim}=1)))$ of shape `[B, aux_dim]`.
     - During training (`self.training and self.aa_drop > 0`), apply per-sample Bernoulli branch mask $m_{\text{drop}} \sim \text{Bernoulli}(1 - p_{\text{aa\_drop}})$ of shape `[B, 1]` so for `20%` of training fragments $z_{\text{aa}}$ is completely zeroed out.
     - Fuse $z = \text{self.drop}(\text{torch.cat}([z_{\text{dna}}, m_{\text{drop}} \cdot g \cdot z_{\text{aa}}], \text{dim}=1))$.
3. **Wire env vars in `remote/run_pilot.sh`:**
   - Add `AA_MODE="${AA_MODE:-plain}"` and `AA_DROP="${AA_DROP:-0}"` and pass `--aa-mode "$AA_MODE" --aa-drop "$AA_DROP"`.

**Kaggle Dataset & Runs for E1:**
- **Dataset:** Mount existing `daniellecallan/omniphyl-5group-mini-corpus` (for fast 3-arm screening, ~12 min/arm) and run the winner on `daniellecallan/omniphyl-v4-corpus` (~55 min).
- **Arms to run (`BACKBONE=msfcgr`, `HEAD=flat`, `AUX=aa`, `PRIOR_TAU=1.0`, `FRAG_VAL=0.1`):**
  - `E1a`: `AA_MODE=bottleneck`, `AA_DROP=0.0`
  - `E1b`: `AA_MODE=gated`, `AA_DROP=0.0`
  - `E1c`: `AA_MODE=gated`, `AA_DROP=0.2`
- **Evaluation command & Gate:**
  - Run `eval_pilot.py` twice on each checkpoint: (1) normal, (2) `--zero-aux aa`.
  - **Pass Gate:** Under `--zero-aux aa`, `E1c` must retain $\ge 0.65$ single-frag / $\ge 0.80$ $M=20$ Domain accuracy (matching standalone `A2` instead of collapsing to `0.56 / 0.67` like `A3f`), while normal eval beats `A3f` on both single-frag and $M=20$ Phylum/Class/Order precision and coverage (`cov_at_p90`).

---

#### Package P2 (Experiment E2) — Stochastic Read-Length Truncation (`150–500 bp`) & Paired-End (`PE250`) Support
**Goal:** Make a single trained checkpoint accurate across Illumina short reads (`150 bp`, `250 bp`), merged/paired reads (`2×150 bp`, `2×250 bp`), and standard `500 bp` contig windows without regressing at `500 bp`.

**Exact Code Changes:**
1. **`scripts/train_omniphyl.py` (`FragmentDataset.__getitem__`, lines `391–402`):**
   - Add CLI args `--min-train-len` (default `500`) and `--var-len-prob` (default `0.0`).
   - Inside `FragmentDataset.__getitem__`, when `self.augment` (or `self.var_len_prob > 0`) and `np.random.rand() < self.var_len_prob`:
     - With probability `0.7`, sample a contiguous sub-window of length $L \sim \text{Uniform}(\text{min\_train\_len}, |s|)$ at a random offset $o \in [0, |s|-L]$: `s = s[o:o+L]`.
     - With probability `0.3`, simulate an unmerged paired-end read by taking the `150 bp` prefix and `150 bp` suffix of `s` joined by `"N"`: `s = s[:150] + "N" + s[-150:]` (`N` breaks cross-junction k-mers and codons cleanly because both `fcgr_encode` and `aa_encode` drop windows containing `N`).
2. **`scripts/eval_pilot.py`:**
   - Add CLI flag `--eval-crop-len` (`int`, default `0` = full 500 bp) and `--eval-paired-len` (`int`, default `0`; e.g., `150` takes `s[:150] + "N" + s[-150:]`) so the evaluator can benchmark any checkpoint at `150 bp`, `250 bp`, `PE150`, `PE250`, and `500 bp`.

**Kaggle Dataset & Runs for E2:**
- **Dataset:** `daniellecallan/omniphyl-5group-mini-corpus`.
- **Arm (`E2-varlen`):** Best config from `E1` + `--min-train-len 150 --var-len-prob 0.35`.
- **Evaluation:** Run `eval_pilot.py` on `E2-varlen` at `--eval-crop-len 150`, `--eval-crop-len 250`, `--eval-paired-len 150`, and full `500 bp`.
- **Pass Gate:** Improves single-end `150 bp` and `250 bp` Phylum/Class accuracy and `cov_at_p90` by $\ge +5$ pts over zero-shot `E1`, with $\le 0.5$ pt change at `500 bp`.

---

#### Package P3 (Experiments E3, E4, E5) — Rich GFF Extraction & Multi-Task Structural / Functional Heads
**Goal:**
- **E3:** Upgrade coding detection from a biased scalar to a calibrated 5-class **Region-Type Composition Head** (`CDS`, `intron`, `intergenic`, `rRNA/tRNA`, `pseudogene`) + class-balanced coding loss.
- **E4:** Predict **Active Reading Frame (`7`-way)** and **Structural Transition Presence (`[has_start, has_stop, has_splice_donor, has_splice_acceptor]`)** on 500 bp windows, plus a lightweight Stage-2 local motif scanner for exact coordinate resolution.
- **E5:** Predict present **Pfam / InterPro protein superfamilies** (`multi-label binary head` over top-`K` domains, default `K=256`) directly from 500 bp windows.

**Exact Code Changes:**
1. **Step 1 — Extend `scripts/fetch_refseq_corpus.py` (`load_gff_cds` -> `load_gff_features`, lines `319–383`):**
   - Parse GFF3 lines in a single pass per assembly (retaining 100% stdlib compatibility):
     - **Feature intervals by type:** Record 0-based half-open intervals for `CDS`, `exon`, `mRNA`/`transcript`, `rRNA`, `tRNA`, and `pseudogene`.
     - **CDS strand, phase & true ends:** For each `CDS` line `(seqid, start, end, strand, phase, attrs)`:
       - Compute the genomic reading frame index $f \in \{0,1,2,3,4,5\}$ aligned with `aa_encode`'s 6 frames:
         - On `+` strand: the first complete codon of the CDS starts at `cds_codon0 = start + ((3 - phase) % 3)`. Relative to a fragment `[frag_start, frag_end)`, the forward frame offset is `off_plus = (cds_codon0 - frag_start) % 3` $\implies$ frame index `0, 1, 2`.
         - On `-` strand: the first complete codon from the `3'` end (`end`) is at `cds_codon0_rc = end - ((3 - phase) % 3)`. Relative to the reverse-complemented fragment of length `L = frag_end - frag_start`, `off_minus = (frag_end - cds_codon0_rc) % 3` $\implies$ frame index `3, 4, 5`.
       - Group `CDS` intervals by `Parent=` (or `protein_id=`) to distinguish **translation start (`start_pos`)**, **translation stop (`stop_pos`)**, and **internal exon–intron splice junctions (`donor_pos`, `acceptor_pos`)**:
         - For a multi-exon `Parent`, internal boundaries between consecutive sorted `CDS` exons are splice sites (`donor` at exon $i$ end on `+` / start on `-`; `acceptor` at exon $i+1$ start on `+` / end on `-`), whereas the outermost ends of the first/last `CDS` are the true `start_codon` and `stop_codon` coordinates.
       - Extract `Dbxref=` tokens matching `InterPro:IPR\d+` and `PFAM:PF\d+` from the 9th GFF column (`attrs`).
   - **New columns emitted to `labels.tsv` (backward-compatible append after the 7 `RANKS` columns):**
     - `intron_frac`: fraction of window inside `mRNA` but outside `exon`/`CDS` (`[0,1]`).
     - `rna_frac`: fraction of window overlapping `rRNA` or `tRNA` (`[0,1]`).
     - `active_frame`: integer `0..5` if a single CDS frame covers $\ge 50\%$ of the coding bases in the window, `6` if multiple overlapping/operon frames are present, `7` if non-coding (`coding_frac < 0.1`), `-1` if GFF unavailable.
     - `struct_flags`: 4-bit integer mask (`0..15`) or 4-char bitstring (`b0=has_start`, `b1=has_stop`, `b2=has_splice_donor`, `b3=has_splice_acceptor`) indicating which structural transitions fall strictly inside `(frag_start, frag_end)`.
     - `domain_ids`: comma-separated `IPR.../PF...` accessions overlapping the window by $\ge 90\text{ bp}$ (`30 AA`), or `""`.

2. **Step 2 — Keep Per-Frame Structure in `aa_encode` & Add Optional Heads in `scripts/train_omniphyl.py`:**
   - **Per-Frame AA Summary for Frame Prediction (`E4`):**
     - In `aa_encode(seq)`, in addition to the summed `9,261-d` spectrum (which is frame-invariant for taxonomy), emit a compact per-frame feature matrix `frame_feats` of shape `[6, 24]`: for each of the 6 frames, the `21` single-amino-acid + stop frequencies plus `(stop_density, longest_orf_frac, codon_pos3_gc)`. Flatten `[6, 24] -> 144-d` and append to `x["aa"]`. Because the 6 frames are in a fixed order `(F0, F1, F2, R0, R1, R2)`, a linear/MLP head on `frame_feats + z_dna` can directly classify `active_frame ∈ {0..7}`!
   - **New Optional Multi-Task Heads in `OmniPhyl` (enabled via CLI flags so taxonomy-only runs are unchanged):**
     - `--region-head`: Adds `self.heads["region_fracs"] = nn.Linear(self.latent, 4)` predicting `[coding_frac, intron_frac, rna_frac, intergenic_frac]` trained with KL-divergence / soft cross-entropy + inverse-prior weighting on pure non-coding windows (`w_noncoding = 3.0`) to fix the `91%` coding class imbalance (`E3`).
     - `--struct-head`: Adds `self.heads["active_frame"] = nn.Linear(self.latent + 144, 8)` (`F.cross_entropy`, ignore_index `-1`) and `self.heads["struct_flags"] = nn.Linear(self.latent + 144, 4)` (`F.binary_cross_entropy_with_logits` with `pos_weight=torch.tensor([5., 5., 8., 8.])`) (`E4`).
     - `--domain-head-k 256`: Builds `domain_vocab` from the top `K` (`256` or `512`) most frequent `IPR/PF` IDs in `train_meta` (requiring $\ge 20$ training windows per domain), adds `self.heads["domains"] = nn.Linear(self.latent, K)`, and trains with masked `F.binary_cross_entropy_with_logits` (only on windows from assemblies that had `InterPro`/`PFAM` annotations in their GFF so unannotated genomes are not treated as false negatives) (`E5`).

3. **Step 3 — Stage-2 Exact Coordinate Scanner (`scripts/eval_pilot.py`, for `E4`):**
   - On validation windows where `struct_flags` predicts `has_start` or `has_stop` and `active_frame` predicts frame $f \in \{0..5\}$, scan the raw 500 bp sequence in frame $f$ for canonical start (`ATG`/`GTG`/`TTG`) or stop (`TAA`/`TAG`/`TGA`) triplets, score each candidate by distance to predicted `offset` (`sigmoid(logits["offset"]) * 500`), and report **exact base-coordinate hit rate** (`|pred_coord - true_coord| <= 3 bp`).

**Kaggle Dataset & Runs for E3–E5:**
- **Dataset Build Kernel (`kaggle/fetch-annot-mini/`):**
  - Run `fetch_refseq_corpus.py` with `GROUPS="protozoa fungi bacteria viral archaea"`, `ONE_PER_RANK=family`, `PER_RANK_N=2`, `NT_BUDGET=15000`, `REQUIRE_GFF=1`, `VEUPATH=1`, `FETCH_ONLY=1` (CPU kernel, ~1.5 hours).
  - Promote output tarball to Kaggle Dataset `daniellecallan/omniphyl-5group-annot-mini`.
- **Training Arms on `daniellecallan/omniphyl-5group-annot-mini` (~15 min each on T4 GPU):**
  - `E3-region`: `BACKBONE=msfcgr`, `AUX=aa`, `AA_MODE=gated`, `--region-head`
  - `E4-struct`: `E3-region` + `--struct-head`
  - `E5-pfam`: `E4-struct` + `--domain-head-k 256`
- **Evaluation Metrics & Pass Gates in `eval_pilot.py`:**
  - **E3 Gate:** Eukaryotic (`fungi` + `protozoa`) coding/non-coding AUROC $\ge 0.88$ (up from `0.757`), non-coding F1 $\ge 0.75$ (up from `0.511`), and `intron_frac` vs `intergenic_frac` separation AUROC $\ge 0.80$.
  - **E4 Gate:** Active reading frame (`0..7`) top-1 accuracy $\ge 85\%$ on coding windows; `has_start`/`has_stop`/`has_splice` macro-AUROC $\ge 0.85$; Stage-2 in-frame stop/start coordinate localization within $\pm 3\text{ bp}$ on $\ge 70\%$ of single-boundary windows.
  - **E5 Gate:** Macro-averaged AUPRC and AUROC across the 256 Pfam/InterPro families on held-out RefSeq genomes (`AUROC ≥ 0.85`, top-5 domain recall@5 $\ge 0.50$), plus check whether adding `--domain-head-k 256` improves `family`/`genus` taxonomy accuracy via multi-task representation sharing into $z_{\text{aa}}$.

---

### 7.3 Execution Order & Summary Checklist for the Implementing Agent

| Step | Package | Where it runs | Inputs / Dataset | Expected Wall Time | Output / Deliverable |
|---|---|---|---|---|---|
| **1** | **P0: Eval + Flat Prior LA + Tree Reconcile** | Local CPU | Existing `a2_out`, `a3f_out`, `v4_out` checkpoints | ~15 min coding + 5 min eval | Updated `eval_pilot.py`, `train_omniphyl.py`, `tests/test_smoke.py`; baseline FP/FDR/Cov@P90 tables |
| **2** | **P1 (E1): Gated & Bottlenecked `aa`** | Kaggle GPU (T4) | `daniellecallan/omniphyl-5group-mini-corpus` (then `v4`) | 3 × 12 min (mini) + 55 min (`v4`) | Winner `AA_MODE=gated` + `AA_DROP=0.2` verified with `--zero-aux aa` |
| **3** | **P2 (E2): Variable Read Length (`150–500 bp` + `PE`)** | Kaggle GPU (T4) | `daniellecallan/omniphyl-5group-mini-corpus` | 15 min train + 5 min multi-length eval | Single checkpoint supporting `150 bp`, `250 bp`, `PE150/250`, and `500 bp` |
| **4** | **Fetch `annot-mini` Corpus** | Kaggle CPU (`FETCH_ONLY=1`) | NCBI RefSeq + UCSC VEuPathDB | ~1.5 h (no GPU quota used) | New Kaggle dataset `daniellecallan/omniphyl-5group-annot-mini` with region, frame, splice, and Pfam/InterPro labels |
| **5** | **P3 (E3 + E4 + E5): Region, Frame/Start/Stop & Pfam Heads** | Kaggle GPU (T4) | `daniellecallan/omniphyl-5group-annot-mini` | 3 × 15 min | Validated multi-task architecture ready for full-scale (`A5` + GTDB/IMG-VR dark matter) training |



