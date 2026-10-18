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
