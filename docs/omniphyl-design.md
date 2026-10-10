# OmniPhyl — CGR→Abundance Whole-Tree Metagenome Profiler

*Narrow-scope companion to `outbreak-dynamics-design.md` (§"One pipeline, two observables", front-end paragraph). This doc scopes ONE deliverable: a learned unified profiler.*

Status: concept stage, 2026-09-29. Working title — OmniPhyl (omni-tree phylogeny profiler).

## What it is

Three steps, no more:

1. **Teacher labeling** — run the traditional per-domain profiler stack on a massive, ecosystem-diverse metagenome corpus → per-sample species presence/abundance for **bacteria + viruses + eukaryotes in one unified table**.
2. **Encoding** — compute CGR/FCGR (chaos game representation; image position = k-mer suffix identity) signatures for the same samples. Prior art de-risks the narrow versions of this bet: CNN-on-FCGR 16S taxonomy (Rizzo), protein source-organism classification (Dick & Green), mammalian coding/non-coding discrimination (Emam), microbial-genome comparison for metagenomics (Karamichalis) — see Löchel & Heider 2021 review. OmniPhyl's novelty is scale + multi-rank whole-tree + per-fragment granularity, not the encoding itself.
3. **Learned surrogate** — train a new model to predict the unified presence/abundance profile **from the CGR input alone**.

Deliverable: one forward pass returns the whole-tree profile — the "reconciler" nobody has built, except the reconciliation is done by a model that internalized all the teachers.

## Feasibility: is this doable?

**Yes — every component exists; the novelty is the surrogate + one training trick.** Honest assessment:

**Why it's tractable:**
- **Data exists at the required scale.** Public metagenomes: MGnify (10⁵–10⁶ processed runs), Earth Microbiome Project (~27k samples, deliberately biome-diverse), HMP, OceanDNA/Tara (marine), SRA metagenome BioProjects (soil, sewage, host, environmental). Ecosystem diversity — the thing the model needs — is already archived.
- **Teacher labeling is embarrassingly parallel.** sylph ~min/sample at ~5 GB RAM; MetaPhlAn 4.2 similar; Kraken2 heavier but routine. 10⁵ samples × ~10 min ≈ thousands of CPU-hours per tool — well within reach on Galaxy/cluster infrastructure the group already runs (brc-tools/IWC machinery applies directly; teacher-labeling is itself a nice Galaxy workflow).
- **FCGR is cheap.** It's just a normalized k-mer count rastered as an image — jellyfish/KMC/sylph-sketch outputs give it nearly free.
- **The surrogate-model pattern is proven.** Teacher→student distillation (MDSINE→forecasts, cNODE→composition maps, Bury→EWS from simulated universals) is the established play.
- **Ground truth for eval exists without teachers** — CAMI2 mock communities and synthetic spike-ins give *true* answers, not teacher answers.

**The two real risks:**
- **Teacher-label ceiling.** A pure imitation model inherits every teacher's false positives, domain skews, and "unclassified" black holes — including never learning organisms the teachers can't name. *Mitigations:* multi-teacher consensus labels (agreement-weighted, not single-tool truth); an explicit **unknown/novel mass channel** as a predicted output; the consistency loss below; and — with per-fragment inputs — **simulated fragment labels** from reference genomes (perfect labels, no teacher in the loop) for the core supervision.
- **Long-tail output space.** 10⁴–10⁶ species targets with extreme imbalance; rare species get zero training signal. *Mitigation:* hierarchical or genome-bin targets, subspecies collapsed to species, log-scale abundance loss.

**The upgrade that makes it better than imitation — deconvolution consistency loss:**
A predicted abundance profile is *verifiable*: mixing the reference genomes' FCGR signatures weighted by predicted abundances must reproduce the sample's observed FCGR spectrum (FCGR = k-mer spectrum, so this is linear mixing). This is a physics-style self-supervised loss on top of imitation — the model can correct teacher errors *against the raw sequence evidence*, and the residual unexplained spectrum IS the novel-organism mass. This turns "learn to copy tools" into "learn to solve spectral unmixing where tools provide priors."

**What it can't do (scope honesty):**
- Novel organism *naming* — model can flag novel mass and return **partial taxonomy** (deepest confident rank via multi-granularity heads), not assign species names where none exists.
- Sub-k resolution: FCGR is lossy at high k (image resolution cap) — fine for abundance, not for strain-level calls. sylph-style sketching retained for cases needing exact ANI.

## Pipeline sketch

| Component | Contents | Status |
|---|---|---|
| `teacher-run` | Galaxy workflow: {Kraken2+Bracken, MetaPhlAn 4.2, mOTUs 3, sylph (GTDB+IMGVR+FungiRefSeq), Kaiju, geNomad} on corpus → unified per-sample abundance table + per-tool confidence | All tools exist, IUC-wrapped or wrappable |
| `encode` | **Per-contig FCGR** (assembly upstream; k≈6–8 image per contig — contigs are kb-scale so FCGR is informative; short unassembled reads → k-mer-embedding path) + pooled whole-sample FCGR retained as consistency substrate/baseline | Trivial compute; standard technique |
| `model` | **Per-fragment set encoder (primary)**: FCGR image per contig → shared CNN encoder → multi-granularity heads at every rank (domain → … → species) + unknown-mass channel per rank → per-fragment calls pooled to sample profile (abundance = fragment-weighted, genome-size normalized). Novel organism = fragments confidently resolving to *deepest confident rank* — partial taxonomy ("an unknown Rickettsiales") on foreground objects, not residual mass. Loss = simulated-fragment supervision + teacher distillation at sample level + deconvolution consistency + compositional (clr/Dirichlet) | DeepMicrobes/geNomad-lineage per-fragment classification is proven; the corpus is still the hard part |
| `eval` | CAMI2 mocks, synthetic spike-ins, **leave-one-biome-out** generalization (train w/o marine, test on Tara), per-teacher agreement vs. model agreement | CAMI datasets public |

### Architecture detail

```
per-contig FCGR image (128×128, k=7; multi-k channels optional)
  → shared ResNet-style encoder (~ResNet-18) → per-contig latent
  → per-contig heads:
      multi-rank abundance/taxonomy, hierarchical softmax
      P(child)=P(parent)·P(child|parent) → deepest-confident-rank
      calls; factorizes the wide species output layer
      unknown-mass channel per rank
      element-type head (chromosomal/plasmid/viral — à la geNomad;
        horizontal elements confound organism calls)
      aux task heads (shared latent): region-type
        composition fractions (CDS/rRNA/tRNA/ncRNA/
        intergenic/mobile — mixture-aware soft targets,
        not majority class), boundary candidacy +
        within-window offset (pos-channel-informed),
        domain/functional via AA-FCGR translation path
  → pooling: per-contig calls → sample profile
      (fragment-weighted, genome-size normalized)
  → consistency decoder: profile × fixed FCGR signature matrix
      S (species×4^k) → reconstructed pooled spectrum ≈ sample FCGR
```

- **Why CNN**: CGR spatial proximity = shared k-mer suffixes; convolutions learn shared-sequence structure directly. Short raw reads (<FCGR resolution) take a k-mer-embedding path (DeepMicrobes-style) into the same latent/heads. Transformer/attention is not required for v1 — it enters only via the raw-read token path or optional attention pooling across contigs; ablate rather than assume.
- **Backbone ablation (implemented, `--backbone resnet|vit`)**: the "CNN can't see global structure" argument is only half true — heads already consume a *globally pooled* latent, and taxonomy is largely aggregate k-mer statistics, which pooling captures. What a ViT genuinely adds is *structured* global mixing where discriminative cells are sparse and scattered across the quadtree (high-k, strain-level, motif co-occurrence). Precedent: PCVR (ViT + masked-autoencoder pretraining on FCGR) beats CNN baselines at superkingdom/phylum. Two things make the ViT arm unusually attractive *for us specifically*: ViT's data-hunger is the standard objection and it's void here (unbounded simulated fragments), and the whole unlabeled RefSeq/metagenome corpus is free MAE-pretraining data. Deeper options if vanilla ViT wins: Swin (hierarchical shifted windows mirror the quadtree) or per-cell tokens — note FCGR is not a natural image: pixel = a specific k-mer, so neither conv weight-sharing nor square patching is the ideal inductive bias; quadtree-relative-position attention is the principled end-state.
- **Labels**: fragment-level supervision is *simulated* — reads/contigs generated from every reference genome carry perfect labels, so teachers aren't the training bottleneck; teacher profiles serve eval + sample-level consistency. Novel-class behavior trained via leave-clade-out splits.
- **Losses**: fragment CE (simulated labels) + sample-level teacher distillation KL + spectrum reconstruction (consistency) + cross-rank coherence + focal/class-balanced weighting for the long tail. Consistency residual metric: use SSI or descriptor distance rather than plain Euclidean — Karamichalis's 6-metric FCGR comparison showed they dominate for genome-distance work.
- **Inference**: assemble → per-contig forward passes → pooled per-rank abundances + foreground novel calls. A novel genus = a coherent set of contigs confidently calling family-rank with species-rank mass in unknown channel → partial taxonomy free. Pooled whole-sample FCGR retained as ablation baseline (answers open question 2) and consistency substrate.

## Data corpus (training labels layer)

- **MGnify / EBI metagenomics** — largest processed public corpus, multi-biome.
- **EMP** — designed for ecosystem breadth.
- **HMP / human cohort studies** — deep human-gut coverage.
- **Tara Oceans / OceanDNA** — marine.
- **SRA environmental/soil/sewage projects** — fill domain gaps; sewage doubles as CHAODYNE W-compartment proxy data (see design doc data table).
- **CAMI2 mock communities** — held-out truth.
- **Logan contigs** — pre-assembled SRA metagenome contigs at petabase scale w/ run metadata; removes the assembly dependency for corpus + eval (see pilot spec below).

Scale target: order 10⁴–10⁵ samples for v1 corpus (matches what teacher-labeling at ~10 min/sample makes cheap); model wants ≥10⁵ species instances across samples to cover the long tail.

### Pilot corpus (proof-of-legs trial)

Minimum corpus to answer "does this have legs" without building the full pipeline. Three tiers, ordered by increasing cost — each is a go/no-go gate:

| Tier | Contents | Answers | Cost |
|---|---|---|---|
| 0 — simulated fragments | ~10k reference genomes (GTDB subsample + ~1k euk/protozoa/fungi + ~5k viral IMGVR) → 1–50kb fragments → ~1–5M FCGRs; deliberate leave-clade-out holdouts | Can FCGR→taxonomy be learned at all? Does deepest-confident-rank behave on novel clades? | Trivial (k-mer counting + sim); no teachers, no real data |
| 1 — real-contig transfer | ~10⁵–10⁶ Logan contigs, biome-stratified by source run | Does simulated-fragment training transfer to real contigs (errors, strain divergence, unknown content)? **The core risk.** | Low — Logan gives contigs free; eval via per-contig teacher calls + pooled consistency |
| 2 — sample-level | ~300–500 real metagenomes across ~6–8 biomes (gut/marine/soil/sewage/oral/skin) + CAMI2 | Do pooled per-contig calls reproduce teacher profiles? Is novel mass meaningful? | Teacher stack on 500 samples ≈ ~100s CPU-hr — cheap |

Tier 0 alone kills or greenlights the architecture. Tiers 1–2 test distribution shift + pooling, the two places the surrogate could still fail.

**Footprint & hardware (Tier 0)**: store fragment *sequences*, compute FCGRs in the loader — ~10–50 GB packed for 1–5M fragments (vs. ~80–160 GB if FCGRs are precomputed dense uint8; labels are a few ints/fragment, negligible). Training is a CIFAR-scale workload (ResNet-18 × 128×128): a single modern GPU (≥12–16 GB VRAM) does a full run in tens of GPU-hours; CPU-only is survivable for a ~100k-fragment smoke subset, not for full runs. FCGR computation is CPU-trivial (k-mer counting — jellyfish/KMC or a small count kernel).

## Design decisions to lock before Phase 1

1. **Teacher strategy**: consensus across tools vs. per-tool target channels (model predicts each teacher + ensemble)? Consensus is simpler; multi-channel retains disagreement signal for calibration.
2. **Target granularity**: resolved to multi-rank heads (species → … → domain) with deepest-confident-rank reporting; remaining sub-question is leaf type — named-species labels vs. genome bins (ANI-space) at the species tier.
3. **Input granularity**: resolved — per-fragment primary, **contig-level preferred** (kb-scale FCGRs are informative, 10³–10⁵ units/sample, assembly already upstream); raw reads via k-mer-embedding path for unassembled fraction; pooled whole-sample FCGR kept as baseline + consistency substrate, not a product mode.
4. **k for FCGR**: 6–8 balances resolution vs. sparsity; may run multi-k channels. The literature flags resolution as *the* dominant FCGR hyperparameter (Löchel & Heider) — budget a k-sweep ablation, and take the free augmentation: permuting the 4 corner vertices is a label-preserving image permutation (Dick & Green used vertex-reordering as augmentation), implemented as a digit-remap at encode time.
5. **Unknown-mass channel**: required, or the model forces errors onto known species.
6. **FCGR normalization**: current implementation normalizes per-fragment (`img /= img.max() + 1e-9`); Löchel & Heider note dataset-level normalization may preserve inter-sample differences — an ablation, not a lock.
7. **FCGR input channels**: extend the 1-channel frequency image to 3-channel (frequency + mean-CGR-x + mean-CGR-y per cell) — PhageCGRNet's ablation shows positional channels alone beat frequency alone; the within-cell mean position summarizes the context *preceding* each k-mer, extending effective context beyond k. Implemented as `--pos-channels` in `train_omniphyl.py`. Second role, discovered via the annotation work: the pos-channels are FCGR's only position-recovery mechanism — for the annotation fuzzy pass they carry boundary *directionality and mass placement* (which side of a mixed window is coding), trained explicitly by fraction targets. Taxonomy mostly needs the spectrum; annotation needs where the spectrum sits.
8. **Corpus decontamination**: the simulated-fragment labels inherit whatever contamination lives in the reference assemblies — a human-alu-containing scaffold in a bacterial RefSeq genome silently stains the training set (the model learns "alu → that bacterium"). Required pre-slicing filter in `fetch_refseq_corpus.py`, layered: (a) assembly-level metadata confidence — prefer complete/annotated assemblies, flag suspicious fields (mislabelled organism_taxid, low-contamination-check scores); (b) per-contig contaminant screen — cheap k-mer containment vs. host/vector/common-contaminant databases (mash/Kraken-style; GCF_000001405 et al., UniVec, phiX); (c) self-consistency check — a contig whose k-mer spectrum is an outlier vs. the rest of the assembly is suspect (the FCGR itself is a contaminant detector: human-tagged kmers light up specific cells). A mislabeled reference is worse than a dropped one: filter aggressively, log what was removed per assembly.

## Why this matters beyond itself

- Fills the **reconciler gap** (docs above) with a model, not a rules engine.
- Produces a **labeled corpus at scale** — the same corpus-generator logic as CHAODYNE `corpus-gen`, but for composition rather than dynamics; the unified profiles ARE the state vectors for the microbiome track.
- The trained encoder (FCGR → profile) is a **sequence encoder** usable downstream: embeddings for the gene-pool dynamics layer, fast similarity search, anomaly detection (novel-mass flagging as outbreak/biosurveillance signal).

**Bonus mode — universal fragment classifier.** Per-contig forward pass alone = "identify this sequence": one contig in → organism call out (partial taxonomy at deepest confident rank, no abundance needed). For identification alone, training data is effectively unbounded — every reference genome (GTDB/RefSeq/IMGVR/euk collections + MAGs) simulates unlimited perfectly-labeled fragments, no teacher in the loop. Caveats: signal scales with contig length (short fragments degrade to deeper-rank calls by design); mobile elements need the element-type head so a prophage/plasmid isn't assigned its carrier's taxonomy.

**Bonus mode — targeted presence/absence.** "Is *Giardia* present?" = per-fragment evidence aggregation: fraction of contigs confidently assigned to the target at any rank → calibrated sample-level yes/no. Works on any sample type (stool, blood, water concentrate) since it's just fragment calls. Detection is the *strong* case vs. abundance — a few confident target-assigned contigs suffice; consistency-spectrum residual bounds false positives. Caveats: sensitivity bounded by sequencing depth + assembly success (low-biomass protozoa in stool = hard case); cheap baseline to beat is a sylph containment-ANI query of the target genome — the model's edge is divergent/novel-lineage targets where ANI thresholds fail.

**Multi-task heads — the latent is task-agnostic.** Same heads pattern, different questions; every label below comes *free* from reference-genome annotations and inherits to simulated fragments exactly like taxonomy (unbounded scale, no teachers):
- **Region-type head** (CDS/rRNA/tRNA/ncRNA/intergenic/mobile-element): coding DNA's 3-periodicity and rRNA/tRNA k-mer signatures are strong FCGR-visible features — CGR was historically used for alignment-free gene finding (yin-yang scans); this head is the modern version — and it already has ML precedent: Emam et al. classified mammalian coding vs. non-coding on CGR features; Hoang et al. detected splice sites on CGR coordinates. Labels = GFF/CDS boundaries — emitted as **composition fractions** (a 0.6-coding window is labeled 0.6/0.4, not "intergenic"), which makes boundary candidacy a first-class output (non-pure prediction = candidate) and forces the pos-channels to carry directionality. This head is **co-primary in training** with taxonomy (annotation detail keeps positional structure in the latent) and is phase 1 of the two-stage annotation cascade — see `annotation-cascade.md`.
- **Domain/functional head**: nucleotide FCGR carries weak domain signal (domains are residue-level); the meaningful variant is a translation path — six-frame translate → amino-acid FCGR → domain classifier. Proteins need the **polyflake** form, not the square: >4 vertices require kissing-number scaling (Fiser) to avoid overlap, and Dick & Green showed protein polyflakes beat back-translated DNA FCGRs for source-organism prediction — so the AA stem should encode a binned 20-gon polyflake grid. **This is a second input modality, not a weight-free adapter**: AA-FCGR texture (residue motifs — zinc-finger Cys runs, heptads, TM hydrophobics) differs from nt-FCGR texture (codon periodicity), and the 20-symbol alphabet forces smaller k (20⁴=160k cells at k=4 vs. 4⁷=16k at k=7). Wire as modality-specific stems (nt-FCGR stem + aa-FCGR stem) → shared trunk → heads, so the domain head trains without disturbing taxonomy weights. Frame resolution either via region-type head output or six-frame set input with pooling over frames. Labels are again free: Pfam-annotated reference proteins → simulated domain-window fragments at arbitrary scale. Honest caveat: hmmscan/Pfam is near-perfect for *known* families — edge is divergent/unannotated families, same as taxonomy.
- **Gene-category head** (COG/GO-style "what kind of gene"): plausible mid-fit; pathway-level function is context-dependent and a fragment sees one gene at a time — cap at category, not role.
- **Host-range head** (virus → host taxon): proven task — PhageCGRNet (frequency + CGR positional features → tiny CNN) beats RaFAH/vHULK/DeepHost/CHERRY/PHPGCA/CL4PHI/PB-LKS on phage→bacterial-host prediction at species/genus/family (~80/89/91% acc). Product framing: "can this virus infect humans" = host-species head with Human as a class + unknown channel, reported at deepest confident rank — generalizes to euk viruses (mammal/bird/plant host classes) with the same multi-rank pattern. Labels free from Virus-Host DB / IMG/VR / euk-virus host records, inherited to simulated viral fragments like taxonomy. Honest caveats: accuracy degrades with train/test similarity (their stratified-similarity experiment — novelty is the hard part, same as taxonomy); the FCGR host signal is a compositional proxy (host-adapted composition + entry-gene motifs), not mechanistic proof — true human infectivity lives in host factors too. Bonus interpretability: their SHAP analysis found discriminative k-mers are low-frequency/high-specificity and map onto host-interaction genes (tail fiber, lysozyme) — cell → k-mer → genome locus is a ready-made explanation path for ours.
- **Same-organism equivalence head** (pairwise: fragments i,j → same-source?): the graceful fallback when a taxonomy call can't be made — a coherent anonymous cluster is still a usable answer ("these contigs are one organism" even if it's an unknown genus). No label bottleneck: same-reference fragment pairs are free positives. Wire as a contrastive objective on the latent (organism-level grouping) or a small pairwise head; also upgrades contig binning and novel-clade coherence from post-hoc clustering into a trained property.
Aux heads are free-ish to add at training time; product question is which beat incumbents (open question 6).

**Near-term concrete plan:** train a single small shared model producing taxonomy + fuzzy annotation tracks jointly — taxonomy multi-rank + unknown-mass heads, region-type fraction head, boundary-candidacy head, optional within-window offset head. Strided-window inference over contigs yields parallel tracks; the same model serves the fragment-classifier product and phase 1 of the annotation cascade. If joint training shows no degradation on either task, the cascade design's "one encoder" claim is verified; if annotation detail degrades taxonomy, heads split before backbones do.

## Phases

- **Phase 1 — corpus + teachers.** Assemble multi-biome corpus (~10⁴ samples), run teacher stack on Galaxy, unify labels; assemble contigs + compute per-contig and pooled FCGRs; build simulated-fragment label set from reference genomes. *Gate:* corpus + manifest + label-quality audit vs. CAMI2.
- **Phase 2 — surrogate v1 (per-fragment).** Contig-FCGR set encoder → multi-rank calls → pooled sample profile; simulated-fragment supervision + sample-level distillation + consistency loss; leave-one-biome-out + leave-clade-out evals; pooled-spectrum baseline ablation. *Gate:* beats/matches per-teacher accuracy on mocks; novel-mass flags recover held-out clades at partial taxonomy.
- **Phase 3 — exceed the teachers.** Novel-mass calibration, spike-in recovery, raw-read path for unassembled fraction. *Gate:* recovers species teachers missed (CAMI truth), calibrates unknown mass.
- **Phase 4 — delivery.** Galaxy tool (fits brc-tools tools/ conventions), corpus + model release, docs.

## Open questions

1. How much does the consistency loss rescue teacher blind spots vs. amplify teacher bias? (Phase 2–3 gate.)
2. Ablation: how much does per-fragment beat pooled-spectrum at low abundance (the baseline comparison), and does per-contig dominate per-read enough to justify the assembly dependency?
3. Leave-one-biome-out transfer — does the model generalize across ecosystems or memorize biome priors?
4. Does the learned representation beat raw k-mer/sketch features for downstream tasks (the encoder-as-product question)?
5. Partial-taxonomy calibration: does leave-clade-out training (hold out whole genera/families) produce well-calibrated deepest-confident-rank calls — an unknown genus still landing on the right family/order? Borrow the taxonomic leave-out protocol standard in phage-host-prediction evals.
6. Aux-head justification: does each multi-task head beat its cheap incumbent — region-type vs. prodigal+infernal, domains vs. hmmscan/Pfam? Heads that can't beat a free baseline stay off the product; the shared latent is the durable output regardless.

## References

Use-case catalog: [`use-cases.md`](use-cases.md) — every application considered, per-use-case plan deltas, feasibility.

Pilot plan: [`pilot-plan.md`](pilot-plan.md) — focused 500 bp / tree-of-life proof-of-concept; scripts, pipeline, sequence-based contaminant screen design, evaluation targets.

Annotation cascade: [`annotation-cascade.md`](annotation-cascade.md) — structural-annotation relationship analysis; two-stage design (shared-encoder fuzzy pass → clade-parameterized base-resolution scorers); EuPathDB-tier targets (apicomplexans/kinetoplastids/fungi); genetic-code and label-schema adaptations.

Method anchors in `~/Documents/idresolver/lit_search_chaos_microbiology.md`: §2 (CGR/FCGR entry — Jeffrey 1990, Randhawa 2020), §8 toolbox (sequence representations). Core review: [Löchel & Heider, Chaos game representation and its applications in bioinformatics](https://doi.org/10.1016/j.csbj.2021.11.008) (CSBJ 2021). Primary precedents: Rizzo 2016 (CNN on FCGR, 16S taxonomy); Dick & Green 2019 (protein polyflakes, source-organism CNN; vertex-permutation augmentation); Emam 2020 (CGR coding/non-coding); Hoang 2019 (CGR splice sites); Karamichalis 2013/2015 (FCGR microbial-genome comparison, SSI/descriptor distance); Deschavanne 1999 (genomic signature); Sengupta 2020 (CGR+ML viral taxonomy); [Wang et al., PhageCGRNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC13065082/) (PLoS Comput Biol — 3-channel FCGR+CNN phage-host prediction; positional>frequency ablation; similarity-stratified robustness eval; SHAP→host-interaction k-mers); PCVR (ViT + MAE pretraining on FCGR — backbone-ablation precedent); CouGaR-g (FCGR+ResNet50, SARS-CoV-2 intra-species clades, 96.3%); FCGR+CNN SARS-CoV-2 VOC typing (99.9% on 47k genomes, temporal-drift + geographic-shift evals). Teacher stack: sylph (Nat Biotech 2024), Kraken2/Bracken, MetaPhlAn 4.2 (vJan25 DB incl. euk + viral SGBs), mOTUs 3, Kaiju, geNomad. Eval: CAMI2. Consistency-loss precedent: spectral-unmixing/deconvolution framing is standard in bulk-RNA cell-type deconvolution — same math, k-mer spectra instead of expression profiles.

---
*Last Updated: 2026-10-10*
