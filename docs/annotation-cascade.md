# OmniPhyl — Structural Annotation: Two-Stage Cascade Design

*Companion to [`omniphyl-design.md`](omniphyl-design.md). Records the analysis of whether fragment classification (taxonomy/identification) and structural annotation (gene models) should be one model or two, and the resulting cascade architecture. Last updated: 2026-10-10.*

---

## The question

A related proposal: **positional FCGR (PFCGR) over sliding windows + 2D CNN → gene structural annotation** (exon/intron/intergenic prediction — a gene finder). Does this belong inside OmniPhyl, or is it a separate project?

## The dividing line: does the output live on genomic coordinates?

Everything else follows from this single architectural fact.

- **Per-window labels** (taxonomy, region-type, boundary candidacy, contaminant flags): the FCGR rasterization is a *feature*, not a cost — it compresses the k-mer spectrum into a fixed-size image that a CNN handles well. All such tasks share one encoder, one forward pass, one dataset. This is OmniPhyl.
- **Base-resolution outputs** (a GFF of exon boundaries): the FCGR rasterization is *fatal*. The image's spatial axes are k-mer suffix coordinates, not genomic coordinates — there is no axis to attach a per-base output head to. Positional channels (mean-x/mean-y per cell) decode a location only when a k-mer occurs exactly once; repeated motifs — which is what splice/start signals are — average to ambiguity. FCGR compresses the composition axis and keeps ~no position; gene finding needs the opposite compression (keep every position, abstract over motif families).

**Conclusion: one model for everything classification-shaped; a second stage for the one task where output resolution must equal input resolution.** Not two models for two domains — one model plus a resolution bridge.

## Signal asymmetry (which direction "gives free" the other)

A genuine asymmetry exists between the task families:

- Annotation features (splice motifs, 3-periodicity, start/stop context) are *local and positional*. A latent trained to preserve them retains fine structure; global composition — most of what taxonomy needs — tends to fall out of finer-grained training more easily than the reverse.
- Taxonomy-primary training *can* legitimately destroy annotation signal: pooled latents are rewarded for composition and penalized for nothing if boundary detail is dropped.
- Counterweight: annotation signal is partially taxon-invariant (a donor looks similar across eukaryotes). An annotation-*only* encoder could learn to discard the taxon-discriminative composition taxonomy needs.

Resolution: **joint heads, not task inversion.** Make annotation co-primary in *training* (it forces positional detail into the latent) while taxonomy stays the *product*. This is loss weighting, not architecture.

## On PFCGR moment channels

The 5-channel PFCGR proposal (freq + mean + std + skewness + kurtosis per cell) is denser in storage, not in usable signal at 500 bp. Most k-mer cells see 0–5 observations; first moments (what `--pos-channels` already encodes) survive that regime, third/fourth moments are noise. Moment channels start paying off at 5–10 kb windows. **Not adopted** — the existing 3-channel encoding is the right density for the pilot regime.

## Two-stage cascade

```
stage 1 (fuzzy):  strided windows over contig → shared OmniPhyl encoder
                  → region-type + boundary-probability tracks per window
                  → candidate intervals ("boundary ∈ [x, x+300)")
stage 2 (exact):  raw-sequence scorer over candidate intervals only
                  → base-exact boundary calls → GFF
```

### Stage 1 is free — and unusually clean

A sliding window over a contig *is* a fragment. Training samples uniform random windows; inference samples strided windows. **Identical input distribution, no domain adaptation** — a property most cascades don't get. Cost: one more linear head (boundary/transition-type) on the shared latent, plus a sampling-mode change in `fetch_refseq_corpus.py` to emit boundary-stratified fragments (uniform windows rarely contain boundaries; GFF coordinates are already parsed).

**Label schema — fractions, not majority class.** Stage-1 windows carry composition-fraction soft targets (e.g., 0.6 coding / 0.4 intergenic), not a hard class. Majority-class labeling would teach the model that mixed spectra *mean* the dominant class — erasing exactly the mixture signature candidacy depends on, and quietly degrading boundary recall, which is the cascade's governing constraint. Fraction targets are strictly more informative: candidacy falls out as any non-pure prediction, triple-boundary windows need no special class, and the boundary's rough direction is encoded by which side's mass dominates. GFF coordinates make the fractions free to compute at sampling time. This is also where the pos-channels earn their keep: they are the position-recovery mechanism for FCGR's position blindness, and fraction targets force the model to *place* mass rather than merely classify — the mean-x/mean-y offset of coding-typical k-mers carries the directionality.

One strided pass emits **parallel tracks**: taxonomy + region-type + boundary probability per window. Two emergent properties:

- **Cross-task coherence**: a confident boundary call inside a window whose taxonomy is "unknown" at all ranks is a novel-organism *coding-region* flag — sharper than novel mass alone.
- **Cheap smoothing**: region-type shouldn't flicker across adjacent windows; a light HMM over the track cleans calls before stage 2 runs.

Limits: stage 1 stays fuzzy (±half-stride, ~150–250 bp). It does triage, not detection — honest framing matters for claims.

**Concrete plan:** train one small shared model producing both task families jointly — taxonomy multi-rank + unknown-mass, region-type fraction head, boundary-candidacy head, optional offset head. If joint training degrades neither task, the one-encoder claim is verified; if annotation detail degrades taxonomy, heads split before backbones do. See `omniphyl-design.md` ("Near-term concrete plan").

### Stage 2 is mostly not a model

Once a ~500 bp candidate interval + transition type are known, classical scorers are near-optimal:

- **Splice donor/acceptor**: PWM argmax over the interval; MaxEntScan-style scoring where dependencies matter. The model picks *which* PWM via its predicted class.
- **CDS start**: ORF logic — among 3 frames × 2 strands, the ATG/GTG/TTG whose downstream ORF runs stop-free through the coding region (+ Shine-Dalgarno scan in bacteria). This is prodigal's core mechanic; nearly deterministic *given a genetic code*.
- **CDS stop**: first in-frame stop entering non-coding. Free.

Division of labor: stage 1 answers **what kind of boundary, roughly where** (the learned part); stage 2 answers **which base** (the mechanical part). Stage 2's scorers are also the incumbent baseline open question 6 requires — build them first, and a learned stage 2 must beat them on divergent held-out clades to earn its seat.

*Middle tier*: an offset-regression head on the shared latent (predict boundary position within window, continuous) plausibly reaches ±10–50 bp — worth trying, narrows stage-2 scan intervals, zero extra models, never base-exact alone.

### Governing constraint: stage-1 recall

A boundary the fuzzy pass misses is unrecoverable. Stage 1 must run high-recall/tolerant-precision; the cascade's compute savings depend entirely on the candidate-flag rate. If 20% of the genome gets flagged to hold 99% recall the cascade wins; if it needs 60%, it's a slow version of the raw model.

### Deployment-length split

- **500 bp unassembled reads**: cascade degenerates — the fragment *is* the window; region-type call is the final answer.
- **kb+ contigs**: boundary head feeds stage 2. Same encoder front end, resolution-appropriate backend only where context exists.

## Reframed value: the clade-agnostic long tail

The cascade was originally justified on compute — fuzzy pass prunes, exact pass finishes. A stronger justification emerged comparing incumbent ab initio tools (SINTEZ, Tiberius, ANNEVO): **every incumbent ships clade-trained weights; a two-phase stack can avoid needing any.** The niche is organisms with canonical-ish biology but no training data — exactly where nobody's model has weights, and exactly our target clades' long tail.

### Phase 1 stays clade-agnostic by construction

The k-mer spectrum is a clade-agnostic feature space: coding/intergenic contrast lives in codon bias and compositional asymmetry, far more conserved than any positional motif. Weak targets also tolerate weak labels — "this window is mostly coding" needs only silver-standard annotation; boundary slop barely perturbs the spectrum.

### Phase 2 = self-calibrated content sensor + parameterized grammar

The mechanism that makes no-weights viable: **estimate everything from the target genome itself** — the GeneMark-ES trick, which is the existing existence proof that self-training can replace clade models. Phase-1 candidacy fixes self-training's usual failure mode (contaminated seeds on messy genomes).

- **Content score**: in-frame hexamer log-likelihood ratio + 3-periodicity against self-estimated tables — universal signals, arithmetic not learning.
- **Grammar decode**: start/stop/splice DAG where the biology is config, not weights — genetic code table, donor/acceptor dinucleotide set (GT-AG / GC-AG / atypical), intron priors, polycistron or trans-splice mode. Non-canonical clades become user parameters.
- **Code inference**: stop-codon frequency per candidate table over confident ORFs (EukPhylo `stop_codon_freq` precedent) — genetic code becomes a phase-1.5 output with a confidence flag, not a required input. The same enumeration trick detects splice consensus.

### Phase 2.5 = protein-space adjudication

Protein chemistry doesn't drift where nucleotide motifs do — a nonsense ORF translates to garbage in every clade. Borrow the oracle pattern as a cost gradient, not a fixed stage: amino-acid composition norms → polyflake protein-ness → pLM (ESM-2-class) only on near-ties. The oracle sees only the final candidate set — the position where SINTEZ's every-candidate-both-strands cost problem mostly dissolves. Self-BLAST paralog consistency is a free, orthogonal check: real genes form families, garbage translations don't.

### Evidence and honest ceiling

- **Existence proof**: SINTEZ trained a ~238k-param backbone on yeast/worm/fly and hit 0.81–0.88 MCC zero-shot on 80%-AT *P. falciparum* — universal features + grammar transfer farther than per-clade intuition suggests. Its DAG decoder is already most of the parameterized-grammar piece; code table and splice set are hardcoded but a config file away.
- **Ceiling**: this is GlimmerM-grade machinery (~70–80% gene-level historically). UTR edges, tiny exons, overlapping genes, isoforms stay soft — content sensors are mushy and grammar junctions do the sharpening. It competes where no trained model exists, not against ANNEVO where one does.
- **Failure modes to design around**: frontier starvation (peak-dependent decoding collapses when emissions are weak — observed empirically in soft-NMS-vs-Viterbi experiments on a semi-trained encoder; candidate generation must run high-recall); AT degeneracy compressing FCGR contrast at ~80% AT; pLM systematic bias against asparagine-rich apicomplexan proteins (relative ranking survives a uniform shift; absolute energy cutoffs may not).

Where curated GFFs exist, the per-clade parameter library (adaptation 3) remains the fast path. Self-calibration is the path for the rest of the tree.

## Target clades change where the difficulty sits

Primary targets — **apicomplexans, kinetoplastids, fungi** — are precisely where incumbent tools underperform, for mechanistic reasons:

| Organism | Genome feature | Impact |
|---|---|---|
| *P. falciparum* et al. | ~80% AT | FCGR mass concentrates in quadtree corner (effective alphabet shrinks → consider k=5–6 or multi-k); acceptor polypyrimidine tract loses discrimination because the whole genome is Py-rich — metazoan PWMs degrade on flanks |
| Apicomplexa broadly | Short sparse introns (*Plasmodium* ~50% genes, 100–300 bp; *Cryptosporidium* ~0) | 500 bp window contains exon–intron–exon triplets → mixed labels; stride/window tradeoff or smaller windows for boundary task |
| Kinetoplastids | Polycistrons + trans-splicing, ~no cis-introns | "Boundary" = SL-acceptor/polyA junction + ORF-in-polycistron; a universal splice class trains on a signal that barely exists |
| Fungi | Canonical splice signals, intron-poor in yeasts, dense genes | Closest to metazoan regime; easiest of the target set |
| Ciliates et al. (scope adjacent) | Genetic code reassignments (UAA/UAG→Gln, UGA→Trp/Cys) | "First in-frame stop" is code-dependent; ORF logic needs a per-clade code parameter |

### Required adaptations

1. **Label schema**: transition-type classes covering donor/acceptor + SL-acceptor + CDS start/stop + polyA — not a universal "splice site" class. Possibly clade-conditional heads.
2. **Training data tier**: RefSeq is thin here (the corpus-design culture-bias gap). EuPathDB annotations are the gold labels — curated, RNA-seq-informed. Implemented as a fetch tier in `fetch_refseq_corpus.py --veupath`: UCSC GenArk mirrors the models as the `VEuPathDBGeneModels` contrib track (~440 assemblies, release 68), pulled via the UCSC JSON API with `chromAlias.txt` seqid bridging — simpler than the WDK API route, one uniform endpoint, per-assembly fallback to RefSeq GFF (`label_source` column records the tier).
3. **Stage 2 = parameterized scorer library**, Augustus-style: per-clade PWMs refit from curated GFFs + per-clade genetic code table. Not one universal scorer. Cross-task option: taxonomy head's call could select the code table; or infer code from stop-usage in the contig itself (EukPhylo precedent — genetic-code breakpoint step).
4. **Learned stage 2 earns its seat exactly here**: motif determinants drift off published parameter sets in these lineages. A learned boundary resolver evaluated on leave-phylum-out apicomplexans is the same novelty-recall story as taxonomy — and the pilot's eval harness already measures it.

### Eval truth set

EuPathDB curated models on held-out species, covering the schema's corners:

- *Cryptosporidium* — extreme minimal case (near-zero introns, overlapping genes, compact genome)
- *T. brucei* — polycistron/trans-splicing case
- *P. falciparum* — AT-extreme case
- Compare vs. Augustus (with species-specific params) and Helixer as incumbents; PWM+ORF stage 2 as the internal baseline.

## Relationship summary

| Question | Answer |
|---|---|
| Same encoder? | Yes — sliding windows are fragments; identical input distribution |
| Same model? | Yes for everything classification-shaped (taxonomy, region-type, boundary candidacy); no for base-resolution output. Near-term test: one small joint model — heads split before backbones if the tasks interfere |
| PFCGR adopted? | No — 3-channel pos encoding suffices at 500 bp; moment channels need 5–10 kb windows |
| Annotation primary? | Co-primary in *training* (forces positional detail into latent); taxonomy stays the product head |
| Stage 2 a model? | Mostly no — clade-parameterized PWM + ORF logic; learned resolver only if it beats them on divergent clades |
| New deliverable? | Base-resolution gene finder for apicomplexans/kinetoplastids/fungi — justified precisely because incumbents underperform there |
| Clade-specific weights? | Not required — self-calibrated content stats + parameterized biology + protein-space adjudication keep the stack clade-agnostic; per-clade parameter libraries stay the fast path where curated GFFs exist |

## Open questions added

1. What is stage-1 boundary recall vs. candidate-flag rate at stride 100–250 bp on *P. falciparum*? (The cascade-economics gate.)
2. Does offset regression on the pos-channels latent reach ±50 bp — enough to shrink stage-2 intervals meaningfully?
3. Does a learned stage 2 beat clade-parameterized PWM+ORF on leave-phylum-out apicomplexans — or only on organisms with no fitted parameter set?
4. Can the taxonomy head reliably predict genetic code where code tables matter (ciliates, kinetoplastid UGA variants)?
5. Does FCGR-seeded self-calibration converge where GeneMark-ES-style self-training fails on signal-poor genomes?
6. Does near-tie-gated pLM adjudication lift precision enough to justify its cost on the final candidate set?
7. Does coding/intergenic FCGR contrast survive ~80% AT compression — and at what k?
