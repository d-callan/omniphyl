# OmniPhyl — Use-Case Catalog

*Living list of everything OmniPhyl could plausibly do, what each would require beyond the core plan in `omniphyl-design.md`, and how feasible it looks. Order ≈ increasing distance from the core. Last updated: 2026-10-02.*

**Feasibility legend** — HIGH: precedent at similar scope or trivial extension of core · MED: proven at narrow scope or needs real additions · LOW: speculative / weak precedent / known hard regime.

---

## Tier A — free with the core model (no plan changes)

| Use case | What it is | Feasibility |
|---|---|---|
| **Sample taxonomic profile** | Whole-tree abundance table per metagenome (the core product). | HIGH — the design itself. |
| **Universal fragment classifier** | One contig/sequence in → organism call at deepest confident rank. Bonus mode already in design doc. | HIGH — per-fragment heads already exist; labels unbounded via simulated fragments. |
| **Targeted presence/absence** | "Is taxon X present?" — aggregate per-fragment calls → calibrated yes/no. Stronger than abundance (few confident contigs suffice). | HIGH — aggregation + calibration layer only. |
| **Sequence embeddings as a product** | Export latent vectors for contigs/samples → similarity search, clustering, downstream features. | HIGH — latent exists for free; needs an export API + eval vs. sketch baselines (open question 4). |
| **Contig binning** | Cluster latent embeddings → metagenome bins. k-mer composition is *the* classic binning signal (TNF in VAMB et al.). | MED-HIGH — needs clustering layer + eval vs. MetaBAT/VAMB; no architecture change. |
| **Novel-lineage flagging** | Unknown-mass channel + latent OOD → "coherent clade not in training set" alerts; biosurveillance hook for CHAODYNE. | MED-HIGH — required by design anyway (unknown channel + leave-clade-out); product = calibration + thresholds. |
| **Same-organism equivalence** | When identity can't be resolved: "are fragments i and j from the same organism?" — anonymous organism clustering as the graceful degradation below named-taxonomy. | MED-HIGH — pairwise latent distances may suffice (eval); small contrastive head if not. Free labels: same-reference fragment pairs. Also strengthens contig binning + novel-clade coherence. |
| **Consistency/QC checker** | Spectrum-reconstruction residual flags samples where the profile doesn't explain the data → chimera/contamination/label-error detection. | MED — consistency decoder already in design; residual analysis needs building. |

## Tier B — aux heads on the shared latent (labels exist, wiring is cheap)

| Use case | Head | Labels | Changes to plan | Feasibility |
|---|---|---|---|---|
| **Region-type calling** | CDS / rRNA / tRNA / ncRNA / intergenic / mobile | GFF boundaries on reference genomes → inherited by fragments | +1 head; done. | MED-HIGH — Emam (coding/non-coding) + Hoang (splice sites) precedent; but must beat prodigal+infernal to ship (open q6). |
| **Mobile-element typing** | prophage / plasmid / chromosomal | geNomad-style element calls | +1 head; also *required* to stop prophage contigs inheriting carrier taxonomy. | MED — a real need, doubles as taxonomy-quality fix. |
| **Gene-category calling** | COG/GO-style functional category | Reference annotations | +1 head | MED — mid-fit expected; cap at category, not pathway role. |
| **Host-range prediction** | virus → host taxon (phage→bacterial host; euk-virus→mammal/bird/plant; "can it infect humans" = Human-as-class + unknown channel) | Virus-Host DB, IMG/VR, euk-virus host records | +1 multi-rank head; benefits from `--pos-channels` | MED-HIGH for phages (PhageCGRNet ≈ SOTA: 80/89/91% sp/g/f); MED-LOW for novel-virus human infectivity — degrades with train/test similarity, and true infectivity needs host factors. |
| **AMR / virulence-class calling** | CARD / VFDB categories | Resistance + virulence DBs | +1 head | MED-LOW — AMR often = few SNPs; k-mer context may capture lineage proxies more than mechanism. Evaluate, don't promise. |
| **Strain-virulence / pathotype calling** | "How dangerous is this new strain of species X?" — pathotype class (EHEC vs commensal, hvKP, hypervirulent ribotypes) or VF-bearing-contig flag | Strain metadata (NCBI pathovar/serotype/ST fields), VFDB annotations → inherited to fragments | +1 head (pathotype class, or per-contig VF flag → aggregated); no architecture change | MED — the mechanism is usually accessory-element-borne (stx prophage, rmpA/iro plasmid, tcd locus), which is *per-contig detectable* — the fragment granularity is the strength here. PaPrBaG-style read-level pathogen/non-pathogen precedent. Honest limit: frame as pathotype/VF-content call; outcome-level virulence labels don't exist at scale and are host/dose-confounded — virulence *regression* stays LOW until labels do. |
| **Biome/source prediction** | Sample-level: which ecosystem (gut/soil/marine/sewage…) | Sample metadata | Pooled latent → sample head | MED-HIGH — biome signatures are strong; trivial head at sample level; also doubles as leave-one-biome-out eval. |

## Tier C — real additions (new modality or finer granularity)

| Use case | What it is | Changes to plan | Feasibility |
|---|---|---|---|
| **Domain/functional calling (Pfam)** | AA-FCGR stem (20-gon polyflake, kissing-number scaling) → domain classifier | New input modality: polyflake encoder + AA stem + domain-window corpus | MED — Dick & Green prove protein CGR works; genuine engineering cost; hmmscan is a near-perfect incumbent for known families. |
| **Strain/variant-level typing** | Below-species calls: lineages, genotypes, cultivars, clones (e.g., SARS-CoV-2 Pango, norovirus genotypes, outbreak strain tracking) | Sub-species labels in taxonomy layer; likely wants longer k / pos channels (variant signal = sparse k-mer deltas); possibly variant-window fragments | MED→MED-HIGH — viral evidence is strong: CouGaR-g (FCGR+ResNet50) hit 96.3% on SARS-CoV-2 GISAID clades; FCGR+CNN VOC typing hit 99.9% on 47k genomes with temporal-drift + geographic-shift evals. Grapevine WGS (vbaf193): 99% on 5 species, 80% on 41 cultivars — cultivar/clonal resolution is the hard regime. Strain head also fixes "closest strain" reporting for outbreak use. |
| **Protein-level identification** | Direct AA-sequence input (not just domains) — translated metagenomic reads, proteomes | Same polyflake stem as domain head | MED — rides the AA modality investment. |

## Tier D — analysis modes / parked ideas

| Use case | What it is | Feasibility / status |
|---|---|---|
| **Alignment-free phylogeny** | FCGR distances (SSI/descriptor) or latent distances → trees | HIGH as analysis mode (Karamichalis metrics proven); LOW as product — phylogeny is a solved, cheap problem. Keep as eval/diagnostic, not a feature. |
| **FCGR-data-augmentation via vertex permutations** | Already implemented (`--vertex-permute-augment`) | Done. |
| **Raw-read path** | K-mer-embedding input for unassembled fraction | In design as secondary path; DeepMicrobes precedent; MEDIUM priority behind contigs. |
| **Teacher-channel prediction** | Predict each profiler's output + disagreement (calibration signal) | Alternative label scheme noted in design decision 1; no new head type needed. |
| **Structure prediction** | Fold/secondary-structure calls from FCGR | PARKED — no credible precedent; domains is the ambitious-enough version. |

---

## Cross-cutting notes

- **Every aux head is optional at product time** — open question 6 gates each on beating a free incumbent (prodigal, hmmscan, sylph query). The shared latent is the durable output regardless.
- **Labels are the currency**: any use case whose labels can be inherited to simulated fragments for free is cheap to try; use cases needing wet-lab truth (AMR phenotypes, true infectivity) are expensive no matter the head size.
- **Granularity is the axis that costs most**: sample → contig → fragment → variant is a spectrum; each step finer needs proportionally more signal per unit. The variant/strain tier is the frontier — the grapevine result says CGR resolves cultivars, not that it resolves SNVs.
- **Adjacent modalities bound the comparison**: OmniPhyl is sequence-space ID (no culture, works on metagenomes). Physical-fingerprint ID — dried-droplet desiccation patterns → ResNet (91.8% acc, 7 BSI pathogens, Guan 2025 J Nanobiotech) — is faster and cheaper *per isolate* but needs pure culture and one organism at a time. Position OmniPhyl against MALDI-TOF/droplet-class tools for isolates and against k-mer profilers for mixtures; never against both at once. Broader field frame: AI/ML infectious-disease prediction review (Decoding Infect. Transm. 2026).
