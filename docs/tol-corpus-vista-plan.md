# Plan 2: Tree-of-Life Corpus Construction & Production Training on TACC Vista (`OmniPhyl-ToL`)

**Status:** Ready for autonomous agent execution **after Plan 1 (`docs/tol-arch-plan.md`) completes**
**Required input contract:** `docs/tol-arch-locked.json` (produced at the end of Plan 1)
**Target HPC system:** TACC Vista (`vista.tacc.utexas.edu` — Grace-Grace `gg` ARM64 CPU nodes & Grace-Hopper `gh`/`gh-dev` GH200 96 GB HBM3 nodes)
**Target codebase:** `d-callan/omniphyl` (`scripts/`, `slurm/vista/`, `docs/`)

---

## 1. Entry Gate & Hardware Ground Rules on TACC Vista

### 1.1 Hard Prerequisite Gate
Before running any corpus build or SLURM job in this plan, verify:
1. `docs/tol-arch-locked.json` exists and passes schema check (`schema == "omniphyl-tol-arch-v1"`).
2. `python3 tests/test_smoke.py && python3 tests/test_tol_arch.py` exits `0`.
3. Read all Stage-1 (`width`, `head_bottleneck`, `domain_head_k`, `aa_extra_v2`, `dna_complexity_stats`, `replicon_head`, `var_len_prob`) and Stage-2 (`arch_version`, `chunk`, `chunk_stride`, `hidden`, `dilations`, `cond_mode`) flags directly from `docs/tol-arch-locked.json` — **never hardcode stale v5/v6 hyperparameters in SLURM scripts**.

### 1.2 TACC Vista Hardware & Policy Constraints (Critical)
Vista is **not** an x86 cluster. Every script and environment setup in `slurm/vista/` must obey these Vista-specific invariants:
- **CPU Architecture (`aarch64` / ARM64):** All login, `gg` (Grace-Grace, 144 cores, ~237 GB LPDDR5), and `gh` (Grace-Hopper, 72 Grace cores + 116 GB LPDDR5X + **1 × NVIDIA H200/GH200 96 GB HBM3** connected via 900 GB/s NVLink-C2C) nodes are **ARMv9 Neoverse V2**. Never copy x86_64 wheels/conda envs or x86 binaries (`hmmer`, `minimap2`); install native `linux-aarch64` packages on Vista.
- **1 GPU per `gh` node:** Each `gh` / `gh-dev` node has **1 GH200 GPU** (96 GB HBM3). Multi-GPU DDP therefore means **multi-node DDP** (`#SBATCH -p gh -N 4 -n 4`, 1 task per node, 72 CPU threads/workers available to feed 1 H200).
- **Filesystems:**
  - `$HOME` (~23 GB quota, VAST NFS): code checkout + lightweight configs only. **Never put environments, caches, or corpora in `$HOME`.**
  - `$WORK` (~1 TB quota, shared Lustre): persistent software environment (`$WORK/envs/omniphyl-arm64`), curated manifests, final `.pt` checkpoints, evaluation reports, and compressed release archives.
  - `$SCRATCH` (multi-PB VAST flash NFS — **NOT Lustre; do NOT run `lfs setstripe` on `$SCRATCH`**, purged on inactivity): all high-IOPS transient data — NCBI/GenBank/GTDB download caches, transient genome `.fna.gz`/`.gff.gz`, unpacked `fragments.fna` + `labels.tsv`, `<corpus>.packed/` `mmap` binary arrays, and Stage-1 `.res.pkl` refiner caches.
  - `/tmp` (~286 GB node-local SSD): optional per-node staging during training if needed, though VAST `$SCRATCH` + `mmap` into 116 GB Grace LPDDR5X per node is already ultra-fast.
- **SLURM Directive Rules on Vista (`docs.tacc.utexas.edu/hpc/vista/running`):**
  - First line must be `#!/bin/bash` (never `#!/bin/sh`).
  - **Never pass `--mem` in `#SBATCH`** — Vista's Slurm rejects `--mem` (whole-node allocation is standard).
  - **Never use `$WORK` or `$SCRATCH` inside `#SBATCH -o` / `#SBATCH -e` lines** (Slurm does not expand shell env vars in directives; use relative `logs/%x.%j.out` and submit from `$SCRATCH/omniphyl-runs` or `$WORK/omniphyl`).
  - **Never pass `--export`** to `sbatch` (breaks TACC module propagation).
  - **Never run builds, `pyhmmer`, slicing, or training on login nodes.** Login nodes are strictly for lightweight orchestration, `sbatch` submission, and throttled `aria2c`/`urllib` staging when compute-node egress is restricted; check compute-node HTTPS egress in a 5-min `gg` test job first, and support both compute-node fetch (`gg` partition, 0.33 SU/node-hr!) and login/datamover prefetch seamlessly via `--cache-dir $SCRATCH/omniphyl-cache`.
  - **Queues:**
    - `gg` (CPU, 144 cores, 0.33 SU/hr, 48h max): **use `gg` for all corpus slicing, GFF parsing, `pyhmmer` Pfam annotation, contaminant screening, and `pack_corpus.py`!** Spending 1.0 SU/hr `gh` GPU hours on CPU bound `pyhmmer`/GFF work wastes 3× SUs.
    - `gh-dev` (1–8 GH200 nodes, 2h max, max 1 running job): smoke tests, throughput calibration, single-epoch timing.
    - `gh` (1–64 GH200 nodes, 48h max, 1.0 SU/node-hr): full Stage-1 and Stage-2 production runs.

---

## 2. Part A — Full Tree-of-Life Stage-1 Corpus Specification (`omniphyl-tol-v1`)

### 2.1 Why Naive Flat Sampling Fails Across All 9 Groups
In RefSeq + GenBank:
- A single mammal (`3.1 Gb`) or hexaploid plant (`5–17 Gb`) is **1,000× larger** than a bacterium (`3 Mb`) and **100,000× larger** than an RNA virus (`10 kb`), while coding density drops from `~88%` in prokaryotes to `~1.2–2%` in mammals!
- Conversely, RefSeq has `>350,000` bacterial assemblies (`E. coli`, *Klebsiella*, *Salmonella*, *Staphylococcus*) vs `~250` protozoan species and `~700` mammalian species.
- If we sample uniformly across genomes with a small `nt_budget` (`25 kb`), a `3 Gb` mammal gets `50` random 500 bp windows — **49 of which are intergenic LINE/Alu repeats and `<1` hits a coding exon**, making eukaryotic gene/domain/frame heads blind on metazoans and plants! Conversely, if we give every genome `5 Mb`, 40,000 bacteria would produce `400M` redundant prokaryotic windows.

### 2.2 Clade-Stratified Taxonomic Selection & Content-Stratified Window Budget
Extend `scripts/fetch_refseq_corpus.py` (or drive via a declarative config `--tol-profile configs/tol_v1_corpus.json`) so **all 9 NCBI groups + organelles + dark-matter MAGs/viruses** are sampled with biologically matched rank granularity and **feature-stratified window budgets**:

| Group / Tier | Taxonomic Sampling Rule | Sources & Assembly Levels | Per-Assembly `nt_budget` (500 bp windows) | Content Stratification (`--region-quota`) | Expected Assemblies | Expected Windows |
|---|---|---|---|---|---|---|
| **`bacteria`** | **`--one-per-rank species --per-rank-n 2`** (species-resolved names only — see §2.2.1) **+ genus floor `3/genus`** (top-up with extra strains/species, incl. species-unresolved assemblies, for any genus under 3 picks) + GenBank MAG gap-fill (`1/family` for GTDB-only families, `1/phylum` for GTDB-only phyla) | RefSeq + GenBank (`--gtdb --include-mags`), Complete..Scaffold | `30,000 bp` (60 windows) | `30%` boundary, `70%` uniform | ~52,000 | ~3,100k |
| **`archaea`** | `--one-per-rank species --per-rank-n 2` + GenBank GTDB MAG gap-fill (`2/genus` for Asgard/DPANN/TACK) | RefSeq + GenBank (`--gtdb --include-mags`), Complete..Contig | `45,000 bp` (90 windows) | `30%` boundary, `70%` uniform | ~2,200 | ~198k |
| **`viral`** | `--one-per-rank species --per-rank-n 2` (all ICTV-resolved RefSeq + GenBank phages/giant viruses/RNA viruses; segment-aware so segmented viruses sample across segments) | RefSeq + GenBank, Complete..Scaffold | `min(2 * genome_len, 25,000 bp)` (tiling short RNA viruses completely without 50× exact duplicates!) | `30%` boundary, `70%` uniform | ~36,000 | ~900k |
| **`protozoa`** | `--one-per-rank species --per-rank-n 2` + VEuPathDB release-68/71 GTF/bigBed tier | RefSeq + GenBank (`--veupath`), Complete..Contig | `100,000 bp` (200 windows) | **Euk quota** (see below) | ~650 | ~130k |
| **`fungi`** | **`--one-per-rank species --per-rank-n 2`** (species-resolved names only) **+ genus floor `2/genus`** | RefSeq + GenBank, Complete..Contig | `75,000 bp` (150 windows) | **Euk quota** | ~8,700 | ~1,300k |
| **`plant`** (*Viridiplantae* + *Rhodophyta* + *Glaucophyta*) | `--one-per-rank genus --per-rank-n 2` (all RefSeq Annotated species + GenBank order/family gap-fill for bryophytes, ferns, gymnosperms, chlorophytes, red algae) | RefSeq + GenBank, Complete..Scaffold | `200,000 bp` (400 windows) | **Euk quota** | ~1,400 | ~560k |
| **`invertebrate`** (arthropods, nematodes, molluscs, helminths, cnidarians, echinoderms, sponges, tunicates) | `--one-per-rank genus --per-rank-n 2` (ensure `2/species` on vector/parasite clades: *Culicidae*, *Ixodida*, *Schistosoma*, *Onchocerca*) | RefSeq + GenBank, Complete..Scaffold | `200,000 bp` (400 windows) | **Euk quota** | ~2,200 | ~880k |
| **`vertebrate_other`** (actinopterygii, chondrichthyes, amphibia, lepidosauria, aves) | `--one-per-rank genus --per-rank-n 1` (`2` on model/agricultural/reservoir genera) | RefSeq (GFF-bearing) + GenBank family gap-fill | `200,000 bp` (400 windows) | **Euk quota** | ~1,500 | ~600k |
| **`vertebrate_mammalian`** | `--one-per-rank genus --per-rank-n 2` + **explicit host anchor** (`Homo sapiens` GRCh38 + T2T-CHM13, *Mus musculus*, *Rattus norvegicus*, *Sus scrofa*, *Bos taurus*, *Canis*, *chiroptera* reservoir bats at `500,000 bp` each so host depletion is rock-solid across chromosomes + repeats) | RefSeq (GFF-bearing) + GenBank family gap-fill | `225,000 bp` (450 windows; `1,000` windows on primary hosts) | **Euk quota** | ~900 | ~415k |
| **Organelle & Plasmid Explicit Tier** | Dedicated organelle pass (`mitochondrion`, `chloroplast`, `apicoplast`, `kinetoplast`, bacterial megaplasmids) extracted from selected assemblies **plus** NCBI RefSeq Organelle collection (`mitochondrion.*.genomic.fna.gz`, `plastid.*.genomic.fna.gz`, 1 per genus) so mitochondrial/plastid contigs are never starved by `round(acc_budget * len(chrM) / 3e9) == 0`! | RefSeq assembly organelle seqids + RefSeq organelle FTP | `15,000 bp` per organelle replicon (up to `2 * len(replicon)`) | `40%` boundary, `60%` uniform | ~8,000 replicons | ~220k |

**Total Corpus Scale (`omniphyl-tol-v1`):**
- **~105,000 genomes + ~8,000 organelle replicons** spanning all 9 NCBI groups, **~90,000+ distinct species** (species-depth `≤2/sp` on bacteria/fungi/viral — matching the A5 selection rule — plus genus-top-up floors), ~14,000+ genera, all GTDB bacterial/archaeal phyla, all ICTV viral realms, and all eukaryotic supergroups.
- **~8.3M – 8.5M 500 bp fragments** (`~4.2 GB` packed `.2bit` / `~9 GB` total `.packed/` `mmap` store — still fits **~13× over** inside a single Vista node's `116 GB` Grace RAM!).

### 2.2.1 Species-Rank Primary Selection with Genus Floor (decided 2026-11-15)

The corpus is selected in **two passes per group** (reuses the existing `covered`/`covered_n` deficit mechanism; requires a new `--drop-unresolved` flag on pass 1):
1. **Species pass:** `one_per_rank=species`, `per_rank_n=2`, keeping only assemblies whose `species` label resolves to a real name (NCBI rankedlineage, GTDB `s__` cluster, or ICTV species for viruses). Assemblies with placeholder/unresolved species values (`uncultured …`, `… sp.`, missing taxid) are **skipped entirely in this pass** — they do not consume species slots.
2. **Genus floor pass:** `one_per_rank=genus`, `per_rank_n=floor` (bacteria `3`, fungi `2`, viral `0`), `covered=` species-pass counts: only genera under the floor receive top-up picks, which MAY be species-unresolved (they still train genus/family/phylum heads; their masked species label contributes `-1` to the species head).

Rationale: species-rank depth is the product surface (`omniphyl profile` species calls, clinical strain resolution); the genus floor preserves the strain-generalization `val_within_species` instrument and clade breadth where species are scarce. Species-unresolved MAGs/SAGs enter via the floor and the GTDB gap-fill tiers only.
- **Backbone group-balance guard (`--group-sample-temp 0.7`):** Moving bacteria/fungi/viral to `species × 2` puts ~63% of all windows in `bacteria + fungi + viral` vs `1.5%` in `protozoa` and `2.3%` in `archaea`. While `flat_prior_tau = 1.0` corrects class-frequency skew in the taxonomy heads, use mild group-temperature sampling ($p_g \propto N_g^{0.7}$) in the Stage-1 sampler so underrepresented eukaryotic/archaeal clades still receive adequate shared-trunk and `struct_head` gradient share.

### 2.2.2 Dark-Matter & Uncultured-Organism Coverage Requirements (verified against `fetch_refseq_corpus.py`)

Lab-inaccessible diversity is primarily a **Stage-1 corpus problem** (windows need only taxonomy + sequence — MAGs/SAGs have no GFF and still emit fragments with `cod=NA` coding labels, contributing to taxonomy heads via the `-1` sentinel mechanism). The refiner panel stays lab-accessible by necessity (it needs trustworthy GFF3 supervision).

Existing mechanisms confirmed in `fetch_refseq_corpus.py` that this plan builds on:
- `--include-mags` + `_excl_ok`: GenBank MAGs/SAGs/`derived from environmental source`/`uncultured` assemblies pass while hard flags (`contaminated`, `chimeric`, `low quality`) stay blocked for **RefSeq** rows.
- `--gtdb` + `attach_gtdb_lineage` + `_gtdb_lin`: GTDB lineages drive rank bucketing in `select_assemblies` AND are preferred over NCBI in label writing (`tax = row.get("_gtdb_lin") or tax_map[...]`).
- `--genbank-groups` + `covered`/`covered_n` deficit-aware top-up + `mark_unresolved=False`: GenBank-only picks are train-only (`seq_source=genbank`, never flagged `_rank_unresolved` test-only).

Required upgrades (new work):
- **GTDB QC gate for GenBank rows:** `excluded_from_refseq` is absent on GenBank rows so `_excl_ok` passes chimeras! Load `bac120_metadata`/`ar53_metadata` (GTDB release-pinned) alongside the taxonomy file and require `checkm_completeness >= 80` and `checkm_contamination <= 10` (or the GTDB `gtdb_genome_representative`/quality tier flag) for MAG/SAG picks.
- **GTDB↔NCBI name normalization:** because `tax = _gtdb_lin or tax_map` selects GTDB names per-accession, a RefSeq assembly WITH a GTDB hit gets GTDB phylum names (`Proteobacteria`) while one without gets NCBI names (`Pseudomonadota`) — silently forking the phylum vocab on the same biological phylum! Fix: normalize output labels to a single convention — prefer NCBI `resolve_taxonomy` whenever `species_taxid` resolves, using `_gtdb_lin` only to fill ranks NCBI leaves `None` (and record `tax_source={ncbi,gtdb,mixed}` in `labels.tsv`). GTDB is the sole label source only for taxids absent from NCBI rankedlineage entirely.
- **Uncultured placeholder names:** NCBI species labels like `uncultured bacterium` / `uncultured archaeon` are shared placeholders across thousands of unrelated MAGs — never emit them as `species`/`genus` labels. Mask any label matching `^(uncultured|unidentified|Candidatus\s*$|metagenome)` to `-1`/blank so the species head never learns a fake class; GTDB cluster names (`s__GCA_xxxxx` style) are legitimate and used as-is.
- **GTDB-only phylum floor (not just family):** `1/family` gap-fill misses GTDB-only *phyla* whose families are partially covered by RefSeq. Add a floor: ≥1 GenBank pick per GTDB-only phylum (bacteria: Patescibacteria/CPR lineages, archaea: all DPANN/Asgard phyla) even when family coverage exists.
- **Uncultured viruses:** GTDB covers only bacteria/archaea (`load_gtdb_taxonomy` hardcodes `ar53`/`bac120`) so viral rank bucketing has no fallback. Add an ICTV VMR (Virus Metadata Resource) lineage fallback for viral rows; drop assemblies with unresolvable `species_taxid` AND no ICTV rank (they would emit all-`-1` taxonomy windows = dead weight).
- **Eukaryotic SAGs:** allowed by `_excl_ok` already (`derived from single cell`) — add a small explicit GenBank SAG gap-fill tier (`1/family`, cap ~100 assemblies) for uncultured protist/microeukaryote lineages; these contribute taxonomy-only windows.
- **GenBank eukaryotes without GFF:** `require_gff=False` already emits `cod=NA` taxonomy-only windows; route all GenBank-only euk gap-fill picks through this path explicitly (do NOT require GFF) and record `label_source=genbank_orf` when ORF/Pfam pseudo-labels are derived.
- **Dual-convention labels (`tax_source` + `gtdb_*` columns):** NCBI rankedlineage remains the **primary label convention** emitted in the 7 `RANKS` columns (it is what downstream tools — Kraken2/MetaPhlAn baselines, query taxid resolution, clinical reporting — speak). GTDB is used for (a) diversity-normalized *selection* bucketing (`_gtdb_lin`), (b) label fallback for taxids absent from NCBI entirely, and (c) emitted verbatim as extra `gtdb_phylum`/`gtdb_family`/`gtdb_genus`/`gtdb_species` provenance columns in `labels.tsv` so eval can reconcile GTDB-only and NCBI-renamed (e.g. `Proteobacteria`→`Pseudomonadota`) mappings. GTDB is never the primary `species` head label: its `s__` names are genome-accession-cluster IDs (`s__Escherichia_coli_A`), not names users or comparators speak.
- **Release pinning:** record GTDB release (e.g. R226), taxdump retrieval date, Pfam release, and VEuPathDB release in `tol_v1_selection.tsv`/`manifest.json` (provenance requirement).

### 2.3 Crucial Fix in `slice_assembly` for Large Eukaryotes (`--euk-region-quota` & Organelle Floor)
Today `slice_assembly` (`scripts/fetch_refseq_corpus.py:948-986`) has two fatal flaws on large genomes:
1. `seq_budget = round(acc_budget * len(seq) / total_len)`: on a `3.1 Gb` human assembly with a `16.5 kb` `NC_012920.1` mitochondrion, `225000 * 16569 / 3.1e9 = 1.2 bp` → **zero mitochondrial windows are ever sampled!**
   - **Fix:** Give non-nuclear replicons (`mitochondrion`, `chloroplast`, `plastid`, `apicoplast`) a dedicated floor `min(len(seq) * 2, args.organelle_nt_budget)` *outside* the nuclear chromosome length-proportional pool.
2. Uniform `rng.randint(0, L - flen)` on a `3.1 Gb` genome with `1.5%` CDS lands in intergenic/intronic DNA `98.5%` of the time; even drawing `4×` candidates yields almost zero CDS or splice-junction windows!
   - **Fix (`slice_euk_stratified`):** When an assembly has GFF/VEuPathDB annotations and raw genome CDS density `< 0.25` (plants, invertebrates, vertebrates, repeat-rich protists), draw windows from **four explicit genomic strata** derived from `feats`:
     - **35% Coding-interior / Multi-exon CDS windows** (centered inside annotated CDS exons ±50 bp) → guarantees rich eukaryotic AA, active-frame, and Pfam domain supervision!
     - **25% Structural-boundary windows** (centered within ±200 bp of start, stop, splice donor, or splice acceptor sites) → supervises `boundary`, `offset`, `struct_flags`, and exon-intron transitions!
     - **20% Intronic + ncRNA windows** (`rRNA`, `tRNA`, `lncRNA`, short & long intron interiors) → teaches Stage 1 that intronic/RNA sequence belongs to the host/eukaryotic taxon *without* firing coding heads!
     - **20% True intergenic + interspersed repeat windows** (sampled uniformly from non-genic genomic background, including Alu/LINE/LTR/centromeric repeats) → ensures honest taxonomic classification and calibrated repeat uncertainty on real shotgun metagenomic reads!
     - **TE-ORF caveat:** vertebrate/plant intergenic space contains thousands of intact transposon ORFs (LINE-1 ORF1p/ORF2p, ERV pol, LTR gag) that look genuinely 'coding' and hit Pfam RT/integrase domains — they would poison `coding` supervision. Since the Pfam scan already runs, tag windows whose ORFs hit only TE-associated Pfam clans (RT/integrase/transposase/endonuclease/gag/env) as `sample_stratum=te_orf` and rewrite `coding_frac = intron_frac = active_frame = struct_flags = "NA"` in `labels.tsv` (so `encode_cont_labels`/`encode_annot_labels` automatically map them to the `-1.0` unlabeled sentinel, keeping taxonomy + Pfam supervision while excluding them from coding-negative supervision).
   - **Record provenance:** Add a `sample_stratum` column (`uniform`, `cds`, `boundary`, `intron_rna`, `intergenic`, `organelle`) to `labels.tsv` and `manifest.json` so evaluation can report **both** unweighted validation metrics and **natural-genomic-prior importance-weighted** shotgun read metrics (preventing stratified training from distorting shotgun false-positive evaluation!).

### 2.4 Viral Deduplication & Short-Genome Tiling Fix
In `slice_fragments` (`fetch_refseq_corpus.py:725-747`), a `3,500 bp` RNA virus with `nt_budget = 25,000` draws `50` random 500 bp windows covering the same `3.5 kb` **7× redundantly**, leaking near-identical windows if `--frag-val` is active and over-weighting tiny ssRNA/ssDNA viruses per base.
- **Fix:** Cap `effective_budget = min(acc_budget, 2 * total_len)` for genomes `< 25 kb`, and disable `--frag-val` within-genome window holdout on assemblies `< 50 kb` (small viruses are evaluated strictly via whole-genome / whole-genus holdouts!).

### 2.5 Holdout & Benchmark Split Hygiene (Zero Leakage)
`build_split` (`train_omniphyl.py`) must enforce **four disjoint validation tiers** saved explicitly in `<run>/splits.json`:
1. **`val_within_species` (`frag_val = 0.03` on genomes $\ge 50\text{ kb}$ + dedicated `val_strain_frac = 0.025` draw from `multi_sp` 2nd accessions):** Measures strain/isolate generalization within trained species.
2. **`val_novel_species` (dedicated `val_species_frac = 0.025` draw strictly from `single_sp_multi_g` whole-species holdouts in multi-species genera):** Measures genus-level placement + `p_ood` species abstention. *(Bug fix in `build_split`: currently `train_omniphyl.py:458` iterates `multi_sp + single_sp_multi_g` under one combined `n_val` cap; with ~11.7k multi-assembly species in `species × 2`, `multi_sp` would fill 100% of `n_val` and leave `val_novel_species` empty unless the two quotas are drawn separately!)*
3. **`val_novel_genus` (`holdout_genus_frac = 0.03` stratified across ALL 9 groups, not pooled globally so `bacteria` doesn't monopolize the draw):** Measures family/order placement + genus/species OOD abstention across every kingdom. *(Bug fix for GenBank-supplemented genera: instead of excluding all `gb_genera` (`train_omniphyl.py:484`), allow mixed RefSeq+GenBank genera to be held out by dropping their GenBank accessions from `train` while evaluating metrics strictly on their RefSeq accessions.)*
4. **Hard Exclusion List (`configs/benchmark_exclusion_accessions.txt`):** Every genome accession in `scripts/clade_bench.py`, the held-out Stage-2 Refiner Evaluation Panel (§3.2), CAMI II marine/plant-rhizosphere/mouse-gut test sets, and clinical/host benchmark sets **must be passed via `--exclude-accessions`** at corpus fetch time so zero windows of any downstream benchmark assembly ever enter Stage-1 training. (Note: `exclude_accs` exists inside `select_assemblies` but is currently only fed by `--base-corpus` — a new `--exclude-accessions <file>` CLI flag is required.)
   - **Human chromosome-level holdout:** because *H. sapiens* is a training anchor AND contributes held-out refiner eval chromosomes (chr21+chr22), exclude `NC_000021.9` and `NC_000022.11` seqids from all human stratified slicing so refiner eval chromosomes are unseen by Stage-1.

---

## 3. Part B — Full Tree-of-Life Stage-2 Refiner Corpus (`omniphyl-refiner-tol-v1`)

### 3.1 Why the 8-Genome v5 Refiner Corpus Was Insufficient
The v5 refiner trained on 8 small genomes (~120 Mb total, zero plants, zero vertebrates, zero arthropods/nematodes/helminths, zero organelles, zero alternative genetic codes), causing FiLM to memorize per-genome confidence signatures (e.g. *P. vivax* collapsing onto *L. major*).

### 3.2 Curated 72-Genome Stratification Panel (60 Train / 16 Strictly Held-Out Eval Units)
Create `configs/refiner_tol_panel.json` + `scripts/fetch_refiner_panel.py` fetching high-quality **RefSeq Complete/Chromosome + MANE/VEuPathDB** curated `(FASTA, GFF3)` pairs with **exact pinned accessions** (e.g., GRCh38 = `GCF_000001405.40`, GRCm39 = `GCF_000001635.27`, *S. cerevisiae* S288C = `GCF_000146045.2`), documented release version, genetic code, and length-capped chromosome/contig slices for gigabase genomes:

| Clade Stratum | Train Genomes (`60` total) | Held-Out Benchmark Genomes (`16` eval units, **never in Stage-1 or Stage-2 train**) | Slice / Budget Rule (`--max-bp-per-genome 25_000_000`) |
|---|---|---|---|
| **Bacteria (diverse GC, operons, GTG/TTG starts, Code 4 & 25)** | 10 species (*E. coli* K-12, *P. aeruginosa*, *M. tuberculosis* [high GC], *S. aureus* [low GC], *Bacteroides fragilis*, *Cyanobacterium* *Synechocystis*, *Streptomyces coelicolor*, *Borrelia*, *Mycoplasma genitalium* [Code 4], *Candidatus* Gracilibacteria [Code 25]) | *Bacillus subtilis* 168, *Mycoplasma pneumoniae* (Code 4) | Full genome (`0.6–8 Mb`) |
| **Archaea (Euryarchaeota, Crenarchaeota, Asgard)** | 5 species (*Halobacterium salinarum*, *Sulfolobus acidocaldarius*, *Pyrococcus furiosus*, *Methanosarcina*, *Promethearchaeum*) | *Methanocaldococcus jannaschii* | Full genome (`1.6–4 Mb`) |
| **Viruses & Phages (compact dual-strand, polyproteins, giant viruses)** | 6 genomes (T4 phage, Lambda, SARS-CoV-2, EBV/HSV-1, Mimivirus, CrAssphage) | Bacteriophage T7 + Mpox/Vaccinia | Full genome |
| **Protists / VEuPathDB (intronless kinetoplastids, intron-dense apicomplexans, ciliates Code 6, amoebae, diatoms, oomycetes, diplomonads)** | 10 species (*P. falciparum* 3D7, *Toxoplasma gondii* ME49, *Cryptosporidium parvum*, *Leishmania major*, *Trypanosoma cruzi*, *Tetrahymena thermophila* [Code 6], *Entamoeba histolytica*, *Phaeodactylum tricornutum*, *Phytophthora infestans*, *Giardia duodenalis* [diplomonad, mitosome, ~zero cis-spliced introns]) | *Plasmodium vivax* P01 (VEuPathDB), *Trypanosoma brucei*, *Dictyostelium discoideum* | Full genome or gene-richest chromosomes up to `25 Mb` |
| **Fungi (compact budding yeasts, intron-rich basidiomycetes, filamentous ascomycetes, *Candida* Code 12, microsporidia, mucoromycetes, chytrids)** | 8 species (*S. pombe*, *Neurospora crassa*, *Aspergillus nidulans*, *Cryptococcus neoformans* [intron-rich basidio!], *Candida albicans* [Code 12], *Encephalitozoon intestinalis* [microsporidian], *Mucor circinelloides* [Mucoromycota], *Batrachochytrium dendrobatidis* [Chytridiomycota, amphibian pathogen]) | *Saccharomyces cerevisiae* S288C, *Coprinopsis cinerea* (intron-rich basidiomycete) | Up to `25 Mb` |
| **Plants (eudicots, monocots, bryophytes, prasinophytes — GC gradients + AT-rich introns)** | 7 species (*Arabidopsis thaliana*, *Medicago truncatula*, *Solanum lycopersicum*, *Brachypodium distachyon*, *Zea mays* [repeat-rich!], *Marchantia polymorpha*, *Ostreococcus tauri*) | *Oryza sativa* Japonica (monocot) | Stratified chromosome slices totaling `25 Mb`/species (preserving intact 200 kb gene-containing windows + flanking intergenic repeats!) |
| **Invertebrates (nematodes, dipterans, hymenopterans, molluscs, platyhelminths, cnidarians)** | 7 species (*C. elegans*, *D. melanogaster*, *Apis mellifera*, *Crassostrea gigas*, *Schistosoma mansoni*, *Nematostella vectensis*, *Ixodes scapularis*) | *Anopheles gambiae* PEST | Stratified `25 Mb`/species |
| **Vertebrates (teleosts, amphibians, reptiles/birds, marsupials, placental mammals)** | 7 species (*Danio rerio*, *Xenopus tropicalis*, *Anolis carolinensis*, *Monodelphis domestica*, *Mus musculus* GRCm39, *Canis lupus familiaris*, *Homo sapiens* GRCh38 MANE Select **chr1–chr20 subset**) | *Gallus gallus* (bird), *Homo sapiens* **chr21 + chr22** (strictly held-out chromosomes for human + whole-species bird holdout!) | Stratified `30 Mb`/species (intact `250 kb` gene+flank blocks so `50 kb` introns are never severed!) |
| **Organelles (embedded in every euk assembly above)** | Mitochondrial + plastid + apicoplast replicons (`Code 2, 4, 5, 11, 16`) | Human `chrM` (`NC_012920.1`, Code 2) + *O. sativa* chloroplast (`Code 11`) | Full replicon |

**Total Stage-2 Refiner Corpus Size:**
- `60` diverse genomes × `~15 Mb` average = **`~900 Mb` of base-exact annotated sequence** ≈ `~37,000` `32,768 bp` chunk starts at `24,576 bp` stride × 2 (RC augmentation) = **`~75,000` training chunks** (per-epoch cost roughly 2× the v5 refiner corpus, acceptable on GH200).
- The held-out column enumerates **16 eval units across 15 taxa** (*H. sapiens* contributes both chr21+22 and chrM as separate replicons); all are strictly excluded from Stage-1 corpus (§2.5) and refiner training.
- With `--balance-genomes`, each of the 60 genomes contributes equal expected gradient steps per epoch (`1/60` weight), preventing `30 Mb` vertebrate slices from drowning out `1.8 Mb` *Mycoplasma* or `16 kb` mitochondrial replicons.
- Optionally attach **real LexicMap RNA-seq / transcript split-alignment tracks** on 10–15 of the training eukaryotes (and keep `p_no_ev = 0.35` evidence dropout) so the refiner learns **both** pure *ab initio* decoding and evidence-guided splicing!

---

## 4. Part C — End-to-End TACC Vista Pipeline & SLURM Orchestration (`slurm/vista/`)

Create the directory `slurm/vista/` with modular, idempotent scripts chained via `sbatch --dependency=afterok:<jobid>`.

### 4.1 Environment Bootstrap (`slurm/vista/00_setup_env.sh`)
Run inside a short `gg` job (`sbatch slurm/vista/00_setup_env.sbatch`, 1 node, 30 min — **not** on login node):
1. Load Vista base toolchain: `module load gcc cuda python3` (verify `uname -m` == `aarch64` and `python3 --version` $\ge 3.11$).
2. Create persistent conda/micromamba or venv on `$WORK/envs/omniphyl-arm64` (using `micromamba` `linux-aarch64` or Vista's module Python + NVIDIA ARM64 CUDA PyTorch wheel index `https://download.pytorch.org/whl/cu124` or system PyTorch module if `torch.cuda.is_available()` is already compiled for `sm_90` Hopper — test `python3 -c "import torch; assert torch.cuda.get_device_capability()[0] >= 9"` on a 5-min `gh-dev` sanity job!).
3. Install `numpy`, `pyhmmer` (`aarch64` wheel/source build with GCC 14 NEON/SVE support — verify `pyhmmer` imports and runs a 10-ORF smoke scan!), and `pip install -e $HOME/omniphyl` (or `$WORK/omniphyl`).
4. Create directory layout:
   - `$SCRATCH/omniphyl-tol/cache` (NCBI/GenBank/UCSC/GTDB/Pfam raw downloads)
   - `$SCRATCH/omniphyl-tol/corpus_v1` (`fragments.fna`, `labels.tsv`, `corpus.packed/`)
   - `$SCRATCH/omniphyl-tol/refiner_v1` (72-genome panel + Stage-1 `.res.pkl` caches)
   - `$SCRATCH/omniphyl-tol/runs` (active training working dirs & SLURM logs)
   - `$WORK/omniphyl-tol/artifacts` (curated manifests, `splits.json`, `best.pt`, bundled checkpoints, eval reports).

### 4.2 Stage 1A — Metadata Selection & Network Prefetch (`slurm/vista/01_prefetch_corpus.sh`)
**Why split network download from CPU slicing:** NCBI/EBI/UCSC throttle >8–12 concurrent HTTPS streams per IP, whereas 144 Grace CPU cores want massive local parallelism. Also handles any compute-node firewall edge cases cleanly.
1. Run `fetch_refseq_corpus.py --select-only --tol-profile configs/tol_v1_corpus.json --exclude-accessions configs/benchmark_exclusion_accessions.txt --out-plan $WORK/omniphyl-tol/artifacts/tol_v1_selection.tsv`:
   - Deterministically selects the ~105,000 assemblies + ~8,000 organelle replicons, resolves NCBI + GTDB + ICTV lineages, verifies GFF URLs, and writes a frozen **Selection Ledger** (`tol_v1_selection.tsv` with SHA256/ETag/URLs, taxid, group, rank, `nt_budget`, `stratum_mode`).
   - Run `fetch_refiner_panel.py --panel configs/refiner_tol_panel.json --out-plan $WORK/omniphyl-tol/artifacts/refiner_v1_selection.tsv`.
2. Download all selected `.fna.gz`, `.gff.gz`, VEuPathDB tracks, `Pfam-A.hmm.gz`, `UniVec_Core`, `phiX174`, `GRCh38` repeat sketch, and the 10 pinned `100k`-read SRA evaluation subsets (§5.1.4, so `05_eval_release.sbatch` runs 100% offline on compute nodes) into prefix-sharded subdirectories `$SCRATCH/omniphyl-tol/cache/raw_genomes/${acc:0:7}/` (avoiding 200k-file single-directory metadata contention) with bounded concurrency (`--dl-workers 10`, atomic `.part` rename, retry/backoff, resumable if interrupted).
   - Disk check: ~98k compressed microbial/viral `.fna.gz` (~1.2 MB avg = ~115 GB) + ~7,000 eukaryotic `.fna.gz`/`.gff.gz` (~1.55 TB compressed) = **~1.7 TB on `$SCRATCH`** (trivial on Vista's multi-PB VAST `$SCRATCH`, and raw `.fna.gz` cache can be deleted once `corpus.packed/` passes audit).

### 4.3 Stage 1B — Massively Parallel Offline Slicing, Pfam HMM Scan, Contaminant Screen & Binary Packing (`slurm/vista/02_build_corpus_gg.sbatch`)
**Queue:** `#SBATCH -p gg -N 3 -n 3 -t 06:00:00` (3 Grace-Grace CPU nodes = **432 ARM64 cores**, expected ~3.5–4h = **`~4.0 SUs`** at `0.33 SU/node-hr`!).
1. **Sharded Slicing (Node-parallel across 432 cores):**
   - Shard `tol_v1_selection.tsv` into `N_SHARDS = 24` balanced by uncompressed genome bytes; run `fetch_refseq_corpus.py --from-plan ... --shard $i/24 --offline-cache $SCRATCH/omniphyl-tol/cache --workers 18` across the 3 `gg` nodes via `ibrun` / task wrapper.
   - Emits sharded `fragments.part_XX.fna` + `labels.part_XX.tsv` with exact `sample_stratum`, `replicon`, `transl_table`, `struct_flags`, `active_frame`, `intron_frac`, `rna_frac`.
2. **Parallel `pyhmmer` Pfam Annotation (432 Grace cores):**
   - Run `annotate_fragments_pfam` per shard (`cpus=36` per process × 4 processes/node) over extracted ORFs translated with each row's `transl_table`. On 432 Neoverse V2 cores, ~8.4M fragments scan in **~55–85 minutes**.
3. **Merge + Clade-Aware Contaminant Screen (`screen_contaminants.py`):**
   - Concatenate shards deterministically, run clade-aware `UniVec` + `phiX174` + non-chordate `GRCh38` contamination screen, emit `fragments_screened.fna`, `labels_screened.tsv`, `fragments_rejected.tsv`, and `corpus_qc_report.json`.
4. **Corpus QC Hard Gate (`scripts/audit_corpus.py`):**
   - Job fails fast before GPU spend if any invariant fails:
     - Zero benchmark-excluded accessions (or human `chr21`/`chr22` seqids) present.
     - All 9 NCBI groups + organelles present with $\ge 90\%$ of target assembly count, and zero `uncultured`/`unidentified` placeholder tokens in `species`/`genus` vocab.
     - Eukaryotic (`plant`, `invertebrate`, `vertebrate_*`) coding window share between `30%` and `55%` (verifying `--euk-region-quota` worked and isn't 98% repeat desert!).
     - Domain vocabulary size (`>= 20` train windows) $\ge$ `domain_head_k` (`512`).
5. **Pack to Binary `mmap` Store (`scripts/pack_corpus.py`):**
   - Produce `$SCRATCH/omniphyl-tol/corpus_v1/corpus.packed/` (`~9 GB` total) and copy `manifest.json`, `corpus_qc_report.json`, `splits.json`, and `vocab.json` to `$WORK/omniphyl-tol/artifacts/`.

### 4.4 Stage 2 — Distributed Stage-1 `OmniPhyl-ToL` Training (`slurm/vista/03_train_stage1_gh.sbatch`)
**Queue & Throughput Sizing on GH200:**
- `#SBATCH -p gh -N 4 -n 4 -t 16:00:00` (4 Grace-Hopper nodes = **4 × H200 96 GB GPUs** + **288 Grace CPU cores**, `64 SUs` max, actual expected wall time **~5–6 hours** = `~22 SUs`!).
- **Why 4 GH200 nodes is so fast:**
  - Single T4 (`16 GB`, FP16, PCIe, 2 vCPUs) took ~30 min/epoch on 1M windows (`~550 frags/s`).
  - A single **H200 (1,979 TFLOPS BF16, 4 TB/s HBM3, 72 Grace cores feeding `mmap` FCGR+AA over 900 GB/s NVLink-C2C)** achieves **~8,000–12,000 windows/sec/GPU** at `width=96`, `batch_size=1024` per GPU (`global_batch=4096`).
  - Across **4 × GH200**, `~8.4M` windows = `~2,050` steps/epoch $\approx$ **`4.5–6.5 minutes/epoch`**!
  - **30 full epochs** (vs v6's budget-starved 4 epochs on Kaggle!) completes in **~3.5–5.5 hours**, leaving ample walltime inside the 16h allocation for validation and calibration sweeps.
- **Launch pattern on Vista (`docs.tacc.utexas.edu/hpc/vista/ml`):**
  ```bash
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  MASTER_PORT=29500
  export OMP_NUM_THREADS=4
  ibrun -np "$SLURM_NNODES" bash -c '
    NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-$SLURM_PROCID}}
    torchrun --nnodes='"$SLURM_NNODES"' --nproc_per_node=1 \
      --node_rank="$NODE_RANK" \
      --master_addr='"$MASTER_ADDR"' --master_port='"$MASTER_PORT"' \
      scripts/train_omniphyl.py \
        --packed "$SCRATCH/omniphyl-tol/corpus_v1/corpus.packed" \
        --arch-config docs/tol-arch-locked.json \
        --out "$SCRATCH/omniphyl-tol/runs/stage1_tol_v1" \
        --epochs 30 --batch-size 1024 --lr 6e-4 --wd 1e-2 \
        --group-sample-temp 0.7 \
        --amp bf16 --num-workers 16 \
        --resume-auto
  '
  ```
  *(Note: Verify `PMI_RANK` / `SLURM_PROCID` on a 5-min 2-node `gh-dev` smoke run `03a_smoke_ddp_ghdev.sbatch` before submitting the 30-epoch `gh` job! Also scale `--lr` linearly/sqrt from `3e-4` @ 256 to `6e-4`–`8e-4` @ global batch 4096 with 2-epoch linear warmup + cosine decay.)*
- Copy `best.pt`, `last.pt`, `vocab.json`, and training curves to `$WORK/omniphyl-tol/artifacts/stage1_tol_v1/` immediately upon completion.

### 4.5 Stage 3 — Stage-1 Feature Caching & Stage-2 `OmniPhylRefiner-ToL` Training (`slurm/vista/04_train_refiner_gh.sbatch`)
**Queue:** `#SBATCH -p gh -N 2 -n 2 -t 12:00:00` (`2 × GH200`, `24 SUs` max, actual expected **~3–5 hours** = `~8 SUs`):
1. **Step 3A — Parallel Stage-1 `.res.pkl` Cache Generation (`~900 Mb` + `~250 Mb` held-out panel = `~1.15 Gb` total = `~11.5M` strided 500 bp windows at stride 100):**
   - Shard the 72 genomes across the 2 GH200 GPUs (36 genomes per H200); at `~10,000 windows/sec/H200`, all 72 genomes finish Stage-1 inference + `.res.pkl` serialization into `$SCRATCH/omniphyl-tol/refiner_v1/stage1_cache/` in **~10–15 minutes total**!
2. **Step 3B — Multi-GPU Refiner v2 Training (`train_refiner.py`):**
   - Launch 2-node DDP `train_refiner.py` with `--arch-config docs/tol-arch-locked.json`, `--balance-genomes`, `--amp bf16`, `--epochs 25`, `--batch-size 16` per GPU (`32 × 32,768 bp = 1.05 Mb/step`), `--resume-auto`, `--data` (60 train genomes) and `--val-seqids` (1 held-out chromosome/scaffold per stratum across 12 train genomes for early-stopping / `best.pt` selection — keeping all **16 external benchmark eval units** strictly untouched until Step 4!).
   - `~75,000` chunks / `32` chunks/step = `~2,350` steps/epoch $\approx$ `~3–5 min/epoch` on 2 × H200 $\rightarrow$ **25 epochs in ~1.5–2.5 hours**.
3. **Step 3C — Bundle Unified, Hardware-Portable Checkpoint:**
   - Unwrap DDP (`model.module`), strip any `torch.compile` `_orig_mod.` keys, convert all state_dict tensors to contiguous CPU `float32`, ensure all metadata dicts contain only standard Python/JSON primitives (no `Path` or pickle-bound objects), and bundle `stage1_tol_v1/best.pt` + `refiner_tol_v1/best.pt` into `$WORK/omniphyl-tol/artifacts/omniphyl_tol_v1_bundled.pt` (plus an optional `safetensors` + `metadata.json` export) so `omniphyl annotate` and `omniphyl profile` work out-of-the-box on **any downstream `x86_64` (AMD/Intel), `aarch64` (Apple Silicon / ARM server), NVIDIA CUDA, AMD ROCm, MPS, or CPU-only machine**.

---

## 5. Part D — Comprehensive Tree-of-Life Evaluation & Release Gates (`slurm/vista/05_eval_release.sbatch`)

Run on `gh-dev` or `gh` (1 node, 2h) via `scripts/eval_tol_suite.py`, emitting `$WORK/omniphyl-tol/artifacts/eval_report/` (`metrics.json` + `REPORT.md` with full data provenance):

### 5.1 Taxonomic & Metagenomic Profiling Gates (`eval_pilot.py` + synthetic/CAMI mixtures)
1. **Stratified + Natural-Prior Accuracy & Calibration across all 9 Groups:**
   - Report Top-1 accuracy, Macro-F1, ECE, and Selective Coverage @ `Precision >= 0.90` (`Cov@P90`) separately for each of the 9 groups (`archaea`, `bacteria`, `fungi`, `protozoa`, `plant`, `invertebrate`, `vertebrate_other`, `vertebrate_mammalian`, `viral` + `organelle`) and across all 4 read regimes (`150 bp`, `250 bp`, `PE150`, `500 bp`, `2 kb` contigs).
   - **Importance-weighted shotgun metric:** Re-weight eukaryotic strata by true genomic composition (`cds`, `intron_rna`, `intergenic`) so reported accuracy reflects unenriched shotgun reads.
2. **Novel-Species & Novel-Genus OOD Abstention (`val_novel_species`, `val_novel_genus`):**
   - Over-resolution rate (calling a confident training species on a held-out novel genus at `conf_tau=0.5`) must be **$\le 0.08$**, while family/phylum recall remains **$\ge 0.80$**.
3. **Host-Contaminated Metagenome Benchmark:**
   - Simulate a `90%` human (`GRCh38` held-out regions) + `8%` gut/blood bacteria + `1.5%` *Plasmodium*/*Candida* + `0.5%` viral read mixture:
     - `omniphyl profile --host-filter Chordata`: host read recall $\ge 0.98$, false host removal of microbial/parasite reads $\le 0.005$, L1 abundance error on the non-host community $\le 0.10$.
4. **Real SRA Validation (uncatalogued-diversity instrument):**
   - SRA cannot supply *training* labels (a read set has no reference taxonomy beyond NCBI anyway) but it IS the deployment distribution: run `omniphyl profile` + Kraken2/mOTUs on ~10 real SRA metagenomes (2× human gut, 2× marine, 2× soil, 2× clinical/pathogen, 2× host-illumina isolate sets) and report **abstention rate vs comparator false-positive rate** — the species-depth corpus should map known species confidently while *correctly abstaining* (`p_ood`) on novel/uncatalogued lineages rather than over-calling.

### 5.2 Whole-Genome Gene Structure & Functional Annotation Gates (`clade_bench.py` + 16 Held-Out Eval Units)
Evaluate `omniphyl annotate` (both pure *ab initio* `--no-lexicmap` and `+LexicMap` where applicable) on all **16 strictly held-out eval units** against reference GFF3s (`eval_genes.py` + `eval_errors.py` + `eval_refiner_panel.py`):

| Clade Regime | Held-Out Genomes | Target Nucleotide F1 (*ab initio*) | Target Exon F1 (*ab initio*) | Special Gate |
|---|---|---|---|---|
| **Prokaryotes (Std + Code 4)** | *B. subtilis*, *M. pneumoniae*, *M. jannaschii* | $\ge 0.95$ | $\ge 0.85$ | *M. pneumoniae* auto-selects `gcode = 4` (`TGA` readthrough, zero split genes) |
| **Compact / Intronless & Intron-rich Protists** | *T. brucei*, *D. discoideum*, *P. vivax* P01 | $\ge 0.90$ | $\ge 0.72$ | *P. vivax* intron recall $\ge 0.75$ (confirming zero FiLM suppression regression!) |
| **Fungi (Compact + Intron-rich)** | *S. cerevisiae*, *C. cinerea* | $\ge 0.92$ / $\ge 0.86$ | $\ge 0.82$ / $\ge 0.68$ | Zero false introns injected into single-exon yeast genes ($\le 2\%$) |
| **Plants** | *Oryza sativa* | $\ge 0.86$ | $\ge 0.65$ | GC-gradient 5'→3' monocot exons decoded cleanly |
| **Viruses (compact genomes, programmed frameshifts)** | Phage T7, Vaccinia | $\ge 0.90$ | $\ge 0.75$ | No double-counted dual-strand gene calls above background |
| **Invertebrates & Vertebrates (Long introns)** | *A. gambiae*, *G. gallus*, *H. sapiens* chr21+22 | $\ge 0.84$ / $\ge 0.80$ | $\ge 0.62$ / $\ge 0.55$ (*ab initio*), $\ge 0.78$ (`+LexicMap`) | Intron recall on `5–35 kb` introns $\ge 0.50$; repeat FP gene calls in intergenic LINE/Alu deserts $\le 5\%$ of total predicted bp |
| **Organelles** | Human `chrM` (`Code 2`), Rice chloroplast (`Code 11`) | $\ge 0.92$ | $\ge 0.80$ | Per-contig `res.gcode` = `2` on `chrM` alongside `1` on nuclear `chr21` in the same FASTA! |

**Comparator baselines (external validity):** run Prodigal (prokaryotes) and Helixer or AUGUSTUS (eukaryotes, where installable on `aarch64` — else run on any x86 host since inference is portable) on 2–3 held-out genomes per regime, and Kraken2/mOTUs on the simulated metagenomes, so `REPORT.md` can state OmniPhyl-ToL results relative to established tools.

---

## 6. SU Budget Summary & Execution Checklist for the Agent

### Total Estimated TACC Vista SU Cost
| Stage | Partition | Nodes × Walltime (Expected) | SU Rate | Estimated SUs |
|---|---|---|---|---|
| `00` Env setup + `03a` 2-node DDP smoke | `gg` + `gh-dev` | `0.5h` + `0.25h × 2` | `0.33` / `1.0` | `< 1 SU` |
| `02` Corpus build + `pyhmmer` + screen + pack | `gg` (432 ARM cores) | `3 nodes × 4h` (~105k genomes; NCBI rate-limited download in `01`, CPU slice/Pfam in `02`) | `0.33 SU/hr` | **`~4.0 SUs`** |
| `03` Stage-1 `OmniPhyl-ToL` (30 epochs, ~8.4M windows) | `gh` (4 × GH200) | `4 nodes × 5.5h` | `1.0 SU/hr` | **`~22.0 SUs`** |
| `04` Stage-2 `OmniPhylRefiner-ToL` (cache + 25 epochs) | `gh` (2 × GH200) | `2 nodes × 3.5h` | `1.0 SU/hr` | **`~7.0 SUs`** |
| `05` Full ToL Eval + Clade Bench + Bundling | `gh` (1 × GH200) | `1 node × 1.5h` | `1.0 SU/hr` | **`~1.5 SUs`** |
| **Total End-to-End Production Run** | | | | **`~35 SUs`** *(~70 SUs incl. 1 full tuning/re-run buffer)* |

### Agent Execution Order
1. **Verify Plan 1 lock:** Check `docs/tol-arch-locked.json` + unit tests.
2. **Write configs & corpus scripts:** Create `configs/tol_v1_corpus.json`, `configs/refiner_tol_panel.json`, `configs/benchmark_exclusion_accessions.txt`, `scripts/fetch_refiner_panel.py`, `scripts/audit_corpus.py`, `scripts/eval_tol_suite.py`, and add `--tol-profile` / `--from-plan` / `--shard` / `slice_euk_stratified` + organelle floor to `scripts/fetch_refseq_corpus.py`.
3. **Write Vista SLURM suite:** Create `slurm/vista/{00_setup_env.sh,01_prefetch_corpus.sh,02_build_corpus_gg.sbatch,03a_smoke_ddp_ghdev.sbatch,03_train_stage1_gh.sbatch,04_train_refiner_gh.sbatch,05_eval_release.sbatch}` obeying every Vista rule in §1.2 (parameterizing `#SBATCH -A` via a small `slurm/vista/submit_pipeline.sh` wrapper that checks `qlimits` / `$TACC_PROJECT`).
4. **Execute on Vista step-by-step:** Run `00` → `01` → `02` (verify `audit_corpus.py` gates) → `03a` (verify 2-node NCCL + throughput on `gh-dev`) → `03` → `04` → `05`.
5. **Archive & Document:** Save `omniphyl_tol_v1_bundled.pt`, `tol_v1_selection.tsv`, `refiner_v1_selection.tsv`, `corpus_qc_report.json`, and `eval_report/REPORT.md` to `$WORK/omniphyl-tol/artifacts/`, ensuring every dataset URL, accession, version, and slice coordinate is permanently recorded.
