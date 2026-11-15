# Plan 1: Tree-of-Life Architecture Audit, Upgrades, and Locking Contract (`OmniPhyl-ToL`)

**Status:** Ready for autonomous agent execution
**Prerequisite for:** `docs/tol-corpus-vista-plan.md` (Plan 2 — Full Tree-of-Life Corpus & TACC Vista Training)
**Target codebase:** `d-callan/omniphyl` (`src/omniphyl/`, `scripts/`, `tests/`)

---

## 1. Context & Problem Statement

OmniPhyl's twin-model stack (**Stage 1 `OmniPhyl`** 500 bp window model + **Stage 2 `OmniPhylRefiner`** per-nucleotide FiLM + dilated-ResNet + BiGRU + semi-Markov decoder) was validated on the **5-group microbial/protist regime** (`protozoa`, `fungi`, `archaea`, `bacteria`, `viral`, ~21k genomes / ~1.06M windows in `v5`/`v6` + an 8-genome refiner set).

Moving to **full Tree-of-Life (ToL)** coverage — adding `plant`, `invertebrate`, `vertebrate_other`, `vertebrate_mammalian` (including human host), organelles (mitochondria, plastids, apicoplasts), and 50k–100k+ species — breaks seven concrete assumptions hardcoded into the current codebase:

1. **Vertebrate/metazoan intron horizon & chunk geometry (`src/omniphyl/refiner.py`, `scripts/train_refiner.py`):**
   - `CHUNK = 8192`, `CHUNK_STRIDE = 6144`, `max_intron = 5000`, and `OmniPhylRefiner`'s receptive field (~1.3 kb conv + ~4 kb 4×-pooled BiGRU) cannot span mammalian/arthropod introns (median human intron ~1.5–3.5 kb; ~25–30% of mammalian introns exceed 5 kb, frequently reaching 20–50 kb).
   - In `_phase_loss` (`scripts/train_refiner.py:310-326`), `hi_w = min(d + max_intron + 1, pa.shape[-1])`: any donor whose partner acceptor lies beyond `CHUNK` (`8,192 bp`) has `pa_max` evaluated over an acceptor-free slice, forcing `1.0 - pd[b, d] * 0 = 1.0` — **actively penalizing true donors of long introns**!
   - In `_refiner_loci` (`src/omniphyl/refiner.py:513-534`), active loci require a 100 bp smoothed `p_cds + p_int >= 0.20`; if `p_int` sags inside a 25 kb repeat-rich mammalian intron, a single gene splits into disconnected loci and its exon chain can never be joined.

2. **Host-genome rejection in contaminant screening (`scripts/screen_contaminants.py:39-48`):**
   - `CONTAM_ACC_FRAC = 0.5` only exempts GRCh38 itself when `>50%` of an assembly's fragments hit the human 21-mer sketch. Non-human primates (*Pan*, *Gorilla*, *Macaca*) and conserved mammalian orthologs hit the human 21-mer sketch on 10–45% of coding windows and get **selectively stripped of their conserved genes**, biasing vertebrate representations toward non-coding repeats.

3. **Repeat-induced compositional collapse & clustering explosion (`src/omniphyl/encode.py`, `src/omniphyl/profile.py`):**
   - Vertebrate and plant genomes are 40–85% transposable elements (Alu, LINE-1, LTR retrotransposons) and simple tandem repeats. A 500 bp window inside a young family-shared repeat has high k-mer multiplicity, little species-specific signal, and lacks an explicit repeat/low-complexity signal for Stage 1 or `profile`.
   - In `cluster_reads` (`src/omniphyl/profile.py:165-175`), `cap = max(2, int(max_kmer_freq * n))` with `max_kmer_freq = 0.10` is safe for small microbial metagenomes, but when a sample contains 5% human host reads (`20,000` reads out of `400k`), Alu 21-mers sit *below* `0.10` yet exceed `200` only if abundant enough — between `2` and `200` reads they quadratically chain unrelated host loci into chimeric clusters.

4. **Incomplete taxonomy graphs across Eukaryota & Viruses (`scripts/fetch_refseq_corpus.py`, `src/omniphyl/model.py`):**
   - `RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]` uses NCBI `rankedlineage.dmp` (`RL_COLS`), where plants (`Viridiplantae`) and fungi sit under `kingdom` while `superkingdom` is `Eukaryota`, many protist phyla/classes are rankless clades in NCBI (`Sar`, `Discoba`, `Amoebozoa`), and viruses under ICTV use `realm` rather than `domain` and frequently omit `class`/`phylum`.
   - In `build_tree` (`src/omniphyl/model.py:157-160`), when an intermediate rank is `""` (e.g., a protist with `domain=Eukaryota`, `phylum=""`, `class=""`, `order=Trypanosomatida`), `parent_votes["order"]` votes on `ranks[ri - 1]` (`class == ""`), dumping `po[c + 1] = 0` (root catch-all) instead of linking to the nearest resolved ancestor (`Eukaryota`). In `reconcile_logp`, missing intermediate ranks get `0` in `chains`, emitting `<unknown>` or `unclassified_<anchor>` mid-lineage.

5. **Genetic code & organelle blindness (`src/omniphyl/encode.py`, `src/omniphyl/annotate.py`, `src/omniphyl/refiner.py`):**
   - `aa_encode` and `aa_frame_encode` hardcode `_AA_LUT` (standard genetic code 1), and `annotate.py` only distinguishes code `1` vs code `4` (`_CODE4_STOPS = {"TAA", "TAG"}`).
   - Across ToL: **Code 2** (vertebrate mitochondria: `AGA`/`AGG` = stop, `TGA` = Trp, `ATA` = Met), **Code 5** (invertebrate mitochondria: `TGA` = Trp), **Code 11** (bacterial/archaeal/plant plastid: standard stops, `GTG`/`TTG`/`CTG`/`ATT`/`ATC`/`ATA` starts), **Code 6** (*Ciliata*/*Hexamita*: `TAA`/`TAG` = Gln, only `TGA` = stop), **Code 12** (*Candida* CTG clade), and **Code 22/23/16** (green algal/protist mitochondria) occur routinely. Worse, a single eukaryotic assembly file contains **nuclear chromosomes (Code 1) alongside mitochondrial (Code 2/5) and plastid (Code 11) contigs** — genome-wide `gcode` inference in `run_annotate` (`annotate.py:321-354`) misclassifies organelle contigs!

6. **50k–100k species head scaling & multi-GPU training (`src/omniphyl/model.py`, `scripts/train_omniphyl.py`, `scripts/train_refiner.py`):**
   - At 80,000 species and `latent = 640` (`width=64` → `512` + `aux_dim=128`), `self.heads["species"]` alone has `640 × 80,001 = 51.2M` parameters (5× the entire backbone), and a batch of `1024` windows materializes `[1024, 80001]` fp32 logits (`328 MB` fwd + `328 MB` grad per GPU step, and `4096 × 80001 × 4B = 1.31 GB` during `annotate` sub-batching).
   - Neither `train_omniphyl.py` nor `train_refiner.py` supports PyTorch `DistributedDataParallel` (`torchrun`), `bfloat16` mixed precision (`torch.amp.autocast("cuda", dtype=torch.bfloat16)` — native on Hopper/Ampere), resume-from-checkpoint (`--resume`), or memory-mapped binary sequence stores (currently `cache_seqs` reads an ASCII `.fna` sequentially in every DDP process, duplicating 15–30 GB × `N_procs` RAM if multi-GPU is used).

7. **Alt-splicing & overlapping-gene label corruption in `build_refiner_labels` (`scripts/train_refiner.py:72-119`):**
   - Mammalian/plant GFF3s contain 3–15 alternative mRNA isoforms per gene. Iterating over all `(tx, strand)` in `by_tx` overwrites `state[s:e] = cds_i` and `state[e:next_s] = int_i` in arbitrary dict order: a constitutive exon in isoform A is overwritten as `ST_INT` by an intron of minor isoform B! Furthermore, `ST_RNA` (`5`) and `ST_DUAL` (`6`) exist in `N_STATES = 7` and `STATE_W` (`train_refiner.py:66`) **but are never populated anywhere in `build_refiner_labels`**.

---

## 2. Guiding Design Principles

Following project conventions:
- **Prefer unified mechanisms over clade-specific `if/else` trees:** Do not write separate "mammalian decoder", "plant decoder", and "bacterial decoder" forks. Parameterize chunking, receptive field, genetic-code tables, and ancestor chains cleanly so one forward/decode path handles all clades.
- **Backward compatibility:** Existing v5/v6 checkpoints must continue to load in `OmniPhylPredictor` and `load_refiner`. Gate any tensor-shape changes via `ckpt["args"]` / `ckpt["config"]` with safe defaults matching v6.
- **Cross-architecture & cross-accelerator portability (`aarch64` ↔ `x86_64` Intel/AMD, CUDA ↔ ROCm ↔ Apple MPS ↔ CPU):** Models trained on TACC Vista (`aarch64` Grace + Hopper `sm_90`) must ship as 100% hardware-agnostic checkpoints and run identically on x86_64 AMD/Intel CPUs, AMD ROCm GPUs, NVIDIA GPUs, and Apple Silicon (`mps`) with zero custom C++/CUDA extensions (`numpy` + pure PyTorch ops only). Specifically:
  - Always unwrap DDP/compiled wrappers before saving (`raw_model = model.module if hasattr(model, "module") else model`, stripping any `_orig_mod.` prefix) and cast saved weights to canonical `float32` CPU tensors (never serialize `bfloat16` master weights or CUDA-pinned tensors that break older CPU/MPS builds).
  - Serialize metadata (`vocab`, `tree`, `flat_log_prior`, `config`, `args`) as plain Python/JSON primitives or explicit little-endian `np.ndarray.tolist()` — never pickle custom classes, OS `Path` objects (`PosixPath`), or architecture-specific NumPy dtypes.
  - Keep `device="auto"` in `OmniPhylPredictor` and `annotate`/`profile` resolving `"cuda"` (covers both NVIDIA CUDA and AMD ROCm `torch.cuda`) $\rightarrow$ `"mps"` $\rightarrow$ `"cpu"`, with standard `nn.MultiheadAttention` / `nn.TransformerEncoderLayer` (which uses FlashAttention on Hopper automatically and falls back to math SDPA on AMD/CPU/MPS seamlessly).
- **Document non-obvious thresholds and biological constants inline** (citing empirical genome statistics where relevant).
- **Every architectural change must pass a fast mini-ToL gate** before Plan 2 spends TACC SUs building and training on the full corpus.

---

## 3. Work Packages & Exact Code Specifications

### WP1: Multi-Node / Multi-GPU DDP, BF16, Packed Binary Store & Scalable Species Head
**Files:** `src/omniphyl/model.py`, `scripts/train_omniphyl.py`, `scripts/train_refiner.py`, `scripts/pack_corpus.py` (new)

1. **Zero-copy `mmap` Corpus Format (`scripts/pack_corpus.py` + `FragmentDataset`):**
   - At ToL scale (5M–15M fragments), ASCII FASTA + Python `dict` `labels.tsv` loading in `train_omniphyl.py` takes ~40–80 GB RAM per process and minutes of startup.
   - Add `scripts/pack_corpus.py` converting `(fragments.fna, labels.tsv)` into a directory `<corpus>.packed/` containing:
     - `seqs.2bit` (or `seqs.u8` 4-bit IUPAC/ACGTN codes) + `seq_offsets.npy` (`int64 [N+1]`): `np.memmap(..., mode="r")` shared across all DDP ranks and DataLoader workers with zero RAM duplication.
     - `tax_ids.npy` (`int32 [N, 7]`), `cont.npy` (`float32 [N, 2]`), `annot.npy` (`float32 [N, ANNOT_COLS]`), `acc_idx.npy` (`int32 [N]`), `genus_idx.npy` (`int32 [N]`), `domain_csr_{indptr,indices}.npy` (sparse CSR for Pfam hits), and `vocab.json` + `meta.json`.
   - Keep `FragmentDataset` accepting raw `--fragments` + `--labels` unchanged when `--packed` is not passed, so `tests/test_smoke.py` and small runs need zero extra steps.

2. **`DistributedDataParallel` + `bf16` + `--resume` in `train_omniphyl.py` and `train_refiner.py`:**
   - Detect `RANK` / `LOCAL_RANK` / `WORLD_SIZE` in `os.environ` (standard `torchrun` / SLURM wrapper env vars). When `WORLD_SIZE > 1`:
     - Initialize `torch.distributed.init_process_group("nccl")`, bind `cuda:{LOCAL_RANK}`, use `DistributedSampler` on train/val loaders, all-reduce epoch loss and val confusion/accuracy counters across ranks, and write checkpoints/logs on `rank == 0` only.
     - When `train_refiner.py` uses `--balance-genomes` under DDP, partition indices per rank deterministically from the global `WeightedRandomSampler` draw seeded by `seed + epoch` so genome balancing composes with DDP.
   - Add `--amp {none,fp16,bf16,auto}` (default `auto`: `bf16` when CUDA capability $\ge (8, 0)$, `fp16` on older GPUs like T4, `none` on CPU). Keep `log_softmax` / loss reductions in `float32`.
   - Add `--resume <ckpt.pt>` and `--resume-auto` (resumes `<out>/last.pt` if present, no-op on fresh start — ideal for idempotent SLURM retries) saving and restoring `optimizer`, `scheduler`, `scaler`, `epoch`, and `best_metric` in `last.pt`.
  - Add `--arch-config <locked.json>` to BOTH `train_omniphyl.py` and `train_refiner.py`: when present, hyperparameters listed in the corresponding `stage1`/`stage2_refiner` block of `docs/tol-arch-locked.json` become the defaults (explicit CLI flags still override). This is how Plan 2's SLURM scripts consume the lock contract without duplicating ~15 flags.

3. **Bottlenecked Cosine/Linear Species Head (`--species-bottleneck` in `OmniPhyl`):**
   - In `src/omniphyl/model.py`, add optional `head_bottleneck: int = 0` (recommended ToL value: `256` when `n_classes[r] > 8192`).
   - Replace `nn.Linear(self.latent, C_r)` for large-vocabulary ranks (`C_r > 8192`, i.e., `genus` and `species` at ToL scale) with `nn.Sequential(nn.Linear(self.latent, head_bottleneck, bias=False), nn.LayerNorm(head_bottleneck), nn.GELU(), nn.Linear(head_bottleneck, C_r))`.
   - At `C_species = 100,000` (Plan 2's `species × 2` ToL corpus reaches `~90k+` species) and `latent = 896` (`width=96`), `head_bottleneck = 256` cuts `species` head parameters from `89.6M` down to `25.9M` (a **3.5× reduction** in head params and optimizer states, and >3× smaller final projection GEMM).
   - Also add **chunked cross-entropy / selective evaluation**: during training, compute species CE only on `m = y[:, i] != 0` rows (`z_bottleneck[m] @ W.T + b`), skipping the `[B, 80000]` matmul on rows with masked species labels (e.g. MAGs or genus-only viral/environmental entries).
   - In `OmniPhylPredictor.predict_sequences` (`src/omniphyl/model.py:420-485`), add `return_window_tax: bool = True` and streaming rank-sum mode (`predict_and_sum_taxonomy`) or lower `PRED_SUBBATCH` dynamically as `max(256, int(2e8 // max_n_classes))` so `80k–150k` species classes never exceed ~800 MB transient RAM during `annotate` or `profile`.

---

### WP2: Gap-Aware Taxonomy Graph & Organelle/Host-Aware Heads
**Files:** `src/omniphyl/model.py`, `scripts/fetch_refseq_corpus.py`, `src/omniphyl/profile.py`, `scripts/screen_contaminants.py`

1. **Gap-Bridging Ancestor Chains (`build_tree`, `ancestor_chains`, `reconcile_logp`):**
   - Currently `build_tree` (`src/omniphyl/model.py:143-170`) only inspects `ranks[ri - 1]` (`parent_votes[r][m[r]][m.get(ranks[ri - 1], "")]`). When a taxon lacks an NCBI `phylum` or `class` (standard across eukaryotic supergroups like `Sar`/`Amoebozoa` and many viral realms), `po[c + 1]` becomes `0`.
   - **Upgrade:** Store `ancestor_of: dict[str, np.ndarray]` of shape `[n_r, ri + 1]` directly inside `ckpt["tree"]` built by majority vote over **every** shallower rank `j < ri` independently (`votes[r][j][name][m.get(ranks[j], "")]`).
     - If `class` is missing (`""`) on *Trypanosoma*, `ancestor_of["order"][c_tryp]` has `col_domain = id("Eukaryota")`, `col_phylum = id("Euglenozoa")` (or `0` if genuinely absent in NCBI), `col_class = 0` (`GAP_SENTINEL = -2`, distinct from `OOV_SENTINEL = 0`), `col_order = c_tryp`.
     - Distinguishing `-2` (**true taxonomic rankless gap**: NCBI has no `class` for this clade) from `0` (**parent exists in biology, out-of-vocab**) fixes two bugs at once:
       1. `path_scores` averages log-probabilities over **defined** ranks (`chain >= 1`), without penalizing clades that lack formal Linnaean `phylum`/`class` names.
       2. `_emit_calls` emits `""` (or `"incertae_sedis"` / `f"no_{t}_{anchor_name}"` per convention — keep `;`-joined 7-slot format intact with `""` or explicit tag, documented in `TaxonCall`) instead of misreporting a well-resolved *Trypanosoma brucei* call as `<unknown>` at `class`.
   - **Viral realm & Eukaryotic kingdom fallback in `fetch_refseq_corpus.py`:**
     - In `resolve_taxonomy` (`fetch_refseq_corpus.py`), verify `rankedlineage.dmp` mapping: `domain` maps from `superkingdom` (or `realm` for viruses in newer NCBI taxdumps where `superkingdom == ""` and viral realms `Riboviria`/`Duplodnaviria` populate the top slot — normalize all viral taxids to `domain = "Viruses"` and, when `phylum` is empty but `kingdom` is populated, keep standard `phylum..species` from `RL_COLS` while recording `kingdom` for gap grouping).

2. **Repeat / Low-Complexity Channel & Host-Depletion Mode:**
   - **Encoder (`src/omniphyl/encode.py`):**
     - Add a lightweight sequence-complexity summary (3 scalars appended to `AA_EXTRA_DIM` when `tol_feats=True`, or 1 spatial channel on `msfcgr` when `--complexity-channel` is set — **prefer appending 4 composition/repeat scalars** `[shannon_k3_entropy, homopolymer_max_frac, tandem_period_autocorr, lowercase_or_dust_frac]` to a dedicated `dna_stats` vector fused into `z_dna` so the 2D conv stem `cin` stays unchanged unless needed!).
     - Why scalars on `z_dna` rather than changing `msfcgr` channels: changing `cin` breaks weight-shape compatibility and conv FLOPs, whereas 4 exact window-level complexity statistics (`k=1..4` normalized Shannon entropy, max dinucleotide/trinucleotide tandem repeat run fraction, GC% and CpG observed/expected ratio — critical in vertebrates where CpG islands mark promoters/5' exons!) cost `<5 µs`/window in NumPy and give both taxonomy calibration (`p_ood` / abstention on pure simple repeats) and Stage-1 coding/promoter detection direct access to CpG O/E and repeat entropy.
   - **Contaminant Screener (`scripts/screen_contaminants.py`):**
     - Fix the vertebrate gene-stripping bug: replace the single `--contam-acc-frac 0.5` rule with clade-aware screening:
       - **Always** screen all groups against **technical/synthetic** artefacts (`UniVec_Core` adapters/vectors + `phiX174`, exempting `phiX174` itself).
       - Apply the **human GRCh38** k-mer screen **only** to non-chordate groups (`bacteria`, `archaea`, `fungi`, `protozoa`, `plant`, `viral`, and non-chordate `invertebrate`), OR use a **human-specific repeat/non-conserved sketch** (Alu/LINE1/alpha-satellite + primate-specific intronic k-mers) rather than coding exons when screening `vertebrate_other` / `vertebrate_mammalian`, and explicitly exempt `Homo sapiens` (`species_taxid == 9606`) as the legitimate **host** reference class!
   - **Profiler (`src/omniphyl/profile.py`):**
     - Add `--host-filter <taxon_substring>` (e.g., `--host-filter Chordata` or `--host-filter "Homo sapiens"`): emits host read/base fraction separately in `summary.json` (`host_fraction`) and optionally excludes host units from microbial relative abundance (`taxonomic_profile.tsv`).
     - Tighten `cluster_reads`: replace the fixed `m > 200` / `max_kmer_freq = 0.10` rule with `cap = min(64, max(2, int(max_kmer_freq * n)))` and skip low-entropy 21-mers (<=2 distinct bases or dinucleotide tandem repeats covering >=16/21 bp) before building `(kmer, read)` pairs so satellite/microsatellite reads in eukaryotic metagenomes cannot blow up sort memory or chain unrelated contigs.

---

### WP3: Multi-Genetic-Code & Organelle Support Across Stage 1, Stage 2, and Inference
**Files:** `src/omniphyl/encode.py`, `src/omniphyl/annotate.py`, `src/omniphyl/refiner.py`, `scripts/fetch_refseq_corpus.py`, `scripts/train_refiner.py`

1. **Canonical NCBI Genetic Code Table (`src/omniphyl/encode.py`):**
   - Replace the single `_AA_LUT` with `GCODE_LUTS: dict[int, np.ndarray]` and `GCODE_STOPS: dict[int, set[str]]`, `GCODE_STARTS: dict[int, set[str]]` covering all biologically major NCBI translation tables:
     - `1`: Standard (`TAA, TAG, TGA` stops; `ATG` euk / `ATG, GTG, TTG` prok starts)
     - `2`: Vertebrate mitochondrial (`TAA, TAG, AGA, AGG` stops; `TGA` = Trp; `ATG, ATA, ATT, ATC, GTG` starts)
     - `4`: Mold/Protozoan/Coelenterate mitochondrial & *Mycoplasma*/*Spiroplasma* (`TAA, TAG` stops; `TGA` = Trp)
     - `5`: Invertebrate mitochondrial (`TAA, TAG` stops; `TGA` = Trp; `AGA, AGG` = Ser)
     - `6`: Ciliate/Dasycladacean/*Hexamita* nuclear (`TGA` only stop; `TAA, TAG` = Gln)
     - `11`: Bacterial, Archaeal, and Plant Plastid (`TAA, TAG, TGA` stops; `ATG, GTG, TTG, CTG, ATT, ATC, ATA` starts)
     - `12`: Alternative Yeast nuclear (*Candida* `CTG` = Ser)
     - `16`, `22`, `23`: Chlorophycean / *Scenedesmus* / *Thraustochytrium* mitochondrial/organellar tables
     - `25`: Candidate Division SR1 / *Gracilibacteria* (`TGA` = Gly — common in gut/environmental MAGs!).
   - Keep `aa_encode(seq)` using standard code `1` as the fixed-basis representation for Stage-1 window input (at window-prediction time the genetic code is not yet known!), **AND** expand `AA_EXTRA_DIM` (currently `18 = 6 frames × [stop_density, longest_orf_frac, tga_share]`) with per-frame `taa_tag_only_orf_frac` and `aga_agg_density` (`+12` dims → `30`, gated by `ckpt` config `aa_extra_v2: bool`) so the gated AA branch and `struct_head` can natively recognize ciliate (`TAA/TAG`=Gln) and vertebrate-mitochondrial (`AGA/AGG`=stop) reading frames!

2. **Per-Contig Organelle + Genetic-Code Resolution (`src/omniphyl/annotate.py`):**
   - **Bug today:** `run_annotate` (`annotate.py:321-354`) infers a **single** scalar `gcode` across the entire FASTA. On a plant, protist, fungal, or animal assembly containing chr1..chrN + chrM + chrPlastid, the nuclear chromosomes overwhelm `total`, forcing `gcode = 1` onto the mitochondrion (`gcode = 2/4/5`) and truncating every mitochondrial COX1/COB/ND gene at internal `TGA` codons!
   - **Upgrade:**
     - Add an **organelle / repliconhead or classifier** (4-way window head `replicon` ∈ `{nuclear_or_chrom, mitochondrion, plastid, plasmid_or_viral}` labeled for free from RefSeq GFF `##sequence-region` / `genome=mitochondrion|chloroplast|apicoplast|plasmid` attributes on feature `region` in `fetch_refseq_corpus.py`!).
     - Even without retraining, infer `gcode` **per contig** (falling back to the assembly-wide call only when a contig has `< 300` coding codons AND no replicon/taxonomy hint):
       - **Stop-reassigned codes (`{2, 4, 5, 6, 25}` vs standard-stop `{1, 11, 12}`):** resolve via log-likelihood ratio of in-frame `TGA`, `TAA/TAG`, and `AGA/AGG` frequencies on high-confidence `p_frame` windows combined with the contig's `replicon` + `taxon` posterior.
       - **Sense-reassigned / start-expanded codes with standard stops (`11` bacterial/plastid alternative starts, `12` *Candida*/CTG-clade `CTG`→Ser):** stop counts are identical to Code 1, so resolve directly from `(replicon, res.taxon)` (`Bacteria`/`Archaea`/`plastid` → `11`; `Debaryomycetaceae`/*Candida* CTG clade → `12` for protein translation).
     - Store `res.gcode` on each `ContigResult`, pass `res.gcode` into `build_contig_cond` (replace binary `cond[9] = (gcode == 4)` with a 5-dim genetic-code / replicon feature slice or backward-compatible appended dims in `N_COND`), and pass `stops = GCODE_STOPS[res.gcode]`, `starts = GCODE_STARTS[res.gcode]` into `decode_with_refiner` / `_decode_genes`.
   - ** Corpus label fix (`scripts/fetch_refseq_corpus.py`):**
     - Parse `transl_table=` from GFF3 `CDS` attributes (`transl_table=4`, `transl_table=2`, `transl_table=11`, etc.) and `genome=` from `region` features! Use the GFF's true `transl_table` when extracting ORFs for `annotate_fragments_pfam` (`fetch_refseq_corpus.py:850-930`) — currently standard-code translation chops code-4, code-6, and mitochondrial ORFs before `pyhmmer` even runs.

---

### WP4: Stage-2 Neural Refiner & Semi-Markov Decoder for Long-Intron Metazoans & Plants
**Files:** `src/omniphyl/refiner.py`, `scripts/train_refiner.py`, `src/omniphyl/annotate.py`

This is the single highest-impact architectural upgrade for ToL.

1. **Multi-Scale Long-Range Refiner Architecture (`OmniPhylRefiner` v2):**
   - Current v1: `CHUNK = 8192` (`stride = 6144`), 6 dilated Conv1d blocks (`d = 1, 2, 4, 8, 16, 32` → RF = `7 + 6 × 63 = 385`... actually `2 * 3 * (1+2+4+8+16+32) + 7 = 385 bp` per side = `771 bp` conv RF!) + `avg_pool1d(4)` → 1-layer BiGRU over `2,048` steps.
   - Over a `32,768 bp` chunk (needed so ~95% of plant, invertebrate, fish, bird, and mammalian introns fit *inside* a single chunk with both donor and acceptor visible), `avg_pool1d(4)` gives `8,192` BiGRU steps — slow sequentially and forgetful across 20 kb introns.
   - **Upgrade (`OmniPhylRefiner` v2, config-switched via `arch_version=2` so v1 checkpoints still load):**
     - Increase `CHUNK = 32768`, `CHUNK_STRIDE = 24576` (configurable; `8192` retained for v1 checkpoints). **Make chunk geometry config-driven, not module-global:** `_chunk_starts` (`refiner.py:64`) and `refiner_tracks` (`refiner.py:345`) currently read the module constants `CHUNK`/`CHUNK_STRIDE`; v2 must read `refiner.cfg["chunk"]`/`["chunk_stride"]` stored in the checkpoint so v1 (`8192`) and v2 (`32768`) models coexist in one process.
     - Expand dilated residual stack to **8 blocks** `d = (1, 2, 4, 8, 16, 32, 64, 128)` (local nucleotide + splice-branchpoint + exon-definition RF $\approx 3.1\text{ kb}$).
     - Replace the single 4×-downsampled BiGRU with a **two-scale context pyramid**:
       - **Meso-scale (8× pool, `4,096` steps for 32 kb):** 1-layer BiGRU (`hidden`) capturing exon-to-exon phase & medium introns (`50 bp – 5 kb`).
       - **Macro-scale (64× pool, `512` steps for 32 kb, `1 token = 64 bp`):** 2-layer TransformerEncoder (or 2-layer BiGRU with RoPE/conv positional encoding; `512` tokens makes self-attention trivial: $512^2$ is tiny, runs in `<0.5 ms` on GPU, and has **zero recurrence decay across the entire 32.8 kb window**, directly pairing a donor at bp `2,000` with its acceptor at bp `29,000`!).
       - Upsample both scales linearly and add residually to `h` before `head_state` and `head_site`.
     - Add an explicit **3-way donor/acceptor splice-phase head** or **exon-reading-frame head** (`head_phase: Conv1d(hidden, 3, 1)` predicting `cds_phase ∈ {0, 1, 2}` on coding bases, supervised directly from GFF CDS phase):
       - Why this matters for long introns: in human/vertebrate genes with 15 exons separated by 10 kb introns, `Stage-1 p_frame` (500 bp windows) washes out on tiny `60–120 bp` internal vertebrate exons! Having the refiner itself emit per-base reading phase `[L, 3]` per strand gives **sharp exon-definition and exact splice-junction frame compatibility** (`(donor_phase + acceptor_phase) == 0 mod 3`) across arbitrary intron lengths!

2. **Fix Canonical Representative Isoform + `ST_RNA` / `ST_DUAL` Labels (`scripts/train_refiner.py`):**
   - Update `parse_cds` / `build_refiner_labels`:
     - **Isoform selection:** Group transcripts by `Parent` gene ID (not all isoforms blindly flattened!). Select the **MANE Select / longest-CDS canonical isoform** per gene locus for the primary `state` map (`ST_CDS_P/M`, `ST_INT_P/M`) so minor intron-retention or exon-skipping isoforms never corrupt constitutive exons into `ST_INT`. Optionally union high-confidence splice boundaries into a soft secondary mask so alternative splice sites aren't penalized as hard negatives in `L_site`.
     - **Populate `ST_RNA` (`5`) and `ST_DUAL` (`6`):** Parse `rRNA`, `tRNA`, `ncRNA`, `snRNA`, `snoRNA` intervals from GFF3 into `ST_RNA`, and mark bases covered by **both** `+` and `-` canonical CDSs as `ST_DUAL` (common in compact viral, bacterial, and protist genomes).
   - **Fix `_phase_loss` truncation bug (`scripts/train_refiner.py:310-326`):**
     - Only include a true donor `d` in `_phase_loss` **when its ground-truth partner acceptor `a_true` from the same transcript also falls inside `[0, chunk_len)`**! Never penalize a donor whose true acceptor lies past the right edge of the chunk (`a_true >= chunk_len`) or exceeds `max_intron`. Even better: supervise `(d_true, a_true)` directly from transcript exon pairs `(ivs[i][1], ivs[i+1][0] - 1)` passed alongside `sites`, rather than guessing via `pa[b, lo_w:hi_w].max()`.

3. **Long-Intron & Micro-Exon Semi-Markov Decoder Upgrades (`src/omniphyl/refiner.py`):**
   - **Clade-conditioned `max_intron`:** Instead of a flat `--max-intron 5000` default for all eukaryotes (`cli.py:67`), add `--max-intron auto` (default) resolved from `_INTRON_LENGTH_PRIORS` / `res.taxon`:
     - Compact protists/fungi (`apicomplexa`, `kinetoplast`, `ascomycota`, `basidiomycota`, `microsporidia`): `max_intron = 5_000`
     - Plants / nematodes / basal metazoans (`streptophyta`, `chlorophyta`, `nematoda`): `max_intron = 15_000`
     - Arthropods / molluscs / vertebrates (`arthropoda`, `chordata`, `vertebrata`, `mammalia`): `max_intron = 50_000` (overridable on CLI).
   - **Non-canonical U12 (`AT..AC`) splice support:** Vertebrates, plants, and some fungi have ~0.5% U12-type introns (`AT..AC` or `GT..AG` with sharp `CTATCCTT` branchpoint). Allow `AT..AC` donor/acceptor pairs when **both** `p_don[d] >= 0.35` and `p_acc[a - 1] >= 0.35` (high site-posterior gate prevents `AT..AC` decoy explosion).
   - **Locus bridging across long introns (`_refiner_loci`):** Bridge adjacent coding/intronic islands separated by up to `clade_max_intron` when the facing edges carry unmatched high-posterior donor (`p_don >= 0.25`) → acceptor (`p_acc >= 0.25`) polarity on the same strand and no intervening high-confidence `p_stop` + polyA/intergenic block terminates the gene.
   - **Short internal exon recovery:** Lower the per-exon length/mean-CDS gate on internal exons flanked by high-confidence splice sites (`p_don >= 0.3, p_acc >= 0.3`) down to `>= 9 bp` (median human internal exon is `~120 bp`, ~10% are `15–60 bp` and currently fail when coarse Stage-1 tracks blur them).
   - **Conditioning hygiene (`COND_MODES`):** Make `reduced` + clade-embedding (or `regularize` over a ≥50-genome ToL refiner panel) the **default** `cond_mode` instead of `full`, removing raw contig GC and raw `path_conf` genome-identity leaks permanently.

4. **Documented biological edge cases (record as known limitations in `docs/`; do NOT special-case in v1):**
   - **Selenocysteine / pyrrolysine recoding:** ~25 human selenoproteins and Methanosarcina pyl proteins translate `TGA`/`TAG` as Sec/Pyl — these genes will be emitted as two fragments by the decoder. GFF `transl_except=` attributes exist for such CDS; optionally exclude them from `L_stop` supervision rather than decoding correctly.
   - **Programmed ribosomal frameshifts (viral −1 PRF, bacterial prfB):** overlapping same-strand CDS in two frames (e.g. SARS-CoV-2 ORF1a/1b) produce contradictory `active_frame` labels; policy: label the LONGER upstream frame as primary, count the overlap region toward neither `struct_flags` negative supervision, and document.
   - **Kinetoplastid RNA editing / SL trans-splicing / operons:** gene-level CDS labels are still correct (editing happens post-transcription; SL sites aren't gene boundaries), so no code change — document that cryptogenes and polycistronic internal boundaries are out of scope for ab initio prediction.

---

## 4. Mini-ToL Stress-Test & Validation Benchmark (Local CPU + Kaggle T4 — Zero TACC Spend)

All architecture development and empirical validation in Plan 1 runs on **local CPU (`tests/test_tol_arch.py`) + Kaggle GPU kernels (`kaggle/tol-*`)** so **zero TACC SUs are consumed until the architecture is completely locked**.

Every Kaggle kernel in Plan 1 must fit comfortably inside Kaggle's hard VM ceilings (**16 GB T4/P100 VRAM** in `fp16` AMP, **13–29 GB host RAM**, **19.5 GB `/kaggle/working`**, **12h walltime cap**):
- Provide three dedicated, self-contained kernel directories following the existing `kaggle/<name>/{kernel-metadata.json,kernel.py}` convention:
  1. `kaggle/tol-mini-corpus/` (CPU internet kernel, ~45–60 min, `< 4 GB` disk): fetches & packs the ~150k-window 9-group Mini-ToL Stage-1 corpus (`corpus.packed/` ~180 MB) + slices the 15-species Mini-ToL Refiner chromosome/organelle panel (~110 MB compressed), saved as reusable Kaggle dataset `omniphyl-tol-mini`.
  2. `kaggle/tol-stage1-bench/` (1× T4 GPU kernel, ~60–75 min, `< 8 GB` VRAM in `fp16` via `--amp auto`): runs the Stage-1 Mini-ToL ablations (`v6` baseline vs `OmniPhyl` ToL upgrades + synthetic `C_species=100,000` bottleneck memory/speed probe) and emits `stage1_mini_best.pt` + `stage1_gate_report.json`.
  3. `kaggle/tol-refiner-bench/` (1× T4 GPU kernel, ~90–120 min, `< 10 GB` VRAM at `batch_size=4, chunk=32768` in `fp16`): caches Stage-1 `.res.pkl` tracks using `stage1_mini_best.pt`, trains `OmniPhylRefiner` v2 vs v1 (10 epochs, `--balance-genomes`), runs held-out decode across prokaryotes, protists, plants, arthropods, vertebrates, and organelles, and emits `refiner_gate_report.json`.

### 4.1 Offline Unit & Invariant Suite (`tests/test_tol_arch.py`, CPU < 30s)
Must pass 100% with zero network access:
1. **Packed binary dataset parity:** `pack_corpus.py` output produces bit-identical encoded sequences, labels, continuous targets, and sparse Pfam targets to `FragmentDataset` over raw `.fna` + `.tsv`.
2. **Gap-aware taxonomy reconciliation:** Synthetic lineage with rankless gaps (`domain=Eukaryota, phylum="", class="", order=Trypanosomatida, ...`) scores ancestor paths without penalty and emits valid lineages without `<unknown>` corruption.
3. **Genetic code tables & per-contig inference:** Verify translation and start/stop scanning for Codes `1, 2, 4, 5, 6, 11, 12, 25`; verify a multi-contig synthetic assembly (`chr1` in Code 1 + `chrM` in Code 2 + `chrPlastid` in Code 11) gets `res.gcode == 1, 2, 11` per contig and decodes full-length `TGA`-containing mitochondrial ORFs.
4. **Refiner v2 long-intron forward & loss:** Synthetic `32,768 bp` contig with a `22,000 bp` intron and a `45 bp` internal micro-exon:
   - Verify `OmniPhylRefiner(arch_version=2)` shapes, memory (`< 2 GB` at batch 4 in `bf16`), and backward pass.
   - Verify `_phase_loss` does **not** penalize a donor at `bp 30,000` whose true acceptor is at `bp 42,000` (outside chunk).
   - Verify `build_refiner_labels` picks the canonical isoform when given 3 conflicting alt-splice mRNAs and populates `ST_RNA` and `ST_DUAL`.
   - Verify `decode_with_refiner` recovers a 3-exon gene with a `20 kb` intron and `45 bp` middle exon given ideal/peaked posteriors.
5. **Backward compatibility:** Load a mock v1 refiner checkpoint (`CHUNK=8192`, no macro-transformer, no `head_phase`) and a mock v6 Stage-1 checkpoint (no `head_bottleneck`, 18-d `AA_EXTRA_DIM`) and run `omniphyl annotate` + `omniphyl profile` end-to-end.
6. **Cross-platform checkpoint portability invariant:** Verify saved Stage-1, Stage-2, and bundled `.pt` checkpoints contain zero `"module."` or `"_orig_mod."` state-dict prefixes, zero non-CPU or `bfloat16` weight tensors, and load cleanly under `map_location="cpu"` with `omniphyl profile` + `omniphyl annotate --device cpu` producing finite, deterministic outputs on any `x86_64` (AMD/Intel) or `aarch64` host.

### 4.2 Empirical Mini-ToL Stress Run (`scripts/tol_arch_bench.py`, 1 GPU, ~90 min)
1. **Mini-ToL Stage-1 Corpus (~150k windows across all 9 RefSeq groups):**
   - Fetch 15–25 diverse assemblies per group across **all 9 groups** (`archaea`, `bacteria`, `fungi`, `protozoa`, `plant`, `invertebrate`, `vertebrate_other`, `vertebrate_mammalian`, `viral`, plus mitochondrial/plastid replicons and ciliate/mycoplasma alternate-code taxa), ~200 assemblies total.
   - Compare Baseline v6 arch vs ToL-upgraded arch (5 epochs, `bf16`):
     - **Gate S1-A (Repeat/Host calibration):** False-discovery rate at `conf_tau=0.5` on held-out mammalian/plant repeat-rich windows $\le 0.10$ (no confident mis-calls on Alu/LTR windows to random species); `Homo sapiens` windows retained by `screen_contaminants.py` and classified at species rank $\ge 90\%$ on non-repeat windows.
     - **Gate S1-B (Throughput & Memory):** `head_bottleneck=256` tested at synthetic `C_species = 120,000` (25%+ headroom above Plan 2's `~90k` species vocab) runs in `fp16`/`bf16` at batch size `512` on 1 GPU under `16 GB` VRAM with `< 5%` macro-accuracy delta on the mini-ToL val split.
     - **Gate S1-C (Gap-lineage reconciliation):** Zero spurious `<unknown>` ancestor collapses on rankless protist/viral clades; hierarchical consistency = `100%`.

2. **Mini-ToL Stage-2 Refiner Panel (12 small/chromosome-slice genomes spanning all gene-structure regimes):**
   - Train `OmniPhylRefiner` v2 (10 epochs) on single chromosomes/scaffolds from 9 training species (`E. coli`, *H. salinarum*, *S. cerevisiae*, *P. falciparum*, *A. thaliana* chr1, *C. elegans* chrI, *D. melanogaster* chr2L, *D. rerio* chr1 slice, *M. musculus* chr19 slice + vertebrate `chrM`), and evaluate held-out on **6 unseen species** (`B. subtilis`, *P. vivax*, *O. sativa* / *Z. mays* slice, *A. gambiae* slice, *G. gallus* slice, *H. sapiens* chr22 10 Mb slice + human `chrM`):
     - **Gate S2-A (Microbial/Protist non-regression):** Held-out bacteria/yeast/apicomplexan nucleotide F1 within `-0.01` of v1 refiner (or better), exon F1 $\ge$ v1.
     - **Gate S2-B (Plant & Metazoan multi-exon recovery):** Held-out *A. thaliana*/*O. sativa* and *A. gambiae* exon F1 improves by $\ge +0.15$ absolute over v1 (`CHUNK=8192`, `max_intron=5000`); held-out vertebrate (*H. sapiens* chr22 / *G. gallus*) achieves non-trivial multi-exon splicing recovery on introns `5–35 kb` (intron recall on `5–30 kb` introns $\ge 0.40$ in mini run vs `0.00` today!).
     - **Gate S2-C (Organelle & Alt-code):** Human `chrM` (Code 2) and *Tetrahymena*/*Paramecium* (Code 6) or *Mycoplasma* (Code 4) contigs auto-detect the right `res.gcode` and achieve nucleotide F1 $\ge 0.85$ (vs catastrophic truncation under Code 1).

---

## 5. Architecture Lock Contract (Deliverable Handed to Plan 2)

Plan 1 is complete **only** when the implementing agent writes `docs/tol-arch-locked.json` (and updates `README.md` / CLI defaults) recording the exact frozen configuration consumed by Plan 2 (`docs/tol-corpus-vista-plan.md`):

```json
{
  "schema": "omniphyl-tol-arch-v1",
  "stage1": {
    "backbone": "msfcgr",
    "ms_ks": [4, 5, 6],
    "pos_channels": true,
    "dna_complexity_stats": true,
    "width": 96,
    "aux": ["aa"],
    "aa_mode": "gated",
    "aa_extra_v2": true,
    "aa_drop": 0.2,
    "head": "flat",
    "flat_prior_tau": 1.0,
    "head_bottleneck": 256,
    "bottleneck_min_classes": 8192,
    "struct_head": true,
    "replicon_head": true,
    "domain_head_k": 512,
    "annot_ctx": true,
    "var_len_prob": 0.25,
    "min_train_len": 150
  },
  "stage2_refiner": {
    "arch_version": 2,
    "chunk": 32768,
    "chunk_stride": 24576,
    "hidden": 128,
    "dilations": [1, 2, 4, 8, 16, 32, 64, 128],
    "meso_pool": 8,
    "macro_pool": 64,
    "macro_layers": 2,
    "phase_head": true,
    "cond_mode": "reduced",
    "canonical_isoform_only": true
  },
  "corpus_contract": {
    "packed_format": "v1",
    "labels_extra_columns": [
      "replicon", "transl_table", "sample_stratum", "tax_source",
      "gtdb_phylum", "gtdb_family", "gtdb_genus", "gtdb_species"
    ],
    "contam_screen_mode": "clade_aware_v2"
  }
}
```
*(Any parameter adjusted based on `tol_arch_bench.py` ablations must be updated in `docs/tol-arch-locked.json` with a 2-line justification citing the ablation numbers.)*

---

## 6. Step-by-Step Execution Checklist for the Agent

1. **Step 1 (Data & Taxonomy plumbing):** Implement gap-bridging `ancestor_of` (`-2` gap sentinel vs `0` OOV), `GCODE_LUTS`/`GCODE_STOPS`/`GCODE_STARTS` (codes `1,2,4,5,6,11,12,16,22,23,25`), `replicon` + `transl_table` GFF extraction in `fetch_refseq_corpus.py`, and clade-aware `screen_contaminants.py`.
2. **Step 2 (Stage-1 scaling & ToL heads):** Implement `dna_stats` complexity/CpG features, `aa_extra_v2`, `replicon` head, `head_bottleneck`, `scripts/pack_corpus.py` + `mmap` `FragmentDataset`, DDP + `bf16` + `--resume` in `train_omniphyl.py`, and streaming/bounded memory in `OmniPhylPredictor` + `--host-filter` in `profile.py`.
3. **Step 3 (Stage-2 Refiner v2 & Decoder):** Implement canonical-isoform + `ST_RNA`/`ST_DUAL` labelling, transcript-paired `_phase_loss` fix, `OmniPhylRefiner` v2 (`32,768 bp` chunk + 8 dilated blocks + 8× BiGRU + 64× Transformer + `head_phase`), per-contig `res.gcode`, and clade-adaptive long-intron + micro-exon decoding in `refiner.py` + DDP/`bf16`/`--resume` in `train_refiner.py`.
4. **Step 4 (Offline tests):** Write and pass `tests/test_smoke.py` + `tests/test_tol_arch.py`.
5. **Step 5 (Empirical Mini-ToL gate):** Run `scripts/tol_arch_bench.py`, verify Gates `S1-A..C` and `S2-A..C`, freeze `docs/tol-arch-locked.json`, and commit cleanly before starting Plan 2.
