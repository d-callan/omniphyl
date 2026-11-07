# Plan 2: Clade-Agnostic Stage-2 Neural Boundary Refiner (`OmniPhylRefiner`)

This document is a self-contained implementation specification for an independent agent with **no prior conversation context**. It specifies the exact files, tensor shapes, neural architecture, training pipeline, Viterbi/HSMM-style constrained decoder, and CLI integration to replace or augment OmniPhyl's heuristic Stage-2 ORF/splice decoder with a lightweight (~1.5M parameter), clade-agnostic per-nucleotide neural refiner conditioned on Stage-1 OmniPhyl tracks and taxonomic context.

---

## 0. Environment, Repository & Data Paths

- **Repository root:** `/home/danielle/Documents/veg/omniphyl`
- **Python / test runner:** `/home/danielle/miniconda3/envs/omniphyl/bin/python`
- **Existing files to reference / edit:**
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py` — Stage-1 window aggregation (`ContigResult`, `_fuse_contig`) and Stage-2 decoding (`run_annotate`, `_decode_genes`).
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/model.py` — `OmniPhyl`, `OmniPhylPredictor`, `TaxonCall`, `RANKS`.
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/encode.py` — `BASE2DIGIT`, `STOPS`, `STARTS`.
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/cli.py` — `omniphyl annotate` CLI flags.
  - `/home/danielle/Documents/veg/omniphyl/scripts/eval_genes.py` — nucleotide / exon / transcript sensitivity, specificity, and F1 evaluator.
  - `/home/danielle/Documents/veg/omniphyl/tests/test_inference.py` — unit and integration test suite.
- **New files to create:**
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/refiner.py` — `OmniPhylRefiner` PyTorch module, feature tensor builder, constrained Viterbi gene decoder, and checkpoint loader/saver.
  - `/home/danielle/Documents/veg/omniphyl/scripts/train_refiner.py` — standalone training and validation script for the Stage-2 refiner.
- **Local checkpoints & benchmark data:**
  - Stage-1 v6 checkpoint: `/home/danielle/omniphyl-work/ckpts/v6/w64_e15_best.pt`
  - Cached Stage-1 benchmark contig (*P. falciparum* chr1 `NC_004325.2`, 640,851 bp, 142 genes, 208 introns):
    - `/home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl`
    - `/home/danielle/omniphyl-work/infer_test/NC_004325.2.fasta`
    - `/home/danielle/omniphyl-work/infer_test/NC_004325.2.gff3`

---

## 1. Architectural Motivation & Design Overview

### Why Stage 1 Alone Cannot Resolve Exact Exon Boundaries
OmniPhyl Stage 1 (`OmniPhylPredictor`) operates on 500 bp sliding windows (`stride = 100 bp`, aggregated onto `B = 50 bp` bins in `ContigResult`). It excels at:
- **Clade & taxonomic identity** across all domains of life (`domain` through `species` + `p_ood`).
- **Coarse functional density** (`p_coding`, 6-frame `p_frame`, 4 boundary impulse tracks `b_tracks`, and optional RNA-seq `dtilde` / `e_lex` evidence).

However, 50 bp bins blur short exons (<120 bp), closely spaced splice junctions, and exact start/stop/splice coordinates. Currently, `_decode_genes` in `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py` uses ~850 lines of hand-tuned rules (`_splice_candidates`, `_intron_sets`, `_emit_spliced_orfs`, `_rescued_spliced_orfs`) to snap those blurry tracks to base-level ORFs.

### Why a Small Conditioned Refiner Works Across All Clades with One Set of Weights
Unlike standalone ab initio callers (e.g., Tiberius or Augustus, which must memorize clade-specific GC/hexamer composition and intron distributions from raw DNA alone and therefore require separate species/clade weight files), our Stage-2 refiner receives **OmniPhyl's Stage-1 predictions as input channels at every base** plus a **clade-conditioning vector**:
1. OmniPhyl's `p_frame` and `p_coding` already tell the refiner *where* the coding exons are and *which reading frame* is active.
2. OmniPhyl's taxonomic call (`domain` probabilities + contig-level summary statistics such as coding density, GC, and genetic code) tells the refiner via FiLM conditioning whether the contig is bacterial/viral (no spliceosome -> intron transitions suppressed), fungal/apicomplexan (short ~50–250 bp introns), or metazoan/plant (longer introns, distinct branch/poly-Y tracts).
3. Therefore, a **single ~1.5M-parameter refiner** only needs to learn how to combine local nucleotide syntax (start contexts, splice donor/acceptor motifs, branch-point/polypyrimidine tracts, stop codons) with OmniPhyl's upsampled 50 bp tracks to emit base-exact state posteriors and boundary logits.

---

## 2. Exact Input & Output Tensor Specification (`src/omniphyl/refiner.py`)

### 2.1 Chunking Grid
- Contigs are processed in overlapping windows of length `L_chunk = 8192` bp with stride `S_chunk = 6144` bp (`1024` bp overlap on each side; raised-cosine blending across overlaps). Contigs shorter than `8192` bp are passed unpadded at their exact length `L` (all layers are 1D convolutions + BiGRU, so arbitrary sequence lengths `L` work natively).

### 2.2 Per-Base Input Channels (`C_in = 22` float32 channels, shape `[B, 22, L]`)
Build a deterministic helper `build_refiner_inputs(seq: str, res: ContigResult, lo: int, hi: int, w_ev: float = 0.25) -> tuple[np.ndarray, np.ndarray]` in `/home/danielle/Documents/veg/omniphyl/src/omniphyl/refiner.py`:

For slice `seq[lo:hi]` of length `L = hi - lo`:
1. **Channels `0..3` (4 ch) — One-hot DNA (`A, C, G, T`):**
   - Map `seq[lo:hi]` via `BASE2DIGIT` (`0..3 -> 1.0`; `N`/ambiguous `-1 -> 0.25` across all 4 channels).
2. **Channel `4` (1 ch) — Softmask / ambiguous indicator:**
   - `1.0` where `BASE2DIGIT < 0`, else `0.0`.
3. **Channels `5..19` (15 ch) — Linearly interpolated Stage-1 `ContigResult` bin tracks at each base `x in [lo, hi)`:**
   - Convert base coordinate `x` to continuous bin coordinate `u = (x - B / 2) / B`, clamped to `[0, res.n_bins - 1]`, and linearly interpolate between `floor(u)` and `ceil(u)` (vectorized via `np.interp` or index gather):
     - **Ch `5`:** `p_coding` (`[nbins]`, raw model coding prob)
     - **Ch `6`:** `cscore` (`[nbins]`, fused coding + LexicMap evidence score from `annotate.py:492`)
     - **Ch `7`:** `p_boundary` (`[nbins]`)
     - **Ch `8..15` (8 ch):** `p_frame[:, 0..7]` (`[nbins, 8]`, contig-global frames `+0, +1, +2, -0, -1, -2, noncoding, dual`)
     - **Ch `16..19` (4 ch):** `np.tanh(res.b_tracks[:, 0..3])` (`[nbins, 4]`, squashed boundary impulses `start, stop, donor, acceptor`)
4. **Channels `20..21` (2 ch) — External evidence tracks:**
   - **Ch `20`:** `res.dtilde` linearly interpolated (`0.0` when no LexicMap evidence is supplied).
   - **Ch `21`:** `np.tanh(res.e_lex)` linearly interpolated (`0.0` when no RNA-seq split endpoints).

> **Coordinate frame alignment note:** In `ContigResult`, `p_frame[:, 0..2]` (`+` strand) and `p_frame[:, 3..5]` (`-` strand) are indexed by **contig-global** reading frame `x % 3`. To make the refiner translation-invariant to chunk start offset `lo`, rotate the 3 `+` frame channels `[8, 9, 10]` and the 3 `-` frame channels `[11, 12, 13]` by `lo % 3` so that within any chunk `[lo, hi)`, channel `8 + f` is always **chunk-local** frame `f = (x - lo) % 3` on `+` and channel `11 + f` is chunk-local frame `f = (x - lo) % 3` on `-`.

### 2.3 Clade-Conditioning Vector (`cond`, shape `[B, D_cond = 12]`)
Constructed once per contig by `build_contig_cond(res: ContigResult, seq: str, gcode: int = 1) -> np.ndarray`:
- **Dims `0..3` (4 dims):** Domain one-hot / probability vector `[Eukaryota, Bacteria, Archaea, Viruses]` derived from `res.taxon.names.get("domain")` weighted by `res.taxon.path_conf.get("domain", 0.0)` (if unclassified, `[0.25, 0.25, 0.25, 0.25]`).
- **Dim `4`:** Contig mean `p_coding` (`float(res.p_coding.mean())`).
- **Dim `5`:** Contig coding-run density (fraction of bins with `p_coding >= 0.5`).
- **Dim `6`:** Contig frame sharpness (mean over coding bins of `res.p_frame[:, :6].max(axis=1)`).
- **Dim `7`:** Contig boundary impulse rate (mean of `np.tanh(res.b_tracks[:, 2:4].sum(axis=1))` over coding bins — high in intron-rich eukaryotes, near zero in intron-poor clades).
- **Dim `8`:** Contig GC fraction in `[0, 1]`.
- **Dim `9`:** Genetic code 4 flag (`1.0` if `gcode == 4` else `0.0`).
- **Dim `10`:** Has LexicMap RNA-seq evidence flag (`1.0` if `res.dtilde.max() > 0` else `0.0`).
- **Dim `11`:** `min(1.0, math.log10(max(res.length, 100)) / 7.0)` (log contig length scale).

### 2.4 Per-Base Output Heads
The refiner outputs two tensors for each chunk `[B, L]`:
1. **`state_logits`: `[B, 7, L]`** — 7 mutually exclusive per-base genomic state classes:
   - `0`: `INTERGENIC_OR_UTR` (non-CDS, outside introns)
   - `1`: `CDS_PLUS` (exonic CDS base on `+` strand)
   - `2`: `INTRON_PLUS` (intronic base between `+` strand CDS exons)
   - `3`: `CDS_MINUS` (exonic CDS base on `-` strand)
   - `4`: `INTRON_MINUS` (intronic base between `-` strand CDS exons)
   - `5`: `RNA_NONCODING` (reserved for tRNA/rRNA/ncRNA; supervised as `0` when GFF lacks ncRNA)
   - `6`: `BOTH_STRAND_CDS` (rare overlapping opposite-strand CDS; optional auxiliary class)
2. **`site_logits`: `[B, 8, L]`** — 8 binary sigmoid boundary-site heads (supervised at exact single-base transition coordinates):
   - `0`: `START_PLUS` — first base `A` of `+` strand `ATG` (`x = cds_start`)
   - `1`: `STOP_PLUS` — first base of `+` strand stop codon (`x = cds_end - 3`)
   - `2`: `DONOR_PLUS` — first intronic base `G` of `+` strand donor (`x = intron_start`)
   - `3`: `ACCEPTOR_PLUS` — last intronic base `G` of `+` strand acceptor (`x = intron_end - 1`)
   - `4`: `START_MINUS` — genomic coordinate `x = cds_end - 1` (the `T` on `+` corresponding to transcript-5' `A` of `ATG` on `-`)
   - `5`: `STOP_MINUS` — genomic coordinate `x = cds_start + 2` (first base of stop codon in `-` transcript order)
   - `6`: `DONOR_MINUS` — genomic coordinate `x = intron_end - 1` (first intronic base on `-` transcript)
   - `7`: `ACCEPTOR_MINUS` — genomic coordinate `x = intron_start` (last intronic base on `-` transcript)

---

## 3. Model Architecture (`OmniPhylRefiner` in `src/omniphyl/refiner.py`)

Keep the architecture compact (~1.5M parameters, pure PyTorch `nn.Module`, zero external dependencies):

1. **Stem:**
   - `nn.Conv1d(22, 96, kernel_size=7, padding=3)` + `nn.GroupNorm(8, 96)` + `nn.GELU()`.
2. **FiLM-Conditioned Dilated Residual Conv Stack (6 blocks, `channels = 96`):**
   - Dilations: `(1, 2, 4, 8, 16, 32)` with `kernel_size = 7` (receptive field `~380 bp` before the recurrent layer).
   - Each `RefinerResBlock(channels=96, dilation=d, cond_dim=12)`:
     - Depthwise-separable or standard `Conv1d(96, 96, kernel_size=7, padding=3*d, dilation=d)` -> `GroupNorm(8, 96)`
     - FiLM projection `Linear(12, 192)` producing `(gamma, beta)` applied as `(1 + gamma[:, :, None]) * h + beta[:, :, None]`
     - `GELU()` -> `Dropout(0.1)` -> `Conv1d(96, 96, kernel_size=1)` -> residual add.
3. **Long-Range Context Layer:**
   - Downsample by `4x` via `AvgPool1d(kernel_size=4, stride=4)` (`[B, 96, L//4]`), run a 1-layer bidirectional GRU `nn.GRU(input_size=96, hidden_size=96, num_layers=1, batch_first=True, bidirectional=True)` (`[B, L//4, 192]` -> `Linear(192, 96)`), linearly upsample back to length `L` via `F.interpolate(..., size=L, mode="linear", align_corners=False)`, and add residually to the conv features.
   - *Why downsample 4x before the BiGRU:* Running a BiGRU at `L//4 = 2048` steps instead of `8192` is **4x faster and 4x lighter in VRAM** while still propagating exon/intron phase and gene-level context across the entire 8 kb window; the un-pooled residual conv stack preserves exact single-nucleotide position for the output heads.
4. **Output Heads:**
   - `head_state = nn.Conv1d(96, 7, kernel_size=1)` -> `state_logits [B, 7, L]`
   - `head_site = nn.Conv1d(96, 8, kernel_size=3, padding=1)` -> `site_logits [B, 8, L]`

---

## 4. Constrained Base-Exact Decoder (`decode_with_refiner` in `src/omniphyl/refiner.py`)

After running `OmniPhylRefiner` across a contig (blending overlapping 8,192 bp chunks with raised-cosine weights), we have contig-wide probabilities:
- `p_state`: `float32 [L, 7]` (`softmax` over axis 0)
- `p_site`: `float32 [L, 8]` (`sigmoid`)

Decode each strand (`+` and `-`) independently into valid, phase-consistent gene dicts matching the exact return schema of `_decode_genes` (`[{"start": int, "end": int, "strand": str, "frame": int, "introns": [(d, acc), ...], "score": float, "domains": []}, ...]`) so downstream GFF3 writing, domain assignment, and reporting in `annotate.py` work unchanged.

### 4.1 Strand Normalization
To avoid duplicating `+` and `-` decoder logic (**single codepath for both strands**):
- For `+` strand:
  - `s_seq = seq.upper()`
  - `p_cds = p_state[:, 1]`, `p_int = p_state[:, 2]`, `p_bg = p_state[:, 0] + p_state[:, 3] + p_state[:, 4] + p_state[:, 5]`
  - `p_start = p_site[:, 0]`, `p_stop = p_site[:, 1]`, `p_don = p_site[:, 2]`, `p_acc = p_site[:, 3]`
- For `-` strand:
  - Reverse-complement the sequence `s_seq = revcomp(seq.upper())` and reverse the 1D arrays `[::-1]` using the `-` channels (`p_cds = p_state[::-1, 3]`, `p_int = p_state[::-1, 4]`, `p_start = p_site[::-1, 4]`, `p_stop = p_site[::-1, 5]`, `p_don = p_site[::-1, 6]`, `p_acc = p_site[::-1, 7]`).
  - Run the exact same forward-strand decoder on `s_seq`, then map any emitted transcript coordinates `[a, b)` back to genomic `[L - b, L - a)` and introns `(d, acc)` to `(L - acc, L - d)`.

### 4.2 Semi-Markov / Constrained Exon-Chain DP on Coding Loci
1. **Identify active locus intervals `[lo, hi)` on the strand:**
   - Smooth `p_cds + p_int` with a 100 bp box filter; take contiguous regions where smoothed `p_cds + p_int >= 0.20` and `max(p_cds) >= coding_threshold`, padded by `300 bp` on each side.
   - If `len(locus) <= 100`, skip.
2. **Candidate splice junctions inside `[lo, hi)` (when `euk=True`):**
   - Scan `x in [lo + 3, hi - min_intron)` where `s_seq[x:x+2] in ("GT", "GC")` and `p_don[x] >= site_threshold` (default `0.05`, or hinted by `intron_hints`).
   - Scan `y in [lo + min_intron, hi - 3)` where `s_seq[y-2:y] == "AG"` and `p_acc[y-1] >= site_threshold` (or hinted).
   - Pair valid `(d, acc)` with `min_intron <= acc - d <= max_intron` and score each candidate intron by:
     `intron_ll(d, acc) = log(p_don[d]) + log(p_acc[acc - 1]) + mean_log(p_int[d:acc]) - mean_log(p_cds[d:acc])` (+ `+2.0` bonus if `(d, acc)` matches an RNA-seq `intron_hint` within 6 bp).
   - Keep the top-`K` (`K = 24`) highest-scoring candidate introns `(d, acc)` with `intron_ll > -3.0`.
3. **Top-K intron-subset evaluation + exact ORF scoring:**
   - Using the existing `_intron_sets` DAG DP (or phase-tracked exon graph over the top-`K` introns), evaluate the empty intron set `()` plus top-scoring non-overlapping intron chains.
   - On each spliced sequence `(spliced_seq, pos_map)`, scan canonical `ATG -> stop` ORFs (`min_cds_len` floor) and score each ORF by combining:
     - Exonic CDS log-probability: `mean(log(p_cds[exonic]) - log(p_bg[exonic])) * log1p(cds_len)`
     - Start/stop boundary site log-odds at the exact start base `a` and stop base `b - 3`: `log(p_start[a]) + log(p_stop[b - 3])`
     - Sum of `intron_ll(d, acc)` for all excised introns
     - Frame consistency with `res.p_frame` via `_exon_frames` (from `annotate.py:673`).
   - Arbitrate overlapping candidates with `_nonoverlap_select` (`annotate.py:644`).

---

## 5. Training Pipeline (`scripts/train_refiner.py`)

### 5.1 Training Data Format & Two-Stage Caching
Training the refiner does **not** backpropagate through Stage-1 `OmniPhyl`. Instead, `scripts/train_refiner.py` accepts a directory or manifest of pairs `(fasta_path, gff3_path)` (e.g., from the v6 RefSeq/VEuPathDB corpus or a clade-balanced subset of 50–200 assemblies across Bacteria, Archaea, Fungi, Protozoa/Apicomplexa, Plants, Invertebrates, Vertebrates, Viruses):

1. **Stage-1 Feature Cache (`--cache-dir`):**
   - For each assembly, if `<cache_dir>/<contig_id>.res.npz` does not exist, run frozen `OmniPhylPredictor(args.stage1_model)` once (`window=500, stride=100, bin_width=50`), call `_fuse_contig` + `_pooled_call`, and save a compressed `.npz` containing `p_coding, p_boundary, p_frame, b_tracks, cond` (16-bit float `float16` — a 1 Mb contig takes `< 150 KB` on disk).
   - Subsequent epochs load `.npz` + FASTA slices directly in `< 1 ms` per 8 kb window without touching the Stage-1 model.
2. **Ground-Truth Label Construction from GFF3 (`build_refiner_labels(gff_path, seq_len)`):**
   - Reuse `parse_cds` from `/home/danielle/Documents/veg/omniphyl/scripts/eval_genes.py` (lines 55–74) to extract 0-based half-open CDS intervals per transcript `(tx_id, strand, [(s_0, e_0), ...])`.
   - For each transcript on `+` (and symmetrically on `-`):
     - Mark exonic bases `[s_i, e_i)` as `CDS_PLUS` (`1`) or `CDS_MINUS` (`3`).
     - Mark intronic bases `[e_i, s_{i+1})` between consecutive CDS segments of the same transcript as `INTRON_PLUS` (`2`) or `INTRON_MINUS` (`4`).
     - Mark exact boundary sites in `site_targets [8, L]`:
       - `START_PLUS` at `s_0` (first CDS start on `+`)
       - `STOP_PLUS` at `e_last - 3` (last CDS end minus 3 on `+`)
       - `DONOR_PLUS` at `e_i` for each internal exon end `i < n_exons - 1`
       - `ACCEPTOR_PLUS` at `s_{i+1} - 1` for each internal exon start
       - Symmetrical positions for `-` strand transcripts (`START_MINUS = e_last - 1`, `STOP_MINUS = s_0 + 2`, `DONOR_MINUS = s_{i+1} - 1`, `ACCEPTOR_MINUS = e_i`).
   - Apply label smoothing (`sigma = 1.5 bp` Gaussian `[0.25, 0.5, 1.0, 0.5, 0.25]`) on `site_targets` or focal BCE loss so 1 bp jitter does not destabilize gradients.

### 5.2 Loss Function & Optimization
- **State loss (`L_state`):** Class-weighted cross-entropy (`F.cross_entropy`) on `state_logits [B, 7, L]` with weights `[1.0, 1.5, 2.5, 1.5, 2.5, 1.0, 2.0]` (up-weighting introns so intron-poor training mixes do not collapse intron recall).
- **Site loss (`L_site`):** Focal binary cross-entropy with logits (`pos_weight = 50.0`, `gamma = 2.0`) on `site_logits [B, 8, L]`, masked to coding/intronic loci (`±500 bp` around true genes) so unannotated pseudogenes far in intergenic space are not over-penalized.
- **Phase-continuity auxiliary loss (`L_phase`):** On true introns `[d, acc)`, penalize `1 - p_don[d] * p_acc[acc - 1]` to encourage joint firing of paired donor and acceptor sites.
- **Total loss:** `L = L_state + 2.0 * L_site + 0.5 * L_phase`.
- **Augmentations during training:**
  - Random reverse-complement of the entire 8,192 bp chunk (50% probability): swap `A<->T`, `C<->G`, reverse sequence axis, swap `+` and `-` frame channels (`8..10 <-> 11..13`), swap `+` and `-` state labels (`1<->3`, `2<->4`), and swap `+` and `-` site labels (`0..3 <-> 4..7`).
  - Random LexicMap evidence dropout (50% probability when training with evidence): zero channels `20..21` and re-compute `cscore` without `dtilde` so the refiner works both with and without RNA-seq evidence.
- **Checkpoint format (`refiner.pt`):**
  ```python
  torch.save({
      "model": refiner.state_dict(),
      "config": {"c_in": 22, "cond_dim": 12, "hidden": 96, "n_states": 7, "n_sites": 8},
      "epoch": epoch,
      "val_metrics": val_metrics,
  }, out_path)
  ```
  Also support bundling `"refiner"` directly inside an OmniPhyl `.pt` checkpoint dict so a single `.pt` file can carry both Stage 1 and Stage 2 weights.

---

## 6. Inference & CLI Integration (`annotate.py` & `cli.py`)

1. **In `/home/danielle/Documents/veg/omniphyl/src/omniphyl/cli.py` (`_annotate_parser`, around line 76):**
   Add:
   ```python
   dec.add_argument("--refiner", type=Path, default=None,
                    help="optional Stage-2 neural boundary refiner checkpoint "
                         "(.pt); auto-used if bundled inside --model, unless "
                         "--no-refiner is set")
   dec.add_argument("--no-refiner", action="store_true",
                    help="force heuristic Stage-2 ORF/splice decoder even if "
                         "--model bundles a refiner")
   ```
2. **In `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py` (`run_annotate`, lines 81–264):**
   - Accept `refiner_path: Path | None = None, no_refiner: bool = False`.
   - Check if `refiner_path` is provided or `"refiner"` key is present in `pred.ckpt` (when `not no_refiner`).
   - If a refiner is loaded (`refiner = load_refiner(refiner_path or pred.ckpt, device=pred.device)`):
     - In Step 4c (line 234), call `res.genes = decode_with_refiner(refiner, seq, res, stops=stops, euk=euk and not prok, gcode=gcode, w_ev=w_ev, min_cds_len=min_cds_len, coding_threshold=coding_threshold, min_intron=min_intron, max_intron=max_intron, intron_hints=...)`.
   - Otherwise, fall back to the existing `_decode_genes(...)` heuristic decoder with zero change in default behavior.

---

## 7. Verification & Acceptance Criteria

### Step 1 — Unit Tests (`/home/danielle/Documents/veg/omniphyl/tests/test_inference.py`)
Add a `test_refiner()` function in `tests/test_inference.py` verifying:
1. `build_refiner_inputs` produces shape `(22, L)` with finite values, properly rotated chunk-local frame channels for arbitrary `lo % 3`, and clamping at contig boundaries.
2. `build_contig_cond` produces shape `(12,)` with finite values in `[0, 1]`.
3. `OmniPhylRefiner` forward pass on a batch `[2, 22, 1024]` + `cond [2, 12]` returns `state_logits [2, 7, 1024]` and `site_logits [2, 8, 1024]` and parameter count is between `0.5M` and `3.0M`.
4. `build_refiner_labels` on a synthetic 2-exon `+` gene and 2-exon `-` gene marks exact start, stop, donor, and acceptor base coordinates and satisfies the reverse-complement equivariance test (inverting the sequence + GFF coordinates produces the exact reverse-swapped label tensors).
5. `decode_with_refiner` given synthetic oracle `state_logits` / `site_logits` on a 2-exon eukaryotic gene recovers the exact 2-exon gene model (`start, end, strand, introns`) and `run_annotate(..., refiner_path=...)` writes a valid `.gff3`.

### Step 2 — Overfit / Smoke Test of `scripts/train_refiner.py`
Run 5 epochs of `train_refiner.py` on the cached *P. falciparum* `NC_004325.2` data (`/home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl` + `NC_004325.2.fasta` + `NC_004325.2.gff3`, splitting first 80% of contig for train and last 20% for val) and evaluate with `scripts/eval_genes.py`:
- Verify loss decreases monotonically across epochs.
- Verify held-out validation nucleotide F1 and exon F1 exceed the heuristic Stage-2 baseline on the same split.
