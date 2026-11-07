# OmniPhyl — Inference Package & CLI Implementation Plan

*Self-contained engineering specification for adding an installable `omniphyl` Python package (`src/omniphyl/`) and CLI (`omniphyl annotate`, `omniphyl profile`) to `/home/danielle/Documents/veg/omniphyl`. Written so an agent with zero prior session context can implement, test, and verify the full stack without breaking existing training or Kaggle pipelines.*

---

## 0. Context, Constraints & Non-Breakage Invariants

### Repository & Environment
- **Repository root:** `/home/danielle/Documents/veg/omniphyl`
- **Conda environment:** `omniphyl` (activate via `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate omniphyl`)
- **Runtime dependencies:** Python `>=3.10`, `torch`, `numpy`, and Python standard library only (`argparse`, `csv`, `gzip`, `json`, `math`, `pathlib`, `sys`, `zlib`). Do **not** add heavy external dependencies (no `biopython`, `scipy`, `pandas`, `scikit-learn`, or `pyarrow`).

### Hard Non-Breakage Invariants
1. **Do NOT move or rename `scripts/`, `remote/`, or `kaggle/`:**
   - `remote/run_pilot.sh` and `kaggle/*/kernel.py` invoke `scripts/fetch_refseq_corpus.py`, `scripts/screen_contaminants.py`, `scripts/train_omniphyl.py`, and `scripts/eval_pilot.py` by relative path.
   - `tests/test_smoke.py` imports symbols directly from `scripts/train_omniphyl.py`, `scripts/eval_pilot.py`, `scripts/fetch_refseq_corpus.py`, and `scripts/screen_contaminants.py`.
2. **Single Source of Truth via Re-Export:**
   - Core encoding, model architecture, checkpoint loading, and tree reconciliation move into `src/omniphyl/` (`encode.py`, `model.py`) and are **re-exported** in `scripts/train_omniphyl.py` and `scripts/eval_pilot.py` (with a `sys.path` fallback to `Path(__file__).resolve().parents[1] / "src"` so running `python3 scripts/train_omniphyl.py` or `python3 tests/test_smoke.py` works even without `pip install -e .`).
3. **Existing Smoke Suite Must Stay 100% Green:**
   - Running `python3 -m pytest tests/test_smoke.py` (or `python3 tests/test_smoke.py`) must pass all existing tests before and after every phase.

---

## 1. Package Layout & `pyproject.toml`

Create a standard `src/`-layout package inside the existing repository:

```text
/home/danielle/Documents/veg/omniphyl/
├── pyproject.toml
├── src/
│   └── omniphyl/
│       ├── __init__.py          # __version__ = "0.1.0", public API exports
│       ├── __main__.py          # python -m omniphyl entrypoint
│       ├── encode.py            # Sequence encoders + virtual-fragment k-mer pooling
│       ├── model.py             # OmniPhyl nn.Module, checkpoint loader, batched runner, tree reconciler
│       ├── io.py                # Streaming FASTA/FASTQ(.gz) reader, LexicMap BED/bedGraph/TSV reader, GFF3/bedGraph writers
│       ├── annotate.py          # Use case 1: sliding-window contig annotation + LexicMap evidence + Stage-2 decoder
│       ├── profile.py           # Use case 2: reference-free metagenomic taxonomic + Pfam profiling (contigs, pre-contigs, reads)
│       └── cli.py               # `omniphyl` CLI (`annotate`, `profile`)
├── scripts/                     # Existing scripts preserved at exact paths
├── remote/
├── kaggle/
└── tests/
    ├── test_smoke.py            # Existing unit + e2e smoke suite
    └── test_inference.py        # New unit + e2e tests for `omniphyl annotate` and `omniphyl profile`
```

### `pyproject.toml` Specification
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "omniphyl"
version = "0.1.0"
description = "Reference-free taxonomic & functional profiling and clade-agnostic genome annotation with multi-scale FCGR networks"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.23",
    "torch>=2.0",
]

[project.scripts]
omniphyl = "omniphyl.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

---

## 2. Exact Checkpoint & Model Contract (`src/omniphyl/encode.py` & `src/omniphyl/model.py`)

Every production checkpoint (`best.pt` / `last.pt`, e.g., from `kaggle/train-v5-w32-e15/kernel.py` and `kaggle/train-v5-w64-e15/kernel.py`) is a `torch.save` dictionary with the following exact schema (defined in `scripts/train_omniphyl.py:1668-1685` and loaded in `scripts/eval_pilot.py:396-459`):

### 2.1 Checkpoint Dictionary Keys
| Key | Type | Description |
|---|---|---|
| `"model"` | `dict[str, Tensor]` | `OmniPhyl` `state_dict` |
| `"vocab"` | `dict[str, dict[str, int]]` | Per-rank map `taxon_name -> 0-based vocab index`. Note: in model logits, **class `0` is the `<unknown>` mask class**, and `vocab_idx` maps to **shifted class ID `vocab_idx + 1`**. |
| `"k"` | `int` | Finest FCGR k-mer size (default `6`, giving `side = 1 << k = 64`) |
| `"targets"` | `list[str]` | Ordered target names, typically `RANKS + CONT_TARGETS` where `RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]` and `CONT_TARGETS = ["coding_frac", "boundary", "offset"]` |
| `"args"` | `dict[str, Any]` | Training hyperparameters needed to reconstruct `OmniPhyl`: `width` (`32` or `64`), `backbone` (`"msfcgr"`, `"resnet"`, `"vit"`, `"seq"`), `pos_channels` (`bool`), `dropout` (`float`), `stem_stride` (`int`), `head` (`"flat"` or `"tree"`), `aux` (comma-separated string, e.g. `"hash,aa"`), `aa_mode` (`"plain"`, `"bottleneck"`, `"gated"`), `aa_drop` (`float`), `region_head` (`bool`), `struct_head` (`bool`), `annot_ctx` (`bool`), `prior_tau` (`float`), `flat_prior_tau` (`float`) |
| `"tree"` | `dict[str, dict[str, list]]` | `"parent_of"`: `rank -> list[int]` of length `len(vocab[rank]) + 1` mapping shifted class ID `c` at `rank` to its shifted parent class ID at `rank - 1` (`0` = unknown/orphan); `"log_prior"`: sibling-conditional log-priors |
| `"flat_log_prior"` | `dict[str, list[float]]` | Per-rank unconditional log-priors of length `len(vocab[rank]) + 1` (`lp[0] = -20.0`). **Critical:** when `head == "flat"` and `flat_prior_tau > 0`, raw logits were trained with `logits[t] + flat_prior_tau * flat_log_prior[t]`. At inference, adding `flat_tau * flat_log_prior[t]` restores the training prior distribution (MAP), whereas `flat_tau = 0.0` yields prior-balanced likelihood ratios. Default `load_checkpoint` must apply `flat_tau = float(train_args.get("flat_prior_tau", 0.0))` unless overridden by `--flat-prior-tau`. |
| `"domain_vocab"` | `dict[str, int]` (optional) | Map `pfam_or_ipr_id -> 0..K-1` column index in `logits["domains"]` |

### 2.2 Input Branch Contract (`encode.py`)
Given a DNA string `seq` and checkpoint config:
1. **`"x"` (primary image/sequence tensor):**
   - If `backbone == "seq"`: `seq_encode(seq, seq_len)` $\to$ `float32 [4, seq_len]`.
   - If `backbone == "msfcgr"`: `fcgr_encode_ms(seq, ms_luts, ms_canons, ks=(4,5,6), pos=pos_channels)` $\to$ `float32 [3 + 2*pos, 64, 64]`.
   - Else: `fcgr_encode(seq, k, lut, pos=pos_channels)` $\to$ `float32 [1 + 2*pos, 2^k, 2^k]`.
2. **`"hash"` (when `"hash" in aux`):**
   - `hash_encode(seq, hash_canon)` $\to$ `int64 [512]` (canonical 11-mers hashed into `2^16 = 65536` buckets via multiplicative hash `* 2654435761 % 65536`, padded with `HASH_PAD = 65536`).
   - **Window-Length Cap Invariant:** Because `HASH_LEN = 512` truncates after the first 512 valid 11-mers ($\approx 522\text{ bp}$ of unambiguous sequence), any single sequence chunk passed into `FeatureEncoder` (whether a sliding window or an `'N'`-joined virtual pre-contig fragment) **must not exceed `500 bp`** of non-`N` bases so the hash branch sees the full chunk uniformly.
3. **`"aa"` (when `"aa" in aux`):**
   - `aa_encode(seq, extra=(aa_mode == "gated"))` $\to$ `float32 [9261]` (or `[9279]` when `extra=True`).
   - 6 reading frames in fixed order `(F0, F1, F2, R0, R1, R2)` where `F0..F2` are forward offsets `0, 1, 2` from the window start and `R0..R2` are reverse-complement (`3 - codes[::-1]`) offsets `0, 1, 2` from the window end (`frag_end`).
4. **`"aa_frame"` (when `struct_head == True`):**
   - `aa_frame_encode(seq)` $\to$ `float32 [144]` (`6 frames x 24` features: 20 AA + 1 stop frequency, `stop_density`, `longest_orf_frac`, `3rd-codon-pos GC`).

### 2.3 Output Heads Contract (`model.py`)
A forward pass `z, logits = model(batch_inputs)` returns latent `z: [B, D]` and `logits: dict[str, Tensor]`:
- **Taxonomy ranks (`t in RANKS`):** `logits[t]` has shape `[B, len(vocab[t]) + 1]`.
  - **Two-Stage Softmax (OOD Class `0` vs. Known Classes `1..C`):** During training (`scripts/train_omniphyl.py:1391-1402`), flat heads are trained with `logits_raw[t] + train_flat_tau * flat_log_prior[t]` (where `flat_log_prior[0] = -20.0`), so the network's raw bias for class `0` was calibrated against `train_flat_tau * (-20.0)`. Therefore:
    1. Always compute `p_ood[t] = torch.softmax(logits_raw[t] + train_flat_tau * flat_lp[t], dim=1)[:, 0]` using the checkpoint's **training** `train_flat_tau = float(train_args.get("flat_prior_tau", 0.0))` (even if the user passes `--flat-prior-tau 0.0` for known-taxon relative abundance).
    2. For known-class scoring (`1..C`), compute `logits_adj[t] = logits_raw[t] + user_flat_tau * flat_lp[t]`, mask `logits_adj[t][:, 0] = -1e9`, and compute `logp[t] = torch.log_softmax(logits_adj[t], dim=1)`.
- **`coding_frac`:** `torch.sigmoid(logits["coding_frac"][:, 0])` $\in [0, 1]$.
- **`boundary`:** `torch.sigmoid(logits["boundary"][:, 0])` $\in [0, 1]$ (probability that $0 < \text{coding\_frac} < 1$, i.e. a CDS edge lies inside the window).
- **`offset`:** `torch.sigmoid(logits["offset"][:, 0])` $\in [0, 1]$ (normalized window-relative coordinate of the interior CDS edge nearest the window midpoint; genomic coordinate is `win_start + offset * win_len`).
- **`region_fracs` (if `region_head`):** `logits["region_fracs"].softmax(dim=1)` $\to$ `[B, 4]` probabilities `(coding, intron, rna, intergenic)`.
- **`active_frame` (if `struct_head`):** `logits["active_frame"].softmax(dim=1)` $\to$ `[B, 8]` probabilities:
  - `0, 1, 2`: forward strand reading frames (`F0, F1, F2` relative to window start `frag_start`, defined in `scripts/fetch_refseq_corpus.py:595-605`: `(codon0 - frag_start) % 3`).
  - `3, 4, 5`: minus strand reading frames (`R0, R1, R2` relative to window end `frag_end`: `3 + (frag_end - codon0) % 3`).
  - `6`: multi-frame / overlapping CDS in window.
  - `7`: non-coding (`coding_frac < 0.1`).
- **`struct_flags` (if `struct_head`):** `torch.sigmoid(logits["struct_flags"])` $\to$ `[B, 4]` independent probabilities `(has_start, has_stop, has_donor, has_acceptor)` strictly inside `(frag_start, frag_end)`. Note that `struct_flags` are strand-agnostic in the checkpoint; strand comes from `active_frame` (`0..2` vs `3..5`).
- **`domains` (if `domain_k > 0`):** `torch.sigmoid(logits["domains"])` $\to$ `[B, K]` multi-label probabilities for each Pfam/InterPro entry in `domain_vocab`.
- **Graceful Degradation for Ablated Checkpoints:** If a loaded checkpoint was trained without `region_head`, `struct_head`, or `domains`, `OmniPhylPredictor` must emit a clear startup warning stating which heads are absent and fall back cleanly (e.g., using `coding_frac` + 6-frame ORF scanning when `struct_head` is absent, and emitting empty domain tables when `domains` is absent).

### 2.4 NCBI Cross-Rank Tree Reconciliation (`reconcile_taxonomy`)
Even when `OmniPhyl` is trained with `head == "flat"` (independent linear heads per rank), `scripts/train_omniphyl.py:1676-1680` saves the NCBI taxonomy parent map `ckpt["tree"]["parent_of"]` into every checkpoint. Without reconciliation, a flat model can predict `Family X` at the family head and a `Genus B` (whose true NCBI family is `Family Y`) at the genus head.

Port `ancestor_chains` and `path_scores` from `scripts/eval_pilot.py:232-273` into `src/omniphyl/model.py` and enforce **strict tree-consistent lineage projection**:
- Given per-rank log-probabilities `logp[r]` (`[N, C_r]`, where `logp` is already pooled across $M$ windows of a contig, whole genome, or pre-contig read cluster):
  1. **Path-score rescoring:** For each rank $r \in \text{RANKS}$, compute `score[r] = path_scores(logp, chains[r], cls_targets)` — the mean log-probability along each class $c$'s full NCBI ancestor chain $(\text{domain} \to \dots \to r)$, with class `0` (`<unknown>`) masked to `-inf`.
  2. **Per-rank candidate & ancestor veto:** Let `c_hat[r] = score[r].argmax(dim=1)` and `path_conf[r] = exp(score[r].max(dim=1).values)`. For `c_hat[r]`, compute `min_anc_prob[r]` = the minimum marginal probability $\min_{r' < r} \exp(\text{logp}[r'][\text{ancestor}_{r'}(c\_hat[r])])$ across all shallower NCBI ancestors in `chains[r][c_hat[r]]`.
  3. **Deepest Confident Rank Selection ($r^*$):** A rank $r$ passes the gate iff `path_conf[r] >= tau_conf` (default `0.5`) AND `min_anc_prob[r] >= tau_anc` (default `0.2`). Note: do **not** require shallower ranks $r' < r$ to pass their own independent `argmax` gate, as long as the specific ancestor lineage `chains[r][c_hat[r]]` satisfies `min_anc_prob[r] >= tau_anc` and has valid (non-zero) ancestor IDs at every shallower rank. Let $r^*$ be the deepest rank that passes.
  4. **Strict NCBI Ancestor Projection (Zero Cross-Rank Contradictions):**
     - Lock the anchor prediction to $c^* = \text{c\_hat}[r^*]$ at rank $r^*$.
     - **Populate all shallower ranks $r < r^*$ deterministically from `chains[r*][c*]`** (the true NCBI parent/ancestor IDs of $c^*$), NOT from independent `argmax` calls at shallower ranks. This mathematically guarantees that the emitted `(domain, phylum, class, order, family, genus, species)` lineage is a valid path in the NCBI taxonomy tree and that child abundances at rank $r$ sum cleanly into their parent taxon at rank $r - 1$.
     - Populate all deeper ranks $r > r^*$ as abstentions: `unclassified_<taxon_name_at_r*>`, and report both `1 - path_conf[r]` and `p_ood[r]` as novelty/OOD indicators.

---

## 3. Use Case 1: Genome Annotation (`omniphyl annotate`)

### 3.1 Goal & Scope
Given an input FASTA of genomic contigs/scaffolds (and optionally a Logan/LexicMap `.bed`, `.bedGraph`, or raw `.tsv` alignment file), slide a $W=500\text{ bp}$ window at stride $S=100\text{ bp}$, aggregate the overlapping window predictions onto genomic coordinates, integrate optional LexicMap conservation/transcription evidence, and run a Stage-2 base-exact boundary resolver + reading-frame grammar decoder to emit:
1. **`<prefix>.gff3`** — Valid GFF3 with `gene`, `mRNA`, `exon`, `CDS` (and `ncRNA_gene` / `ncRNA` when `rna` dominates), including predicted Pfam domains, taxonomy context, confidence, and LexicMap support attributes.
2. **`<prefix>.report.md` and `<prefix>.report.json`** — Human-readable Markdown + machine-readable JSON **whole-genome summary report** (also printed to stdout):
   - **Whole-genome taxonomic identity:** Pools `logp` across all windows of the entire assembly (`M` = total windows) and runs `reconcile_taxonomy()` to report the predicted species (or deepest confident rank), full 7-rank NCBI lineage with per-rank confidence, and novelty/OOD status (`p_ood` and abstention depth).
   - **Contaminant / Chimeric Contig Flagging:** Compares each individual contig's confident phylum/class call against the whole-genome consensus and lists any discordant contigs (requiringcontig length $\ge 1500\text{ bp}$ or $M \ge 3$ windows and `path_conf >= 0.6` before flagging to avoid single-window noise on tiny fragments).
   - **Assembly & Composition Stats:** Total bp, contig count, N50, GC%, model-estimated genome-wide composition breakdown (% coding, % intron, % RNA, % intergenic), and inferred genetic code (`Code 1` vs `Code 4` with observed in-frame `TAA`/`TAG`/`TGA` stop ratios).
   - **Annotated Feature Counts:** Total genes, protein-coding genes (`CDS`), single-exon vs. multi-exon gene counts, total exons, total introns (with mean & median intron length), ncRNA features, mean/median CDS length (bp and aa), and fraction of genes with LexicMap SRA support.
   - **Functional Profile:** Top Pfam domains detected across the predicted proteome with gene counts.
3. **`<prefix>.coding.bedGraph`** — Per-stride-bin coding probability track (`0.000`–`1.000`) for UCSC Genome Browser inspection.
4. **`<prefix>.boundary.bedGraph`** — Per-stride-bin boundary probability track.
5. **`<prefix>.contig_summary.tsv`** — Per-contig taxonomy call (deepest confident rank + full lineage + `chimeric_flag`), inferred genetic code, coding density, gene count, and top Pfam domains.

### 3.2 Step-by-Step Algorithm (`src/omniphyl/annotate.py`)

#### Step 1: Strided Window Generation & Cross-Contig Batching
For each contig `seq` of length $L$:
- If $L < W$: evaluate a single window `[0, L)`.
- If $L \ge W$: generate windows starting at $a_w \in \{0, S, 2S, \dots\}$ plus a final window `[L - W, L)` if $(L - W) \bmod S \ne 0$, guaranteeing full coverage of both contig ends.
- Skip windows with $>50\%$ ambiguous (`N`) bases (assign neutral intergenic prior).
- **Cross-Contig Window Batching:** Draft assemblies often contain hundreds or thousands of short contigs ($1\text{–}4$ windows each). Do **not** call `predict_sequences()` once per small contig; accumulate valid windows across contigs into full batches of `--batch-size 256` (recording `(contig_id, a_w, win_len)`) before running `OmniPhylPredictor.predict_sequences()`, then scatter predictions back to their parent contigs.

#### Step 2: Raised-Cosine Window-to-Base Aggregation
Divide the contig into uniform bins of width $B = \gcd(S, 50)$ (for default $W=500, S=100$, bin width $B = 50\text{ bp}$, or base-level $B = 1$ on short contigs; $B=50\text{ bp}$ keeps memory tiny even on 50 Mb chromosomes: $10^6$ bins = 4 MB per track).
For a window $w = [a_w, a_w + W)$ covering bin midpoint $x \in [a_w, a_w + W)$, define the **raised-cosine (Hann) center weight**:
$$u_w(x) = 0.1 + 0.9 \cdot \sin^2\!\left(\pi \frac{x - a_w}{W}\right)$$
(The `0.1` floor ensures contig extremes `[0, S)` and `[L-S, L)` retain non-zero weight.)

Aggregate across all windows $w$ covering bin $x$:
1. **Region composition track** $P_{\text{reg}}(x) \in \Delta^3$ `(coding, intron, rna, intergenic)`:
   - If the checkpoint has `region_fracs`: weighted average $\sum_w u_w(x) P_{\text{reg}, w} / \sum_w u_w(x)$.
   - Fallback if checkpoint only has `coding_frac`: $P_{\text{coding}}(x) = \sum_w u_w(x) p_{\text{coding}, w} / \sum_w u_w(x)$ and $P_{\text{intergenic}}(x) = 1 - P_{\text{coding}}(x)$.
2. **Global Genomic Reading Frame Re-Indexing (Critical Coordinate Transform):**
   - In `OmniPhyl`, `active_frame` indices `0..2` (`F0..F2`) are relative to `a_w` (`(codon0 - a_w) % 3`), and `3..5` (`R0..R2`) are relative to `a_w + W` (`3 + (a_w + W - codon0) % 3`), where `codon0` is defined in `scripts/fetch_refseq_corpus.py:595-605`:
     - On the `+` strand, a codon occupying 0-based half-open `[c_start, c_start + 3)` has `codon0_fwd = c_start` (its $5'$ start coordinate), and `c_start % 3 == g_fwd`.
     - On the `-` strand, a codon occupying 0-based half-open `[c_start, c_end)` (where `c_end = c_start + 3`) has `codon0_rev = c_end` (its $5'$ start coordinate on the minus strand, i.e., its exclusive upper genomic coordinate!). Note that because `c_end - c_start == 3`, `c_end % 3 == c_start % 3`, so a minus-strand codon at genomic `[c_start, c_start + 3)` has global frame `g_rev = 3 + (c_end % 3) = 3 + (c_start % 3)`!
   - Because $a_w$ shifts by $S$ each stride ($100 \equiv 1 \pmod 3$) and $W = 500 \equiv 2 \pmod 3$, **window-local frame indices permute across strides!**
   - Convert every window's 6-frame probabilities into **contig-global frame index** $g \in \{0,1,2,3,4,5\}$ before pooling across overlapping windows:
     - For forward frames $f \in \{0, 1, 2\}$: `codon0 % 3 == (a_w + f) % 3` $\implies$ global forward frame is:
       $$g_{\text{fwd}}(f, a_w) = (a_w + f) \bmod 3 \in \{0, 1, 2\}$$
     - For reverse frames $f \in \{3, 4, 5\}$ (let $r = f - 3 \in \{0, 1, 2\}$): `(a_w + W - codon0) % 3 == r` $\implies$ `codon0 % 3 == (a_w + W - r) % 3` $\implies$ global reverse frame is:
       $$g_{\text{rev}}(f, a_w, W) = 3 + ((a_w + W - (f - 3)) \bmod 3) \in \{3, 4, 5\}$$
   - Now all overlapping windows vote in the **same contig-global 6-frame coordinate system** $P_{\text{frame}}(x, g)$ for $g \in \{0..5\}$, where any forward codon `[i, i+3)` has frame `i % 3` and any reverse codon `[i, i+3)` (with `i+3` as its minus-strand $5'$ edge) has frame `3 + ((i + 3) % 3) = 3 + (i % 3)`.
3. **Strand-Aware Boundary Coordinate Voting & Flag Tracks:**
   - Each window $w$ predicts a boundary probability $p_{\text{bnd}, w}$, an interior offset $\hat{o}_w \in [0, 1]$, and 4 structural flag probabilities $(p_{\text{start}, w}, p_{\text{stop}, w}, p_{\text{donor}, w}, p_{\text{acceptor}, w})$.
   - Let $p_{+, w} = \sum_{f=0}^2 P_{\text{frame}, w}(f)$ and $p_{-, w} = \sum_{f=3}^5 P_{\text{frame}, w}(f)$ be the window's forward and reverse coding-strand weights (normalized by $p_{+, w} + p_{-, w} + \epsilon$).
   - Window $w$ casts a Gaussian-smoothed impulse vote of mass $p_{\text{bnd}, w} \cdot p_{\text{flag}, w} \cdot p_{\pm, w}$ centered at genomic coordinate $\hat{x}_w = a_w + \hat{o}_w \cdot W$ with bandwidth $\sigma = 35\text{ bp}$, producing **strand-separated** boundary tracks $B_{\text{type}, +}(x)$ and $B_{\text{type}, -}(x)$ for $\text{type} \in \{\text{start}, \text{stop}, \text{donor}, \text{acceptor}\}$.
   - **Limitation note:** Because each window emits a single scalar `offset`, windows spanning micro-exons or micro-introns shorter than $\sim 100\text{ bp}$ contain two boundaries in the same window; Stage-2 motif scanning (§3.2 Step 4) must therefore search around both $B_{\text{type}, \pm}(x)$ peaks and any interior dips in $P_{\text{coding}}(x)$ or in-frame stop codons.
4. **Assembly-Wide Taxonomy & Mode Selection:**
   - Pool `logp[r]` across all windows on each contig (and across the whole assembly for the genome-wide prior) and run `reconcile_taxonomy()`.
   - Set **intron mode**: if the contig's reconciled domain is `Bacteria`, `Archaea`, or `Viruses` (or user passes `--kingdom prokaryote`), disable splice-site states (`donor`/`acceptor` prior $= 0$) so prokaryotic/viral ORFs are never artificially split by false introns. If `Eukaryota` (or `--kingdom eukaryote`), enable intron states with minimum intron length `35 bp` and maximum intron length `5000 bp` (configurable).
   - **Self-Calibrated Genetic Code Check (Assembly-Wide with Minimum Evidence Gate):** Pool high-confidence coding bins ($P_{\text{coding}}(x) > 0.8$) across the **entire assembly** (or per contig only if discordant/contaminant). Require at least `1,000` high-confidence in-frame codons (`>= 3,000 bp` of coding sequence) before testing for non-standard genetic codes; below that threshold, default to standard Code `1`. When $\ge 1,000$ codons are available, if `TGA` occurs at $>2\%$ of in-frame codons while `TAA`+`TAG` occur at $<0.2\%$ (classic Code-4 reassignment), set `inferred_gcode=4`; otherwise use Code `1` (overridable via `--genetic-code`).

#### Step 3: Optional Logan-LexicMap Evidence Integration (`--lexicmap-*`)
**Intended Use & Index Provenance (`--lexicmap-source rnaseq|genomic`, default `rnaseq`):**
- Document clearly in the CLI `--help`, README, and report output that **`--lexicmap-bed` / `--lexicmap-tsv` is intended to be generated by querying the target assembly against a Logan/SRA transcriptomic (`rnaseq` / cDNA / EST) LexicMap index**, rather than a whole-genome shotgun (`wgs` / `genomic`) index:
  - **When `--lexicmap-source rnaseq` (default & recommended):** HSPs represent spliced or unspliced transcript alignments from SRA RNA-seq unitigs/contigs. Coverage $d(x)$ reflects transcribed exonic/UTR support, and internal HSP boundaries (where a single RNA-seq unitig `sgenome|sseqid` splits across an intron, or where multiple RNA-seq HSPs terminate at a splice junction) provide candidate **splice-junction** (`donor`/`acceptor`) evidence $E_{\text{lex}}(x)$.
  - **Unstranded Unitig Orientation Warning:** Because Logan RNA-seq unitigs are assembled from short reads in canonical k-mer orientation, the BED/TSV `strand` column (`+` or `-`) indicates the *relative alignment orientation of the subject unitig*, **not** the biological transcription strand of the gene. Therefore, compute coverage $d(x)$ and splice-endpoint density $E_{\text{lex}}(x)$ **unstranded** across the genomic bin, and let OmniPhyl's `active_frame` (`0..2` vs `3..5`) and canonical splice dinucleotides (`GT..AG` vs `CT..AC`) determine the gene's coding strand. (Only require BED `strand` consistency when linking two colinear HSPs of the *same* `sgenome|sseqid` unitig across a split-intron gap.)
  - **UTR vs. CDS & ncRNA Discrimination:** Because RNA-seq unitigs cover $5'$ and $3'$ UTRs and non-coding RNAs (`rRNA`, `lncRNA`, `tRNA`) in addition to coding exons (`CDS`):
    1. If `region_head` is present and $P_{\text{rna}}(x) > P_{\text{coding}}(x)$, route the $\tilde{d}(x)$ boost to `ncRNA` rather than `CDS`.
    2. Gate the coding boost by multiplying $\tilde{d}(x)$ by $\min(1.0,\; P_{\text{coding}}(x) / 0.25)$ so high RNA-seq depth inside an untranslated $5'$ or $3'$ UTR (where the model sees low coding potential $P_{\text{coding}}(x) \ll 0.25$) does not force the ORF scanner to extend CDS coordinates into UTRs.
    3. Apply $E_{\text{lex}}(x)$ strictly to **splice-site tracks** ($B_{\text{donor}, \pm}(x)$ and $B_{\text{acceptor}, \pm}(x)$), **never** to translation start/stop tracks ($B_{\text{start}, \pm}(x)$ or $B_{\text{stop}, \pm}(x)$), because outer transcript HSP ends mark UTR transcription ends (TSS/TTS) or unitig assembly ends, not `ATG`/stop codons!
  - **When `--lexicmap-source genomic`:** HSPs come from whole-genome assemblies or WGS unitigs (e.g., cross-isolate genomic alignments spanning introns and intergenic regions). In `genomic` mode, HSP endpoints reflect assembly breaks, indels, or synteny edges rather than splice junctions, so **HSP endpoint boundary boosting ($E_{\text{lex}}(x)$) is disabled (`w_end = 0.0`)** and coverage $\tilde{d}(x)$ is used strictly as a weak sequence-conservation prior (`w_ev` scaled by `0.5`).
- **Supported Formats (matching `/home/danielle/Documents/brc-analytics/brc-tools/libs/lexicmap-to-ucsc/src/lexicmap_to_ucsc/core.py`):**
  1. **BED6 (`.bed`):** `chrom  start  end  sgenome|sseqid|hspN  score  strand` (0-based half-open; `score` is bitscore capped at 1000; `sgenome` before the first `|` is the SRA run / accession, e.g. `SRR13250831`).
  2. **bedGraph (`.bedGraph`):** `chrom  start  end  hsp_count` (0-based half-open).
  3. **Raw LexicMap TSV (`.tsv`):** Header with `query, qlen, qstart, qend, sgenome, sseqid, hsp, bitscore, sstr` (1-based inclusive; optional `pident`/`qcov` columns when present).
- **Pre-Aggregation HSP Quality, Length & Repeat Filtering (for `.bed` and `.tsv`):**
  Raw LexicMap searches against large SRA indices often include short spurious seeds, low-score distant homolog hits, or mobile-element/simple-repeat pileups. Filter HSP intervals before building tracks:
  1. **Minimum Bitscore / Score Filter (`--lexicmap-min-score`, default `150`):** Drop HSP records with `score < 150` (`bitscore < 150`).
  2. **Minimum HSP Alignment Length (`--lexicmap-min-len`, default `60 bp`):** Drop intervals with `end - start < 60` so short k-mer seed matches do not pollute coverage or boundary tracks.
  3. **Low-Complexity / Repeat-Region Masking:** For any HSP interval whose query sequence has low Shannon dinucleotide entropy ($H_2 < 1.8\text{ bits}$, e.g. poly-AT or microsatellite runs) or $>20\%$ softmasked/ambiguous bases, discard the hit.
  4. **Per-SRA-Run Redundancy & Depth Cap (`--lexicmap-max-per-run`, default `2`):** If a single SRA run (`sgenome`) produces dozens of overlapping paralog/repeat HSPs at the same query bin $x$, cap each distinct `sgenome` run's contribution at `1` (or `max_per_run`) per bin so `d(x)` measures **independent SRA sample support** rather than within-sample repeat copy number.
  5. **Repeat / Transposon Damper:** Even after per-run deduplication, if $d(x) > 10 \times \operatorname{median}_{d(y)>0}(d(y))$ AND the model's own $P_{\text{coding}}(x) + P_{\text{rna}}(x) < 0.35$, cap $\tilde{d}(x)$ at `0.15` and suppress endpoint votes inside that bin so conserved transposable elements or rDNA arrays do not force false protein-coding calls.
- **Evidence Signals & Fusion with OmniPhyl Emissions (`--evidence-weight`, default `w_ev = 0.35`):**
  1. **Normalized Expression / Conservation Coverage $\tilde{d}(x) \in [0, 1]$:**
     $$\tilde{d}(x) = \min\!\left(1.0,\; \frac{\log(1 + d(x))}{\max(\operatorname{P}_{90}(\log(1 + d)),\, 1.0)}\right)$$
  2. **HSP Endpoint Splice Density $E_{\text{lex}}(x)$ (enabled only when `--lexicmap-source rnaseq`):**
     - Accumulate filtered HSP `start` and `end` coordinates (excluding endpoints within `25 bp` of contig ends or `N`-gaps) into a per-bin impulse track $E_{\text{lex}}(x)$ normalized to $[0, 1]$.
     - Furthermore, when two HSPs from the **same transcript/unitig (`sgenome|sseqid`) and same alignment orientation (`strand`)** align colinearly on the query contig separated by a genomic gap of $35\text{–}5000\text{ bp}$ (and minimal overlap/gap on the unitig if coordinates are available), record that gap as a **split-alignment intron candidate** `[intron_start, intron_end)` with a strong splice-junction bonus in Stage-2 decoding.
  3. **Log-Odds Fusion:**
     - Boost coding log-odds in bin $x$ (gated by coding potential vs. UTR/ncRNA):
       $$w_{\text{utr}}(x) = \min\!\left(1.0,\; \frac{P_{\text{coding}}(x)}{0.25}\right), \qquad \text{score}_{\text{coding}}(x, g) = \log P_{\text{coding}}(x) + \log P_{\text{frame}}(x, g) + w_{\text{ev}} \cdot w_{\text{utr}}(x) \cdot \tilde{d}(x)$$
     - When `--lexicmap-source rnaseq`, boost **donor and acceptor** boundary impulse tracks where model boundary probability is also non-trivial:
       $$B_{\text{splice}, \pm}(x) \leftarrow B_{\text{splice}, \pm}(x) + w_{\text{ev}} \cdot E_{\text{lex}}(x) \cdot P_{\text{bnd}}(x) \quad \text{for } \text{splice} \in \{\text{donor}, \text{acceptor}\}$$
     - Record per-gene `lexicmap_source`, `lexicmap_mean_depth`, `lexicmap_frac_covered`, and `lexicmap_sra_runs` (count of distinct `sgenome` accessions supporting the gene after filtering) in the GFF3 attributes.

#### Step 4: Stage-2 Base-Exact Boundary Snapping & ORF/Exon Decoding
1. **Candidate Interval Extraction:**
   - Find local maxima on the smoothed coding/boundary tracks where $P_{\text{bnd}}(x) \ge \tau_{\text{bnd}}$ (default `0.25`) or where the dominant state transitions between `intergenic`, `coding (frame g)`, and `intron`.
2. **Base-Exact Codon & Splice Snapping (`scan_candidates`):**
   - Generalize `scan_transition` from `scripts/eval_pilot.py:142-166` explicitly for **both strands**:
     - **Forward strand (`+`, $g \in \{0, 1, 2\}$):** Start codons `ATG`, `GTG`, `TTG`; stop codons `TAA`, `TAG`, `TGA` (or `TAA`, `TAG` under Code 4); splice donor `GT` (or `GC`) at the $5'$ end of the intron `[d, d+2)` and acceptor `AG` at the $3'$ end `[a-2, a)`.
     - **Reverse strand (`-`, $g \in \{3, 4, 5\}$):** Scan on the reverse-complement sequence (or equivalently on the forward strand: start codon at the $3'$ genomic coordinate is `CAT`/`CAC`/`CAA`, stop codon at the $5'$ genomic coordinate is `TTA`/`CTA`/`TCA`, minus-strand donor at the $3'$ genomic end of the intron `[a-2, a)` is `CT` (`GC` $\to$ `GC`), and minus-strand acceptor at the $5'$ genomic end `[d, d+2)` is `AC`).
   - **Prokaryotic / Intronless Mode (or single-exon genes):**
     - Seed contiguous regions where $P_{\text{coding}}(x) + w_{\text{ev}}\tilde{d}(x) \ge \tau_{\text{code}}$ (default `0.45`).
     - In the dominant global frame $g \in \{0..5\}$ (and any secondary frame with $>0.25$ posterior), walk outward/inward to enumerate valid stop-free ORFs (minimum length `--min-cds-len`, default `90 bp`).
     - Score each start candidate $s$ near the $5'$ boundary by combining:
       - Model coding integral inside $[s, \text{stop})$ minus coding integral wasted upstream of $s$,
       - Distance to the nearest strand-matched $B_{\text{start}, \pm}$ peak,
       - Codon prior (`ATG > GTG > TTG`),
       - Shine-Dalgarno motif score (`AGGAGG` sub-kmer within `[-20, -4]` upstream of $s$ when contig domain is `Bacteria`/`Archaea`).
   - **Eukaryotic Multi-Exon Mode (Exact Intron Phase & Frame-Shift Arithmetic):**
     - **Critical Biological Rule:** Splicing an intron `[d, a)` of length $L_{\text{int}} = a - d$ between coding exon 1 (global frame $g_1$) and coding exon 2 (global frame $g_2$) does **not** require $g_1 == g_2$! Whenever $L_{\text{int}} \bmod 3 \ne 0$, the genomic reading frame of exon 2 shifts by $L_{\text{int}} \bmod 3$:
       - **Forward strand (`+`, $g_1, g_2 \in \{0, 1, 2\}$):** An intron `[d, a)` is frame-compatible with $(g_1, g_2)$ iff:
         $$(g_1 + (a - d)) \bmod 3 == g_2$$
         Equivalently, the intron phase $p_{\text{int}} = (d - g_1) \bmod 3 \in \{0, 1, 2\}$ (number of coding bases between the last complete codon of exon 1 and donor $d$) must satisfy $(p_{\text{int}} + (a - d - p_{\text{int}})) \implies (a - g_2) \bmod 3 == p_{\text{int}}$.
       - **Reverse strand (`-`, $g_1, g_2 \in \{3, 4, 5\}$):** Recall $g_{\text{rev}} = 3 + (\text{codon0} \bmod 3)$. Moving from the $5'$-upstream exon (higher genomic coordinates, frame $g_1$) across intron `[d, a)` to the $3'$-downstream exon (lower genomic coordinates, frame $g_2$), compatibility requires:
         $$((g_1 - 3) - (a - d)) \bmod 3 == (g_2 - 3)$$
       - **Split-Codon Check:** For phase-1 ($p_{\text{int}}=1$) and phase-2 ($p_{\text{int}}=2$) introns, reconstruct the 3-base junction codon formed by concatenating the trailing $p_{\text{int}}$ base(s) of exon 1 with the leading $3 - p_{\text{int}}$ base(s) of exon 2, and **reject any `(d, a)` pair whose spliced junction codon is a stop codon** under the contig's genetic code.
     - Search a $\pm 150\text{ bp}$ window around candidate donor/acceptor transitions (or take exact RNA-seq split-alignment introns from Step 3) and score valid `(d, a)` pairs by donor/acceptor motif log-odds + $B_{\text{donor}, \pm}(d) + B_{\text{acceptor}, \pm}(a)$ + intron interior $\log P_{\text{intron}}$ integral.
     - Compute and emit exact GFF3 `phase` (`0`, `1`, or `2`) on every `CDS` feature (`0` for the first coding exon; `(3 - p_int) % 3` for each subsequent exon).
3. **Non-Overlapping Gene Model Selection (Dynamic Programming / Greedy NMS):**
   - Score each candidate gene model $m$ by its length-normalized log-likelihood:
     $$S(m) = \sum_{x \in \text{CDS}(m)} \text{score}_{\text{coding}}(x, g_m) + \sum_{x \in \text{Intron}(m)} \log P_{\text{intron}}(x) + \text{boundary\_bonus}(m)$$
   - Resolve overlapping candidate ORFs on the same or opposite strands via weighted interval scheduling (allowing up to `60 bp` tail overlap for compact prokaryotic/apicomplexan operons).
4. **Per-Gene Functional Domain Assignment:**
   - For each called gene $m$, average the Pfam domain probability vector $P_{\text{domains}, w}$ across all windows $w$ overlapping $\text{CDS}(m)$ by $\ge 90\text{ bp}$ (or $\ge 50\%$ of the CDS length for short genes $< 180\text{ bp}$). Emit all domains clearing `--domain-threshold` (default `0.5`).

---

## 4. Use Case 2: Reference-Free Metagenomics (`omniphyl profile`)

### 4.1 Goal & Scope
Given one or more metagenomic sample files (raw short reads in `.fq/.fq.gz`, paired-end reads, Logan unitigs, or assembled contigs in `.fa/.fa.gz`), produce a reference-DB-free taxonomic and functional profile across all samples:
1. **`taxonomic_profile.tsv`** — Samples $\times$ Taxa **sequence-abundance** matrix (explicitly documented as fraction of total sequenced bases `--abundance-metric bp` or reads `--abundance-metric reads`, i.e. *taxonomic sequence abundance* rather than genome-size-normalized cell counts, since genome sizes of uncultured/novel clades are unknown). Includes full NCBI lineage `d__...;p__...;c__...;o__...;f__...;g__...;s__...` and explicit `unclassified_<parent>` rows when novel clades abstain at deeper ranks.
2. **`abundance_<rank>.tsv`** — Per-rank (`domain`, `phylum`, `class`, `order`, `family`, `genus`, `species`) Samples $\times$ Taxon sequence-abundance table (columns = samples, values sum to `1.0` per sample).
3. **`pfam_profile.tsv`** — Samples $\times$ Pfam domains matrix (coding-bp-normalized domain abundance per sample).
4. **`taxon_pfam_joint.tsv`** — Long-format table `(sample, rank, taxon, pfam_id, coding_bp, mean_prob)` linking **which taxon carries which Pfam domains** directly from the joint forward pass, without requiring MAG binning.
5. **`sample_summary.tsv`** — Per-sample total bp, coding fraction, intron/RNA/intergenic fractions, domain-level breakdown, **mean class-0 OOD probability (`mean_p_ood`)**, and **novelty index** (fraction of bp abstaining at genus/species while confident at phylum/class/order).

**Prior-Temperature Note for Metagenomic Profiling (`--flat-prior-tau`):**
- By default, `OmniPhylPredictor` uses the checkpoint's training `flat_prior_tau` (typically `0.5`), which weights logits toward RefSeq training-corpus taxon frequencies. Because real environmental metagenomes can have taxonomic compositions very different from RefSeq, `omniphyl profile` exposes `--flat-prior-tau FLOAT` (allowing `--flat-prior-tau 0.0` for uniform-prior likelihood-ratio scoring) and records the active `flat_prior_tau` in `sample_summary.tsv`.

### 4.2 Handling the Three Input Regimes (`--read-mode auto|precontig|raw|contig`)

A single sample may contain long contigs ($\ge 500\text{ bp}$), unitigs ($200\text{–}1000\text{ bp}$), or raw short reads ($100\text{–}300\text{ bp}$). Note that OmniPhyl checkpoints are trained on $500\text{ bp}$ fragments; for raw short reads ($< 500\text{ bp}$), both `precontig` and `raw` modes are empirical inference strategies whose accuracy vs. assembled contigs should be benchmarked on synthetic read mixtures in `tests/test_inference.py`.

#### Mode A: Assembled Contigs / Long Unitigs (`len(seq) >= 500 bp`)
- Slice each contig into non-overlapping (or `stride=250 bp`) $500\text{ bp}$ windows $w_1, \dots, w_M$.
- **Cross-Contig Window Batching:** Accumulate windows across contigs into full batches of `--batch-size 256` before calling `predict_sequences()`, then group the output window log-posteriors back by `contig_id`.
- **Contig-level $M$-fragment consensus (`eval_pilot.py:36-37` principle):**
  - Average the per-rank log-probabilities $\overline{\log p}_r = \frac{1}{M}\sum_{m=1}^M \log p_{r, m}$ across all windows of the contig **before** running `reconcile_taxonomy()`.
  - From our v4/v5 benchmarks on $500\text{ bp}$ fragments, pooling $M=15\text{–}20$ windows boosts species-level accuracy from `0.208` ($M=1$) to `0.842+`.
  - Weight the contig's contribution to sample sequence abundance by its sequence length $L$ (multiplied by contig read-depth if present in FASTA header, e.g., MEGAHIT `multi=12.4` or SPAdes `cov_12.4_`).

#### Mode B: "Pre-Contig" Read Clustering on Raw Reads (`len(seq) < 500 bp`, Default for Short Reads)
Why this helps when reads share local overlaps:
- A single $150\text{ bp}$ read has only $145$ 6-mers ($\sim 3.5\%$ of the $4,096$ FCGR cells) and no training-time length augmentation, making single-read calls noisy.
- When sequencing depth is sufficient that reads from the same genomic locus overlap, grouping reads by shared exact $(k=21)$-mers forms **"pre-contig" read clusters** (micro-bins of reads from the same local locus) without full De Bruijn graph assembly.
- Furthermore, **`fcgr_encode_ms` (without positional channels), `hash_encode`, and `aa_encode` are additive k-mer/tripeptide count histograms** up to $500\text{ bp}$.

**Exact Pre-Contig Clustering & Dual-Pooling Algorithm:**
1. **Paired-End Joining (if `--R1` / `--R2` or interleaved PE provided):**
   - Join mate pairs `(r1, rc(r2))` with a single `'N'` separator (`f"{r1}N{rc(r2)}"`, matching `eval_crop_seq` in `scripts/train_omniphyl.py:488-503` where `'N'` cleanly resets k-mer windows without creating false junction k-mers). A $2\times 150\text{ bp}$ pair immediately yields $300\text{ bp}$ of spectrum from the same DNA molecule.
2. **All-Canonical-21-mer Indexing with Anti-Chaining Guards:**
   - Why not $m=4$ global minimum hashes per read? On $150\text{ bp}$ reads with $50\text{–}80\text{ bp}$ overlaps, two overlapping reads frequently do not share their global minimum hash across the entire read. Instead, index **all** canonical $k_{\text{min}}=21$-mers (or subsampled every $s=3\text{ bp}$, yielding $\sim 43$ canonical 21-mers per $150\text{ bp}$ read; for $200,000$ reads this is $\le 8.6 \times 10^6$ integer hashes, fitting in $< 100\text{ MB}$ of NumPy arrays):
     - Compute canonical 21-mer 64-bit SplitMix64 hashes vectorized across each read.
     - **High-Frequency / Conserved-Region Filter:** Count 21-mer frequencies across the sample first; discard any 21-mer occurring above $\max(20,\; 5 \times \operatorname{median\_count})$ so conserved rRNA, tRNA, mobile elements, or plasmid backbones cannot chain unrelated species into chimeric clusters.
     - **Two-Hit Link Rule & Star/Bounded-Diameter Clustering:** Require two reads (or mate-pairs) to share **at least 2 distinct non-repetitive 21-mers** (or $\ge 1$ 21-mer when `--min-shared-kmers 1` is explicitly requested) to form an edge, and grow clusters greedily around seed reads (maximum cluster size `max_cluster_reads = 32`, maximum graph hop radius $= 2$ from seed) rather than unconstrained transitive union-find. This prevents transitive chaining across conserved gene boundaries.
3. **Two-Level Pooling per Pre-Contig Cluster:**
   - Suppose a cluster has $R$ reads totaling $L_{\text{tot}} = \sum_{j=1}^R \ell_j$ bases:
   - **Level 1 — Virtual $\le 500\text{ bp}$ Spectrum Packing (Pre-CNN):**
     - Partition the cluster's reads into chunks of **at most $500\text{ bp}$ of non-`N` bases** (e.g., three $150\text{ bp}$ reads joined by `'N'` separators: `read1 + "N" + read2 + "N" + read3`, totaling $450\text{ bp} \le 500\text{ bp}$).
     - Keeping each virtual chunk $\le 500\text{ bp}$ ensures `hash_encode` (`HASH_LEN = 512`) never truncates the 3rd read and matches the $500\text{ bp}$ training fragment length distribution.
     - Note: if the checkpoint was trained with `pos_channels=True`, joining arbitrary reads alters the left- vs. right-half positional FCGR channels; in `FeatureEncoder`, if `pos_channels=True` on an `'N'`-joined virtual chunk, set the two positional difference channels to zero (`0.0`, their expected value on a stationary sequence) so arbitrary read ordering cannot introduce artificial positional skew.
   - **Level 2 — Cluster Log-Posterior Consensus (Post-CNN):**
     - If a cluster produces $M \ge 2$ virtual $\le 500\text{ bp}$ chunks, average their per-rank log-probabilities $\overline{\log p}_r = \frac{1}{M}\sum_{m=1}^M \log p_{r, m}$ before running `reconcile_taxonomy()`.
     - Assign the resulting reconciled taxon call to all $R$ reads ($L_{\text{tot}}$ bases) in the cluster.
4. **Singleton Read Handling:**
   - Reads that do not cluster with any other read (singletons, which may be low-abundance taxa or low-coverage samples) are evaluated individually (or as `'N'`-joined mate-pairs if PE) with a stricter abstention threshold (`conf_tau_singleton = max(conf_tau, 0.65)`) so they contribute where confident at shallower ranks without inflating false species calls.

#### Mode C: Raw Read Mode (`--read-mode raw`)
- Scores each read / mate-pair (`r1 + "N" + rc(r2)`) directly without k-mer clustering; used as a baseline and on low-coverage samples where reads rarely overlap.

---

## 5. CLI Specification (`src/omniphyl/cli.py`)

Expose a single `omniphyl` entrypoint with two subcommands (`omniphyl annotate` and `omniphyl profile`):

### 5.1 `omniphyl annotate`
```text
usage: omniphyl annotate [-h] --model MODEL --fasta FASTA --out-prefix OUT_PREFIX
                         [--window 500] [--stride 100] [--batch-size 256]
                         [--device auto|cpu|cuda] [--flat-prior-tau FLOAT]
                         [--lexicmap-bed PATH] [--lexicmap-bedgraph PATH]
                         [--lexicmap-tsv PATH] [--lexicmap-source rnaseq|genomic]
                         [--lexicmap-min-score 150] [--lexicmap-min-len 60]
                         [--lexicmap-max-per-run 2] [--evidence-weight 0.35]
                         [--kingdom auto|eukaryote|prokaryote|viral]
                         [--genetic-code auto|1|4] [--min-cds-len 90]
                         [--coding-threshold 0.45] [--boundary-threshold 0.25]
                         [--domain-threshold 0.5]
```

**Outputs written:**
- `<out_prefix>.gff3`
- `<out_prefix>.report.md` (and printed summary to stdout)
- `<out_prefix>.report.json`
- `<out_prefix>.coding.bedGraph`
- `<out_prefix>.boundary.bedGraph`
- `<out_prefix>.contig_summary.tsv`

### 5.2 `omniphyl profile`
```text
usage: omniphyl profile [-h] --model MODEL --outdir OUTDIR
                        [--samples SAMPLE [SAMPLE ...]]
                        [--sample-sheet TSV]
                        [--read-mode auto|precontig|raw|contig]
                        [--window 500] [--stride 250] [--max-reads 200000]
                        [--cluster-k 21] [--cluster-min-shared 2]
                        [--cluster-max-reads 64] [--min-overlap 30]
                        [--pack-below 300]
                        [--batch-size 256] [--device auto|cpu|cuda]
                        [--flat-prior-tau FLOAT]
                        [--conf-threshold 0.5] [--anc-veto-tau 0.2]
                        [--domain-threshold 0.5] [--abundance-metric bp|reads]
```
- Accepts multiple inputs directly (`omniphyl profile --model best.pt --outdir out/ sampleA.fq.gz sampleB.fa`) where sample names default to filename stems, OR `--sample-sheet samples.tsv` with columns `sample_id\tpath_r1\tpath_r2` (where `path_r2` is optional).

**Outputs written in `<outdir>/`:**
- `taxonomic_profile.tsv` (First column `lineage`, subsequent columns `<sample_1>`, `<sample_2>`, ..., values = relative sequence abundance in `[0, 1]`)
- `abundance_domain.tsv`, `abundance_phylum.tsv`, ..., `abundance_species.tsv`
- `pfam_profile.tsv` (First column `domain_id`, subsequent columns `<sample_1>`, `<sample_2>`, ..., values = domain hits per kb of coding sequence)
- `taxon_pfam_joint.tsv` (`sample\tdeepest_rank\ttaxon\tlineage\tdomain_id\tcoding_bp\tmean_domain_prob`)
- `sample_summary.tsv` (`sample\ttotal_seqs\ttotal_bp\tn_clusters\tcoding_frac\tintron_frac\trna_frac\tmean_p_ood\tnovelty_rate_genus\tmean_conf_phylum\tflat_prior_tau`)

---

## 6. Step-by-Step Implementation Checklist for the Agent

Execute these steps in order:

### Phase 1: Extract Core Modules & Preserve Backwards Compatibility
1. **Create `src/omniphyl/__init__.py` and `src/omniphyl/__main__.py`.**
2. **Create `src/omniphyl/encode.py`:**
   - Move/copy `BASE2DIGIT`, `MS_KS`, `VTX`, `HASH_K`, `HASH_BUCKETS`, `HASH_PAD`, `HASH_LEN`, `AA_LETTERS`, `AA_ALPHA`, `AA_DIM`, `AA_EXTRA_DIM`, `AA_FRAME_DIM`, `_AA_LUT`, `build_cell_lut`, `rc_index_map`, `fcgr_encode`, `fcgr_encode_ms`, `hash_encode`, `aa_encode`, `aa_frame_encode`, `seq_encode`, and `scan_transition` (from `scripts/eval_pilot.py:138-167`).
   - Add a `FeatureEncoder` helper class initialized from a checkpoint's `k` and `train_args` that caches `ms_luts`, `ms_canons`, `lut`, and `hash_canon` once, provides `encode_batch(seqs: list[str], virtual_joined: bool = False) -> Tensor | dict[str, Tensor]`, and zeroes the 2 positional difference channels when `virtual_joined=True` and `pos_channels=True`.
3. **Create `src/omniphyl/model.py`:**
   - Move/copy `RANKS`, `CODING_TARGETS`, `CONT_TARGETS`, `Block`, `ResNetEncoder`, `SeqEncoder`, `ViTEncoder`, `TreeRankHead`, `OmniPhyl`, `build_tree`, `build_flat_priors`, `build_parent_of`, `ancestor_chains`, and `path_scores`.
   - Implement `OmniPhylPredictor`:
     - Loads checkpoint via `torch.load(model_path, map_location="cpu", weights_only=False)`.
     - Emits startup warnings if optional heads (`region_head`, `struct_head`, `domains`) are absent in the checkpoint.
     - Reconstructs `OmniPhyl` using exact logic from `scripts/eval_pilot.py:404-453`.
     - Precomputes `chains = ancestor_chains(...)` from `ckpt["tree"]["parent_of"]` and moves `flat_lp` to device.
     - Provides `predict_sequences(seqs: list[str], batch_size: int = 256, virtual_joined: bool = False) -> BatchPredictions` returning numpy arrays for `logp: dict[str, np.ndarray]`, `p_ood: dict[str, np.ndarray]`, `coding_frac`, `boundary`, `offset`, `region_fracs`, `active_frame`, `struct_flags`, and `domains`.
     - Provides `reconcile_logp(pooled_logp: dict[str, np.ndarray], conf_tau: float = 0.5, anc_tau: float = 0.2) -> list[TaxonCall]` returning per-item reconciled calls across all 7 ranks (using strict NCBI ancestor projection from the deepest confident rank $r^*$) plus deepest confident rank and lineage string.
4. **Wire Re-Exports in `scripts/train_omniphyl.py` and `scripts/eval_pilot.py`:**
   - Add `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))` at top of `scripts/train_omniphyl.py` and import the shared symbols from `omniphyl.encode` and `omniphyl.model`.
   - Verify immediately: run `python3 tests/test_smoke.py` in the `omniphyl` conda env and confirm **all existing smoke tests pass**.

### Phase 2: Streaming Sequence & Evidence I/O (`src/omniphyl/io.py`)
1. Implement `iter_sequences(path: Path) -> Iterator[tuple[str, str, float]]`:
   - Auto-detects `.gz` (via magic bytes `1f 8b` or `.gz` suffix) and FASTA (`>`) vs FASTQ (`@`).
   - Uses length-counted 4-record FASTQ parsing (accumulating sequence lines until `+\n`, then accumulating quality lines until `len(qual) == len(seq)`) so wrapped FASTQ or quality lines starting with `@` never corrupt parsing.
   - Parses optional coverage multiplier from FASTA headers (`cov_(\d+\.?\d*)` or `multi=(\d+\.?\d*)`, defaulting to `1.0`).
2. Implement `iter_paired_sequences(r1: Path, r2: Path) -> Iterator[tuple[str, str, float]]`:
   - Streams R1 and R2 in lockstep and yields `(id, f"{seq1}N{revcomp(seq2)}", 1.0)`.
3. Implement `load_lexicmap_evidence(bed_path=None, bedgraph_path=None, tsv_path=None, source="rnaseq", min_score=150.0, min_len=60, max_per_run=2, contig_seqs=None) -> dict[str, ContigEvidence]`:
   - Parses `.bed` (0-based half-open, extracting `sgenome` run ID before `|` in column 4 and transcript/unitig ID `sgenome|sseqid`), `.bedGraph` (0-based half-open density), or raw LexicMap `.tsv` (1-based inclusive `qstart..qend`).
   - Applies score filtering (`score >= min_score`), length filtering (`end - start >= min_len`), low-entropy repeat filtering on `contig_seqs`, and per-SRA-run depth deduplication (`max_per_run`).
   - Computes per-bin depth $d(x)$, RNA-seq HSP endpoint impulse count $E_{\text{lex}}(x)$ (zeroed when `source == "genomic"`), colinear split-alignment intron candidates, and interval-to-SRA-run sets for GFF3 tagging.
4. Implement `write_gff3(...)` and `write_bedgraph(...)`.

### Phase 3: Implement `omniphyl annotate` (`src/omniphyl/annotate.py`)
1. Implement strided window slicer (`window=500`, `stride=100`).
2. Implement raised-cosine bin aggregator with the **global reading-frame coordinate transform** (§3.2 Step 2.2) and strand-separated boundary tracks (§3.2 Step 2.3).
3. Implement assembly-wide and per-contig taxonomy pooling, chimeric/contaminant contig flagging, and assembly-wide genetic-code inference (with $\ge 1000$-codon minimum evidence gate).
4. Implement LexicMap RNA-seq vs. genomic evidence fusion + repeat damper (§3.2 Step 3).
5. Implement Stage-2 both-strand ORF / splice-junction boundary snapper (with exact phase-0/1/2 frame-shift arithmetic `(g1 + (a - d)) % 3 == g2` and split-codon stop checking) and non-overlapping gene selector (§3.2 Step 4).
6. Emit `<prefix>.gff3`, `<prefix>.report.md`, `<prefix>.report.json`, `<prefix>.coding.bedGraph`, `<prefix>.boundary.bedGraph`, and `<prefix>.contig_summary.tsv`.

### Phase 4: Implement `omniphyl profile` (`src/omniphyl/profile.py`)
1. Implement `cluster_short_reads_precontig()`:
   - Canonical 21-mer vectorized hashing in NumPy (strided every $s=3\text{ bp}$), with sample-wide high-frequency 21-mer masking.
   - Bounded-radius greedy star clustering requiring $\ge 2$ shared non-repetitive 21-mers per read pair (`max_cluster_reads = 32`).
   - Pack cluster reads into `'N'`-joined virtual fragments of $\le 500\text{ bp}$ non-`N` bases.
2. Implement unified sample profiler:
   - Long sequences ($\ge 500\text{ bp}$) $\to$ $500\text{ bp}$ windows $\to$ per-contig log-prob average.
   - Short sequences ($< 500\text{ bp}$) in `precontig` mode $\to$ virtual $\le 500\text{ bp}$ fragments $\to$ per-cluster log-prob average.
   - Run `reconcile_logp` on each contig/cluster (with strict NCBI ancestor projection from $r^*$), weight by bp (or read count), accumulate into per-rank taxon sequence-abundance tables, Pfam domain abundance tables (`pfam_profile.tsv`), and joint `(taxon, pfam)` tables (`taxon_pfam_joint.tsv`).
3. Write multi-sample output matrices.

### Phase 5: CLI & Verification Suite (`src/omniphyl/cli.py` & `tests/test_inference.py`)
1. Wire `omniphyl annotate` and `omniphyl profile` in `src/omniphyl/cli.py`.
2. Install package in editable mode inside the `omniphyl` conda env: `pip install --no-build-isolation --no-deps -e .`.
3. Write `tests/test_inference.py` covering:
   - **Strict NCBI tree-projection test:** craft synthetic flat-head logits where `family` argmax is `Family X` and `genus` argmax is `Genus B` (belonging to `Family Y`), and verify `reconcile_logp` never emits a contradictory `(Family X, Genus B)` pair and that per-rank abundances sum coherently up the tree.
   - **Global reading frame coordinate invariance & intron phase test:** verify that a synthetic CDS starting at coordinate `c0` maps to the exact same contig-global frame `g` across windows at offsets `0, 100, 200, 300, 400`, and verify that phase-0, phase-1, and phase-2 introns on both `+` and `-` strands satisfy the frame-shift equations and reject split stop codons.
   - **LexicMap filtering & `rnaseq` vs `genomic` mode test:** verify that low-score (`<150`), short (`<60 bp`), and low-entropy repeat hits are filtered out, per-SRA-run duplicates are capped, colinear split-HSP introns are detected in `rnaseq` mode, and endpoint boundary boosting is zeroed in `genomic` mode.
   - **End-to-end `omniphyl annotate` test:** train a tiny 2-epoch model on a synthetic 2-contig genome (with known CDS + intron + Pfam domain), run `omniphyl annotate` both with and without a synthetic LexicMap `.bed`/`.bedGraph` file, and verify GFF3 coordinate recovery, `.report.json`/`.report.md` species call + feature counts + chimera detection, `lexicmap_sra_runs` attributes, and `.bedGraph` output.
   - **Real LexicMap data integration test (if `/home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/` exists):** run `omniphyl annotate` on `chr5_first_10k_bases.fasta` with `chr5_annot.bed` / `chr5_annot.bedGraph` (`--lexicmap-source genomic`) and verify valid GFF3 + track generation.
   - **End-to-end `omniphyl profile` test across all 3 read modes (`contig`, `precontig`, `raw`):** simulate 150 bp reads from 2 distinct synthetic genomes (including a shared repetitive 21-mer to test anti-chaining), verify that `precontig` clustering groups same-genome reads without cross-species chaining, and emits valid `taxonomic_profile.tsv`, `abundance_<rank>.tsv`, `pfam_profile.tsv`, `taxon_pfam_joint.tsv`, and `sample_summary.tsv` summing to `1.0` per sample.
4. Run full test suite:
   ```bash
   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate omniphyl
   pytest tests/test_smoke.py tests/test_inference.py
   ```
