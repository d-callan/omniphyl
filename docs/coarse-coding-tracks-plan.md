# Plan 1: Coarse Coding Browser Tracks (`omniphyl annotate`)

This document is a self-contained implementation specification for an independent agent with **no prior conversation context**. It specifies exact files, line numbers, data structures, output formats, CLI options, and unit tests to emit genome-browser tracks (UCSC / IGV compatible BED9 and bedGraph files) showing **coarse coding vs. non-coding intervals and frame/strand dominance** directly from OmniPhyl's Stage-1 aggregated bin signals, without relying on Stage-2 exact ORF/splice-junction decoding.

---

## 0. Environment & Repository Paths

- **Repository root:** `/home/danielle/Documents/veg/omniphyl`
- **Python / test runner:** `/home/danielle/miniconda3/envs/omniphyl/bin/python`
- **Primary files to edit:**
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py`
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/cli.py`
  - `/home/danielle/Documents/veg/omniphyl/tests/test_inference.py`
- **Cached Stage-1 benchmark artifact (optional visual smoke test, <1 s):**
  - `/home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl`
  - `/home/danielle/omniphyl-work/infer_test/NC_004325.2.gff3`

---

## 1. Context & Problem Statement

In `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py`, `run_annotate` (lines 81–296) runs in two stages:

1. **Stage 1 — Window inference & raised-cosine bin aggregation (`_fuse_contig`, lines 302–395):**
   Produces a `ContigResult` dataclass (lines 47–61) per contig on a fixed `B`-bp grid (`B = gcd(stride, 50)`, default `50 bp`):
   - `res.p_coding`: `float64 [nbins]` — smoothed coding probability in `[0, 1]`.
   - `res.p_region`: `float64 [nbins, 4]` — `("coding", "intron", "rna", "intergenic")` probabilities (falls back to `[p_coding, 0, 0, 1 - p_coding]` when the checkpoint has no `region_fracs` head).
   - `res.p_frame`: `float64 [nbins, 8]` — contig-global reading-frame probabilities:
     - `0, 1, 2`: `+` strand global frames (`+0, +1, +2`)
     - `3, 4, 5`: `-` strand global frames (`-0, -1, -2`)
     - `6, 7`: non-coding / dual-strand classes
   - `res.b_tracks`: `float64 [nbins, 4]` — boundary impulse tracks `("start", "stop", "donor", "acceptor")`.
   - `res.p_boundary`: `float64 [nbins]` — window boundary probability.
   - `res.dtilde`: `float64 [nbins]` — normalized LexicMap evidence coverage in `[0, 1]`.

2. **Stage 2 — Exact ORF & splice-junction decoding (`_decode_genes`, lines 465–642):**
   Attempts to snap exact `ATG -> stop` ORFs and `GT..AG` intron chains and writes `.gff3`.

Currently, `_write_outputs` (lines 1404–1540) writes `.gff3`, `.coding.bedGraph`, and `.boundary.bedGraph`. However:
- There is **no discrete interval track (`.coding_regions.bed`)** showing coarse predicted coding blocks with strand, dominant reading frame, score, and UCSC RGB coloring so a user can load the file alongside a reference GFF3 in UCSC Genome Browser or IGV and visually inspect whether OmniPhyl's raw coding/strand/frame signal matches real genes (independent of Stage-2 ORF/intron heuristics).
- There is **no stranded or frame-confidence signal track (`.frame.bedGraph`)** showing signed strand-directed coding confidence (`+` coding above the axis in `[0, 1]`, `-` coding below the axis in `[-1, 0]`).
- Running `omniphyl annotate` always runs the full Stage-2 ORF/splice decoder (`_decode_genes`), even when the user only wants fast browser tracks (`--tracks-only`).
- In `_write_bg` (lines 1560–1569), the last bin `j * bin_width` can exceed `res.length` (e.g., a 640,851 bp contig with `50 bp` bins emits `640850..640900`), which UCSC `bedGraphToBigWig` / `bedToBigBed` rejects as out-of-bounds.

---

## 2. Specification of New Outputs & Behavior

### 2.1 Clamp Coordinates to Contig Length (`_write_bg`)

In `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py:1560-1569`, update `_write_bg` to accept `length: int | None = None` and clamp the interval end `min(j * bin_width, length)`:

```python
def _write_bg(f, seqid: str, values: np.ndarray, bin_width: int,
              length: int | None = None):
    i, n = 0, len(values)
    while i < n:
        v = float(values[i])
        j = i + 1
        while j < n and round(float(values[j]), 3) == round(v, 3):
            j += 1
        start = i * bin_width
        end = min(j * bin_width, length) if length is not None else j * bin_width
        if end > start:
            f.write(f"{seqid}\t{start}\t{end}\t{v:.3f}\n")
        i = j
```
Pass `res.length` in all calls to `_write_bg`.

---

### 2.2 Signed Strand/Frame Track: `<out_prefix>.frame.bedGraph`

Write `<out_prefix>.frame.bedGraph` in `_write_outputs` right after `.boundary.bedGraph`.

- **Header line:**
  ```
  track type=bedGraph name=omniphyl_frame description="OmniPhyl signed strand coding score (+strand > 0, -strand < 0)" viewLimits=-1:1 autoScale=off
  ```
- **Per-bin value computation (`_signed_frame_signal(res, w_ev)`):**
  - Let `coding = res.p_region[:, 0] if res.p_region[:, 0].sum() > 0 else res.p_coding`.
  - Let `utr_gate = np.minimum(1.0, coding / 0.25)`.
  - Let `cscore = np.clip(np.exp(np.log(np.clip(coding, 1e-6, 1.0)) + w_ev * utr_gate * res.dtilde), 0.0, 1.0)`.
  - If `res.p_frame[:, :6].sum() > 0`:
    - `p_plus = res.p_frame[:, 0:3].sum(axis=1)`
    - `p_minus = res.p_frame[:, 3:6].sum(axis=1)`
    - `strand_dir = (p_plus - p_minus) / np.maximum(p_plus + p_minus, 1e-9)` in `[-1, 1]`
    - `signal = cscore * strand_dir`
  - Else (checkpoint without `struct_head`):
    - `signal = cscore`

In UCSC / IGV, this renders as a single two-sided wiggle track where `+` strand coding regions rise to `+1.0` and `-` strand coding regions dip to `-1.0`.

---

### 2.3 Coarse Coding Intervals Track: `<out_prefix>.coding_regions.bed` (BED9)

Write a helper `_coarse_coding_blocks(res: ContigResult, coding_threshold: float = 0.45, w_ev: float = 0.25, min_block_bp: int = 100, max_gap_bins: int = 1) -> list[dict]` in `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py` and emit `<out_prefix>.coding_regions.bed` in `_write_outputs`.

#### Algorithm for `_coarse_coding_blocks`
1. **Compute fused coding score per bin (`nbins`):**
   - `coding = res.p_region[:, 0] if res.p_region[:, 0].sum() > 0 else res.p_coding`
   - `utr_gate = np.minimum(1.0, coding / 0.25)`
   - `cscore = np.clip(np.exp(np.log(np.clip(coding, 1e-6, 1.0)) + w_ev * utr_gate * res.dtilde), 0.0, 1.0)`
   - `in_coding = cscore >= coding_threshold`
2. **Contiguous bin runs `[b0, b1]` (inclusive bin indices):**
   - Scan `in_coding` to collect contiguous runs `[(b0, b1), ...]`.
3. **Morphological gap closing (`max_gap_bins = 1`):**
   - Merge two consecutive runs `(b0, b1)` and `(b2, b3)` if the gap `b2 - b1 - 1 <= max_gap_bins` **and** (when `have_frames` is true) both runs have the same dominant strand (`+` vs `-`). This prevents a single 50 bp noisy bin dip from splitting a coding block while keeping opposite-strand neighbors separate.
4. **Filter by minimum length (`min_block_bp`, default `100`):**
   - For each run `(b0, b1)`:
     - `start = b0 * B`
     - `end = min((b1 + 1) * B, res.length)`
     - If `end - start < min_block_bp`: skip.
5. **Summarize block attributes (no ORF or splice logic):**
   - `mean_coding = float(cscore[b0:b1 + 1].mean())`
   - `score = int(round(min(1000.0, max(0.0, mean_coding * 1000.0))))` (standard BED `0..1000` integer score).
   - If `res.p_frame[:, :6].sum() > 0`:
     - `f_mean = res.p_frame[b0:b1 + 1, :6].mean(axis=0)` (`float[6]`)
     - `p_plus = float(f_mean[0:3].sum())`
     - `p_minus = float(f_mean[3:6].sum())`
     - `strand = "+" if p_plus >= p_minus else "-"`
     - `g_dom = int(f_mean[0:3].argmax()) if strand == "+" else 3 + int(f_mean[3:6].argmax())`
     - `frame_label = f"{'+' if g_dom < 3 else '-'}{g_dom % 3}"`
     - `frame_conf = float(f_mean[g_dom] / max(p_plus + p_minus, 1e-9))`
   - Else:
     - `strand = "."`
     - `frame_label = "none"`
     - `frame_conf = 0.0`
   - **High-confidence core (`thickStart`, `thickEnd`):**
     - Find bins within `[b0, b1]` where `cscore >= max(coding_threshold + 0.15, 0.65)`.
     - If any exist, `thick_start = first_bin * B` and `thick_end = min((last_bin + 1) * B, end)`; otherwise `thick_start = start`, `thick_end = end`.
   - **UCSC RGB color (`itemRgb`):**
     - Color by dominant global reading frame so frame shifts across exons are immediately visible in UCSC/IGV:
       - `+0`: `"31,119,180"` (blue)
       - `+1`: `"44,160,44"` (green)
       - `+2`: `"23,190,207"` (teal)
       - `-0`: `"214,39,40"` (red)
       - `-1`: `"255,127,14"` (orange)
       - `-2`: `"148,103,189"` (purple)
       - `"none"`: `"127,127,127"` (grey)
   - **Name field (`col 4`):**
     - `f"cblock_{idx}|f={frame_label}|p={mean_coding:.2f}|fc={frame_conf:.2f}"`

#### BED9 File Format (`<out_prefix>.coding_regions.bed`)
- Header line:
  ```
  track name="omniphyl_coding_regions" description="OmniPhyl coarse coding regions (colored by global frame)" itemRgb="On"
  ```
- Columns (tab-separated, 9 columns per row):
  `chrom  chromStart  chromEnd  name  score  strand  thickStart  thickEnd  itemRgb`

---

### 2.4 CLI Flag `--tracks-only`

In `/home/danielle/Documents/veg/omniphyl/src/omniphyl/cli.py`:
1. Add `--tracks-only` to `_annotate_parser` (under the `decoding` argument group, around line 88):
   ```python
   dec.add_argument("--tracks-only", action="store_true",
                    help="skip Stage-2 ORF/intron decoding; emit coarse "
                         "coding BED/bedGraph tracks and taxonomy report "
                         "only")
   ```
2. Thread `tracks_only: bool = False` into `run_annotate(...)` in `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py` (line 81) and forward `tracks_only=args.tracks_only` from `cli.py:main` (line 143).
3. In `run_annotate` (line 232):
   - When `tracks_only=True`, skip the `_decode_genes` loop (`res.genes = []` is already the default on `ContigResult`), still compute taxonomy calls, genetic code, chimeric flags, and call `_write_outputs`.
   - Note: `<out_prefix>.coding_regions.bed` and `<out_prefix>.frame.bedGraph` are **always** written by `_write_outputs` (both in normal mode and in `--tracks-only` mode).
4. Update the summary print in `_write_outputs` (line 1536) and `--out` help text in `cli.py` (line 35) to list `coding_regions.bed` and `frame.bedGraph`.
5. Add `"n_coding_blocks": total_blocks` to `report["features"]` in `_write_outputs` (line 1522).

---

## 3. Verification & Unit Tests

### 3.1 Unit Tests in `/home/danielle/Documents/veg/omniphyl/tests/test_inference.py`
Add tests inside `test_annotate()` (or a new `test_coarse_coding_tracks()` function registered in `main`):
1. **BED9 schema & coordinate clamping:**
   - Run `run_annotate` on the synthetic genome in `test_inference.py`.
   - Assert `<prefix>.coding_regions.bed` and `<prefix>.frame.bedGraph` exist.
   - Parse non-header lines of `.coding_regions.bed`: assert every row has 9 tab-separated columns, `0 <= int(start) < int(end) <= contig_len`, `0 <= int(score) <= 1000`, `strand in {"+", "-", "."}`, `int(start) <= int(thickStart) <= int(thickEnd) <= int(end)`, and `itemRgb` matches `^\d+,\d+,\d+$`.
   - Parse `.coding.bedGraph`, `.boundary.bedGraph`, and `.frame.bedGraph`: assert no interval end exceeds `contig_len`, and `.frame.bedGraph` values lie in `[-1.0, 1.0]`.
2. **Direct unit test of `_coarse_coding_blocks`:**
   - Construct a synthetic `ContigResult` with `length=1230`, `bin_width=50`, `n_bins=25`:
     - Bins `2..6` (`100..350 bp`): `p_coding = 0.85`, dominant frame `+1` (`p_frame[:, 1] = 0.8`).
     - Bin `7` (`350..400 bp`): 1-bin dip `p_coding = 0.30`, dominant frame `+1` (`p_frame[:, 1] = 0.7`).
     - Bins `8..11` (`400..600 bp`): `p_coding = 0.80`, dominant frame `+1`.
     - Bins `15..20` (`750..1050 bp`): `p_coding = 0.90`, dominant frame `-2` (`p_frame[:, 5] = 0.85`).
   - Call `_coarse_coding_blocks(res, coding_threshold=0.45, max_gap_bins=1)`:
     - Verify the 1-bin same-strand dip at bin 7 is bridged into a single `+` block `[100, 600)` with `frame_label == "+1"` and RGB `"44,160,44"`.
     - Verify the second block is `[750, 1050)` on `-` strand with `frame_label == "-2"` and RGB `"148,103,189"`.
3. **`--tracks-only` mode:**
   - Run `run_annotate(..., tracks_only=True)`: assert `report["features"]["n_genes"] == 0`, `report["features"]["n_coding_blocks"] >= 1`, and `.coding_regions.bed` is non-empty while `.gff3` has no gene features.

### 3.2 Run Test Suite
```bash
PYTHONPATH=/home/danielle/Documents/veg/omniphyl/src \
  /home/danielle/miniconda3/envs/omniphyl/bin/python -m pytest \
  /home/danielle/Documents/veg/omniphyl/tests/test_inference.py -x -q
```
All existing and new tests must pass.
