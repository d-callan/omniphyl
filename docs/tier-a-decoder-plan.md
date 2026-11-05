# Tier A Stage-2 Decoder & Evidence-Wiring Implementation Plan

This document is a self-contained implementation specification for an independent agent with **no prior conversation context**. It specifies exact files, line numbers, root-cause diagnostics, algorithms, unit tests, and benchmark commands.

---

## 0. Environment, Repository & Benchmark Paths

- **Repository root:** `/home/danielle/Documents/veg/omniphyl`
- **Python / test runner:** `/home/danielle/miniconda3/envs/omniphyl/bin/python`
- **Primary source files to edit:**
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py`
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/io.py`
  - `/home/danielle/Documents/veg/omniphyl/tests/test_inference.py`
- **Benchmark workspace (no GPU / no model inference needed):**
  - Directory: `/home/danielle/omniphyl-work/infer_test`
  - Cached Stage-1 model output pickle (50 bp bins, stride 100, `v5_w64` checkpoint on *P. falciparum* chr1 `NC_004325.2`, 640,851 bp):
    `/home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl`
  - Reference FASTA & GFF3:
    `/home/danielle/omniphyl-work/infer_test/NC_004325.2.fasta`
    `/home/danielle/omniphyl-work/infer_test/NC_004325.2.gff3` (142 reference genes: 48 single-exon, 86 multi-exon, 8 non-coding; 208 introns total)
  - LexicMap RNA-seq evidence for `NC_004325.2`:
    `/home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/rs_full_top50k.bed`
    `/home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/rs_full.bed`
  - Decode-only harness (<2 s runtime):
    `/home/danielle/omniphyl-work/infer_test/redecode.py`
  - Standalone GFF3 comparator:
    `/home/danielle/omniphyl-work/infer_test/compare_gff.py`

### Baseline Numbers to Beat (Current `main` at commit `87bbf46`)

Running:
```bash
PYTHONPATH=/home/danielle/Documents/veg/omniphyl/src \
  /home/danielle/miniconda3/envs/omniphyl/bin/python \
  /home/danielle/omniphyl-work/infer_test/redecode.py \
  /home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl \
  /home/danielle/omniphyl-work/infer_test/NC_004325.2.gff3 \
  /tmp/baseline.gff3
/home/danielle/miniconda3/envs/omniphyl/bin/python \
  /home/danielle/omniphyl-work/infer_test/compare_gff.py \
  /tmp/baseline.gff3 \
  /home/danielle/omniphyl-work/infer_test/NC_004325.2.gff3
```
Currently produces:
- **Without LexicMap:** `predicted genes = 205` (199 single-exon, 6 multi-exon), `matched = 61`, `recall = 0.430`, `precision = 0.298`, `median |start_err| = 391 bp`, `median |stop_err| = 300 bp`, `RefSeq CDS base coverage = 0.763`.
  - Breakdown of the 205 predictions: **61** strict matches, **51** sub-fragments inside multi-exon RefSeq genes, **11** sub-fragments inside single-exon RefSeq genes, **1** marginal overlap, **22** antisense false positives (opposite strand of a real RefSeq gene), **59** intergenic false positives.
- **With LexicMap (`rs_full_top50k.bed`):** `predicted genes = 237` (10 multi-exon), `matched = 63`, `recall = 0.444`, `precision = 0.266`, `RefSeq CDS base coverage = 0.805`.

---

## 1. Task A1 — Fix Two Bugs in LexicMap Evidence Loading (`src/omniphyl/io.py`)

### Root Cause
1. In `load_lexicmap_evidence` (`src/omniphyl/io.py`, lines 237–246):
   ```python
   if tsv_path:
       raw = _parse_tsv(Path(tsv_path))
   elif bed_path:
       raw = _parse_bed(Path(bed_path))
   elif bedgraph_path:
       raw = _parse_bedgraph(Path(bedgraph_path))
   else:
       return {}
   per_hit_meta = not bedgraph_path  # BUG!
   ```
   If a caller passes **both** `bed_path` (or `tsv_path`) **and** `bedgraph_path`, `raw` is parsed from `bed_path`/`tsv_path`, but `per_hit_meta` is `False` because `bedgraph_path` is truthy. That skips per-hit filtering, overwrites `ev.density` as if BED scores were pre-aggregated bedGraph densities, and skips `ev.intron_candidates` completely (`intron_candidates = []`).
2. In `src/omniphyl/io.py`, lines 320–328:
   ```python
   for a, b in zip(group, group[1:]):
       gap = b.start - a.end
       if 35 <= gap <= 5000:
           ev.intron_candidates.append((a.end, b.start))
   ```
   `ev.intron_candidates` is never deduplicated. On `rs_full_top50k.bed`, 3,003 entries are appended for only **62 unique `(donor_end, acceptor_start)` pairs**, causing `_splice_candidates` (`src/omniphyl/annotate.py:794`) to run its nested `O(rad^2)` motif-snapping loop 3,003 times over identical intervals.

### Required Change (`src/omniphyl/io.py`)
1. Set `per_hit_meta = bool(tsv_path or bed_path)` (it is `False` **only** when falling back to `bedgraph_path` alone).
2. Deduplicate `ev.intron_candidates` while preserving support count or deterministic sorted order:
   - Count occurrences of each `(a.end, b.start)` pair across unitigs/runs, or deduplicate via ` sorted(set(ev.intron_candidates))`.
   - Even better: keep unique `(d, acc)` tuples sorted by `(-count, d, acc)` so higher-depth split alignments are evaluated first in `_splice_candidates`.

---

## 2. Task A2 — Fix Self-Defeating Intron Filter Inside `_splice_candidates` (`src/omniphyl/annotate.py`)

### Root Cause
In `_decode_genes` (`src/omniphyl/annotate.py:427–436`), coding runs `[b, b2]` are contiguous stretches where **every bin has `in_coding[x] == True`** (`cscore >= coding_threshold`).
Then `_emit_orfs` calls `_splice_candidates` on `[lo - 300, hi + 300)`.
At the end of `_splice_candidates` (`src/omniphyl/annotate.py:840–845`):
```python
ib0, ib1 = (d + B - 1) // B, acc // B
if ib1 > ib0:
    if in_coding[ib0:ib1].mean() > 0.5:
        continue
    if res.p_boundary[ib0:ib1].max() > boundary_threshold:
        continue
```
Because `[lo, hi)` is a contiguous `in_coding == True` run, **every interior bin `ib0..ib1` inside the run has `in_coding == True`**, so `in_coding[ib0:ib1].mean()` is identically `1.0 > 0.5`!
- Consequently, **100% of true introns inside a coding run that span ≥1 full interior 50 bp bin (`ib1 > ib0`, i.e., any intron ≥ ~60–100 bp) are unconditionally rejected** by line 842.
- Diagnostic confirmation on `NC_004325.2`: all 3 within-run RefSeq introns (median length 160 bp) were proposed and then killed by `in_coding[ib0:ib1].mean() > 0.5`.
- Worse, line 844 (`res.p_boundary[ib0:ib1].max() > boundary_threshold`) also rejects real introns whose splice sites trigger the boundary head `p_boundary`, because `p_boundary` fires on **any** CDS/non-CDS transition (including exon–intron junctions)!

### Required Change (`src/omniphyl/annotate.py:836–848`)
Replace the broken `in_coding[ib0:ib1].mean() > 0.5` and `p_boundary` checks on interior bins `ib0..ib1` with a **relative coding dip / frame-shift / evidence check** that distinguishes a true intron from carving out the middle of a solid single-frame exon:
1. Compute flanking exon coding support around `[d, acc)`:
   - `left_b = max(0, d // B - 1)` to `d // B + 1`
   - `right_b = acc // B` to `min(res.n_bins, acc // B + 2)`
   - `flank_c = 0.5 * (float(cscore[left_b].mean()) + float(cscore[right_b].mean()))`
2. If `ib1 > ib0`:
   - Let `int_c = float(cscore[ib0:ib1].mean())`.
   - Let `has_hint = any(abs(d - hd) <= 12 and abs(acc - ha) <= 12 for hd, ha in hints)`.
   - Let `g_lo = 0 if strand == "+" else 3`, and check whether the dominant same-strand frame shifts across the intron:
     `frame_shifts = have_frames and (int(res.p_frame[max(0, d // B), g_lo:g_lo+3].argmax()) != int(res.p_frame[min(res.n_bins - 1, acc // B), g_lo:g_lo+3].argmax()))`.
   - Reject (`continue`) **only if** the interior looks just as strongly coding in the same frame as its flanks and has no hint:
     `if not has_hint and not frame_shifts and int_c > max(0.75, flank_c - 0.05): continue`
   - Instead of rejecting on `res.p_boundary[ib0:ib1].max() > boundary_threshold` (which fires on splice junctions too), only reject if there is a strong **start or stop** structural impulse (`res.b_tracks[ib0:ib1, 0]` or `res.b_tracks[ib0:ib1, 1]`) that exceeds the splice donor/acceptor tracks (`res.b_tracks[ib0:ib1, 2:4].max()`), indicating two separate genes rather than an intron.

---

## 3. Task A3 — Cross-Run Exon Chaining & Short-Exon Extension (`src/omniphyl/annotate.py`)

### Root Cause
On `NC_004325.2`, **51 predicted single-exon genes** are actually individual exons of multi-exon RefSeq genes, and out of 208 RefSeq introns:
- **15 introns** split a gene across two adjacent `in_coding` runs (`cscore >= 0.45` in both exons, `cscore < 0.45` in the intron between them, median intron length 233 bp). Because `_decode_genes` (`src/omniphyl/annotate.py:429–464`) processes each contiguous `in_coding` run `[b, b2]` independently, `_emit_orfs` never sees both runs together.
- **150 introns** connect a main coding exon (`cscore >= 0.45`) to a short terminal exon (<120 bp, typically 54–72 bp in *Plasmodium*) whose 500 bp window average `cscore` is `0.15–0.40` (`< 0.45`).

### Required Change (`src/omniphyl/annotate.py`)
Modify `_decode_genes` and `_emit_orfs` in two clean, general ways (avoiding organism-specific thresholds):

1. **Bridge adjacent `in_coding` runs separated by `<= max_intron` in eukaryotic mode:**
   - First collect all contiguous `in_coding` bin-runs `runs = [(b_start, b_end), ...]`.
   - In addition to decoding each individual run `(b, b2)`, when `euk=True`, form **chained super-runs** by merging consecutive runs `(b_i, b2_i)` and `(b_{i+1}, b2_{i+1})` whenever the gap `(b_{i+1} - b2_i - 1) * B <= max_intron` and both runs share the same dominant strand (`p_frame[:, 0:3].sum()` vs `p_frame[:, 3:6].sum()`).
   - For each chained super-run `[lo_chain, hi_chain)`, run the eukaryotic spliced scan (`_splice_candidates` + `_emit_spliced_orfs`) across `[lo_chain, hi_chain)` where every gap bin interval between the constituent sub-runs is automatically seeded as a `(gap_start_bp, gap_end_bp, snap_radius)` proposal in `_splice_candidates` (with `snap_radius = min(150, B * 3)` so the donor/acceptor search reaches into the edges of the flanking coding runs).
   - Any multi-exon ORF emitted across the super-run gets scored by `_score_orf` (`body * log1p(cds_len) + 0.5 * (imp + step) + prior`). Because `log1p(cds_len)` rewards the combined exon length and `_exon_frames` verifies that each exon's implied reading frame matches `res.p_frame`, a true multi-exon chain naturally outscores its individual single-exon sub-fragments in `_nonoverlap_select`!

2. **Widen the 5'/3' splice-scan margin (`s_lo`, `s_hi`) in eukaryotic mode to capture short terminal exons:**
   - In `_emit_orfs` (`src/omniphyl/annotate.py:670–671`), `s_lo = max(0, lo - 300)` and `s_hi = min(L, hi + 300)`.
   - A 60 bp 5' leader exon separated by a 180–400 bp intron from the main coding run sits `240–460 bp` upstream of `lo`, just outside `lo - 300`, and its bin has `in_coding == False`.
   - When `euk=True`, widen the spliced-scan window on the transcript-5' side to `min(max_intron, 600)` bp (i.e., `lo - 600` on `+`, `hi + 600` on `-`), and in `_splice_candidates` seed a candidate intron whenever a upstream/downstream bin within 600 bp has local same-strand frame mass `res.p_frame[b, g_lo:g_lo+3].sum() >= 0.20` (even if `cscore[b] < coding_threshold`).

---

## 4. Task A4 — Cross-Strand Overlap Arbitration & Short Intergenic FP Filter (`src/omniphyl/annotate.py`)

### Root Cause
1. **22 Antisense False Positives:**
   In `_decode_genes` (`src/omniphyl/annotate.py:442`), both `+` and `-` strands are decoded in every coding run. In `_score_orf` (`src/omniphyl/annotate.py:572–576`), a candidate on the minority strand is only rejected if its strand has `< 25%` of total frame mass (`own < 0.25 * (s_pos + s_neg)`). And in `_nonoverlap_select` (`src/omniphyl/annotate.py:480–495`):
   ```python
   for k in kept:
       if k["strand"] != g["strand"]:
           continue
   ```
   `_nonoverlap_select` **never compares opposite-strand candidates**. So whenever a real gene on `+` has a stop-free shadow ORF on `-` whose bins have e.g. 26% minus-strand frame mass (or noisy edge bins), both the real `+` gene and the fake `-` shadow gene are emitted.
2. **59 Intergenic False Positives:**
   Short single-exon ORFs (e.g. 90–180 bp, 2–3 bins) in AT-rich intergenic regions that barely clear `cscore >= 0.45` with weak `body` frame support (`body < 0.30`) are accepted on the same footing as 1,500 bp high-confidence genes.

### Required Change (`src/omniphyl/annotate.py`)
1. **In `_score_orf` (`src/omniphyl/annotate.py:569–598`):**
   - Store `g["strand_share"] = own / max(1e-9, s_pos + s_neg)` when `have_frames` is `True` (else `1.0`).
   - Add a length-aware minimum body support gate for **short unspliced** candidates: if `not g.get("introns")` and `cds_len < 180` and `body < 0.35`, return `-1.0`. (True short single-exon genes have sharp frame signal; marginal 90–150 bp intergenic ORFs do not.)
2. **In `_nonoverlap_select` (`src/omniphyl/annotate.py:480–495`):**
   - Keep the existing same-strand overlap suppression (`ov > max_overlap_frac * span`).
   - For **opposite-strand** overlaps (`k["strand"] != g["strand"]` where `k` is an already-kept higher-scoring gene):
     If `ov > 0.5 * span` (i.e. more than half of the lower-scoring candidate `g` lies inside the higher-scoring opposite-strand gene `k`), drop `g` unless `g` has strong independent strand evidence (`g.get("strand_share", 0.5) >= 0.45` and `g["score"] >= 0.8 * k["score"]`).
   - Why this respects `test_adjacent_opposite_strand_genes`: adjacent opposite-strand genes do **not** overlap by >50% of their span (they sit side-by-side in a merged run), so `ov <= 0.5 * span` and both are preserved!

---

## 5. Verification & Acceptance Criteria

### Step 1 — Unit Tests (`tests/test_inference.py`)
Run existing + new unit tests:
```bash
PYTHONPATH=/home/danielle/Documents/veg/omniphyl/src \
  /home/danielle/miniconda3/envs/omniphyl/bin/python \
  /home/danielle/Documents/veg/omniphyl/tests/test_inference.py
```
All 21 existing tests **must** pass (especially `test_adjacent_opposite_strand_genes`, `test_frameshift_intron`, `test_upstream_start_rejected`, `test_rev_frame_grid`), plus add 3 new unit tests in `tests/test_inference.py`:
1. `test_antisense_shadow_orf_suppressed`: a strong `+` gene (`p_frame[:, 0] = 0.72`) and a co-located `-` shadow ORF (`p_frame[:, 3] = 0.26`) on the same span -> only the `+` gene is emitted.
2. `test_within_run_long_intron_not_killed`: a 2-exon gene whose 150 bp GT..AG intron spans 2 interior bins inside an all-coding run (`cscore = 0.65` in intron vs `0.85` in exons, with frame shift) -> spliced 2-exon gene is emitted.
3. `test_cross_run_exon_chaining`: a 2-exon gene whose 250 bp GT..AG intron dips `cscore = 0.20` (< `0.45`) for 4 bins between two strong coding runs -> chained into a single 2-exon gene rather than 2 separate single-exon genes.

### Step 2 — RefSeq Benchmark via `redecode.py` & `compare_gff.py`
Run the decode-only benchmark on `/home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl`:
```bash
cd /home/danielle/omniphyl-work/infer_test
PYTHONPATH=/home/danielle/Documents/veg/omniphyl/src \
  /home/danielle/miniconda3/envs/omniphyl/bin/python \
  redecode.py chr1_s100_bauto.res.pkl NC_004325.2.gff3 /tmp/tier_a.gff3
/home/danielle/miniconda3/envs/omniphyl/bin/python \
  compare_gff.py /tmp/tier_a.gff3 NC_004325.2.gff3
```
**Acceptance criteria against baseline (`recall = 0.430`, `precision = 0.298`, `CDS cov = 0.763`, `preds = 205`):**
- Recall must be `>= 0.430` (ideally higher as split multi-exon fragments merge into full-length matches).
- Precision must improve significantly above `0.298` (target `>= 0.35+` from suppressing the 22 antisense FPs, filtering short weak intergenic ORFs, and collapsing split sub-fragment calls).
- Antisense false positives (predicted genes overlapping a RefSeq gene only on the opposite strand) must drop from `22` to `<= 5`.
- Multi-exon gene calls must increase from `6` while maintaining or improving RefSeq CDS base coverage (`>= 0.75`).
- Test each sub-change (A2, A3, A4) incrementally with `redecode.py` and keep only components that improve or preserve the RefSeq benchmark metrics without overfitting to *P. falciparum*.

## 6. Implementation Results (2026-11)

All of A1–A4 implemented. 24/24 tests pass (3 new:
`test_decode_antisense_suppression`, `test_decode_intron_inside_coding_run`,
`test_decode_chained_runs`).

### Deviations from the plan above
- **Terminal-exon seed threshold 0.35, not 0.20**: measured on the chr1
  pkl, non-coding bins carry median same-strand frame mass 0.227 — the
  plan's 0.20 would propose from ~half of all non-coding bins. Real
  short terminal exons (<120 bp) carry median 0.17 max strand mass, so
  most are undetectable regardless; 0.35 only seeds where signal exists.
- **`p_boundary` interior check replaced by structural-head comparison**:
  splice sites themselves fire `p_boundary` (any CDS/non-CDS transition),
  so it cannot discriminate gene boundaries from junctions. The new check
  rejects a candidate only when interior start/stop impulse mass
  (`b_tracks[:, :2]`) exceeds interior splice mass (`b_tracks[:, 2:4]`).
- **Score-first dedupe**: chained spliced calls can share (start, end,
  strand) with a per-run unspliced call; dedupe now keeps the
  higher-scored variant instead of first-seen.
- **`redecode.py` gained `--lex-bed=<path>`**: loads LexicMap evidence,
  fuses density/endpoints (w_ev=0.35) mirroring `run_annotate`, and
  passes intron candidates as hints.

### Benchmark vs acceptance criteria (chr1, NC_004325.2, decode-only)
| metric | baseline | Tier A no-ev | Tier A +LexicMap | criterion |
|---|---|---|---|---|
| predictions | 205 | 172 | 186 | — |
| recall | 0.430 | 0.444 | 0.458 | >= 0.430 PASS |
| precision | 0.298 | 0.366 | 0.349 | >= 0.35 PASS |
| antisense FP | 22 | 3 | — | <= 5 PASS |
| multi-exon calls | 6 | 27 | — | increase PASS |
| RefSeq CDS cov | 0.763 | 0.787 | 0.824 | >= 0.75 PASS |

Outputs: `/home/danielle/omniphyl-work/infer_test/tierA_noev.gff3`,
`tierA_ev.gff3`.
