# Tier B Implementation Plan: Softmasking Support, LexicMap Evidence Leveraging & Kaggle CPU Sweep

This document is a self-contained implementation specification for an independent agent with **no prior conversation context**. It specifies exact files, line numbers, root-cause diagnostics, algorithms, unit tests, and Kaggle CPU job instructions (to avoid running multi-hundred-MB BED sweeps on the local machine).

---

## 0. Environment, Repository & Hardware Constraints

- **Hardware constraint (CRITICAL):** The local machine runs a RAM-hungry IDE.
  - **DO NOT** run `load_lexicmap_evidence` or Python scripts on large BED files (`chr5_annot.bed` at 110 MB / 1.69M rows, or `rs_full.bed` at 472 MB / ~7M rows) locally until streaming memory fixes are in place, and prefer running any multi-arm BED/threshold sweeps as **Kaggle CPU kernels**.
  - Running unit tests (`tests/test_inference.py`, <2 s, ~40 MB RAM) and single-pass `redecode.py` on `chr1_s100_bauto.res.pkl` + `rs_full_top50k.bed` (3.3 MB) is lightweight, but all threshold/dataset grid sweeps belong on Kaggle CPU.
- **Repository root:** `/home/danielle/Documents/veg/omniphyl`
- **Python / test runner:** `/home/danielle/miniconda3/envs/omniphyl/bin/python`
- **Primary source files to edit:**
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/io.py`
  - `/home/danielle/Documents/veg/omniphyl/src/omniphyl/annotate.py`
  - `/home/danielle/Documents/veg/omniphyl/tests/test_inference.py`
  - `/home/danielle/omniphyl-work/infer_test/redecode.py`
- **Benchmark & LexicMap files:**
  - Cached Stage-1 model output pickle (50 bp bins, stride 100, `v5_w64` checkpoint on *P. falciparum* chr1 `NC_004325.2`, 640,851 bp):
    `/home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl`
  - Reference FASTA & GFF3 (142 reference genes: 48 single-exon, 86 multi-exon, 8 non-coding; 208 introns):
    `/home/danielle/omniphyl-work/infer_test/NC_004325.2.fasta`
    `/home/danielle/omniphyl-work/infer_test/NC_004325.2.gff3`
  - LexicMap BED files for `NC_004325.2`:
    - `/home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/rs_full_top50k.bed` (3.3 MB, 50,000 HSPs truncated at bitscore `>= 944`)
    - `/home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/chr5_annot.bed` (110 MB, 1,692,651 HSPs with bitscore `>= 300`)
    - `/home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/rs_full.bed` (472 MB, unfiltered HSPs)

### Current Tier A Baseline (Commit + Tier A edits)
- **Tier A without LexicMap:** `preds = 172`, `matched = 63`, `recall = 0.444`, `precision = 0.366`, `antisense FP = 3`, `multi-exon = 27`, `RefSeq CDS cov = 0.787`.
- **Tier A with `rs_full_top50k.bed`:** `preds = 186`, `matched = 65`, `recall = 0.458`, `precision = 0.349`, `RefSeq CDS cov = 0.824`.

---

## 1. Root-Cause Diagnosis: Why LexicMap Currently Helps So Little

Six interacting bugs and design bottlenecks explain why adding LexicMap currently moves recall by only `+1.4%` (`+2` genes) while **reducing** precision (`0.366 -> 0.349`):

1. **Softmasked lowercase repeats break both `io.py` and `annotate.py` (Task B1):**
   - `iter_sequences` (`src/omniphyl/io.py:65–75`) and `revcomp` (`src/omniphyl/io.py:21–25`) preserve lowercase characters (`a, c, g, t`) from softmasked FASTA inputs.
   - While Stage-1 `encode.py` (`BASE2DIGIT`, lines 17–19) maps `acgt` to `0..3` properly, Stage-2 `_decode_genes` (`src/omniphyl/annotate.py:405`) operates on raw `seq` with **case-sensitive** string comparisons:
     - `_scan_codons` (`annotate.py:373`): `s2[i:i + 3] not in cands` misses any start/stop codon containing a lowercase base.
     - `_orf_extends` (`annotate.py:396`): `s2[i:i + 3] in stops` **fails to recognize lowercase in-frame stop codons (`taa`, `tag`, `tga`)**, causing ORFs to walk straight through internal stop codons in softmasked regions!
     - `_splice_candidates` (`annotate.py:898–918`): `seq[d:d + 2] == "GT"` / `"AG"` / `"CT"` / `"AC"` misses any splice junction in softmasked sequence.
     - `_emit_spliced_orfs` (`annotate.py:1015, 1018`): `tseq[t:t + 3] != "ATG"` and `tseq[u:u + 3] not in stops` fail on lowercase bases.
   - Meanwhile, `load_lexicmap_evidence` (`src/omniphyl/io.py:269–272`) counts lowercase bases (`b.islower()`) as ambiguous `N` bases and rejects any HSP with `>20%` softmasked bases, plus rejects HSPs with `_dinuc_entropy(sub) < 1.8` (which drops genuine AT-rich *Plasmodium* coding exons whose 80%-AT sequence has dinucleotide entropy `1.5–1.8 bits`).

2. **Stray `break` inside `_splice_candidates` acceptor loop (`src/omniphyl/annotate.py:929`, Task B2):**
   ```python
   for acc in range(max(a0 - rad, d + min_intron),
                    min(a0 + rad, d + max_intron, hi) + 1):
       ...
       if best is None or sc > best[0]:
           best = (sc, d, acc)
       break
   ```
   - `break` at `annotate.py:929` is unconditionally executed on the **first** valid acceptor motif (`AG` on `+`, `AC`/`GC` on `-`) in `[a0 - rad, a0 + rad]`.
   - For every donor `d`, it stops at the leftmost `AG` in `[a0 - rad, a0 + rad]` and **never checks `a0` itself** or higher-scoring acceptors to the right!
   - Also, at `annotate.py:898` and `910`, `canon` is mutated in-place (`canon = canon and seq[acc - 2:acc] == "AG"`), which corrupts `canon` across acceptor iterations once `break` is removed unless `d_canon` (donor) and `canon` (pair) are separated.

3. **`_score_orf` and `_better_shared_stop` ignore `dtilde` and `intron_hints` when `have_frames == True` (`src/omniphyl/annotate.py:621–710`, Task B3):**
   - When `have_frames` is true (all v5/v6 models), `_score_orf` (`line 622`) computes `body` strictly from `res.p_frame[b0:b1, gx]`, and `_better_shared_stop` (`line 704`) computes 5' extension support `sup` strictly from `res.p_frame[ext_bins, g_orf]`.
   - Neither uses `res.dtilde` or checks whether an intron is corroborated by LexicMap `intron_hints`!
   - Consequently, if LexicMap split-read evidence proposes a true intron linking a main exon (`p_frame = 0.80`) to a shorter or weaker exon (`p_frame = 0.30`), adding the second exon *lowers* the mean `body` score (`(0.80 + 0.30)/2 < 0.80`), and `_score_orf` / `_better_shared_stop` prefer the truncated single-exon fragment over the LexicMap-corroborated multi-exon gene!

4. **`cscore` fusion (`src/omniphyl/annotate.py:422–425`) spawns unstranded UTR/intergenic FPs while failing to reach exons outside `in_coding` runs (Task B4):**
   - Currently, `cscore = exp(log(coding) + w_ev * utr_gate * dtilde)`.
   - Because $\exp(0.35) = 1.42$, `cscore` can only cross `coding_threshold = 0.45` when `coding >= 0.32`. In UTRs or intergenic regions where `coding` is `0.33–0.44` and `dtilde` is high, `cscore` crosses `0.45` and creates **new standalone `in_coding` runs** that emit false-positive single-exon ORFs (`preds` jumps `172 -> 186`, dropping precision from `0.366 -> 0.349`).
   - Conversely, when a real short partner exon has `coding = 0.20` (`cscore < 0.45`), it never forms a run, and even if a LexicMap split-alignment `(hd, ha)` points directly to it across an intron `> 600 bp`, `_decode_genes` never expands the adjacent coding run's search span `[lo, hi)` to include the partner exon!

5. **Global bitscore truncation (`rs_full_top50k.bed` at `score >= 944`) destroys split alignments, while unfiltered BED loading (`io.py`) blows up RAM (Task B5):**
   - In `rs_full_top50k.bed`, the minimum bitscore is `944` (HSP length `>= 520 bp`). Any exon `< 500 bp` has bitscore `< 900` and was stripped out. For example, in `chr5_annot.bed` (lines 8–9), unitig `SRR13250831_9609` has `hsp8` (`408557..410135`, score `1000`) and `hsp10` (`410552..410954`, score `719`) spanning a `417 bp` intron; `rs_full_top50k.bed` kept `hsp8` and discarded `hsp10`, destroying the intron candidate!
   - Conversely, `chr5_annot.bed` (1.69M rows) and `rs_full.bed` (~7M rows) cannot be loaded naively because `_parse_bed` builds a full in-memory list of millions of `Hsp` objects, and single-HSP hits at low bitscores (`150–350`) include distant-species homologs and repeat pileups.
   - **Solution:** Two-tier unitig-aware filtering + streaming aggregation in `load_lexicmap_evidence`:
     - Keep a **lower per-HSP floor** (`score >= min_score` default `150`, `len >= 45 bp`) for **multi-HSP colinear chains** from the same `(sgenome, sseqid, strand)` unitig (so short exons paired with a longer exon on the same transcript unitig are preserved!).
     - Apply a **stricter singleton floor** (`score >= max(min_score, 350.0)`, `len >= 100 bp`) to **isolated single-HSP unitigs** (suppressing distant-species / repeat noise without losing spliced partner exons).
     - Stream/release `ev.hsps` after building `density`, `endpoints`, and `intron_candidates` so memory stays `< 30 MB` regardless of BED size.

---

## 2. Step-by-Step Code Changes

### Task B1 — Softmasking Support & Entropy/Endpoint Fixes in `src/omniphyl/io.py` and `src/omniphyl/annotate.py`

1. **In `src/omniphyl/annotate.py`:**
   - At the start of `_decode_genes` (`src/omniphyl/annotate.py:417`), normalize `seq` once to uppercase before any codon or splice scanning:
     ```python
     seq = seq.upper()
     L = res.length
     ```
   - In `run_annotate` Step 4b genetic-code inference (`src/omniphyl/annotate.py:253`), use `seq = contig_seqs[res.contig].upper()` so codon walking is uppercase throughout.
   - Verify `_n_frac(seq)` (`src/omniphyl/annotate.py:62–63`) already uses `b.upper() not in "ACGT"` (it does: softmasked `acgt` are not treated as `N`).

2. **In `src/omniphyl/io.py` (`load_lexicmap_evidence`, lines 266–276):**
   - Replace the softmask-as-ambiguous check (`sum(1 for b in sub if b.islower())`) so lowercase softmasked bases are **not** counted as ambiguous `N`s:
     ```python
     if seq is not None:
         sub = seq[max(0, h.start):min(clen, h.end)]
         if sub:
             ambig = sum(1 for b in sub if b.upper() not in "ACGT")
             if ambig > 0.2 * len(sub):
                 continue
             # 1.5 bits allows genuine AT-rich (80% AT) coding exons while
             # still rejecting homopolymers and dinucleotide microsatellites
             if _dinuc_entropy(sub) < 1.5:
                 continue
     ```
   - In `src/omniphyl/io.py:321–322`, normalize `ev.endpoints` by the 95th percentile of non-zero endpoint bins rather than `ev.endpoints.max()` so a single high-depth locus does not squash all other splice-junction impulses to ~0:
     ```python
     nz = ev.endpoints[ev.endpoints > 0]
     if len(nz):
         scale = max(float(np.percentile(nz, 95)), 1.0)
         ev.endpoints = np.minimum(1.0, ev.endpoints / scale)
     ```

---

### Task B2 — Fix `_splice_candidates` Acceptor `break` Bug & Hint Bonus (`src/omniphyl/annotate.py`)

In `_splice_candidates` (`src/omniphyl/annotate.py:839–935`):
1. Preserve whether a candidate came from `hints` (split-read evidence) vs heuristic proposals:
   - Let `hint_xy = [(iv[0], iv[1]) for iv in hints if len(iv) == 2]`.
2. In the motif-snapping loop (`lines 893–931`):
   - Compute donor canonicality (`d_canon`) in the outer `for d` loop and pair canonicality (`canon = d_canon and ...`) in the inner `for acc` loop.
   - **Remove the stray `break` at `line 929`** so all valid acceptor positions in `[max(a0 - rad, d + min_intron), min(a0 + rad, d + max_intron, hi)]` are evaluated and the highest-scoring `(sc, d, acc)` is kept.
   - Add a **direct split-read evidence bonus** (`+1.5`) to `sc` when `(d0, a0)` is a 2-tuple LexicMap hint (`rad == 12` and `(d0, a0)` in `hint_set`), and increase the distance tie-breaker on hints (`-0.02 * (abs(d - d0) + abs(acc - a0))`) so snapping strongly prefers the canonical `GT..AG` / `CT..AC` closest to the exact split-alignment coordinates:
     ```python
     hint_set = {(iv[0], iv[1]) for iv in hints if len(iv) == 2}
     scored = []
     for d0, a0, rad in cand:
         is_hint = (d0, a0) in hint_set and rad <= 12
         best = None
         for d in range(max(lo + 3, d0 - rad),
                        min(hi - min_intron, d0 + rad + 1)):
             if strand == "+":
                 d_canon = seq[d:d + 2] == "GT"
                 if seq[d:d + 2] not in ("GT", "GC"):
                     continue
             else:
                 d_canon = seq[d:d + 2] == "CT"
                 if seq[d:d + 2] != "CT":
                     continue
             for acc in range(max(a0 - rad, d + min_intron),
                              min(a0 + rad, d + max_intron, hi) + 1):
                 if strand == "+":
                     if seq[acc - 2:acc] != "AG":
                         continue
                     canon = d_canon
                 else:
                     if seq[acc - 2:acc] not in ("AC", "GC"):
                         continue
                     canon = d_canon and seq[acc - 2:acc] == "AC"
                 sc = (2.0 if canon else 1.0) + (1.5 if is_hint else 0.0)
                 sc += float(d_track[min(res.n_bins - 1, d // B)])
                 sc += float(a_track[min(res.n_bins - 1, acc // B)])
                 ib0, ib1 = (d + B - 1) // B, acc // B
                 if ib1 > ib0:
                     sc += float(res.p_region[ib0:ib1, 1].mean())
                 pen = 0.02 if is_hint else 0.001
                 sc -= pen * (abs(d - d0) + abs(acc - a0))
                 if best is None or sc > best[0]:
                     best = (sc, d, acc)
         if best:
             scored.append(best)
     ```
3. In the interior start/stop impulse veto (`lines 966–968`):
   - Do **not** let a noisy interior start/stop impulse veto a junction that has direct split-alignment evidence (`has_hint`):
     ```python
     if (not has_hint
             and res.b_tracks[ib0:ib1, :2].max()
             > res.b_tracks[ib0:ib1, 2:4].max()):
         continue
     ```

---

### Task B3 — Evidence-Aware ORF Scoring & Shared-Stop Arbitration (`src/omniphyl/annotate.py`)

Currently `_score_orf` (`lines 581–664`) and `_better_shared_stop` (`lines 667–710`) ignore `res.dtilde` and `intron_hints` whenever `have_frames == True`.

1. **Pass `w_ev` and `hint_xy` (or attach `res.intron_hints`) so `_score_orf` and `_better_shared_stop` can see evidence:**
   - Simplest clean interface: store `hint_xy = [(iv[0], iv[1]) for iv in intron_hints if len(iv) == 2]` and `w_ev` on `res` (or pass via `cand["hint_introns"]` count computed when `cand` is built in `_emit_spliced_orfs`).
   - Specifically, in `_emit_spliced_orfs` (`annotate.py:1028–1040`), when building `cand`:
     ```python
     n_hint = sum(1 for d, acc in ivs
                  if any(abs(d - hd) <= 12 and abs(acc - ha) <= 12
                         for hd, ha in hint_xy))
     cand = {"start": a, "end": bnd, "strand": strand, "frame": None,
             "introns": ivs, "n_hint_introns": n_hint, "domains": []}
     ```
2. **In `_score_orf(seq, res, cscore, have_frames, g)` (`src/omniphyl/annotate.py:621–664`):**
   - When `have_frames` is true, blend strand-consistent RNA-seq coverage `res.dtilde` into `body` on exonic bins where the candidate's reading frame already carries non-trivial support, and add a splice-evidence bonus for each LexicMap-corroborated intron:
     ```python
     if have_frames:
         raw_body = float(sum(res.p_frame[b0:b1, gx].sum()
                              for gx, b0, b1 in ex_bins)) / n_bins_tot
         s_pos = float(res.p_frame[mask, :3].sum())
         s_neg = float(res.p_frame[mask, 3:6].sum())
         own = s_pos if strand == "+" else s_neg
         if own < 0.25 * (s_pos + s_neg):
             return -1.0
         strand_share = own / max(1e-9, s_pos + s_neg)
         g["strand_share"] = strand_share
         # Evidence boost on exonic bins: gated by strand_share so
         # unstranded RNA-seq coverage only boosts the dominant strand
         ev_cov = float(res.dtilde[mask].mean()) if res.dtilde is not None else 0.0
         body = raw_body + 0.15 * strand_share * ev_cov
     ```
   - Add `0.4 * g.get("n_hint_introns", 0)` to `g["score"]` at `line 662`:
     ```python
     g["score"] = (body * math.log1p(cds_len)
                   + 0.5 * (imp + step) + prior
                   + 0.4 * g.get("n_hint_introns", 0))
     ```
     Why `+0.4` per hint-corroborated intron: it is enough to prevent a real multi-exon gene with a slightly weaker short terminal exon from losing to its own single-exon sub-fragment, while still requiring valid splice motifs, reading-frame continuity (no stops across the spliced transcript), and `>=25%` strand dominance.
3. **In `_better_shared_stop` (`src/omniphyl/annotate.py:667–710`):**
   - If the longer candidate `lng` contains a LexicMap-corroborated intron (`lng.get("n_hint_introns", 0) > sht.get("n_hint_introns", 0)`) spanning the 5' extension, include `0.20 * float(res.dtilde[ext_bins].mean())` in `sup` so a short 5' leader exon backed by a literal split-read intron is not rejected when its window-diluted `p_frame` is `0.15–0.25`:
     ```python
     if have_frames:
         ...
         sup = float(res.p_frame[ext_bins, g_orf].mean())
         if lng.get("n_hint_introns", 0) > sht.get("n_hint_introns", 0):
             sup += 0.25 * float(res.dtilde[ext_bins].mean())
     ```

---

### Task B4 — Stop Unstranded `dtilde` From Spawning UTR/Intergenic False-Positive Runs & Bridge Hint-Linked Exons (`src/omniphyl/annotate.py`)

Look at why precision dropped from `0.366 -> 0.349` (`172 -> 186` predictions) when enabling LexicMap:
- In `_decode_genes` (`annotate.py:422–428`), `cscore` boosts `coding` everywhere `coding >= 0.32` and `dtilde > 0`.
- When a 5'/3' UTR or non-coding bin has `coding = 0.35` and `dtilde = 1.0`, `cscore` crosses `0.45` and creates a **new standalone run** `(b0, b1)` where the model itself had `coding < coding_threshold` on every bin!
- Conversely, when a real multi-exon gene has a split-alignment hint `(hd, ha)` whose outer exon sits `600–2500 bp` away from the main coding run in bins with `coding < 0.45`, `_emit_orfs` only scans `m5 = 600 bp` (`line 740`) and never reaches the outer exon!

**Required Change in `_decode_genes` (`src/omniphyl/annotate.py:422–475`):**
1. **Gate standalone runs on model coding + frame support:**
   - Keep `cscore` as defined, **but** drop any standalone run `(b0, b1)` that was pushed over `coding_threshold` *solely* by `dtilde` (i.e. `coding[b0:b1 + 1].max() < coding_threshold`) **unless** it either:
     - is chained to an adjacent run via `spans` / `intron_hints`, OR
     - has strong concentrated single-frame support (`res.p_frame[b0:b1 + 1, :6].max(1).mean() >= 0.40`).
   - This immediately eliminates the ~14 spurious UTR/ncRNA single-exon false positives spawned by unstranded RNA-seq coverage!
2. **Expand span search bounds (`lo`, `hi`) when a LexicMap split-alignment hint `(hd, ha)` anchors inside a coding run:**
   - For any coding run `[r_lo, r_hi)` in `spans`, check if any 2-tuple `(hd, ha)` in `intron_hints` has one junction endpoint inside `[r_lo - B, r_hi + B]` and the other endpoint within `max_intron`:
     - If `r_lo - B <= ha <= r_hi + B` and `r_lo - hd <= max_intron`: extend the span's left bound to `max(0, hd - 300)` (so `_splice_candidates` and `_emit_spliced_orfs` cover `[hd, ha)` and up to 300 bp of the upstream partner exon!).
     - If `r_lo - B <= hd <= r_hi + B` and `ha - r_hi <= max_intron`: extend the span's right bound to `min(L, ha + 300)`.
   - Also, if `(hd, ha)` bridges two separate runs `runs[i]` and `runs[j]` (even across a gap where an intermediate run or weak bin sat), emit a fused span `[runs[i][0] * B, min((runs[j][1] + 1) * B, L))` seeded with `(hd, ha)`.

---

### Task B5 — Two-Tier Unitig-Aware Filtering & Low-RAM Streaming in `load_lexicmap_evidence` (`src/omniphyl/io.py`)

To make full/untruncated LexicMap files (`chr5_annot.bed`, `rs_full.bed`) usable without drowning in distant-species/repeat singletons or running out of RAM:

1. **Stream and group by unitig `(contig, sgenome, sseqid, strand)` without storing all raw `Hsp` objects in memory:**
   - In `src/omniphyl/io.py`, LexicMap TSV and BED outputs are naturally emitted per `(sgenome, sseqid)` unitig (or can be grouped by `(sseqid, strand)` per contig using lightweight `(start, end, int_score)` tuples instead of full `Hsp` dataclasses).
   - Add parameter `min_single_score: float = 350.0` (or `max(min_score, 300.0)`) and lower default `min_len: int = 45` for multi-HSP unitigs:
     - **Multi-HSP unitig (`len(group) >= 2` with at least one colinear gap in `[35, 5000]` and at least one anchor HSP with `score >= min_single_score`):** keep all partner HSPs in the group down to `score >= min_score` (`150.0`) and `length >= 45 bp`! This rescues short 50–250 bp exons that belong to a spliced transcript whose main exon had a high bitscore.
     - **Singleton unitig (`len(group) == 1` or no valid intron gap):** require `score >= max(min_score, min_single_score)` and `length >= max(min_len, 90)` so short low-bitscore hits from distant relatives or repetitive elements are filtered out.
2. **Require minimum split-read corroboration (`min_intron_support`) when total HSP count is huge:**
   - When a BED file has `>100,000` kept HSPs on a contig (like `chr5_annot.bed` or `rs_full.bed`), a single noisy alignment (`support == 1`) can propose a false gap; require `support >= 2` (at least 2 unitigs/runs corroborating the `(d, acc)` junction within `+-3 bp`) when `len(kept) > 100_000`, or rank `intron_candidates` by support and cap at top `500` per contig so `_splice_candidates` stays sub-second.
3. **Do not retain `ev.hsps = kept` by default (`keep_hsps: bool = False`):**
   - Nothing in `annotate.py` reads `ev.hsps`! Dropping `ev.hsps` (or setting `ev.hsps = []` unless requested) frees hundreds of megabytes of RAM immediately after `load_lexicmap_evidence` finishes a contig.

---

## 3. Unit Tests to Add (`tests/test_inference.py`)

Add 3 unit tests in `/home/danielle/Documents/veg/omniphyl/tests/test_inference.py`:
1. **`test_softmasked_sequence_decoding_and_evidence`:**
   - Construct a synthetic 2-exon gene where half of exon 1, the `gt..ag` splice sites, and an intronic `taa` stop codon (plus the terminal `taa` stop codon) are lowercase (`a, c, g, t`), and feed a matching LexicMap BED hit over the softmasked sequence.
   - Assert that `load_lexicmap_evidence` keeps the HSP (does not reject lowercase as ambiguous `N`s) and `_decode_genes` emits the exact 2-exon gene (snapping lowercase `gt..ag` and stopping at the lowercase `taa`).
2. **`test_splice_candidates_acceptor_break_fix`:**
   - Construct a sequence with a true acceptor `AG` at `a0 = 600` (matching a LexicMap hint `(450, 600)`) and a decoy `AG` 8 bp earlier at `592` that produces an immediate in-frame stop codon.
   - Before Task B2, the stray `break` at `annotate.py:929` snapped only to `592` and missed `600`. Assert that `_splice_candidates` now snaps to `(450, 600)` and `_decode_genes` recovers the full 2-exon ORF.
3. **`test_lexicmap_hint_rescues_weak_partner_exon`:**
   - Construct a 2-exon `+` gene where exon 1 (`200..320`, 120 bp) has weak model coding (`p_coding = 0.25`, `p_frame[:, g] = 0.22`, below `0.45`) and exon 2 (`750..1450`, 700 bp) has strong coding (`p_coding = 0.85`), linked by a LexicMap split-alignment hint `(320, 750)` and `dtilde = 0.9` on exon 1.
   - Assert that with the LexicMap hint and `dtilde`, `_decode_genes` bridges the `430 bp` intron into the weak upstream exon and emits the full `200..1450` 2-exon gene rather than truncating at an internal `ATG` inside exon 2.

---

## 4. Kaggle CPU Sweep Specification (For `chr5_annot.bed` / `rs_full.bed` Threshold Tuning)

**Why Kaggle CPU:** `chr5_annot.bed` (110 MB, 1.69M HSPs) and `rs_full.bed` (472 MB) are too large to sweep interactively on the local machine alongside the IDE.

### Step 1 — Local Verification on `rs_full_top50k.bed` (Lightweight, <3 s, <50 MB RAM)
After implementing Tasks B1–B5, run the unit test suite and the fast decode-only benchmark on `rs_full_top50k.bed`:
```bash
/home/danielle/miniconda3/envs/omniphyl/bin/python -m pytest \
  /home/danielle/Documents/veg/omniphyl/tests/test_inference.py -q

cd /home/danielle/omniphyl-work/infer_test
/home/danielle/miniconda3/envs/omniphyl/bin/python redecode.py \
  chr1_s100_bauto.res.pkl NC_004325.2.gff3 \
  --lex-bed=/home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/rs_full_top50k.bed \
  tierB_top50k.gff3
/home/danielle/miniconda3/envs/omniphyl/bin/python compare_gff.py \
  tierB_top50k.gff3 NC_004325.2.gff3
```
Note: Also update `redecode.py` (`lines 97–105`) so it includes the repeat damper (`lines 197–203` of `annotate.py`) and matches `run_annotate` 1:1.

### Step 2 — Upload `chr5_annot.bed.gz` to `daniellecallan/omniphyl-infer-test` & Push a Kaggle CPU Sweep Kernel
To find the optimal `(min_single_score, min_partner_score, min_intron_support, w_ev)` operating point on the un-truncated 1.69M-row `chr5_annot.bed`:
1. Compress `chr5_annot.bed` (or a pre-filtered unitig-chain + `score >= 300` subset) into `/home/danielle/omniphyl-work/infer-test-dataset/` alongside `chr1_s100_bauto.res.pkl`, and version-bump the Kaggle dataset:
   ```bash
   cp /home/danielle/omniphyl-work/infer_test/chr1_s100_bauto.res.pkl \
      /home/danielle/omniphyl-work/infer-test-dataset/
   gzip -c /home/danielle/Documents/brc-analytics/brc-tools/data/lexicmap_pf8_chr5/chr5_annot.bed \
      > /home/danielle/omniphyl-work/infer-test-dataset/chr5_annot.bed.gz
   kaggle datasets version -p /home/danielle/omniphyl-work/infer-test-dataset \
      -m "Add chr1_s100_bauto.res.pkl and chr5_annot.bed.gz for decode sweep"
   ```
2. Create `/home/danielle/Documents/veg/omniphyl/kaggle/lexicmap-sweep/kernel-metadata.json` (`enable_gpu: false`, `dataset_sources: ["daniellecallan/omniphyl-infer-test"]`) and `kernel.py` that sweeps:
   - `min_single_score` $\in \{300, 400, 500, 700\}$
   - `min_score` (multi-HSP partner floor) $\in \{150, 250, 300\}$
   - `min_intron_support` $\in \{1, 2, 3, 5\}$
   - `w_ev` $\in \{0.25, 0.35, 0.50\}$
   and prints a TSV summary of `(preds, matched, recall, precision, multi_exon, exact_introns_matched, cds_cov)` for each arm in `/kaggle/working/sweep_results.tsv`.

### Acceptance Criteria for Tier B
- **Softmasking invariance:** `test_softmasked_sequence_decoding_and_evidence` passes; decoding an all-lowercase or softmasked FASTA produces identical gene calls and LexicMap evidence tracks to an all-uppercase FASTA.
- **No precision regression when adding LexicMap:** On `chr1_s100_bauto.res.pkl` + `rs_full_top50k.bed`, precision with LexicMap must be `>= 0.366` (matching or beating the no-evidence precision instead of dropping to `0.349`), while recall is `>= 0.458` and RefSeq CDS coverage is `>= 0.80`.
- **Memory bound:** `load_lexicmap_evidence` does not retain raw `Hsp` lists (`ev.hsps == []` by default) and processes BED files in bounded memory.
