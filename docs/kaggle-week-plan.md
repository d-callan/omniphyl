# OmniPhyl Kaggle Week — Experiment Plan

*Goal: convert `docs/arch-proposal-500bp.md` into a sequence of attributable experiments, maximizing learning per GPU-hour. Local machine is CPU-only (20 cores, ~15 GB RAM, torch 2.14+cpu); Kaggle T4/P100 quota ~30 h/week. The single biggest waste in weeks 1–2 was paying the RefSeq fetch cost and full-scale training cost on every experiment — this week's plan eliminates both.*

---

## 0. The core strategy change: freeze the corpus, iterate the model

Every prior kernel run did fetch → train → eval end-to-end. Fetch dominated wall-clock and added corpus variance between arms, making comparisons ambiguous. This week we split the pipeline:

1. **Corpus is built once, frozen, and versioned** (Kaggle Dataset or `/kaggle/working` output pulled back).
2. **All architecture ablations train on the identical frozen corpus** — same fragments, same labels, same split seed. Any metric difference is attributable to the arm, not the data.
3. **Ablations run on a mini corpus (~20–30k frags)** where an honest val-genome split still has enough genomes to give signal, and a full train+eval cycle is ~20–40 min on T4 instead of 6–10 h.
4. **Only the winning arm gets a v4-scale confirmation run.**

### Frozen corpus assets (already in hand, zero new fetch needed)

| Asset | Location | Reuse |
|---|---|---|
| v4 corpus | `/tmp/v4_out/omniphyl-work/corpus/fragments.fna` (254 MB, ~460k frags) + `labels.tsv`, `best.pt` inside `omniphyl-pilot-20261012-093651.tar.gz` | v4-scale confirmation arm + local CPU eval |
| fungi-probe | `/tmp/fungi_res/omniphyl-work/run/best.pt` + `corpus/labels.tsv` | Step-0 offline eval only (fragments.fna wasn't pulled back — pull it with the next kernel output, or regenerate) |
| pilot_v2 | `/tmp/pilot_v2/omniphyl-work/corpus/{fragments.fna,labels.tsv}` + `run/best.pt` | small fast smoke corpus for CPU pipeline tests (44 MB fragments) |

Extract v4 to a stable local path (`data/v4/` or `/tmp/omni_v4/`) so nothing in `/tmp` evaporates mid-week.

### New corpus to fetch once on Kaggle: `5group-mini`

Same five groups and family-level sampling as v4, but `NT_BUDGET≈15000` per assembly (~30 frags/genome) and `PER_RANK_N=1` — target ~500–1000 genomes, ~20–30k fragments, ~10% val-genome holdout + the existing `holdout_genus_frac` clade holdout. Save `fragments.fna` + `labels.tsv` as a **Kaggle Dataset output** from the fetch kernel; every ablation kernel then mounts that dataset and skips fetch entirely. If Kaggle Dataset creation from kernel output is friction, the fallback is pulling `fragments.fna` to local and re-uploading once.

**GenBank gap-fill (implemented 2026-10-17, enabled via env).** The Step-0 analysis showed protozoa/archaea/fungi fail differently than bacteria — not enough training coverage (protozoa contributed ~650 fragments; its clades get systematically wrong attractors or none at all). `fetch_refseq_corpus.py` now supports `--genbank-groups` (e.g. `archaea protozoa fungi`), which loads the identically-formatted `genomes/genbank/{group}/assembly_summary.txt`, keeps only assemblies with no RefSeq twin (`gbrs_paired_asm == na`), and fills rank values the RefSeq pass missed (`--genbank-extra N` relaxes to a bounded supplement on covered values). GenBank picks emit `seq_source=genbank` in `labels.tsv` and `build_split` forces them train-only — val stays purely RefSeq so the benchmark is unchanged and the arm measures exactly "does GenBank depth help weak groups". Annotation labels just mask out on unannotated assemblies (existing `require_gff=False` path; `label_source=none`). Caveats: submitter-declared taxonomy is noisier and draft euk assemblies are more contaminated — `screen_contaminants.py` is now wired into `run_pilot.sh` for **all** fragments (`SCREEN_CONTAM=1` default; ambiguity gate + phiX/UniVec/human sketches; `HUMAN_FASTA` env enables the human sketch; accessions that are themselves contaminant species, e.g. phiX in the viral set, are exempted via `--contam-acc-frac`). Known blind spot: contamination *inside* deposits (e.g. bacterial contigs in a protozoa assembly) is invisible to the screen — that needs a taxonomic-concordance check, and may partly explain the v4 protozoa consensus failures. Groups to apply: protozoa + archaea for sure; fungi likely (GenBank is much richer than RefSeq there); bacteria/viral no (depth adequate, mislabel surface largest).

---

## 1. Work partition: this machine vs Kaggle

### Local (CPU) — everything except gradient updates at scale

- **All implementation + unit tests** for Steps 1–3 of the proposal ladder (tree head, RC/multiscale FCGR, hash/AA branches). CPU is fine for correctness; GPU only decides whether the idea *learns*.
- **Step 0 offline eval** on the recovered v4 checkpoint: calibration, risk-coverage, M-fragment consensus pooling, per-class recall tables. Inference-only on a subsample (~20–46k frags) is tractable on 20 cores.
- **Micro smoke runs** (~5k frag subset of `pilot_v2` corpus, 1–2 epochs, width 32): verifies each new code path end-to-end (loss decreases, shapes, checkpoint round-trips) before it costs GPU time. Estimated ~10–30 min/run on CPU.
- **Timing profiling**: measure FCGR encode throughput in the DataLoader to predict Kaggle epoch times before launching (FCGR compute is CPU-side even on Kaggle).
- **Kernel packaging**: edit `kaggle/*.py` env config + `remote/run_pilot.sh` passthroughs; push with the `omniphyl` conda env's `kaggle` CLI.

### Kaggle (GPU) — only the things that need gradient descent at honest scale

- **One fetch kernel** → `5group-mini` dataset (fetch only, `EPOCHS=0` or a `FETCH_ONLY=1` mode in `run_pilot.sh`).
- **Ablation arms** (Section 2), each = clone → mount mini dataset → train → eval → save `best.pt` + `results.json`. ~20–40 min each.
- **One v4-scale confirmation run** on the winner (~6–10 h, run overnight mid-week).

**What never goes to Kaggle without local verification first:** any code path that hasn't passed a CPU smoke run + unit test on this machine. A syntax error or shape bug discovered at minute 240 of a Kaggle run is the failure mode this plan exists to kill.

---

## 2. Arm sequence (each arm vs. A0 control, identical corpus/split/seed)

Order follows the proposal's ablation ladder, re-sorted so the highest-information-per-hour experiments go first.

| Arm | Change | Where built | GPU est | Decision it drives |
|---|---|---|---|---|
| **A0** | Baseline: flat heads, k=7 FCGR, current ResNet | — (control re-run on mini corpus) | ~30 min | Reference numbers on the *new* frozen corpus |
| **A1** | Conditional tree softmax + cosine prototypes + logit-adjusted priors (`--head tree`), flat heads kept as `--head flat` default | local | ~30 min | Does prior collapse (Uroviricota 41.7%) disappear? Val macro-F1 phylum/family should ≥2× |
| **A2** | RC-canonical multi-scale FCGR (k=6 + k4/k5 pyramid channels, stride-1 stem, quadtree 2×2 pooling, log-count normalization) | local | ~40 min | Does representation ceiling lift? Expect +10–15 pts class/order/family |
| **A3** | + canonical 11-mer hash `EmbeddingBag` + 6-frame AA 3-mer branch | local | ~45 min | Family/genus lift specifically; watch whether species moves at all |
| **A4** | Optional diagnostics on existing flags: `--backbone seq` and/or `--stem-stride 1` | already wired | ~30 min each | Raw-sequence floor/ceiling check; stem-stride isolates quadtree-destruction hypothesis |
| **A5** | Winner combination on v4 frozen corpus | Kaggle | ~6–10 h | Scale-up confirmation; compare directly against v4's val metrics |

**Defer to next week:** SupCon paired sampler (Step 4) — needs the prototype head working first anyway; pos-channels/annotation-head interaction — orthogonal to taxonomy; ViT backbone — only if CNN saturates after input fixes.

**Guardrail:** one variable per arm. If A2 underperforms, we learn "multi-scale/RC hurts" — not "the whole proposal is wrong". Arms A1–A3 are cumulative *only after* each individually clears its gate; otherwise they're independent vs A0.

### Arm results — 5group-mini corpus (2,874 genomes, 63,372 screened frags)

**A0 (flat)** — kernel `omniphyl-a0-flat` v2, `~/omniphyl-work/a0_out/`:

| rank | frag acc | macro-F1 | consensus (all frags) |
|---|---|---|---|
| domain | 0.662 | 0.266 | 0.768 |
| phylum | 0.531 | 0.035 | 0.624 |
| class | 0.492 | 0.015 | 0.553 |
| order | 0.135 | 0.010 | 0.172 |
| family–species | n=0 (no in-vocab val labels) | — | — |

Collapse check (predicted-class entropy from `predictions.tsv`): phylum H=1.72 nats, Uroviricota takes 50.8% of all phylum calls — the v4 collapse is reproduced on the frozen corpus.

**A1 (tree head)** — kernel `omniphyl-a1-tree` v5. First read looked catastrophic (val domain 0.19, deep ranks 0.000), but that was an **eval artifact, not a training failure**: the head is trained so `softmax(logits + τ·log ν) = P(y)` with `PRIOR_TAU=1.0`, while the evaluator computed marginals at τ=0 — i.e. argmax over `P/ν` odds-over-prior ratios, which systematically inflates rare taxa (9,820 Bacteria frags called *Singelaviria*; a rare chytrid lineage took 61.6% of genus calls). Re-running eval with `--infer-tau 1.0` (MAP, `f6eceba` — now the default for tree ckpts) on the same `best.pt`:

| rank | frag acc | macro-F1 | consensus |
|---|---|---|---|
| domain | **0.873** | **0.482** | **0.928** |
| phylum | **0.665** | **0.104** | **0.713** |
| class | 0.504 | 0.041 | 0.523 |
| order | 0.118 | 0.016 | 0.101 |
| family | 0.056 | 0.005 | 0.079 |
| genus | 0.034 | 0.003 | 0.049 |
| species | 0.040 | 0.003 | 0.052 |

**Gate: passed.** Phylum macro-F1 3× A0, domain ~2×, class ~3×; phylum entropy 1.72→2.12 with Uroviricota top-share 50.8→41.5 (collapse undone); calibration also improves (domain ECE 0.31→0.07). Family macro-F1 is formally unevaluable for A0 (n=0 on val) but the macro-F1 trend favors the tree head at every comparable rank. Deep-rank *accuracy* stays low on held-out genera — expected: those classes are unseen by construction. A1 ships to scale; tree head becomes the default for cumulative arms pending A2's backbone verdict.

---

## 3. Step 0 — free analysis before any GPU spend (local, day 1)

Extend `scripts/eval_pilot.py` to emit, per rank: per-class recall table, reliability diagram / ECE, risk–coverage curve (AURC), and **M-fragment genome consensus accuracy** (log-odds pooling over fragments per accession, M ∈ {1, 5, 20, all}). Run it on the extracted v4 `best.pt` + `labels.tsv` + `fragments.fna`, subsampling val genomes if CPU inference is too slow.

**Why first:** it answers two questions that change the arm order for free — (a) is the v4 model's softmax confidence rank-informative at all (if not, abstention work jumps priority), and (b) does genome-level pooling already rescue family/genus accuracy (if M=20 consensus hits 80%+, the product story shifts from per-fragment to per-contig aggregation and the architecture stakes change).

### Step 0 results — v4 checkpoint, val-genome split (45k frags, ~352 genomes)

Artifacts: `~/omniphyl-work/v4/run/eval3/results.json` (+ `per_class_*.tsv`, `calibration.tsv`, `risk_coverage.tsv`, `predictions.tsv`).

| rank | acc | macro-F1 | ECE | AURC |
|---|---|---|---|---|
| domain | 0.729 | 0.309 | 0.042 | 0.117 |
| phylum | 0.575 | 0.102 | 0.049 | 0.198 |
| class | 0.540 | 0.055 | 0.039 | 0.215 |
| order | 0.204 | 0.040 | 0.078 | 0.651 |
| family | 0.106 | 0.018 | 0.082 | 0.805 |
| genus | 0.143 | 0.013 | 0.022 | 0.741 |
| species | 0.208 (n=2448) | 0.002 | 0.057 | 0.662 |

**Confidence is rank-informative.** At τ=0.5, ~90% of frags get a call somewhere on the tree and the depth distribution is sensible; AURC rises steeply at order+ (0.65→0.80), meaning the risk-coverage trade is usable — abstention below order is warranted, at domain/phylum it is cheap. This keeps A1's calibrated-confidence story relevant.

**Genome consensus rescues deep ranks.** Corrected product-of-experts pooling (sparse top-512 logp per frag, per-frag floor subtracted):

| rank | M=1 | M=5 | M=20 | all |
|---|---|---|---|---|
| domain | 0.699 | 0.852 | 0.849 | 0.861 |
| phylum | 0.575 | 0.673 | 0.711 | 0.711 |
| class | 0.531 | 0.635 | 0.656 | 0.671 |
| order | 0.224 | 0.318 | 0.365 | 0.370 |
| family | 0.126 | 0.252 | 0.318 | 0.351 |
| genus | 0.136 | 0.435 | 0.461 | **0.526** |
| species | 0.158 | 0.789 | 0.842 | 0.789 (n=19 genomes only) |

Per-fragment errors at family/genus are near-independent across frags of the same genome, so ~20 frags roughly triples accuracy there — the signal is in the genome, not the fragment. This partially decouples the product story from per-fragment top-1: a metagenome-assembled contig with ≥20 500-bp windows already lands ~52% genus / ~84% species-where-species-is-in-vocab. It does **not** remove the case for A1 — per-fragment priors/collapse still depress macro-F1 (phylum F1=0.10) and conditional calibration would lift every pooling step — but it lowers the stakes of per-fragment species accuracy and raises the value of a *sequence-level* product surface. For single-read use cases M=1 stays the honest (weak) deep-rank number.

**Not a fragment-subsampling artifact.** Held-out genomes carry ~90–130 fragments in *every* group (fetcher's NT budget gives fixed fragment count, not fixed coverage), so M=20 is a ~15% subset everywhere — but ~1% of a bacterial genome vs ~20% of a small viral genome. The lift is largest and cleanest in **bacteria** (genus 0.09→0.58, domain 0.82→0.99), i.e. in the sparse-coverage regime where 20 windows are scattered across Mb-scale genomes. A coverage artifact would predict the opposite. Where errors are *systematic*, pooling fails as expected: protozoa consensus is flat at every M (phylum 0.0, n=6) and archaea's pooled phylum dips below its M1 (0.18 vs 0.27, n=11) — correlated wrong votes get amplified, not averaged out.

**Group gaps:** archaea domain acc only 0.259 (n=1374, likely too few assemblies); bacteria 0.838, viral 0.687. Leave-phylum-out confirms expected degradation vs in-vocab at every rank.

*Note:* an earlier sparse-pooling version summed raw logps and biased toward classes absent from fragments' top-512 (floor≈0 vs accumulated mass); corrected 2026-10-17 to subtract each frag's top-K floor before pooling.

---

## 4. Week schedule

| Day | Local (CPU) | Kaggle (GPU) |
|---|---|---|
| Mon | Extract v4 artifacts to stable path; Step 0 eval extensions + run on v4 ckpt | — (any in-flight kernel finishes; pull outputs incl. fragments.fna) |
| Tue | Implement A1 tree head + unit tests (tree coherence, sibling normalization, RC index map); CPU smoke on pilot_v2 subset | Fetch-only kernel → `5group-mini` dataset (~1–2 h) |
| Wed | Implement A2 RC/multiscale encoder + unit tests; profile FCGR throughput | A0 + A1 back-to-back (~1 h total); pull results |
| Thu | Implement A3 hash/AA branches; write eval extension for per-sibling entropy report | A2 (+ A4 diagnostics if A2 is promising) |
| Fri | Review arm results vs gates; pick winner combo | A3; queue A5 v4-scale overnight if gates cleared |
| Weekend | Analysis write-up into `docs/`; next-week plan | A5 completes; pull checkpoint |

Gate logic: A1 ships to scale only if val family macro-F1 ≥2× A0 **and** predicted-class entropy at phylum rises (collapse undone). A2 needs ≥ +8 pts val order/family accuracy vs its control. A3 judged on family/genus lift + leave-genus-out parent accuracy, not species top-1 (physically bounded at 500 bp).

---

## 5. Known blockers / checks before Monday

- **In-flight kernel**: confirm which step is currently running and that it finishes cleanly before the fetch kernel starts (quota overlap is fine, output naming isn't).
- **`run_pilot.sh` needs a `FETCH_ONLY` path** and a `SKIP_FETCH` path (mount dataset → train directly). ~20 lines; local edit.
- **fungi fragments.fna** wasn't pulled back — next kernel output should include it for completeness.
- **RAM**: 15 GB total, ~3 GB free at check — keep `num_workers` ≤ 8 on local smokes and don't memory-map the full 254 MB fragments.idx casually.
- The `omniphyl` conda env has the `kaggle` CLI + `~/.kaggle/access_token` — use that env for `kaggle kernels push/output`.
