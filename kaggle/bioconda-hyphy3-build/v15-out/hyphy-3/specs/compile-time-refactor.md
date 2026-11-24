# Compile-time refactor spec (working doc — do not commit)

Goal: reduce HyPhy3 compile time / peak memory enough that bioconda builds stop
timing out and OOMing, without changing any analysis behavior. Measured
motivation and structure below; every step is gated on the regression harness in
section 3.

Related, already landed on `fix/standalone-alias-symlinks`:
`HYPHY3_STANDALONE_SYMLINKS=ON` skips ~20 duplicate app-TU compiles and installs
`hyphy3-*` as symlinks to the driver. This spec covers the heavier work that
option alone cannot fix.

## 1. Measured structure of the problem

`hyphy3_core` and `hyphy3_report` are INTERFACE (header-only) libraries — no
library TU exists, so every consumer re-parses whatever it includes.

| item | size / count | fan-in | note |
|---|---|---|---|
| `include/nlohmann/json.hpp` | 920 KB | all emitters + apps | in PCH for exe targets; **emitters excluded** (CMakeLists ~line 582) |
| `src/python/bindings.cpp` | 962 lines, 31 includes | 1 TU, ×4 in bioconda (per py variant) | pulls in *every* analysis header in one TU — prime OOM suspect |
| `include/hyphy/analyses/busted.hpp` | 6,623 lines, ~980 defs | ~5 TUs (app, emitter, tests, bindings, driver) | worst impl-in-header |
| `include/hyphy/analyses/characterize.hpp` | 6,418 lines, ~670 defs | ~3 TUs (app, wasm, driver) | |
| `include/hyphy/analyses/joint_align.hpp` | 5,805 lines, ~737 defs | ~2 TUs | |
| `include/hyphy/analyses/absrel.hpp` | 110 KB | | medium |
| `include/hyphy/analyses/tree_engine.hpp` | 105 KB | | medium |
| `include/hyphy/analyses/relax.hpp` | 105 KB | | medium |
| `include/hyphy/core/likelihood.hpp` | 101 KB | ~33 includers | mostly already inside PCH path |
| `include/hyphy/core/tree.hpp`, `alignment.hpp`, `json_output.hpp` | 39–44 KB | ~30 includers | in PCH |
| ~25 `tests/test_*.cpp` executables | | | each a heavy TU with its own main |

Before starting: capture a baseline using the protocol in §2. Build once with
Ninja in CI (or Kaggle) and record the scorecard metrics in the table below.
Update the table above if the measured ranking differs from this initial profile.

## 2. Build metrics & inter-phase benchmarking protocol

To know between phases if and how much each refactor helps without relying on
flaky local cmake, we track three primary metrics on clean builds:

1. **Peak Memory (Max RSS)**: Captured via `/usr/bin/time -v cmake --build build`.
   This is the direct proxy for the Bioconda OOM failure mode.
2. **Top bottleneck TUs and Cumulative CPU time**: Captured via `build/.ninja_log`.
   Ninja logs `<start_ms>\t<end_ms>\t<mtime>\t<output_file>\t<hash>`. A 20-line
   python snippet (`scripts/parse_ninja_log.py`) sorts TUs by duration and prints:
   - Top 10 slowest compilation units (identifies the long pole).
   - Cumulative TU duration (sum of all TU times, unaffected by core count).
   - Target-group totals (`hyphy3_emitters`, `src/python/bindings`, `tests/*`).
3. **End-to-end wall clock time**: Clean build time in GitHub Actions CI
   (`ubuntu-latest`, 4 vCPUs) and optional Kaggle run.

### CI integration
In `.github/workflows/test.yml`, wrap the build step with `/usr/bin/time -v` and
run the log parser immediately after:
```yaml
      - name: Build with profiling
        run: |
          ccache --zero-stats
          /usr/bin/time -v cmake --build build -j"$(nproc)" 2>&1 | tee build_stats.log
          ccache --show-stats
          python3 scripts/parse_ninja_log.py build/.ninja_log
```
Every commit on the refactor branch will print the exact TU bottlenecks and peak
RSS in the Actions log.

### Phase scorecard table

Record these numbers after each phase before proceeding to the next:

| Metric | Baseline (`1e491ce`, Kaggle `-j2`) | Post-R1 (Bindings) | Post-R2 (Emitters PCH) | Post-R3 (Impl out of hdr) | Post-R4 (Test merge) | Target win |
|---|---|---|---|---|---|---|
| Peak RSS (Max RAM, GB) | 4.1 | | | | | < 2.5 GB (safe for 4GB bioconda workers) |
| Total build wall time (CI) | 5436.6 s tests-ON (3822.9 s tests-OFF) | | | | | < 10 min |
| Cumulative TU time (CPU-s) | 18214.9 tests-ON (7499.2 tests-OFF, 98 edges) | | | | | > 30% reduction |
| Heaviest single TU (s & name)| 476.3 `src/apps/hyphy_characterize.cpp` (`hyphy3` target) | | | | | < 35s |
| Python bindings total TU time | 351.7 (`_hyphy3`) | | | | | > 50% reduction in peak TU mem |
| `hyphy3_emitters` compile time | 1568.3 | | | | | > 30% reduction |
| Test binaries compile time | ~3374.4 (29 `hyphy_*_test` targets) | | | | | > 40% reduction |

### Inter-phase Go / No-Go decision rules
- **Post-R1 (Bindings split)**: If heaviest single TU does not drop below 40s or
  peak RSS doesn't decrease, check if template instantiations are being duplicated
  across split files.
- **Post-R2 (Emitters PCH)**: If `hyphy3_emitters` cumulative time does not drop
  by at least 25%, verify whether the compiler is actually reusing the PCH or
  silently recompiling (check compiler diagnostics / `-Winvalid-pch`).
- **Post-R3 (Impl out of header)**: If golden diff (G1) is not 100% clean, DO NOT
  PROCEED. If benchmark runtime (G5) slows down >5%, mark critical helper functions
  `inline` in the header until parity is restored.
- **Post-R4 (Test merge)**: Verify all individual ctest cases still report
  distinct pass/fail status.

## 3. Refactors, ranked by value/cost

### R1 — Split `src/python/bindings.cpp` per analysis (highest value/cost)

- New TUs `src/python/bindings_<analysis>.cpp`, one per bound analysis
  (fel, meme, busted, absrel, relax, fubar, prime, gard, slac, dfel,
  contrast-fel, mss, ohmic, characterize, joint-align, expcm…).
- Each defines `void register_<analysis>(nanobind::module_&)` and includes only
  its own analysis header.
- `bindings.cpp` keeps `NB_MODULE` and calls the registrars; registrars declared
  in a small `bindings_registry.hpp`.
- CMake: replace the one TU with the list; module target links same libs.
- Effect: one ~60 s monolith TU → many small parallelizable TUs; peak memory
  per TU drops because no TU parses all analysis headers at once. In bioconda
  the module is built once per Python variant (4×), so the win multiplies.
- Risk: low. Only binding/registration code moves; no analysis code touched.

### R2 — PCH for `hyphy3_emitters`

- CMakeLists ~line 582 excludes `hyphy3_emitters` from PCH reuse. Find out why
  (probably a compile-flag mismatch when the exclusion was introduced).
- If nothing blocks it, drop the exclusion or give emitters their own PCH via
  `target_precompile_headers(hyphy3_emitters REUSE_FROM hyphy3_pch)`.
- Saves ~16 re-parses of json.hpp + Eigen + core headers.
- Risk: low; verify `cmake --build` and one emitter's output unchanged.

### R3 — Move implementation bodies out of the three monster headers

Order: `joint_align.hpp` (smallest consumer set) → `characterize.hpp` →
`busted.hpp` (largest). Same pattern each:

- Create `src/analyses/<name>_impl.cpp`; move all non-template, non-`inline`
  function bodies out of the header; leave declarations + templates + hot
  inner-loop functions (mark those `inline` and document why).
- Add the new TU to `hyphy3_emitters` (or a new `hyphy3_analyses` static lib —
  decide when the first one lands; a separate lib keeps emitter/impl concerns
  clean).
- Effect: each header shrinks to declarations; its ~5 consumer TUs stop
  re-parsing megabytes of definitions.
- Risk: medium. Inline-surface shrinkage could slow hot paths → the perf
  spot-check in §4 (G5) is mandatory, not optional. ODR mistakes show up as link
  errors — cheap to catch.

### R4 — Merge test executables

- ~25 `tests/test_*.cpp` each with `main()` and heavy includes. Group into a
  few binaries (`test_core_bin`, `test_analyses_bin`, …) or one doctest/own
  runner TU. Keep test names distinct in ctest via `add_test` per group or a
  test-case filter.
- Effect: cuts the `fast` CI build directly; biggest single win for CI
  wall-clock after R3.
- Risk: medium-low. Watch for `static`/anonymous-namespace collisions when
  merging TUs — the reason this is grouped, not one mega-TU.

### R5 — `likelihood.hpp` into the PCH (measure first)

- 101 KB × ~33 includers, but it is included by the analysis headers, so its
  cost rides inside every analysis TU anyway. Only worth doing if the
  `.ninja_log` baseline shows it as the marginal driver after R3.

### Explicitly deferred

- pImpl or redesign of `likelihood.hpp` / core types — too invasive for the
  compile-time win.
- Unity/jumbo builds — ODR and anonymous-namespace hazards, worse parallelism.
- Touching `nlohmann/json.hpp` or Eigen versions.
- Moving `hyphy3_core`/`hyphy3_report` to compiled libraries — that is the
  heavy refactor; only reconsider if R1–R4 are insufficient.

## 4. Regression gates — implement BEFORE refactoring

Existing coverage (verify each is wired in ctest before relying on it):
per-analysis unit tests incl. `test_fel_cd2_parity`, `test_fel_multi_dataset`,
`test_gradient_validation`, `test_busted_mh`; `benchmarks/ground_truth/*.json`;
`schemas/v3/*.schema.json`.

Gaps to close:

### G1 — Golden-output equivalence harness (primary net)

- Script (python, `scripts/` or `tests/`): for every benchmark dataset in
  `benchmarks/data/` run `hyphy3 <analysis> <canonical args>`, canonicalize the
  emitted JSON (recursively sort object keys, round floats to ~1e-12), and diff
  against a checked-in snapshot file.
- Record in the snapshot the exact dataset + args used (source + transforms,
  per project convention).
- Workflow per refactor step: `harness --record` on unmodified code → refactor →
  `harness --check` must be byte-identical. Unit tests check values; this
  checks that *every emitted field* survived the move.

### G2 — Schema validation for all outputs

- Extend tests so every analysis's JSON output on the benchmark datasets is
  validated against its `schemas/v3/<analysis>.schema.json` (deep job already
  has `jsonschema` installed; mirror in the harness or a ctest).

### G3 — CLI surface test

- ctest/script asserting: `hyphy3 <sub>` works for every dispatched name,
  `hyphy3-<sub> --help` exits 0 through a symlink alias, unknown arg/subcommand
  gives the expected error and non-zero exit, no-arg prints usage.
- Protects both the alias dispatch and any future argv handling changes.

### G4 — Build-tree alias smoke (packaging)

- In `HYPHY3_STANDALONE_SYMLINKS` mode the `hyphy3-*` names only exist
  post-install. If in-build testing is wanted, add a `POST_BUILD`
  `create_symlink` for each alias into the build dir so ctest can exercise
  alias dispatch without installing.

### G5 — Perf spot-check (R3 gate)

- Wall-time one FEL and one BUSTED benchmark run before vs. after each
  impl-out-of-header move. Report numbers in the refactor PR. Threshold is
  judgement, not an assertion — flag if >5% slower on either.

## 5. Implementation order

1. §4 G1–G3 harnesses (land first, verify they catch a deliberately-broken
   output field before trusting them).
2. §2 build profiling script (`scripts/parse_ninja_log.py`) and capture baseline.
3. R1 bindings split (measure delta on Python variant compilation).
4. R2 emitters PCH (measure delta on `hyphy3_emitters`).
5. R3: joint_align → characterize → busted (G1+G5 gates each; measure per-TU drop).
6. R4 test-binary merge (measure CI wall time drop).
7. R5 only if baseline shows it still matters.

## 6. Open questions

- Why is `hyphy3_emitters` excluded from the PCH? (R2 blocker if non-obvious.)
- `hyphy3_analyses` static lib vs. folding impl TUs into `hyphy3_emitters` —
  decide at first R3 commit.
- Upstream appetite for merging test binaries (touches ctest layout).
- Bioconda: at next tag add `-DHYPHY3_STANDALONE_SYMLINKS=ON` to `build.sh`;
  harmless earlier (unused-variable warning only).
