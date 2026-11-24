# Golden-output regression snapshots

These snapshots back the G1 regression gate (`tests/golden_diff.py`): each
manifest case runs the `hyphy3` CLI, its emitted v3 JSON is canonicalized
(sorted keys, floats rounded to 12 significant digits, volatile fields such as
timers/git-commit/abs-paths masked), and compared byte-for-byte against the
committed snapshot. Where the manifest declares a `schema`, the raw output is
also validated against `schemas/v3/*.schema.json` (the G2 gate).

## Data provenance

All inputs come from `benchmarks/data/` (CD2 alignment and
`{Group1-4}`-tagged tree — the standard HyPhy CD2 benchmark set). Exact
arguments and per-case source notes are in `manifest.json`. No data is
transformed: files are passed to the CLI as-is.

## Usage

```bash
# record snapshots against a built binary (run on the pre-refactor baseline)
python3 tests/golden_diff.py --record --bin build/hyphy3 --tier all

# check (fast tier only — this is what CI runs)
python3 tests/golden_diff.py --bin build/hyphy3

# single case
python3 tests/golden_diff.py --bin build/hyphy3 --cases fel_cd2
```

Via ctest: `GoldenOutputTest` (fast tier), `GoldenOutputDeepTest` (slow tier,
label `deep`), `CLISurfaceTest` (driver dispatch/alias surface).

## Adding a case

1. Add an entry to `manifest.json` (name, subcommand, args, schema, tier,
   source). Keep `--threads` fixed so results don't depend on the machine.
   Disable stochastic features for determinism (e.g. SLAC `--samples 0`).
2. `python3 tests/golden_diff.py --record --cases <name> --bin build/hyphy3`
3. Commit the new snapshot.

## Notes

- Snapshots are recorded on the baseline commit and must not change across
  pure refactorings; a legitimately changed analysis requires a recorded
  re-baseline and review of the diff.
- `tree`, `gard`, `fubar`, `prime`, `mss`, `ohmic`, `expcm`, `dfel`, `report`,
  `busted-ph` have no cases yet — either stochastic/search-heavy or untested
  for determinism; extend the manifest as needed.
- Snapshots are machine-independent only to float-rounding precision
  (12 sig figs). Record/check on the same toolchain family for refactors.
