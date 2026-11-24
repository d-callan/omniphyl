#!/usr/bin/env python3
"""Golden-output regression harness for the hyphy3 CLI.

Runs each case from tests/golden/manifest.json through
``hyphy3 <subcommand> <args> --output <tmp>``, canonicalizes the emitted
modern (v3) JSON, and either records it as a snapshot (--record) or compares
it against the committed snapshot (--check).

Canonicalization:
  - object keys sorted recursively;
  - floats normalized to FLOAT_PRECISION significant digits (see
    --float-precision) so insignificant summation-order jitter across thread
    counts / BLAS vectorization does not flake the gate;
  - values under keys matching VOLATILE_KEYS (timers, timestamps, git commit,
    build info) replaced with "<masked>" — these legitimately differ between
    the commit snapshots are recorded on and later refactor commits;
  - strings that are absolute filesystem paths reduced to their basename;
  - occurrences of the repository root inside strings replaced with <REPO>.

When the 'jsonschema' package is importable and a case declares a schema,
the raw (non-canonicalized) output is also validated against
schemas/v3/<schema> — this doubles as the G2 schema gate.

Exit code is 0 when every selected case passes, 1 otherwise.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "golden" / "manifest.json"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "tests" / "golden"
SCHEMA_DIR = REPO_ROOT / "schemas" / "v3"

DEFAULT_FLOAT_PRECISION = 12

# Keys whose values are expected to change between record time and refactor
# time (wall-clock timers, timestamps, git commit strings, absolute paths
# recorded under explicit path keys). Matched case-insensitively as a
# substring of the key name.
VOLATILE_KEYS = re.compile(
    r"time|timer|duration|elapsed|timestamp|commit|build|^path$|_path$|cli_command",
    re.IGNORECASE,
)

ABS_PATH = re.compile(r"^(?:[A-Za-z]:[/\\]|/)")

try:
    import jsonschema
except ImportError:
    jsonschema = None


def canonicalize(node, key="", precision=DEFAULT_FLOAT_PRECISION):
    """Recursively canonicalize parsed JSON for stable comparison."""
    if isinstance(node, dict):
        return {
            k: canonicalize(v, k, precision) for k, v in sorted(node.items())
        }
    if isinstance(node, list):
        return [canonicalize(v, key, precision) for v in node]
    if key and VOLATILE_KEYS.search(key):
        return "<masked>"
    if isinstance(node, float):
        return float(f"{node:.{precision}g}")
    if isinstance(node, str):
        node = node.replace(str(REPO_ROOT), "<REPO>")
        if ABS_PATH.match(node) and len(node) > 1:
            node = "<PATH>/" + os.path.basename(node)
        return node
    return node


def find_diffs(expected, actual, path="$", diffs=None, limit=50):
    """Collect human-readable mismatch descriptions between two canonical trees."""
    if diffs is None:
        diffs = []
    if len(diffs) >= limit:
        return diffs
    if type(expected) is not type(actual):
        # Allow int-vs-float equivalence (1 vs 1.0)
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if expected == actual:
                return diffs
        diffs.append(f"{path}: type {type(expected).__name__} != {type(actual).__name__}")
        return diffs
    if isinstance(expected, dict):
        for k in expected.keys() | actual.keys():
            if k not in expected:
                diffs.append(f"{path}.{k}: missing in snapshot, present in output")
            elif k not in actual:
                diffs.append(f"{path}.{k}: present in snapshot, missing in output")
            else:
                find_diffs(expected[k], actual[k], f"{path}.{k}", diffs, limit)
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            diffs.append(f"{path}: list length {len(expected)} != {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual)):
            find_diffs(e, a, f"{path}[{i}]", diffs, limit)
    elif expected != actual:
        diffs.append(f"{path}: {expected!r} != {actual!r}")
    return diffs


def load_manifest(path):
    with open(path) as f:
        data = json.load(f)
    cases = data.get("cases", [])
    names = [c.get("name") for c in cases]
    if len(names) != len(set(names)):
        raise SystemExit(f"manifest {path}: duplicate case names")
    return cases


def run_case(case, binary, workdir):
    """Run one manifest case; returns (raw_output_dict, completed_process, out_path)."""
    out_path = Path(workdir) / f"{case['name']}.json"
    argv = [binary, case["subcommand"]]
    argv += [str(REPO_ROOT / a) if ABS_PATH.match(a) is None and (REPO_ROOT / a).exists() else a
             for a in case["args"]]
    argv += ["--output", str(out_path)]
    proc = subprocess.run(argv, capture_output=True, text=True)
    raw = None
    if out_path.exists():
        try:
            raw = json.loads(out_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  output file is not valid JSON: {e}")
    return raw, proc, out_path


def schema_check(case, raw):
    """Validate raw output against the case's v3 schema. Returns None=skip/pass, str=failure."""
    schema_name = case.get("schema")
    if not schema_name:
        return None
    if jsonschema is None:
        print("  schema: SKIP (jsonschema not installed)")
        return None
    schema_path = SCHEMA_DIR / schema_name
    if not schema_path.exists():
        return f"schema file not found: {schema_path}"
    with open(schema_path) as f:
        schema = json.load(f)
    try:
        jsonschema.validate(instance=raw, schema=schema)
        print(f"  schema: OK ({schema_name})")
        return None
    except jsonschema.ValidationError as e:
        return f"schema violation: {e.message} (at {list(e.absolute_path)})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true",
                      help="run cases and (re)write snapshots")
    mode.add_argument("--check", action="store_true",
                      help="run cases and diff against snapshots (default)")
    ap.add_argument("--bin", default=os.environ.get("HYPHY3_BIN", str(REPO_ROOT / "build" / "hyphy3")),
                    help="path to the hyphy3 binary (or HYPHY3_BIN env)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOT_DIR))
    ap.add_argument("--cases", default=None,
                    help="comma-separated case names to run (default: all in tier)")
    ap.add_argument("--tier", choices=["fast", "slow", "all"], default="fast",
                    help="which manifest tiers to run (default: fast)")
    ap.add_argument("--workdir", default=None,
                    help="directory for case outputs (default: temp dir)")
    ap.add_argument("--float-precision", type=int, default=DEFAULT_FLOAT_PRECISION,
                    help="significant digits floats are rounded to before comparison")
    ap.add_argument("--no-schema", action="store_true",
                    help="skip v3 schema validation")
    args = ap.parse_args()

    binary = args.bin
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        print(f"error: hyphy3 binary not found/executable: {binary}", file=sys.stderr)
        print("build first or pass --bin / set HYPHY3_BIN", file=sys.stderr)
        return 2

    cases = load_manifest(args.manifest)
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["name"] in wanted]
    elif args.tier != "all":
        cases = [c for c in cases if c.get("tier", "fast") == args.tier]
    if not cases:
        print("error: no cases selected", file=sys.stderr)
        return 2

    snapshot_dir = Path(args.snapshot_dir)
    workdir = args.workdir or tempfile.mkdtemp(prefix="hyphy3_golden_")
    os.makedirs(workdir, exist_ok=True)
    if args.record:
        os.makedirs(snapshot_dir, exist_ok=True)

    failures = 0
    for case in cases:
        name = case["name"]
        snap_path = snapshot_dir / f"{name}.json"
        print(f"[{name}] hyphy3 {case['subcommand']} {' '.join(case['args'])}")

        raw, proc, out_path = run_case(case, binary, workdir)
        if proc.returncode != 0:
            print(f"  FAIL: exit code {proc.returncode}")
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            for line in tail[-5:]:
                print(f"    {line}")
            failures += 1
            continue
        if raw is None:
            print(f"  FAIL: no JSON output written to {out_path}")
            failures += 1
            continue

        if not args.no_schema:
            schema_failure = schema_check(case, raw)
            if schema_failure:
                print(f"  FAIL: {schema_failure}")
                failures += 1
                continue

        canonical = canonicalize(raw, precision=args.float_precision)

        if args.record:
            snap_path.write_text(json.dumps(canonical, indent=1, sort_keys=True) + "\n")
            print(f"  recorded -> {snap_path}")
            continue

        if not snap_path.exists():
            print(f"  FAIL: no snapshot {snap_path} — record it with --record")
            failures += 1
            continue

        expected = json.loads(snap_path.read_text())
        diffs = find_diffs(expected, canonical)
        if diffs:
            print(f"  FAIL: {len(diffs)} difference(s) vs snapshot")
            for d in diffs[:20]:
                print(f"    {d}")
            failures += 1
        else:
            print("  PASS")

    verb = "recorded" if args.record else "checked"
    print(f"\n{len(cases) - failures}/{len(cases)} case(s) {verb} cleanly")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
