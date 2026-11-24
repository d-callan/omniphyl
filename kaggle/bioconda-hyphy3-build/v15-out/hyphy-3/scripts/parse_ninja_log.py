#!/usr/bin/env python3
"""Parse a Ninja .ninja_log into per-TU compile-time stats.

Usage:
    parse_ninja_log.py <build_dir> [--top N] [--json] [--group-by app|target|all]

Ninja log format (v5+): start_ms end_ms restat_mtime_ms output hash
Per-target duration = end_ms - start_ms for each completed edge.

Reports cumulative edge time (CPU-seconds equivalent), wall duration of the
build (first start to last end), and the N longest-compiling outputs — the
metrics used by the compile-time refactor scorecard (specs/
compile-time-refactor.md). Source provenance: .ninja_log in the build dir
produced by `ninja` during a build; no transformations beyond duration math
and basename grouping.
"""

import argparse
import collections
import json
import sys
from pathlib import Path


def read_log(log_path):
    """Yield (start_ms, end_ms, output) for each log entry."""
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            entries.append((start, end, parts[3]))
    return entries


def group_key(output, mode):
    if mode == "all":
        return "all"
    if mode == "app":
        # Group object files by which binary/app they feed, inferred from
        # CMakeFiles/<target>.dir/ path segments.
        parts = Path(output).parts
        for p in parts:
            if p.endswith(".dir"):
                return p[:-4]
        return "(other)"
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("build_dir", help="build directory containing .ninja_log")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--group-by", choices=["target", "app", "all"],
                    default="app", help="aggregate durations by CMake target dir, all, or per-output")
    args = ap.parse_args()

    log_path = Path(args.build_dir) / ".ninja_log"
    if not log_path.exists():
        print(f"error: no .ninja_log in {args.build_dir}", file=sys.stderr)
        return 2

    entries = read_log(log_path)
    if not entries:
        print("error: .ninja_log is empty", file=sys.stderr)
        return 2

    total_cpu_ms = sum(e - s for s, e, _ in entries)
    wall_ms = max(e for _, e, _ in entries) - min(s for s, _, _ in entries)

    per_group = collections.Counter()
    per_output = collections.Counter()
    for s, e, out in entries:
        dur = e - s
        per_group[group_key(out, args.group_by)] += dur
        per_output[out] = per_output.get(out, 0) + dur

    heaviest = per_output.most_common(args.top)
    groups = per_group.most_common()

    if args.json:
        print(json.dumps({
            "log": str(log_path),
            "edges": len(entries),
            "cumulative_edge_time_s": total_cpu_ms / 1000,
            "wall_time_s": wall_ms / 1000,
            "parallelism": round(total_cpu_ms / wall_ms, 2) if wall_ms else None,
            "groups": [{"name": g, "time_s": t / 1000} for g, t in groups],
            "heaviest_outputs": [{"output": o, "time_s": t / 1000} for o, t in heaviest],
        }, indent=1))
        return 0

    print(f"log: {log_path}")
    print(f"edges:                  {len(entries)}")
    print(f"cumulative edge time:   {total_cpu_ms / 1000:.1f} s")
    print(f"wall time (log span):   {wall_ms / 1000:.1f} s")
    if wall_ms:
        print(f"effective parallelism:  {total_cpu_ms / wall_ms:.2f}x")
    print(f"\ntop {args.top} heaviest outputs:")
    for out, t in heaviest:
        print(f"  {t / 1000:8.1f} s  {out}")
    if args.group_by != "target":
        print(f"\ntime by {args.group_by}:")
        for g, t in groups[:args.top]:
            print(f"  {t / 1000:8.1f} s  {g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
