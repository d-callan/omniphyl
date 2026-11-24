#!/usr/bin/env python3
"""CLI surface regression test for the hyphy3 combined driver.

Checks:
  - driver usage / help / version exit codes and output
  - unknown subcommand exits non-zero with a clear error
  - every documented subcommand's --help exits 0
  - argv[0] alias dispatch: a hyphy3-* or hyphy-* symlink to the binary
    selects the subcommand without argv[1]
  - unknown arguments exit non-zero

Usage: test_cli_surface.py [path-to-hyphy3-binary]
Falls back to HYPHY3_BIN env or ./build/hyphy3.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

BIN = os.path.abspath(
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "HYPHY3_BIN",
        str(Path(__file__).resolve().parent.parent / "build" / "hyphy3")))

# Subcommands routed by dispatch_subcommand() in src/apps/hyphy3.cpp.
SUBCOMMANDS = [
    "fel", "contrast-fel", "meme", "busted", "absrel", "relax", "fubar",
    "prime", "gard", "slac", "dfel", "mss", "ohmic", "tree",
    "joint-align", "expcm", "report", "characterize",
]

failures = []


def check(name, argv, expect_rc, expect_substr=None):
    """Run argv[0]+argv[1:], verify exit code and optional output substring."""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        failures.append(f"{name}: timed out")
        print(f"FAIL {name}: timed out")
        return
    out = (proc.stdout or "") + (proc.stderr or "")
    ok = proc.returncode == expect_rc
    if ok and expect_substr is not None:
        ok = expect_substr.lower() in out.lower()
    if ok:
        print(f"PASS {name}")
    else:
        failures.append(name)
        print(f"FAIL {name}: rc={proc.returncode} (want {expect_rc})"
              + (f", missing {expect_substr!r}" if expect_substr else ""))
        tail = out.strip().splitlines()[-3:]
        for line in tail:
            print(f"    {line}")


def main():
    if not (os.path.isfile(BIN) and os.access(BIN, os.X_OK)):
        print(f"error: hyphy3 binary not found/executable: {BIN}", file=sys.stderr)
        return 2

    check("no-args-usage", [BIN], 0, "usage")
    check("help", [BIN, "--help"], 0, "usage")
    check("version", [BIN, "--version"], 0, "HyPhy 3")
    check("unknown-subcommand", [BIN, "bogus-analysis"], 1, "unknown analysis")

    for sub in SUBCOMMANDS:
        check(f"help:{sub}", [BIN, sub, "--help"], 0)

    check("unknown-arg", [BIN, "fel", "--bogus-flag-xyz"], 1)

    # argv[0] alias dispatch: symlink the driver under a standalone-style name.
    try:
        with tempfile.TemporaryDirectory() as td:
            for alias, sub in [("hyphy3-fel", "fel"), ("hyphy-meme", "meme")]:
                link = os.path.join(td, alias)
                os.symlink(BIN, link)
                check(f"alias:{alias}", [link, "--help"], 0)
            bogus = os.path.join(td, "hyphy3-bogus")
            os.symlink(BIN, bogus)
            check("alias:hyphy3-bogus", [bogus], 1, "unknown analysis")
    except (OSError, NotImplementedError) as e:
        print(f"SKIP alias dispatch (symlink unavailable: {e})")

    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
