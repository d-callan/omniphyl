"""Kaggle kernel: hyphy3 post-refactor (R1-R4) build + gate verification.

Captures the post-refactor scorecard for specs/compile-time-refactor.md:
  - wall time + peak RSS of a ninja build mirroring the bioconda recipe
    (-j2, HYPHY3_BUILD_TESTS=OFF)
  - per-TU compile times via scripts/parse_ninja_log.py on .ninja_log
  - golden-output CHECK against committed tests/golden snapshots FIRST
    (this is the real regression gate for the refactor), then --record
    for artifact salvage
  - a ctest pass over all tests incl. merged-bin -sf filtering (R4)

Source: clones d-callan/hyphy-3 branch test/regression-gates (post R1-R4:
R1 bindings split ddd772c, R2 emitters PCH c771dd5, R3 impl-out-of-header
f9e39ab, R4 merged test bins 1b121f0). Eigen 3.4.0 is cloned separately
and handed to FetchContent, exactly like the recipe's second source entry.

Caveat vs the bioconda build: this is a host build on Kaggle's Debian image,
not the cos7 container; timings characterize relative TU costs, not CI-exact
numbers.

Artifacts land in /kaggle/working/out/ (build.log, time logs, ninja JSONs,
golden snapshots to commit back to tests/golden/).
"""

import os
import resource
import shutil
import subprocess
import time

WORKDIR = "/kaggle/working"
OUT = f"{WORKDIR}/out"
MF = f"{WORKDIR}/miniforge"
CONDA = f"{MF}/bin/conda"
ENV = f"{MF}/envs/h3"
ENV_BIN = f"{ENV}/bin"
os.environ["PATH"] = f"{ENV_BIN}:" + os.environ["PATH"]
REPO = f"{WORKDIR}/hyphy-3"
BRANCH = "test/regression-gates"
EIGEN = f"{WORKDIR}/eigen-src"
LOG = f"{OUT}/build.log"
JOBS = "2"  # mirrors the recipe's -j2 RAM cap (~6G peak)


def run(cmd, fatal=True):
    print(f"+ {cmd}", flush=True)
    with open(LOG, "ab") as log:
        p = subprocess.run(
            ["bash", "-c", f"set -o pipefail; {cmd} 2>&1 | tee -a {LOG}"],
            check=False,
        )
    if p.returncode != 0 and fatal:
        raise SystemExit(f"FAILED ({p.returncode}): {cmd}")
    if p.returncode != 0:
        print(f"nonzero exit ({p.returncode}), continuing: {cmd}", flush=True)


def timed(report_file, cmd):
    """Run cmd like `time -v` — Kaggle's image has no /usr/bin/time, so use
    wait4 rusage: wall clock + peak RSS (maxrss propagates up from ninja's
    waited children, i.e. the heaviest TU)."""
    print(f"+ {cmd}", flush=True)
    t0 = time.monotonic()
    p = subprocess.Popen(
        ["bash", "-c", f"set -o pipefail; {cmd} 2>&1 | tee -a {LOG}"])
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    wall = time.monotonic() - t0
    with open(report_file, "w") as f:
        f.write(f'\tCommand being timed: "{cmd}"\n')
        f.write(f"\tElapsed (wall clock) seconds: {wall:.1f}\n")
        f.write(f"\tMaximum resident set size (kbytes): {ru.ru_maxrss}\n")
        f.write(f"\tExit status: {p.returncode}\n")
    if p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): {cmd}")


os.makedirs(OUT, exist_ok=True)

# --- toolchain ------------------------------------------------------------
run("curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/"
    "download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh")
run(f"bash /tmp/mf.sh -b -p {MF}")
run(f"{CONDA} create -y -n h3 -c conda-forge "
    "cmake ninja cxx-compiler python=3.12 jsonschema nanobind")
with open(os.path.expanduser("~/.condarc"), "w") as f:
    f.write("channels:\n  - conda-forge\n")

# --- sources ----------------------------------------------------------------
run(f"git clone --depth 50 --single-branch -b {BRANCH} "
    f"https://github.com/d-callan/hyphy-3 {REPO}")
run(f"git clone --depth 1 --branch 3.4.0 "
    f"https://gitlab.com/libeigen/eigen.git {EIGEN}")
os.chdir(REPO)
run("git rev-parse HEAD | tee /kaggle/working/out/postrefactor-commit.txt")

# --- phase 1: recipe-fidelity build (tests OFF, -j2), timed -------------------
run(f"{ENV_BIN}/cmake -S . -B build -G Ninja "
    f"-DCMAKE_BUILD_TYPE=Release "
    f"-DCMAKE_PREFIX_PATH={ENV} "
    f"-DHYPHY3_BUILD_TESTS=OFF "
    f"-DFETCHCONTENT_SOURCE_DIR_EIGEN3={EIGEN} "
    f"-DPython3_EXECUTABLE={ENV_BIN}/python")
timed(f"{OUT}/time-ninja-off.txt", f"{ENV_BIN}/ninja -C build -j{JOBS}")
run("cp build/.ninja_log /kaggle/working/out/ninja_log_postrefactor.txt")
run(f"{ENV_BIN}/python scripts/parse_ninja_log.py build --json "
    f"2>&1 > {OUT}/ninja_postrefactor.json")
run(f"{ENV_BIN}/python scripts/parse_ninja_log.py build")

# --- phase 2: tests ON (adds test TUs), timed ---------------------------------
run(f"{ENV_BIN}/cmake -S . -B build -DHYPHY3_BUILD_TESTS=ON")
timed(f"{OUT}/time-ninja-testson.txt", f"{ENV_BIN}/ninja -C build -j{JOBS}")
run("cp build/.ninja_log /kaggle/working/out/ninja_log_testson.txt")
run(f"{ENV_BIN}/python scripts/parse_ninja_log.py build --json "
    f"2>&1 > {OUT}/ninja_testson.json")

# --- gates ------------------------------------------------------------------
# CHECK FIRST against committed tests/golden snapshots: for a pure refactor
# this is the real regression gate. --record second for artifact salvage only
# (a record-then-check ordering would compare the run against itself).
# non-fatal: a failed case still leaves its siblings' outputs for salvage
run(f"{ENV_BIN}/python tests/golden_diff.py --check --tier all "
    f"--bin build/hyphy3 --workdir {OUT}/golden-work-check", fatal=False)
run(f"{ENV_BIN}/python tests/golden_diff.py --record --tier all "
    f"--bin build/hyphy3 --workdir {OUT}/golden-work", fatal=False)
shutil.copytree(f"{REPO}/tests/golden", f"{OUT}/golden-snapshots",
                dirs_exist_ok=True)
run(f"{ENV_BIN}/python tests/test_cli_surface.py build/hyphy3", fatal=False)
run(f"{ENV_BIN}/ctest --test-dir build -j{JOBS} "
    "--output-on-failure", fatal=False)

# --- artifacts ----------------------------------------------------------------
run("uname -a; nproc; free -g | head -2")
run(f"git -C {REPO} log --oneline -1")
# prune heavy dirs so the kernel-output download stays small (build tree ~GBs)
for d in (f"{WORKDIR}/miniforge", f"{REPO}/build", f"{WORKDIR}/eigen-src",
          f"{REPO}/.git"):
    shutil.rmtree(d, ignore_errors=True)
print("BASELINE CAPTURE DONE", flush=True)
