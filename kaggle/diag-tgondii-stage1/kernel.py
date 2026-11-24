"""Kaggle CPU kernel: isolate the tgondii Stage-1 annotate OOM.

History: the ~65 Mb TgondiiME49 genome OOM-kills (rc=-9) inside
run_annotate even split into ~20 Mb contig-group subprocesses
(STAGE1_SPLIT_BP=20e6). Kaggle kills ~29 GB RSS. A 20 Mb part
should not need 29 GB, so either one pathological contig or a
growth curve that scales super-linearly with input bp.

This kernel measures, not guesses:

  Phase A  contig census: id, length, N-frac for every contig;
           replicate the train_refiner._stage1_cache 20 Mb grouping
           to identify exactly which contigs formed the failed _p1.
  Phase B  run_annotate(tracks_only) on the LARGEST single contig
           alone, in a subprocess; parent polls /proc/<pid>/status
           VmHWM + VmRSS every 5 s and logs the RSS curve.
  Phase C  same subprocess monitor on the full _p1 group -- the
           repro of the OOM. If Phase B already peaks near the kill
           bound, the pathology is inside one contig's annotate; if
           only the group OOMs, growth scales with contig count.

Memory numbers land in stdout (kernel log) + memdiag.json.
"""

import glob
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM -- verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
REPO = "https://github.com/d-callan/omniphyl.git"
os.environ.setdefault("OMP_NUM_THREADS", "4")
DIAG = WORK / "diag"
DIAG.mkdir(exist_ok=True)

subprocess.run(["git", "clone", "--depth", "1", REPO,
                str(WORK / "omniphyl")], check=True)
sys.path.insert(0, str(WORK / "omniphyl/src"))
sys.path.insert(0, str(WORK / "omniphyl/scripts"))

from omniphyl.io import iter_sequences                    # noqa: E402


def find_one(pattern: str) -> str:
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}",
                            recursive=True))
    if not hits:
        raise SystemExit(f"missing {pattern}; "
                         f"mounted={os.listdir('/kaggle/input')}")
    return hits[0]


FASTA = Path(find_one("ToxoDB-71_TgondiiME49_Genome.fasta"))
MODEL = Path(find_one("w64_e15_best.pt"))

# ---- Phase A: census + reproduce the 20 Mb grouping ------------------
STAGE1_SPLIT_BP = 20_000_000
contigs = [(cid, seq) for cid, seq, _cov
           in iter_sequences(FASTA)]
print(f"{len(contigs)} contigs, "
      f"{sum(len(s) for _, s in contigs):,} bp total")
for cid, seq in sorted(contigs, key=lambda x: -len(x[1])):
    n_frac = seq.upper().count("N") / len(seq)
    print(f"  {cid:50s} {len(seq):>12,} bp  N={n_frac:.4f}",
          flush=True)

groups, cur, cur_bp = [], [], 0
for cid, seq in contigs:
    if cur and cur_bp + len(seq) > STAGE1_SPLIT_BP:
        groups.append(cur)
        cur, cur_bp = [], 0
    cur.append((cid, seq))
    cur_bp += len(seq)
if cur:
    groups.append(cur)
for gi, g in enumerate(groups):
    print(f"_p{gi}: {len(g)} contigs, "
          f"{sum(len(s) for _, s in g):,} bp "
          f"[{', '.join(cid for cid, _ in g)}]", flush=True)


def write_fasta(path, items):
    with open(path, "w") as out:
        for cid, seq in items:
            out.write(f">{cid}\n" + "\n".join(
                seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")


CHILD = (
    "import sys, json\n"
    "from pathlib import Path\n"
    "from omniphyl.annotate import run_annotate\n"
    "kw = json.loads(sys.argv[1])\n"
    "kw['out_prefix'] = Path(kw['out_prefix'])\n"
    "kw['model_path'] = Path(kw['model_path'])\n"
    "kw['fasta_path'] = Path(kw['fasta_path'])\n"
    "run_annotate(**kw)\n")


def rss_kb(pid):
    try:
        st = open(f"/proc/{pid}/status").read()
    except OSError:
        return 0, 0
    rss = hwm = 0
    for ln in st.splitlines():
        if ln.startswith("VmRSS:"):
            rss = int(ln.split()[1])
        elif ln.startswith("VmHWM:"):
            hwm = int(ln.split()[1])
    return rss, hwm


def monitored_annotate(tag, fasta_path):
    """run_annotate in a subprocess, VmRSS/VmHWM sampled every 5 s."""
    spec = json.dumps({
        "model_path": str(MODEL), "fasta_path": str(fasta_path),
        "out_prefix": str(DIAG / fasta_path.stem),
        "window": 500, "stride": 100, "bin_width": 0,
        "batch_size": 256, "device": "cpu",
        "tracks_only": True, "verbose": True})
    env = {**os.environ,
           "OMNIPHYL_DUMP_RES": "1",
           "PYTHONPATH": str(WORK / "omniphyl/src") + os.pathsep
           + os.environ.get("PYTHONPATH", "")}
    print(f"[{tag}] launch annotate on {fasta_path.name}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-c", CHILD, spec],
                            env=env, stdout=None, stderr=None)
    curve, peak = [], 0
    while proc.poll() is None:
        rss, hwm = rss_kb(proc.pid)
        peak = max(peak, hwm)
        if len(curve) == 0 or time.time() - curve[-1][0] >= 25:
            curve.append((time.time(),
                          round(rss / 1e6, 2),
                          round(hwm / 1e6, 2)))
            print(f"[{tag}] t={time.time() - t0:7.0f}s "
                  f"rss={rss / 1e6:5.2f}GB hwm={hwm / 1e6:5.2f}GB",
                  flush=True)
        time.sleep(5)
    rc = proc.returncode
    rss, hwm = rss_kb(proc.pid)
    peak = max(peak, hwm)
    print(f"[{tag}] DONE rc={rc} wall={time.time() - t0:.0f}s "
          f"peak_hwm={peak / 1e6:.2f}GB", flush=True)
    return {"tag": tag, "rc": rc,
            "wall_s": round(time.time() - t0, 1),
            "peak_hwm_gb": round(peak / 1e6, 2),
            "curve": [(round(t - t0, 0), r, h)
                      for t, r, h in curve]}


report = {"groups": [[cid for cid, _ in g] for g in groups]}
big = max(contigs, key=lambda x: len(x[1]))

# Phase B: largest contig alone
fa_one = DIAG / "largest_only.fasta"
write_fasta(fa_one, [big])
print(f"Phase B: largest contig {big[0]} ({len(big[1]):,} bp) alone",
      flush=True)
report["largest_alone"] = monitored_annotate("largest", fa_one)

# Phase C: the failing _p1 group as one annotate call (OOM repro)
p1 = groups[1] if len(groups) > 1 else groups[0]
fa_p1 = DIAG / "group_p1.fasta"
write_fasta(fa_p1, p1)
print(f"Phase C: _p1 group ({sum(len(s) for _, s in p1):,} bp, "
      f"{len(p1)} contigs)", flush=True)
report["group_p1"] = monitored_annotate("group_p1", fa_p1)

json.dump(report, open(WORK / "memdiag.json", "w"), indent=1)
print("wrote memdiag.json", flush=True)
