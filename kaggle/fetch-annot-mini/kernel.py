"""Kaggle kernel entrypoint: annotation-rich mini corpus for P3 (E3-E5).

Same scale as the 5group-mini corpus, but every assembly MUST have a
RefSeq GFF (REQUIRE_GFF=1) and the VEuPathDB tier is OFF: the rich label
columns (intron_frac, rna_frac, active_frame, struct_flags, domain_ids)
are only populated on the refseq_gff tier — VEuPathDB gene models carry
CDS intervals but no strand/phase/Dbxref, so its rows emit NA.

Groups cover intron-bearing eukaryotes (protozoa/fungi) where the
region-composition and splice-site tasks are non-degenerate, plus
bacteria/viral/archaea for taxonomy coverage. BOUNDARY_FRAC=0.3 (runner
default) enriches transition windows so struct_flags has positives.

FETCH_ONLY=1: fetch + contaminant screen + bundle, exit before training.
Output tarball lands in /kaggle/working; promote to a Kaggle Dataset so
the e3/e4/e5 kernels can mount it.
"""

import os
import socket
import subprocess

try:
    socket.create_connection(("github.com", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet + GPU), then re-push.")

WORK = "/kaggle/working"
os.environ.setdefault("WORKDIR", f"{WORK}/omniphyl-work")
os.environ.setdefault("OUTDIR", WORK)
# annot-mini: same groups/sampling as 5group-mini, GFF required, RefSeq
# tier (VEUPATH=0) so the P3 columns are populated on every row.
os.environ.setdefault("GROUPS", "protozoa fungi bacteria viral archaea")
os.environ.setdefault("LEVELS", "Complete Genome,Chromosome,Scaffold")
os.environ.setdefault("ONE_PER_RANK", "family")
os.environ.setdefault("PER_RANK_N", "1")
os.environ.setdefault("NT_BUDGET", "15000")
os.environ.setdefault("VEUPATH", "0")
os.environ.setdefault("REQUIRE_GFF", "1")
# transition windows are the positives for struct_flags and the
# frame-scan gate; the runner default is already 0.3.
os.environ.setdefault("BOUNDARY_FRAC", "0.3")
os.environ.setdefault("FETCH_ONLY", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# os.environ["REPO_URL"] = "https://<token>@github.com/d-callan/omniphyl.git"

subprocess.run(["git", "clone", "--depth", "1",
                os.environ.get(
                    "REPO_URL", "https://github.com/d-callan/omniphyl.git"),
                f"{WORK}/omniphyl"], check=True)
subprocess.run(["bash", f"{WORK}/omniphyl/remote/run_pilot.sh"], check=True)
