#!/usr/bin/env python3
"""Pack a fragment corpus (fragments.fna + labels.tsv) into a zero-copy
memory-mapped store for ``train_omniphyl.py --packed``.

Writes ``<out>/`` containing::

    seqs.u8          concatenated uppercase ASCII bases (1 byte/bp)
    seq_offsets.npy  int64 [N+1] cumulative byte offsets into seqs.u8
    seq_ids.txt      one frag_id per line, aligned with offset rows
    labels.tsv       verbatim copy of the input labels
    meta.json        format + provenance record

Sequences are opened via ``np.memmap`` so every DataLoader worker and
every DDP rank shares one OS page cache -- the multi-GB-per-process RAM
duplication of ``--cache-seqs`` disappears. labels.tsv is copied
verbatim so packed and raw corpora parse through the identical code
path (test_tol_arch checks bit-identical parity).

Usage:
  python scripts/pack_corpus.py --fragments fragments.fna \
      --labels labels.tsv --out corpus.packed
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np


def pack(fragments: Path, labels: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    seqs_path = out / "seqs.u8"
    ids_path = out / "seq_ids.txt"
    offs: list[int] = [0]
    n = 0
    total = 0
    with open(fragments, "rb") as fin, \
            open(seqs_path, "wb") as fseq, \
            open(ids_path, "w", encoding="utf-8") as fid_f:
        fid: str | None = None
        buf: list[bytes] = []

        def flush() -> None:
            nonlocal n, total
            if fid is None:
                return
            blob = b"".join(buf).upper()
            fseq.write(blob)
            total += len(blob)
            offs.append(offs[-1] + len(blob))
            fid_f.write(fid + "\n")
            n += 1

        for line in fin:
            if line.startswith(b">"):
                flush()
                fid = line[1:].strip().split()[0].decode("ascii", "replace")
                buf = []
            else:
                buf.append(line.strip())
        flush()
    np.save(out / "seq_offsets.npy", np.asarray(offs, dtype=np.int64))
    shutil.copy2(labels, out / "labels.tsv")
    meta = {
        "format": "packed-v1",
        "n_fragments": n,
        "total_nt": total,
        "source_fragments": str(fragments),
        "source_labels": str(labels),
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"packed {n} fragments ({total / 1e6:.1f} Mb) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fragments", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="output packed directory")
    args = ap.parse_args()
    pack(args.fragments, args.labels, args.out)


if __name__ == "__main__":
    main()
