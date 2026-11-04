"""omniphyl profile — reference-free taxonomic + functional profiling.

Modes (docs/inference-plan.md §4):
  contig    - assembly contigs/scaffolds windowed like annotate; per-
              contig reconciled calls; abundance = fraction of BASES.
  precontig - raw reads joined into ~500 bp virtual fragments by
              canonical 21-mer shared-kmer clustering; abundance =
              fraction of READS assigned to each lineage.
  raw       - each read a unit (truncated/padded to model length);
              abundance = fraction of READS.

Abundance is *sequence* abundance (bases or reads), NOT cell/genome-copy
abundance — the report says so explicitly.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .encode import BASE2DIGIT, HASH_LEN
from .io import iter_paired_sequences, iter_sequences
from .model import OmniPhylPredictor, RANKS

MAX_INPUT = 500          # model input budget (HASH_LEN bound)


# ------------------------------------------------------- k-mer utils ---


def _canon_kmers(seq: str, k: int = 21) -> set:
    """Canonical (min of fwd/RC) base-4 k-mer ids for a sequence."""
    codes = BASE2DIGIT[np.frombuffer(seq.upper().encode("ascii", "replace"),
                                     dtype=np.uint8)]
    n = len(codes)
    if n < k:
        return set()
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    valid = (win >= 0).all(axis=1)
    if not valid.any():
        return set()
    w = win[valid]
    idx = (w * (4 ** np.arange(k - 1, -1, -1, dtype=np.int64))).sum(1)
    comp = 3 - w[:, ::-1]
    ridx = (comp * (4 ** np.arange(k - 1, -1, -1, dtype=np.int64))).sum(1)
    return set(np.minimum(idx, ridx).tolist())


def cluster_reads(ids: list[str], seqs: list[str], k: int = 21,
                  min_shared: int = 2, max_kmer_freq: float = 0.10,
                  max_chunk: int = 500, sep: str = "NNNNN"
                  ) -> list[tuple[list[str], str]]:
    """Bounded shared-kmer clustering -> virtual 'N'-joined fragments.

    Union-find over reads sharing >= min_shared canonical k-mers.
    K-mers seen in > max_kmer_freq fraction of reads are ignored
    (repeats / universal markers would chain everything together).
    A cluster stops accepting new reads once joining would exceed
    max_chunk bp.
    """
    n = len(seqs)
    ksets = [_canon_kmers(s, k) for s in seqs]
    freq = defaultdict(int)
    for ks in ksets:
        for km in ks:
            freq[km] += 1
    # floor at 2: a k-mer present in exactly 2 reads is always a
    # legitimate pairwise signal — the mask exists to drop near-universal
    # kmers, which needs a large n to be meaningful
    cap = max(2, int(max_kmer_freq * n))
    index: dict[int, list[int]] = defaultdict(list)
    for i, ks in enumerate(ksets):
        for km in ks:
            if freq[km] <= cap:
                index[km].append(i)

    parent = list(range(n))
    size = [len(s) for s in seqs]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    shared: dict[tuple[int, int], int] = defaultdict(int)
    for km, members in index.items():
        for a_i in range(len(members)):
            for b_i in range(a_i + 1, len(members)):
                if len(members) > 200:   # skip combinatorial blowup
                    break
                shared[(members[a_i], members[b_i])] += 1

    for (i, j), cnt in sorted(shared.items(), key=lambda x: -x[1]):
        if cnt < min_shared:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # bounded cluster length: joining must stay under max_chunk
        if size[ri] + len(sep) + size[rj] > max_chunk:
            continue
        parent[rj] = ri
        size[ri] += len(sep) + size[rj]

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for members in groups.values():
        joined = sep.join(seqs[i][:max_chunk] for i in members)
        out.append(([ids[i] for i in members], joined[:max_chunk]))
    return out


# --------------------------------------------------------- run_profile ---


def run_profile(model_path, inputs, out_prefix: Path,
                mode: str = "auto", reads2=None,
                window: int = 500, stride: int = 100,
                batch_size: int = 256, device: str = "auto",
                flat_prior_tau: float | None = None,
                conf_tau: float = 0.5, anc_tau: float = 0.2,
                cluster_k: int = 21, cluster_min_shared: int = 2,
                cluster_max_kmer_freq: float = 0.10,
                cluster_max_chunk: int = 500,
                domain_threshold: float = 0.5,
                verbose: bool = True) -> dict:
    """Run profiling; writes taxonomy/pfam/reads TSVs; returns report."""
    out_prefix = Path(out_prefix)
    pred = OmniPhylPredictor(model_path, device=device,
                             flat_prior_tau=flat_prior_tau,
                             verbose=verbose)
    W, S = int(window), int(stride)

    # resolve mode
    if mode == "auto":
        first = str(inputs[0])
        n_sample = 0
        mode = "contig"
        for _i, s, _c in iter_sequences(Path(first)):
            n_sample += 1
            if len(s) < 800:
                mode = "precontig"
            if n_sample >= 200:
                break

    units: list[tuple] = []   # (unit_id, [member_ids], [seqs or windows])
    if mode == "contig":
        for path in inputs:
            for cid, seq, cov in iter_sequences(Path(path)):
                ws = []
                starts = ([0] if len(seq) <= W else
                          list(range(0, len(seq) - W + 1, S)))
                if starts and starts[-1] + W < len(seq):
                    starts.append(len(seq) - W)
                for a in starts:
                    w = seq[a:a + W]
                    if sum(1 for b in w if b.upper() not in "ACGT") \
                            <= 0.5 * len(w):
                        ws.append(w)
                units.append((cid, [cid], seq, ws, float(cov), len(seq)))
    elif mode in ("precontig", "raw"):
        ids, seqs = [], []
        for path in inputs:
            for rid, s, c in iter_sequences(Path(path)):
                ids.append(rid)
                seqs.append(s)
        if reads2:
            ids2 = []
            seqs2 = []
            for rid, s, c in iter_paired_sequences(Path(inputs[0]),
                                                   Path(reads2)):
                ids2.append(rid)
                seqs2.append(s)
            ids, seqs = ids2, seqs2
        if mode == "precontig":
            groups = cluster_reads(ids, seqs, k=cluster_k,
                                   min_shared=cluster_min_shared,
                                   max_kmer_freq=cluster_max_kmer_freq,
                                   max_chunk=cluster_max_chunk)
            for gi, (members, joined) in enumerate(groups):
                units.append((f"cluster_{gi}", members, joined,
                              [joined], 1.0, len(joined)))
        else:
            for rid, s in zip(ids, seqs):
                w = s[:W]
                units.append((rid, [rid], s, [w], 1.0, len(s)))

    if verbose:
        print(f"omniphyl profile [{mode}]: {len(units)} units, "
              f"{sum(len(u[3]) for u in units):,} model inputs")

    # batch inference
    all_seqs = [w for u in units for w in u[3]]
    virtual = mode == "precontig"
    bp = pred.predict_sequences(all_seqs, batch_size=batch_size,
                                virtual_joined=virtual)

    # per-unit pooled calls
    tax_rows = []
    read_assign: dict[str, tuple] = {}
    pos = 0
    for uid, members, full_seq, ws, cov, ulen in units:
        idxs = list(range(pos, pos + len(ws)))
        pos += len(ws)
        if not idxs:
            continue
        pooled = {t: bp.logp[t][idxs].mean(0, keepdims=True)
                  for t in pred.cls_targets}
        pod = {t: bp.p_ood[t][idxs].mean() for t in pred.cls_targets}
        call = pred.call(
            pooled, conf_tau=conf_tau, anc_tau=anc_tau,
            p_ood={t: np.array([v]) for t, v in pod.items()})[0]
        doms = []
        if bp.domains is not None and pred.domain_vocab:
            id2dom = {i: d for d, i in pred.domain_vocab.items()}
            mean_p = bp.domains[idxs].mean(0)
            doms = [id2dom[i] for i in np.argsort(-mean_p)
                    if mean_p[i] >= domain_threshold]
        row = {
            "unit": uid, "mode": mode, "n_members": len(members),
            "n_windows": len(idxs), "length": ulen,
            "cov": cov if mode == "contig" else "",
            "deepest_rank": call.deepest_rank or "unclassified",
            "lineage": call.lineage,
            "conf_at_deepest": (round(call.path_conf[call.deepest_rank], 4)
                                if call.deepest_rank else 0.0),
            "p_ood_domain": round(call.p_ood.get("domain", 0.0), 4),
            "domains": ",".join(doms),
        }
        tax_rows.append(row)
        for m in members:
            read_assign[m] = (call.lineage, call.deepest_rank)

    # ---------------- abundance ------------------------------------------
    if mode == "contig":
        # sequence abundance = fraction of total BASES (weighted by
        # contig coverage when available)
        weights = {r["unit"]: r["length"] * max(r["cov"] or 1.0, 1e-9)
                   for r in tax_rows}
    else:
        weights = {r["unit"]: float(r["n_members"]) for r in tax_rows}
    total_w = sum(weights.values()) or 1.0
    abund: dict[str, dict[str, float]] = {}
    for r in tax_rows:
        lin = r["lineage"]
        abund.setdefault(lin, {"weight": 0.0, "n_units": 0})
        abund[lin]["weight"] += weights[r["unit"]]
        abund[lin]["n_units"] += 1
    abundance_rows = [
        {"lineage": lin,
         "sequence_abundance": round(v["weight"] / total_w, 6),
         "n_units": v["n_units"]}
        for lin, v in sorted(abund.items(),
                             key=lambda x: -x[1]["weight"])]

    # ---------------- outputs ---------------------------------------------
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_prefix) + ".taxonomy.tsv", "w") as f:
        f.write("unit\tmode\tn_members\tn_windows\tlength\tcov\t"
                "deepest_rank\tlineage\tconf_at_deepest\tp_ood_domain\t"
                "domains\n")
        for r in tax_rows:
            f.write("\t".join(str(r[k]) for k in
                              ("unit", "mode", "n_members", "n_windows",
                               "length", "cov", "deepest_rank", "lineage",
                               "conf_at_deepest", "p_ood_domain",
                               "domains")) + "\n")
    with open(str(out_prefix) + ".abundance.tsv", "w") as f:
        unit_word = "bases" if mode == "contig" else "reads"
        f.write(f"# sequence_abundance = fraction of {unit_word} "
                "(NOT cell/genome-copy abundance)\n")
        f.write("lineage\tsequence_abundance\tn_units\n")
        for r in abundance_rows:
            f.write(f"{r['lineage']}\t{r['sequence_abundance']}\t"
                    f"{r['n_units']}\n")
    if mode != "contig":
        with open(str(out_prefix) + ".reads.tsv", "w") as f:
            f.write("read_id\tdeepest_rank\tlineage\n")
            for rid in sorted(read_assign):
                lin, dr = read_assign[rid]
                f.write(f"{rid}\t{dr or 'unclassified'}\t{lin}\n")

    uncl = sum(weights[r["unit"]] for r in tax_rows
               if r["deepest_rank"] == "unclassified") / total_w
    report = {
        "mode": mode,
        "n_units": len(tax_rows),
        "n_lineages": len(abundance_rows),
        "unclassified_fraction": round(uncl, 4),
        "abundance_basis": ("contig bases (cov-weighted when available)"
                            if mode == "contig" else "reads assigned"),
        "top_lineages": abundance_rows[:10],
    }
    (Path(str(out_prefix) + ".report.json")).write_text(
        json.dumps(report, indent=1))
    if verbose:
        print(f"omniphyl profile: wrote {out_prefix}.{{taxonomy.tsv,"
              f"abundance.tsv,report.json}}")
    return report
