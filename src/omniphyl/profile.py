"""omniphyl profile — reference-free taxonomic + functional profiling.

Modes (docs/inference-plan.md §4):
  contig    - assembly contigs/scaffolds windowed like annotate; per-
              contig reconciled calls; abundance = fraction of BASES.
  precontig - raw reads assembled into local unitigs ("pre-contigs") by
              canonical-kmer clustering + greedy overlap-layout merge;
              windows pooled per unit. abundance = fraction of READS.
  raw       - each read a unit (truncated/padded to model length);
              abundance = fraction of READS.

Abundance is *sequence* abundance (bases or reads), NOT cell/genome-copy
abundance — the report says so explicitly.

Per-unit pooled predictions go beyond taxonomy: coding probability
(coding_frac mean), dominant active frame (active_frame head), Pfam /
InterPro domain hits (domains head), and an in-frame translated peptide
for units the model believes are coding.
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .encode import _AA_LUT, AA_LETTERS, BASE2DIGIT
from .io import iter_paired_sequences, iter_sequences, revcomp
from .model import OmniPhylPredictor

MAX_INPUT = 500          # model input budget (HASH_LEN bound)

# coding status thresholds on the mean sigmoid coding_frac of a unit's
# windows: deliberate gap between the labels because fragment-level
# coding estimates are noisy at read scale.
CODING_HI = 0.5
CODING_LO = 0.3


# ------------------------------------------------------- k-mer utils ---


def _kmer_ids(seq: str, k: int = 21) -> np.ndarray:
    """Canonical k-mer id per valid position (raw — intra-seq dupes
    kept). Clustering dedupes (kmer, read) pairs globally via one
    argsort, which is much cheaper than a Python set() per read."""
    codes = BASE2DIGIT[np.frombuffer(seq.upper().encode("ascii", "replace"),
                                     dtype=np.uint8)]
    if len(codes) < k:
        return np.empty(0, dtype=np.int64)
    win = np.lib.stride_tricks.sliding_window_view(codes, k)
    w = win[(win >= 0).all(axis=1)]
    if not len(w):
        return np.empty(0, dtype=np.int64)
    pow4 = 4 ** np.arange(k - 1, -1, -1, dtype=np.int64)
    idx = (w * pow4).sum(1)
    ridx = ((3 - w[:, ::-1]) * pow4).sum(1)
    return np.minimum(idx, ridx)


def _canon_kmers(seq: str, k: int = 21) -> set:
    """Canonical (min of fwd/RC) base-4 k-mer ids for a sequence."""
    return set(_kmer_ids(seq, k).tolist())


# ------------------------------------------------ pre-contig assembly --


def _overlap_len(a: str, b: str, k: int = 21,
                 min_id: float = 0.90) -> int:
    """Longest suffix-prefix overlap of `a` onto `b` (>=k, identity
    >= min_id), or 0. Vectorized per candidate length."""
    la, lb = len(a), len(b)
    maxl = min(la, lb)
    if maxl < k:
        return 0
    A = np.frombuffer(a.encode("ascii", "replace"), dtype=np.uint8)
    B = np.frombuffer(b.encode("ascii", "replace"), dtype=np.uint8)
    best = 0
    for L in range(k, maxl + 1):
        if int((A[la - L:] == B[:L]).sum()) >= min_id * L:
            best = L
    return best


def _end_anchors(seq: str, flank: int = 200, k: int = 15) -> tuple:
    """(head, tail) canonical k-mer sets over the sequence's ends.

    Any true >= k overlap must share an anchor; used to prune the
    O(n^2) overlap scan. Short anchor k (15) tolerates mismatch-dense
    overlaps that would erase all 21-mer anchors."""
    return (_canon_kmers(seq[:flank], k), _canon_kmers(seq[-flank:], k))


def assemble_cluster(seqs: list[str], k: int = 21, min_overlap: int = 30,
                     min_id: float = 0.90) -> list[str]:
    """Greedy overlap-layout merge of one cluster's reads -> unitig(s).

    Best-overlap-first via a lazy edge heap: candidate a->b edges
    (a-tail shares an anchor with b-head) are scored once; each merge
    bumps a per-item version so stale heap entries are skipped, and
    only the merged item's edges are rescanned. revcomp(b)'s head
    anchors ARE b's tail anchors (canonical kmers are rc-invariant),
    so both orientations cost one anchor pair. Returns one or more
    merged sequences -- an edge-connected cluster can still resolve
    into disjoint contigs (insertion gap / divergent haplotype), and
    reads fully contained in a contig simply stay separate units.
    """
    items = [s.upper() for s in seqs]
    anc = [_end_anchors(s) for s in items]      # (head, tail)
    alive = [True] * len(items)
    ver = [0] * len(items)
    heap: list[tuple] = []

    def scan_edges(i: int):
        """Push all valid i->j edges (both b orientations) for item i."""
        for j in range(len(items)):
            if j == i or not alive[j]:
                continue
            a_tail = anc[i][1]
            # b fwd: b-head; b rc: rc-head anchors == b-tail anchors
            for b_rc in (False, True):
                bh = anc[j][1] if b_rc else anc[j][0]
                if not (a_tail & bh):
                    continue
                bb = revcomp(items[j]) if b_rc else items[j]
                L = _overlap_len(items[i], bb, k, min_id)
                if L >= min_overlap:
                    heapq.heappush(
                        heap, (-L, i, j, b_rc, ver[i]))

    for i in range(len(items)):
        scan_edges(i)

    while heap:
        negs, i, j, b_rc, v = heapq.heappop(heap)
        if not (alive[i] and alive[j]) or ver[i] != v:
            continue                                  # stale
        L = -negs
        bb = revcomp(items[j]) if b_rc else items[j]
        items[i] = items[i] + bb[L:]
        anc[i] = _end_anchors(items[i])
        alive[j] = False
        ver[i] += 1
        scan_edges(i)

    return [s for s, a in zip(items, alive) if a]


def cluster_reads(ids: list[str], seqs: list[str], k: int = 21,
                  min_shared: int = 2, max_kmer_freq: float = 0.10,
                  max_cluster_reads: int = 64,
                  min_overlap: int = 30,
                  assemble: bool = True
                  ) -> list[tuple[list[str], list[str]]]:
    """Shared-kmer clustering -> per-cluster assembled unit sequences.

    Union-find over reads sharing >= min_shared canonical k-mers; a
    union is only allowed while the merged cluster stays under
    max_cluster_reads (bounds memory + transitive chaining). K-mers
    occurring in > max_kmer_freq of reads are masked (repeats /
    universal markers would chain unrelated loci). Each cluster is then
    overlap-assembled; returns (member_ids, [assembled seqs]) --
    usually one seq, more when assembly splits into disjoint contigs.

    Single-sort design: an earlier version held per-read k-mer sets +
    a kmer->reads dict over ALL distinct kmers, which needed tens of
    GB at ~400k reads (SIGKILL on Kaggle). Here all (kmer, read) pairs
    go into two flat int64 arrays; one argsort groups equal kmers, an
    adjacent-pair dedup makes each read count a kmer once, run lengths
    are the frequencies, and only non-singleton capped groups emit
    member pairs. Peak ~ n * read-kmers * ~30 B — a few GB at 400k
    reads vs ~30 GB before.
    """
    n = len(seqs)
    ka, ra = [], []
    for i, s in enumerate(seqs):
        kids = _kmer_ids(s, k)
        if kids.size:
            ka.append(kids)
            ra.append(np.full(kids.size, i, dtype=np.int32))
    shared: dict[tuple[int, int], int] = defaultdict(int)
    cap = max(2, int(max_kmer_freq * n))
    if ka:
        km = np.concatenate(ka)
        rd = np.concatenate(ra)
        del ka, ra
        order = np.argsort(km)
        km, rd = km[order], rd[order]
        del order
        # dedupe adjacent (kmer, read) repeats — a read counts a kmer
        # once regardless of how many positions it occurs at
        keep = np.r_[True, (km[1:] != km[:-1]) | (rd[1:] != rd[:-1])]
        km, rd = km[keep], rd[keep]
        del keep
        starts = np.flatnonzero(np.r_[True, km[1:] != km[:-1]])
        ends = np.r_[starts[1:], len(km)]
        del km
        # groups >200 members are combinatorially skipped (same rule
        # as before: each member pair is O(1) work only for small
        # groups; huge groups are near-mask kmers anyway)
        for st, en in zip(starts.tolist(), ends.tolist()):
            m = en - st
            if m < 2 or m > cap or m > 200:
                continue
            mem = rd[st:en]
            for a_i in range(m):
                for b_i in range(a_i + 1, m):
                    shared[(int(mem[a_i]), int(mem[b_i]))] += 1

    parent = list(range(n))
    size = [1] * n                     # cluster READ count

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), cnt in sorted(shared.items(), key=lambda x: -x[1]):
        if cnt < min_shared:
            continue
        ri_, rj_ = find(i), find(j)
        if ri_ == rj_:
            continue
        if size[ri_] + size[rj_] > max_cluster_reads:
            continue
        parent[rj_] = ri_
        size[ri_] += size[rj_]

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    out = []
    for members in groups.values():
        if len(members) == 1 or not assemble:
            # singleton, or pool-reads mode: emit member reads as-is
            out.append(([ids[i] for i in members],
                        [seqs[i] for i in members]))
        else:
            out.append(([ids[i] for i in members],
                        assemble_cluster([seqs[i] for i in members],
                                         min_overlap=min_overlap)))
    return out


def pack_fragments(seqs: list[str], max_chunk: int = MAX_INPUT,
                   sep: str = "NNNNN") -> list[str]:
    """'N'-join short unit seqs into <= max_chunk fragments, each seq
    kept intact (k-mer spectrum is additive across the N break)."""
    frags, cur = [], ""
    for s in seqs:
        piece = s[:max_chunk]
        cand = piece if not cur else cur + sep + piece
        if len(cand) > max_chunk:
            frags.append(cur)
            cur = piece
        else:
            cur = cand
    if cur:
        frags.append(cur)
    return frags


# ---------------------------------------------------- translation -----


def translate_frame(seq: str, frame: int) -> str:
    """In-frame peptide (standard code) for a 0-5 frame index.

    frames 0/1/2 = forward offsets; 3/4/5 = reverse-complement offsets
    (matches the model's active_frame head ordering). Codons containing
    non-ACGT bases -> 'X'; stop -> '*'."""
    if frame >= 3:
        seq = revcomp(seq)
        frame -= 3
    codes = BASE2DIGIT[np.frombuffer(seq.upper().encode("ascii",
                                                          "replace"),
                                     dtype=np.uint8)]
    aas = []
    for i in range(frame, len(codes) - 2, 3):
        c = codes[i:i + 3]
        if (c < 0).any():
            aas.append("X")
            continue
        aa = int(_AA_LUT[c[0] * 16 + c[1] * 4 + c[2]])
        aas.append("*" if aa == len(AA_LETTERS) else AA_LETTERS[aa])
    return "".join(aas)


# --------------------------------------------------------- run_profile ---


def _windows(seq: str, W: int, S: int) -> list:
    """Strided windows over a unit seq, tail-window included."""
    starts = ([0] if len(seq) <= W else
              list(range(0, len(seq) - W + 1, S)))
    if starts and starts[-1] + W < len(seq):
        starts.append(len(seq) - W)
    return [seq[a:a + W] for a in starts]


def _rows(arr, idxs):
    """Row-slice helper: None-safe array index, returns stacked rows."""
    if arr is None or not idxs:
        return None
    return arr[idxs]


def run_profile(model_path, inputs, out_prefix: Path,
                mode: str = "auto", reads2=None,
                window: int = 500, stride: int = 250,
                batch_size: int = 256, device: str = "auto",
                flat_prior_tau: float | None = None,
                conf_tau: float = 0.5, anc_tau: float = 0.2,
                cluster_k: int = 21, cluster_min_shared: int = 2,
                cluster_max_kmer_freq: float = 0.10,
                cluster_max_reads: int = 64,
                min_overlap: int = 30,
                pack_below: int = 300,
                domain_threshold: float = 0.5,
                pool_reads: bool = False,
                verbose: bool = True) -> dict:
    """Run profiling; writes precontig/taxonomy/function outputs;
    returns the summary dict."""
    out_prefix = Path(out_prefix)
    pred = OmniPhylPredictor(model_path, device=device,
                             flat_prior_tau=flat_prior_tau,
                             verbose=verbose)
    W, S = int(window), int(stride)

    # resolve mode
    if mode == "auto":
        mode = "contig"
        n_sample = 0
        for _i, s, _c in iter_sequences(Path(str(inputs[0]))):
            n_sample += 1
            if len(s) < 800:
                mode = "precontig"
            if n_sample >= 200:
                break

    # units: id, member read ids, emitted seqs, windows split into
    # real vs 'N'-joined virtual (encoded with zeroed pos channels)
    units: list[dict] = []
    if mode == "contig":
        for path in inputs:
            for cid, seq, cov in iter_sequences(Path(path)):
                ws = [w for w in _windows(seq, W, S)
                      if sum(1 for b in w if b.upper() not in "ACGT")
                      <= 0.5 * len(w)]
                units.append({"id": cid, "members": [cid],
                              "seqs": [seq], "windows": ws,
                              "virtual": [False] * len(ws),
                              "cov": float(cov), "bp": len(seq)})
    elif mode in ("precontig", "raw"):
        ids, seqs = [], []
        if reads2:
            for rid, s, _c in iter_paired_sequences(Path(inputs[0]),
                                                    Path(reads2)):
                ids.append(rid)
                seqs.append(s)
        else:
            for path in inputs:
                for rid, s, _c in iter_sequences(Path(path)):
                    ids.append(rid)
                    seqs.append(s)
        if mode == "precontig":
            groups = cluster_reads(ids, seqs, k=cluster_k,
                                   min_shared=cluster_min_shared,
                                   max_kmer_freq=cluster_max_kmer_freq,
                                   max_cluster_reads=cluster_max_reads,
                                   min_overlap=min_overlap,
                                   assemble=not pool_reads)
            by_id = dict(zip(ids, seqs))
            for gi, (members, useqs) in enumerate(groups):
                if pool_reads:
                    # cluster defines the pooling group only; each
                    # member read is its own real model input
                    frags = useqs
                    virt = [False] * len(frags)
                else:
                    # >= pack_below = real pre-contig windows; shorter
                    # leftovers 'N'-packed into <=500 bp virtual frags
                    big = [s for s in useqs if len(s) >= pack_below]
                    frags = big + pack_fragments(
                        [s for s in useqs if len(s) < pack_below])
                    virt = [False] * len(big) + \
                        [True] * (len(frags) - len(big))
                ws, wvirt = [], []
                for sq, v in zip(frags, virt):
                    w = _windows(sq, W, S)
                    ws += w
                    wvirt += [v] * len(w)
                units.append({"id": f"pc_{gi}", "members": members,
                              "seqs": frags, "windows": ws,
                              "virtual": wvirt, "cov": 1.0,
                              "bp": sum(len(by_id[m]) for m in members)})
        else:
            for rid, s in zip(ids, seqs):
                units.append({"id": rid, "members": [rid], "seqs": [s],
                              "windows": [s[:W]], "virtual": [False],
                              "cov": 1.0, "bp": len(s)})

    if verbose:
        print(f"omniphyl profile [{mode}]: {len(units)} units, "
              f"{sum(len(u['windows']) for u in units):,} model inputs")

    # chunked inference + incremental pooling. predict_sequences
    # materializes every head output for every input row (the domains
    # head alone is B x D floats), so holding one BatchPredictions
    # across ALL windows OOMs at metagenome scale — observed SIGKILL
    # on Kaggle. Pool each unit's rows as fixed-size chunks complete;
    # only the pooled running sums are retained. A single unit with
    # more windows than the chunk is sliced across chunks (its sums
    # still pool correctly).
    POOL_CHUNK = 20000                          # windows per predict call

    def _pool_chunk(slices):
        """Predict one chunk -> accumulate per-unit running sums."""
        seqs_r, seqs_v = [], []
        rows_map = []                            # per slice: (u, rr, rv)
        for u, off, take in slices:
            rr, rv = [], []
            for w, v in zip(u["windows"][off:off + take],
                            u["virtual"][off:off + take]):
                if v:
                    rv.append(len(seqs_v))
                    seqs_v.append(w)
                else:
                    rr.append(len(seqs_r))
                    seqs_r.append(w)
            rows_map.append((u, rr, rv))
        bp_r = (pred.predict_sequences(
            seqs_r, batch_size=batch_size, virtual_joined=False)
            if seqs_r else None)
        bp_v = (pred.predict_sequences(
            seqs_v, batch_size=batch_size, virtual_joined=True)
            if seqs_v else None)

        def rows(name, rr, rv, dict_key=None):
            parts = []
            for bp, idxs in ((bp_r, rr), (bp_v, rv)):
                if bp is None or not idxs:
                    continue
                arr = getattr(bp, name)
                if arr is None:
                    continue
                a = arr[dict_key] if dict_key is not None else arr
                parts.append(a[idxs])
            return np.concatenate(parts) if parts else None

        for u, rr, rv in rows_map:
            a = u.setdefault("acc", {
                "n": 0, "logp": {}, "pod": {},
                "cf": 0.0, "af": None, "dom": None})
            n_win = len(rr) + len(rv)
            a["n"] += n_win
            for t in pred.cls_targets:
                a["logp"][t] = a["logp"].get(t, 0.0) + \
                    rows("logp", rr, rv, t).sum(0)
                a["pod"][t] = a["pod"].get(t, 0.0) + \
                    float(rows("p_ood", rr, rv, t).sum())
            r = rows("coding_frac", rr, rv)
            if r is not None:
                a["cf"] += float(r.sum())
            r = rows("active_frame", rr, rv)
            if r is not None:
                a["af"] = r.sum(0) + (a["af"]
                                      if a["af"] is not None else 0)
            r = rows("domains", rr, rv)
            if r is not None:
                a["dom"] = r.sum(0) + (a["dom"]
                                       if a["dom"] is not None else 0)

    # build slices: units in order, intra-unit slicing for units
    # larger than POOL_CHUNK
    slices, nw = [], 0
    for u in units:
        off = 0
        while off < len(u["windows"]):
            take = min(POOL_CHUNK - nw, len(u["windows"]) - off)
            slices.append((u, off, take))
            off += take
            nw += take
            if nw >= POOL_CHUNK:
                _pool_chunk(slices)
                if verbose:
                    done = sum(a["n"] for a in
                               (x.get("acc") or {} for x in units))
                    print(f"  profile: pooled {done:,} windows",
                          flush=True)
                slices, nw = [], 0
    if slices:
        _pool_chunk(slices)

    # per-unit pooled calls
    tax_rows = []
    read_assign: dict[str, tuple] = {}
    total_reads = sum(len(u["members"]) for u in units) or 1
    total_bp = sum(u["bp"] for u in units) or 1
    for u in units:
        if not u["windows"]:
            continue
        a, nn = u["acc"], max(u["acc"]["n"], 1)
        pooled = {t: (a["logp"][t] / nn).reshape(1, -1)
                  for t in pred.cls_targets}
        call = pred.call(
            pooled, conf_tau=conf_tau, anc_tau=anc_tau,
            p_ood={t: np.array([a["pod"][t] / nn])
                   for t in pred.cls_targets})[0]
        doms = []
        if pred.domain_vocab and a["dom"] is not None:
            mp = a["dom"] / nn
            id2dom = {i: d for d, i in pred.domain_vocab.items()}
            doms = [id2dom[i] for i in np.argsort(-mp)
                    if mp[i] >= domain_threshold]
        coding_prob = a["cf"] / nn
        status = ("coding" if coding_prob >= CODING_HI else
                  "noncoding" if coding_prob < CODING_LO else "ambig")
        frame = (int((a["af"] / nn).argmax())
                 if a["af"] is not None else -1)
        longest = max(u["seqs"], key=len) if u["seqs"] else ""
        pep = ""
        if status == "coding" and 0 <= frame < 6 and longest:
            pep = translate_frame(longest, frame)
        frame_lbl = (f"{'+' if frame < 3 else '-'}{frame % 3}"
                     if 0 <= frame < 6 else "none")
        tax_rows.append({
            "unit": u["id"], "mode": mode,
            "n_members": len(u["members"]), "n_windows":
            len(u["windows"]),
            "length": max((len(s) for s in u["seqs"]), default=0),
            "member_bp": u["bp"],
            "cov": u["cov"] if mode == "contig" else "",
            "deepest_rank": call.deepest_rank or "unclassified",
            "lineage": call.lineage,
            "conf_at_deepest": (round(call.path_conf[call.deepest_rank],
                                      4) if call.deepest_rank else 0.0),
            "p_ood_domain": round(call.p_ood.get("domain", 0.0), 4),
            "coding_prob": round(coding_prob, 4),
            "coding_status": status, "frame": frame_lbl,
            "domains": ",".join(doms), "pep": pep,
            "_seqs": u["seqs"],
        })
        for m in u["members"]:
            read_assign[m] = (call.lineage, call.deepest_rank)

    # ---------------- abundance ------------------------------------------
    if mode == "contig":
        weights = {r["unit"]: r["member_bp"]
                   * max(r["cov"] or 1.0, 1e-9) for r in tax_rows}
    else:
        weights = {r["unit"]: float(r["n_members"]) for r in tax_rows}
    total_w = sum(weights.values()) or 1.0
    abund: dict[str, dict[str, float]] = {}
    for r in tax_rows:
        a = abund.setdefault(r["lineage"], {"weight": 0.0, "bp": 0.0,
                                            "n_units": 0})
        a["weight"] += weights[r["unit"]]
        a["bp"] += r["member_bp"]
        a["n_units"] += 1

    # ---------------- outputs ---------------------------------------------
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cols = ("unit", "mode", "n_members", "n_windows", "length",
            "member_bp", "cov", "deepest_rank", "lineage",
            "conf_at_deepest", "p_ood_domain", "coding_prob",
            "coding_status", "frame", "domains")
    with open(str(out_prefix) + ".precontigs.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in tax_rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    with open(str(out_prefix) + ".precontigs.fasta", "w") as f:
        for r in tax_rows:
            for si, s in enumerate(r["_seqs"]):
                tag = r["unit"] if len(r["_seqs"]) == 1 \
                    else f"{r['unit']}.{si}"
                f.write(f">{tag} len={len(s)} reads={r['n_members']} "
                        f"taxon=\"{r['lineage']}\" "
                        f"conf={r['conf_at_deepest']} "
                        f"coding={r['coding_prob']} "
                        f"frame={r['frame']} domains={r['domains']}\n"
                        f"{s}\n")
    n_pep = 0
    with open(str(out_prefix) + ".precontigs.pep.fasta", "w") as f:
        for r in tax_rows:
            if r["pep"]:
                n_pep += 1
                f.write(f">{r['unit']} frame={r['frame']} "
                        f"taxon=\"{r['lineage']}\"\n{r['pep']}\n")
    unit_word = "bases" if mode == "contig" else "reads"
    with open(str(out_prefix) + ".taxonomic_profile.tsv", "w") as f:
        f.write(f"# sequence_abundance = fraction of {unit_word} "
                "(NOT cell/genome-copy abundance)\n")
        f.write("deepest_rank\tlineage\tabundance\tfrac_bp\tn_units\n")
        for lin, v in sorted(abund.items(),
                             key=lambda x: -x[1]["weight"]):
            rnk = next((r["deepest_rank"] for r in tax_rows
                        if r["lineage"] == lin), "unclassified")
            f.write(f"{rnk}\t{lin}\t{v['weight'] / total_w:.6f}\t"
                    f"{v['bp'] / total_bp:.6f}\t{v['n_units']}\n")
    # functional profile: domain hits normalized per Mb of coding seq
    coding_bp = sum(r["member_bp"] for r in tax_rows
                    if r["coding_status"] == "coding") or 1
    dcount = defaultdict(int)
    for r in tax_rows:
        for d in (r["domains"].split(",") if r["domains"] else []):
            dcount[d] += 1
    with open(str(out_prefix) + ".functional_profile.tsv", "w") as f:
        f.write("domain_id\tn_units\thits_per_mb_coding\n")
        for d, cnt in sorted(dcount.items(), key=lambda x: -x[1]):
            f.write(f"{d}\t{cnt}\t{1e6 * cnt / coding_bp:.1f}\n")
    if mode != "contig":
        with open(str(out_prefix) + ".reads.tsv", "w") as f:
            f.write("read_id\tdeepest_rank\tlineage\n")
            for rid in sorted(read_assign):
                lin, dr = read_assign[rid]
                f.write(f"{rid}\t{dr or 'unclassified'}\t{lin}\n")

    uncl = sum(weights[r["unit"]] for r in tax_rows
               if r["deepest_rank"] == "unclassified") / total_w
    n_coding = sum(1 for r in tax_rows
                   if r["coding_status"] == "coding")
    summary = {
        "mode": mode,
        "n_units": len(tax_rows),
        "n_reads": (total_reads if mode != "contig" else ""),
        "n_lineages": len(abund),
        "unclassified_fraction": round(uncl, 4),
        "coding_units": n_coding,
        "coding_fraction_units": round(n_coding / max(len(tax_rows), 1),
                                       4),
        "peptides_emitted": n_pep,
        "abundance_basis": ("contig bases (cov-weighted when available)"
                            if mode == "contig" else "reads assigned"),
        "top_lineages": [
            {"lineage": lin, "abundance": round(v["weight"] / total_w, 6)}
            for lin, v in sorted(abund.items(),
                                 key=lambda x: -x[1]["weight"])[:10]],
    }
    (Path(str(out_prefix) + ".summary.json")).write_text(
        json.dumps(summary, indent=1))
    if verbose:
        print(f"omniphyl profile: wrote {out_prefix}.{{precontigs.tsv,"
              f"precontigs.fasta,precontigs.pep.fasta,"
              f"taxonomic_profile.tsv,functional_profile.tsv,"
              f"summary.json}}")
    return summary
