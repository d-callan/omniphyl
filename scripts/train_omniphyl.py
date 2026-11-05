#!/usr/bin/env python3
"""Train OmniPhyl Tier-0: per-fragment FCGR -> multi-rank taxonomy.

Reads fragments.fna + labels.tsv emitted by fetch_refseq_corpus.py.
FCGRs are computed on the fly in the DataLoader workers (no big tensors
on disk): sequence -> k-mer counts -> quadtree pixel image [1, 2^k, 2^k].

Splits are by *genome accession* so train/val never share a genome.
Optional --holdout-genus-frac holds out whole genera entirely (the
leave-clade-out probe: does the model still call their family/order?).

Deps: torch, numpy. Usage:
  python train_omniphyl.py --fragments ../data/tier0/fragments.fna \
      --labels ../data/tier0/labels.tsv --out ../runs/tier0_v1 \
      --epochs 30 --batch-size 256 --holdout-genus-frac 0.1
"""
import argparse
import csv
import json
import math
import random
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Single source of truth for encoders + model classes: the installable
# package in src/omniphyl/. The path insert keeps `python3
# scripts/train_omniphyl.py` (and tests) working without `pip install -e .`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from omniphyl.encode import (  # noqa: E402
    BASE2DIGIT, MS_KS, VTX,
    HASH_K, HASH_BUCKETS, HASH_PAD, HASH_LEN,
    AA_LETTERS, AA_ALPHA, AA_DIM, AA_EXTRA_DIM, AA_FRAME_DIM, _AA_LUT,
    build_cell_lut, rc_index_map,
    fcgr_encode, fcgr_encode_ms, hash_encode, aa_encode,
    aa_frame_encode, seq_encode)
from omniphyl.model import (  # noqa: E402
    RANKS, CODING_TARGETS, CONT_TARGETS,
    Block, ResNetEncoder, SeqEncoder, ViTEncoder,
    TreeRankHead, OmniPhyl,
    build_tree, build_flat_priors)


def jitter_seq(s: str, rate: float, seed: int) -> str:
    """Apply Poisson(rate)-distributed point substitutions per position.

    Deterministic in ``seed`` so a jitter sweep is reproducible and a
    given fragment mutates identically across workers. Used at eval to
    probe whether predictions rely on exact long k-mers (instance
    features) or compositional spectrum: one SNP destroys ~k overlapping
    k-mers but leaves GC/codon/dinucleotide statistics nearly unchanged,
    so the accuracy-vs-rate curve traces instance-feature reliance."""
    a = np.frombuffer(s.upper().encode("ascii"), dtype=np.uint8).copy()
    rng = np.random.default_rng(seed)
    n = rng.poisson(rate * len(a))
    if n == 0:
        return s
    pos = rng.choice(len(a), size=min(int(n), len(a)), replace=False)
    pos = pos[np.isin(a[pos], np.frombuffer(b"ACGT", dtype=np.uint8))]
    if len(pos) == 0:
        return s
    bases = np.frombuffer(b"ACGT", dtype=np.uint8)
    alt = bases[rng.integers(0, 4, len(pos))]
    neq = alt != a[pos]
    while not neq.all():
        redo = np.nonzero(~neq)[0]
        alt[redo] = bases[rng.integers(0, 4, len(redo))]
        neq = alt != a[pos]
    a[pos] = alt
    return a.tobytes().decode("ascii")


PE_LEN = 150    # unmerged paired-end mate length for var-len augment


def varlen_crop(s: str, min_len: int) -> str:
    """Stochastic read-length crop (E2): 70% uniform contiguous
    sub-window in [min_len, |s|]; 30% unmerged paired-end
    (PE_LEN prefix + 'N' + PE_LEN suffix). The 'N' junction breaks
    cross-insert k-mers and codons: both fcgr_encode and aa_encode
    drop windows/codons containing it.
    """
    if len(s) <= min_len:
        return s
    if np.random.random() < 0.3 and len(s) >= 2 * PE_LEN:
        return s[:PE_LEN] + "N" + s[-PE_LEN:]
    L = int(np.random.randint(min_len, len(s) + 1))
    o = int(np.random.randint(0, len(s) - L + 1))
    return s[o:o + L]


def eval_crop_seq(s: str, crop: int = 0, paired: int = 0) -> str:
    """Deterministic eval-time read simulation (E2):
    paired>0 -> prefix/suffix mates of `paired` bases joined by 'N';
    crop>0 -> truncate to the first `crop` bases (150/250bp reads).
    Seqs already <= the target length pass through unchanged.
    """
    if paired and len(s) > 2 * paired:
        return s[:paired] + "N" + s[-paired:]
    if crop and len(s) > crop:
        return s[:crop]
    return s


class FragmentDataset(Dataset):
    """Line-offset-indexed FASTA -> (FCGR tensor, per-target label ids)."""

    def __init__(self, fasta: Path, labels: dict[str, dict], k: int,
                 lut: np.ndarray, augment: bool = False, pos: bool = False,
                 targets: list[str] | None = None,
                 cont: dict[str, tuple] | None = None,
                 seq: bool = False, seq_len: int = 500,
                 ms_luts: dict[int, np.ndarray] | None = None,
                 ms_canons: dict[int, np.ndarray] | None = None,
                 aux: set | None = None,
                 hash_canon: np.ndarray | None = None,
                 jitter: float = 0.0,
                 aa_extra: bool = False,
                 var_len_prob: float = 0.0,
                 min_train_len: int = 500,
                 eval_crop: int = 0,
                 eval_paired: int = 0,
                 aa_frame: bool = False,
                 annot: dict | None = None,
                 cache_seqs: bool = False):
        self.fasta = Path(fasta)
        self.k = k
        self.lut = lut
        self.augment = augment
        self.pos = pos
        self.seq = seq
        self.seq_len = seq_len
        self.ms_luts = ms_luts      # set -> multi-scale RC-canonical FCGR
        self.ms_canons = ms_canons
        self.aux = aux or set()   # {"hash","aa"} extra input branches (A3)
        self.hash_canon = hash_canon
        self.jitter = jitter      # eval-only SNP rate; train keeps 0
        self.aa_extra = aa_extra  # gated aa mode: append frame/stop feats
        self.var_len_prob = var_len_prob   # E2: stochastic crop prob
        self.min_train_len = min_train_len
        self.eval_crop = eval_crop         # E2: deterministic read crop
        self.eval_paired = eval_paired     # E2: paired-end mate length
        self.aa_frame = aa_frame   # E4: emit per-frame aa summary vec
        self.annot = annot         # frag_id -> annot label tuple or None
        self._annot_default = (
            (-1.0,) * (ANNOT_BASE + annot["_k"]
                       + (1 if annot["_k"] else 0))
            if annot is not None else ())
        self.labels = labels  # frag_id -> {target: int}
        self.cont = cont or {}  # frag_id -> (coding_frac, boundary_offset)
        self.ids = list(labels.keys())
        self.targets = targets if targets is not None else RANKS
        idx_path = fasta.with_suffix(fasta.suffix + ".idx.json")
        if idx_path.exists():
            self.offsets = json.loads(idx_path.read_text())
        else:
            self.offsets = self._build_index(idx_path)
        # Optional in-RAM sequence store: one concatenated bytes blob plus
        # per-fid (start, len) offsets. Replaces per-sample open()+seek()
        # reads -- on 2-core VMs the syscall+decode overhead dominates the
        # dataloader (~6ms/frag) and starves the GPU. A single sequential
        # pass loads 900k x ~500bp fragments in <1GB; worker processes
        # share the blob copy-on-write.
        self._seq_blob: bytes | None = None
        self._seq_pos: dict[str, tuple[int, int]] = {}
        if cache_seqs:
            self._load_seqs()

    def _build_index(self, idx_path: Path):
        offsets = {}
        with open(self.fasta, "rb") as f:
            pos = f.tell()
            line = f.readline()
            while line:
                if line.startswith(b">"):
                    fid = line[1:].strip().decode()
                    offsets[fid] = pos
                pos = f.tell()
                line = f.readline()
        try:
            idx_path.write_text(json.dumps(offsets))
            print(f"indexed {len(offsets)} records -> {idx_path}")
        except OSError:
            # fasta dir may be read-only (mounted corpus); the in-memory
            # index is still valid, we just can't cache it to disk.
            print(f"indexed {len(offsets)} records (cache write skipped: "
                  f"{idx_path} not writable)")
        return offsets

    # Optional in-RAM sequence store: one concatenated bytes blob plus
    # per-fid (start, len) offsets. Replaces per-sample open()+seek()
    # reads -- on 2-core VMs the syscall+decode overhead dominates the
    # dataloader (~6ms/frag) and starves the GPU. A single sequential
    # pass loads 900k x ~500bp fragments in <1GB; worker processes
    # share the blob copy-on-write.

    def _load_seqs(self):
        """One sequential pass storing every needed record in memory."""
        wanted = set(self.ids)
        parts: list[bytes] = []
        pos: dict[str, tuple[int, int]] = {}
        cursor = 0
        fid = None
        with open(self.fasta, "rb") as f:
            for line in f:
                if line.startswith(b">"):
                    fid = line[1:].strip().decode()
                elif fid in wanted:
                    chunk = line.strip()
                    if fid not in pos:
                        pos[fid] = (cursor, 0)
                    parts.append(chunk)
                    cursor += len(chunk)
                    st, ln = pos[fid]
                    pos[fid] = (st, ln + len(chunk))
        missing = wanted - set(pos)
        if missing:
            raise KeyError(f"{len(missing)} frag_ids absent from "
                           f"{self.fasta} (first: {sorted(missing)[0]})")
        self._seq_blob = b"".join(parts)
        self._seq_pos = pos
        self.offsets = {}  # release the line-offset index
        print(f"cached {len(pos)} sequences "
              f"({len(self._seq_blob) / 2**20:.0f} MiB)")

    def _read_seq(self, fid: str) -> str:
        if self._seq_blob is not None:
            st, ln = self._seq_pos[fid]
            return self._seq_blob[st:st + ln].decode("ascii")
        off = self.offsets[fid]
        with open(self.fasta, "rb") as f:
            f.seek(off)
            f.readline()  # header
            chunks = []
            for line in f:
                if line.startswith(b">"):
                    break
                chunks.append(line.strip())
        return b"".join(chunks).decode("ascii")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        s = self._read_seq(fid)
        perm = np.random.permutation(4) if self.augment else None
        if perm is not None:
            # Permute at the SEQUENCE level so every encoder — FCGR
            # image, hash k-mers, AA histogram, frame summary — sees
            # the same permuted molecule. Digit-remap inside
            # fcgr_encode alone would leave the aux branches encoding
            # the unpermuted sequence (mismatched input branches).
            tbl = str.maketrans("ACGTacgt",
                                "".join("ACGT"[d] for d in perm) * 2)
            s = s.translate(tbl)
            perm = None
        if self.jitter:
            s = jitter_seq(s, self.jitter,
                           zlib.crc32(fid.encode("ascii")))
        cropped = False
        if self.var_len_prob and np.random.random() < self.var_len_prob:
            n0 = len(s)
            s = varlen_crop(s, self.min_train_len)
            # varlen_crop passes through unchanged when len(s) <= min_len
            cropped = len(s) != n0
        if self.eval_crop or self.eval_paired:
            s = eval_crop_seq(s, self.eval_crop, self.eval_paired)
        img = (seq_encode(s, self.seq_len, perm) if self.seq
               else fcgr_encode_ms(s, self.ms_luts, self.ms_canons,
                                   perm=perm, pos=self.pos)
               if self.ms_luts is not None
               else fcgr_encode(s, self.k, self.lut, perm, self.pos))
        lab = np.array([self.labels[fid].get(t, 0) for t in self.targets],
                       dtype=np.int64)
        c = self.cont.get(fid, (-1.0, -1.0))  # -1 = unlabeled/missing
        x = torch.from_numpy(img)
        if self.aux:
            x = {"x": x}
            if "hash" in self.aux:
                x["hash"] = torch.from_numpy(hash_encode(s, self.hash_canon))
            if "aa" in self.aux:
                x["aa"] = torch.from_numpy(aa_encode(s, self.aa_extra))
        if self.aa_frame:
            if not isinstance(x, dict):
                x = {"x": x}
            x["aa_frame"] = torch.from_numpy(aa_frame_encode(s))
        if self.annot is not None:
            c = c + self.annot.get(fid, self._annot_default)
        if cropped:
            # The crop shifts reading frame by (offset % 3), moves or
            # drops structural transitions, and the coding fraction was
            # measured on the full window — every window-derived target
            # (coding_frac, boundary, offset, region, frame, flags,
            # domains) is wrong for the cropped read. Mask them all to
            # the -1 unlabeled sentinel the losses already respect.
            c = (-1.0,) * len(c)
        return (x, torch.from_numpy(lab),
                torch.tensor(c, dtype=torch.float32))




def load_labels(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["frag_id"]] = r
    return rows


def encode_cont_labels(rows: dict[str, dict]) -> dict[str, tuple]:
    """frag_id -> (coding_frac, boundary_offset), -1.0 when absent.

    Boundary flag is NOT stored: it is derived at loss time as
    0 < coding_frac < 1, matching the candidacy semantics in
    docs/annotation-cascade.md. Missing/non-numeric/out-of-range
    values become -1.0 and are masked out of the loss.
    """
    out = {}
    for fid, m in rows.items():
        vals = []
        for col in ("coding_frac", "boundary_offset"):
            try:
                v = float(m.get(col, "") or -1.0)
            except (TypeError, ValueError):
                v = -1.0
            vals.append(v if 0 <= v <= 1 else -1.0)
        out[fid] = tuple(vals)
    return out


ANNOT_BASE = 9   # region_fracs(4) + active_frame(1) + struct_flags(4)


def encode_annot_labels(rows: dict[str, dict],
                        domain_vocab: dict | None = None
                        ) -> dict[str, tuple]:
    """frag_id -> (r_coding, r_intron, r_rna, r_intergenic,
                   active_frame, f_start, f_stop, f_donor, f_acceptor,
                   dom_0..K-1, dom_mask)  for the P3 annotation heads.

    -1.0 marks unlabeled dims (masked out of each loss).  domain_ids
    uses a trailing mask column because the column distinguishes three
    states: 'NA'/absent = no annotation (mask off), '' = annotated, no
    domains (all-negative label), 'IPR..,PF..' = multi-hot positive.
    out["_k"] carries K so callers can build a matching default row.
    """
    domain_vocab = domain_vocab or {}
    K = len(domain_vocab)
    out: dict[str, tuple] = {}
    for fid, m in rows.items():
        vals, ok = [], True
        for col in ("coding_frac", "intron_frac", "rna_frac"):
            v = m.get(col, "")
            if v in ("", "NA", None):
                ok = False
                break
            vals.append(float(v))
        region = ([vals[0], vals[1], vals[2],
                   max(0.0, 1.0 - sum(vals))] if ok else [-1.0] * 4)
        fv = m.get("active_frame", "")
        frame = float(int(fv)) if fv not in ("", "NA", None) else -1.0
        sv = m.get("struct_flags", "")
        flags = ([float((int(sv) >> b) & 1) for b in range(4)]
                 if sv not in ("", "NA", None) else [-1.0] * 4)
        doms: list[float] = []
        dm: list[float] = []
        if K:
            dv = m.get("domain_ids")
            if dv is None or dv == "NA":
                doms, dm = [0.0] * K, [-1.0]
            else:
                dm = [1.0]
                pos = {t for t in dv.split(",") if t}
                doms = [1.0 if i in pos else 0.0 for i in domain_vocab]
        out[fid] = tuple(region + [frame] + flags + doms + dm)
    out["_k"] = K
    return out


def build_split(meta: dict[str, dict], val_frac: float, genus_holdout: float,
                seed: int, frag_val: float = 0.0):
    """(val_accs, holdout_genera, val_frags).

    frag_val > 0 withholds a CONTIGUOUS span of ~frag_val fragments from
    every train-side accession as within-genome validation. Contiguous,
    not random: interleaved holdout leaks through near-identical
    neighbouring windows, whereas a held-out region tests whether the
    class signature persists in unsampled sequence context. This is what
    gives singleton-species genomes (one assembly, so genome-level val
    can never pair them) species-level val coverage."""
    rng = random.Random(seed)
    # genomes whose requested rank was unresolved in taxonomy are
    # test-only: never trained on, always held out (OOD probe set)
    forced_val = {m["accession"] for m in meta.values()
                  if m.get("rank_unresolved") == "1"}
    # genbank-sourced genomes are train-only: their taxonomy is
    # submitter-declared (no RefSeq QC), so keeping the held-out set
    # purely RefSeq keeps the benchmark clean when measuring whether
    # genbank augmentation helps underrepresented groups
    forced_train = {m["accession"] for m in meta.values()
                    if m.get("seq_source") == "genbank"}
    forced_val -= forced_train
    gb_genera = {m["genus"] for m in meta.values()
                 if m["accession"] in forced_train}
    all_accs = sorted({m["accession"] for m in meta.values()}
                      - forced_val - forced_train)
    acc_genus: dict[str, str] = {}
    acc_species: dict[str, str] = {}
    for m in meta.values():
        a = m["accession"]
        if a not in acc_genus:
            acc_genus[a] = m.get("genus", "")
            acc_species[a] = m.get("species", "")
    genus_accs: dict[str, set[str]] = defaultdict(set)
    species_accs: dict[str, set[str]] = defaultdict(set)
    for a, g in acc_genus.items():
        if g:
            genus_accs[g].add(a)
    for a, s in acc_species.items():
        if s:
            species_accs[s].add(a)
    # Prefer holding out genomes whose genus has >=2 assemblies so val_accs
    # does not silently orphan singleton genera from the training vocab
    # (holdout_genera owns open-set genus evaluation). Within multi-genome
    # genera, prioritize multi-assembly species and keep >=1 assembly per
    # genus/species in train.
    multi_sp = [a for a in all_accs
                if len(genus_accs.get(acc_genus[a], ())) >= 2
                and len(species_accs.get(acc_species[a], ())) >= 2]
    single_sp_multi_g = [a for a in all_accs
                         if len(genus_accs.get(acc_genus[a], ())) >= 2
                         and len(species_accs.get(acc_species[a], ())) < 2]
    single_g = [a for a in all_accs
                if len(genus_accs.get(acc_genus[a], ())) < 2]
    rng.shuffle(multi_sp)
    rng.shuffle(single_sp_multi_g)
    rng.shuffle(single_g)
    n_val = max(1, int(len(all_accs) * val_frac)) if val_frac > 0 else 0
    g_rem = {g: len(s) for g, s in genus_accs.items()}
    s_rem = {s: len(ac) for s, ac in species_accs.items()}
    picked: list[str] = []
    deferred: list[str] = []
    for a in multi_sp + single_sp_multi_g:
        if len(picked) >= n_val:
            break
        g, s = acc_genus[a], acc_species[a]
        if g and g_rem.get(g, 0) <= 1:
            deferred.append(a)
            continue
        if s and len(species_accs.get(s, ())) >= 2 and s_rem.get(s, 0) <= 1:
            deferred.append(a)
            continue
        picked.append(a)
        if g:
            g_rem[g] -= 1
        if s:
            s_rem[s] -= 1
    for a in deferred + single_g:
        if len(picked) >= n_val:
            break
        picked.append(a)
    val_accs = set(picked) | forced_val

    holdout_genera = set()
    if genus_holdout > 0:
        # genera containing any train-only (genbank) member are excluded:
        # holding one out would drag its genbank members into val anyway
        train_genera = sorted({m["genus"] for m in meta.values()
                               if m["genus"] and m["genus"] not in gb_genera
                               and m["accession"] not in val_accs})
        holdout_genera = set(rng.sample(
            train_genera, max(1, int(len(train_genera) * genus_holdout))))

    val_frags: set = set()
    if frag_val > 0:
        acc_genus: dict[str, str] = {}
        by_acc_seq: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list))
        for fid, m in meta.items():
            acc = m["accession"]
            acc_genus.setdefault(acc, m.get("genus", ""))
            by_acc_seq[acc][m.get("seqid", "")].append(
                (fid, int(m.get("start") or 0)))
        for acc, seqs in by_acc_seq.items():
            if acc in val_accs or acc_genus.get(acc) in holdout_genera:
                continue
            big = max(seqs.values(), key=len)
            big.sort(key=lambda t: t[1])
            w = int(round(len(big) * frag_val))
            if w < 1 or len(big) <= w:
                continue  # too few windows to spare
            i0 = rng.randint(0, len(big) - w)
            val_frags.update(fid for fid, _ in big[i0:i0 + w])
    return val_accs, holdout_genera, val_frags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragments", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="fraction of genomes held out for val")
    ap.add_argument("--holdout-genus-frac", type=float, default=0.02,
                    help="fraction of train genera moved to val entirely "
                         "(leave-clade-out probe)")
    ap.add_argument("--frag-val", type=float, default=0.05,
                    help="fraction of each train genome's windows withheld "
                         "as one contiguous span for within-genome val "
                         "(species coverage for singleton-species genomes)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--vertex-permute-augment", action="store_true",
                    help="randomly permute CGR corner vertices per fragment "
                         "(label-preserving augmentation)")
    ap.add_argument("--pos-channels", action="store_true",
                    help="add mean-CGR-(x,y) channels per cell "
                         "(PhageCGRNet positional features)")
    ap.add_argument("--backbone", choices=["resnet", "vit", "seq", "msfcgr"],
                    default="resnet",
                    help="encoder: resnet=FCGR CNN (default), vit=FCGR "
                         "global attention, seq=1D CNN over one-hot bases "
                         "(keeps motif/positional info FCGR discards), "
                         "msfcgr=RC-canonical k4/5/6 pyramid + "
                         "quadtree-aligned 2x2 resnet (A2)")
    ap.add_argument("--aux", type=str, default="",
                    help="comma list of extra input branches fused into "
                         "the latent (A3): hash=canonical 11-mer "
                         "EmbeddingBag, aa=6-frame AA 3-mer spectrum")
    ap.add_argument("--stem-stride", type=int, default=2,
                    help="resnet stem conv stride; 1 keeps full FCGR "
                         "resolution (stride-2 merges each k-mer's "
                         "terminal-base digit on both grid axes)")
    ap.add_argument("--seq-len", type=int, default=500,
                    help="pad/truncate fragments to this length for "
                         "--backbone seq")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout on the encoder latent before heads")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CE label smoothing; keeps softmax confidence "
                         "honest for the abstention metric")
    ap.add_argument("--wd", type=float, default=1e-4,
                    help="AdamW weight decay (default 1e-4)")
    ap.add_argument("--head", choices=["flat", "tree"], default="flat",
                    help="tree = cosine-prototype heads with sibling-"
                         "conditional softmax + prior logit adjustment")
    ap.add_argument("--prior-tau", type=float, default=1.0,
                    help="tau for the tau*log(nu) sibling-prior offset "
                         "(tree head, applied during training only)")
    ap.add_argument("--flat-prior-tau", type=float, default=0.0,
                    help="tau for unconditional prior logit adjustment on "
                         "FLAT heads: adds tau*log(nu) to logits inside "
                         "the training cross-entropy only (Menon et al. "
                         "2021); raw logits stay prior-balanced at eval")
    ap.add_argument("--aa-mode", choices=["plain", "bottleneck", "gated"],
                    default="plain",
                    help="aa aux fusion: plain concat; bottleneck forces a "
                         "64-d compress; gated adds a z_dna-driven scalar "
                         "gate + 18-d frame/stop features and moves "
                         "coding_frac onto a z_dna-only head")
    ap.add_argument("--aa-drop", type=float, default=0.0,
                    help="per-sample Bernoulli prob of zeroing the gated "
                         "aa branch during training (branch dropout)")
    ap.add_argument("--min-train-len", type=int, default=500,
                    help="E2: shortest stochastic sub-window when "
                         "--var-len-prob > 0")
    ap.add_argument("--var-len-prob", type=float, default=0.0,
                    help="E2: per-fragment prob of applying a variable "
                         "length crop (70% contiguous sub-window, 30% "
                         "unmerged paired-end) during training")
    ap.add_argument("--region-head", action="store_true",
                    help="E3: add region_fracs 4-way soft head "
                         "[coding, intron, rna, intergenic] with "
                         "inverse-prior weight on non-coding windows")
    ap.add_argument("--struct-head", action="store_true",
                    help="E4: add active_frame (8-way) + struct_flags "
                         "(4-bit) heads on cat([z, 144-d per-frame aa "
                         "summary]); emits x['aa_frame']")
    ap.add_argument("--domain-head-k", type=int, default=0,
                    help="E5: add a K-way multi-label domains head over "
                         "the top-K IPR/PF ids seen on >=20 train windows")
    ap.add_argument("--annot-weight", type=float, default=0.25,
                    help="gradient weight on the optional annotation-head "
                         "loss group (region_fracs, active_frame, "
                         "struct_flags, domains) relative to the "
                         "unit-weight taxonomy+continuous group")
    ap.add_argument("--annot-ctx", action="store_true",
                    help="condition annotation heads on detached "
                         "posteriors: domain-class probs feed "
                         "region/domains/struct heads and region probs "
                         "feed domains/struct (cross-task context with "
                         "no gradient path)")
    ap.add_argument("--cache-seqs", action="store_true",
                    help="load all fragment sequences into RAM once "
                         "(one sequential FASTA pass) instead of "
                         "open()+seek() per sample per epoch; ~1GB per "
                         "900k x 500bp fragments, removes the dataloader "
                         "bottleneck on low-core remote VMs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    meta = load_labels(args.labels)
    val_accs, holdout_genera, val_frags = build_split(
        meta, args.val_frac, args.holdout_genus_frac, args.seed,
        frag_val=args.frag_val)
    print(f"{len(val_accs)} val genomes, "
          f"{len(holdout_genera)} holdout genera, "
          f"{len(val_frags)} frag-val windows")

    def is_val(fid, m):
        return (m["accession"] in val_accs
                or m["genus"] in holdout_genera
                or fid in val_frags)

    train_meta = {k: v for k, v in meta.items() if not is_val(k, v)}
    val_meta = {k: v for k, v in meta.items() if is_val(k, v)}
    print(f"frags: {len(train_meta)} train / {len(val_meta)} val")

    # vocab from TRAIN rows only; val rows outside vocab -> -1 (unknown)
    vocab = {r: {} for r in RANKS}
    for m in train_meta.values():
        for r in RANKS:
            name = m.get(r, "")
            if name and name not in vocab[r]:
                vocab[r][name] = len(vocab[r])
    cls_targets = RANKS
    all_targets = cls_targets + CONT_TARGETS
    n_classes = {r: len(vocab[r]) + 1 for r in cls_targets}  # +1 = unknown
    n_classes.update({t: 1 for t in CONT_TARGETS})  # scalar heads
    # build_tree is computed for every head: the sibling priors drive
    # TreeRankHead, while parent_of is saved into the checkpoint so the
    # evaluator can reconcile flat-head calls along the taxonomy tree
    # at inference without retraining.
    tree = build_tree(train_meta, vocab, cls_targets)
    flat_priors = build_flat_priors(train_meta, vocab, cls_targets)

    def encode_labels(rows):
        out = {}
        for fid, m in rows.items():
            rec = {r: vocab[r].get(m.get(r, ""), -1)
                   if m.get(r, "") else -1 for r in cls_targets}
            # +1 offset unknown->0: shift all ids by 1 so 0 = unknown class
            out[fid] = {r: (v + 1 if v >= 0 else 0) for r, v in rec.items()}
        return out

    lut = build_cell_lut(args.k)
    cont_train = encode_cont_labels(train_meta)
    cont_val = encode_cont_labels(val_meta)

    # P3 annotation labels. domain_vocab comes from TRAIN rows only and
    # requires >=20 windows per domain so the multi-label head is not
    # fitted to singleton noise. The annot tuple is appended to yc when
    # any of the three heads is enabled.
    annot_any = (args.region_head or args.struct_head
                 or args.domain_head_k > 0)
    domain_vocab: dict[str, int] = {}
    annot_train = annot_val = None
    if annot_any:
        if args.domain_head_k:
            from collections import Counter
            cnt: Counter = Counter()
            for m in train_meta.values():
                dv = m.get("domain_ids")
                if dv and dv != "NA":
                    cnt.update(t for t in dv.split(",") if t)
            domain_vocab = {
                d: i for i, (d, c) in enumerate(cnt.most_common())
                if c >= 20 and i < args.domain_head_k}
            print(f"domain head: {len(domain_vocab)} of "
                  f"{args.domain_head_k} slots filled")
        annot_train = encode_annot_labels(train_meta, domain_vocab)
        annot_val = encode_annot_labels(val_meta, domain_vocab)

    use_seq = args.backbone == "seq"
    ms_maps = ({k: build_cell_lut(k) for k in MS_KS},
               {k: rc_index_map(k) for k in MS_KS}) \
        if args.backbone == "msfcgr" else (None, None)
    aux = {a for a in args.aux.split(",") if a}
    hash_canon = rc_index_map(HASH_K) if "hash" in aux else None
    aa_extra = args.aa_mode == "gated"
    train_ds = FragmentDataset(args.fragments, encode_labels(train_meta),
                               args.k, lut,
                               augment=args.vertex_permute_augment,
                               pos=args.pos_channels,
                               targets=cls_targets, cont=cont_train,
                               seq=use_seq, seq_len=args.seq_len,
                               ms_luts=ms_maps[0], ms_canons=ms_maps[1],
                               aux=aux, hash_canon=hash_canon,
                               aa_extra=aa_extra,
                               var_len_prob=args.var_len_prob,
                               min_train_len=args.min_train_len,
                               aa_frame=args.struct_head,
                               annot=annot_train,
                               cache_seqs=args.cache_seqs)
    val_ds = FragmentDataset(args.fragments, encode_labels(val_meta),
                             args.k, lut, pos=args.pos_channels,
                             targets=cls_targets, cont=cont_val,
                             seq=use_seq, seq_len=args.seq_len,
                             ms_luts=ms_maps[0], ms_canons=ms_maps[1],
                             aux=aux, hash_canon=hash_canon,
                             aa_extra=aa_extra,
                             aa_frame=args.struct_head,
                             annot=annot_val,
                             cache_seqs=args.cache_seqs)
    pin = device == "cuda"
    train_ld = DataLoader(train_ds, args.batch_size, shuffle=True,
                          num_workers=args.num_workers, drop_last=True,
                          pin_memory=pin,
                          persistent_workers=args.num_workers > 0)
    val_ld = DataLoader(val_ds, args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin,
                        persistent_workers=args.num_workers > 0)

    base_ch = len(MS_KS) if args.backbone == "msfcgr" else 1
    model = OmniPhyl(n_classes, args.width,
                     cin=base_ch + 2 if args.pos_channels else base_ch,
                     backbone=args.backbone, side=1 << args.k,
                     targets=all_targets, dropout=args.dropout,
                     stem_stride=args.stem_stride,
                     head=args.head, tree=tree if args.head == "tree" else None,
                     aux=aux, aa_mode=args.aa_mode,
                     aa_drop=args.aa_drop,
                     region_head=args.region_head,
                     struct_head=args.struct_head,
                     domain_k=len(domain_vocab),
                     annot_ctx=args.annot_ctx).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    steps = len(train_ld) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    amp = device == "cuda"

    # Device-resident tensors reused across all batches — hoisting them
    # out of the loop avoids a host->device transfer (and implicit sync)
    # per batch per rank.
    flat_priors_dev = ({t: flat_priors[t].to(device)
                        for t in cls_targets}
                       if args.flat_prior_tau > 0 else {})
    struct_pos_w = torch.tensor([5., 5., 8., 8.], device=device)

    def run(loader, train=True):
        model.train(train)
        # All epoch statistics accumulate on-device; host conversion
        # happens once at the end, so the batch loop issues zero
        # cudaStreamSynchronize calls beyond the ones autocast already
        # needs for grad updates.
        tot_loss_t = torch.zeros((), device=device)
        tot_n = 0
        correct = defaultdict(lambda: torch.zeros((), device=device))
        total = defaultdict(lambda: torch.zeros((), device=device))
        frac_err = torch.zeros((), device=device)
        frac_n = torch.zeros((), device=device)
        bnd_ok = torch.zeros((), device=device)
        bnd_n = torch.zeros((), device=device)
        off_err = torch.zeros((), device=device)
        off_n = torch.zeros((), device=device)
        for x, y, yc in loader:
            if isinstance(x, dict):
                x = {kk: v.to(device) for kk, v in x.items()}
            else:
                x = x.to(device)
            y, yc = y.to(device), yc.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=amp):
                z, logits = model(x)
                # optional P3 annotation-head losses are collected
                # separately so they can be scaled by --annot-weight
                # without diluting the taxonomy/cont group
                annot_terms: list = []
                if args.head == "tree":
                    # sibling-conditional NLL with prior offset; the
                    # conditional softmax is the training target while
                    # accuracy below still reads the chained marginals
                    clp = model.cond_logps(z, tau=args.prior_tau)
                    terms = []
                    for i, t in enumerate(cls_targets):
                        m = y[:, i] != 0
                        if not m.any():
                            continue
                        ce = -clp[t][m].gather(
                            1, (y[m, i] - 1)[:, None]).squeeze(1)
                        if args.label_smoothing:
                            ce = ((1 - args.label_smoothing) * ce
                                  - args.label_smoothing
                                  * clp[t][m].mean(1))
                        terms.append(ce.mean())
                else:
                    # flat-prior logit adjustment applies at train AND
                    # eval: the model's raw outputs are prior-negated
                    # (raw = posterior - tau*lp), so every consumer —
                    # loss, val acc, best-ckpt selection, eval_pilot —
                    # must score in the same adjusted space.
                    if args.flat_prior_tau > 0:
                        for t in cls_targets:
                            logits[t] = logits[t] + args.flat_prior_tau \
                                * flat_priors_dev[t]
                    # skip heads whose batch labels are all masked —
                    # cross_entropy(ignore_index=0) returns NaN there (val
                    # batches whose rank labels are entirely out-of-vocab)
                    terms = []
                    for i, t in enumerate(cls_targets):
                        if not (y[:, i] != 0).any():
                            continue
                        terms.append(F.cross_entropy(
                            logits[t], y[:, i], ignore_index=0,
                            label_smoothing=args.label_smoothing))
                fm = yc[:, 0] >= 0          # coding_frac label present
                btgt = ((yc[:, 0] > 0) & (yc[:, 0] < 1)).float()
                if fm.any():
                    terms.append(F.binary_cross_entropy_with_logits(
                        logits["coding_frac"][:, 0][fm], yc[fm, 0]))
                    terms.append(F.binary_cross_entropy_with_logits(
                        logits["boundary"][:, 0][fm], btgt[fm]))
                om = yc[:, 1] >= 0          # boundary_offset label present
                if om.any():
                    terms.append(F.mse_loss(torch.sigmoid(
                        logits["offset"][:, 0][om]), yc[om, 1]))
                if yc.size(1) > 2:          # P3 annotation columns
                    ya = yc[:, 2:]
                    if "region_fracs" in logits:
                        rm = ya[:, 0] >= 0
                        if rm.any():
                            # soft-CE on the composition simplex; pure
                            # non-coding windows are inverse-prior
                            # upweighted (91% coding imbalance)
                            pr = F.log_softmax(
                                logits["region_fracs"][rm], 1)
                            w = torch.where(
                                ya[rm, 0] < 0.1, 3.0, 1.0)
                            annot_terms.append(
                                -(w[:, None] * ya[rm, :4]
                                  * pr).sum(1).mean())
                    if "active_frame" in logits:
                        fm2 = ya[:, 4] >= 0
                        if fm2.any():
                            annot_terms.append(F.cross_entropy(
                                logits["active_frame"][fm2],
                                ya[fm2, 4].long()))
                    if "struct_flags" in logits:
                        sm = ya[:, 5] >= 0
                        if sm.any():
                            annot_terms.append(
                                F.binary_cross_entropy_with_logits(
                                    logits["struct_flags"][sm],
                                    ya[sm, 5:9],
                                    pos_weight=struct_pos_w))
                    if "domains" in logits:
                        dm = ya[:, -1] >= 0
                        if dm.any():
                            K = logits["domains"].size(1)
                            annot_terms.append(
                                F.binary_cross_entropy_with_logits(
                                    logits["domains"][dm],
                                    ya[dm, ANNOT_BASE:
                                       ANNOT_BASE + K]))
                if not terms and not annot_terms:
                    continue
                # taxonomy + the always-on continuous heads keep unit
                # weight, so enabling annotation heads can never dilute
                # their gradient share; the optional annotation group is
                # scaled by --annot-weight as a single block
                loss = (torch.stack(terms).mean() if terms
                        else z.new_zeros(()))
                if annot_terms:
                    loss = loss + args.annot_weight * torch.stack(
                        annot_terms).mean()
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            # Batch size from y: x may be a dict (len() would count
            # keys, not samples).
            bsz = y.size(0)
            tot_loss_t += loss.detach() * bsz
            tot_n += bsz
            if not train:
                # Per-rank accuracies + annotation extras are only
                # consumed by the caller on the val pass — computing
                # them in train mode wastes a full argmax over the
                # 12k-class species head per batch and forces host
                # syncs the training loop never uses.
                for i, t in enumerate(cls_targets):
                    m = y[:, i] != 0  # unknown/NA labels excluded
                    total[t] += m.sum()
                    correct[t] += (logits[t].argmax(1)[m]
                                   == y[:, i][m]).sum()
                if fm.any():
                    p_frac = torch.sigmoid(logits["coding_frac"][:, 0][fm])
                    frac_err += (p_frac - yc[fm, 0]).detach().abs().sum()
                    frac_n += fm.sum()
                    p_bnd = torch.sigmoid(logits["boundary"][:, 0][fm]) > 0.5
                    bnd_ok += (p_bnd == btgt[fm].bool()).sum()
                    bnd_n += fm.sum()
                if om.any():
                    p_off = torch.sigmoid(logits["offset"][:, 0][om])
                    off_err += (p_off - yc[om, 1]).detach().abs().sum()
                    off_n += om.sum()
        if train:
            return float(tot_loss_t) / max(tot_n, 1), {}, {}
        accs = {t: (float(correct[t]) / max(float(total[t]), 1))
                for t in cls_targets}
        extras = {
            "frac_mae": float(frac_err) / max(float(frac_n), 1),
            "boundary_acc": float(bnd_ok) / max(float(bnd_n), 1),
            "offset_mae": float(off_err) / max(float(off_n), 1),
        }
        return float(tot_loss_t) / max(tot_n, 1), accs, extras

    best = 0.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run(train_ld, train=True)
        va_loss, accs, extras = run(val_ld, train=False)
        msg = "  ".join(f"{t}:{a:.3f}" for t, a in accs.items())
        xmsg = "  ".join(f"{t}:{v:.3f}" for t, v in extras.items())
        print(f"ep {ep:03d} train {tr_loss:.4f} val {va_loss:.4f} | "
              f"{msg} | {xmsg} | {time.time()-t0:.0f}s")
        mean_acc = float(np.mean(list(accs.values())))
        ckpt = {"model": model.state_dict(), "vocab": vocab,
                "k": args.k, "targets": all_targets, "epoch": ep,
                "val_mean_acc": mean_acc,
                "args": {k: str(v) if isinstance(v, Path) else v
                         for k, v in vars(args).items()}}
        # parent_of + flat priors are saved for every head so
        # eval_pilot can do cross-rank tree reconciliation and
        # prior diagnostics without rebuilding them from labels.tsv
        ckpt["tree"] = {
            "parent_of": {r: tree["parent_of"][r].tolist()
                          for r in cls_targets},
            "log_prior": {r: tree["log_prior"][r].tolist()
                          for r in cls_targets}}
        ckpt["flat_log_prior"] = {
            r: flat_priors[r].tolist() for r in cls_targets}
        if domain_vocab:
            ckpt["domain_vocab"] = domain_vocab
        # Atomic per-epoch checkpoint: if the kernel is killed mid-save
        # (Kaggle 12h cap), last.pt still holds the previous complete
        # epoch rather than a truncated file.
        tmp_ck = args.out / "last.pt.tmp"
        torch.save(ckpt, tmp_ck)
        tmp_ck.replace(args.out / "last.pt")
        if mean_acc > best:
            best = mean_acc
            tmp_ck = args.out / "best.pt.tmp"
            torch.save(ckpt, tmp_ck)
            tmp_ck.replace(args.out / "best.pt")
    (args.out / "vocab.json").write_text(json.dumps(vocab, indent=1))
    print(f"done. best mean val acc {best:.3f} -> {args.out}/best.pt")


if __name__ == "__main__":
    main()
