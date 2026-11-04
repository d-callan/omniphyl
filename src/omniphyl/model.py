"""OmniPhyl model architecture + inference predictor — single source of truth.

The nn.Module classes, tree builders, and cross-rank reconciliation
helpers are extracted verbatim from ``scripts/train_omniphyl.py`` and
``scripts/eval_pilot.py``; those scripts re-export these names so
checkpoints remain bit-compatible and existing tests keep passing.

New inference-facing API:
  - :class:`OmniPhylPredictor` — checkpoint loading, batched
    ``predict_sequences`` (raw logits -> calibrated log-probs, OOD mass,
    continuous + optional heads), and ``reconcile_logp`` (strict NCBI
    ancestor projection, docs/inference-plan.md §2.4).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import (AA_DIM, AA_EXTRA_DIM, AA_FRAME_DIM, HASH_BUCKETS,
                     HASH_PAD, MS_KS, FeatureEncoder)

RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
CODING_TARGETS = ["coding", "non-coding"]  # legacy binary labels; superseded by fractions
# Continuous annotation heads (see docs/annotation-cascade.md):
#   coding_frac  mixture-aware fraction target (replaces binary coding)
#   boundary     candidacy: derived at loss time as 0 < coding_frac < 1
#   offset       normalized position of nearest interior CDS edge
CONT_TARGETS = ["coding_frac", "boundary", "offset"]


# ------------------------------------------------------------ encoder ---


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1, qt=False):
        super().__init__()
        if qt and stride == 2:
            # quadtree-aligned downsample: a 2x2 block is exactly the four
            # extensions of one (k-1)-mer prefix; 3x3s2 convs straddle
            # quadtree boundaries and mix unrelated prefixes.
            self.c1 = nn.Conv2d(cin, cout, 2, 2, 0, bias=False)
            self.short = nn.Conv2d(cin, cout, 2, 2, 0, bias=False)
        else:
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.short = (nn.Conv2d(cin, cout, 1, stride, bias=False)
                          if stride != 1 or cin != cout else nn.Identity())
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)

    def forward(self, x):
        s = self.short(x)
        x = F.relu(self.b1(self.c1(x)))
        x = self.b2(self.c2(x))
        return F.relu(x + s)


class ResNetEncoder(nn.Module):
    def __init__(self, cin: int, width: int, stem_stride: int = 2,
                 qt: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            # qt: 3x3 stride-1 stem preserves the finest FCGR resolution;
            # all halving happens in the quadtree-aligned stage blocks.
            nn.Conv2d(cin, width, 3 if qt else 5,
                      1 if qt else stem_stride,
                      1 if qt else 2, bias=False),
            nn.BatchNorm2d(width), nn.ReLU())
        self.stages = nn.Sequential(
            Block(width, width, qt=qt), Block(width, width, qt=qt),
            Block(width, width * 2, 2, qt),
            Block(width * 2, width * 2, qt=qt),
            Block(width * 2, width * 4, 2, qt),
            Block(width * 4, width * 4, qt=qt),
            Block(width * 4, width * 8, 2, qt),
            Block(width * 8, width * 8, qt=qt),
        )
        self.latent = width * 8

    def forward(self, x):
        return self.stages(self.stem(x)).mean(dim=(2, 3))


class SeqEncoder(nn.Module):
    """1D CNN over one-hot bases -- keeps motif + positional info the FCGR
    composition histogram discards.

    Global mean+max concat pooling: mean carries broad compositional
    signal, max carries presence of discriminative motifs anywhere in the
    window.
    """

    def __init__(self, width: int):
        super().__init__()

        def blk(i, o, s=1):
            return nn.Sequential(nn.Conv1d(i, o, 9, s, 4, bias=False),
                                 nn.BatchNorm1d(o), nn.ReLU())

        self.net = nn.Sequential(
            blk(4, width),
            blk(width, width * 2, 2),
            blk(width * 2, width * 2),
            blk(width * 2, width * 4, 2),
            blk(width * 4, width * 4),
            blk(width * 4, width * 8, 2),
        )
        self.latent = width * 8 * 2

    def forward(self, x):
        h = self.net(x)
        return torch.cat([h.mean(2), h.amax(2)], dim=1)


class ViTEncoder(nn.Module):
    """Patch-embed + TransformerEncoder + CLS latent (PCVR-style ViT on FCGR).

    Global self-attention mixes distant quadtree regions directly; useful
    where discriminative k-mers are sparse and spread across the grid.
    No conv inductive bias -> relies on data scale, which simulated
    fragments provide for free.
    """

    def __init__(self, cin: int, side: int, dim: int = 256, depth: int = 6,
                 heads: int = 8, patch: int = 8):
        super().__init__()
        n = (side // patch) ** 2
        self.patch = nn.Conv2d(cin, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pe = nn.Parameter(torch.randn(1, n + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.latent = dim

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)  # B, N, D
        t = torch.cat([self.cls.expand(t.size(0), -1, -1), t], 1)
        return self.norm(self.enc(t + self.pe))[:, 0]


# ------------------------------------------------- taxonomy structure ---


def build_tree(meta: dict[str, dict], vocab: dict[str, dict],
               ranks: list[str]) -> dict:
    """Sibling-conditional tree structure for the A1 head.

    For each rank r and shifted class id c (1..C-1):
      parent_of[r][c] = shifted id of the parent-rank class seen on the
      training rows carrying name c; 0 = parent missing/out-of-vocab ->
      the class joins the catch-all root group.
      log_prior[r][c] = log nu(c | parent) with add-one smoothing inside
      each sibling group; applied as a tau*log(nu) logit offset during
      training only (removed at inference for balanced conditionals).

    Parent names are resolved by majority vote over training rows, which
    absorbs the occasional lineage inconsistency in RefSeq labels.
    """
    counts = {r: np.zeros(len(vocab[r]) + 1) for r in ranks}
    parent_votes = {r: defaultdict(Counter) for r in ranks[1:]}
    for m in meta.values():
        for ri, r in enumerate(ranks):
            c = vocab[r].get(m.get(r, ""), -1)
            if c == -1:
                continue
            counts[r][c + 1] += 1
            if ri:
                parent_votes[r][m[r]][m.get(ranks[ri - 1], "")] += 1
    parent_of: dict[str, torch.Tensor] = {}
    log_prior: dict[str, torch.Tensor] = {}
    for ri, r in enumerate(ranks):
        n = len(vocab[r]) + 1
        po = np.zeros(n, dtype=np.int64)
        for name, c in vocab[r].items():
            if ri and parent_votes[r].get(name):
                pname = parent_votes[r][name].most_common(1)[0][0]
                po[c + 1] = vocab[ranks[ri - 1]].get(pname, -1) + 1
        parent_of[r] = torch.tensor(po)
        groups = defaultdict(list)
        for c in range(1, n):
            groups[int(po[c])].append(c)
        lp = np.zeros(n)
        for cs in groups.values():
            tot = counts[r][cs].sum() + len(cs)  # add-one smoothing
            for c in cs:
                lp[c] = float(np.log((counts[r][c] + 1) / tot))
        log_prior[r] = torch.tensor(lp, dtype=torch.float32)
    return {"parent_of": parent_of, "log_prior": log_prior}


def build_flat_priors(meta: dict[str, dict], vocab: dict[str, dict],
                      ranks: list[str]) -> dict[str, torch.Tensor]:
    """Unconditional per-rank class log-priors (flat-head logit adjustment).

    lp[r][c] = log nu(c) with add-one smoothing over TRAIN rows only.
    lp[r][0] is pinned to a large negative: class 0 is the unknown mask,
    never a learned prediction, so it must never be boosted.
    During training, adding tau * lp to the logits before cross-entropy
    (Menon et al. 2021 logit-adjusted loss) makes the learned raw logit
    approximate a prior-balanced posterior, undoing majority-class
    collapse without touching the inference head.
    """
    counts = {r: np.zeros(len(vocab[r]) + 1) for r in ranks}
    for m in meta.values():
        for r in ranks:
            c = vocab[r].get(m.get(r, ""), -1)
            if c == -1:
                continue
            counts[r][c + 1] += 1
    out: dict[str, torch.Tensor] = {}
    for r in ranks:
        n = len(vocab[r]) + 1
        lp = np.full(n, -20.0, dtype=np.float32)
        tot = counts[r][1:].sum() + (n - 1)
        for c in range(1, n):
            lp[c] = float(np.log((counts[r][c] + 1.0) / tot))
        out[r] = torch.tensor(lp, dtype=torch.float32)
    return out


def build_parent_of(rows_iter, vocab: dict, ranks: list[str]) -> dict:
    """Majority-vote parent map when the checkpoint lacks ckpt["tree"].

    parent_of[r][c] = shifted class id of the most common parent-rank
    name on label rows carrying class c; 0 = parent missing/out-of-vocab.
    Identical construction to build_tree's parent vote.
    """
    votes = {r: defaultdict(Counter) for r in ranks[1:]}
    for m in rows_iter:
        for ri in range(1, len(ranks)):
            c_name = m.get(ranks[ri], "")
            if c_name in vocab[ranks[ri]]:
                votes[ranks[ri]][c_name][m.get(ranks[ri - 1], "")] += 1
    parent_of = {}
    for ri, r in enumerate(ranks):
        po = np.zeros(len(vocab[r]) + 1, dtype=np.int64)
        if ri:
            for name, c in vocab[r].items():
                pv = votes[r].get(name)
                if pv:
                    pname = pv.most_common(1)[0][0]
                    po[c + 1] = vocab[ranks[ri - 1]].get(pname, -1) + 1
        parent_of[r] = po
    return parent_of


class TreeRankHead(nn.Module):
    """Cosine-prototype head for one taxonomy rank (A1 arm).

    Logit for class c is a scaled cosine similarity against unit-norm
    prototype w_c, which removes the weight-norm advantage majority
    classes get from plain Linear heads. Class probabilities normalize
    *within sibling groups* (classes sharing the same parent id), not
    across the whole rank, so predictions chain into a coherent tree.
    """

    def __init__(self, latent: int, n_classes: int,
                 parent_of, log_prior):
        super().__init__()
        self.proto = nn.Parameter(
            F.normalize(torch.randn(n_classes - 1, latent), dim=1))
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.register_buffer("parent_of",
                             torch.as_tensor(parent_of, dtype=torch.long))
        self.register_buffer("log_prior",
                             torch.as_tensor(log_prior, dtype=torch.float32))

    def cond_logits(self, z, tau: float = 0.0):
        """log P(c | parent(c), z) for classes 1..C-1 as a [B, C-1] tensor."""
        zn = F.normalize(z, dim=1)
        w = F.normalize(self.proto, dim=1)
        # fp32: bf16 under autocast breaks scatter_add_ (exp is autocast to
        # fp32) and is too coarse for the logsumexp below anyway
        lg = (self.scale * (zn @ w.T)).float()
        if tau:
            lg = lg + tau * self.log_prior[1:]
        pidx = self.parent_of[1:].expand(lg.size(0), -1)
        n_groups = int(self.parent_of.max().item()) + 1
        denom = torch.zeros(lg.size(0), n_groups,
                            dtype=lg.dtype, device=lg.device)
        denom.scatter_add_(1, pidx, lg.exp())
        return lg - denom.gather(1, pidx).clamp_min(1e-30).log()

    def marginal_logp(self, z, prev_ext=None, tau: float = 0.0):
        """Chained log-marginals for classes 1..C-1.

        prev_ext: [B, C_prev] log-marginals of the previous rank with a
        leading 0 column (log 1) so orphan classes (parent_of==0) inherit
        root mass 1.0. tau reinstates the training-time prior offset for
        MAP decisions; tau=0 gives prior-balanced conditionals instead.
        """
        cond = self.cond_logits(z, tau)
        if prev_ext is None:
            return cond
        pidx = self.parent_of[1:].expand(z.size(0), -1)
        return cond + prev_ext.gather(1, pidx)


# ------------------------------------------------------------- model ----


class OmniPhyl(nn.Module):
    def __init__(self, n_classes: dict[str, int], width: int = 64,
                 cin: int = 1, backbone: str = "resnet", side: int = 128,
                 targets: list[str] | None = None, dropout: float = 0.0,
                 stem_stride: int = 2, head: str = "flat", tree=None,
                 aux: set | None = None,
                 aa_mode: str = "plain", aa_drop: float = 0.0,
                 region_head: bool = False, struct_head: bool = False,
                 domain_k: int = 0, annot_ctx: bool = False):
        super().__init__()
        self.annot_ctx = annot_ctx
        if backbone == "vit":
            self.encoder = ViTEncoder(cin, side, dim=width * 4)
        elif backbone == "seq":
            self.encoder = SeqEncoder(width)
        elif backbone == "msfcgr":
            self.encoder = ResNetEncoder(cin, width, qt=True)
        else:
            self.encoder = ResNetEncoder(cin, width, stem_stride)
        self.aux = aux or set()
        aux_dim = 128
        self.hash_bag = (nn.EmbeddingBag(HASH_BUCKETS + 1, aux_dim,
                                         mode="mean",
                                         padding_idx=HASH_PAD)
                         if "hash" in self.aux else None)
        self.aa_mode = aa_mode if "aa" in self.aux else "plain"
        self.aa_drop = aa_drop
        aa_in = AA_DIM + (AA_EXTRA_DIM if self.aa_mode == "gated" else 0)
        if "aa" not in self.aux:
            self.aa_proj = None
        elif self.aa_mode == "bottleneck":
            # force aa evidence through a 64-d bottleneck so it cannot
            # memorise genome identity and starve msfcgr (A3f pathology)
            self.aa_proj = nn.Sequential(nn.Linear(aa_in, 64),
                                         nn.LayerNorm(64), nn.GELU(),
                                         nn.Linear(64, aux_dim),
                                         nn.LayerNorm(aux_dim))
        else:
            self.aa_proj = nn.Sequential(nn.Linear(aa_in, aux_dim),
                                         nn.LayerNorm(aux_dim), nn.ReLU())
        # gated fusion: scalar gate per aux dim driven by z_dna + a
        # detached coding-fraction estimate + the 18-d frame summary.
        # coding_frac is predicted from z_dna alone so coding detection
        # never co-depends on the aa branch it gates.
        gated = self.aa_mode == "gated"
        self.coding_head_dna = (nn.Linear(self.encoder.latent, 1)
                                if gated else None)
        self.aa_gate = (nn.Linear(
            self.encoder.latent + 1 + AA_EXTRA_DIM, aux_dim)
            if gated else None)
        self.latent = self.encoder.latent + aux_dim * len(self.aux)
        self.infer_tau = 0.0   # prior offset at inference (MAP if >0)
        self.drop = nn.Dropout(dropout)
        keys = targets if targets is not None else list(n_classes.keys())
        if gated:              # coding_frac comes from z_dna, not z
            keys = [t for t in keys if t != "coding_frac"]
        tree_po = (tree or {}).get("parent_of", {}) if head == "tree" else {}
        self.heads = nn.ModuleDict(
            {t: (TreeRankHead(self.latent, n_classes[t],
                              tree["parent_of"][t], tree["log_prior"][t])
                 if t in tree_po else nn.Linear(self.latent, n_classes[t]))
             for t in keys})
        # P3 annotation heads (E3 region composition / E5 domain multi-
        # label) are plain linears on z; they ride the heads dict since
        # their input is the fused latent. The struct pair (E4) needs
        # cat([z, aa_frame]) so it lives outside heads and is applied
        # explicitly in forward.
        # annot_ctx=True instead conditions each annotation head on
        # DETACHED posteriors from the other tasks: every annot head
        # sees the domain-rank class probs, and the domains/struct pair
        # also sees the region-composition probs. Taxonomy can thus
        # inform annotation explicitly (splice sites are euk-only; TGA
        # is a stop except in code-4 clades) without a gradient path
        # that would let annot losses distort the taxonomy posterior.
        # Because their input dims differ, ctx heads sit outside the
        # heads dict under new names (head_region/head_domains), so
        # checkpoints trained without the flag stay loadable.
        n_dom = (n_classes.get("domain", 1) - 1) if annot_ctx else 0
        if annot_ctx:
            self.head_region = (nn.Linear(self.latent + n_dom, 4)
                                if region_head else None)
            self.head_domains = (nn.Linear(self.latent + n_dom + 4,
                                           domain_k)
                                 if domain_k else None)
        else:
            self.head_region = self.head_domains = None
            if region_head:
                self.heads["region_fracs"] = nn.Linear(self.latent, 4)
            if domain_k:
                self.heads["domains"] = nn.Linear(self.latent, domain_k)
        self.struct_head = struct_head
        if struct_head:
            struct_in = (self.latent + AA_FRAME_DIM
                         + (n_dom + 4 if annot_ctx else 0))
            self.head_frame = nn.Linear(struct_in, 8)
            self.head_flags = nn.Linear(struct_in, 4)

    def cond_logps(self, z, tau: float = 0.0):
        """Per-rank sibling-conditional log-probs (tree heads only)."""
        return {t: h.cond_logits(z, tau) for t, h in self.heads.items()
                if isinstance(h, TreeRankHead)}

    def forward(self, x):
        lcf = None
        if isinstance(x, dict):
            z_dna = self.encoder(x["x"])
            parts = [z_dna]
            if self.hash_bag is not None:
                h = self.hash_bag(x["hash"])
                parts.append(F.layer_norm(h, (h.size(1),)))
            if self.coding_head_dna is not None:
                lcf = self.coding_head_dna(z_dna)
            if self.aa_proj is not None:
                za = self.aa_proj(x["aa"])
                if self.aa_gate is not None:
                    gi = torch.cat(
                        [z_dna, lcf.detach(),
                         x["aa"][:, -AA_EXTRA_DIM:]], 1)
                    g = torch.sigmoid(self.aa_gate(gi))
                    if self.training and self.aa_drop > 0:
                        keep = torch.bernoulli(torch.full(
                            (za.size(0), 1), 1.0 - self.aa_drop,
                            device=za.device))
                        g = g * keep
                    za = g * za
                parts.append(za)
            z = self.drop(torch.cat(parts, 1))
        else:
            z = self.drop(self.encoder(x))
        logits = {}
        prev_ext = None  # [B, C_prev] chained log-marginals, col 0 = log 1
        for t, h in self.heads.items():
            if isinstance(h, TreeRankHead):
                logm = h.marginal_logp(z, prev_ext, self.infer_tau)
                prev_ext = torch.cat(
                    [logm.new_zeros(logm.size(0), 1), logm], 1)
                # col 0 (unknown class) gets a large negative so argmax
                # never lands there; cols 1.. are log-marginals
                logits[t] = torch.cat(
                    [logm.new_full((logm.size(0), 1), -1e4), logm], 1)
            else:
                logits[t] = h(z)
        if lcf is not None:
            logits["coding_frac"] = lcf
        if self.annot_ctx:
            # detached taxonomy->annotation context. Softmax over cols
            # 1.. only: col 0 (unknown mask) is an untrained free logit.
            # reg_p stays zero when no region head exists, keeping the
            # downstream input width constant.
            dl = logits.get("domain")
            dom_p = (dl.softmax(1)[:, 1:].detach() if dl is not None
                     else z.new_zeros(z.size(0), 0))
            if self.head_region is not None:
                logits["region_fracs"] = self.head_region(
                    torch.cat([z, dom_p], 1))
                reg_p = logits["region_fracs"].softmax(1).detach()
            else:
                reg_p = z.new_zeros(z.size(0), 4)
            if self.head_domains is not None:
                logits["domains"] = self.head_domains(
                    torch.cat([z, dom_p, reg_p], 1))
        if self.struct_head:
            xf = (x["aa_frame"] if isinstance(x, dict)
                  and "aa_frame" in x
                  else torch.zeros(z.size(0), AA_FRAME_DIM,
                                   device=z.device))
            zs = (torch.cat([z, xf, dom_p, reg_p], 1)
                  if self.annot_ctx else torch.cat([z, xf], 1))
            logits["active_frame"] = self.head_frame(zs)
            logits["struct_flags"] = self.head_flags(zs)
        return z, logits


# --------------------------------------------- reconciliation helpers ---


def ancestor_chains(parent_of: dict, ranks: list[str]) -> dict:
    """chains[r] -> int64 [n_r, ri+1]: col j = ancestor class id at
    ranks[j] for each class id at rank r (col ri = self; 0 = ancestor
    exists but is out-of-vocab/mask class; -1 = no parent known)."""
    chains = {}
    for ri, r in enumerate(ranks):
        n = len(parent_of[r])
        cols = [np.arange(n, dtype=np.int64)]
        cur = np.arange(n, dtype=np.int64)
        for k in range(ri, 0, -1):
            po_k = parent_of[ranks[k]]
            nxt = np.where(cur > 0, po_k[np.maximum(cur, 0)], -1)
            cols.insert(0, nxt)
            cur = nxt
        chains[r] = np.stack(cols, axis=1)
    return chains


def path_scores(logp: dict, chain: np.ndarray,
                ranks: list[str]) -> torch.Tensor:
    """Mean log-probability of each rank-r class's full ancestor path.

    chain: [C_r, L] numpy ancestor ids (col L-1 = self, -1 = missing).
    logp[rk]: [B, C_rk] log-softmax tensors. Ancestor id 0 (the
    out-of-vocab mask class) is treated as missing — its logit is an
    untrained free parameter and carries no signal.
    Returns [B, C_r] path-consistent scores (geometric-mean log-space).
    """
    ch = torch.from_numpy(chain)
    valid = ch > 0                     # mask of usable ancestor evidence
    denom = valid.sum(1).float()
    acc = torch.zeros(next(iter(logp.values())).size(0), ch.size(0))
    for j in range(ch.size(1)):
        ids = ch[:, j].clamp(min=0)
        g = logp[ranks[j]][:, ids]     # [B, C_r]
        acc += g * valid[:, j].float()
    # classes with zero usable ancestor evidence (the mask class 0, or
    # fully-orphaned lineages) must not get score 0.0 — that would beat
    # every real (negative) mean-logp path score.
    score = acc / denom.clamp(min=1).unsqueeze(0)
    score[:, denom == 0] = float("-inf")
    return score


# -------------------------------------------------------- predictor -----


@dataclass
class TaxonCall:
    """One reconciled taxonomy call for a pooled evidence unit.

    ``ids`` / ``names`` hold the STRICT NCBI-projected lineage: every
    rank shallower than ``deepest_rank`` is copied from the deepest
    confident class's true ancestor chain, never from independent
    argmax — so a call can never pair Family X with a genus that
    belongs to Family Y. Deeper ranks are 0 / unclassified.
    """
    ids: dict[str, int]
    names: dict[str, str]
    path_conf: dict[str, float]
    min_anc_prob: dict[str, float]
    p_ood: dict[str, float]
    deepest_rank: str | None
    lineage: str


@dataclass
class BatchPredictions:
    """Stacked numpy predictions for N input sequences."""
    logp: dict[str, np.ndarray]          # rank -> [N, C_r] log-probs (col 0 masked)
    p_ood: dict[str, np.ndarray]         # rank -> [N] softmax mass on class 0
    coding_frac: np.ndarray              # [N]
    boundary: np.ndarray                 # [N]
    offset: np.ndarray                   # [N]
    region_fracs: np.ndarray | None      # [N, 4]
    active_frame: np.ndarray | None      # [N, 8]
    struct_flags: np.ndarray | None      # [N, 4]
    domains: np.ndarray | None           # [N, K]
    extra: dict = field(default_factory=dict)


class OmniPhylPredictor:
    """Load a checkpoint once; run batched inference + reconciliation."""

    def __init__(self, model_path, device: str = "auto",
                 flat_prior_tau: float | None = None,
                 verbose: bool = True):
        model_path = Path(model_path)
        ckpt = torch.load(model_path, map_location="cpu",
                          weights_only=False)
        self.ckpt = ckpt
        self.vocab: dict[str, dict] = ckpt["vocab"]
        self.all_targets: list[str] = ckpt.get("targets", RANKS)
        self.k: int = ckpt["k"]
        self.train_args: dict = ckpt.get("args", {})

        # classification targets have vocab entries; continuous heads
        # (coding_frac/boundary/offset) are scalar outputs, n_classes=1.
        self.cls_targets = [t for t in self.all_targets
                            if t in self.vocab]
        n_classes = {t: len(self.vocab[t]) + 1 for t in self.cls_targets}
        n_classes.update({t: 1 for t in self.all_targets
                          if t not in n_classes})
        ta = self.train_args
        # old checkpoints store None for unset args — normalize to
        # defaults with `or` (dict.get default only fires on missing keys)
        base_ch = len(MS_KS) if ta.get("backbone") == "msfcgr" else 1
        self.model = OmniPhyl(
            n_classes,
            width=ta.get("width") or 64,
            cin=base_ch + 2 if ta.get("pos_channels") else base_ch,
            backbone=ta.get("backbone") or "resnet",
            side=1 << self.k,
            targets=self.all_targets,
            dropout=ta.get("dropout", 0.0) or 0.0,
            stem_stride=ta.get("stem_stride") or 2,
            head=ta.get("head") or "flat",
            tree=ckpt.get("tree"),
            aux={a for a in (ta.get("aux") or "").split(",") if a},
            aa_mode=ta.get("aa_mode") or "plain",
            aa_drop=float(ta.get("aa_drop", 0.0) or 0.0),
            region_head=bool(ta.get("region_head")),
            struct_head=bool(ta.get("struct_head")),
            domain_k=len(ckpt.get("domain_vocab", {})),
            annot_ctx=bool(ta.get("annot_ctx")))
        self.model.load_state_dict(ckpt["model"])

        # --- missing-head warnings (graceful degradation) ---------------
        self.has_region = bool(ta.get("region_head"))
        self.has_struct = bool(ta.get("struct_head"))
        self.has_domains = bool(ckpt.get("domain_vocab"))
        if verbose:
            for name, ok in (("region_fracs", self.has_region),
                             ("active_frame/struct_flags", self.has_struct),
                             ("domains", self.has_domains)):
                if not ok:
                    print(f"omniphyl: checkpoint lacks {name} head; "
                          "downstream outputs will be skipped or "
                          "fallback-computed", file=sys.stderr)

        # --- prior offsets -------------------------------------------------
        # tree head: reinstate the training tau so softmax(logits + tau*nu)
        # equals the posterior the model was fit to.
        self.model.infer_tau = (float(ta.get("prior_tau", 0.0) or 0.0)
                                if ta.get("head") == "tree" else 0.0)
        # flat head: logits were trained with tau * flat_log_prior added;
        # inference must re-add it (train tau restores the MAP posterior,
        # user tau rescales for metagenomic mixtures).
        self.train_flat_tau = float(ta.get("flat_prior_tau", 0.0) or 0.0)
        self.user_flat_tau = (flat_prior_tau
                              if flat_prior_tau is not None
                              else self.train_flat_tau)
        self.flat_lp: dict[str, torch.Tensor] = {}
        if ckpt.get("flat_log_prior"):
            self.flat_lp = {t: torch.as_tensor(
                ckpt["flat_log_prior"][t], dtype=torch.float32)
                for t in self.cls_targets
                if t in ckpt["flat_log_prior"]}
        elif (self.train_flat_tau or self.user_flat_tau):
            print("omniphyl: WARNING — checkpoint lacks flat_log_prior; "
                  "raw logits are prior-negated", file=sys.stderr)

        self.device = ("cuda" if device == "auto"
                       and torch.cuda.is_available() else
                       ("cpu" if device == "auto" else device))
        self.model.to(self.device).eval()
        self.encoder = FeatureEncoder(self.k, self.train_args)

        # --- NCBI reconciliation state --------------------------------------
        tree_po = (ckpt.get("tree") or {}).get("parent_of")
        self.chains: dict | None = None
        if tree_po is not None:
            self.chains = ancestor_chains(
                {r: np.asarray(tree_po[r], dtype=np.int64)
                 for r in self.cls_targets}, self.cls_targets)
        self.id2name: dict[str, dict[int, str]] = {
            t: {i + 1: n for n, i in v.items()}
            for t, v in self.vocab.items()}
        for t in self.id2name:
            self.id2name[t][0] = "<unknown>"
        self.domain_vocab = ckpt.get("domain_vocab", {})

    # ------------------------------------------------------------------
    def _batch_logits(self, seqs: list[str], batch_size: int,
                      virtual_joined: bool):
        """Yield per-batch (adjusted-logits dict, raw-logits dict)."""
        enc_dtype = {"x": torch.float32, "hash": torch.int64,
                     "aa": torch.float32, "aa_frame": torch.float32}
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            enc = self.encoder.encode_batch(chunk,
                                            virtual_joined=virtual_joined)
            x = {k: torch.as_tensor(v, dtype=enc_dtype[k])
                 .to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                _, logits = self.model(x)
            yield logits

    def predict_sequences(self, seqs: list[str], batch_size: int = 256,
                          virtual_joined: bool = False) -> BatchPredictions:
        """seqs -> stacked numpy predictions for all available heads."""
        n = len(seqs)
        logp: dict[str, list[np.ndarray]] = {t: []
                                             for t in self.cls_targets}
        p_ood: dict[str, list[np.ndarray]] = {t: []
                                              for t in self.cls_targets}
        cont: dict[str, list[np.ndarray]] = {t: [] for t in CONT_TARGETS}
        ann: dict[str, list[np.ndarray]] = {
            k: [] for k in ("region_fracs", "active_frame",
                            "struct_flags", "domains")}
        for logits in self._batch_logits(seqs, batch_size, virtual_joined):
            for t in self.cls_targets:
                raw = logits[t].float()
                # p_ood: softmax at the TRAINING flat tau — the network's
                # class-0 logit was calibrated against train_tau * lp[0],
                # so user overrides must not touch this statistic.
                if t in self.flat_lp:
                    raw_ood = raw + self.train_flat_tau \
                        * self.flat_lp[t].to(raw.device)
                else:
                    raw_ood = raw
                p_ood[t].append(raw_ood.softmax(1)[:, 0].cpu().numpy())
                # known-class logp: user-selected prior tau, class 0 masked.
                if t in self.flat_lp:
                    adj = raw + self.user_flat_tau \
                        * self.flat_lp[t].to(raw.device)
                else:
                    adj = raw
                adj = adj.clone()
                adj[:, 0] = -1e9
                logp[t].append(F.log_softmax(adj, 1).cpu().numpy())
            for t in CONT_TARGETS:
                if t in logits:
                    cont[t].append(
                        torch.sigmoid(logits[t][:, 0]).cpu().numpy())
            if "region_fracs" in logits:
                ann["region_fracs"].append(
                    logits["region_fracs"].softmax(1).cpu().numpy())
            if "active_frame" in logits:
                ann["active_frame"].append(
                    logits["active_frame"].softmax(1).cpu().numpy())
            if "struct_flags" in logits:
                ann["struct_flags"].append(
                    logits["struct_flags"].sigmoid().cpu().numpy())
            if "domains" in logits:
                ann["domains"].append(
                    logits["domains"].sigmoid().cpu().numpy())

        def cat(parts, shape_tail=()):
            if not parts or not parts[0]:
                return (np.zeros((0,) + shape_tail, np.float32)
                        if parts is None else np.zeros((n,) + shape_tail,
                                                       np.float32))
            return np.concatenate(parts)

        return BatchPredictions(
            logp={t: np.concatenate(v) if v else
                  np.zeros((n, len(self.vocab.get(t, {})) + 1), np.float32)
                  for t, v in logp.items()},
            p_ood={t: np.concatenate(v) if v else np.zeros(n, np.float32)
                   for t, v in p_ood.items()},
            coding_frac=(np.concatenate(cont["coding_frac"])
                         if cont["coding_frac"] else np.zeros(n, np.float32)),
            boundary=(np.concatenate(cont["boundary"])
                      if cont["boundary"] else np.zeros(n, np.float32)),
            offset=(np.concatenate(cont["offset"])
                    if cont["offset"] else np.zeros(n, np.float32)),
            region_fracs=(np.concatenate(ann["region_fracs"])
                          if ann["region_fracs"] else None),
            active_frame=(np.concatenate(ann["active_frame"])
                          if ann["active_frame"] else None),
            struct_flags=(np.concatenate(ann["struct_flags"])
                          if ann["struct_flags"] else None),
            domains=(np.concatenate(ann["domains"])
                     if ann["domains"] else None))

    # ------------------------------------------------------------------
    def reconcile_logp(self, pooled_logp: dict[str, np.ndarray],
                       conf_tau: float = 0.5, anc_tau: float = 0.2,
                       p_ood: dict[str, np.ndarray] | None = None
                       ) -> list[TaxonCall]:
        """Strict NCBI tree projection for pre-pooled log-probs.

        pooled_logp[r]: [B, C_r] numpy log-probs (e.g. the mean over a
        contig's windows). For each row b:
          1. path-score every class at every rank (mean logp along its
             NCBI ancestor chain);
          2. find the deepest rank r* whose argmax path score clears
             conf_tau AND whose weakest in-vocab ancestor prob clears
             anc_tau;
          3. emit shallower ranks by copying c*'s TRUE ancestor ids out
             of chains[r*] — independent per-rank argmaxes are never
             consulted, so emitted lineages are always tree-valid.
        """
        if self.chains is None:
            raise RuntimeError("checkpoint carries no NCBI parent map; "
                               "tree reconciliation unavailable")
        n = len(next(iter(pooled_logp.values())))
        lp = {t: torch.from_numpy(pooled_logp[t])
              for t in self.cls_targets}
        scores: dict[str, torch.Tensor] = {}
        cand: dict[str, torch.Tensor] = {}
        conf: dict[str, torch.Tensor] = {}
        for t in self.cls_targets:
            s = path_scores(lp, self.chains[t], self.cls_targets)
            s[:, 0] = -1e9  # class-0 mask id is not a real call
            scores[t] = s
            rs, rp = s.max(1)
            cand[t] = rp
            conf[t] = rs.exp()
        # weakest in-vocab ancestor probability along each argmax chain
        min_anc: dict[str, torch.Tensor] = {}
        for ri, t in enumerate(self.cls_targets):
            crow = torch.from_numpy(self.chains[t])[cand[t]]
            minp = torch.ones(n)
            for j in range(ri):
                ids = crow[:, j]
                m = ids > 0
                ap = lp[self.cls_targets[j]].exp().gather(
                    1, ids.clamp(min=0)[:, None]).squeeze(1)
                minp = torch.where(m, torch.minimum(minp, ap), minp)
            min_anc[t] = minp

        return self._emit_calls(lp, cand, conf, min_anc,
                                conf_tau, anc_tau, p_ood)

    def call(self, pooled_logp: dict[str, np.ndarray],
             conf_tau: float = 0.5, anc_tau: float = 0.2,
             p_ood: dict[str, np.ndarray] | None = None
             ) -> list[TaxonCall]:
        """Reconciled call when a parent map exists, else per-rank argmax
        fallback (deepest rank by raw softmax confidence)."""
        if self.chains is not None:
            return self.reconcile_logp(pooled_logp, conf_tau, anc_tau,
                                       p_ood)
        lp = {t: torch.from_numpy(pooled_logp[t])
              for t in self.cls_targets}
        cand = {t: lp[t].argmax(1) for t in self.cls_targets}
        conf = {t: lp[t].exp().max(1).values for t in self.cls_targets}
        min_anc = {t: torch.ones(len(cand[t])) for t in self.cls_targets}
        return self._emit_calls(lp, cand, conf, min_anc,
                                conf_tau, anc_tau, p_ood)

    def _emit_calls(self, lp, cand, conf, min_anc,
                    conf_tau, anc_tau, p_ood) -> list[TaxonCall]:
        n = len(next(iter(lp.values())))
        calls: list[TaxonCall] = []
        for b in range(n):
            ids: dict[str, int] = {}
            names: dict[str, str] = {}
            r_star: str | None = None
            c_star = 0
            for ri, t in enumerate(self.cls_targets):
                ok = (float(conf[t][b]) >= conf_tau
                      and float(min_anc[t][b]) >= anc_tau)
                if ok:
                    r_star, c_star = t, int(cand[t][b])
            if r_star is not None:
                ri_star = self.cls_targets.index(r_star)
                if self.chains is not None:
                    # project shallower ranks deterministically from the
                    # anchor class's true NCBI chain
                    chain_row = self.chains[r_star][c_star]
                    proj = {t: (max(0, int(chain_row[j]))
                                if j <= ri_star else 0)
                            for j, t in enumerate(self.cls_targets)}
                else:
                    # tree-less checkpoint: no parent map, so keep the
                    # per-rank argmax for ranks at/above the anchor
                    proj = {t: (int(cand[t][b])
                                if j <= ri_star else 0)
                            for j, t in enumerate(self.cls_targets)}
                for t in self.cls_targets:
                    cid = proj[t]
                    ids[t] = cid
                    names[t] = (self.id2name[t].get(cid, "<unknown>")
                                if cid else
                                f"unclassified_"
                                f"{self.id2name[r_star][c_star]}")
            else:
                for t in self.cls_targets:
                    ids[t] = 0
                    names[t] = "unclassified"
            lineage = ";".join(
                names[t] for t in self.cls_targets)
            calls.append(TaxonCall(
                ids=ids, names=names,
                path_conf={t: float(conf[t][b])
                           for t in self.cls_targets},
                min_anc_prob={t: float(min_anc[t][b])
                              for t in self.cls_targets},
                p_ood={t: float(p_ood[t][b])
                       for t in self.cls_targets}
                if p_ood is not None else {},
                deepest_rank=r_star,
                lineage=lineage))
        return calls
