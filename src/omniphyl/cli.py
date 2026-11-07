"""omniphyl CLI — `omniphyl annotate` and `omniphyl profile`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _common_model_args(ap):
    ap.add_argument("--model", required=True, type=Path,
                    help="OmniPhyl checkpoint (.pt) with vocab/tree/args")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--flat-prior-tau", type=float, default=None,
                    help="override flat-head prior tau for known-class "
                         "relative abundance; p_ood always uses the "
                         "checkpoint's training tau")
    ap.add_argument("--conf-tau", type=float, default=0.5,
                    help="minimum path confidence to emit a rank call")
    ap.add_argument("--anc-tau", type=float, default=0.2,
                    help="minimum weakest-ancestor prob for a rank call")
    ap.add_argument("--quiet", action="store_true")


def _annotate_parser(sub):
    ap = sub.add_parser("annotate",
                        help="annotate contigs with gene models + "
                             "taxonomy (GFF3, tracks, report)")
    _common_model_args(ap)
    ap.add_argument("--fasta", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="output prefix; writes PREFIX.{gff3,report.md,"
                         "report.json,coding.bedGraph,boundary.bedGraph,"
                         "contig_summary.tsv}")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--stride", type=int, default=100)
    ap.add_argument("--bin-width", type=int, default=0,
                    help="aggregation bin width in bp "
                         "(0 = gcd(stride, 50))")
    # LexicMap evidence
    le = ap.add_argument_group("lexicmap evidence")
    le.add_argument("--lexicmap-bed", type=Path, default=None,
                    help="LexicMap BED6 HSP intervals")
    le.add_argument("--lexicmap-bedgraph", type=Path, default=None,
                    help="bedGraph overlap-density track (no per-hit "
                         "filters possible)")
    le.add_argument("--lexicmap-tsv", type=Path, default=None,
                    help="raw LexicMap TSV (retains full per-HSP metadata)")
    le.add_argument("--lexicmap-source", choices=["rnaseq", "genomic"],
                    default="rnaseq",
                    help="index provenance: rnaseq enables splice/UTR "
                         "interpretation of endpoints; genomic treats "
                         "coverage as weak conservation evidence only")
    le.add_argument("--lexicmap-min-score", type=float, default=150.0,
                    help="minimum bitscore for partner HSPs of an "
                         "anchored split alignment")
    le.add_argument("--lexicmap-min-single-score", type=float,
                    default=300.0,
                    help="minimum bitscore for unchained (singleton) "
                         "HSPs -- stricter than the partner floor so "
                         "distant/repetitive hits need split "
                         "alignments to count")
    le.add_argument("--lexicmap-min-len", type=int, default=60)
    le.add_argument("--lexicmap-min-intron-support", type=int,
                    default=1,
                    help="minimum split alignments corroborating an "
                         "intron candidate")
    le.add_argument("--lexicmap-max-per-run", type=int, default=2,
                    help="per-SRA-run depth cap per bin")
    le.add_argument("--evidence-weight", type=float, default=0.25,
                    help="w_ev log-space boost of d~(x) into coding score")
    # decoding
    dec = ap.add_argument_group("decoding")
    dec.add_argument("--kingdom", default="auto",
                     choices=["auto", "prokaryote", "eukaryote", "viral"])
    dec.add_argument("--genetic-code", default="auto",
                     help="NCBI genetic code id, or 'auto' to infer "
                          "code-4 (TGA recoding) from in-frame codon usage")
    dec.add_argument("--min-cds-len", type=int, default=90)
    dec.add_argument("--coding-threshold", type=float, default=0.45)
    dec.add_argument("--boundary-threshold", type=float, default=0.25)
    dec.add_argument("--domain-threshold", type=float, default=0.5)
    dec.add_argument("--min-intron", type=int, default=35)
    dec.add_argument("--max-intron", type=int, default=5000)
    dec.add_argument("--ablate", default="",
                     help="comma-separated evidence streams to disable "
                          "post-aggregation (contribution audit): "
                          "frame,flags,region,coding,boundary,evidence,"
                          "domains")
    return ap


def _profile_parser(sub):
    ap = sub.add_parser("profile",
                        help="taxonomic + Pfam profiling of contigs or reads")
    _common_model_args(ap)
    ap.add_argument("inputs", nargs="+", type=Path,
                    help="FASTA/FASTQ files (gz ok)")
    ap.add_argument("--reads2", type=Path, default=None,
                    help="mate-2 file for paired-end reads")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["auto", "contig", "precontig",
                                       "raw"], default="auto")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--stride", type=int, default=250)
    ap.add_argument("--domain-threshold", type=float, default=0.5)
    # pre-contig clustering + assembly
    cl = ap.add_argument_group("pre-contig clustering")
    cl.add_argument("--cluster-k", type=int, default=21)
    cl.add_argument("--cluster-min-shared", type=int, default=2)
    cl.add_argument("--cluster-max-kmer-freq", type=float, default=0.10)
    cl.add_argument("--cluster-max-reads", type=int, default=64,
                    help="max reads per pre-contig cluster")
    cl.add_argument("--min-overlap", type=int, default=30,
                    help="min suffix-prefix overlap for unitig merge")
    cl.add_argument("--pack-below", type=int, default=300,
                    help="assembled seqs shorter than this are "
                         "'N'-packed into virtual <=500 bp fragments "
                         "instead of strided as real pre-contigs")
    cl.add_argument("--pool-reads", action="store_true",
                    help="skip unitig assembly: clusters only define "
                         "pooling groups, each member read is its own "
                         "model input")
    return ap


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="omniphyl",
        description="OmniPhyl inference: annotate + profile")
    ap.add_argument("--version", action="version",
                    version=f"omniphyl {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _annotate_parser(sub)
    _profile_parser(sub)
    args = ap.parse_args(argv)

    if args.cmd == "annotate":
        from .annotate import run_annotate
        run_annotate(
            model_path=args.model, fasta_path=args.fasta,
            out_prefix=args.out, window=args.window,
            stride=args.stride, bin_width=args.bin_width,
            batch_size=args.batch_size,
            device=args.device, flat_prior_tau=args.flat_prior_tau,
            lexicmap_bed=args.lexicmap_bed,
            lexicmap_bedgraph=args.lexicmap_bedgraph,
            lexicmap_tsv=args.lexicmap_tsv,
            lexicmap_source=args.lexicmap_source,
            lexicmap_min_score=args.lexicmap_min_score,
            lexicmap_min_len=args.lexicmap_min_len,
            lexicmap_min_single_score=args.lexicmap_min_single_score,
            lexicmap_min_intron_support=
            args.lexicmap_min_intron_support,
            lexicmap_max_per_run=args.lexicmap_max_per_run,
            evidence_weight=args.evidence_weight,
            kingdom=args.kingdom, genetic_code=args.genetic_code,
            min_cds_len=args.min_cds_len,
            ablate=args.ablate,
            coding_threshold=args.coding_threshold,
            boundary_threshold=args.boundary_threshold,
            domain_threshold=args.domain_threshold,
            conf_tau=args.conf_tau, anc_tau=args.anc_tau,
            min_intron=args.min_intron, max_intron=args.max_intron,
            verbose=not args.quiet)
    elif args.cmd == "profile":
        from .profile import run_profile
        run_profile(
            model_path=args.model, inputs=args.inputs,
            out_prefix=args.out, mode=args.mode, reads2=args.reads2,
            window=args.window, stride=args.stride,
            batch_size=args.batch_size, device=args.device,
            flat_prior_tau=args.flat_prior_tau,
            conf_tau=args.conf_tau, anc_tau=args.anc_tau,
            cluster_k=args.cluster_k,
            cluster_min_shared=args.cluster_min_shared,
            cluster_max_kmer_freq=args.cluster_max_kmer_freq,
            cluster_max_reads=args.cluster_max_reads,
            min_overlap=args.min_overlap,
            pack_below=args.pack_below,
            pool_reads=args.pool_reads,
            domain_threshold=args.domain_threshold,
            verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
