"""Build the refiner panel (train + held-out decode genomes).

Standalone port of the panel-build section of
kaggle/tol-mini-corpus/kernel.py -- the corpus kernel never reached it
(12h limit hit mid-fetch), and the refiner bench expects
``refiner_panel/{train,heldout}/<name>.fna + .gff3 + panel.json``.

Each panel entry slices ONE seqid out of an NCBI assembly zip, caps its
length, keeps the CDS rows inside the cap, and writes fna + gff3.
``gcode`` entries (chrM code 2, tetrahymena code 6, mycoplasma code 4)
feed gate S2-C's alternate-code check.

    python build_refiner_panel.py --out refiner_panel [--dl /tmp/panel_dl]
"""

import argparse
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

NCBI_DL = ("https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
           "{acc}/download?include_annotation_type=GENOME_FASTA,"
           "GENOME_GFF&filename={acc}.zip")

# name -> (accession, seqid substring, clade, expected gcode, role,
#          max bases kept from seq start)
PANEL = [
    # ---- training species (single chrom/scaffold each) ----
    ("ecoli",      "GCF_000005845.2",  "NC_000913", "bacteria", None,
     "train", 5_000_000),
    ("hsalinarum", "GCF_000006805.1",  "NC_002607", "archaea", None,
     "train", 3_000_000),
    ("scerevisiae", "GCF_000146045.2", "NC_001136", "fungi", None,
     "train", 2_000_000),
    # GCF_000002765.6 = Pf3D7 v3 (v2 suppressed); chr14 is the
    # largest chr -> 'largest' keeps continuity w/ ablation panel
    ("pfalciparum", "GCF_000002765.6", "largest", "apicomplexa",
     None, "train", 4_000_000),
    ("athaliana",  "GCF_000001735.4",  "NC_003070", "plant", None,
     "train", 30_000_000),
    ("celegans",   "GCF_000002985.6",  "NC_003279", "invertebrate",
     None, "train", 16_000_000),
    ("dmelanogaster", "GCF_000001215.4", "NT_033779", "arthropod",
     None, "train", 24_000_000),
    # GCF_052040795.2 = GRCz13ab (GRCz11 suppressed)
    ("drerio",     "GCF_052040795.2",  "largest", "vertebrate", None,
     "train", 30_000_000),
    ("mmusculus",  "GCF_000001635.27", "NC_000085", "vertebrate", None,
     "train", 30_000_000),
    # vertebrate chrM in the training set (code 2)
    ("mmus_chrm",  "GCF_000001635.27", "NC_005089", "vertebrate", 2,
     "train", 0),
    # coverage additions: virus + non-Plasmodium protozoa
    ("phlambda",  "GCF_000840245.1",  "NC_001416", "viral", None,
     "train", 0),
    ("lmajor",    "GCF_000002725.2",  "largest", "protozoa", None,
     "train", 3_000_000),
    ("tgondii",   "GCF_000006565.2",  "largest", "protozoa", None,
     "train", 5_000_000),
    # ---- held-out decode species ----
    ("bsubtilis",  "GCF_000009045.1",  "NC_000964", "bacteria", None,
     "heldout", 5_000_000),
    # NC_009917 = PvP01_12 (chr12) -- same chr as the ablation panel
    ("pvivax",     "GCF_000002415.2",  "NC_009917", "apicomplexa",
     None, "heldout", 4_000_000),
    # GCF_034140825.1 = AGIS-1.0 Nipponbare T2T (IRGSP suppressed)
    ("osativa",    "GCF_034140825.1",  "largest", "plant", None,
     "heldout", 30_000_000),
    # GCF_943734735.2 = idAnoGambNW_F1_1 Sanger ToL (AgamP3 suppressed)
    ("agambiae",   "GCF_943734735.2",  "largest", "arthropod",
     None, "heldout", 24_000_000),
    # GCF_016699485.2 = GRCg7b VGP (GRCg6a suppressed)
    ("ggallus",    "GCF_016699485.2",  "largest", "vertebrate", None,
     "heldout", 30_000_000),
    ("hsapiens",   "GCF_000001405.40", "NC_000022", "vertebrate", None,
     "heldout", 10_000_000),
    ("hsap_chrm",  "GCF_000001405.40", "NC_012920", "vertebrate", 2,
     "heldout", 0),
    # alt-code taxa for gate S2-C
    ("tetrahymena", "GCF_000189635.1", "scaffold", "ciliate", 6,
     "heldout", 3_000_000),
    ("mycoplasma", "GCF_000027325.1",  "NC_000908", "bacteria", 4,
     "heldout", 1_000_000),
]


_FNA_SUF = (".fna", ".fa", ".fasta")
_GFF_SUF = (".gff", ".gff3")


def _fasta(p: Path) -> bool:
    n = p.name.lower()
    return (p.suffix.lower() in _FNA_SUF
            and "rna" not in n and "cds" not in n)


def fetch_asm(acc: str, dl: Path) -> tuple[Path, Path]:
    """-> (genomic.fna, genomic.gff) for an assembly accession."""
    zpath = dl / f"{acc}.zip"
    if not zpath.exists() and not (dl / acc).is_dir():
        print(f"  download {acc}", flush=True)
        req = urllib.request.Request(
            NCBI_DL.format(acc=acc),
            headers={"User-Agent": "omniphyl-refiner-panel"})
        with urllib.request.urlopen(req, timeout=600) as r:
            zpath.write_bytes(r.read())
    if zpath.exists() and not (dl / acc).is_dir():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(dl / acc)
    fna_hits = sorted(p for p in dl.glob(f"{acc}/**/*")
                      if p.is_file() and _fasta(p)
                      and "genomic" in p.name.lower()) or \
        sorted(p for p in dl.glob(f"{acc}/**/*")
               if p.is_file() and _fasta(p))
    gff_hits = sorted(p for p in dl.glob(f"{acc}/**/*")
                      if p.is_file() and p.suffix.lower() in _GFF_SUF)
    if not fna_hits or not gff_hits:
        listing = sorted(str(p) for p in (dl / acc).rglob("*")
                         if p.is_file())
        raise RuntimeError(f"{acc}: no fasta/gff inside zip; "
                           f"contents={listing[:25]}")
    return fna_hits[0], gff_hits[0]


def read_fna(path: Path) -> list:
    seqs, h, buf = [], None, []
    for line in open(path):
        if line.startswith(">"):
            if h is not None:
                seqs.append((h, "".join(buf)))
            h, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if h is not None:
        seqs.append((h, "".join(buf)))
    return seqs


def write_panel(name, acc, seqmatch, clade, gcode, role, cap,
                panel_dir: Path, dl: Path, manifest: dict):
    fna, gff = fetch_asm(acc, dl)
    target = None
    for h, s in read_fna(fna):
        sid = h.split()[0]
        if seqmatch in sid or seqmatch in h:
            target = (sid, s)
            break
    if target is None and seqmatch in ("scaffold", "largest"):
        # tetrahymena-style / NC-id-unknown assemblies: largest seq
        target = max(read_fna(fna), key=lambda x: len(x[1]))
    if target is None:
        print(f"  {name}: seqid *{seqmatch}* not in {acc}, skipped")
        return
    sid, seq = target
    if cap and len(seq) > cap:
        seq = seq[:cap]
        cap_end = cap
    else:
        cap_end = len(seq)
    rows = []
    for line in open(gff):
        if line.startswith("#"):
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 9 or c[0] != sid or c[2] != "CDS":
            continue
        s, e = int(c[3]), int(c[4])
        if s >= 1 and e <= cap_end:
            rows.append(line.rstrip("\n"))
    if not rows:
        print(f"  {name}: no CDS rows on {sid}, skipped")
        return
    d = panel_dir / role
    with open(d / f"{name}.fna", "w") as f:
        f.write(f">{sid}\n" + "\n".join(
            seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    with open(d / f"{name}.gff3", "w") as f:
        f.write("##gff-version 3\n")
        f.writelines(r + "\n" for r in rows)
    manifest[name] = {"clade": clade, "gcode": gcode, "acc": acc,
                      "seqid": sid, "len": len(seq),
                      "n_cds": len(rows), "role": role}
    print(f"  {name}: {sid} {len(seq):,} bp, {len(rows)} CDS -> {role}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                    help="refiner_panel dir (gets train/ heldout/ "
                         "panel.json)")
    ap.add_argument("--dl", type=Path, default=None,
                    help="scratch dir for NCBI zips (default "
                         "<out>/../_panel_dl)")
    args = ap.parse_args()
    panel_dir = args.out
    dl = args.dl or panel_dir.parent / "_panel_dl"
    (panel_dir / "train").mkdir(parents=True, exist_ok=True)
    (panel_dir / "heldout").mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for i, spec in enumerate(PANEL):
        try:
            write_panel(*spec, panel_dir, dl, manifest)
        except Exception as ex:
            print(f"  {spec[0]}: FAILED ({ex}) -- skipping", flush=True)
        acc = spec[1]
        if acc not in {s[1] for s in PANEL[i + 1:]}:
            # last panel entry for this accession: release zip + extract
            shutil.rmtree(dl / acc, ignore_errors=True)
            (dl / f"{acc}.zip").unlink(missing_ok=True)
    (panel_dir / "panel.json").write_text(json.dumps(manifest,
                                                   indent=1))
    print(f"\npanel manifest: {len(manifest)} genomes "
          f"({sum(1 for m in manifest.values() if m['role'] == 'train')}"
          f" train / "
          f"{sum(1 for m in manifest.values() if m['role'] == 'heldout')}"
          f" heldout)")


if __name__ == "__main__":
    main()
