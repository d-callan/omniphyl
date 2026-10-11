# Remote runs: free-GPU training without local hardware

The pipeline is deliberately cheap to move: fetch is stdlib-only, the
model is ~11M params, and a pilot run fits inside any free-GPU
envelope. The env-agnostic runner is `remote/run_pilot.sh`
(clone -> smoke test -> fetch -> train -> eval -> tarball); everything
below wraps it.

## Option A: Kaggle Kernels (recommended)

Kaggle gives a real batch API with free GPU — T4, ~30h GPU/week — and
the official `kaggle` CLI pushes jobs from the terminal.

### Setup (once)

1. kaggle.com -> account settings -> **verify phone number** (required
   for GPU + internet on kernels).
2. Account -> API -> "Create New Token" -> `~/.kaggle/access_token`
   (`chmod 600`). Requires a current `kaggle` CLI; older versions used
   `~/.kaggle/kaggle.json` (username+key) — either works, and
   `KAGGLE_API_TOKEN` env var is a fallback.
3. `pip install -U kaggle`

### Push a run

```bash
cd kaggle/
# edit kernel-metadata.json: set "id" to "<you>/omniphyl-pilot"
kaggle kernels push
kaggle kernels status <you>/omniphyl-pilot
kaggle kernels output <you>/omniphyl-pilot -p ./results
```

`kernel.py` clones the GitHub repo onto the kernel VM and executes
`remote/run_pilot.sh` with `WORKDIR=/kaggle/working`. Outputs —
`omniphyl-pilot-<ts>.tar.gz` (labels.tsv, best.pt, eval JSON) — land in
`/kaggle/working` and come back via `kernels output`.

Parameters are env vars in `remote/run_pilot.sh` (documented in its
header); override in `kernel.py`'s `os.environ.setdefault` block, e.g.
`os.environ.setdefault("GROUPS", "protozoa")`, `("EPOCHS", "50")`.

Caveats: run cap is ~9-12h wall-clock; GPU kernels queue when the free
pool is busy; internet must be on (needed for the NCBI fetch — already
set in `kernel-metadata.json`).

## Option B: Google Colab

Works, but note GPU is now pay-as-you-go on most accounts (`colab pay`,
`colab usage` show credits). The official `google-colab-cli` still makes
it the best *interactive* remote when you have credits:

```bash
pip install google-colab-cli && colab auth
colab run --gpu T4 remote/run_pilot.sh   # ephemeral job -> tarball back
# or interactively:
colab new -s omni --gpu T4 && colab ssh -s omni
```

Browser path: File -> Open notebook -> GitHub -> `d-callan/omniphyl`,
then `!git clone ...` + `!bash remote/run_pilot.sh`. Mount Drive
(`from google.colab import drive; drive.mount('/content/drive')`) and
point `WORKDIR`/`OUTDIR` at it for persistence; the fetch caches every
download so a re-run resumes where it stopped.

## Option C: laptop CPU

Feasible for pilots by design. Keep the corpus small (~10-20k
fragments) or drop to `--k 6`; the 11M-param ResNet trains in hours on
a modern CPU. The smoke test already runs CPU-only in seconds.

## Tuning the run

All env vars on `remote/run_pilot.sh` (defaults in header):

| Var | Default | Effect |
|---|---|---|
| `GROUPS` | `bacteria viral` | RefSeq groups fetched |
| `NT_BUDGET` | 100000 | nt sampled per seqid -> fragments/genome |
| `FRAG_LEN` | 500 | fragment length bp |
| `BOUNDARY_FRAC` | 0.3 | mixed-content windows fraction |
| `VEUPATH` | 1 | VEuPathDB label tier |
| `K` / `WIDTH` | 7 / 64 | FCGR order / ResNet width |
| `EPOCHS` / `BATCH` | 30 / 256 | training schedule |
| `POS_CHANNELS` | 1 | PFCGR positional channels |
