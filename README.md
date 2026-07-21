# SG-Share Reproduction Package

This directory is a clean-room, auditable reconstruction of the experiments for
SG-Share. The former experiment repository and raw datasets were not present on
the machine when this package was created. The reconstruction therefore keeps a
strict provenance boundary:

- **Recovered**: settings and results visible in the manuscript and historical
  Codex execution logs.
- **Reconstructed**: implementation implied by the documented equations and
  execution traces, but not recovered byte-for-byte.
- **Paper-faithful cross-domain baseline**: capacity-controlled adaptations of
  PDFK, Budgeted Online Adaptation, and Supermask audited against the papers and,
  where available, official repositories. They are not claimed to reproduce the
  papers' original image-benchmark numbers.

See docs/PROVENANCE.md and docs/SOTA_FAITHFULNESS_AUDIT.md before interpreting
reproduced numbers.

## Repository boundary

This public repository contains SG-Share source code, configurations, tests,
data-preparation programs, provenance records, and lightweight expected-value
tables. It intentionally does **not** contain:

- participant-level CES or GLOBEM raw/processed data;
- generated event logs, checkpoints, or full experiment result directories;
- copies of third-party repositories used to audit baseline implementations.

Dataset files must be obtained from the official providers and placed under
the ignored `data/raw/` tree. See `docs/OFFICIAL_DATA_SOURCES.md` and
`docs/DATA.md` for the exact directory contracts and preparation commands.
Third-party source audits can be restored at pinned commits with:

```bash
bash scripts/fetch_third_party.sh
```

See `docs/THIRD_PARTY.md` for the upstream repositories, commit identifiers,
licensing boundary, and the distinction between source-grounded ports and
mechanism reproductions.

## Reproduction target

The named final configuration is:

```text
p0_bias_best_cflsplit_a_m005_u10_e50
```

It contains clean route-gradient agglomerative grouping, warmup-v2, rank-4
group LoRA, a scalar per-user calibration bias, holdout-verified local splitting
with margin 0.005/minimum 10 users/50 events, and mature-user loss refinement
after 20 observations with margin 0.001.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Validate the package without private datasets.
python -m unittest discover -s tests -v
python -m sgshare.experiment --dataset synthetic --method full_final \
  --output results/smoke

# Convert a licensed dataset to the canonical stream format.
python -m sgshare.data \
  --input data/raw/ces.csv --output data/processed/ces.csv \
  --user-column user_id --time-column timestamp --label-column label

# Run the registered comparison suite.
python -m sgshare.reproduce --datasets ces,globem \
  --data-root data/processed --output-root results/reproduction --no-scaling
```

## Canonical data format

Each processed stream is a CSV or Parquet table with:

```text
user_id,timestamp,label,<numeric feature columns...>
```

Rows are sorted by timestamp. Labels must be binary (`0`/`1`). Dataset-specific
raw extraction is intentionally separate from model code because CES access is
licensed and GLOBEM may require credentialed PhysioNet access. See
`docs/DATA.md`.

## Expected outputs

Each run writes:

- `metrics.json`: full-stream and post-warmup metrics;
- `cold_start.csv`: pooled and user-macro first-K metrics;
- `events.csv`: prequential predictions, emitted before each update;
- `groups.csv`: regroup diagnostics;
- `config.yaml`: the resolved configuration.

The suite additionally writes `summary.csv` and `paper_comparison.csv`, where
reproduced values are compared with the manuscript values in
`expected/paper_results.csv`.
