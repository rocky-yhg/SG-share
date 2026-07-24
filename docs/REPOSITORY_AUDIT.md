# Repository Content Audit

This audit records why potentially redundant files are retained or excluded.

## Retained intentionally

- `assets/framework.pptx` is the editable source for the framework figure. It
  is not a runtime dependency, but its Git history shows that it was added
  deliberately for figure editing.
- `paper/figures/*.pdf` are the publication vector assets; matching PNG files
  are retained for GitHub rendering. The pairs are intentional, not duplicate
  experiment outputs.
- `expected/paper_results.csv` is an older compatibility target consumed by
  `sgshare/reproduce.py`. It is separated from the versioned
  `manuscript_*_results.csv` files in `expected/README.md`.
- `_v100.pbs` scripts preserve an optional PBS launch environment. They are
  classified separately in `scripts/README.md` and are not presented as the
  SeetaCloud validation path.
- Historical analysis runners under `process/` are retained when they provide
  provenance or are imported by maintained publication runners.

## Excluded from publication

- raw and processed participant-level data;
- `data/processed.zip` and `data/releases/`;
- per-event experiment outputs, logs, checkpoints, and temporary files;
- local `AGENTS.md`, environment files, credentials, private keys, and
  AppleDouble metadata;
- downloaded third-party repositories.

The root `.gitignore` enforces these boundaries. The published repository
contains aggregate paper references and non-sensitive audit metadata only.

## Execution completeness

Commit `c81444c` passed 49 unit tests in the retained SeetaCloud Python 3.12
environment. The paper bundle and documentation added after that commit do not
modify model, metric, dataset, preset, or runner logic. `pyproject.toml` and
`requirements.txt` include the ML baselines, Parquet engine, and Markdown-table
dependency used by the advertised code paths.
