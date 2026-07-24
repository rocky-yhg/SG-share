# Expected Result Boundaries

This directory separates the selected paper display values from older
reconstruction targets and baseline provenance.

## Selected paper snapshot

The following files transcribe the bundled paper artifact at commit
`297f60f3b87fba18ac7ae4f48a83dd79c17abc68`:

- `manuscript_main_results.csv`;
- `manuscript_cold_start_results.csv`;
- `manuscript_ablation_results.csv`.

The selected PDF reports CES OCAP F1 as `0.7060` in the main table and
component ablation, while the gradient-grouping point is `0.7051`. These files
preserve the displayed values instead of forcing them to match.

## Historical compatibility files

`paper_results.csv` is the older reconstruction target consumed by
`python -m sgshare.reproduce`; it is not a transcription of the bundled PDF.
`historical_sota_results.csv` records baseline provenance. Neither file should
be mixed with the three versioned manuscript CSVs.

Reference values are comparison targets, not evidence that a new run
reproduced them. Use the resolved configuration, event log, data hash, and
metric contract from the same run when reporting new results.
