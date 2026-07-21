# Reproduction Status

## Completed

- Self-contained package and dependency manifest.
- Official CES release restored and matched to the historical labeled stream:
  35,289 events, 218 users, and 10,053 positive labels.
- Historical CES stage-1 feature construction, stable event ordering, UID
  factorization, and per-user normalization recovered from Codex execution
  logs. The resulting 37-feature stream matches the logged first rows and UID
  order exactly.
- Canonical chronological stream converter with strict online scaling.
- Exact historical GLOBEM seven-modality weekly alignment and top-128
  variance-selection/global-standardization recipe recovered from execution
  logs.
- Historical ARFF compatibility converter preserving event order.
- TinyTFT-style shared backbone and rank-4 LoRA adapters.
- Temporary-user to reusable-group adapter transition.
- Route-gradient EMA signatures and cosine agglomeration.
- `full_final` refinements: scalar user calibration, verified local split, and
  mature-user refinement.
- Framework, grouping-signal, and component variants.
- Ported PDFK, Budgeted, and Supermask online baselines.
- Full-stream, post-warmup, first-K user-macro, and pooled metrics.
- Unit tests and a successful ten-method synthetic end-to-end matrix.

## Numerical reproduction status

The historical CES input has been recovered as
`data/processed/ces_historical_usernorm.csv`. A run must use `--no-scaling` so
the stored historical values are not transformed a second time.

The package's current `sgshare/` learner remains a clean-room approximation,
not the historical implementation. A full run on an earlier 649-feature stream
produced F1 0.5600 and post-warmup F1 0.5622, far below the historical 0.6910
and 0.6961. A second run using the recovered historical input was terminated
after code audit established that it still used the non-equivalent learner and
an impractical cubic agglomeration path. Its partial execution is not a result.

Historical logs identify the missing implementation as
`AsyncBackboneLoRAGradPartitionV5Classifier` with a two-layer, four-head
TinyTFT configuration, LoRA rank 4/alpha 8, backbone learning rate 2e-4, and
LoRA learning rate 7e-3. The former classifier, runner, and dependency tree
were stored on the now-unmounted Lenovo volume and are not present locally or
in a public repository. Numerical paper reproduction is therefore blocked on
recovering or faithfully rebuilding that implementation, not on CES data.

The complete credentialed GLOBEM release is also unavailable locally. The
recovered pipeline passes all checks on the four official sample cohorts:
120 rows, 40 participant-year users, positive rate 0.425, and 130 final model
inputs. This verifies the processing contract, not the protected full-cohort
paper result. The full target remains 8,225 rows, 704 users, positive rate
0.4621276596, and 130 inputs.

The machine-readable audit is
`data/audit/historical_data_alignment_audit.json`; its companion report is
`data/audit/HISTORICAL_DATA_ALIGNMENT_AUDIT.md`. CES passes the complete
historical-stream audit, while GLOBEM is explicitly marked as an official
sample-contract audit.

## Acceptance sequence

1. Restore/rebuild the historical learner and runner.
2. Pass static, unit, and short-stream prequential regression tests.
3. Run CES `full_final` seed 42 on `ces_historical_usernorm.csv` with
   `--no-scaling`; compare event counts, positive rate, configuration, and
   metrics.
4. Stop if F1 remains materially below the historical target; do not spend
   compute on SOTA or ablations until the main path is aligned.
5. Run framework/grouping ablations and the three ported SOTA methods on the
   identical stream.
6. Run seeds 42, 43, and 44 and report mean/std rather than selecting a maximum.
7. Accept numerical reproduction only after `paper_comparison.csv` and event
   traces are reviewed.
