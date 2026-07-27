# Manuscript Result Audit, 2026-07-27

This directory organizes the retained SG-Share results in the same order as
the manuscript experiments. It distinguishes three evidence levels:

- `complete`: aggregate outputs, resolved configuration, and a retained private
  event artifact with a published SHA-256;
- `aggregate_only`: the displayed aggregate is retained, but its event-level
  prediction file is unavailable;
- `manuscript_only`: a diagnostic value is transcribed from the manuscript and
  its original analysis program is unavailable;
- `partial_alignment`: at least one displayed value is recoverable, but the
  complete displayed row is not produced by the identified event source.

The authoritative map is [audit/coverage.csv](audit/coverage.csv). Known
cross-file discrepancies are listed explicitly in
[audit/alignment_issues.csv](audit/alignment_issues.csv); an item is not marked
aligned merely because one displayed metric can be recovered. Server
availability at audit time is recorded in
[audit/server_inventory.csv](audit/server_inventory.csv).

## Directory order

1. `RQ1_dataset_analysis`: asynchronous participation and evidence
   accumulation.
2. `RQ2_main_results`: Table 2 classification and the online/offline
   efficiency comparison.
3. `RQ3_low_evidence`: Table 3 selected cold-safe SG-Share runs.
4. `RQ4_ablation_study`: both the class-macro tables in the current LaTeX
   working tree and the later positive-class reruns.
5. `RQ5_hyperparameter_sensitivity`: route-gradient EMA, minimum-group,
   mature-evidence, and movement-margin sweeps.
6. `diagnostic_evidence`: Figure 1 values and Section 3 relation/transfer
   diagnostics that motivate the method.

## Selected SG-Share runs

| Purpose | Dataset | Private event source | Displayed result |
|---|---|---|---:|
| Classification | CES | `mature_parameter_sensitivity_seed42_20260723/ces/lambda20_d010/events.csv` | F1(+) `0.7060458652` |
| Classification | GLOBEM | `no_split_focal_alpha_k_cross_seed42_20260723/globem/a010_k5/events.csv` | F1(+) `0.7358834244` |
| First-K | CES | `no_split_focal_alpha_k_cross_seed42_20260723/ces/a005_k3/events.csv` | F1@5 `0.2713225075` |
| First-K | GLOBEM | `globem_cold_alpha_refine_seed42_20260723/alpha_012/events.csv` | F1@5 `0.4379660125` |

The event files remain on the controlled SeetaCloud volume because they contain
longitudinal participant-level mental-health labels. Their paths, byte sizes,
and SHA-256 values are listed in
[audit/private_artifact_manifest.tsv](audit/private_artifact_manifest.tsv).
No password, raw sensor feature, timestamp, or participant-level prediction
sequence is published here.

## Recomputing additional metrics

Run the repository script in the controlled environment:

```bash
python process/recompute_event_metrics.py \
  --events /root/autodl-tmp/sgshare-runtime/results/<run>/events.csv \
  --output metrics_extended.json \
  --ks 5,10,20,50
```

The script reports positive-class, class-macro, discrimination, calibration,
and first-K user-macro/pooled metrics from the same prediction file. It never
changes the stored prediction threshold.

## Audit boundaries

- The current CES main-result row is not a single-run row: its displayed
  `F1=0.7060` matches the selected `lambda20_d010` run, while the displayed
  Recall, Accuracy, and AUC do not. The selected run yields Recall `0.7580`,
  Accuracy `0.8202`, and AUC `0.8743`. See `audit/alignment_issues.csv`.
- The current LaTeX working tree and the bundled paper snapshot do not use the
  same ablation display. Both result families are retained under
  `RQ4_ablation_study`; do not combine their columns.
- CES classification and cold-start use different selected configurations.
  GLOBEM classification and cold-start also use different configurations.
- Offline `AUC` values were reconstructed from hard class labels and equal
  balanced accuracy. They are not probability-based ROC AUC.
- Transfer AUC `0.702` is a linear-surrogate diagnostic, not a direct
  TinyTFT/LoRA result.

`remote_metadata_snapshot/sgshare_result_metadata_20260727.tar.gz` is the
read-only metadata snapshot downloaded from the more complete `westb-32117`
instance. Its SHA-256 is recorded in `MANIFEST.sha256`.
