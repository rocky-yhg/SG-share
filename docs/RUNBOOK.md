# Reproduction Runbook

## 1. Restore streams

For an existing historical ARFF:

```bash
python scripts/convert_legacy_arff.py \
  --input-arff /path/to/ces_phq4_stage1_ordered_bin_usernorm.arff \
  --output-csv data/processed/ces.csv
```

For the historical GLOBEM cleaned weekly CSV:

```bash
python scripts/prepare_globem_weekly.py \
  --input-csv /path/to/globem_weekly_clean_all_cohorts.csv.gz \
  --output-arff data/processed/globem.arff --top-k-features 128
python scripts/convert_legacy_arff.py \
  --input-arff data/processed/globem.arff \
  --output-csv data/processed/globem.csv
```

StudentLife's historical prepared ARFF can be converted with the same legacy
converter. The recovered raw builder used PHQ-9 `pre`, threshold `10`, top-16
application packages, top-16 application classes, daily rows, removal of
constant/duplicate features, and all 46 users.

## 2. Validate cohort identity

```bash
python scripts/validate_stream.py --dataset ces --input data/processed/ces.csv
python scripts/validate_stream.py --dataset globem --input data/processed/globem.csv
```

Do not tune the model until unexpected row/user counts or positive rates are
resolved; otherwise a numerical difference is not attributable to the method.

## 3. Run final method first

```bash
PYTHONPATH=. python -m sgshare.experiment \
  --dataset ces --data data/processed/ces.csv --method full_final \
  --seed 42 --no-scaling --output results/ces_full_seed42
```

The reconstructed CES and GLOBEM historical streams already contain their
legacy normalization. Use `--no-scaling` for numerical alignment. Omitting it
selects the stricter transform-then-update online standardizer and therefore
defines a different experiment.

## 4. Run all comparisons

```bash
make reproduce
```

Review, in order:

1. `summary_mean_std.csv` for new mean/std results;
2. `paper_comparison.csv` for deltas against the manuscript;
3. per-run `events.csv` for prequential ordering;
4. per-run `cold_start.csv` for each K and both macro/pooled metrics;
5. per-run `groups.csv` for group count, churn, split, and refinement activity.

## 5. Numerical acceptance

The target historical full-final point estimates are:

| Dataset | Overall F1 | Recall | Specificity | Post F1 | Mean first-K macro F1 |
|---|---:|---:|---:|---:|---:|
| CES | 0.691 | 0.773 | 0.815 | 0.696 | 0.285 |
| GLOBEM | 0.727 | 0.862 | 0.561 | 0.730 | 0.404 |

Treat a higher value as a new result, not an automatic reproduction success.
First verify identical cohort, event count, positive rate, preprocessing mode,
seed policy, and method configuration.
