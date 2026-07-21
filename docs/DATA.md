# Dataset Preparation

The package never fabricates paper metrics when source data are unavailable.
Place licensed raw files under `data/raw/`, then convert them to the canonical
stream schema.

## CES

The official CES release is available under `data/raw/ces/official_v5/`. The
historical stage-1 stream can be rebuilt with:

```bash
python scripts/prepare_ces_historical.py
```

This produces `data/processed/ces_historical_usernorm.csv` with 35,289 labeled
events, 218 users, 10,053 positive labels, and 37 model inputs. The recovered
protocol uses 36 stage-1 sensing/device features plus raw epoch `day`, stable
`day,uid` ordering, zero-filled missing values, and
per-user population z-scores for sensing features. `uid` and `day` are excluded
from normalization. The CSV keeps the official hash UID; sorting those hashes
is order-equivalent to the historical sorted factorization into integer UIDs.

Run this stored-value stream with `--no-scaling`. Using strict online scaling
or legacy global scaling changes the historical representation.

Historical prepared path:

```text
data_streams/ces_phq4/ces_phq4_stage1_ordered_bin_usernorm.arff
```

## GLOBEM

Use the official GLOBEM/PhysioNet v1.1 release. The recovered historical
pipeline uses weekly depression labels as the base and left-joins seven raw
feature tables: bluetooth, call, location, screen, sleep, steps, and wifi.
RAPIDS features are excluded. It drops discrete-rank, mixed-type, and
greater-than-90%-missing columns, selects 128 behavior features by full-stream
population variance, retains `is_ios` and `is_android`, median-imputes, and
applies a full-stream population z-score. The expected stream contains 8,225
rows, 704 participant-year users, and 130 inputs. Use `--no-scaling` when
loading the resulting ARFF.

Historical prepared path:

```text
data_streams/globem_weekly/stage1/globem_weekly_v5_top128.arff
```

## StudentLife

Build daily sensing features aligned to PHQ-9 observations and binarize using
the study protocol. This dataset was supplemental, not part of the two-dataset
main claim.

Historical prepared path:

```text
data_streams/studentlife/stage1/studentlife_daily_phq9_pre.arff
```

## Leakage policy

The default canonical loader sorts chronologically and scales online: statistics
used at time `t` are updated only after the row at `t` is transformed. This is
stricter than the historical helpers. The reconstructed historical CES and
GLOBEM files already store normalized values, so exact legacy comparison uses
`--no-scaling`. The separate `--legacy-global-scaling` option is only for an
unscaled canonical table that still needs the old full-stream transform; it
must not be applied to the reconstructed normalized files.

The recovered CES helper is more specific: it performs full-user offline
normalization while excluding `day`. This historical behavior can use future
observations from the same user when estimating mean and variance. It is
retained only for exact legacy comparison and must be disclosed separately
from the stricter online-scaling protocol.

The GLOBEM top-128 builder likewise uses all stream rows for median imputation,
variance ranking, and population z-scoring. This is retained for exact legacy
comparison and is not a leakage-free online feature-selection protocol.
