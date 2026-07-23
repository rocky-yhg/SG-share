# Manuscript-aligned SG-Share presets

The registry in `sgshare/presets.py` is the executable source of truth for the
SG-Share configurations used by the current `main.pdf`. The full-stream table
and the early-monitoring table use different selected points.

| Dataset | Preset | EMA alpha | EMA decay | Eligibility observations | `k_min` | Verified split | Mature reassignment |
|---|---|---:|---:|---:|---:|---:|---|
| CES | `classification_tuned` | 0.20 | 0.80 | 20 | 5 | off | `20 / 0.01` |
| CES | `cold_safe` | 0.05 | 0.95 | 20 | 3 | off | `20 / 0.01` |
| GLOBEM | `classification_tuned` | 0.10 | 0.90 | 2 | 5 | off | `20 / 0.01` |
| GLOBEM | `cold_safe` | 0.12 | 0.88 | 2 | 4 | off | `20 / 0.01` |

`EMA alpha` is the weight of the current route gradient. The implementation
stores its complement as `training.route_grad_ema_decay`.

`Eligibility observations` controls regular admission to grouping. Initial
warm-up grouping retains the dataset YAML settings of five observations with a
one-observation fallback. `k_min` is the lower target on the number of
agglomerative groups; it is not a minimum group size or an observation gate.

The mature-reassignment entry is
`mature_min_observations / mature_loss_margin`. Both datasets use
`20 / 0.01`. The historical `verified_split` refinement is disabled for all
four manuscript-aligned presets.

## Commands

Full-stream classification:

```bash
python process/run_sg_share_named_preset.py \
  --dataset ces --preset classification_tuned \
  --data data/processed/ces_historical_usernorm.csv \
  --output results/manuscript/main_classification \
  --seeds 42 --device cuda --torch-threads 8

python process/run_sg_share_named_preset.py \
  --dataset globem --preset classification_tuned \
  --data data/processed/globem_full.csv \
  --output results/manuscript/main_classification \
  --seeds 42 --device cuda --torch-threads 8
```

First-K early monitoring:

```bash
python process/run_sg_share_named_preset.py \
  --dataset ces --preset cold_safe \
  --data data/processed/ces_historical_usernorm.csv \
  --output results/manuscript/cold_start \
  --seeds 42 --device cuda --torch-threads 8

python process/run_sg_share_named_preset.py \
  --dataset globem --preset cold_safe \
  --data data/processed/globem_full.csv \
  --output results/manuscript/cold_start \
  --seeds 42 --device cuda --torch-threads 8
```

Every run writes:

- `preset.json`, containing the selected registry point;
- `config.yaml`, containing the complete resolved configuration;
- `manifest.json`, containing the input path and SHA-256;
- `events.csv`, `groups.csv`, `metrics.json`, and `cold_start.csv`.

The output configuration and data hash, rather than the preset name alone,
identify a numerical run.

## Operational boundary

These presets use `group_adapter_mode=standalone`. After a user is assigned to
a group, the group adapter receives the normal online adapter update; the
temporary personal adapter is not updated in parallel. The experimental
`user_mean` and `shadow_personal` modes are not part of the manuscript-aligned
configuration.

All prediction metrics use the configured SG-Share deployment path, including
the dataset YAML's smoothing and online threshold behavior. This is distinct
from the online-SOTA raw fixed-0.5 protocol.
