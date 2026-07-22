# Classification-tuned and cold-safe presets

The named presets reproduce the parameter points selected by the recovered
parameter-search and confirmation runs. They change only the route-gradient EMA
alpha, the observation eligibility threshold (`lambda`), `k_min`, and whether
the initial eligibility threshold is strict. All other settings come from the
dataset YAML and the `full_final` method.

| Dataset | Preset | EMA alpha | EMA decay | Lambda | `k_min` | Initial eligibility |
|---|---|---:|---:|---:|---:|---|
| CES | `classification_tuned` | 0.20 | 0.80 | 20 | 4 | legacy: minimum 5, fallback 1 |
| CES | `cold_safe` | 0.05 | 0.95 | 20 | 4 | legacy: minimum 5, fallback 1 |
| GLOBEM | `classification_tuned` | 0.20 | 0.80 | 2 | 4 | legacy: minimum 5, fallback 1 |
| GLOBEM | `cold_safe` | 0.20 | 0.80 | 2 | 4 | strict: minimum 2, fallback 2 |

`lambda` is the regular group-sharing eligibility threshold. Under legacy
initial eligibility, the warmup boundary may use the dataset YAML's smaller
minimum and fallback. Strict eligibility applies the same `lambda` to regular
and initial admission.

Run one preset with:

```bash
python process/run_sg_share_named_preset.py \
  --dataset globem \
  --preset classification_tuned \
  --data data/processed/globem.csv \
  --output results/globem_classification_tuned \
  --seeds 42 --device cpu

python process/run_sg_share_named_preset.py \
  --dataset globem \
  --preset cold_safe \
  --data data/processed/globem.csv \
  --output results/globem_cold_safe \
  --seeds 42 --device cpu
```

Each run writes `preset.json`, the complete resolved `config.yaml`, and the
input stream SHA-256 in `manifest.json`; these artifacts identify the exact
configuration and immutable processed stream used for the run.

To reproduce all named SG-Share presets and the existing online SOTA ports in
both global and per-user scopes, use:

```bash
bash scripts/run_named_reproduction.sh data/processed results/named_reproduction
```

The default seed protocol is CES `42` and GLOBEM `42,43,44`, matching the
stored comparison artifacts. Override these independently with `CES_SEEDS` and
`GLOBEM_SEEDS`, or set `SEEDS` to force one shared seed list.

The script reads `data/processed/ces.csv` and
`data/processed/globem.csv`; it never creates, copies, replaces, or commits
those files. SOTA predictions use raw probability, fixed threshold `0.5`, no
probability smoothing, and no threshold search. CES SOTA runs use the shared
causal online standardizer from the stored comparison protocol; GLOBEM SOTA
runs consume the already normalized stored features. The implementations retain
the fidelity boundaries documented in
[Online SOTA reproduction](ONLINE_SOTA_REPRODUCTION.md): OLI2DS, HBP/ODL, and
KOIL are source-grounded ports; OBAL is an independent mechanism
reproduction; OLIFL and OLFL are compatibility analyses rather than official
numerical reproductions.

## Difference from the previous GitHub interface

Before these named presets, `process/run_table3_experiments.py` embedded only
the classification-tuned points in `ECAP_POINTS`. The cold-safe points existed
only inside dataset-specific parameter-confirmation scripts. There was no
single command that selected either version by name, and users had to infer the
strict-initial-eligibility difference from `SearchPoint` arguments.

The named runner does not change the SG-Share algorithm or historical target.
It exposes the already tested points through one registry and records the exact
resolved configuration. It intentionally uses the current GitHub `main`
default group-adapter mode, `standalone`. The deleted experimental
`agent/cross-entropy-warmup-backbone` branch used Q/V LoRA and default
`user_mean` group adapters; those experimental changes are not silently folded
into these historical presets.

Participant-level processed data remain ignored by Git. This change does not
add or modify any file under `data/processed/`.
