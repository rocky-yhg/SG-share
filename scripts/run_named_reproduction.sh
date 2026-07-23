#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed}"
OUTPUT_ROOT="${2:-results/named_reproduction}"
SEEDS="${SEEDS:-}"
CES_SEEDS="${CES_SEEDS:-${SEEDS:-42}}"
GLOBEM_SEEDS="${GLOBEM_SEEDS:-${SEEDS:-42}}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-8}"

CES_DATA="${DATA_ROOT}/ces_historical_usernorm.csv"
GLOBEM_DATA="${DATA_ROOT}/globem_full.csv"

for path in "${CES_DATA}" "${GLOBEM_DATA}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Missing processed stream: %s\n' "${path}" >&2
    exit 2
  fi
done

python process/run_online_sota_suite.py \
  --datasets ces \
  --methods oli2ds,hbp,koil,olifl,olfl \
  --scopes global,per_user \
  --seeds "${CES_SEEDS}" \
  --output "${OUTPUT_ROOT}/online_sota_raw05/ces" \
  --ces-data "${CES_DATA}" \
  --no-scaling --raw-fixed-05

python process/run_online_sota_suite.py \
  --datasets globem \
  --methods oli2ds,hbp,koil,olifl,olfl \
  --scopes global,per_user \
  --seeds "${GLOBEM_SEEDS}" \
  --output "${OUTPUT_ROOT}/online_sota_raw05/globem" \
  --globem-data "${GLOBEM_DATA}" \
  --no-scaling --raw-fixed-05

for dataset in ces globem; do
  data_var="${DATA_ROOT}/${dataset}.csv"
  if [[ "${dataset}" == "ces" ]]; then
    dataset_seeds="${CES_SEEDS}"
  else
    dataset_seeds="${GLOBEM_SEEDS}"
  fi
  for preset in classification_tuned cold_safe; do
    python process/run_sg_share_named_preset.py \
      --dataset "${dataset}" \
      --preset "${preset}" \
      --data "${data_var}" \
      --output "${OUTPUT_ROOT}/sgshare" \
      --seeds "${dataset_seeds}" \
      --device "${DEVICE}" \
      --torch-threads "${THREADS}"
  done
done
