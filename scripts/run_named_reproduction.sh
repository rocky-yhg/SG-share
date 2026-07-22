#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/processed}"
OUTPUT_ROOT="${2:-results/named_reproduction}"
SEEDS="${SEEDS:-42}"
DEVICE="${DEVICE:-cpu}"
THREADS="${THREADS:-1}"

CES_DATA="${DATA_ROOT}/ces.csv"
GLOBEM_DATA="${DATA_ROOT}/globem.csv"

for path in "${CES_DATA}" "${GLOBEM_DATA}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Missing processed stream: %s\n' "${path}" >&2
    exit 2
  fi
done

python process/run_online_sota_suite.py \
  --datasets ces,globem \
  --methods oli2ds,obal,hbp,koil,olifl,olfl \
  --scopes global,per_user \
  --seeds "${SEEDS}" \
  --output "${OUTPUT_ROOT}/online_sota_raw05" \
  --ces-data "${CES_DATA}" \
  --globem-data "${GLOBEM_DATA}" \
  --no-scaling --raw-fixed-05

for dataset in ces globem; do
  data_var="${DATA_ROOT}/${dataset}.csv"
  for preset in classification_tuned cold_safe; do
    python process/run_sg_share_named_preset.py \
      --dataset "${dataset}" \
      --preset "${preset}" \
      --data "${data_var}" \
      --output "${OUTPUT_ROOT}/sgshare" \
      --seeds "${SEEDS}" \
      --device "${DEVICE}" \
      --torch-threads "${THREADS}"
  done
done
