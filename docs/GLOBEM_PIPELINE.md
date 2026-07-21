# GLOBEM Official Pipeline

After obtaining authorized GLOBEM v1.1 files from PhysioNet, place `INS-W_1`
through `INS-W_4` under `data/raw/globem/physionet-1.1`, then run the
recovered historical two-stage pipeline:

```bash
python scripts/align_globem_official.py \
  --root data/raw/globem/physionet-1.1 \
  --out-dir data/processed/globem_stage1
python scripts/prepare_globem_weekly.py \
  --input-csv data/processed/globem_stage1/globem_weekly_clean_all_cohorts.csv.gz \
  --output-arff data/processed/globem.arff --top-k-features 128
python scripts/convert_legacy_arff.py \
  --input-arff data/processed/globem.arff \
  --output-csv data/processed/globem.csv
python scripts/validate_stream.py --dataset globem --input data/processed/globem.csv
```

The weekly labels are the row-preserving base. Seven modality tables
(`bluetooth`, `call`, `location`, `screen`, `sleep`, `steps`, and
`wifi`) are filtered and left-joined on `(pid,date)`; `rapids.csv` is not
used. Columns matching `*_dis:*`, mixed-type columns, and columns with more
than 90% missingness are removed. Participant-year identity is
`cohort::pid`.

The second stage median-imputes behavior features, selects the 128 highest
full-stream population-variance features, retains `is_ios` and `is_android`,
and applies the historical full-stream population z-score (`ddof=0`). Run
experiments with `--no-scaling`; any loader scaling would transform the
historical representation a second time.

The protected full release should produce 8,225 rows, 704 participant-year
users, positive rate 0.4621276596, and 130 model inputs (128 behavior + 2
platform). The bundled four-cohort sample validates the file contract only.
