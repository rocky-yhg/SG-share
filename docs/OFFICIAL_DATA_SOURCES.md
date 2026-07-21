# Official Dataset Sources

This reproduction uses only sources linked by the corresponding dataset papers.
No third-party data mirror is required.

Participant-level files and generated canonical streams are deliberately
excluded from the public Git repository. The model-facing interface is always
a local file under `data/processed/`; acquisition and conversion remain the
responsibility of an authorized user.

## CES

Paper:

- Subigya Nepal et al., *Capturing the College Experience: A Four-Year
  Mobile Sensing Study of Mental Health, Resilience, and Behavior of College
  Students during the Pandemic*, IMWUT 2024.
- DOI: https://doi.org/10.1145/3643501

Official data release cited by the paper:

- https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset
- Release used here: Kaggle dataset version 5, updated 2025-04-15.
- License: CC BY-NC-SA 4.0 plus the release's privacy terms.
- Archive SHA-256:
  `5acb29a6374cdc47f80d9b6c07bf0ef2d6d98562ed88eea004458c8afdcd7805`.

Download and extract the files needed for SG-Share:

```bash
mkdir -p data/raw/ces/official_v5
curl -L --fail --retry 3 \
  -o data/raw/ces/ces-kaggle-v5.zip \
  'https://www.kaggle.com/api/v1/datasets/download/subigyanepal/college-experience-dataset?datasetVersionNumber=5'
unzip -q data/raw/ces/ces-kaggle-v5.zip \
  'Demographics/*' 'EMA/*' 'Sensing/*' 'Readme.md' \
  -d data/raw/ces/official_v5
```

Prepare the canonical stream:

```bash
python scripts/prepare_ces_phq4.py
python scripts/validate_stream.py --dataset ces --input data/processed/ces.csv
```

For exact historical comparison, use the recovered 37-input preparation path:

```bash
python scripts/prepare_ces_historical.py \
  --source-root data/raw/ces/official_v5 \
  --output-csv data/processed/ces_historical_usernorm.csv
python scripts/validate_stream.py \
  --dataset ces --input data/processed/ces_historical_usernorm.csv
```

The official v5 release gives 35,348 non-missing PHQ-4 observations. Joining
them to daily sensing on `(uid, day)` gives 35,289 observations from 218 users,
which exactly matches the historical SG-Share CES stream size. The recovered
binary target is `phq4_score >= 4`.

## GLOBEM

Paper:

- Xuhai Xu et al., *GLOBEM Dataset: Multi-Year Datasets for Longitudinal
  Human Behavior Modeling Generalization*, NeurIPS Datasets and Benchmarks 2022.

Official data release:

- PhysioNet v1.1: https://physionet.org/content/globem/1.1/
- Version DOI: https://doi.org/10.13026/r9s1-s711
- Official processing and benchmark code: https://github.com/UW-EXP/GLOBEM

The data files are not open anonymous downloads. PhysioNet requires all of:

1. a credentialed PhysioNet account;
2. completed CITI Data or Specimens Only Research training;
3. acceptance of the GLOBEM Data Use Agreement.

After authorization, download v1.1 directly from PhysioNet. Do not commit or
redistribute the protected files:

```bash
wget -r -N -c -np --user YOUR_PHYSIONET_USERNAME --ask-password \
  https://physionet.org/files/globem/1.1/
```

The expected full release contains `INS-W_1` through `INS-W_4`, each with:

```text
SurveyData/dep_weekly.csv
FeatureData/rapids.csv
ParticipantsInfoData/platform.csv
```

Place those cohort directories under
`data/raw/globem/physionet-1.1/`, then use the exact conversion interface in
`docs/GLOBEM_PIPELINE.md`. The final model-facing file is
`data/processed/globem.csv` and must be loaded with `--no-scaling` for the
historical protocol.

The official repository can be restored under `vendor/GLOBEM` with
`bash scripts/fetch_third_party.sh GLOBEM` for data-format and preprocessing
audit. It includes synthetic/sample raw folders, not the protected full
participant data. The local `vendor/` tree is ignored by Git.

## Provenance Boundary

CES source alignment is now directly reproducible from the public official
release. GLOBEM source alignment and code are reproducible, but full numerical
experiments remain blocked until an authorized user downloads the credentialed
PhysioNet files. Authentication credentials and protected files must never be
stored in this repository.
