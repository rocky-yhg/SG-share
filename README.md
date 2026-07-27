# OCAP

This anonymous repository contains the implementation of OCAP. It includes
the OCAP framework, dataset preparation scripts, and the runner for the main
experiments.

Participant-level data are not included. Reviewers must download CES and
GLOBEM from their official sources and prepare them locally as described
below. The repository intentionally excludes baseline implementations, paper
sources, reported-result files, and author-identifying infrastructure.

## Repository structure

```text
configs/                 Dataset-specific OCAP configurations
data/raw/                Official downloads (not tracked by Git)
data/processed/          Prepared streams (not tracked by Git)
process/run_ocap.py       Main OCAP experiment runner
scripts/                 CES and GLOBEM preparation utilities
ocap/                    OCAP implementation
tests/                   Tests for the OCAP implementation
```

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Dataset preparation

### CES

Download the **College Experience Dataset (CES)** from its
[official Kaggle page](https://www.kaggle.com/datasets/subigyanepal/college-experience-dataset).
After extracting the archive, place the release files so that the following
two paths exist:

```text
data/raw/ces/official_v5/EMA/general_ema.csv
data/raw/ces/official_v5/Sensing/sensing.csv
```

Prepare and validate the chronological CES stream:

```bash
python scripts/prepare_ces_historical.py \
  --source-root data/raw/ces/official_v5 \
  --output-csv data/processed/ces_historical_usernorm.csv

python scripts/validate_stream.py \
  --dataset ces \
  --input data/processed/ces_historical_usernorm.csv
```

The preparation script joins the daily PHQ-4 assessments with sensing
features, defines the risk label as `PHQ-4 >= 4`, performs the preprocessing
used in the experiments, and orders the resulting data by collection time.

### GLOBEM

Download **GLOBEM v1.1** from its
[official PhysioNet page](https://physionet.org/content/globem/1.1/).
GLOBEM is credentialed health data. Reviewers must sign in to PhysioNet,
complete its required training, and accept the dataset data-use agreement.
Do not redistribute the downloaded or processed participant-level data.

Place the four extracted cohort directories as follows:

```text
data/raw/globem/physionet-1.1/INS-W_1/
data/raw/globem/physionet-1.1/INS-W_2/
data/raw/globem/physionet-1.1/INS-W_3/
data/raw/globem/physionet-1.1/INS-W_4/
```

Prepare and validate the chronological GLOBEM stream:

```bash
python scripts/align_globem_official.py \
  --root data/raw/globem/physionet-1.1 \
  --out-dir data/processed/globem_stage1

python scripts/prepare_globem_weekly.py \
  --input-csv data/processed/globem_stage1/globem_weekly_clean_all_cohorts.csv.gz \
  --output-arff data/processed/globem_top128.arff \
  --top-k-features 128

python scripts/convert_legacy_arff.py \
  --input-arff data/processed/globem_top128.arff \
  --output-csv data/processed/globem_full.csv

python scripts/validate_stream.py \
  --dataset globem \
  --input data/processed/globem_full.csv
```

The pipeline joins weekly depression labels with Bluetooth, call, location,
screen, sleep, step, and Wi-Fi features. It then selects 128 behavioral
features, retains the two platform indicators, and preserves chronological
order.

The final files used by the experiment runners are:

```text
data/processed/ces_historical_usernorm.csv
data/processed/globem_full.csv
```

Both `data/raw/` and `data/processed/` are ignored by Git.

## Running OCAP

Each data triplet is evaluated in predict-then-update order. Every run uses
the fixed main-experiment configuration and saves its resolved configuration,
prediction history, metrics, and group assignments under the selected output
directory.

```bash
python process/run_ocap.py \
  --dataset ces \
  --data data/processed/ces_historical_usernorm.csv \
  --output results/ocap_main \
  --seeds 42 \
  --device cuda \
  --torch-threads 8

python process/run_ocap.py \
  --dataset globem \
  --data data/processed/globem_full.csv \
  --output results/ocap_main \
  --seeds 42 \
  --device cuda \
  --torch-threads 8
```

All model and optimization hyperparameters are fixed by
`configs/ces.yaml`, `configs/globem.yaml`, and `ocap/settings.py`. The
commands above require no manual hyperparameter selection.

## Tests

The tests use synthetic or temporary inputs and do not require either private
dataset:

```bash
python -m unittest discover -s tests -v
```

## Data policy

The `.gitignore` rules prevent raw data, processed participant records,
predictions, and checkpoints from being committed. Before publishing a fork,
verify that `git status` contains no files under `data/raw/`,
`data/processed/`, or `results/`.
