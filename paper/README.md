# Bundled Paper Snapshot

This directory contains the paper artifact and result displays bundled with the
code repository.

- Paper repository:
  `rocky-yhg/Online-Learning-for-Mental-Health-Monitoring-with-Passive-Sensing-Data`
- Snapshot commit: `297f60f3b87fba18ac7ae4f48a83dd79c17abc68`
- PDF Git blob: `82ed3f551adcefe6904f8050e0ad8f3ea384184f`
- PDF SHA256:
  `c7a06941b9db365b3e499977ea57ddbcc504aaf36b1ed574c149625eb6615599`

The user explicitly selected this tracked PDF artifact instead of rebuilding
the later paper source. `OCAP.pdf`, the files under `tables/`, and the files
under `figures/` were all exported from the same snapshot commit. The displayed
values and visual forms are consequently version-consistent.

## Contents

- [`OCAP.pdf`](OCAP.pdf): complete 10-page paper artifact.
- `tables/main_results.tex`: full-stream result table.
- `tables/cold_start_results.tex`: low-evidence result table.
- `tables/evidence_accumulation.tex`: evidence-accumulation figure wrapper.
- `tables/ablation_results.tex`: component/grouping ablation figure wrapper.
- `tables/hyperparameter_sensitivity.tex`: sensitivity figure wrapper.
- `figures/`: exact PDF and PNG figure assets referenced by the result section.
- `MANIFEST.sha256`: integrity hashes for every bundled artifact.

## Relationship to the executable code

The paper snapshot is the reporting reference. The repository retains the
historical `sgshare` import package and SG-Share/ECAP runner names, while the
bundled PDF uses the public method name OCAP.

The executable baseline is code commit
`c81444cc75c728543fcdfeb39f8f6b318b01613e`. It packages the named
`classification_tuned` and `cold_safe` presets reconstructed from the
SeetaCloud experiments run from `/root/autodl-tmp/sgshare-runtime`, principally
the `code-91f7d5b` experiment snapshot. The remote Python 3.12 environment
passed all 49 repository unit tests during that release audit. The audit did
not rerun every full CES/GLOBEM experiment.

The paper and executable release are related but not byte-identical:

- the selected deployment mode is `standalone`;
- the PDF retains an earlier CES grouping-signal value of `0.7051`, while the
  selected current-code center is `0.7060`;
- the PDF retains CES Accuracy/AUC from an earlier stored event file;
- the first-K table uses `cold_safe`, while full-stream and ablation results
  use `classification_tuned`.

See the root README and `docs/REPRODUCTION_STATUS.md` before replacing any
displayed paper value.
