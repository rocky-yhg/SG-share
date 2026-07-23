# Reproduction Status

## Executable coverage

The repository currently provides executable paths for:

- CES and GLOBEM processed-stream preparation and validation;
- SG-Share classification-tuned and cold-safe presets;
- prequential full-stream and first-K evaluation;
- offline ML/DL experiments;
- online HBP/ODL, KOIL, OLFL, OLI2DS, and OLIFL in global/per-user scopes;
- component and grouping-signal ablations;
- route-gradient EMA alpha and `k_min` sensitivity;
- mature-reassignment threshold/margin sensitivity;
- V100 efficiency profiling.

The complete experiment-to-runner map and commands are in the root README.

## Current selected point estimates

Full-stream positive-class metrics, seed 42:

| Dataset | F1(+) | Recall(+) | Accuracy | AUC |
|---|---:|---:|---:|---:|
| CES | 0.7060 | 0.7580 | 0.8202 | 0.8743 |
| GLOBEM | 0.7359 | 0.8503 | 0.7179 | 0.7926 |

Cold-safe user-averaged positive-class first-K metrics:

| Dataset | F1@5 | F1@10 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| CES | 0.2713 | 0.2632 | 0.3848 | 0.3932 |
| GLOBEM | 0.4380 | 0.4697 | 0.5180 | 0.5445 |

These tables use different selected presets. They must not be reconstructed by
mixing columns from one event file.

## Verification completed for the current code update

- remote Python syntax compilation of the modified runners;
- 49 remote unit tests, all passing;
- explicit tests for the four named presets;
- explicit tests that the full ablation uses the classification-tuned center;
- first-K tests using recorded deployment predictions;
- positive-class and class-macro metric naming separated.

Full data experiments were not rerun as part of this documentation update.
The numerical values above are the selected existing result artifacts.

## Remaining non-executable manuscript analyses

The source programs for the relation-profile analysis and transfer AUC 0.702
are not present in this code repository. Their values remain manuscript
analysis artifacts rather than executable reproduction targets here.

## Known alignment issues

1. The compiled PDF contains CES Accuracy/AUC from an earlier stored output
   while the current selected rerun reports `0.8202/0.8743`.
2. The CES grouping-signal figure still contains the earlier gradient F1
   `0.7051`; the selected center is `0.7060`.
3. Manuscript prose says personal adapters remain trainable after grouping,
   while the selected code uses `standalone` group adapters.
4. The historical target label contains `cflsplit`, but verified split is off
   in the current manuscript presets.

These issues should be resolved in the manuscript or by new audited reruns
before claiming byte-level paper/code equivalence.
