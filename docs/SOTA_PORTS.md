# Paper-Faithful SOTA Adaptations

All baselines use the same canonical stream, feature preprocessing, two-hidden-
layer per-user model, memory/replay budget, weighted focal task loss, strict
prediction-before-update order, and metric implementation. The first event of
every user remains in evaluation. The implementations preserve each paper's
defining online update rule while adapting image-specific inputs and
augmentations to tabular passive-sensing features.

## PDFK

The implementation follows the official PDFK code: random reservoir replay, one
EMA teacher (alpha 0.01), temperature-3 weighted KD (coefficient 5.5), and
parameter-space adversarial perturbation on correctly classified replay
observations (gamma 0.0005, consistency coefficient 0.005).

## Budgeted Online Adaptation (aL-SAR)

The implementation follows the paper's two defining components: batch freezing
criterion (BFC) from layer-wise Fisher information per computation, and
similarity-aware retrieval using discounted/effective use frequency. The paper
defaults k=4, T=0.125, Fisher/similarity EMA 0.01, and a Fisher refresh every four
batches are retained.

## Supermask Online Adaptation

No official repository was linked by the paper. This implementation follows
Equations 13--21 and Algorithm 1: first-order and Fisher-approximated second-
order sigmoid masks merge old/new parameters with alpha 0.01, and a dual-model
dual-view KL term uses coefficient 5.5. A small Gaussian feature perturbation is
the tabular counterpart of the paper's image augmentation.

## Interpretation

These are cross-domain paper-faithful adaptations, not claims of reproducing the
papers' image-benchmark numbers. PDFK and aL-SAR are audited against vendored
official repositories in `vendor/`; Supermask is audited against its published
equations. Historical local point estimates remain provenance records only and
are not implementation acceptance targets.
