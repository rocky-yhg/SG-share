# SOTA Faithfulness and Result Audit

## Scope

The goal of this audit is paper-method faithfulness under the SG-Share
passive-sensing protocol, not numerical reproduction of the papers' original
image benchmarks. Every method uses:

- the same chronological, preprocessed passive-sensing stream;
- one independent two-hidden-layer model per user;
- prediction before label observation and update;
- the same weighted focal task loss and binary metrics;
- memory size 128 and replay batch size 8;
- every user's first event in evaluation.

## Source audit

| Method | Primary source | Code source | Audited revision |
|---|---|---|---|
| PDFK | [AAAI 2026, *Perturbing to Preserve*](https://ojs.aaai.org/index.php/AAAI/article/view/40129) | [official code](https://github.com/colaudiolab/PDFK), vendor/PDFK | ff4f58a320eeab35e76bb75999c75c6a862b967d |
| Budgeted (aL-SAR) | [ICLR 2025, *Budgeted Online Continual Learning*](https://proceedings.iclr.cc/paper_files/paper/2025/hash/760564ebba4797d0dcf1678e96e8cbcb-Abstract-Conference.html) | [official code](https://github.com/snumprlab/budgeted-cl), vendor/budgeted-cl | fb4d79cd928d5f00b013d3a2c525a5a1e374b337 |
| Supermask | [AAAI 2026, *Parameter Merging with Gradient-Guided Supermasks*](https://ojs.aaai.org/index.php/AAAI/article/view/39687) | no official code linked | equations 13--21 and Algorithm 1 |

The executable implementation is sgshare/baselines.py; mechanism-level
regression tests are in tests/test_paper_baselines.py.

## Mechanisms retained

- **PDFK:** random reservoir replay, one EMA teacher, temperature-3 weighted
  distillation with coefficient 5.5, and parameter-space adversarial
  perturbation on correctly classified replay observations.
- **aL-SAR:** layer-wise Fisher information, the batch freezing criterion,
  greedy balanced memory insertion, and similarity-aware retrieval based on
  discounted/effective use frequency.
- **Supermask:** first-order and Fisher-approximated second-order masks,
  old/new parameter merging with alpha 0.01, and dual-model dual-view
  distillation with coefficient 5.5.

Image augmentations are replaced by a small Gaussian feature perturbation for
the Supermask dual-view term. No task identities or future labels are used.

## Full CES results

The full audited CES stream contains 35,289 events, 218 users, 37 features, and
a positive rate of 0.284876. Results below are seed 42.

| Method | F1(+) | Recall(+) | Specificity | Balanced Acc. | Post F1(+) |
|---|---:|---:|---:|---:|---:|
| SG-Share full_final | 0.6919 | 0.7601 | 0.8259 | 0.7930 | 0.6972 |
| aL-SAR paper-faithful adaptation | 0.5597 | 0.8996 | 0.4762 | 0.6879 | 0.5607 |
| PDFK paper-faithful adaptation | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 0.0000 |
| Supermask paper-faithful adaptation | 0.0000 | 0.0000 | 1.0000 | 0.5000 | 0.0000 |

Result directories:

- results/real_ces_full_final_restored_v4_seed42
- results/real_ces_budgeted_paper_faithful_v2_seed42
- results/real_ces_pdfk_paper_faithful_v2_seed42
- results/real_ces_supermask_paper_faithful_seed42

PDFK and Supermask collapse to the negative class in this per-user sparse-data
adaptation. This is an applicability result, not evidence that the methods fail
on their original global image streams. The result must not be presented as the
authors' published benchmark performance.

## CES first-K

| Method | Macro F1 K=5 | K=10 | K=20 | K=50 |
|---|---:|---:|---:|---:|
| SG-Share full_final | 0.2649 | 0.2571 | 0.2582 | 0.2926 |
| aL-SAR adaptation | 0.0000 | 0.0297 | 0.1840 | 0.2855 |
| PDFK adaptation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Supermask adaptation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

SG-Share's early advantage is clear at K=5, 10, and 20 against the strongest
paper-faithful adaptation. At K=50 the aL-SAR macro F1 is slightly lower than
SG-Share (0.2855 versus 0.2926).

## GLOBEM status

Only the official 120-event sample is locally available. It verifies execution
and preprocessing contracts but is too small for a performance claim. The
protected full stream used by the historical experiments is absent. Historical
full-stream point estimates remain available in
expected/historical_sota_results.csv; they are provenance records rather than
fresh reproduction outputs.

## Interpretation boundary

Two comparisons should remain separate:

1. **Controlled cross-domain adaptation:** the executable results above compare
   algorithms under one passive-sensing protocol.
2. **Historical strongest point estimate:** the older per-user PDFK result
   (CES F1 0.6350; GLOBEM F1 0.7283) is a stronger conservative reference, but
   its original runner is not recovered byte-for-byte.

SG-Share exceeds the conservative PDFK point estimate on CES. On GLOBEM its
historical overall F1 (0.7268) is close to, but not above, PDFK (0.7283);
therefore the defensible statement is comparable overall F1 with a different
recall/specificity operating point, not uniform dominance.
