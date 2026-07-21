# Historical `full_final` divergence audit

This audit compares the reconstructed clean-room implementation with the
historical target variant:

```text
p0_bias_best_cflsplit_a_m005_u10_e50
```

The historical target was composed as:

```text
SG_BACKBONE_BEST[dataset]
+ STRICT_CLEAN_OVERRIDES[dataset]
+ routegrad_agglom_clean
+ backbone_direct_k_min=4
+ mature20
+ p0_bias_lr005_l2_1e3
+ cflsplit_a_m005_u10_e50
```

## Result validity

The surviving run at `results/real_ces_full_final_seed42` is not an aligned
reproduction. It used 649 model features and obtained overall F1 0.55999 and
post F1 0.56222. The recovered historical CES stream has 37 model inputs.
Therefore, the observed 0.56222 versus historical 0.69610 gap cannot be
attributed only to learner parameters. No completed full-CES run currently
combines the corrected 37-input stream with a faithful historical learner.

## Highest-impact structural divergences

| Area | Historical target | Clean-room implementation | Expected impact |
|---|---|---|---|
| Backbone | TinyTFT-style temporal model; CES dim 16/depth 2/heads 4, GLOBEM dim 16/depth 1/heads 2 | 32-dimensional gated MLP with one explicit route layer (`sgshare/model.py:27-49`) | Very high: changes logits and the route-gradient geometry used for grouping |
| LoRA | rank 4, alpha 8; CES LR 0.007, GLOBEM LR 0.005 | rank 4 with scale `1/rank`; LR 0.001 on both datasets (`sgshare/model.py:14-24`, configs) | Very high: adapter scale and learning speed differ by 5-7x |
| Prediction rule | Online F1 threshold search, grid 101, minimum history 200 CES/100 GLOBEM; smoothing; GLOBEM user pos-rate threshold bias 0.09 | Fixed probability threshold 0.5 (`sgshare/learner.py:84-90`) | Very high: directly changes F1, recall, and specificity |
| Warmup | Global warmup 300 CES/120 GLOBEM plus individual warmup 20/2; warmup-v2 initial grouping | No global warmup state; ordinary online updates start at event 1 | Very high: changes both model state and first-K behavior |
| Cold-start prediction | Safe mixture of global/personal/group experts until K=20 plus prototype initialization | Exactly one hard active adapter (`sgshare/learner.py:63-70`) | Very high for first-K; also affects later adapter states |
| Backbone update | Dataset LR plus `backbone_lr_scale=0.3`, direct-route mode, route-gradient clip 1.0 | All backbone parameters stepped at raw LR 0.0003, without scale or clipping (`sgshare/learner.py:119-132`) | High: current CES backbone step is not the target update |

## Exact parameter mismatches

| Parameter | Historical target | Current CES | Current GLOBEM |
|---|---:|---:|---:|
| backbone base LR | 0.0002 | 0.0003 | 0.0003 |
| backbone LR scale | 0.3 | missing | missing |
| route-gradient clip | 1.0 | missing | missing |
| LoRA LR | 0.007 | 0.001 | 0.001 (target 0.005) |
| LoRA alpha | 8.0 | implicit scale 1/rank | implicit scale 1/rank |
| route-signature EMA alpha | 0.05 historical default | 0.10 | 0.10 |
| per-user bias LR | 0.005 | 0.010 | 0.010 |
| per-user bias initialization | group mean, weight 0.5 | zero/defaultdict | zero/defaultdict |
| mature loss margin | 0.01 | 0.001 | 0.001 |
| mature cadence | every 10 grouping boundaries | every grouping boundary | every grouping boundary |
| mature minimum buffer | 5 | no check | no check |
| verified split holdout minimum users | 5 | missing | missing |
| verified split fallback | no recent-buffer fallback | second half of same recent buffer | second half of same recent buffer |

## Grouping semantic divergences

The clean merge configuration itself was recovered exactly. It used pure
route-gradient cosine scoring and disabled all pair filters and loss/churn
acceptance gates:

```text
backbone_direct_posrate_merge_tol=1.0
backbone_direct_min_pair_finite=0
backbone_direct_threshold_factor=1e9
backbone_direct_w_grad=1.0
backbone_direct_w_loss=0.0
backbone_direct_w_lora=0.0
backbone_direct_min_loss_support=0
regroup_tau_low=0.0
regroup_tau_high=1.1
regroup_mid_churn_loss_gate_enabled=False
backbone_direct_k_min=4
```

The clean-room `agglomerative_groups` has the same high-level rule: start from
singletons, repeatedly merge the pair with maximum centroid cosine, stop at
four groups or when the best cosine is negative (`sgshare/grouping.py:40-71`).
This is the closest aligned component, but it still consumes signatures from a
different backbone and optimizer. Its adapter lifecycle also differs:

- Current regrouping uses the currently active adapter as the donor, which may
  already be a group adapter (`sgshare/learner.py:157-159`).
- Only newly created or unmatched groups receive a donor mean; matched groups
  retain their old state (`sgshare/learner.py:301-307`).
- Group identity is reconstructed with a Jaccard >= 0.5 heuristic
  (`sgshare/grouping.py:109-132`).

These are clean-room approximations, not verified historical behavior.

## Refinement semantic divergences

### Per-user bias

The online update equation is structurally similar, but the current LR is
twice the selected target and initialization is not implemented. Historical
`group_mean` initialization with weight 0.5 matters when a user enters a group;
the current `defaultdict(float)` always starts from zero
(`sgshare/learner.py:44-45`, `133-139`).

### Verified split

The current implementation uses a farthest-seed binary split, constructs
diagnostic adapters from source means, and evaluates the second half of each
user's same 64-event recent deque (`sgshare/learner.py:160-225`). The target
required a CFL-style proposal and an explicit holdout source with at least five
users, with `split_holdout_fallback_to_recent=False`. Matching the margin,
minimum child users, and minimum child events does not make these procedures
equivalent.

### Mature refinement

The current implementation evaluates current plus top-two prototype groups on
the entire recent buffer and may move a user at every regroup
(`sgshare/learner.py:227-269`). The selected target required at least five
buffer items, margin 0.01, and execution every ten grouping boundaries. The
current 0.001 margin and missing cadence make migration substantially more
aggressive.

## Evaluation divergences

1. Current first-K includes every user with at least one event and uses up to K
   events (`sgshare/metrics.py:39-63`). Historical outputs show decreasing user
   counts as K increases (CES: 216/214/209/191 at K=5/10/20/50), which means
   users without K observations were excluded. Current reports 218 users at
   every K. This changes user-macro and pooled first-K values.
2. Current post-warmup drops a fixed number of global events
   (`sgshare/learner.py:352-360`). Historical evaluation checked the learner's
   actual warmup phase. The counts can match at 300/120 only if the restored
   warmup state machine is also matched.
3. Current predictions use fixed threshold 0.5 and no smoothing, while the
   historical event log used the online thresholding path. Metric files are
   therefore not directly comparable even when probabilities were similar.

## Causal priority

The current gap should be addressed in this order:

1. Restore the historical TinyTFT and LoRA parameterization.
2. Restore online threshold search, smoothing, and GLOBEM threshold bias.
3. Restore global/individual warmup, warmup-v2 initial grouping, safe expert
   mixing, and prototype initialization.
4. Restore optimizer scales and route-gradient clipping.
5. Restore exact bias, mature-refinement, and split semantics.
6. Align first-K inclusion and post-warmup evaluation.
7. Only then tune `k_min`, split margins, or other grouping parameters.

Changing only the visible YAML values cannot reproduce the historical target,
because the highest-impact differences are missing execution paths rather than
scalar hyperparameters.

## Required verification ladder

Use one seed and the corrected 37-input CES stream for a staged comparison:

1. Current clean-room baseline.
2. Historical backbone + LoRA scale/LRs.
3. Add historical thresholding and smoothing.
4. Add historical warmup and cold-start mixing.
5. Add exact clean grouping and adapter lifecycle.
6. Add bias, verified split, and mature refinement.
7. Recompute historical first-K and post-warmup metrics from the same event log.

This ladder is necessary to quantify attribution. The current artifacts support
identifying divergences, but they do not support assigning a numerical fraction
of the F1 gap to any one parameter.
