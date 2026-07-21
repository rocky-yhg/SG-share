# Online SOTA reproduction boundary

The suite evaluates every method in two state scopes:

- `global`: one model state receives the unified chronological event stream.
- `per_user`: one independent model state is created for each user.

Every run is strictly prequential: predict, reveal the current label, then update.
The first event is included. All methods share the same causal online feature
standardizer, probability smoothing, causal window-F1 threshold selection, and
positive-class F1 metrics. Cold-start outputs include user-macro and pooled F1
for each configured first-K value.

## Implementation status

| Method | Implementation source | Protocol boundary |
|---|---|---|
| OLI2DS | Port of the authors' public Python update | Fixed complete features; missing-feature branch is inactive |
| HBP/ODL | Reimplementation from the public HBP code | HBP-19, 100 hidden units, online batch size one |
| KOIL | Python port of the authors' MATLAB equations | RBF kernel, fixed class buffers, FIFO compensation |
| OBAL | Independent mechanism reproduction | No public author code found; users are treated as asynchronous streams |
| OLIFL | Mechanism-aligned specialization | Features and labels are complete in the benchmark |
| OLFL | Labeled protocol surrogate | Original task is label-free and cannot identify positive-risk semantics without an external anchor |

Only OLI2DS, HBP/ODL, and KOIL should be described as source-grounded
reproductions. OBAL is an independent mechanism reproduction. OLIFL and OLFL
are compatibility analyses, not official numerical reproductions.

## Commands

```bash
python process/run_online_sota_suite.py \
  --datasets ces \
  --methods oli2ds,obal,hbp,koil,olifl,olfl \
  --scopes global,per_user \
  --seeds 42 \
  --output results/online_sota_ces_20260720

python process/summarize_online_sota_suite.py
```

The restored repository currently contains only a 120-event GLOBEM sample.
Runs on that sample validate the pipeline but are not a full-dataset performance
comparison. A full GLOBEM table requires the protected complete stream.
