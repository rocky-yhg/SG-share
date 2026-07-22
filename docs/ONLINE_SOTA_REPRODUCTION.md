# Online SOTA reproduction boundary

The suite evaluates every method in two state scopes:

- `global`: one model state receives the unified chronological event stream.
- `per_user`: one independent model state is created for each user.

Every run is strictly prequential: predict, reveal the current label, then update.
The first event is included. With `--raw-fixed-05`, all methods use
`raw_probability >= 0.5`, without probability smoothing or threshold search.
Without that flag, the legacy configured prediction path remains available.
Cold-start outputs include user-macro and pooled F1 for each configured first-K
value.

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
  --datasets ces,globem \
  --methods oli2ds,obal,hbp,koil,olifl,olfl \
  --scopes global,per_user --raw-fixed-05 --no-scaling \
  --seeds 42 \
  --ces-data data/processed/ces.csv \
  --globem-data data/processed/globem.csv \
  --output results/online_sota_raw05

python process/summarize_online_sota_suite.py
```

Participant-level streams are intentionally not committed. A complete GLOBEM
comparison requires the protected full stream at the path supplied through
`--globem-data`.
