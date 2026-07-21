# Provenance and Reproducibility Boundary

## Recovered facts

The following facts were recovered from the manuscript and historical execution
logs:

- final variant name `p0_bias_best_cflsplit_a_m005_u10_e50`;
- rank-4 LoRA and a shared TinyTFT-style backbone;
- route-gradient EMA signatures and cosine agglomeration;
- `K_min=4` for CES and GLOBEM;
- minimum observations/regroup periods: CES `20/100`, GLOBEM `5/120`;
- split margin/minimum support: `0.005/10 users/50 events`;
- mature refinement threshold/margin: `20/0.001`;
- per-user scalar bias;
- first-K sets and post-warmup exclusions encoded in the YAML files;
- the manuscript point estimates in `expected/paper_results.csv`.

## Reconstructed implementation

The source repository previously used to produce the numbers is unavailable on
the current machine. `sgshare/` is therefore a clean-room implementation of the
documented algorithm, not a byte-identical recovery. In particular, the exact
former TinyTFT layer graph, random initialization order, buffer eviction policy,
and all undocumented optimizer details cannot be certified from the available
record. These choices are explicit in code and configuration so that new runs
are reproducible.

## Baseline status

The PDFK, Budgeted, and Supermask baselines are online adaptations reconstructed
from historical local code. They are marked as `ported` in output metadata.
They should be used for capacity-controlled comparison in this package, while
the paper point estimates remain the authoritative comparison to the historical
experiment.

## Claim policy

A run is considered a numerical reproduction only if its event stream is built
from the same licensed raw data and its metrics fall within the tolerance in the
comparison report. A successful synthetic smoke test establishes software
correctness only; it does not reproduce the paper result.
