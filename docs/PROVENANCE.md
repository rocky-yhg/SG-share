# Provenance and Reproducibility Boundary

## Current manuscript-aligned facts

The current `main.pdf`, recovered result artifacts, and selected-parameter
reruns identify the following execution points:

- historical target label `p0_bias_best_cflsplit_a_m005_u10_e50`;
- rank-4 LoRA and a shared TinyTFT-style backbone;
- route-gradient EMA signatures and cosine agglomeration;
- classification-tuned `(alpha, admission observations, k_min)`:
  CES `(0.20, 20, 5)`, GLOBEM `(0.10, 2, 5)`;
- cold-safe points:
  CES `(0.05, 20, 3)`, GLOBEM `(0.12, 2, 4)`;
- regroup periods: CES 100 events, GLOBEM 120 events;
- mature reassignment threshold/margin: `20 / 0.01`;
- scalar per-user calibration bias;
- verified split disabled in all current selected presets;
- seed 42 for the point estimates in the current manuscript tables.

The executable registry is `sgshare/presets.py`. A result is identified by its
resolved `config.yaml`, processed-stream SHA-256, Git commit, and event file,
not by the historical target label alone.

## Data provenance

The selected streams are:

- CES: 35,289 events, 218 users, 37 model inputs, SHA-256
  `771a8973ac58e1b50a071e0e55ed681a022b7f283cc7f19c631e365720745eb5`;
- GLOBEM: 8,225 events, 704 participant-year users, 130 model inputs,
  SHA-256
  `d7ead91d72b647d3e881ae43c8bd0837906935720969081a59c1c6c996caeacc`.

The participant-level streams are restricted and are not distributed through
Git. CES comes from the authorized College Experience Dataset release. GLOBEM
comes from credentialed PhysioNet GLOBEM v1.1 access.

## Reconstructed implementation

The original historical source tree was not recovered byte-for-byte.
`sgshare/` is a clean-room implementation reconstructed from manuscript
equations, execution records, configurations, and recovered event outputs.
Undocumented historical choices such as the exact former layer graph,
initialization order, and buffer eviction policy cannot be certified as
identical.

The selected executable mode is `group_adapter_mode=standalone`: after group
assignment, the group adapter is updated and the temporary personal adapter is
not updated in parallel. This differs from manuscript prose that currently
states all personal adapters remain trainable.

## Baseline boundary

The current manuscript compares offline ML/DL methods and online HBP/ODL,
KOIL, OLFL, OLI2DS, and OLIFL in global and per-user scopes. Implementations
and source-fidelity limits are documented in
`docs/ONLINE_SOTA_REPRODUCTION.md`.

These passive-sensing ports should not be presented as numerical reproductions
of results reported on the methods' original benchmark domains.

## Claim policy

A value is a current-code numerical result only when the run directory retains:

- the selected Git commit;
- the complete resolved configuration;
- the processed-stream path and SHA-256;
- the seed, device, and command;
- the event-level predictions used to recompute the metric.

Values copied from `main.pdf`, figure-generation scripts, or historical
records remain manuscript artifacts until their corresponding event files are
available and audited.
