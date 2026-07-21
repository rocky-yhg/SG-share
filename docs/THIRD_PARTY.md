# Third-Party Source Boundary

Third-party repositories are not committed into SG-Share. Keeping them outside
the main Git history avoids redistributing code without a clear license,
preserves upstream attribution and history, prevents accidental publication of
upstream credentials or large model assets, and keeps the reproduction package
small. The `vendor/` directory is ignored and is used only for local source
audits.

## Pinned repositories

| Local path | Upstream repository | Pinned commit | Use in SG-Share |
|---|---|---|---|
| `vendor/PDFK` | <https://github.com/colaudiolab/PDFK.git> | `ff4f58a320eeab35e76bb75999c75c6a862b967d` | Audit the PDFK online continual-learning port |
| `vendor/budgeted-cl` | <https://github.com/snumprlab/budgeted-cl.git> | `fb4d79cd928d5f00b013d3a2c525a5a1e374b337` | Audit Budgeted Online Adaptation/aL-SAR |
| `vendor/OnlineDeepLearning` | <https://github.com/phquang/OnlineDeepLearning.git> | `90456299654b742b45374c83f8923faf895db13c` | Audit the HBP/ODL equations and update path |
| `vendor/GLOBEM` | <https://github.com/UW-EXP/GLOBEM.git> | `4f140fc5290dc97204298cca28b956165aa0a29f` | Audit the official GLOBEM file contract and preprocessing |
| `vendor/OLI2DS` | <https://github.com/youdianlong/OLI2DS.git> | `119e7c1764d2c4f592f7582e0857a351a2819dfe` | Audit the OLI2DS public Python implementation |
| `vendor/KOIL` | <https://github.com/JunjieHu/koil.git> | `5b181412795a2c9777e9d54eef9f94233c01d9c0` | Audit the KOIL MATLAB/C++ implementation |

Run the pinned checkout helper from the repository root:

```bash
bash scripts/fetch_third_party.sh
```

To restore only selected repositories, pass their names:

```bash
bash scripts/fetch_third_party.sh PDFK GLOBEM KOIL
```

The helper checks out detached commits. Do not commit the resulting `vendor/`
tree. Consult each upstream repository for its current license and usage terms.
At the audited commits, only GLOBEM exposed a top-level license file (Apache
2.0); no top-level license file was found in the other five local checkouts, so
their source is linked rather than redistributed.

## Runtime interface

The core `sgshare` package does not import code from `vendor/`. Baseline ports
used by the comparison runner are implemented in:

- `sgshare/baselines.py` for PDFK, Budgeted Online Adaptation, and Supermask;
- `sgshare/online_sota.py` for OLI2DS, HBP/ODL, KOIL, OBAL, OLIFL, and OLFL.

The pinned repositories are evidence for implementation audits, not hidden
runtime dependencies. `docs/SOTA_FAITHFULNESS_AUDIT.md` and
`docs/ONLINE_SOTA_REPRODUCTION.md` state which methods are source-grounded,
paper-mechanism reproductions, or protocol surrogates.
