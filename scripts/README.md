# Script Inventory and Runtime Boundary

The portable, paper-facing commands are listed in the root README. They were
audited against the retained SeetaCloud runtime under
`/root/autodl-tmp/sgshare-runtime`.

## Maintained portable scripts

- `prepare_ces_historical.py`, `prepare_ces_phq4.py`: CES preparation.
- `prepare_globem_weekly.py`, `align_globem_official.py`: GLOBEM preparation
  and alignment.
- `validate_stream.py`: canonical stream validation.
- `audit_historical_data_alignment.py`: non-sensitive alignment audit.
- `convert_legacy_arff.py`: historical ARFF conversion.
- `run_named_reproduction.sh`: wrapper for the named manuscript presets.
- `fetch_third_party.sh`: fetches documented baseline revisions without
  vendoring them into this repository.

## Optional cluster-specific launchers

Files ending in `_v100.pbs` are retained as optional launchers for a separate
PBS/V100 environment. They were not used to validate commit `c81444c`, are not
required to reproduce the selected SeetaCloud results, and should not be
treated as the canonical runtime description.

Keeping these launchers preserves prior cluster provenance without mixing
their paths or scheduler assumptions into the portable commands.
