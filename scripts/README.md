# scripts/ — research/reference validation and experiment harnesses

This directory supports the root Python research/reference stack; it is not
imported by the canonical `cortex_web` runtime. `_parallel.py` is the
process-pool helper, `run_*.py` contains the dated SBC/coverage/retest and
simulation drivers, and the `cortex_*` modules support the local Python
integration reference.

Canonical corpus-building and tier-consolidation ownership remains in
`pipeline/`. Do not recreate historical copies under `scripts/`; duplicate
builders are a provenance and drift hazard.
