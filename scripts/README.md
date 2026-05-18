# scripts/ — validated validation & experiment harness (from the methodology repo)

Carried verbatim in Phase 2 so the methodology repo's test suite
(`tests/test_parallel_determinism.py` and the Phase-7 validation drivers)
runs unchanged. `_parallel.py` is the process-pool helper; `run_*.py` are
the SBC / coverage / sparcnet-retest / gold-chain / phase1 / phase4
drivers re-run in Phase 7.

NOTE: `build_unified_labels.py` and `consolidate_label_tiers.py` are
**deliberately NOT here** — their canonical, Phase-1-corrected copies live
in `pipeline/` (build_unified_labels.py path-patched; consolidate_label_tiers.py
R3-corrected to avoid Centaur double-count). Do not re-introduce the
methodology-repo originals here (divergent-copy hazard).
