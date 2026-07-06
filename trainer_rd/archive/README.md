# archive/ — stale files moved aside (never deleted)

Project convention (mirrors `sandbox/state/archive-*`): files that no longer
serve the working tree are MOVED here with a line in this README — deletion
is reserved for reproducible build products (`__pycache__`). Check
`docs/PROJECT_MEMORY.md` §1 before archiving anything: if a file is
referenced there (or by code — grep first), it is not stale.

| archived | file | why |
|---|---|---|
| 2026-07-05 | `sbc_run.log` (from `figures/`) | one-off captured console log of the M14 EXTSET SBC run; its results are recorded in `figures/data_extset_sbc.npz` and PROJECT_MEMORY F51/F52 |
| 2026-07-06 | `relocation_baseline_suite.log` | Gate G0 pre-move baseline for the `docs/REPO_RELOCATION_PLAN.md` relocation into `<repo>/trainer_rd/`: full 21-file / 299-check script suite ALL PASSED (interpreter + numpy/scipy/pandas versions in the header). |
| 2026-07-06 | `relocation_repo_baseline_pytest.log` | Gate G0 repo-side baseline: `<repo>/.venv/bin/python -m pytest -q` on the dirty tree = **485 passed / 2 failed / 11 skipped / 28 deselected**; both failures are pre-existing dirt (`test_phase1_data.py::test_canonical_table_sizes[datasets.csv-5]`, `::test_carry_forwards_present`), NOT relocation-related. This tally is the frozen parity target for Gates G3/G5. |
| 2026-07-06 | `relocation_g0_dirt_manifest.txt` | Gate G0 snapshot of the repo working tree's pre-existing `git status --short` (unrelated web-app / data work); the Phase-5 commits must leave exactly this manifest behind. |
