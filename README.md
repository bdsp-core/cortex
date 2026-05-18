# ILAE Skill Certification (unified)

> **Phase 0 skeleton.** This README is a placeholder; the merged, authoritative
> README + CLAUDE.md land in Phase 8. See `../UNIFIED_REPO_MERGE_PLAN.md` for
> the full plan and `docs/MERGE_SOURCE_MANIFEST.md` for source provenance.

Unified repo merging the methodology/simulation engine (SMC+MCMC Paper-1) and
the PI clinical deployment runtime (Laplace/EKF). Two engines, cleanly
separated (D1); one shared data + calibration layer; Full-7 / K=7 (D5).

## Quickstart (target env: Python 3.11.9)

```sh
python -m venv .venv && .venv/bin/pip install -e ".[test]"
.venv/bin/pytest          # Phase-0 structural self-tests
```

Console entry points (Phase 0 = loud stubs; wired in later phases):
`ilae-paper` (Phase 2) · `ilae-calibrate` (Phase 3) · `ilae-deploy` (Phase 4).
