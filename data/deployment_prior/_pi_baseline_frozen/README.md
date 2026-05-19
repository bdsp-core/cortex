# Frozen PI-lineage deployment baseline (archived at Phase 4.6-A)

Phase 4.6 re-fits/re-freezes `data/deployment_prior/` on the unified
corpus (K=7, v13 ℓ*), OVERWRITING the canonical `Sigma.csv`,
`case_bank.csv`, `ell_thresholds.csv`, and (4.6-B) `sim/`.

The committed Phase-4.3 / 4.3b / 4.5 controlled studies are scoped
"under PI `ell_thresholds` ℓ* / PI fixed-θ population" — their
provenance + committed JSON artifacts are defined against THESE files.
They (and their slow regression tests) are repointed here so they stay
fully reproducible as historical studies after 4.6 regenerates the
canonical artifacts.

  ell_thresholds.csv  — PI K=6 Youden ℓ* lineage (pre-4.5/4.6)
  candidates.csv      — PI K=6 sim, the fixed-θ population those
                        controlled studies replay
  sim_summary.json    — its provenance (seed=0, cfg, tiers)

DO NOT regenerate these — they are an immutable historical anchor.
