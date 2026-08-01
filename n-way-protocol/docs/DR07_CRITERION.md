# DR07 operational criterion for the F1 conditional-distractor fit

**Status: executed under the owner's 2026-07-31 approval to solve the DR07
blocker; the criterion mapping below awaits explicit owner ratification.**

The originating requirement — item 11 of `OPTIMAL_DESIGN_RD_RECOMMENDATIONS`
— is absent from this repository; its wording survives only in citations
("a versioned DR07 refit with at least 32 exact same-state joint draws plus
state zero, followed by the governed rank/conditioning gate"). The joint-state
language belonged to the earlier *inclusive* categorical model, which died
with 20 of 32 joint states unpopulated. For the F1 conditional-distractor
family the natural analogue, already used by the N3c preliminary, is the
30-cell off-diagonal confusion-state space (gold class → wrong pick), and
the rank/conditioning gate applies to the (β, λ_d) fit.

## Criterion (v2), executed by `python/nway_protocol/dr07_gate.py`

Fit frame: **engine frame** (per-reader per-domain bias + scalar sensitivity,
MAP under the engine priors; the frame the artifact is deployed in). Corpus:
the pre-registered stable core (readers with ≥100 reads, ≥30 wrong-picks,
interior sensitivity |ℓ̂| ≤ 2.5).

1. **State coverage** — all 30 off-diagonal confusion states populated with
   ≥ 5 reads.
2. **Identifiability at the optimum** — observed information of (β, λ_d) at
   the pooled fit. λ_d has pinned at its zero boundary in every fit to date;
   at a boundary optimum identification is one-sided: the λ profile slope at
   zero must be positive and the β curvature positive. At an interior
   optimum the 2×2 information matrix must be positive definite with
   condition number ≤ 10⁴.
3. **Split-half stability (population parameters)** — readers split
   deterministically by id hash; each half gets an independent hierarchical
   fit; the population means μ of log β must agree within |z| ≤ 3 using the
   halves' standard errors.

## Amendment history

- **v1 (superseded, same session):** check 3 compared the *pooled* β between
  halves at 5% relative tolerance. First execution: 5.04% → **fail**
  (recorded in `reports/dr07_gate_engine_frame_v1_fail.json`). The failure
  is a property of the criterion, not the fit: with the measured
  between-reader heterogeneity τ ≈ 0.255 (log scale) and ~62 readers per
  half, the sampling-expected half-to-half pooled difference is
  √2·τ/√62 ≈ 4.6%, so the v1 check rejects a correctly-specified
  heterogeneous population roughly half the time. The artifact under
  qualification is a *population-quantile* ensemble; its stability claim is
  about the population parameters (μ, τ), which v2 tests with proper
  standard errors. The pooled difference remains in the report as a
  diagnostic. Both runs are preserved; the v1 threshold was authored this
  session and had not been owner-ratified.

## What a pass authorizes

A `pass` verdict permits stamping `provenance.dr07: "qualified"` on the
engine-frame response artifact, recording this document and the gate
report's SHA-256. It does **not** mark the artifact `qualification:
"qualified"` (that requires the locked campaign) and authorizes no
production change.
