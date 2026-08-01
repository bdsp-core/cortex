# Label-agnostic validation verdict — F1 PARKED (2026-07-31)

Per the owner-approved kill rule (one bounded pass, two exits, no
iteration), the categorical F1 engine **parks before the locked campaign**:
the pre-registered smoke cells predict a cell-1/cell-2 absolute-gate
failure at power, and no locked 65M-series seed was burned. This document
is the park record. All artifacts, gates, and evidence remain on branch
`nway-f1-promotion-20260731`.

## Smoke results (192 CRN seeds each, production profile, engine-frame artifact)

| cell | world | nway skill cov [Wilson] | binary | paired Δ |
|---|---|---|---|---|
| continuum (pre-reg 1) | β ~ LogNormal(μ̂, τ̂) | 0.9227 [0.906, 0.937] | 0.9488 | −0.026 |
| served-bank + Precision stop (pre-reg 2) | same, real bank, ~349 q | **0.8698** [0.849, 0.888] | 0.9531 | −0.083 |
| stress (pre-reg 3, informational) | β = 1.903 | 0.8108 | 0.9505 | −0.140 |
| collapse (pre-reg 4, informational) | λ_d = 1 | 0.4288 | 0.9566 | −0.528 |
| **on-grid diagnostic** | β uniform on the 9 atoms | **0.9375** [0.922, 0.950] | 0.9497 | −0.012 |

## Attribution

The on-grid diagnostic separates the two failure sources:

1. **Static mixture averaging (~half the deficit, and it compounds).** The
   engine averages the nine (β, λ) draws with fixed weights per
   observation — equivalent to assuming the sitter's β resamples every
   question — rather than inferring which draw explains the session. It
   therefore undercoves (0.9375) even when the world sits exactly on its
   own prior atoms, where a joint Bayesian treatment would be calibrated
   by construction. Longer sessions accumulate more misspecified
   cross-domain updates: the served-bank Precision cell (~349 questions)
   drops to 0.870.
2. **Continuum off-grid mass (~half).** Worlds between and beyond the
   nine atoms add the rest (0.9375 → 0.9227); the β = 1.903 world (the
   independent expert median, outside the ensemble's 1.475 top draw)
   shows the tail cost (0.811).

## What is NOT in question

- The channel is real: shadow replay on 39,272 real reads shows +0.36
  bits/wrong-pick over uniform, out-of-reader, margin guard intact.
- The asked-class margin, binary engine, monitor (trips collapse 100% at
  median 15 wrong-picks), floor pricing, DR07 identifiability, and the
  Precision-stopping join are all green.
- Accuracy: categorical skill RMSE beats binary in every non-collapse
  cell (0.35–0.47 vs 0.54). The failure is calibration, not information.

## The bounded un-park option (owner decision; no work started)

Construction B — the per-session draw latent: each particle carries a
draw index, making β a jointly-inferred session latent. By construction
it restores exact calibration against the discrete prior (closing
deficit 1), lets atom count grow without a blur penalty (closing much of
deficit 2), and with a λ_d≈1 atom nests binary as graceful degradation.
Scope: Python oracle change + TS engine port with parity artifacts +
re-validation under this same pre-registration (amended for the new
model, disclosed) ≈ 2–3 working days. The alternative stands: keep
binary, which is valid, shipped, and unaffected by everything above.
