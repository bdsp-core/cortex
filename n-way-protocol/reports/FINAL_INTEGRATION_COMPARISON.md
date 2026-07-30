# Final frozen-versus-integrated n-way comparison

Status: **designated the standard isolated n-way approach; development gate
passed; production promotion remains forbidden**.

Run date: 2026-07-20. The comparison used 2,000 matched seeds, 1,200 particles,
30 MH steps, 46 process workers, and 24 questions per session. The raw report
contains exactly 4,000 arm rows in
`final_frozen_vs_integrated_1200p30_2000.json`.

## Compared implementations

The frozen arm used the immutable F1 recovery-point behavior:

- Scalar exploratory artifact: beta 0.9912 and distractor lapse 0.
- Original deterministic categorical total-variance shortlist.
- Exact six-outcome final loss.
- Multinomial resampling, ESS threshold 0.5, 1,200 particles, and 30 MH steps.

The integrated arm changed only the retained R&D features:

- Nine-draw reader-bootstrap artifact mixture.
- Fisher-augmented shortlist using a weighted-mean artifact only for the cheap
  screen.
- Exact nine-draw, six-outcome total-variance final choice.
- Exact ensemble likelihood for update and complete-history MH replay.
- The same resampling, ESS, particle, MH, and 24-question settings as frozen.

The skill and bias truths were drawn from the inference prior. For response
behavior, each matched seed drew one of the nine artifact settings and held it
fixed for the session. This is a prior-predictive skill/bias study plus an
artifact-heterogeneity stress. It is deliberately stricter than generating
every response from the integrated per-observation mixture and is not a fully
specified hierarchical artifact SBC.

## Overall result

| Metric | Frozen F1 | Integrated R&D | Integrated change |
|---|---:|---:|---:|
| Skill coverage | 0.94592 | 0.94250 | -0.00342 |
| Bias coverage | 0.94642 | 0.94550 | -0.00092 |
| Skill interval width | 2.40935 | 2.21002 | 8.04% tighter |
| Bias interval width | 2.43695 | 2.36522 | 2.58% tighter |
| Skill RMSE | 0.59517 | 0.54639 | 8.20% lower aggregate mean |
| Bias RMSE | 0.60727 | 0.59672 | 1.74% lower aggregate mean |
| Mean resamples | 9.92 | 11.10 | +1.18 |
| MH acceptance | 0.2225 | 0.2218 | essentially unchanged |
| Distinct ancestry | 0.4303 | 0.4280 | essentially unchanged |

Paired uncertainty:

- Skill coverage difference: -0.00342, 95% CI
  `[-0.00863, 0.00180]`.
- Bias coverage difference: -0.00092, 95% CI
  `[-0.00593, 0.00409]`.
- Skill width ratio: 0.91958, 95% CI `[0.91674, 0.92242]`.
- Bias width ratio: 0.97420, 95% CI `[0.96998, 0.97842]`.
- Skill RMSE difference: -0.04878, 95% CI
  `[-0.05804, -0.03952]`.
- Bias RMSE difference: -0.01055, 95% CI
  `[-0.01939, -0.00172]`.

Both coverage-difference lower bounds pass the preregistered development
noninferiority rule of integrated minus frozen >= -0.03. Both paired width
confidence intervals lie below 1. The combined relative development gate
therefore passes.

Ratios of per-seed RMSEs are unstable when the frozen denominator is very
small. The RMSE interpretation above uses the paired absolute-difference
intervals and ratios of aggregate means; it does not claim superiority from
the unstable mean of individual ratios.

## Artifact-behavior strata

All nine descriptive strata tightened both parameter blocks:

- Skill width ratios ranged from 0.9121 to 0.9264.
- Bias width ratios ranged from 0.9622 to 0.9892.
- The largest skill coverage decrease was 0.01615 at beta 1.09155.
- The largest bias coverage decrease was 0.01542 at beta 1.04186 and lapse
  0.01305.

No stratum showed a three-point coverage loss. The strata contain only 182–245
replicates each and are descriptive, not separately powered noninferiority
tests.

## Absolute coverage warning

The integrated skill coverage interval was `[0.93818, 0.94682]`; the integrated
bias interval was `[0.94138, 0.94962]`. Both point estimates are below 0.95 and
both intervals narrowly exclude nominal coverage. Frozen coverage was also
below 0.95, with intervals `[0.94178, 0.95005]` and
`[0.94227, 0.95056]`.

No absolute-coverage margin was preregistered for this development test. The
result shows relative coverage retention and real tightening, but it does not
close the absolute-coverage gate. That gate requires a locked run using the
qualified artifact, served bank, and complete Precision stopping workflow.

## Integration completed in the isolated protocol

- `ConditionalF1ResponseArtifact` supports stamped scalar and ensemble
  artifacts without reinterpreting frozen profiles.
- The likelihood, posterior update, replay, MH rejuvenation, expected loss,
  and predictive distribution accept the ensemble exactly.
- `categorical_fisher_totalvar_v1` dispatches only integrated profiles to the
  Fisher shortlist; frozen profiles keep `categorical_totalvar_v1`.
- Artifact and engine-profile schemas include the new immutable contracts.
- Independent Python/TypeScript parity covers ensemble probabilities, wrong-
  pick updates, log likelihoods, and exact six-outcome loss.
- The actual R&D artifact and profile remain `exploratory_unqualified`,
  non-DR07, and `promotionForbidden`.

The original archive remains read-only and independently recoverable. No
production `cortex_web`, API, database, or rollout code was replaced.

Following review of these results, this integrated profile was designated the
standard for new isolated n-way work. Frozen F1 remains available for replay,
comparison, and rollback. The designation is recorded in the fail-closed
machine-readable `artifacts/nway_standard_rd.json` pointer and does not alter
the production authorization status.

## Final verification

- Strict TypeScript type-check passed.
- Standard suite passed 37 TypeScript tests; three opt-in tests were skipped by
  design in that command.
- Python suite passed 32 tests.
- Independent integrated Python/TypeScript fixture parity passed.
- The opt-in selector-regret audit passed: mean regret 0.05166 to 0.01209 and
  exact-choice recall 0 to 0.875 in its eight-state tractable audit.
- The final isolated 35,000-segment benchmark passed in 1.53 seconds for frozen
  and 4.03 seconds for integrated ensemble/Fisher selection. These are host
  measurements, not browser/device qualification.
- The frozen archive and its 52-file digest manifest remained valid and
  read-only.
- All retained JSON reports parsed, and the final report contained 2,000
  unique seeds with exactly one complete row per arm.
