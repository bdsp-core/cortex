# Frozen cut-independent PrecisionPolicy

## Implementation status

The approved stopping design and selected `frontier_p90guard_m3` profile are
implemented in the standalone `../precision-policy/` package and connected to
this Python reference through the existing policy seam. PrecisionPolicy
development is closed. Its material extreme-skill tail undercoverage is an
accepted, disclosed limitation and is not repaired or tuned here. AD6 remains
the default and rollback policy in `../ad6-policy/`. Nothing in `cortex_web`
has been modified or promoted.

The prerequisite numerical profile corrections are included: full Python sessions use 1,200
particles, `Corr_l` for skill, `Corr_t` for bias, and the v15 AD6 block. Invalid
posterior updates now reject transactionally and end the session with
`invalid_posterior_update`; the old silent uniform reset is gone.

## Runtime contract

For domain `k`, the skill tolerance is
`r_l,k = c_k * SD_prior(l_k)`. In task order spike, sz, lpd, gpd, lrda, grda,
iic, the selected development multipliers are
`[1.50, 1.30, 1.35, 1.30, 1.30, 1.30, 1.25]`. A domain is
`ESTIMATE_COMPLETE` only when:

- the maximum distance from its posterior mean to either realized central 95%
  marginal skill-interval endpoint is at most `r_l,k`;
- the configured reliability guard passes for two consecutive evaluations;
- at least 20 unique own-domain segments have been answered; and
- each of the three signal-content bands contains at least 3 unique
  own-domain answers.

Completion is derived fresh after every joint-cloud update and can revert to
`ACTIVE`. A non-complete domain at its 60th own answer becomes terminal
`UNDETERMINABLE_CAP`. An exhausted or floor-infeasible bank becomes terminal
`UNDETERMINABLE_BANK`. No domain can receive a 61st question, so K=7 sessions
are bounded by 420 questions as a consequence of seven 60-question caps; there
is no separate 420 constant in the runtime.

Selection remains the shipped `total_var` A-optimal objective. Candidate
filtering reserves items whose band supply equals an outstanding floor deficit,
and when content spread is the only completion blocker it exposes only unmet
bands. The consecutive-domain rule may use a complete non-terminal domain only
while it remains below 60; otherwise the rule relaxes for the lone active
domain.

The frozen local Precision session profile is pinned to the deterministic
uncertainty-aware coarse-to-fine candidate scan with `n_subsample=128`. The
exhaustive selector remains the AD6/reference default outside that explicit
factory. The plain coarse scan is not an allowed profile variant, and no
bitwise-exact equivalence claim is made between approximate and exhaustive
selection at the 35,193-segment bank scale.

The stopping layer never accepts or loads a cut. The downstream
`classify_determined_interval_against_cut` helper first enforces a determined
status, then compares the realized final interval endpoints with a cut and
reports `ABOVE_CUT`, `BELOW_CUT`, or `INDETERMINATE_AT_CUT`.

## Reliability guard and qualification boundary

Two explicit modes exist:

- `surrogate`: ESS plus configurable rejuvenation-acceptance and
  distinct-ancestor floors, with two-evaluation persistence;
- `quantile_mcse`: ESS plus
  `radius + z * inflation * MCSE(radius) <= r_l` using a covariance-aware
  influence-function estimator for the mean and interval endpoints. Acceptance
  and ancestry remain
  diagnostics in this qualified mode and gate only the surrogate fallback.

The radius-specific replicate-cloud study qualified `quantile_mcse` with
z=1.645, inflation 1.5962415321, and effective multiplier 2.6258173203 over
6,048 evaluations. ESS=0.50 and M=2 remain fixed. The verified 840-trajectory
served-bank frontier and 100-reader/three-arm composed sweep selected the
profile above with `m=3`; see `calibration/precision_frontier/`,
`calibration/PRECISION_FRONTIER_PROTOCOL.md`, and
`calibration/precision_frontier/COMPOSED_SWEEP_DECISION.md`. The old
half-width profile is retained only as a marked, superseded audit artifact.

The selected composed arm determined 93.29% of 700 reader-domains, with 95.56%
skill and 95.10% bias interval coverage among determined domains, median/p95
session burden 193/297.25, 47 `UNDETERMINABLE_CAP` outcomes, no BANK or
execution failure, and an observed maximum of 366 session questions. Every CAP
was individually audited; none was caused by evidence, content, or bank supply.

## Accepted D-6 tail-coverage limitation

A 270-session stopping-time screen used 10 registered development response
replicates in each of the 27 factorial cells under the unchanged selected arm.
It determined 1,724/1,890 reader-domains. Skill coverage among determined
domains was 94.84% overall, 93.57% for `|l_true| >= 1`, and 96.18% below 1.
Fifteen cell-domain estimates with at least eight determined replicates fell
below the descriptive 85% screen floor; the clearest were IIC 3/9 at true skill
+2.5, spike 6/10 at -2.5, and LPD 7/10 at +0.75. Cut calls were correct in
746/749 decidable cases, with zero false PASS and three false FAIL outcomes.

This is an interval-calibration limitation, not evidence of observed false
certification. The abandoned repair evidence remains historical. The frozen
factory does not accept calibration controls and asserts that the operative
scale/ramp/bias fields are `None`; there is no frozen `g` parameter.

The generic policy class reports raw equal-tailed intervals and computes the
point-centred stopping radius separately. The factory pins
`reliability_mode="quantile_mcse"`,
`precision_statistic="point_centered_radius"`, and the complete controller
profile described in `../POLICY_LAYOUT.md`.

Implementation is not production promotion. TypeScript porting, storage/UI
migration, rollout, and production sign-off remain separately approved steps.
