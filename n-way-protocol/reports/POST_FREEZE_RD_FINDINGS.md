# Post-freeze n-way R&D findings

Status: **research-only; no production promotion authorized**.

Run date: 2026-07-20. All implementation, staged inputs, and generated reports
are contained in `n-way-protocol`. Production `cortex_web` was not changed.
Qualification-scale runs used 1,200 particles, 30 MH steps, up to 46 worker
processes, one numerical-library thread per worker, two reserved logical CPUs,
and at least 20% memory headroom.

## Recovery point

The pre-iteration F1 implementation is recoverable from
`frozen/F1_1200P30_20260720.tar.gz`. Its SHA-256 is
`015068f250827bcfa42b8ec3d923b473591079ffd60f12674d0ae884267c4a52`.
The archive is read-only, has an independently verified 52-file digest
manifest, and passed 26 TypeScript tests, 9 Python tests, and the 35,000-bank
benchmark before freezing. Nothing in this R&D series overwrites that archive.

## Decision

Retain the following combined challenger for the next governed qualification:

1. The frozen conditional-F1 n-way likelihood and 1,200-particle/30-MH
   numerical profile.
2. A leakage-reduced, uncertainty-propagating nine-draw artifact ensemble.
3. A Fisher-augmented shortlist followed by the unchanged exact six-outcome
   total-variance final decision.

This is a candidate, not a promoted profile. It has the best supported
precision/coverage tradeoff of the tested variants, but the absolute bias
coverage result and the remaining governed gates prohibit production use.

## Powered combined result

`selector_fisher_artifact_ensemble9_1200p30_2000.json` contains 2,000 disjoint
formal-SBC seeds and 4,000 complete arm rows. Both arms used the same nine-draw
artifact mixture; the only difference was shortlist augmentation. Each session
asked four direct questions in each of the six IIIC domains (24 total).

| Metric | Frozen selector | Fisher-augmented | Paired/aggregate change |
|---|---:|---:|---:|
| Skill coverage | 0.94783 | 0.94933 | +0.00150, 95% CI `[-0.00351, 0.00651]` |
| Bias coverage | 0.94275 | 0.94542 | +0.00267, 95% CI `[-0.00235, 0.00768]` |
| Skill width | 2.42103 | 2.21755 | 8.17% lower paired mean ratio |
| Bias width | 2.41221 | 2.34771 | 2.30% lower paired mean ratio |
| Skill RMSE | 0.58949 | 0.53696 | 8.91% lower ratio of aggregate means |
| Bias RMSE | 0.61099 | 0.59532 | 2.56% lower ratio of aggregate means |

The 95% intervals for the paired width ratios were `[0.91547, 0.92119]` for
skill and `[0.97278, 0.98129]` for bias. Both paired coverage lower bounds pass
the development noninferiority margin of -0.03.

The RMSE evidence is weaker than the aggregate ratios alone suggest. The mean
of per-seed ratios was 0.99049 for skill (95% CI `[0.97117, 1.00982]`) and
1.04595 for bias (`[1.02746, 1.06444]`), because ratios become unstable when a
seed's frozen-arm RMSE is small. Therefore this study supports interval
tightening and coverage retention, but does not claim paired RMSE superiority.

The Fisher arm's absolute skill coverage was 0.94933 (95% CI
`[0.94532, 0.95335]`). Absolute bias coverage was 0.94542
(`[0.94118, 0.94965]`). The latter is slightly below nominal 95% coverage and
its interval barely excludes 0.95. No absolute-coverage margin was
preregistered for this development run. This is a mandatory qualification
issue, not a result to tune away after observing it.

## Artifact refit and uncertainty propagation

The staged response panel contains 68,783 wrong IIIC picks from 685 readers
and 4,957 external cases. Expert responses use leave-one-expert-out plurality
gold; novice responses use the expert plurality. Five-fold reader-held-out and
external-case-held-out panels were fit separately because the crossed data form
a connected graph that prevents simultaneous independent partitioning.

- Full fit: beta 1.03032, distractor lapse 0.
- Reader-held-out beta range: 0.99006–1.06589; one fold estimated lapse
  0.01222.
- External-case-held-out beta range: 1.02785–1.03773; all lapses 0.
- Reader-clustered 200-draw intervals: beta `[0.91538, 1.13698]`, lapse
  `[0, 0.01258]`.

The plug-in artifact narrowly passed the central 192-replicate relative
coverage gate, but failed the skill gate under the low-beta/high-lapse stress:
the lower confidence bound was -0.04632. The nine-draw ensemble passed the same
stress with a skill lower bound of -0.02958 and retained 30.75% skill and
13.70% bias tightening versus binary. It also passed the high-beta stress.
This supports propagating artifact uncertainty rather than treating the fitted
parameters as exact.

The artifact is still `promotionForbidden: true`. External case is the
available source grouping; patient-level grouping and DR07 qualification have
not been established.

## Selector evidence

On 24 posterior states with 420 deterministically sampled eligible empirical
axis segments per state, exhaustive exact loss was used as the reference:

- Exact-best recall improved from 0.4583 to 0.8750.
- Mean regret fell from 0.01516 to 0.001495, a 90.1% reduction.
- p95 regret fell from 0.04940 to 0.01107.
- Mean shortlist size increased from 422.08 to 454.21.

The TypeScript audit independently showed the same direction, and the
35,000-segment benchmark remained operational. The production-shaped baseline
took about 1.25 seconds on this host; Fisher augmentation took about 2.25
seconds. These are host measurements, not browser/device latency
qualification. The empirical-axis audit samples an eligible research bank; it
is not a full exhaustive audit of the eventually served bank.

Bias-weighted total variance and response-particle mutual information were
retained as tested controls. Their 192-replicate results did not justify
advancing them over the simpler Fisher-augmented exact-total-variance design.

## SMC and Monte Carlo findings

Twelve fixed categorical histories compared 12 repeats at 1,200/30 with four
repeats at 4,800/60. Candidate/reference posterior-radius ratios were 0.98586
for skill and 0.98497 for bias. Thus the production numerical profile was only
about 1.4–1.5% narrower than the higher-fidelity reference in this panel.

A particle-bootstrap plug-in MCSE did not capture SMC dependence: validation
required q95 radius-inflation factors of 2.85 for skill and 2.76 for bias.
Those factors must not be applied as posterior interval multipliers; they
diagnose an invalid naive MCSE estimator. No interval guard reduction or
profile reduction is supported.

Forty-eight histories with four repeats per profile compared frozen
multinomial resampling, stratified, residual, and a scrambled-Sobol/PCA-sorted
hybrid. Width ratios stayed within about 0.0–0.33% of frozen, with no meaningful
RMSE advantage. The hybrid is a research precursor, not full SQMC, and its
reported timing excluded initialization and was measured under concurrent
load. None of these SMC variants advances.

## Bank-axis audit

The leakage-reduced eligibility rule produced 20,502 segments: at least 10
IIIC votes and all six IIIC signal means/SDs finite. On a deterministic
4,000-segment sample, total-information quantiles were 0.32379, 0.94627,
1.97647, and 2.92891 at q10/q50/q90/q99. This supports information-aware
screening because bank heterogeneity is large.

Artificially reducing signal SD by 25% increased median local information by
12.75%, but an end-to-end synthetic run changed interval width by only about
0.3%. No bank pruning, evidence-axis rewrite, or claimed precision gain is
authorized until the real served-bank operating characteristics are evaluated
with content, exposure, source, and coverage constraints.

## Promotion gates still open

- New versioned DR07 fit and governed rank/conditioning gate.
- Patient/source leakage control beyond the available external-case grouping.
- Preregistered absolute skill and bias coverage margins and a locked powered
  qualification; bias coverage deserves specific attention.
- Full served-bank adaptive operation under the unchanged Precision stopping
  rule, including pass/fail/refer/cap and burden metrics.
- Source-shift, lapse/scale drift, correlated-criterion, and evidence-axis
  misspecification panels beyond the limited artifact stresses here.
- TypeScript/Python complete-trajectory parity for the artifact ensemble and
  Fisher shortlist when wired into the production-shaped state engine.
- Exact served-bank selector regret plus browser/device p50/p95/max latency,
  memory, worker failure, and serial/speculative parity.
- Binary-session golden, resume, rollback, API, persistence, and rollout gates.

Until these pass and the owner gives a separate green light, the production
binary protocol remains authoritative.

## Final verification

After all post-freeze changes:

- TypeScript strict type-check passed.
- Standard suite passed 32 TypeScript tests; three opt-in tests were skipped by
  design in that command.
- Python suite passed 29 tests.
- Both opt-in 35,000-segment performance tests passed: 1.37 seconds for the
  frozen shortlist and 2.19 seconds for Fisher augmentation on this host.
- The opt-in TypeScript selector-regret audit passed.
- The frozen archive digest and per-file recovery manifest were rechecked.
- Every retained JSON report parsed, and the powered combined report contained
  exactly 2,000 unique seeds with one complete row per arm.
- `POST_FREEZE_RD_FILES.sha256` records the final digests for the retained
  implementation, tests, contracts, artifacts, and reports.

## Integrated follow-up

The retained artifact-ensemble/Fisher candidate was subsequently wired through
the isolated production-shaped likelihood, update, replay, selector, and
profile paths. Its final 2,000-seed frozen-versus-integrated comparison is in
`FINAL_INTEGRATION_COMPARISON.md`. It passed relative coverage noninferiority
and produced an additional 8.04% skill and 2.58% bias interval tightening.
Absolute coverage remained an open gate; no production promotion followed.
