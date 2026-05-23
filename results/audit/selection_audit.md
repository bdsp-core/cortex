# CORTEX — adaptive-selection audit
- Engine: SMC + MCMC particle cloud, K=6 IIIC tasks, N=600 particles
- Bank: 100 IIIC segments (each served at most once)
- Arms: **adaptive** (`choose_item`, expected-variance minimiser) vs **random** (uniform null baseline)
- Panel: 13 simulated raters of known (t, l)

## Headline — does the native delta=0.05 stop fire?

0 of 13 raters reach AUROC half-width < 0.05 within the 100-segment bank under adaptive selection. A 100-segment IIIC bank spread over 6 tasks is too small to pin every task's AUROC to +/-0.05; the test bank-exhausts first. See recommendation below.

## Questions to reach an AUROC half-width target

`A` = adaptive, `R` = random; `-` = target not reached within the bank.

| Rater | arm | hw<0.2 | hw<0.15 | hw<0.12 | hw<0.1 | hw<0.08 | hw<0.05 |
|---|---|---|---|---|---|---|---|
| real:M. Brandon Westover | A | 1 | 59 | - | - | - | - |
| real:M. Brandon Westover | R | 1 | 67 | - | - | - | - |
| real:Aaron F. Struck | A | 1 | 24 | 43 | - | - | - |
| real:Aaron F. Struck | R | 1 | 42 | 67 | - | - | - |
| real:Marcus Ng | A | 1 | 49 | - | - | - | - |
| real:Marcus Ng | R | 1 | 81 | - | - | - | - |
| real:Aline Herlopian | A | 1 | 22 | - | - | - | - |
| real:Aline Herlopian | R | 1 | - | - | - | - | - |
| synth:fail-biasLow | A | 1 | 32 | - | - | - | - |
| synth:fail-biasLow | R | 1 | 89 | - | - | - | - |
| synth:fail-biasNeutral | A | 1 | 36 | 77 | - | - | - |
| synth:fail-biasNeutral | R | 1 | 17 | 40 | - | - | - |
| synth:fail-biasHigh | A | 1 | 87 | - | - | - | - |
| synth:fail-biasHigh | R | 1 | 91 | - | - | - | - |
| synth:borderline-biasLow | A | 1 | 51 | 76 | - | - | - |
| synth:borderline-biasLow | R | 1 | 15 | 47 | 74 | - | - |
| synth:borderline-biasNeutral | A | 1 | 21 | 44 | 63 | - | - |
| synth:borderline-biasNeutral | R | 1 | - | - | - | - | - |
| synth:borderline-biasHigh | A | 1 | 53 | - | - | - | - |
| synth:borderline-biasHigh | R | 1 | - | - | - | - | - |
| synth:pass-biasLow | A | 1 | 44 | 77 | 94 | - | - |
| synth:pass-biasLow | R | 1 | 29 | 41 | - | - | - |
| synth:pass-biasNeutral | A | 1 | 23 | 50 | 70 | 92 | - |
| synth:pass-biasNeutral | R | 1 | 21 | 61 | - | - | - |
| synth:pass-biasHigh | A | 1 | 53 | 78 | - | - | - |
| synth:pass-biasHigh | R | 1 | 96 | - | - | - | - |

## Aggregate — mean questions to target (lower is better)

| Target | adaptive | random | adaptive advantage |
|---|---|---|---|
| hw<0.2 | 1.0 (n=13) | 1.0 (n=13) | 1.00x fewer |
| hw<0.15 | 42.6 (n=13) | 54.8 (n=10) | 1.29x fewer |
| hw<0.12 | 63.6 (n=7) | 51.2 (n=5) | 0.81x fewer |
| hw<0.1 | 75.7 (n=3) | 74.0 (n=1) | 0.98x fewer |
| hw<0.08 | 92.0 (n=1) | - (0) | - |
| hw<0.05 | - (0) | - (0) | - |

## Final precision + collapse toward known truth

At matched questions (bank exhaustion), adaptive should reach lower posterior variance and estimates closer to truth.

| Rater | arm | n_q | final hw | total var | l RMSE | t RMSE |
|---|---|---|---|---|---|---|
| real:M. Brandon Westover | A | 100 | 0.125 | 3.419 | 0.326 | 0.385 |
| real:M. Brandon Westover | R | 100 | 0.165 | 4.227 | 0.219 | 0.172 |
| real:Aaron F. Struck | A | 100 | 0.107 | 3.322 | 0.275 | 0.169 |
| real:Aaron F. Struck | R | 100 | 0.103 | 3.941 | 0.361 | 0.318 |
| real:Marcus Ng | A | 100 | 0.125 | 3.167 | 0.488 | 0.242 |
| real:Marcus Ng | R | 100 | 0.143 | 3.472 | 0.418 | 0.293 |
| real:Aline Herlopian | A | 100 | 0.137 | 3.931 | 0.642 | 0.425 |
| real:Aline Herlopian | R | 100 | 0.158 | 5.369 | 0.282 | 0.534 |
| synth:fail-biasLow | A | 100 | 0.130 | 2.375 | 0.436 | 0.232 |
| synth:fail-biasLow | R | 100 | 0.139 | 2.987 | 0.221 | 0.281 |
| synth:fail-biasNeutral | A | 100 | 0.123 | 2.476 | 0.367 | 0.264 |
| synth:fail-biasNeutral | R | 100 | 0.115 | 3.222 | 0.361 | 0.409 |
| synth:fail-biasHigh | A | 100 | 0.138 | 3.095 | 0.349 | 0.337 |
| synth:fail-biasHigh | R | 100 | 0.158 | 3.876 | 0.354 | 0.316 |
| synth:borderline-biasLow | A | 100 | 0.112 | 2.241 | 0.292 | 0.213 |
| synth:borderline-biasLow | R | 100 | 0.100 | 3.174 | 0.769 | 0.199 |
| synth:borderline-biasNeutral | A | 100 | 0.100 | 2.233 | 0.408 | 0.321 |
| synth:borderline-biasNeutral | R | 100 | 0.160 | 3.045 | 0.399 | 0.218 |
| synth:borderline-biasHigh | A | 100 | 0.134 | 3.012 | 0.285 | 0.363 |
| synth:borderline-biasHigh | R | 100 | 0.158 | 3.608 | 0.512 | 0.407 |
| synth:pass-biasLow | A | 100 | 0.087 | 2.125 | 0.350 | 0.231 |
| synth:pass-biasLow | R | 100 | 0.119 | 2.838 | 0.384 | 0.378 |
| synth:pass-biasNeutral | A | 100 | 0.088 | 2.674 | 0.243 | 0.150 |
| synth:pass-biasNeutral | R | 100 | 0.131 | 3.267 | 0.444 | 0.404 |
| synth:pass-biasHigh | A | 100 | 0.119 | 2.954 | 0.518 | 0.164 |
| synth:pass-biasHigh | R | 100 | 0.140 | 3.450 | 0.587 | 0.349 |

## Summary — adaptive vs random (single replicate per rater)

Count of raters (of 13) where adaptive beats random:

- Final posterior variance: **13/13** (mean 2.848 vs 3.575).
- Final AUROC half-width: **10/13** (mean 0.117 vs 0.138).
- Skill-l RMSE vs truth: **7/13** (mean 0.383 vs 0.409).
- Bias-t RMSE vs truth: **9/13** (mean 0.269 vs 0.329).
- `choose_item` latency: **13.5 ms** per question (no human-perceptible engine pause).

## Findings

1. **The selector computes the correct choice.** `choose_item` provably returns the global expected-posterior-variance argmin over the live candidate pool — verified to 1e-9 in `tests/test_cortex_session_controller.py`.
2. **Realized variance reduction is consistent.** Adaptive selection yields lower total posterior variance than random for 13/13 raters. The AUROC-half-width and RMSE-vs-truth gains are directionally favorable but noisy at one replicate per rater — a rigorous efficiency claim needs a multi-replicate simulation study (`scripts/run_phase1_experiments_v2.py` is built for that).
3. **The native delta=0.05 stop is unreachable.** 0/13 raters reach AUROC half-width < 0.05 within the 100-segment bank; the live test runs to bank exhaustion at ~100 questions.

## Decision needed (PI)

Either (a) accept a fixed ~100-question test (the delta-stop is effectively inert), or (b) relax delta to a bank-deliverable value — the questions-to-target table shows hw<0.15 is reached by every rater (mean ~43 questions) and hw<0.12 by about half. This choice sets the session-length distribution and the consent copy.
