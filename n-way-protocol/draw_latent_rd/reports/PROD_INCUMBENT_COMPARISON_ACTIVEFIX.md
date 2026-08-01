# Four-arm head-to-head under the production-faithful selector (2026-08-01)

Phase-1b re-run of `PROD_INCUMBENT_COMPARISON.md` after fixing the harness
fidelity gap that report disclosed: the research selector offered every
under-cap domain until the safety cap, where production restricts selection
to domains the Precision policy reports ACTIVE (`advance.ts:283,399`) with
the session.ts:71 variety semantics. `_precision_eligible` now mirrors
production exactly (tested); everything else — 96 CRN seeds at base
66,000,000, production profile, real served bank, unchanged Precision
stopping, continuum truth world — is identical to the previous run, so the
two runs isolate the selector fix. Results:
`prod_incumbent_head_to_head_activefix.json` (wall clock 1,494 s).

## Headline (was: cap-dominated ~351–354 questions in every arm)

| arm | mean questions [95% CI] | median | p90 | skill RMSE | bias RMSE | skill cov [Wilson95] | bias cov | skill width |
|---|---|---|---|---|---|---|---|---|
| binary | 292.05 [284.54, 299.56] | 292.5 | 337.0 | 0.4101 | 0.3473 | 0.9531 [0.933, 0.968] **at nominal** | 0.9479 | 1.7911 |
| **prod incumbent** | 256.70 [248.50, 264.89] | 253.5 | 305.5 | 0.3472 | 0.3227 | **0.8872 [0.859, 0.910] BELOW nominal** | 0.9462 | 1.2369 |
| floored ensemble9 | 249.76 [241.91, 257.61] | 245.5 | 298.5 | 0.3612 | 0.3074 | **0.9045 [0.878, 0.926] BELOW nominal** | 0.9288 | 1.3492 |
| **draw-latent atoms17** | **234.95** [227.29, 242.61] | 236.5 | 285.0 | 0.3277 | 0.3049 | **0.9583 [0.939, 0.972] at nominal** | 0.9531 | 1.4297 |

Terminal states over 576 domains: draw-latent certifies 435 as
`ESTIMATE_COMPLETE` (vs incumbent 354, floored 400, binary 272); zero
draw-latent sessions spent the full cap budget (incumbent 1, binary 4).

## The burden conclusion, restated (the Phase-1b question)

The previous report's open caveat was that a production-faithful selector
might let the incumbent "convert its 12% narrower intervals into genuinely
earlier stopping." The corrected measurement answers it: **the incumbent
does stop earlier than binary (−35.4 questions), but the draw-latent engine
stops earlier still — 21.75 questions (−8.5%) below the incumbent, paired
CI [−30.0, −13.5] excluding zero — while being the only n-way arm with
honest coverage (0.9583, Wilson containing nominal).** B stops no later
than the incumbent; it stops materially sooner. At matched honesty the
comparison is binary 292.1 vs draw-latent 234.9: the honest price of a
session fell by 57.1 questions (−19.6%) and the cheapest honest arm is
draw-latent.

The apparent paradox — draw-latent's intervals are wider (+0.193) yet it
stops sooner — is the selector fix at work: honest per-session inference
(each sitting resolving its own sharpness atom) reaches the policy's
precision target on more domains, and the ACTIVE-only selector stops
spending questions on domains that are already done. Overconfident narrow
intervals no longer buy earlier stopping; they buy re-opened domains and
wasted questions.

## Phase-1d: the bias-RMSE regression is explained and gone (non-blocking)

The +0.0348 [+0.0074, +0.0621] draw-latent bias-RMSE regression measured in
the cap-dominated run does not survive the selector fix: under the
production-faithful regime the paired difference is **−0.0178
[−0.0502, +0.0147]** (draw-latent 0.3049 vs incumbent 0.3227; point
estimate now *favors* draw-latent, CI spans zero). The regression was a
property of the non-production questioning regime — with every arm forced
to spend ~360 questions, the extra post-completion questions interacted
with the mixture's overconfident skill posterior differently than with the
draw-latent joint posterior — not a property of the engine production
would run. Per the owner's decision this item is measured and documented,
not gated; no unexplained regression ships in a reported parameter block.

## Paired contrasts (draw-latent − incumbent, n = 96)

| metric | mean | 95% CI |
|---|---|---|
| questions | **−21.75** | [−30.00, −13.50] |
| skill RMSE | −0.0195 | [−0.0490, +0.0100] |
| bias RMSE | −0.0178 | [−0.0502, +0.0147] |
| skill coverage | **+0.0712** | [+0.0388, +0.1035] |
| bias coverage | +0.0069 | [−0.0162, +0.0301] |
| skill width | +0.1928 | [+0.1654, +0.2202] |

Reference contrasts vs binary: draw-latent −57.10 questions
[−65.67, −48.54] with coverage +0.0052 [−0.0201, +0.0306]; the incumbent's
skill-coverage deficit vs binary is −0.0660 [−0.1010, −0.0309].

## Scope

Research-only, unqualified artifacts, atoms17 arm (the head-to-head's
draw-latent arm; the locked campaign qualifies the nesting34 promotion
grid). Contrasts are CRN-paired; the selector fix applies identically to
all four arms.
