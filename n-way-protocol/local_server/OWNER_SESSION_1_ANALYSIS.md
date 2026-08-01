# First owner UI session — analysis (2026-08-01)

Session `369df1ab`, real cortex_web SPA, draw-latent atoms17 profile,
167 questions, 14m36s, ran to a genuine Precision stop
(`all_estimated_or_undeterminable`), all seven domains `DETERMINED`.

## What this session is

A fast click-through, not a skill measurement: median reaction time 0.5–0.7 s,
132 of 167 answers under one second, overall accuracy 0.317 with the six IIIC
domains at 0.15–0.39 against a 1/6 chance floor (spike, a binary task, sat at
0.815 against a 1/2 floor). Wrong picks are spread near-uniformly across
classes. The skill estimates from this sitting therefore measure the
machinery, not the reader.

## Machinery verdict: clean

Full production path exercised end-to-end — real SPA, real bundle, real API,
worker architecture, unchanged Precision stopping. Every domain reached
`ESTIMATE_COMPLETE`, final ESS 818/1200, 167 trials persisted with reaction
times, client stamps, and complete per-trial diagnostics.

## The headline finding: the safety layer fired on real data

Replaying the reference misspecification monitor over the real picks
(floor-0.15 reference, threshold 5.213, raw `s_mean` evidence frame as the
calibration uses):

**Tripped at wrong-pick 15** (statistic 6.24). The calibration predicted a
median of 14 wrong-picks to trip under a uniform-distractor world; this
session tripped at 15.

> **Corrections (2026-08-01).** The first version of this file (a) computed
> the evidence in the skill-scaled z frame while `misspec_monitor.load_axes`
> calibrates on raw `s_mean` — corrected above, the trip index is unchanged
> and the statistic moves 6.04 → 6.24; and (b) stated that "in production a
> trip replays binary" and that the binary engine is what production serves.
> **Both are false.** Production has served the categorical n-way engine
> since 2026-07-20, and the misspecification monitor is not deployed there at
> all. See `reports/PROD_ENGINE_STATE_FINDINGS.md`.

## Fixed-sequence three-engine replay (same items, same answers)

| engine | mean 95% skill-interval width |
|---|---|
| binary (the AD6 fallback, not what prod serves) | 1.929 |
| static mixture floor015 | 1.580 |
| draw-latent atoms17 (new) | 1.639 |

Note the deployed production engine is a *fourth* configuration not replayed
here — the **unfloored** ensemble9 mixture — which is strictly more
aggressive than the floor015 arm above.

Both categorical arms produce ~15–18% narrower intervals from identical
responses — but for THIS session that narrowing is not a quality claim: the
monitor says the response model is misspecified here, and unwarranted
narrowing is precisely the overconfidence the monitor guards. The comparison
is also information-per-response only: item selection was adaptive under
draw-latent, so binary would have asked different items and stopped elsewhere.
A true head-to-head is the locked campaign, not a replay.

Draw-latent atom posterior ended with mean β 0.714 on 6 surviving atoms — the
engine correctly inferring "this responder's wrong picks are diffuse" and
migrating toward the low-β atoms. Partly legitimate concentration, partly the
known atom-depletion effect at 70 particles per atom; it reinforces the
particle-per-atom floor as a locked-campaign design item.

## What a useful next session looks like

A genuinely attempted reading sitting — answering deliberately, even if
uncertain — is what exercises the calibration claim. Two things to watch: the
monitor should NOT trip (its 1% false-trip calibration applies to
good-faith responding), and the atom posterior should settle somewhere
interpretable rather than collapsing to the low-β floor.
