# Phase-1c real-response arbitration: draw-latent beats the incumbent (2026-08-01)

The hard kill gate of the promotion plan. All 13 real production n-way
sessions containing IIIC trials (1,483 categorical picks; extract limited to
session_id/trial_index/seg_id/task_k/pick/is_correct, no participant data)
replayed one-step-ahead through five engines under the production profile
(1,200 particles, ESS 0.5, 30 MH) and the SERVED manifest corrL/corrT prior
blocks. Score: prequential log predictive of the observed pick — before each
trial the engine's filtered posterior prices the pick that then arrives.
Axes are the RAW `s_mean`/`s_sd` staged bank frame (the frame the artifact
was fitted in — the misspec-monitor evidence-frame rule; no skill scaling).
Runner: `real_response_arbitration.py`, seed base 68,000,000; results
`real_response_arbitration.json`.

## Headline: total log predictive on real picks (higher = describes them better)

| arm | total | mean/trial |
|---|---|---|
| binary (+uniform distractors) | −2742.39 | −1.8492 |
| **prod incumbent** (unfloored mixture) | −2588.08 | −1.7452 |
| floored ensemble9 | −2582.63 | −1.7415 |
| draw-latent atoms17 | −2577.21 | −1.7378 |
| **draw-latent nesting34 (ship candidate)** | **−2572.62** | **−1.7347** |

Paired per-session contrasts (13 sessions, 10k bootstrap over sessions):

| contrast | total Δ | bootstrap 95% | sessions won |
|---|---|---|---|
| **nesting34 − incumbent (GATE)** | **+15.46** | [−1.61, +34.16] | **8/13** |
| atoms17 − incumbent | +10.87 | [−1.05, +23.11] | 8/13 |
| floored − incumbent | +5.46 | [−3.26, +13.77] | 8/13 |
| nesting34 − binary | **+169.77** | [+80.25, +269.38] | 11/13 |
| binary − incumbent | −154.30 | [−269.77, −49.38] | 2/13 |

## Gate decision

The kill conditions were: (a) draw-latent does not beat the incumbent on
real responses, or (b) B ≈ incumbent while both fall below binary on the
categorical channel (the fit-population mismatch dominating, making a refit
the correct action rather than promotion).

- **(a) fails to trigger**: nesting34 beats the incumbent on the total
  (+15.5 nats), on mean-per-trial, and on 8 of 13 sessions; every design
  step is monotone in the right direction
  (incumbent → floored → atoms17 → nesting34 improves the real-pick score at
  each step, entirely on the wrong-pick channel: −2294.3 → −2279.8 →
  −2268.9 → −2258.7).
- **(b) is refuted with a wide margin**: every n-way arm beats binary by
  ~150–170 nats with a bootstrap CI excluding zero (nesting34 − binary
  +169.8 [+80.3, +269.4], 11/13 sessions). The categorical channel carries
  real information about the real response population; the mismatch does
  not dominate.

**Verdict: PASS — promotion is defensible on real responses.** Honest
caveat: the gate contrast's own session-level bootstrap CI ([−1.6, +34.2])
does not exclude zero at n = 13 sessions; the direction is supported by the
point estimate, the session majority, the per-trial mean, and the monotone
arm ordering, not by a significant paired interval. The asked≠gold refit of
the conditional (option 3 of PROD_ENGINE_STATE_FINDINGS.md) remains the
right *next* fit-lineage R&D, but on this evidence it is an improvement
opportunity, not a promotion blocker.

## Why draw-latent wins on real data — the mechanism, visible per session

- Session `0d815d8b…` (27 trials; the owner sitting whose picks tripped the
  reference monitor) puts **0.984 posterior mass on the λ_d = 1
  binary-nesting atom**: the cloud identified a diffuse responder and
  effectively became binary mid-session, gaining +7.0 nats over the
  incumbent, which kept applying the fitted conditional. This is the
  graceful-degradation claim of construction B observed on production data.
- The other 12 sessions leave the nesting atom at ~0 mass and concentrate
  on per-session sharpness atoms with MAP β from 0.45 to 1.39 — the
  draw-latent cloud resolves *individual* distractor sharpness instead of
  averaging the population mixture over every observation.
- Real picks match the asked class in only 20.97% of trials (matching the
  production finding); the arms differ almost entirely on the 79% wrong-pick
  channel, which is exactly the channel the misspecification monitor flagged.

## Scope

Research-only evidence over unqualified artifacts; no production surface
touched. Fixed-sequence historical replay (QUALIFICATION.md family 4): it
checks the response model against real responses and does not qualify
adaptive selection — the locked campaign (families 1–3) does that.
