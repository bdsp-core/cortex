# What the production server is actually running (2026-08-01)

Investigation prompted by the owner's recollection that production already
runs an n-way engine. **The owner is correct; earlier statements in this
program's reporting that "binary is the incumbent, shipped and serving" were
wrong.** Evidence and consequences below.

## 1. Production has served the categorical n-way engine since 2026-07-20

- `/etc/cortex/cortex.env` on the prod box: `CORTEX_PRECISION_POLICY_ROLLOUT=all`.
- `routers/testing.py` stamps `drawn["nwayProfile"] = production_nway_profile(...)`
  for **every** `precision_v1` sitting. There is no separate response-model
  rollout gate in the deployed code (the `nway_response_rollout` flag
  described in `PHASE2_REPORT.md` does not exist in this codebase).
- Prod database: 18 sessions carry a non-null `nway_profile`, all stamped
  `precision_nway_f1_ensemble9_fisher_v1`, from 2026-07-20T23:20Z to
  2026-07-27T13:44Z — 1,483 `categorical_f1` trials and 372 `binary` trials
  (the binary rows are the spike domain, which is binary in both arms).
- Exposure: 7 accounts — **6 external participants (8 sittings, 3 completed)**
  plus one owner/test account (10 sittings, 9 completed).

The owner's second recollection also checks out: the 2026-07-21 latency work
(`Compact selector worker messages`, `Reuse MH worker history buffers`, `Keep
calibrated worker pool fixed per session`) was the n-way engine optimization.

## 2. The deployed configuration is the pre-Phase-2 one

Release stamp `285f264a…` dated 2026-07-25, so it predates the 2026-07-30
Phase-2 commits. Deployed n-way is therefore:

| layer | production today | on `main` (undeployed) |
|---|---|---|
| artifact | `iiic-f1-crossfit-ensemble9-rd-20260720`, **unfloored** (λ_d 0.0–0.019) | floor-0.15 |
| aggregation | static per-observation mixture | static mixture (draw-latent on the F1 branch) |
| misspecification monitor | **absent** | ported 2026-07-30 |

That combination — unfloored artifact, no monitor — is exactly the
configuration Phase-1 gate A3 measured at **skill coverage 0.28** under a
uniform-pick world (vs binary's 0.957), the case the floor and monitor were
designed to bound.

## 3. Every real n-way sitting trips the misspecification monitor

Reference monitor replayed over real prod picks (deployed unfloored artifact
as reference, its own calibrated threshold 7.410, raw `s_mean` frame):

**13 of 13 sessions containing IIIC trials trip** — median trip index 22
wrong-picks (range 8–84), final statistics 7.6–9.8. Per-session "answered the
asked class" rates run 0.11–0.40.

## 4. Root cause: the response model is applied out of its fitted domain

The artifact was fit on a corpus where the asked class **is** the item's gold
class (`stage_real_artifact.py` sets `asked_k = CLASS_INDEX[gold]`), so every
"wrong pick" there is a genuine confusion among near-miss classes. Production
asks whichever class is most informative, which frequently is *not* the item's
gold class. In the prod trials:

| | gold-correct | gold-wrong |
|---|---|---|
| picked the asked class | 9.0% | 11.9% |
| did **not** pick the asked class | **22.8%** | 56.2% |

**28.8% of the responses the model feeds to its distractor term are
gold-correct answers** — a participant correctly naming the item's true class,
which the model scores as a confusion. The fitted conditional simply does not
describe this response population, which is what the monitor is detecting.

This also explains the low "accuracy" figures: `pick == asked_k` is a model
construct, not a competence measure. Gold-derived `is_correct` is 0.318 on the
same trials — the two agree only 65.3% of the time.

## 5. Consequence and options

The skill posterior's *asked-class margin* term is unaffected; the harm is
confined to the cross-domain distractor channel, which is currently
contributing information derived from a likelihood that does not match the
responses. Phase-1 A3 bounds the worst case at coverage 0.28 (unfloored,
fully uniform picks); real prod picks are diffuse relative to the model but
not literally uniform, so the true figure lies between that and nominal and
**cannot be pinned without ground-truth skill** for those participants.
Three completed external sittings are affected.

Options, in increasing order of intervention:

1. **Deploy the Phase-2 floor + monitor** (built, tested, on `main`, never
   deployed): bounds per-update damage and fails closed to binary-reduced
   replay when a session's picks go diffuse. Smallest change; would have
   caught all 13.
2. **Roll n-way back to AD6/binary in prod** (`CORTEX_PRECISION_POLICY_ROLLOUT`
   is the documented switch, "set off and restart for AD6 rollback") until a
   qualified engine lands.
3. **Re-fit the conditional on asked≠gold responses** — the scientifically
   correct fix, since it targets the actual serving population. This is new
   R&D and belongs in the locked-campaign design, not in an incident response.

The draw-latent (construction B) work is orthogonal to this finding and does
not resolve it on its own: fixing *how* the ensemble is aggregated does not
fix a conditional fit on the wrong response population.
