# Does the F1 work beat the PROD INCUMBENT? Equal-burden study (2026-08-01)

Every earlier comparison in this program used binary as the reference, which
is not what production serves. This study compares the three n-way variants
against each other and against binary, all at the same 90-question burden,
192 CRN-matched replicates, production profile (1200p/30MH), formal SBC,
truth β drawn per replicate from the fitted reader population
(LogNormal(−0.020575, 0.255378)), truth λ_d = 0.

The prod-incumbent arm is the deployed nine draws exactly: the crossfit
quantile-spread was verified equal bit-for-bit to the engine artifact
production stamps. CRN integrity check: the binary arm's skill RMSE is
0.538250 in all four runs.

| arm | skill RMSE | bias RMSE | skill cov [Wilson 95] | bias cov | skill width |
|---|---|---|---|---|---|
| **prod incumbent** (unfloored mixture) | 0.3611 | 0.3984 | **0.8811** [0.861, 0.899] | 0.9557 | 1.292 |
| Phase-2 floored mixture | 0.3656 | 0.3928 | 0.9253 [0.909, 0.939] | 0.9392 | 1.425 |
| **draw-latent atoms17 (B)** | **0.3563** | **0.3923** | **0.9505** [0.936, 0.962] | 0.9566 | 1.459 |
| draw-latent atoms33 (B) | 0.3566 | 0.3928 | 0.9523 [0.938, 0.963] | 0.9523 | 1.464 |
| binary (reference) | 0.5382 | 0.4872 | 0.9488 [0.934, 0.960] | 0.9444 | 2.379 |

## 1. The n-way premise holds decisively

Every n-way arm cuts skill RMSE by ~34% versus binary (0.356–0.366 vs 0.538)
and bias RMSE by ~19%, at identical burden. Nothing in this program disputes
the reason production runs n-way.

## 2. Versus the incumbent, B's accuracy gain is marginal — that is not the point

Skill RMSE 0.3563 vs 0.3611 (−1.3%); bias RMSE 0.3923 vs 0.3984 (−1.5%).
Treat these as a wash. **The gain is calibration**: the incumbent's 95%
skill intervals cover 88.1% of the time, B's cover 95.1%. The incumbent's
Wilson interval excludes nominal by a wide margin; B's contains it.

## 3. The incumbent's tighter intervals are unearned

The incumbent reports the *narrowest* skill intervals of any arm (1.292 vs
B's 1.459) — but only because they are too narrow to be true. Rescaling each
arm's interval to the width it would need for honest 95% coverage
(approximate: normal marginal, symmetric miscoverage):

| arm | coverage | raw width | honest width |
|---|---|---|---|
| prod incumbent | 0.8811 | 1.292 | **1.625** |
| floored mixture | 0.9253 | 1.425 | 1.567 |
| draw-latent B | 0.9505 | 1.459 | **1.455** |
| binary | 0.9488 | 2.379 | 2.392 |

**At matched honesty B is 10.4% tighter than the prod incumbent** and 39.1%
tighter than binary. The incumbent's apparent tightness is a 12-point
coverage deficit wearing a narrow interval.

## 4. Implication for burden (the questions-to-stop question)

Precision stopping halts when the interval reaches a width target, so an
engine with unearned narrow intervals reaches that target *sooner*. Expect
the incumbent to post the lowest raw question count, and expect B to need
somewhat more — that difference is the price of honest intervals, not
inefficiency. The like-for-like statement is the honest-width row above:
at equal *real* precision B needs less information than the incumbent, so
against an honestly-calibrated target B is the cheaper engine. The four-arm
Precision-stopping study measures the raw numbers directly.

## 5. Scope limit that matters more than any row above

This study plants truth from the same conditional-F1 form the engines
assume, so every arm is evaluated under correct specification. The
2026-08-01 production finding (`PROD_ENGINE_STATE_FINDINGS.md`) shows the
real serving population violates that assumption — the conditional was fit
where asked = gold, while production asks the most informative class. None
of the numbers here transfer to production until that mismatch is resolved;
they compare engines, not engines-on-real-responses.
