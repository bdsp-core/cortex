# Head-to-head against the PRODUCTION INCUMBENT (2026-07-31)

Every earlier comparison in this program scored the n-way work against the
**binary** arm. Binary is not the incumbent: production has served the
categorical n-way engine with the **unfloored** nine-draw static mixture since
2026-07-20 (`reports/PROD_ENGINE_STATE_FINDINGS.md`). This run scores the new
work against what is actually deployed.

96 CRN-paired replicates, seed base 66,000,000 (disjoint from every prior
grid), production profile (1,200 particles / 30 MH / ESS 0.5), formal SBC,
real served bank `.artifacts/categorical_bank_axes.csv`, and the UNCHANGED
production Precision stopping policy through the `src/precision_cli.ts`
sidecar. Continuum truth world: β ~ LogNormal(μ = −0.020575228248171405,
σ = 0.25537783624586124), truth λ_d = 0. Wall clock 2,112 s on 48 workers.
Machine-readable results: `prod_incumbent_head_to_head.json`.

| arm | artifact | sha256 (12) |
|---|---|---|
| A binary | — (shipping binary arm, CRN reference) | — |
| **B PROD INCUMBENT** | `iiic_conditional_f1_integrated_ensemble9_rd.json` (9 draws, **unfloored**, λ_d 0.0–0.0186) | `6f5ddc6d244c` |
| C floored | `iiic_conditional_f1_integrated_ensemble9_rd_floor015.json` (9 draws, λ_d ≡ 0.15) | `b450095226eb` |
| D draw-latent | `iiic_conditional_f1_engine_frame_atoms17_rd.json` (17 atoms, λ_d ≡ 0.15) | `70dc2c25f314` |

Arms B and C run the existing static-mixture inference path; arm D runs the
draw-latent cloud. All four face the same planted truths, the same bank draw
and the same world β per seed.

## Headline

| arm | mean questions [95% CI] | median | skill RMSE | bias RMSE | skill cov [Wilson95] | bias cov [Wilson95] | skill width | bias width |
|---|---|---|---|---|---|---|---|---|
| A binary | 353.07 [349.18, 356.97] | 360 | 0.3973 | 0.3338 | 0.9444 [0.9226, 0.9604] **at nominal** | 0.9358 [0.9127, 0.9530] at nominal | 1.6269 | 1.2459 |
| **B PROD INCUMBENT** | **354.26** [351.93, 356.59] | 360 | 0.3096 | 0.2711 | **0.8698 [0.8398, 0.8948] BELOW nominal** | 0.9549 [0.9347, 0.9690] at nominal | 1.0581 | 1.0204 |
| C floored ensemble9 | 351.45 [348.48, 354.42] | 360 | 0.3196 | 0.2794 | **0.8924 [0.8644, 0.9151] BELOW nominal** | 0.9444 [0.9226, 0.9604] at nominal | 1.1558 | 1.0064 |
| **D draw-latent atoms17** | **351.04** [347.72, 354.36] | 360 | **0.2871** | 0.3059 | **0.9497 [0.9286, 0.9647] at nominal** | 0.9410 [0.9186, 0.9575] at nominal | 1.2036 | 1.0402 |

RMSE / width interval estimates:

| arm | skill RMSE [95% CI] | bias RMSE [95% CI] | skill width [95% CI] | bias width [95% CI] |
|---|---|---|---|---|
| A binary | 0.3973 [0.3694, 0.4252] | 0.3338 [0.3017, 0.3660] | 1.6269 [1.5919, 1.6619] | 1.2459 [1.1939, 1.2979] |
| B incumbent | 0.3096 [0.2871, 0.3321] | 0.2711 [0.2424, 0.2998] | 1.0581 [1.0287, 1.0875] | 1.0204 [0.9674, 1.0734] |
| C floored | 0.3196 [0.2952, 0.3440] | 0.2794 [0.2493, 0.3095] | 1.1558 [1.1266, 1.1851] | 1.0064 [0.9522, 1.0606] |
| D draw-latent | 0.2871 [0.2649, 0.3093] | 0.3059 [0.2727, 0.3391] | 1.2036 [1.1774, 1.2299] | 1.0402 [0.9899, 1.0905] |

## Per-domain question counts (mean; domain 0 = spike, never asked here)

| arm | spike | sz | lpd | gpd | lrda | grda | iic |
|---|---|---|---|---|---|---|---|
| A binary | 0.00 | 58.26 | 59.23 | 58.64 | 58.85 | 58.68 | 59.42 |
| B incumbent | 0.00 | 59.41 | 59.58 | 59.38 | 59.16 | 58.65 | 58.09 |
| C floored | 0.00 | 58.57 | 59.85 | 59.19 | 58.88 | 56.98 | 57.98 |
| D draw-latent | 0.00 | 57.83 | 59.53 | 59.20 | 58.64 | 57.43 | 58.42 |

Final per-domain policy states over 576 domains (96 sessions × 6):

| arm | ESTIMATE_COMPLETE | UNDETERMINABLE_CAP | sessions spending the full 360 |
|---|---|---|---|
| A binary | 268 (46.5%) | 308 | 64 / 96 |
| B incumbent | 355 (61.6%) | 221 | 58 / 96 |
| C floored | 373 (64.8%) | 203 | 52 / 96 |
| D draw-latent | 396 (68.8%) | 180 | 53 / 96 |

Every session in every arm ended with all six domains terminal — no arm was
ended by bank exhaustion.

## Paired (CRN) differences

### D (draw-latent atoms17) − B (prod incumbent), n = 96

| metric | mean | 95% CI | reading |
|---|---|---|---|
| questions | **−3.22** | [−6.23, −0.21] | D is marginally cheaper (<1%) |
| skill RMSE | −0.02255 | [−0.05096, +0.00586] | −7.3%, not resolved at n=96 |
| bias RMSE | **+0.03477** | [+0.00739, +0.06214] | D is worse on bias RMSE |
| skill coverage | **+0.07986** | [+0.04670, +0.11302] | **D recovers 8.0 coverage points** |
| bias coverage | −0.01389 | [−0.03933, +0.01156] | no difference |
| skill width | **+0.14557** | [+0.12803, +0.16311] | D's intervals are 13.8% wider |
| bias width | +0.01981 | [−0.00096, +0.04057] | no difference |

Per-domain question differences (D − B): sz −1.57 [−2.85, −0.30] is the only
one whose interval excludes zero; lpd −0.05, gpd −0.18, lrda −0.52,
grda −1.22, iic +0.32 are all indistinguishable from zero.

### C (floored ensemble9) − B (prod incumbent), n = 96

| metric | mean | 95% CI | reading |
|---|---|---|---|
| questions | **−2.81** | [−5.58, −0.05] | marginally cheaper (<1%) |
| skill RMSE | +0.00997 | [−0.02055, +0.04049] | no difference |
| bias RMSE | +0.00829 | [−0.02232, +0.03891] | no difference |
| skill coverage | +0.02257 | [−0.01509, +0.06023] | **the floor alone does not close the gap** |
| bias coverage | −0.01042 | [−0.03683, +0.01600] | no difference |
| skill width | **+0.09777** | [+0.08021, +0.11532] | floor widens intervals |
| bias width | −0.01402 | [−0.03709, +0.00905] | no difference |

Per-domain question differences (C − B): grda −1.67 [−2.75, −0.58] only.

### Reference: each n-way arm − binary, skill coverage (house margin −0.03)

| contrast | mean | 95% CI | point pass | confidence pass |
|---|---|---|---|---|
| B incumbent − binary | −0.07465 | [−0.11301, −0.03630] | **FAIL** | **FAIL** |
| C floored − binary | −0.05208 | [−0.09058, −0.01359] | **FAIL** | **FAIL** |
| D draw-latent − binary | +0.00521 | [−0.02235, +0.03277] | **PASS** | **PASS** |

D is the only n-way arm that meets absolute nominal skill coverage *and*
passes the −0.03 house noninferiority margin against binary. The deployed
incumbent fails both, by a wide margin.

## Burden must never be read without coverage

The Precision policy stops a domain when its **skill posterior radius**
contracts inside a fixed tolerance (`PRECISION_CONTRACTION_BY_DOMAIN` × the
prior SD), guarded by a quantile MCSE term. The tolerance does not know
whether the posterior is honest. An engine whose intervals are too narrow
therefore reaches the tolerance sooner and stops earlier — it has bought
questions with unearned confidence, not with efficiency. Any burden number
read on its own will rank an overconfident engine as the cheapest one.

Per-arm reading of this run:

| arm | skill coverage | burden | verdict on the burden |
|---|---|---|---|
| A binary | 0.9444, Wilson contains 0.95 | 353.07 | **honest** |
| B PROD INCUMBENT | 0.8698, Wilson entirely below 0.95 | 354.26 | **not honest — and not cheap either** |
| C floored | 0.8924, Wilson entirely below 0.95 | 351.45 | **not honest**; its −2.81 saving vs B is not creditable |
| D draw-latent | 0.9497, Wilson contains 0.95 | 351.04 | **honest** |

### Burden at matched honesty

Two arms clear the honesty bar: **A (binary)** and **D (draw-latent)**. Their
burdens are 353.07 and 351.04 questions, and the paired difference
D − A = −2.03 [−6.65, +2.59] does not resolve. So the honest price of a
session on this bank at this profile is ~351–353 questions, and the
draw-latent engine pays it without asking for a coverage discount.

The incumbent's overconfidence buys it **nothing**. B's skill intervals are
12.1% narrower than D's (1.0581 vs 1.2036) and 35.0% narrower than binary's,
yet its mean burden (354.26) is the **highest** of the four arms, and it is
not lower than either honest arm. On this evidence B is dominated: it is no
cheaper, and it is 8.0 coverage points short of D on the paired contrast. Its
narrow intervals are pure undercoverage, not efficiency.

Note that B's undercoverage is confined to the **skill** block — the
certification quantity. Its bias coverage (0.9549) is the highest of the four.
An arm cannot be cleared on bias calibration.

### Two limits on the burden reading — state them with the number

1. **Burden here is cap-dominated, so it is a weak discriminator.** The median
   session in every arm spent the full 360 questions (6 domains × the frozen
   60-question per-domain ceiling), and 52–64 of 96 sessions per arm did. The
   arm-to-arm spread is ≤ 3.2 questions, under 1%. The differentiating signal
   in this run is coverage and the terminal-state mix, not burden.
2. **This harness's selector does not skip completed domains, so its burden is
   an upper bound on production burden.** Production `advance.ts` restricts
   item selection to domains whose status is `ACTIVE`
   (`advance.ts:283`, `:399`); `qualification.py::_select` keeps offering every
   domain until the safety cap. `ESTIMATE_COMPLETE` is re-derived on each
   evaluation and is not sticky, so a domain that completes early can be pushed
   back to `ACTIVE` by continued questioning and end at
   `UNDETERMINABLE_CAP`. This is a pre-existing property of the shipped
   `--stopping precision` join (it applies equally to the banked
   `reports/prereg_smokes/served_precision_192.json`), not something this study
   introduced, and it applies identically to all four arms — so the *contrasts*
   remain valid while the *levels* are inflated.

   Consequence for the owner: **this run cannot rule out that under a
   production-faithful, ACTIVE-restricted selector the incumbent would convert
   its 12% narrower intervals into genuinely earlier stopping** — which is
   exactly the purchased-burden failure mode. Measuring true production burden
   requires an ACTIVE-restricted selector and is the recommended next step. The
   terminal-state mix is the closest available proxy, and it already points the
   other way: D certifies 396/576 domains as `ESTIMATE_COMPLETE` against B's
   355/576, so the honest engine reaches the policy's own precision target on
   **more** domains, not fewer.

### Why the within-arm split does not demonstrate purchase here

Splitting each arm's own sessions by realized coverage would ordinarily expose
the exchange directly. It does not, because burden is cap-limited and the split
is confounded by truth difficulty: in arm B the sessions that missed used
*more* questions (+4.85 [+0.27, +9.44]), and in A, C, D the split does not
resolve (−2.52 [−10.21, +5.17], +0.37 [−5.68, +6.43], +0.75 [−7.58, +9.08]).
Sessions that end early are the easy ones, which are also the well-covered
ones. The cross-arm evidence — narrower intervals at sub-nominal coverage for
equal burden — is what carries the finding.

## What this means for the deployed engine

1. **The deployed configuration is measurably overconfident on the
   certification quantity.** Skill coverage 0.8698, Wilson [0.8398, 0.8948],
   entirely below nominal, on the real served bank under the real stopping
   policy, in the pre-registered continuum truth world. It fails the −0.03
   house margin against binary by a wide margin.
2. **The 0.15 lapse floor alone does not fix it.** C recovers only
   +0.0226 [−0.0151, +0.0602] skill coverage — not resolved at n=96 — and stays
   below nominal at 0.8924 [0.8644, 0.9151].
3. **The draw-latent construction does.** D restores nominal coverage
   (0.9497 [0.9286, 0.9647]) and recovers +0.0799 [+0.0467, +0.1130] against
   the incumbent, while keeping most of the accuracy prize: skill RMSE 0.2871
   vs binary's 0.3973 (−27.7%), and its skill intervals are still 26.0%
   narrower than binary's.
4. **The price of honesty is measurable and small.** D's skill intervals are
   +0.1456 wider than B's and its bias RMSE is +0.0348 worse. Burden is
   unchanged within noise.

## Status

Research-only sandbox result on an unqualified artifact
(`promotionForbidden: true` on all three response artifacts). It does not
constitute the locked campaign in `docs/LOCKED_CAMPAIGN_PREREG.md`, and at
n=96 it is powered for the coverage contrasts, not for the RMSE ones. The
burden limitation in §2 above should be closed before any burden claim is made
to the owner.
