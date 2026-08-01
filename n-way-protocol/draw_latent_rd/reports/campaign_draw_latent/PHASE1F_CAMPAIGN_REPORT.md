# Locked campaign result: 9 of 10 gate clauses pass; the campaign FAILS on the pre-registered absolute-bias clause (2026-08-01)

Executed exactly as registered in `docs/LOCKED_CAMPAIGN_PREREG_DRAW_LATENT.md`
(committed before the first seed ran): draw-latent nesting34 artifact,
disjoint 67M seed series, production profile (1,200p / 30 MH / ESS 0.5),
formal SBC, the production-faithful ACTIVE-only selector, binary CRN
control in every cell. Machine-readable verdict: `GATE_VERDICT.json`
(`campaign_pass: false`). Under the standing park rule the promotion is
**PARKED before Phase 2** — no artifact was qualified, nothing merged,
nothing deployed, no serving change of any kind.

## Gating cells — every clause, exactly as pre-registered

### Cell 1 — continuum SBC (2,000 seeds, own-cap 90-question frame)

| clause | value | verdict |
|---|---|---|
| absolute skill coverage (Wilson contains/above 0.95) | 0.95542 [0.95158, 0.95897] — **above** nominal | PASS |
| absolute bias coverage (same rule) | 0.94575 [0.94155, 0.94966] | **FAIL** (upper edge 0.00034 below 0.95) |
| skill noninferiority ≥ −0.03, point AND confidence | +0.01042 [+0.00490, +0.01594] | PASS |
| bias noninferiority ≥ −0.03, point AND confidence | +0.00008 [−0.00531, +0.00547] | PASS |
| tightening: skill width ratio < 1, cluster-95 excl. 1 | 0.62368 [0.62001, 0.62734] | PASS |

Skill RMSE 0.3529 vs binary 0.5746 (−38.6%); SBC KS skill 0.0697
(conservative direction), bias 0.0180; atoms alive 19.7/34.

### Cell 2 — served-bank adaptive comparison (2,000 seeds, UNCHANGED Precision policy, real bank)

| clause | value | verdict |
|---|---|---|
| absolute skill coverage | 0.94833 [0.94423, 0.95215] — contains nominal | PASS |
| absolute bias coverage | 0.94475 [0.94052, 0.94870] | **FAIL** |
| skill noninferiority | +0.00400 [−0.00177, +0.00977] | PASS |
| bias noninferiority | −0.00425 [−0.00972, +0.00122] | PASS |
| tightening | 0.82316 [0.81868, 0.82764] | PASS |

Burden: draw-latent **228.52** questions vs binary **290.90** (−21.4%) at
honest skill coverage — the corrected Phase-1b finding replicated at
n = 2,000 under the real bank and the real stopping policy. Skill RMSE
0.3544 vs 0.4363; MH acceptance 0.217; ancestry 0.480; SBC KS skill
0.0387, bias 0.0125; atoms alive 16.0/34.

## Informational cells (never gating, per the pre-registration)

- **Cell 3, β = 1.903 point mass** (500 seeds): skill coverage 0.91233
  [0.90168, 0.92193] vs binary 0.94400 — the documented sharp-tail
  residual, improved from the battery's 0.9062 by the widened grid
  (MAP atom tracks the world 62.0%). Same disclosed open edge as at
  pre-registration; owner options unchanged (tail mass, particles, or
  documented residual).
- **Cell 4, uniform collapse λ_d = 1** (500 seeds): skill coverage 0.92167
  [0.91150, 0.93075] vs static mixture's historical 0.43; sessions
  concentrate 93.6% mean posterior mass on the binary-nesting atom
  (MAP tracks world 93.2%, 3.9 atoms alive) — graceful degradation at
  scale, before the monitor is ever needed.

## Why the failing clause failed — and what it does NOT say

The absolute-bias clause has never been met by ANY arm of this platform at
a powered n, including every binary control: floored campaign categorical
0.94267 [0.93836, 0.94669] and its binary control 0.94600 [0.94181,
0.94990]; the 2,000-seed integrated development runs 0.94250/0.94550
(QUALIFICATION.md's standing open gate); cell 1's own binary control
0.94567 (also failing); cell 2's binary control 0.94900 (its one
above-threshold draw). Cell 1's draw-latent bias coverage (0.94575) is
the highest categorical bias coverage any 2,000-seed campaign here has
recorded, four intervals short (11,349/12,000 vs the 11,353 the Wilson
edge requires). The paired bias contrasts (+0.00008 and −0.00425, both
far inside −0.03) show the artifact under test adds no bias miscoverage
over the control; the shortfall is the documented 1,200-particle
profile-machinery property (PHASE2_REPORT.md: "profile machinery, not an
F1 defect"), which the pre-registered clause nevertheless binds to.

The park rule exists precisely so that this argument is made TO THE OWNER
rather than silently applied by the executor after seeing the data. It is
recorded here and nothing else was concluded from it.

## Owner decisions that unblock (any one)

1. **Ratify a skill-primary absolute gate**: the absolute nominal-coverage
   clause applies to the certification (skill) block; the bias block is
   governed by the −0.03 paired-noninferiority guardrail (which passes in
   both cells). On the recorded rows the campaign then passes 10/10 with
   no re-run — gate evaluation is arithmetic on committed data
   (`campaign_gates.py`), not a new experiment.
2. **Waive the absolute-bias clause for this run** with the measured
   0.9457/0.9448 recorded in the qualified artifact's provenance.
3. **Require particle-profile requalification** (the only path that can
   actually move absolute bias coverage; a new engine profile and a new
   campaign — weeks, and it re-opens the frozen 1,200p/30MH profile).

## Status

Campaign data complete and committed; verdict evaluated exactly as
pre-registered; promotion parked before Phase 2. Every other Phase-1 gate
(1b, 1c hard kill gate, 1d, 1e) passed and is committed on this branch.
