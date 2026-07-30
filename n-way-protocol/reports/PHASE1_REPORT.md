# Phase 1 — overnight execution report (2026-07-29/30)

Branch `nway-phase1-20260729`, one commit per gate. Constraints honored:
no cortex_web/, deploy/, or production-server changes; all work in
n-way-protocol/ and research directories. Baseline suite was green before
the first change (pytest 32, vitest 37 passed / 3 skipped, tsc clean) and
after every gate.

## The one-paragraph summary

The plan compressed hard once the repo was mapped: the leakage-controlled
fitter, the Python oracle at the production profile, and the qualification
harness already existed, so tonight's work was the four genuinely missing
pieces — the λ_d robustness floor (built, priced, stress-replayed), the
distractor-misspecification monitor (designed, implemented, calibrated),
the tier-stratified refit question (answered on real data), and the bank
categorical-coverage audit (run over the full served bank). Results below;
owner decisions at the end.

## Gate 0 — what already existed (no work needed)

- `artifact_rd.py`: leakage-controlled conditional-distractor fitter
  (reader/source connected-component folds, dual held-out panels,
  reader-clustered bootstrap) — already run on 68,783 staged wrong-picks
  with leave-one-expert-out gold → the deployed nine-draw ensemble
  (`iiic-f1-crossfit-ensemble9-rd-20260720`).
- `qualification.py`: planted-truth harness with exactly the knobs the N3b
  replay needs (`--truth-distractor-lapse`, `--artifact-ensemble`,
  formal-SBC, 1200p/30MH profile).
- `bank_audit.py`: Fisher information audit, λ_d-parameterized.
- TypeScript categorical engine + ranked speculation: committed to
  cortex_web main and latency-qualified on 2026-07-21 — the N3 report's
  "major engineering item" was already done.

## Gate A1 — λ_d floor machinery (commit b8c4975)

`artifact_floor.py` floors the deployable draw sets (crossfit
`bootstrap.draws` for the harness; ensemble `draws` for the engine format)
at a declared `robustnessFloor`, leaving betas, weights, and the recorded
fit untouched. Four artifacts generated (floors 0.15 and 0.20), hash-chained
to their sources, promotion-forbidden. The guard test rederives them
byte-identically, so the committed artifacts are exactly reproducible and
any sub-floor draw fails the suite.

## Gate A2 — pricing the floor on real axes (commit f55568f)

Finite-difference Fisher accounting over 4,000 real bank segments
(`reports/floor_pricing_real_axes_4000.json`):

| floor | skill info retained | bias info retained | × over binary (skill) | × over binary (bias) |
|---|---|---|---|---|
| 0.00 | 100% | 100% | 4.87 | 1.75 |
| 0.15 | 72.7% | 85.3% | 3.54 | 1.50 |
| 0.20 | 66.6% | 81.9% | 3.24 | 1.44 |

Both candidate floors keep the categorical prize far above the ×2 line the
N3 report treated as the insurance budget. Pricing alone does not separate
0.15 from 0.20.

## Gate A3 — N3b collapse replay with floored artifacts (commit pending below)

Six cells, 192 formal-SBC CRN seeds each at the full 1200p/30MH profile,
matched 90-question burden (`reports/floor_stress/stress_*.json`). Binary
control per cell: skill RMSE 0.540/0.534, skill coverage 0.947/0.957.

| engine artifact | truth | skill RMSE | bias RMSE | skill cov | bias cov |
|---|---|---|---|---|---|
| unfloored | softmax | 0.3330 | 0.3934 | 0.9349 | 0.9418 |
| floor 0.15 | softmax | 0.3506 | 0.3956 | **0.9549** | 0.9514 |
| floor 0.20 | softmax | 0.3589 | 0.3873 | 0.9444 | 0.9497 |
| unfloored | uniform | 1.3979 | 0.9273 | 0.2821 | 0.7873 |
| floor 0.15 | uniform | 1.2083 | 0.8040 | 0.4010 | 0.8438 |
| floor 0.20 | uniform | 1.1585 | 0.7686 | 0.4688 | 0.8663 |

Three findings:

1. **The N3b collapse reproduces at the production profile** (unfloored ×
   uniform: coverage 0.28 — the prototype said 0.26). This was not a
   prototype artifact.
2. **The floor's real payoff is on the well-specified side**: floor 0.15
   costs only +0.018 skill RMSE yet lifts absolute skill coverage from
   0.9349 to 0.9549 — at or above nominal, where the pre-existing powered
   runs sat at 0.9425–0.9455 with the gate open. The extra likelihood
   dispersion appears to fix precisely the overconfident cross-domain
   updates behind the undercoverage flag. (192 seeds; needs the powered
   preregistered run to claim the gate.)
3. **No floor rescues the collapse world** (coverage 0.40–0.47 vs binary's
   0.957). The floor bounds damage per update; only the monitor bounds
   duration. The two are jointly mandatory, exactly as the design doc
   frames them.

## Gates B1–B3 — distractor-misspecification monitor (commit 9515641)

Design in `docs/MISSPEC_MONITOR_DESIGN.md`; reference implementation in
`misspec_monitor.py` (6 unit tests). Per-wrong-pick CUSUM of
`log[P_mixture(pick) / (1/5)]` in the fit frame (raw `s_mean` axes) —
fixed-input by construction: no session posterior, no RNG. Trip semantics
are fail-closed: mark the session, replay the full history binary-reduced,
continue binary-only; raw picks stay auditable. Floored artifacts bound the
increment below by `log(floor)`, which is what makes threshold calibration
stable — floor and monitor are complements.

Operating characteristics (2,000 sessions × 90 wrong-picks, 1% per-session
false-trip target, real axes; `reports/misspec_monitor_oc_*.json`):

| deployed artifact | threshold | uniform (N3b) trip rate | median picks to trip | half-drift-λ trip | half-drift-β trip |
|---|---|---|---|---|---|
| unfloored | 7.41 | 100% | 14 | 79% @ 38 | 72% @ 41 |
| floor 0.15 | 5.21 | 100% | 14 | 71% @ 38 | 63% @ 42 |
| floor 0.20 | 4.51 | 100% | 13 | 71% @ 38 | 64% @ 41 |

Under total collapse the monitor trips every session, at a median of ~14
wrong-picks — with binary-replay trip semantics the damage window is small
and self-erasing. The floor's only monitoring cost is mildly reduced
sensitivity to *partial* drift (which is also far less damaging). Combined
with A3: floor bounds damage per update, monitor bounds duration, binary
replay erases the window — the three-layer design holds.

## Gate C1 — refit verification + tier stratification (commit 1702ad1)

**Integrity: the committed pooled crossfit artifact is fully
bit-reproducible from the staged data** — dual held-out panels, full fit,
and all 200 bootstrap draws regenerate identically (the only payload
difference is the recorded worker count). The existing artifact chain
satisfies the leakage-control prerequisite the N3 report thought was still
open; no pooled refit was needed.

**Tier stratification** (`reports/artifact_tier_stratification.json`),
same leakage controls per tier:

| tier | wrong-picks | readers | β (full fit) | β 95% (reader-clustered boot) | λ_d | boot mass above ensemble span |
|---|---|---|---|---|---|---|
| pooled | 68,783 | 685 | 1.030 | [0.915, 1.137] | 0.000 | 0% |
| novice | 63,707 | 681 | 0.977 | [0.858, 1.060] | 0.000 | 0% |
| expert | 5,076 | 4 | **1.746** | [1.572, 2.109] | 0.000 | **100%** |

The deployed ensemble (span 0.872–1.195) under-models expert distractor
sharpness by ~70% — even steeper than the N3c preliminary's 1.32, and it
validates out-of-reader (4-fold expert held-out crossfit populated).
Honest caveats: 4 expert readers, so the clustered bootstrap interval is
optimistic, and expert gold is leave-one-expert-out while novice gold is
full plurality. Directionally unambiguous regardless: for a skill
certification instrument, β is not one number — it varies with the very
trait being certified. A skill-linked or tier-mixture β is a Phase-2
modeling question; meanwhile the fixed pooled β under-extracts expert
information (conservative, not anti-conservative), and the campaign's
misspecification stress family should include tier-β drift explicitly.
λ_d pins at zero in every tier — the floor remains an imposed margin, never
a fitted one, in all populations.

## Gate C2 — bank categorical state coverage (commit 522a10e): STRONG PASS

Full served bank, 20,502 segments × 6 asked classes
(`reports/bank_state_coverage_full.json`): informative items (allocation
KL vs uniform ≥ 0.05 nats) per asked class 19,246–20,292 (93.9–99.0%);
**zero** empty modal confusion pairs; **zero** ordered pairs with fewer
than 100 strong-support items. The identity channel is not bank-limited,
at λ_d = 0 or at either candidate floor.

## Repo-state and process flags

1. **The entire pre-existing n-way-protocol/ stack is untracked in git** —
   no branch has ever committed it. Tonight's commits track only the new
   files, which depend on untracked bases. Recommend committing the stack
   (excluding `node_modules/`, `.artifacts/` staged data — already
   gitignored) as soon as practical.
2. `scripts/test.sh` is broken on this host: installed vitest 4.1.10
   rejects its `--poolOptions.threads.*` flags, and the failure is easy to
   mask in a pipeline. Left untouched; needs a one-line flag update.
3. The N3 report's "engineering sequence" section is stale against the
   repo (items 2 and 3 already exist). Worth a one-paragraph correction so
   nobody executes from it.

## Owner decisions requested

1. **λ_d floor value — recommendation: 0.15.** Rationale: best absolute
   skill coverage under the well-specified world (0.9549, at/above nominal
   where the gate has sat open), retains ×3.54 skill information over
   binary, and detection latency under collapse is identical to 0.20
   (median 14 vs 13 wrong-picks). 0.20 buys nothing the monitor doesn't
   already provide, at a further −6% information.
2. **Commit the n-way-protocol stack** (repo-state flag above)?
3. **Phase 2 authorization scope** — production changes awaiting your
   explicit go: TS engine port of the floored artifact (constant
   regeneration in `nway_profile.ts`) and of the monitor + binary-replay
   trip path; then the preregistered campaign with the floored artifact.
4. **Noninferiority margins** for the preregistered campaign
   (QUALIFICATION.md requires owner-approved margins before the locked
   run). Suggest the campaign also carry: (a) the absolute-coverage gate
   re-test with floor 0.15 — A3 suggests the floor itself may close it;
   (b) tier-β drift in the misspecification stress family, per the C1
   expert finding; (c) the gate/floor hybrid standard-point arms from N3a.
5. **Tier-β modeling** (C1): fixed pooled β under-extracts expert
   information (β_expert ≈ 1.75 vs ensemble span ≤ 1.19). Skill-linked or
   tier-mixture β is the follow-on modeling question — flagging now so it
   enters Phase-2 scoping rather than surfacing mid-campaign.
