# Locked qualification campaign — draw-latent amendment (2026-08-01)

Disclosed amendment to `docs/LOCKED_CAMPAIGN_PREREG.md`, registered BEFORE
the locked run, per QUALIFICATION.md and the promotion plan
(`docs/DRAW_LATENT_PROMOTION_PLAN.md`, owner decisions of 2026-08-01).
Any deviation from this document voids the run. The park rule stands: a
failed gating cell ends the promotion — no iteration, no reruns.

## What changed versus the 2026-07-31 pre-registration, and why

1. **Artifact under test** is now the draw-latent promotion candidate
   `artifacts/iiic_conditional_f1_engine_frame_nesting34_rd.json`
   (sha256 `3390932ad57da3e4647a68d6d7edd1afc88a1c04e99eae5087576f7784094806`):
   33 population-quantile atoms at the OUT-OF-SAMPLE heterogeneity
   (τ = 0.3581, `oos_population_check`) floored at 0.15, plus the λ_d = 1
   binary-nesting atom, uniform 1/34 weights, aggregated by the draw-latent
   cloud (construction B) instead of the static mixture. Same fit lineage
   (`iiic_conditional_f1_engine_frame_rd.json`, DR07 v2 gate pass) as the
   floored-mixture artifact the original pre-registration named. Grid and
   nesting-atom choices are implementer defaults awaiting owner
   ratification, disclosed in the promotion plan; the battery evidence for
   them is `draw_latent_rd/reports/B_VIABILITY_VERDICT.md`.
2. **Fresh disjoint seed series** (67M; every prior grid used 62.1M dev,
   63.8M monitor, 65M original prereg, 66M head-to-head, 68M real-response
   arbitration): cell 1 → 67,000,000; cell 2 → 67,100,000;
   cell 3 → 67,200,000; cell 4 → 67,300,000.
3. **Harness fidelity fix disclosed**: precision-mode selection is now
   restricted to domains the frozen policy reports ACTIVE, with the
   production variety semantics (`advance.ts:283,399`; session.ts:71),
   mirroring production instead of offering every domain to the safety cap
   (`qualification._precision_eligible`, tested). This applies identically
   to every arm; it corrects the burden inflation recorded in
   `reports/PROD_INCUMBENT_COMPARISON.md` §"Two limits".
4. **Particle budget** held at production 1,200 (34 atoms → 35.3 expected
   particles per atom, meeting the documented ≥ 35 floor); atoms-alive is
   reported per cell from the harness diagnostics.
5. **Selector approximation disclosed**: the draw-latent arm's item
   selection uses the static-mixture expected-loss screen over the same 34
   atoms (as in the battery and head-to-head); inference is fully
   draw-latent. The binary arm is the unchanged CRN control.
6. Noninferiority margin: the owner −0.03 house margin, point AND
   confidence, unchanged.
7. DR07: criterion v2 (`docs/DR07_CRITERION.md`) executed as written;
   owner ratification of the criterion mapping remains pending and is
   surfaced in the final promotion report.

## Cells (unchanged design, amended artifact/seeds)

| cell | world | seeds | n | stopping | role |
|---|---|---|---|---|---|
| 1 `campaign_continuum34` | truth β ~ LogNormal(−0.020575, 0.255378), λ_d = 0 | 67,000,000 | 2,000 | own-cap 15 (90-question SBC frame) | **GATING** |
| 2 `campaign_continuum34` | same continuum world | 67,100,000 | 2,000 | UNCHANGED production Precision policy, real served bank `.artifacts/categorical_bank_axes.csv` | **GATING** |
| 3 `campaign_stress190_34` | truth β = 1.903 point mass | 67,200,000 | 500 | own-cap 15 | informational |
| 4 `campaign_collapse34` | truth λ_d = 1.0 (uniform collapse) | 67,300,000 | 500 | own-cap 15 | informational |

Production profile throughout: 1,200 particles, 30 MH steps, ESS 0.5,
formal SBC (truth drawn from the inference prior). Runner:
`draw_latent_rd.harness --cell <name>`; per-seed rows written for the
gating cells; gates rendered by `draw_latent_rd/campaign_gates.py`.

## Gates (cells 1 and 2 must BOTH pass; else the promotion parks)

- **Absolute coverage**: skill AND bias 95% interval coverage with the
  Wilson 95% interval containing or lying above 0.95.
- **Paired noninferiority vs binary**: coverage difference ≥ −0.03, point
  AND confidence.
- **Tightening**: per-seed skill width ratio nway/binary mean < 1.0 with
  the seed-cluster 95% CI excluding 1.0.
- Burden, ESS/acceptance/ancestry, SBC rank KS, and atoms-alive reported;
  monitor and rollout semantics unchanged by this run.

## Evidence banked before this run (not re-litigated)

- Real-response arbitration (family 4 analog):
  `draw_latent_rd/reports/REAL_RESPONSE_ARBITRATION.md` — draw-latent
  nesting34 beats the deployed incumbent on the 13 real prod sessions
  (+15.5 nats; every n-way arm ≫ binary, so the asked≠gold mismatch does
  not dominate).
- Viability battery (`B_VIABILITY_VERDICT.md`) and head-to-head
  (`PROD_INCUMBENT_COMPARISON.md`).
- Monitor OC recalibration for this artifact:
  `reports/misspec_monitor_oc_nesting34.json` (Phase 1e).

## On pass

Exactly the original pre-registration's promotion path: qualified
re-emission chained to the merged cell reports, engine constants
regenerated and validated, contract/parity/golden suites, dark deploy,
then offline shadow → internal test → allowlisted phases behind the
fail-closed server rollout.
