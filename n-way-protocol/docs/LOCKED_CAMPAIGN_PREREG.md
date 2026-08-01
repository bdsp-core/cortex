# Locked qualification campaign — pre-registration (2026-07-31)

Registered BEFORE the locked run, per QUALIFICATION.md. Authorized under the
owner's 2026-07-31 plan approval; the noninferiority margin below formalizes
the −0.03 house margin named in that approved plan. Any deviation from this
document voids the run.

## Artifact under test

`artifacts/iiic_conditional_f1_engine_frame_rd.json` — engine-frame
hierarchical ensemble (variant `engine_frame_hierarchical`), nine
population-quantile draws β ∈ [0.651, 1.475], λ_d floored at 0.15,
`provenance.dr07: qualified` (gate report
`reports/dr07_gate_engine_frame.json`). Population parameters from the
stable core: μ_log β = −0.020575, τ_log β = 0.255378. Binary arm is the CRN
control in every cell.

## Evidence already banked (not re-litigated by this run)

- Shadow replay (family 4): +0.359 bits/wrong-pick over uniform,
  leave-one-reader-out, 122/124 readers positive, margin guard intact
  (`reports/shadow_replay_stable_core_500.json`).
- DR07 gate pass (`docs/DR07_CRITERION.md` v2; owner ratification pending).
- Floor pricing ×3.51 skill info at 0.15; monitor OC threshold 5.204,
  collapse trip 100% at median 15 wrong-picks
  (`reports/floor_pricing_engine_frame_4000.json`,
  `reports/misspec_monitor_oc_engine_frame.json`).

## Design constants

Production profile: 1,200 particles, 30 MH steps, ESS 0.5, formal SBC.
Seed bases disjoint from every development grid (62.1M series):
cell 1 → 65,000,000; cell 2 → 65,100,000; cell 3 → 65,200,000;
cell 4 → 65,300,000. Sharding across hosts via `--seed-base`/`--rows-output`
and `qualification_merge` (verified exact).

## Cells

1. **Continuum SBC gate** (2,000 seeds, GATING). Truth β drawn per
   replicate from LogNormal(μ = −0.020575, σ = 0.255378) — the fitted
   reader population, matching the label-agnostic serving reality — truth
   λ_d = 0.
2. **Served-bank adaptive comparison** (2,000 seeds, GATING; family 3).
   Real bank (`.artifacts/categorical_bank_axes.csv`), each arm's own
   selector, unchanged Precision stopping via the TS bridge. Same truth
   design as cell 1.
3. **Upper-quantile temperature stress** (500 seeds, informational). Truth
   β = 1.903 fixed — the sharpest observed stable-core reader.
4. **Uniform collapse** (500 seeds, informational). Truth λ_d = 1.0;
   documents the damage bound the floor+monitor+replay stack must cover,
   as in Phase 2.

## Gates (cells 1 and 2 must BOTH pass; otherwise F1 parks — no iteration)

- **Absolute coverage**: skill and bias 95% interval coverage with the
  Wilson 95% interval containing or lying above 0.95.
- **Paired noninferiority vs binary**: coverage difference ≥ −0.03, point
  AND confidence (owner margin, formalized per the approved plan).
- **Tightening**: skill interval width ratio nway/binary < 1.0 with
  cluster-95 CI excluding 1.0.
- Burden, ESS/acceptance/MCSE, and SBC rank diagnostics reported; monitor
  and rollout semantics unchanged by this run.

## On pass

Artifact re-emitted with `qualification: "qualified"` chained to the merged
cell reports; engine profile constants regenerated from the Python oracle;
contract/parity/golden suites re-run; deploy dark; offline shadow → internal
test → allowlisted phases per the promotion checklist. Serving-mode changes
remain gated behind the fail-closed server rollout and owner go.
