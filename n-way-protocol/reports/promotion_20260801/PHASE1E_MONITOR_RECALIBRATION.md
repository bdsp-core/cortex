# Phase 1e — misspecification-monitor recalibration for the promotion grid

The monitor's operating characteristics were calibrated against the
floor015 mixture reference (threshold 5.204,
`misspec_monitor_oc_floor015.json` lineage); the promoted artifact is the
nesting34 draw-latent grid, so its reference mixture changes and the
threshold must be re-derived. Recalibrated on the RAW `s_mean` staged bank
axes (the artifact's fitted frame — the monitor never sees skill-scaled z),
2,000 simulated sessions × 90 wrong-picks, false-trip target 0.01, default
seeds: `reports/misspec_monitor_oc_nesting34.json`.

| quantity | floor015 reference | **nesting34 (promoted)** |
|---|---|---|
| CUSUM threshold | 5.204 | **5.101** |
| null false-trip rate | 0.01 | **0.0100** (at target) |
| uniform collapse (N3b) trip rate | 1.00 | **1.000** |
| median wrong-picks to collapse trip | 15 | **15** |
| half lapse-drift trip rate | — | 0.636 (median 42) |
| half beta-drift trip rate | — | 0.5035 (median 44) |

Collapse detection is re-verified for the promoted artifact: every
uniform-collapse session trips, at the same median depth as the floored
reference. The threshold ships with the qualified artifact's monitor
configuration; the calibrated false-trip target (1%) is the Phase-8
observation gate for real-session trip rate.

Note the defense-in-depth ordering observed in Phase 1c: the nesting atom
usually resolves a diffuse responder *before* the monitor would trip (the
real tripped session put 0.984 mass on it), so in production the monitor is
the backstop, not the primary mitigation.
