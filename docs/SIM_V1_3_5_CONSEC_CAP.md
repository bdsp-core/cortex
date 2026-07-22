# Sim report — consecutive-same-domain cap (v1.3.5)

> Historical AD6 simulation record. The result remains provenance for the
> rollback policy, not a current production rollout instruction.

## Problem

The adaptive selector pours every remaining question into the last unresolved
task. When one IIIC task is borderline-hard and the others resolve quickly,
this produces very long single-domain runs. Session `e46cc793` asked
**seizure 99 times in a row** (the last 99 of 240 trials), because seizure was
the sole PENDING task from trial 142 on. This is statistically correct
(`session_controller._compute_active_domains` only offers PENDING tasks, and
AD6 verdicts are monotonic) but monotonous for a human test-taker.

## Mitigation

`CortexSession(max_consecutive_same_domain=N)` (default `None` = off). After
`N` consecutive questions on one task the selector is forced onto a different
domain for the next question — preferring other unresolved tasks, falling back
to a resolved IIIC task only when one task is the sole remainder. The spike
block (Phase A) is exempt. The forced "variety" question still updates the
joint posterior.

## Sweep

`sim_v1_3_5/run_consec_sweep.py` — 300 heterogeneous-rater sessions at the ship
AD6 params (N_MIN=20, ALPHA=0.25, R_STAR=0.30), K=7. Heterogeneous = one IIIC
task at a near-cut-score level (lingers) with the rest clearly below
(resolve fast), varied over all six IIIC tasks; each rater seed shared across
cap values for a controlled comparison. Metric: `max_iiic_run`, the longest
consecutive IIIC single-task run (excludes the uncapped spike block).

| cap | iiic_med | iiic_p95 | iiic_max | q_med | q_p95 | all_resolved | pass | fail | refer |
|----:|---------:|---------:|---------:|------:|------:|-------------:|-----:|-----:|------:|
| off |    7 |  145 |  158 | 142 | 300 | 93.3% | 0.30 | 6.63 | 0.07 |
|   5 |    5 |    5 |    5 | 144 | 280 | 95.0% | 0.35 | 6.60 | 0.05 |
|   8 |    7 |    8 |    8 | 142 | 300 | 93.3% | 0.35 | 6.58 | 0.07 |
|  12 |    7 |   12 |   12 | 142 | 213 | 98.3% | 0.35 | 6.63 | 0.02 |
|  20 |    7 |   20 |   20 | 142 | 245 | 95.0% | 0.33 | 6.62 | 0.05 |

## Findings

- Without a cap the worst IIIC run reaches **158** (p95 145). Every cap holds
  the run at its value exactly.
- The cap has **no statistical downside**: `all_resolved` is equal-or-better
  (the forced variety questions add cross-task evidence via Σ that helps the
  lingering task resolve), worst-case session length is equal-or-shorter, and
  the verdict mix is essentially unchanged (REFER even drops).
- Balanced raters are unaffected at cap≥8 (their natural IIIC run is ~7).

## Decision

**cap = 12** (`MAX_CONSEC_SAME_DOMAIN_DEFAULT`), wired into the live test in
`eeg_bank_viewer`. Best `all_resolved` (98.3%), shortest worst-case sessions
(p95 213 vs 300), runs capped at 12, verdicts flat. Tests/sims/audit keep the
`None` (off) default. Shipped in cortex-v1.3.5.
