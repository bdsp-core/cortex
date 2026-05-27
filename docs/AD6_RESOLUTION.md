# AD6 — termination + verdict policy

The CORTEX live test's stopping and verdict logic. Authoritative
specification lives in `scripts/cortex_policy.py`; this doc records the
calibration choices and any deviations from the panel-derived defaults.

## The rule

For each IIIC task `k` after every trial:

  - `π_k    = Σ_i w_i · 1[ℓ_k^(i) > ℓ*_k]`                    pass-mass
  - `mcse_k = √( π_k(1−π_k) / ESS )`                           MC error
  - `R_k    = 1 − Var_post(ℓ_k) / Var_prior(ℓ_k)`              info gate

Task `k` is RESOLVED iff `n_per_task[k] ≥ N_min` AND `R_k ≥ R*` AND
either:

  - **PASS**: `π_k − Z·mcse_k ≥ 1 − α`
  - **FAIL**: `π_k + Z·mcse_k ≤ α`

Session stops when every task is RESOLVED, or the bank is exhausted.
At end of run, each remaining PENDING task becomes either
REFER_BORDERLINE (gate open, posterior in band) or
REFER_UNINFORMATIVE (data did not constrain ℓ_k).

## Panel-derived defaults

Unanimous output of a four-expert independent derivation panel. Each
parameter is independently OC-calibratable via `scripts/run_oc_validation.py`.

| Constant | Production | Source |
|---|---|---|
| `DEFAULT_N_MIN` | **15** | Minimum direct task-k trials before any verdict can be assigned |
| `DEFAULT_R_STAR` | **0.30** | Information gate — posterior SD(ℓ_k) must contract ≥ ~16% from prior |
| `DEFAULT_ALPHA` | **0.05** | Per-task posterior-probability boundary for PASS / FAIL |
| `DEFAULT_Z` | **2.0** | MC-error buffer (matches engine's existing Z_BUFFER) |

## Internal-test override (2026-05-22)

**v1.1.0 update (2026-05-27).** Settings have been advanced toward
production. Current values in `scripts/cortex_policy.py`:

  * `DEFAULT_N_MIN` = **15** (= production target) — reachable now that
    the bank is 300 IIIC segments (50/task on average)
  * `DEFAULT_ALPHA` = **0.10** (production target: 0.05) — calibration
    midpoint between the v1.0 internal-test override (0.30) and the
    panel-derived production target (0.05). v1.1.0 is a calibration
    trial to characterise pass/refer/fail rates at this strictness
    before committing to α=0.05 in v1.2.0.

**v1.0 historical (for reference):** Earlier internal-test overrides
were `DEFAULT_N_MIN=6` and `DEFAULT_ALPHA=0.30`, scoped to the
100-segment IIIC bank, intended to validate the algorithm's end-to-end
mechanics (verdict assignment + `all_resolved` early-exit) on a bank
too small to support production-grade thresholds. The v1.0 audit
(`results/audit/selection_audit.md`) found that 0/13 raters reached
AUROC half-width < 0.05 in 100 questions — the override gave the
all-resolved early-exit path a chance to fire.

### Why the override

The internal test runs against a 100-segment IIIC bank. With `K=6`
tasks, the production `N_min=15` floor demands 90 direct trials minimum
just to clear the sample-size gate on every task — leaving essentially
zero headroom for `choose_item`'s information-greedy selector, which has
no task-balance constraint.

Empirically (first internal-test session, John Doe random-responder, 99
trials, `bank_exhausted`):

  - `choose_item` allocated `sz=9 lpd=15 gpd=16 lrda=19 grda=17 iic=23`
  - `sz` never reached `n_min=15` despite having the most confident
    FAIL signal in the entire session (`π=0.0002`, `FAIL_ub=0.0017` at
    session end — `4σ` below the FAIL boundary by trial 36)
  - The sample-size gate at `cortex_policy.py:226` (`if n_per_task[k] <
    self.n_min or R[k] < self.R_star: continue`) blocked the verdict
    assignment even though the posterior had been decisive for ~63
    trials before bank-exhaustion

`R*`, `α`, and `Z` were not binding — all six tasks cleared `R*=0.30`
within 26 trials; the three confident FAILs landed cleanly inside
`α=0.05` with `Z=2.0`. The internal-test override targets `N_min`
specifically because it was the only binding constraint.

### Why `N_min = 6`

  - Drops the K × N_min floor from `90 → 36` trials on the 99-trial
    engine pool — leaves substantial headroom for `choose_item` to fire
    the probability gate on the more variance-informative tasks
  - Direct evidence on each task is still required (6 trials is enough
    that an early swing of 1-2 lucky responses cannot fix a verdict),
    but the floor no longer fights the selector's information-greedy
    allocation pattern. Empirically, `sz` is the consistently
    under-sampled task across the internal-test sessions (see below);
    `N_min = 6` lets `sz` resolve in ~70-90 trials instead of waiting
    until trial 98+
  - Genuinely borderline tasks (π in the middle band) correctly remain
    REFER_BORDERLINE — `α` controls the verdict bands; lowering N_min
    only affects when the sample-size gate opens

### Empirical observations from internal-test sessions (2026-05-22)

Four random-responder internal-test sessions, across `N_min` calibration
revisions:

| Session | `N_min` | `α` | n_trials | stop_reason | n_per_task[sz] | sz verdict trial |
|---|---|---|---|---|---|---|
| John    | 15   | 0.05 | 99 | bank_exhausted | 9  | n_min never reached |
| Garrett | 8    | 0.30 | 99 | bank_exhausted | 8  | n_min reached trial 30 |
| Andrew  | 8    | 0.30 | 73 | all_resolved   | 8  | n_min reached trial 72 |
| Jeffrey | 8    | 0.30 | 99 | all_resolved   | 8  | n_min reached trial 98 |

`sz` is the under-sampled task in every session. The bank's IIIC class
balance (17 sz / 16 lpd / 17 gpd / 16 lrda / 17 grda / 17 other) is
even — so the under-sampling is `choose_item`'s information-greedy
selector consistently preferring the other tasks' segments, not a bank
shortage. Likely structural cause: per-segment `s_sd` for seizure
segments, or the seizure cut-score being far from the prior mean. Worth
auditing in `scripts/audit_selection.py` for the production handoff,
but **not relevant at production scale** — the 10k bank gives every
task hundreds of queries even at the same skewed allocation ratio.

### Why the α override (2026-05-22)

The `n_min=8` override above resolved the sample-size gate as the
binding constraint, but a second internal-test session (Garrett Roedel,
99 trials, random responder, `bank_exhausted`) revealed a second
binding constraint: AD6's "every task must resolve" stopping rule
combined with the strict α=0.05 probability boundary.

Empirically, Garrett's session produced 5 confident FAILs (sz, lpd,
gpd, lrda, iic) but `grda` stayed PENDING the entire session — π for
grda oscillated in [0.26, 0.40], unable to cross either α=0.05 (FAIL)
or 1-α=0.95 (PASS). With one task PENDING, the session cannot trigger
`all_resolved` and must bank-exhaust at trial 99.

This pattern is structural for random-guess responders: the cut-score
ℓ*_k for some tasks (grda in particular) appears to sit near the
chance-level posterior equilibrium, leaving π in the middle band
indefinitely. The strict α=0.05 boundary is correct for production but
prevents the internal test from exercising the `all_resolved` exit on
realistic random-responder validation runs.

α=0.30 widens the bands to PASS at π≥0.70, FAIL at π≤0.30 — close
enough to the random-responder equilibrium that most sessions should
trigger `all_resolved` within the 100-segment bank, even when one task
sits near the cut-score. This is **not** a defensible PASS/FAIL
threshold for production certification (it means "70% confident →
PASS"); it is purely a mechanics-validation knob.

### Before pushing to production

Restore both constants in `scripts/cortex_policy.py`:

  * `DEFAULT_N_MIN = 15` (was 6) at line 58
  * `DEFAULT_ALPHA = 0.05` (was 0.30) at line 62

The production bank (>10,000 segments) makes `N_min=15` well within
budget — each task can be queried 1,500+ times, so the information-gate
constraint becomes binding instead of the sample-size gate. The
production bank also gives every task enough trials to drive π away
from the chance-level equilibrium, so α=0.05 becomes routinely
crossable.

Checklist for the production-restore PR:

  - [ ] Revert `DEFAULT_N_MIN` from 6 to 15 in `scripts/cortex_policy.py`
  - [ ] Revert `DEFAULT_ALPHA` from 0.30 to 0.05 in `scripts/cortex_policy.py`
  - [ ] Re-run `tests/test_cortex_ad6_integration.py` — the integration
        tests construct AD6Policy with explicit `n_min` and `alpha`
        kwargs and are independent of the defaults, so they should
        remain green
  - [ ] Re-run the AD6 OC harness against the production bank
  - [ ] Remove this "Internal-test override" section, or move it under
        a "Calibration history" appendix
  - [ ] Rebuild `dist/cortex-internal-test.zip` so the distributed
        bundle ships the production constants

## References

  - Spec: `scripts/cortex_policy.py` (`AD6Policy` class)
  - Integration tests: `tests/test_cortex_ad6_integration.py`
  - OC harness: `scripts/run_oc_validation.py`
  - Audit memory: `results/audit/selection_audit.md`
