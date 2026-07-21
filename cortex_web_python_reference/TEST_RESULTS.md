# Verification result — 2026-07-18

## Organized Python policy implementation

The concrete policies now have one canonical implementation each:

- `../ad6-policy/` — shipped AD6, still the default and rollback path;
- `../precision-policy/` — frozen, uncalibrated `precision_v1` policy and
  complete controller profile; and
- `../termination-policy/` — shared decision/interface contract only.

Both the primary Python algorithm and this integration reference import those
packages through their existing policy seam. `CORTEX_TERMINATION_POLICY`
accepts `ad6` (default) or `precision_v1`; unknown values fail closed. The
frozen Precision factory requires `Corr_t`, 1,200 particles, `total_var`, the
uncertainty-aware 128-candidate scan, top-10 opener, same-domain limit 5,
floor-progress deadline, 60/domain, `quantile_mcse`, and the point-centred
radius statistic. It accepts no calibration controls and asserts all operative
scale/ramp/bias fields are `None`.

The TypeScript implementation was not modified.

### Consolidated gate

Command: `../run_policy_tests.sh`

- Shared-contract package: **1 passed**.
- Standalone AD6 package: **2 passed**.
- Standalone PrecisionPolicy package: **5 passed**.
- Integrated Python reference: **59 passed, 2 opt-in golden tests skipped**.
- Fixed-cloud Python↔TypeScript parity: **2 passed**.
- Pinned browser engine regressions: **48 passed**.
- Primary Python algorithm standard regressions: **70 passed, 1 environment
  skip, 2 slow selector proofs deselected**.

The two slow selector proofs were then run separately and both passed. They
confirmed exact full-grid/coarse-to-fine session equivalence and the existing
≥3× isolated selector-speed requirement.

### Per-engine deterministic golden sessions

Command:
`CORTEX_RUN_POLICY_GOLDENS=1 python -m pytest tests/test_policy_packages.py`

Result: **6 passed** (four integration assertions plus two full sessions).

- AD6: 300 questions, expected PASS/REFER vector, maximum own-domain count 60.
- PrecisionPolicy: 317 questions, five DETERMINED domains and two
  UNDETERMINABLE_CAP domains, maximum own-domain count 60.

These are per-engine Python drift guards. Full random-session equality with
TypeScript is not claimed because the engines intentionally use different RNGs.

### Broader workspace diagnostic

The entire primary `tests/` tree was also run with its explicit slow group
excluded. Policy and engine tests passed. Eleven remaining failures are outside
this change and reflect the pre-existing dirty workspace/environment: removed
`pyproject.toml`, requirements/config/sensitivity files and `slides/`, Python
3.12 versus a historical 3.11 assertion, and changed dataset row counts. The
one policy-era failure in that diagnostic—the immutable v1.3.6 600-particle /
500-question freeze—was corrected additively: its historical manifest remains
unchanged, while the current implementation is pinned by PROVENANCE v6.

The accepted extreme-skill undercoverage limitation remains disclosed and
unchanged; no calibration, tuning, or new stopping family was introduced.

# Verification result — 2026-07-17

## Compact AD6-replacement overnight pricing halt

The Option B artifacts were removed and replaced by the compact, uninterrupted
protocol under `calibration/ad6_replacement_overnight/`. All 840 frozen
frontier trajectories passed their artifact, seed, population, profile,
response, and finite-field integrity checks. The verifier recorded the exact
authorized current-source drift from the default-off calibration/SBC support;
the historical manifest hash remained pinned.

Trajectory-credited pricing through the 60-domain cap found:

| Family | Aggregate determination | Weakest task | Central/tail coverage | Burden median/p95/max | Outcome |
|---|---:|---:|---:|---:|---|
| Constant task factors | 82.05% | IIC 43.33% | 99.24% / 96.92% | 253 / 323.05 / 393 | FAIL per-task floor |
| Estimate-dependent ramp | 79.71% | spike 45.67% | 99.37% / 95.65% | 242 / 316.10 / 405 | FAIL aggregate and per-task floors |

Outcome: `pricing_halt`. No candidate was frozen, MCSE requalification did not
run, and R1/R2 were not consumed. The ramp lifted IIC to 60.33% but required a
3.767504 spike maximum to cover centrally-shrunk D-6 estimates, moving the
bottleneck to spike rather than resolving it.

The full suite after implementation and reconciliation passed 54 isolated
Python tests, 2 Python/TypeScript parity tests, and 48 unchanged browser
regressions. AD6 remains default and `cortex_web` was not modified.

## D-6 factorial stopping-time coverage screen

After the D-5 development freeze, the unchanged `frontier_p90guard_m3` arm ran
on 270 registered development factorial readers: 27 skill/bias/heterogeneity
cells with 10 response replicates per cell. The exact campaign selector was
`total_var` with uncertainty-aware coarse-to-fine refinement and
`n_subsample=128`. All 270 resumable sessions completed with zero execution
failure; maximum burden was 60 own-domain and 406 session questions.

- DETERMINED: 1,724/1,890 reader-domains (91.22%).
- Skill coverage among determined: 94.84% overall, 93.57% for
  `|l_true| >= 1`, and 96.18% below 1.
- Bias coverage among determined: 96.17%.
- Cut decisions: 749/1,724 (43.45%); correctness 746/749 (99.60%), with zero
  false PASS and three false FAIL outcomes.
- Terminal outcomes: 166 `UNDETERMINABLE_CAP`, zero BANK. Every CAP occurred at
  own n=60; no evidence, content, reservation, or bank-supply blocker appeared.
- Denominator-robust cell failures included IIC 3/9 at true skill +2.5, spike
  6/10 at -2.5, and LPD 7/10 at +0.75. Fifteen cell-domain estimates with at
  least eight determined replicates were below the descriptive 85% screen
  floor.

Outcome: `gross_tail_undercoverage_observed`. The unchanged candidate is now
blocked from the locked promotion campaign. A predeclared calibrated-interval
repair must be applied to both stopping and reporting, requalify the radius
MCSE guard, and pass fresh development confirmation before locked seeds are
used. The screen did not observe false certification; its critical failure is
the interval-honesty claim across the registered truth grid.

## Point-centred-radius frontier and composed sweep

The served-bank frontier completed 840/840 cap-100 trajectories: 300 fresh
correlated-prior readers and 540 factorial readers, each with 700 questions.
The original long-running PID exited abruptly after 761 artifacts without
writing a final manifest; the same hash-pinned command safely resumed the 79
missing trajectories. The final fail-closed verifier passed all source and
artifact hashes, exact seed/profile/population invariants, binary response and
task/count alignment, global segment de-duplication, final band counts, and
finite-field checks. Generator failures: 0.

The D-5 report covers skill and bias radius, fixed-checkpoint coverage, burden,
downstream cut decisiveness, diminishing returns, and metric-specific worst
factorial cells. Skill diminishing returns first fell below 1% per added own-
domain question at n=50–60; bias did so at n=50–75. Factorial undercoverage
remains a promotion limitation.

The minimal 100-reader CRN-paired composed sweep evaluated three arms and
completed all 300 sessions with zero numerical execution failures:

| Arm | Determined | Skill coverage | Bias coverage | Median / p95 session n | CAP / BANK |
|---|---:|---:|---:|---:|---:|
| D-5 profile, m=3 | 93.29% | 95.56% | 95.10% | 193 / 297.25 | 47 / 0 |
| D-5 profile, m=5 | 92.86% | 93.69% | 95.08% | 201.5 / 309.8 | 50 / 0 |
| Uniform 1.40, m=5 | 94.29% | 95.45% | 95.76% | 184.5 / 299.5 | 40 / 0 |

The selected opt-in development arm is D-5 profile m=3, with task-ordered
tolerance multipliers `[1.50, 1.30, 1.35, 1.30, 1.30, 1.30, 1.25]`. It was
the only profiled arm to retain aggregate skill coverage above 95% while every
domain remained at least 80% determined. This selection is development-only.

All 137 three-arm CAP/BANK outcomes were audited individually. Every outcome
was CAP at exactly 60 own-domain questions; evidence and content floors passed,
bank supply remained ample, and there were no BANK failures. At terminalization
26 missed raw radius, 101 passed raw radius but missed the MCSE guard, and 10
passed the instantaneous guard but not two-evaluation persistence.

## Complete gate after D-6 evidence freeze

Command: `./run_tests.sh`

- Full isolated Python suite: **46 passed**.
- Extended Python↔TypeScript parity: **2 passed**.
- Selected unchanged `cortex_web` engine regressions: **48 passed**.
- Provenance/current-reference and preserved original-source hashes: matched.
- Hard ceilings: observed maximum 60 own-domain and 366 session questions.
- `cortex_web` modifications: none.

AD6 remains the default. No superiority claim is made: the reserved locked
seeds remain unused, prospective coverage/noninferiority, adversarial,
stopping-time SBC, burden/failure, and approved parity gates remain, and no
TypeScript port is authorized.

# Verification result — 2026-07-16

## Superseded half-width PrecisionPolicy calibration

The historical development calibration selected
`c=1.40`, `m=5`, ESS floor 0.50, persistence M=2, and the qualified
quantile-MCSE rule
`halfwidth + 2.3369699408 * MCSE(halfwidth) <= r_l,k`. The surrogate fallback
thresholds are acceptance 0.20 and ancestry 0.35; they are diagnostic-only in
the qualified mode.

These numbers are now superseded because the candidate changed from interval
half-width to point-centred radius. The final combined N=1200/Corr_t arm had
determined 162/168 (96.43%)
reader-domains, had a 91.67% minimum domain rate, median/p90 own-domain burden
24/35, 94.44% coverage among determined domains, no numerical failures, and no
domain above 60 questions. This is development evidence, not promotion.
Details remain in `calibration/precision_dev/CALIBRATION_REPORT.md` for audit;
they are not evidence for the radius candidate. The radius-specific MCSE
qualification used 6,048 evaluations and selected inflation 1.5962415321
(effective multiplier 2.6258173203), with 98.05% leave-one-out upper coverage
and 97.11% minimum-domain coverage.

Command after freezing: `./run_tests.sh`

- Isolated Python tests: **30 passed**.
- Extended Python↔TypeScript parity: **2 passed**.
- Selected unchanged `cortex_web` engine regressions: **48 passed**.
- Provenance/current-reference and source hashes: all matched.

## Authorized vNext PrecisionPolicy implementation

The cut-independent stopping design is implemented as an opt-in policy in the
isolated Python reference. AD6 remains the default. The implementation includes
the N=1200/Corr_t/v15 profile repair, transactional fail-closed updates,
reversible precision statuses, the hard 60/domain budget, trial-0 and ongoing
bank feasibility, exact-supply content-band reservations, floor-progress
candidate filtering, additive reliability/model-fit/futility diagnostics,
actual skill/bias intervals, downstream-only cut classification, and raw
response preservation.

Command: `./run_tests.sh`

- Isolated Python tests at initial implementation: **27 passed**.
- Extended Python↔TypeScript parity: **2 passed**.
- Selected unchanged `cortex_web` engine regressions: **48 passed**.
- Provenance/current-reference hashes: all matched.
- Original source hashes and independent-inode isolation: all matched.

The deterministic parity artifacts regenerated exactly, confirming that normal
likelihood updates, shipped `total_var` selection, and AD6 numerical behavior
did not drift. The browser files were read for regression checks only; no
`cortex_web` file was modified by this implementation.

The candidate is implemented and development-frozen but not promoted.
Prospective OC gates and explicit approval still block TypeScript transfer.

## Authorized per-domain cap correction

The Python reference now uses the current production backstop: an unresolved
domain is capped after 60 of its own questions, removed from subsequent item
selection, and finalized as `REFER`. A normal production session stops with
`resolved_or_referred` once every domain is resolved or capped. The legacy
global 500-question sweep is no longer the production policy; explicit
NoStop/Delta research paths retain their full-trajectory behavior.

Three focused tests verify the default value, end-to-end cap/referral behavior,
and rejection of invalid cap values. No `cortex_web` file was modified.

## Isolation gate

Command: `./run_tests.sh`

- Isolated Python tests: **10 passed**.
- Extended Python↔TypeScript parity: **2 passed**.
- Selected existing `cortex_web` engine tests: **48 passed**.
- Source/data provenance hashes: all matched.
- Original and isolated files: independent inodes; no symlinks or hardlinks.

The extended parity trajectory used 96 fixed particles, 7 tasks, separate v15
`Corr_t`, signal uncertainty, 10 adaptive selections/updates, and the production
AD6 thresholds. Selected tasks/segments and verdict labels matched exactly.
Continuous values passed at `1e-6`; aggregate ESS passed at `1e-5`, reflecting
the documented browser normal-CDF approximation relative to SciPy.

## Original Python regression selection

The mapped original tests were run directly from the workspace:

```text
tests/test_phase2_port.py
tests/test_cortex_engine_inputs.py
tests/test_cortex_session_controller.py
tests/test_cortex_ad6_integration.py
tests/test_ell_star_v15.py
retired native bank/fetch suite (historical; no longer executable)
```

Result: **69 passed, 1 skipped**.

## Full browser regression

- TypeScript typecheck: **passed**.
- Full Vitest suite: **162 passed, 4 skipped, 1 failed**.

The single failure is the existing real-bank behavioral assertion:

```text
engine/session.test.ts
clearly-unskilled rater FAILs the bulk of tasks
expected at least 4 FAIL domains; observed 3
```

That test took approximately 100 seconds; the three real-bank session cases
took approximately 310 seconds together. This same failure was reproduced
before the isolation work. No `cortex_web` file was changed, and all isolated
source/data hashes match the unchanged originals, so it is a baseline engine
behavior—not a copy/move regression.

## Claim boundary

The Python and TypeScript implementations are decision-equivalent on the
deterministic parity contract. They are not byte-identical for a freely sampled
full session because their documented RNGs differ (NumPy versus xoshiro256**)
and their normal-CDF implementations differ at approximately `1e-7`.
