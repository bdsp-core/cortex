# CORTEX policy implementation layout

The Python stopping policies are organized as small, independently testable
projects at the repository root. The TypeScript application is intentionally
outside this implementation phase and has not been modified.

```text
termination-policy/                 shared StopDecision/TerminationPolicy contract
ad6-policy/                         standalone shipped AD6 policy
precision-policy/                   standalone frozen PrecisionPolicy + profile
cortex_web_python_reference/
  scripts/cortex_policy.py          compatibility imports and AD6 config adapter
  scripts/cortex_policy_k7.py       K=7 policy registry and safe name resolver
  scripts/session_controller.py     local policy-aware session factory
```

## Ownership boundaries

- `ad6-policy` owns AD6's monotone PASS/FAIL/PENDING verdict logic. It accepts
  a cut vector because AD6 is inherently cut-aware. Configuration-file loading
  remains in the reference adapter.
- `precision-policy` owns cut-independent stopping, fresh-derived domain
  statuses, the frozen policy factory, the frozen controller profile, and a
  separate downstream reporting helper.
- `termination-policy` owns only the shared Python interface. It has no NumPy,
  selector, cut, or engine dependency.
- `cortex_web_python_reference` connects the packages to the particle engine.
  Historical `from cortex_policy import ...` callers remain compatible.

There is one canonical copy of each concrete policy implementation. The
reference façade does not duplicate their stopping math.

## Local policy selection and rollback

The explicit local entry point is:

```python
from session_controller import build_cortex_session

session = build_cortex_session(inputs)
```

It reads `CORTEX_TERMINATION_POLICY` with these values:

- `ad6` — default and rollback path.
- `precision_v1` — frozen PrecisionPolicy plus its complete session profile.

An unknown value fails closed. Existing research and unit-test callers may
continue to inject a policy directly into `CortexSession`; that path does not
claim the frozen release profile.

## Frozen PrecisionPolicy profile

`build_frozen_precision_policy` accepts no calibration controls. It pins:

- task order `spike, sz, lpd, gpd, lrda, grda, iic`;
- raw equal-tailed central 95% reported skill and bias intervals;
- point-centred skill radius for stopping;
- `reliability_mode="quantile_mcse"`;
- MCSE constants `z=1.645` and inflation `1.5962415320776275`;
- task tolerance multipliers `[1.50, 1.30, 1.35, 1.30, 1.30, 1.30, 1.25]`;
- 20 own-domain questions, three per signal tercile, ESS at least `0.5N`, and
  two consecutive successful evaluations;
- 1,200 particles with `Corr_l` for skill and `Corr_t` for bias;
- `total_var`, uncertainty-aware 128-candidate coarse-to-fine selection, a
  randomized top-10 first item, five-question same-domain limit, and the
  floor-progress deadline; and
- 60 questions per domain, hence at most 420 for K=7. There is no independent
  420-question runtime constant.

The generic class retains default-off historical calibration hooks solely for
reproducibility. The frozen factory asserts the operative scale/ramp/bias
fields are `None`; no `g` parameter belongs to the frozen candidate.

Certification cuts are unavailable to PrecisionPolicy. Reporting callers must
use `classify_determined_interval_against_cut`, which rejects ACTIVE and
UNDETERMINABLE domains before returning `ABOVE_CUT`, `BELOW_CUT`, or
`INDETERMINATE_AT_CUT`.

The disclosed extreme-skill interval-undercoverage limitation is accepted and
unchanged. This implementation does not calibrate, tune, or introduce a new
stopping family.

## Verification

Run the complete local gate from the repository root:

```bash
./run_policy_tests.sh
```

This runs all three standalone package suites, the Python reference suite, the
primary Python algorithm regressions, the two fixed-cloud Python↔TypeScript
parity checks, and the pinned 48 browser regressions. The browser sources are
read-only in this phase.

The heavier selector proofs and 1,200-particle deterministic golden sessions
are opt-in:

```bash
CORTEX_RUN_POLICY_GOLDENS=1 ./run_policy_tests.sh
```

They are per-engine Python drift guards, not claims of full-session
Python↔TypeScript equality; the engines intentionally use different RNGs.
