# PrecisionPolicy

This project is the standalone Python implementation of the frozen
`frontier_p90guard_m3` PrecisionPolicy. It is cut-independent: certification
cuts are never constructor inputs and never enter stopping or selection.

The frozen runtime reports raw equal-tailed central 95% marginal intervals.
Its stopping statistic is separately computed as the point-centred radius
`max(mean - low, high - mean)`, guarded by the pinned quantile-MCSE constants.
The optional historical calibration hooks remain available in the generic
class for reproducibility, but the `build_frozen_precision_policy` factory
does not accept them and asserts that all operative calibration fields are
disabled.

`FROZEN_PRECISION_PROFILE` also pins the controller settings that travel with
the stopping policy: 1,200 particles, `Corr_l`/`Corr_t`, `total_var`, the
deterministic uncertainty-aware 128-candidate scan, randomized top-10 opener,
five-question same-domain limit, floor-progress deadline, and 60 questions per
domain (therefore at most 420 for K=7).

## Test in isolation

From this directory:

```bash
python -m pytest
```

For an editable development install:

```bash
python -m pip install -e ../termination-policy -e .
```

The accepted extreme-skill interval-undercoverage limitation is documented in
the Python reference evidence. It is deliberately disclosed, not recalibrated
or changed here.
