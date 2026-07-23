# CORTEX percentile policy

This directory is the **governed source-artifact and transition-policy
workspace** for provisional CORTEX domain percentiles. The production
integration lives in `cortex_web`, but it consumes only the immutable compact
runtime built here. Percentiles remain reporting-only and do not control
certification verdicts, stopping, item selection, training allocation, or
mastery.

The initial artifact answers this limited question:

> Where does the candidate's estimated domain skill fall relative to
> quality-screened historical calibration fit records?

It does **not** estimate position among physicians, clinicians, healthcare
workers, or the general population. Product copy must use:

> Provisional historical calibration-cohort percentile

and must show the warning that population-representativeness bias is not
included in the 95% interval.

## What is implemented

- Manifest-pinned ingestion of `data/engine_inputs/sdt_fits_k7.csv`.
- A declared quality screen for nonidentified and optimizer-boundary fits.
- Per-domain two-level bootstrap:
  - resample historical fit rows;
  - draw latent skill from the fit's normal approximation using
    `ell=-log(sigma)` and `SE_ell=SE_sigma/sigma`.
- Immutable K=7 source reference-CDF artifact with a canonical self-hash.
- A compact float32 browser/API runtime with independently verified metadata,
  source-artifact, and binary hashes.
- JSON contracts for candidate input, norm artifacts, and score output.
- Candidate scoring from exact posterior particles and weights.
- Clearly flagged normal approximation when only mean and SD are available.
- A 95% interval combining candidate posterior, historical fit uncertainty,
  and fit-row sampling uncertainty.
- Sensitivity estimates under stricter fit-quality policies.
- Objective, non-automatic gates for replacing the historical artifact with a
  versioned CORTEX first-attempt-user norm.

## Build and verify

From this directory:

```bash
PYTHONPATH=src python -m percentile_policy build
PYTHONPATH=src python -m percentile_policy verify \
  artifacts/historical-calibration-k7-v1.json
PYTHONPATH=src python -m percentile_policy build-runtime
```

The builder rejects a historical CSV whose SHA-256 does not match
`data/engine_inputs/MANIFEST_k7.json`.

Current immutable identities:

- source norm: `e5a532dda595e76e3559efa20a0f8ca45fdaa039bada6e8f99890ac8a3c01262`
- runtime metadata content: `6bb77ba9efb8ef53eb7a0b5cc4d0cb6af48a20cb7482d76017f0e4d9e4f08d1b`
- runtime binary: `c0ca140310b2410d12d61733478c0b8d4999e25e9f77c204a2caa318a7ee4776`

Default build policy:

- converged fit;
- at least 20 trials;
- finite positive `sigma` and finite nonnegative `SE_sigma`;
- `SE_ell <= 1.0`;
- not at the fitter's `sigma` bounds of 0.02 or 5.0;
- 500 bootstrap CDF replicas;
- 201 equally spaced quantile knots;
- fixed seed `20260722`.
- governed creation stamp `2026-07-22T00:00:00Z` (so the default build is
  reproducible; a future norm version must deliberately change it).

The audit report records every inclusion/exclusion count and stricter
sensitivity cohort size.

## Score a candidate

Exact posterior particles are preferred:

```json
{
  "domains": {
    "spike": {
      "ellSamples": [-0.1, 0.0, 0.2],
      "weights": [0.2, 0.3, 0.5]
    }
  }
}
```

```bash
PYTHONPATH=src python -m percentile_policy score \
  artifacts/historical-calibration-k7-v1.json \
  examples/candidate-particles.json
```

For research-only legacy scoring, mean and SD are accepted:

```json
{
  "domains": {
    "spike": {"ellMean": 0.2, "ellSd": 0.15}
  }
}
```

This path is labeled `normal_approximation_from_mean_sd`; it must not be
presented as equivalent to using the engine's weighted posterior cloud.

## Interval interpretation

For candidate particle `m` and bootstrap reference CDF `b`, the implementation
calculates:

```text
rank[m,b] = 100 * F_domain,b(ell_domain,m)
```

The point estimate is the weighted mean over candidate particles and reference
replicas. The lower and upper values are weighted quantiles of the same joint
rank distribution.

The interval includes:

- candidate posterior uncertainty;
- the historical rater-fit standard errors;
- finite historical fit-row sampling uncertainty.

It cannot include unknown bias caused by the historical source mixture,
unreconciled cross-source identities, self-selection, or mismatch with the
future CORTEX user population. The response carries that warning
programmatically.

## Evaluate replacement by CORTEX-user norms

Only the first completed assessment per authenticated user, before CORTEX
training, is eligible for the future reference cohort. Passing the numerical
gates never triggers an automatic switch:

```bash
PYTHONPATH=src python -m percentile_policy evaluate-transition \
  examples/transition-metrics-not-ready.json \
  --policy policy/transition-policy.json
```

Exit status 2 means the cohort is not ready. Even when all gates pass, a new
artifact, scientific review, shadow comparison, and explicit governance
approval are required.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

See [POLICY.md](POLICY.md) for product language, versioning, and cutover rules.
