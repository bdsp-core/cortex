# Frozen Python engine inputs

This directory contains versioned research/reference inputs and their
provenance manifests. The canonical web runtime uses its separately prepared
bundle manifest; it does not read these CSV files at request time.

## Current K=7 artifacts

| File | Role |
|---|---|
| `sdt_fits_k7.csv` | seven-domain per-rater SDT fits |
| `cross_domain_rater_matrix_k7.csv` | governed cross-domain panel used to derive the K=7 prior |
| `MANIFEST_k7.json` | producer, dimensions, source hashes, and K=7 artifact hashes |

The K=7 domain order is `spike, sz, lpd, gpd, lrda, grda, iic`. The K=7
manifest records 16,765 fit rows and the hashes of the fitted inputs and joint
posterior products used by the dated Phase-9 rebuild.

## Historical/reference artifacts

| File | Role |
|---|---|
| `sdt_fits.csv` | six-domain Phase-3 IIIC reference fit |
| `sdt_fits.spike.csv` | spike reference fit |
| `cross_domain_rater_matrix.csv` | historical reconciled panel |
| `cross_domain_rater_matrix.q2locked.csv` | historical locked candidate panel |
| `sdt_fits.legacy_oldcorpus.csv` | frozen pre-unification comparison only; never rebuild or use as current input |
| `MANIFEST.json` | Phase-5 six-domain provenance and hashes |

The historical files remain for reproduction and drift detection. They must
not silently replace the K=7 artifacts.

## Identity and publication boundary

Cross-domain panels may contain identifying or joinable clinician fields. They
are governed artifacts even when the fitted numeric values are scientifically
publishable. Before a public release, inspect the exact columns and either
remove identity fields or publish a non-reversible reviewer table.

Current identity reconciliation uses the approved
`(source_namespace, source_rater_id)` crosswalk. Never join a domain-local
numeric rater ID to `data/labels/raters.csv` by numeric equality, and never use
fuzzy names as an automatic replacement. Missing, ambiguous, or conflicting
mappings fail; unlinked people remain distinct source-scoped identities.

## Producers and verification

- `pipeline/run_unified_calibration.py` owns the six-domain reference
  calibration products.
- `scripts/build_engine_inputs_k7.py` assembled the frozen K=7 fit tables from
  the pinned reference inputs.
- `scripts/build_engine_inputs.py` verifies/regenerates the Phase-5 manifest;
  it does not create a new scientific fit.

Do not hand-edit generated CSVs or manifests. A replacement must be produced by
the governed pipeline, record every input/output SHA-256, pass domain/shape and
crosswalk validation, and be reviewed as a new calibration artifact.

The root `Sigma_l_fitted*.npy` and certification configuration files are
separate frozen inputs. Their path and hash must be resolved through the code
and manifests rather than inferred from this narrative.
