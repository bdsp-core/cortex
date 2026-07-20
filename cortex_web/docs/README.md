# CORTEX web documentation

The code and tests are the source of truth. This index separates active
contracts from historical implementation records so reviewers do not have to
guess which document governs a release.

## Authoritative

- `../README.md` — repository entry point and qualification commands.
- `REPOSITORY_STRUCTURE.md` — ownership boundaries and dependency rules.
- `PLATFORM_ARCHITECTURE.md` — application and deployment architecture.
- `WEB_WORKER_ARCHITECTURE.md` — deterministic browser-compute contract.
- `WEB_WORKER_QUALIFICATION.md` — numerical, regression, and performance evidence.
- `MODULARIZATION_REPORT.md` — repository-hygiene changes, module boundaries,
  compatibility strategy, and qualification evidence.
- `../LOCAL_DEV.md` — local development and production-shaped smoke setup.
- `../deploy/README.md` — operational deployment, backup, and restore runbook.

## Product and data design records

- `DATA_ARCHITECTURE_PLAN.md`
- `PRECISION_POLICY_ACCOUNT_PILOT.md`
- `PRIVACY_POLICY_DRAFT.md`
- `TERMS_OF_SERVICE_DRAFT.md`

These records provide rationale but cannot override current code, tests, or
the authoritative PrecisionPolicy contract in the Python reference.

## Historical records

- `OPTIMIZATION_PASS_2026-07.md`
- `V15_STAGING.md`

Historical records describe completed work and are not release instructions.
The obsolete root `PLAN.md` and `STATUS.md` were removed during repository
hygiene because their phase model and paths no longer described the current
application.
