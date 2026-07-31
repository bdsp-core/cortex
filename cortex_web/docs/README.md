# CORTEX web documentation

The code and tests are the source of truth. This index separates active
contracts from historical implementation records so reviewers do not have to
guess which document governs a release.

## Authoritative

- `PRODUCTION_BASELINE.md` — verified live release, non-secret runtime
  configuration, current API/telemetry assumptions, and rollback anchors.
- `../README.md` — repository entry point and qualification commands.
- `REPOSITORY_STRUCTURE.md` — ownership boundaries and dependency rules.
- `PLATFORM_ARCHITECTURE.md` — application and deployment architecture.
- `WEB_WORKER_ARCHITECTURE.md` — deterministic browser-compute contract.
- `WEB_WORKER_QUALIFICATION.md` — numerical, regression, and performance evidence.
- `WEB_WORKER_ROLLOUT_OPERATIONS.md` — live rollout record, observation
  checklist, aggregate telemetry query, and rollback conditions.
- `PERCENTILE_PREVIEW_RUNBOOK.md` — provisional norm identity, disclosure,
  fail-closed exposure gate, qualification, monitoring, and approval boundary.
- `PERCENTILE_PREVIEW_VALIDATION.md` — local statistical/runtime parity,
  performance, regression, browser-smoke, and release-boundary evidence.
- `MODULARIZATION_REPORT.md` — repository-hygiene changes, module boundaries,
  compatibility strategy, and qualification evidence.
- `../LOCAL_DEV.md` — local development and production-shaped smoke setup.
- `../deploy/README.md` — operational deployment, backup, and restore runbook.

## Product and data design records

- `DATA_ARCHITECTURE_PLAN.md`
- `PRECISION_POLICY_ACCOUNT_PILOT.md`
- `PRIVACY_POLICY_DRAFT.md`
- `TERMS_OF_SERVICE_DRAFT.md`

These records provide rationale but cannot override current code, tests,
`PRODUCTION_BASELINE.md`, or the authoritative PrecisionPolicy contract. The
data plan deliberately preserves its original proposal language; its status
banner identifies the routes and assumptions superseded in production.

## Historical implementation record

- `OPTIMIZATION_PASS_2026-07.md`

This record describes completed work and is not a release instruction. Earlier
staging plans and obsolete root status documents were removed because their
phase models and paths no longer described the current application. Git history
preserves them if implementation archaeology is required.
