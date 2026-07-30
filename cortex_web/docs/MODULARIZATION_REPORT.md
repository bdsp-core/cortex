# Repository hygiene and modularization report

Date: 2026-07-19; production addendum 2026-07-22
Scope: developed in an isolated staging worktree, then promoted into canonical
`cortex_web` on a clean release branch. The staging directory was disposable
and is not part of the repository.

This pass reorganized the isolated replacement candidate without changing the
scientific policy, participant result, persistence meaning, or deployment
state. Promotion was performed from a clean `origin/main` worktree so unrelated
research changes could not enter the release.

## Changes made

### Frontend

- Extracted verdict presentation, participant-safe history, protocol,
  authentication controls, and settings into feature-owned modules under
  `apps/web/src/features/`.
- Retained compatibility re-exports where existing callers depended on the
  previous component entry points.
- Split tab-scoped authentication transport into `src/api/core.ts` and cohort
  operations into `src/api/cohorts.ts`; `src/api.ts` remains the stable public
  facade.
- Added an executable engine-boundary test that rejects React, rendering, API,
  dynamic-code, blob-worker, and shared-memory dependencies in deterministic
  certification code.

### Backend

- Moved canonical DDL into `services/api/persistence/schema.py` and retained
  additive column evolution in `services/api/persistence/migrations.py`; the
  database facade still applies them in the original order.
- Kept the independent fail-closed policy and compute decisions in
  `policy_rollout.py` and `compute_rollout.py`, while moving their shared
  identity matching into `rollout.py`.
- Added legacy-schema, migration-idempotence, uniqueness, allowlist, default,
  and unknown-configuration tests.

### Repository and supply chain

- Removed stale planning documents and generated local outputs, expanded
  ignore rules, and added an authoritative documentation index and ownership
  map.
- Added ESLint with zero-warning enforcement, full TypeScript/Python/build/CSP
  gates, and a full dependency audit to `scripts/quality_gate.sh`.
- Upgraded the Vite/Vitest development toolchain to audited versions and made
  production source maps opt-in and hidden.
- Replaced dirty, in-place production updates with clean-main enforcement,
  pre-deploy backup, versioned staging, health-gated activation, and automatic
  one-release rollback.

## Compatibility and accuracy

No engine formulas, PrecisionPolicy/AD6 behavior, selection ordering, RNG
state, persistence schema meaning, or participant output contracts were
changed. The compatibility facade prevents feature extraction from forcing a
simultaneous caller migration. Database migration tests begin with a legacy
SQLite schema and verify both row preservation and repeat application.

The final qualification includes deterministic goldens, fixed-cloud parity,
serial/helper result equality in real Chromium, the full participant UI path,
the actual 35,193-item bank benchmark, production build/CSP verification, and
dependency auditing. Exact counts and timings live in
`WEB_WORKER_QUALIFICATION.md`.

Subsequent native n-way performance work retained these boundaries while
adding deterministic worker sharding, ranked speculation/preemption,
load-guarding, and persistent MH-history caches. Those runtime details are
owned by `WEB_WORKER_ARCHITECTURE.md`; this report remains the modularization
record rather than a second architecture specification.

## 2026-07-22 production addendum

Release `e68d59b` completed the next behavior-preserving extractions:

- `engine/index.ts` now declares the only UI-facing engine surface, enforced by
  `engine_public_surface.test.ts`.
- The shared waveform pipeline lives in `features/eeg/useEegDisplay.ts`; the
  trainer and both exam viewers now use the same tested filter path.
- Result derivation and the washout banner moved out of `App.tsx` into
  `features/exam/results.ts` and `components/exam/WashoutBanner.tsx`.
- Read-only administrative aggregation moved to `reporting.py`, canonical UTC
  formatting to `timeutil.py`, and shared backend helpers to `helpers.py`.
- One upsert definition now generates both SQLite and PostgreSQL statements,
  with dialect-specific contract tests.
- Retired API routes and demo/sample surfaces were removed, validation errors
  use one envelope, and the runtime release archive excludes test-module
  patterns.
- The UI smoke harness now builds the requested source and reliably terminates
  its temporary server, preventing stale-build qualification.

These commits do not alter certification formulas, RNG order, policy,
selection, stopping, or results. The verified release relationship and
rollback anchors are in `PRODUCTION_BASELINE.md`.

## Deliberate stopping point

At the original stopping point, the shell orchestrator and database repository
remained cohesive integration surfaces. The addendum extracted only domains
with independent tests and stable compatibility seams. Further decomposition
should continue to be driven by an independently testable domain, not a
line-count target.
