# Repository hygiene and modularization report

Date: 2026-07-19  
Scope: developed in `cortex_web_optimized`, then promoted into canonical
`cortex_web` on the clean release branch

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

- Moved additive schema declarations into
  `services/api/persistence/migrations.py`; the existing database facade still
  applies them in the original order.
- Centralized fail-closed server-owned policy and compute-rollout decisions in
  `services/api/policy_rollout.py`.
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

## Deliberate stopping point

Large remaining files such as the shell orchestrator and database repository
are cohesive integration surfaces. Splitting them further in this pass would
increase migration surface without improving the accuracy or runtime boundary.
Future extractions should be driven by an independently testable domain, not a
line-count target.
