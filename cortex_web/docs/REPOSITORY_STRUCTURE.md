# Repository structure and dependency rules

This document defines ownership boundaries for the deployed CORTEX web
application. Changes that cross a boundary must retain the tests at that
boundary and document the reason in the pull request.

## Runtime directories

| Path | Owner | May depend on |
|---|---|---|
| `apps/web/engine/` | deterministic certification computation | engine modules and typed worker protocols; never React, API, cuts, or media rendering |
| `apps/web/engine/index.ts` | declared engine public surface | stable value/type and worker-wire contracts only; UI imports may not bypass it |
| `apps/web/src/features/` | participant-facing product features | shared UI/API contracts and feature-local modules |
| `apps/web/src/features/eeg/` | shared exam/trainer display pipeline | bundle data, montage/DSP helpers, and UI theme; never inference state |
| `apps/web/src/features/exam/` | result derivation/presentation helpers | public engine result types and pure UI-facing calculations |
| `apps/web/src/api/` | authentication-aware transport and domain API clients | transport primitives only; feature code consumes the compatibility facade |
| `apps/web/src/shared/` | reusable browser infrastructure | no feature-specific views |
| `apps/web/src/components/` | compatibility entry points and genuinely shared components | features and shared infrastructure |
| `apps/web/trainer/` | thin browser trainer adapter | authenticated server-session contracts; never a local learning-model fallback |
| `services/api/routers/` | authenticated HTTP boundary | application services and repositories |
| `services/api/persistence/` | additive schema declarations and migration registry | database-neutral schema metadata only |
| `services/api/reporting.py` | read-only administrative aggregations | database query surface and dashboard interpretation; never persistence writes |
| `services/api/rollout.py` | shared identity matching for server-owned gates | participant lookup values supplied by the independent policy/compute/trainer callers |
| `services/api/timeutil.py` | canonical fixed-width UTC timestamps | Python standard library only |
| `services/api/` | API services, persistence, security, and configuration | no browser implementation imports |
| `learning-engine-cleaned/` | runtime-authoritative server training package | its own package and adapter tests |

`research/` is non-serving analysis code. `deploy/` contains operational
assets only. Neither may become an implicit runtime dependency of the browser.

## Accuracy boundaries

- The Python reference is the scientific contract.
- Certification cuts remain downstream reporting only.
- Worker scheduling and performance telemetry never enter selection, posterior
  updates, stopping, intervals, or cut classification.
- Engine state crosses worker boundaries only through versioned typed messages
  and exact snapshots.
- Existing exports from compatibility entry points remain stable while modules
  are decomposed; callers are migrated separately.
- Browser code imports engine contracts through `engine/index.ts`. A new
  deep import from `src/` is a deliberate boundary violation and is rejected
  by `engine_public_surface.test.ts`.
- The exam viewers and trainer share `features/eeg/useEegDisplay.ts`; display
  filtering remains presentation-only and cannot enter inference.

## Generated and sensitive artifacts

Never commit:

- EEG or spectrogram banks;
- SQLite databases, WAL/SHM files, JWT secrets, credentials, or result exports;
- `node_modules`, Python caches, build output, coverage output, screenshots, or
  Playwright reports;
- production `RELEASE` stamps.

Production builds emit no public source maps. Private symbolication jobs may
set `CORTEX_BUILD_SOURCEMAP=1` and upload the hidden maps to protected storage;
those maps are not deployed with the public SPA.

The database, `/etc/cortex/cortex.env`, and `/opt/cortex/bundle` are persistent
deployment state and remain outside immutable application releases.

## Required gates

Every behavior-bearing step must pass the closest unit tests and TypeScript
typecheck or Python tests. Before a replacement release, all of the following
are required:

1. full Vitest and pytest suites;
2. production build and CSP verification;
3. deterministic Precision and AD6 goldens;
4. real-browser coordinator/helper parity and participant UI smoke;
5. dependency audit;
6. release-candidate provenance from a clean commit, with no unrelated or
   generated tracked diff.

Production releases must also be clean, pushed `origin/main` commits. The
deployment builds a versioned tree before changing the stable application path
and retains the prior release for automatic or manual rollback. Test source
patterns are qualification inputs and are intentionally not copied into the
runtime release tree.

The completed decomposition and its compatibility strategy are recorded in
`MODULARIZATION_REPORT.md`.

`scripts/quality_gate.sh` is the executable form of these gates. Canonical CI
must invoke it on pull requests and pushes to `main`; the browser form sets
`CORTEX_BROWSER_GATES=1` after installing Chromium.
