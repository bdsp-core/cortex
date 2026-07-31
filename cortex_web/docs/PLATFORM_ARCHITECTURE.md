# CORTEX web platform architecture

Status: implemented. `cortex_web/` is the canonical product monorepo;
`app.cortexeeg.org` serves the participant application, and the API remains
same-origin under `/api`. This document records the as-built ownership model,
not the superseded migration plan.

The dated live release/configuration snapshot is maintained separately in
[`PRODUCTION_BASELINE.md`](PRODUCTION_BASELINE.md). Architecture changes must
update that handoff when they change an operational assumption.

## Public topology

```text
cortexeeg.org / www
        └─ HTTPS redirect → app.cortexeeg.org

app.cortexeeg.org
        └─ Caddy
            ├─ static SPA from apps/web/dist
            ├─ governed EEG bundle under /bundle
            └─ /api → FastAPI service
```

The retired temporary host redirects to the canonical application. Concrete
cloud resource identifiers, host addresses, credentials, and environment
values are operational state and are deliberately not published here.

## Monorepo layout

```text
cortex_web/
  apps/
    web/                    React SPA and deterministic certification engine
    marketing/              redirect/static marketing surface
  services/
    api/                    FastAPI service, security, persistence, routes
  learning-engine-cleaned/  runtime-authoritative server trainer
  research/                 non-serving export and analysis utilities
  deploy/                   Caddy, systemd, backup, and release automation
  docs/                     architecture, qualification, and operations
  package.json              npm workspace root
```

This nesting avoids a name collision between the TypeScript certification
engine and the repository-root Python research engine.

## Serving and dependency boundaries

- Caddy serves immutable static output and the externally managed bundle.
- Uvicorn serves API routes only; it does not own the SPA build.
- Browser API clients default to same-origin `/api`. A future API-origin split
  is isolated behind `VITE_API_BASE` and the server CORS configuration.
- Bundle location is configuration-owned rather than inferred from source-tree
  layout.
- `POST /api/session` carries the authenticated per-sitting bank, profile, and
  compute/policy stamps. There is no separate manifest API in the live surface.
- Browser selection and posterior updates do not wait on the API. Progress
  requests persist resumability checkpoints only.
- `GET /api/bootstrap` is the dashboard-entry aggregate; independent routes
  remain for section-specific refreshes. The full active-session payload is
  loaded only at test preflight.
- Training trajectories are server-authoritative: `GET /api/trajectories` is
  read-only and owned `POST /api/training-progress` checkpoints write real
  rows. The retired trajectory-write and training-session-list routes remain
  absent.
- `apps/web/engine/` never imports React, API routes, trainer code, media
  rendering, or certification cuts.
- `learning-engine-cleaned/` owns the server trainer; `apps/web/trainer/` is a
  thin authenticated adapter and has no local model fallback.
- `research/` and repository-root Python packages are not serving
  dependencies.

Detailed module rules are in `REPOSITORY_STRUCTURE.md`; deployment and rollback
commands are in `../deploy/README.md`.

## Release model

Production releases are built from a clean pushed commit into an immutable
versioned directory. Database state, the EEG bundle, secrets, and environment
configuration remain outside release directories. Activation changes one
stable application pointer only after build and deep-health checks; the prior
release is retained for health-gated rollback.

A release candidate must first pass locked dependency installation, lint, type
checking, TypeScript and Python tests, production build, CSP verification,
dependency audit, real-browser worker parity, and participant UI smoke. The
deployment script then stages a runtime-only tree, rebuilds on the host,
compiles Python, activates atomically, and runs deep-health and live phone
smoke. The deploy itself does not replace the pre-deploy test gate. See
`WEB_WORKER_QUALIFICATION.md` for the numerical boundary.

## Future expansion hooks

- An independent API hostname can be enabled through `VITE_API_BASE`, CORS,
  DNS, and a Caddy route without changing application-domain logic.
- The bundle URL may become an absolute CDN/object-store URL without changing
  engine selection or media identity.
- The marketing surface can move independently because it does not share SPA
  runtime code.

These are extension seams, not scheduled migrations.
