# CORTEX web platform — domain + monorepo architecture plan

Status: **IMPLEMENTED 2026-06-17** — Phase A (domain cutover to cortexeeg.org,
nip.io retired→301) and Phase B (monorepo restructure: apps/web + services/api,
Caddy file-serves static, uvicorn pure API) are both LIVE on prod. This doc
captures the locked decisions and the as-built architecture.

## Locked decisions

| # | Decision | Choice |
|---|---|---|
| 1 | Domain | `cortexeeg.org` (already owned: Route 53 zone + SES). NOT `.io`. |
| 2 | Subdomains | `cortexeeg.org` = marketing root, `app.cortexeeg.org` = SPA, `api.cortexeeg.org` = backend (later) |
| 3 | Marketing (now) | **Redirect-only** — `cortexeeg.org` 301 → `app.cortexeeg.org`; real marketing later |
| 4 | API split | **Deferred** — app keeps same-origin `/api` under `app.`; `VITE_API_BASE` is the one switch to flip later |
| 5 | Repo layout | **Monorepo restructure** — `apps/web`, `apps/marketing`, `services/api` under the `cortex_web/` container (NOT repo root — avoids colliding with the methodology repo's Python `engine/`) |
| 6 | Serving model | **Caddy serves all static** (SPA + marketing + `/bundle`); **uvicorn = pure API** |
| 7 | Sequencing | **Phase A (domain) first**, then **Phase B (restructure)** |

## Why these are cheap (existing modular seams)

The code already isolates the three things a split needs:
- **API origin** — `apps/web` `src/api.ts:10` `API_BASE = VITE_API_BASE ?? ""` (same-origin default).
- **CORS allow-list** — `server/app.py:252-257` reads `CORTEX_CORS_ORIGINS`.
- **Bundle URL** — `CORTEX_BUNDLE_URL` env; manifest can hand the client an absolute URL.

## Infra facts (verified)

- EC2 `i-006c5c7b11e9cd49c`, SG `launch-wizard-1` (`sg-0a48dd23c8bee356d`).
- **Static Elastic IP `44.233.29.150`** (`eipalloc-09c7c5cd528cd34a7`) — safe DNS target.
- Caddy auto-TLS (Let's Encrypt HTTP-01; ports 80/443 open). One box, one uvicorn :8000.
- Route 53 hosted zone for `cortexeeg.org` already exists (SES).

## Target topology

```
cortexeeg.org / www  ─┐
app.cortexeeg.org     ┼─ A records ─► EIP 44.233.29.150 ─► Caddy :443 ─┬─ cortexeeg.org      → 301 → app.
api.cortexeeg.org(L8r)┘                                                ├─ app.cortexeeg.org  → file_server(SPA) + /bundle(file_server) + /api → uvicorn
                                                                       └─ api.cortexeeg.org  → uvicorn   (LATER)
```

## Phase A — domain cutover (low risk, ship first)  ✅ DONE 2026-06-17

Executed with `deploy/scripts/smoke_domain.sh` (ALL PASS). `app.cortexeeg.org`
serves the app over HTTPS; apex + `www` + the old `nip.io` host all 301→app.
(`nip.io` retired 2026-06-17 — moved to the redirect block + dropped from CORS;
old links still work via the 301). Box config backups:
`/etc/{caddy/Caddyfile,cortex/cortex.env}.bak-20260617T180416Z`. DNS managed via
the scoped `cortexeeg-dns` profile. See the production-architecture memory for
the per-step revert commands.

No source restructure. Steps (each reversible):
1. Route 53: A records `app`, apex `@`, `www` → `44.233.29.150`.
2. Caddy: add `app.cortexeeg.org` vhost (same reverse_proxy config as the `nip.io` vhost today).
3. `/etc/cortex/cortex.env`: `CORTEX_DOMAIN=app.cortexeeg.org`, `CORTEX_CORS_ORIGINS=https://app.cortexeeg.org`. Restart `cortex.service` + reload Caddy.
4. Make `app.` canonical; turn `cortexeeg.org`, `www`, and `cortex-44-233-29-150.nip.io` into **301 redirects → app.** (keeps active-pilot links working; the 6-digit email flow has no magic links, so email is unaffected).

Rollback: revert DNS/redirect + env; the `nip.io` vhost stays until verified.

## Phase B — monorepo restructure (staged, on the stable Phase-A base)

Target tree:

```
cortex_web/                  # web-product monorepo root (npm workspaces: ["apps/*"])
  apps/
    web/                     # SPA: src/, engine/, ui/, public/, index.html, vite.config.ts, package.json
    marketing/               # static; redirect-only for now
  services/
    api/                     # FastAPI: app.py, db.py, security.py, email.py, sample_data.py, tests
  deploy/                    # multi-vhost Caddyfile + scripts + systemd
  package.json               # workspace root
```

Serving model (decision #6): Caddy `file_server`s `apps/web/dist`, `apps/marketing`,
and the EEG `/bundle`. `services/api` **drops the `StaticFiles` mounts**
(`app.py:605-609`) → no cross-package path coupling. uvicorn keeps only the
repo-root `scripts/` access for `/api/videos`, via an **env var** replacing the
current `WEB_ROOT.parent/scripts` relative resolution (`app.py:547`).

### Reference-update checklist
- **Build:** move `src/engine/ui/public/index.html/vite.config.ts` → `apps/web/`; add root + `apps/web` `package.json` (workspaces); marketing build (or static).
- **API:** `server/` → `services/api/`; uvicorn target `server.app:app` → new module path; replace `WEB_ROOT`-relative `DIST_DIR`/`BUNDLE_DIR`/`scripts` (`app.py:58-61,547`) with env-driven paths; drop StaticFiles mounts.
- **Tests:** `services/api` pytest path + `ui_smoke.sh` (`uvicorn server.app:app`); vitest under `apps/web`.
- **Deploy:** `Caddyfile.template` → multi-vhost + `file_server`; `deploy_app.sh` (rsync paths, two npm builds + one pip), `provision.sh`, `systemd/cortex.service` (`WorkingDirectory`/`ExecStart`), `backup_to_box.sh`; on-box `/opt/cortex/cortex_web` re-layout.
- **Docs/memory:** `CLAUDE.md`, `deploy/README.md`, production-architecture memory.

### Staging (each step: build + vitest + pytest green; monolith stays live)
1. Source-only restructure (moves, workspaces, import/path fixes, env-drive static/scripts dirs) → green locally; no deploy.
2. Rewrite `deploy/` for the new layout; dry-run.
3. Provision the new layout **alongside** the old on the box; cut `cortex.service` over; verify; keep old dir for instant rollback.
4. Update docs + memory.

## Future expansion hooks (the modularity payoff)
- Flip `VITE_API_BASE=https://api.cortexeeg.org` + add a Caddy vhost + CORS origin → real API split, zero app-logic change. (Bearer/JWT-in-localStorage → clean cross-origin, no cookies.)
- New surface = one Caddy block + one DNS record; optional wildcard `*.cortexeeg.org` + Caddy DNS-01 (Route 53 plugin) for zero-DNS new subdomains.
- Bundle `bundleUrl` can go absolute → `cdn.`/CloudFront with no client change.
- Marketing is an independent static unit → can move to S3/CloudFront later.
