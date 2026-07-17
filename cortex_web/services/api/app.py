"""CORTEX web backend — FastAPI app factory.

Tiny by design (PLAN §3, §8): the engine runs entirely in the participant's
browser, so the server only (a) authenticates, (b) hands out the bundle URL +
the per-session question draw, and (c) ingests results. No per-question
compute. A single small instance serves 100 concurrent participants because
there is no shared compute.

Layout (one module per concern; routers are thin over these):
    config.py           env-derived paths/constants + the RELEASE stamp
    models.py           pydantic request models
    deps.py             require_auth / require_admin / client_ip / RateLimiter
    helpers.py          validation, auth-code lifecycle, Google verify, report email
    dashboard_logic.py  result→dashboard derivation, truth maps, question breakdown
    routers/            auth, account, testing, dashboard, report, admin
    db.py               Database (SQLite dev / pooled Postgres prod)

Endpoint surface (all JSON, prefix /api). Public: health, register,
verify/{confirm,resend,status}, auth, auth/google, forgot, reset, report,
client-error (SPA crash telemetry; rate-limited, log-only),
ses/events (SNS webhook; capability-token-gated, 404 unless enabled). Bearer-
gated: manifest, tutorial-example, session, progress, results, dashboard,
bootstrap (the dashboard-entry sections in one round-trip),
activity, history (+ /{id}/questions), regimen, trajectories,
training-sessions (+ /finalize), consent (+ /withdraw), profile,
account/{password,email}. X-Admin-Token-gated: admin/participants,
admin/sessions, admin/results/{id}. GET /api/health?deep=1 additionally
checks the DB (503 when it fails) — point uptime monitors at that.

Run locally (from cortex_web/services/):
    uvicorn api.app:app --reload --port 8000
or  python -m api.run

Production runs exactly ONE uvicorn worker (see api/run.py + the systemd
unit): the in-memory rate limiter, the lazily-loaded SessionBank, and the
boot migrations all assume a single process.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, digest
from .db import Database
from .ops_alerts import OpsAlerter
from .routers import ALL_ROUTERS
from .session_bank import SessionBank


def create_app(db_path: Optional[str | Path] = None) -> FastAPI:
    db = Database(db_path)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # Ripeness-digest scheduler (digest.py): in-process, single-worker,
        # same assumption as the rate limiter. The initial delay means
        # short-lived test clients never run a pass; CORTEX_DIGEST_DISABLED=1
        # is the ops kill switch.
        digest_task = None
        if not os.environ.get("CORTEX_DIGEST_DISABLED"):
            digest_task = asyncio.create_task(digest.scheduler_loop(db))
        yield
        if digest_task is not None:
            digest_task.cancel()
            with suppress(asyncio.CancelledError):
                await digest_task
        db.close()   # release the DB pool/connection on clean shutdown

    app = FastAPI(title="CORTEX Web API", version="1.0", lifespan=_lifespan)

    # ── per-session question bank (server-side balanced draw) ──
    # Bundle config is read here (not at import) so tests/dev can point it at a
    # fixture bundle via env. The bank (the full signal index = the bundle
    # manifest) loads LAZILY on first /api/session and is cached for the app's
    # lifetime; a missing manifest (CI / fresh box) leaves it None and yields a
    # clear 503 instead of breaking app construction or unrelated endpoints.
    bundle_url = os.environ.get("CORTEX_BUNDLE_URL", config.DEFAULT_BUNDLE_URL)
    bundle_dir = Path(os.environ.get("CORTEX_BUNDLE_DIR", str(config.BUNDLE_DIR)))
    _bank_cache: dict[str, Optional[SessionBank]] = {}
    _bank_lock = threading.Lock()   # two first-requests must not both parse the 35 MB manifest

    def get_session_bank() -> Optional[SessionBank]:
        with _bank_lock:
            if "bank" not in _bank_cache:
                version = bundle_url.rstrip("/").split("/")[-1]
                mpath = bundle_dir / version / "manifest.json"
                _bank_cache["bank"] = (
                    SessionBank(mpath, bundle_url) if mpath.exists() else None)
            return _bank_cache["bank"]

    # Shared state the routers read (see routers/__init__.py).
    from .deps import RateLimiter
    app.state.db = db
    app.state.limiter = RateLimiter()
    app.state.get_bank = get_session_bank
    # Ops error alerting (ops_alerts.py): backend 500s + SPA crash telemetry
    # email the operator, cooldown-collapsed. Empty CORTEX_OPS_ALERT_TO
    # disables it.
    app.state.alerts = OpsAlerter(
        to_addr=os.environ.get("CORTEX_OPS_ALERT_TO", config.REPORT_TO),
        cooldown_s=int(os.environ.get("CORTEX_OPS_ALERT_COOLDOWN_S",
                                      str(6 * 3600))),
    )
    app.state.cfg = {
        "bundle_url": bundle_url,
        "session_sample": int(os.environ.get("CORTEX_SESSION_SAMPLE",
                                             str(config.DEFAULT_SESSION_SAMPLE))),
        "spacing_days": int(os.environ.get("CORTEX_SPACING_DAYS", "30")),
        "spacing_sessions": int(os.environ.get("CORTEX_SPACING_SESSIONS", "3")),
        "training_mode": os.environ.get("CORTEX_TRAINING_MODE", "all").strip().lower(),
        "training_allowlist": frozenset(
            x.strip().lower()
            for x in os.environ.get("CORTEX_TRAINING_ALLOWLIST", "").split(",")
            if x.strip()),
        # Engine-trainer exposure. Default ALL (2026-07-17 integration
        # decision: the learning engine IS the production trainer; the
        # incumbent client-side trainer is ARCHIVED as the fallback).
        # Kill switch: CORTEX_TRAINER_ENGINE=off reverts every sitting to
        # the incumbent with zero deploys; "cohort" + allowlist scopes it.
        "trainer_engine": os.environ.get(
            "CORTEX_TRAINER_ENGINE", "all").strip().lower(),
        "trainer_engine_allowlist": frozenset(
            x.strip().lower()
            for x in os.environ.get(
                "CORTEX_TRAINER_ENGINE_ALLOWLIST", "").split(",")
            if x.strip()),
    }

    origins = os.environ.get(
        "CORTEX_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Defense-in-depth security headers on the API's OWN responses. In prod
    # Caddy fronts /api and sets the site-wide headers, but these travel WITH
    # the API — so they hold if the service is ever reached directly (the
    # deferred api.cortexeeg.org split, a health probe hitting :8000, a
    # future CDN). Scoped to /api so the optional dev static-serving path
    # (CORTEX_SERVE_STATIC) is untouched. The API only ever returns JSON or a
    # file download, so a `default-src 'none'` CSP can't break anything, and
    # `Cache-Control: no-store` keeps authed JSON (dashboard/history/profile)
    # out of shared and back/forward caches — a real, live improvement that
    # Caddy does not set for /api.
    @app.middleware("http")
    async def _api_security_headers(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/api"):
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            response.headers.setdefault("Cache-Control", "no-store")
            response.headers.setdefault("Content-Security-Policy",
                                        "default-src 'none'; frame-ancestors 'none'")
            response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        return response

    # Request-body cap, mirroring the Caddy edge limit (4 MB) so the API
    # self-protects when reached directly (health probe on :8000, a future
    # split deploy). Declared-length check only — Caddy is the authoritative
    # enforcement at the edge; this stops the honest-but-huge payload (an
    # unbounded result/points blob) before it is buffered/parsed.
    _MAX_BODY = 4 * 1024 * 1024

    @app.middleware("http")
    async def _api_body_cap(request, call_next):
        if request.url.path.startswith("/api"):
            try:
                declared = int(request.headers.get("content-length") or 0)
            except ValueError:
                declared = 0    # malformed header → uvicorn rejects it anyway
            if declared > _MAX_BODY:
                return JSONResponse(status_code=413,
                                    content={"error": "request body too large"})
        return await call_next(request)

    @app.get("/api/health")
    def health(deep: int = 0):
        """Liveness (default) or readiness (?deep=1). The deep probe verifies
        the DB round-trip — the failure mode that took auth down for a day on
        2026-06-24 while the shallow probe kept returning 200 — and reports the
        bank state WITHOUT forcing the lazy 35k-manifest load. Deep failures
        return 503 so external uptime monitors alert on them."""
        info: dict[str, Any] = {"ok": True, "service": "cortex-web",
                                "version": app.version, "release": config.RELEASE}
        if deep:
            try:
                db.ping()
                info["db"] = "ok"
            except Exception as e:
                info["ok"] = False
                info["db"] = f"error: {type(e).__name__}"
            info["bank"] = ("not_loaded_yet" if "bank" not in _bank_cache
                            else "loaded" if _bank_cache["bank"] is not None
                            else "missing")
            if not info["ok"]:
                return JSONResponse(status_code=503, content=info)
        return info

    for r in ALL_ROUTERS:
        app.include_router(r)

    # ── static (bundle + SPA): OPTIONAL — prod serves these via Caddy ──
    if config.SERVE_STATIC and config.BUNDLE_DIR.exists():
        app.mount("/bundle", StaticFiles(directory=str(config.BUNDLE_DIR)), name="bundle")
    if config.SERVE_STATIC and config.DIST_DIR.exists():
        # SPA catch-all LAST so /api/* and /bundle win.
        app.mount("/", StaticFiles(directory=str(config.DIST_DIR), html=True), name="spa")

    @app.exception_handler(HTTPException)
    async def _http_exc(_req, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(Exception)
    async def _unhandled_exc(req, exc: Exception):
        # One greppable journal line (Starlette re-raises after this handler,
        # so the server still logs the full traceback) + an ops email so 500s
        # get noticed instead of resting in the journal. The response body
        # stays generic — no internals leak to the client.
        print(f"[cortex.servererr] method={req.method} path={req.url.path} "
              f"err={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        app.state.alerts.notify(
            "server-error",
            f"{req.method} {req.url.path} → {type(exc).__name__}: {exc}")
        return JSONResponse(status_code=500,
                            content={"error": "internal server error"})

    return app


app = create_app()
