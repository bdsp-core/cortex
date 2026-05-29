"""CORTEX web backend — FastAPI.

Tiny by design (PLAN §3, §8): the engine runs entirely in the participant's
browser, so the server only (a) authenticates, (b) hands out the bundle URL,
and (c) ingests results. No per-question traffic. A single small instance
serves 100 concurrent participants because there is no shared compute.

Endpoints (all JSON, prefix /api):
    GET  /api/health                      liveness
    POST /api/auth      {code, password}  → {token, expiresIn}
    GET  /api/manifest  (Bearer)          → {bundleUrl, version, sessionSample}
    POST /api/session   (Bearer) {participant, sampleSeed} → {sessionId}
    POST /api/progress  (Bearer) {sessionId, trial}        → {ok}
    POST /api/results   (Bearer) {sessionId, result, stopReason, nQuestions} → {ok}
    GET  /api/admin/participants  (X-Admin-Token)          → [...]
    POST /api/admin/participants  (X-Admin-Token) {count, prefix} → [{code,password}]
    GET  /api/admin/sessions      (X-Admin-Token)          → [...]
    GET  /api/admin/results/{id}  (X-Admin-Token)          → {...}

Run locally:
    uvicorn server.app:app --reload --port 8000        (from cortex_web/)
or  python -m server.run
"""
from __future__ import annotations

import os
import secrets
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import security
from .db import Database

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent                       # cortex_web/
BUNDLE_DIR = WEB_ROOT / "public" / "bundle"
DIST_DIR = WEB_ROOT / "dist"

# Default bundle the SPA pulls (overridable via env for S3/CloudFront).
DEFAULT_BUNDLE_URL = os.environ.get("CORTEX_BUNDLE_URL", "/bundle/v1.1-local")
DEFAULT_SESSION_SAMPLE = int(os.environ.get("CORTEX_SESSION_SAMPLE", "500"))
TOKEN_TTL = int(os.environ.get("CORTEX_TOKEN_TTL", str(6 * 3600)))


# ───────────────────────── request models ─────────────────────────

class AuthIn(BaseModel):
    code: str
    password: str


class SessionIn(BaseModel):
    participant: dict[str, Any] = Field(default_factory=dict)
    sampleSeed: Optional[int] = None


class ProgressIn(BaseModel):
    sessionId: str
    trial: dict[str, Any]


class ResultsIn(BaseModel):
    sessionId: str
    result: dict[str, Any] = Field(default_factory=dict)
    stopReason: Optional[str] = None
    nQuestions: Optional[int] = None


class AdminGenIn(BaseModel):
    count: int = Field(ge=1, le=1000)
    prefix: str = "cortex"
    label: str = ""


# ───────────────────────── app factory ─────────────────────────

def create_app(db_path: Optional[str | Path] = None) -> FastAPI:
    app = FastAPI(title="CORTEX Web API", version="1.0")
    db = Database(db_path)
    app.state.db = db

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

    # ── auth dependencies ──────────────────────────────────────
    def require_auth(authorization: str = Header(default="")) -> str:
        if not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "missing bearer token")
        token = authorization[7:].strip()
        try:
            claims = security.decode_token(token)
        except security.TokenError as e:
            raise HTTPException(401, f"invalid token: {e}")
        return claims["sub"]

    def require_admin(x_admin_token: str = Header(default="")) -> bool:
        expected = os.environ.get("CORTEX_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(503, "admin API disabled (set CORTEX_ADMIN_TOKEN)")
        if not secrets.compare_digest(x_admin_token, expected):
            raise HTTPException(403, "bad admin token")
        return True

    # ── public ─────────────────────────────────────────────────
    @app.get("/api/health")
    def health():
        return {"ok": True, "service": "cortex-web", "version": app.version}

    @app.post("/api/auth")
    def auth(body: AuthIn):
        row = db.get_participant(body.code)
        if row is None or not row["active"]:
            raise HTTPException(401, "invalid credentials")
        if not security.verify_password(body.password, row["password_hash"]):
            raise HTTPException(401, "invalid credentials")
        token = security.issue_token(body.code, ttl_seconds=TOKEN_TTL)
        return {"token": token, "expiresIn": TOKEN_TTL, "code": body.code}

    # ── gated ───────────────────────────────────────────────────
    @app.get("/api/manifest")
    def manifest(_code: str = Depends(require_auth)):
        return {
            "bundleUrl": DEFAULT_BUNDLE_URL,
            "version": "v1.1-local",
            "sessionSample": DEFAULT_SESSION_SAMPLE,
        }

    @app.post("/api/session")
    def new_session(body: SessionIn, code: str = Depends(require_auth)):
        session_id = uuid.uuid4().hex
        db.create_session(session_id, code, body.participant, body.sampleSeed)
        return {"sessionId": session_id}

    @app.post("/api/progress")
    def progress(body: ProgressIn, code: str = Depends(require_auth)):
        sess = db.get_session(body.sessionId)
        if sess is None or sess["code"] != code:
            raise HTTPException(404, "unknown session")
        db.upsert_trial(body.sessionId, body.trial)
        return {"ok": True}

    @app.post("/api/results")
    def results(body: ResultsIn, code: str = Depends(require_auth)):
        sess = db.get_session(body.sessionId)
        if sess is None or sess["code"] != code:
            raise HTTPException(404, "unknown session")
        db.store_result(body.sessionId, body.result)
        db.finalize_session(body.sessionId, body.stopReason, body.nQuestions)
        return {"ok": True}

    # ── admin ───────────────────────────────────────────────────
    @app.get("/api/admin/participants")
    def admin_list(_: bool = Depends(require_admin)):
        return [dict(r) for r in db.list_participants()]

    @app.post("/api/admin/participants")
    def admin_gen(body: AdminGenIn, _: bool = Depends(require_admin)):
        created = []
        for _i in range(body.count):
            code = f"{body.prefix}-{secrets.token_hex(4)}"
            password = secrets.token_urlsafe(9)
            db.add_participant(code, security.hash_password(password), body.label)
            created.append({"code": code, "password": password})
        return created

    @app.get("/api/admin/sessions")
    def admin_sessions(_: bool = Depends(require_admin)):
        return [dict(r) for r in db.all_sessions()]

    @app.get("/api/admin/results/{session_id}")
    def admin_result(session_id: str, _: bool = Depends(require_admin)):
        res = db.get_result(session_id)
        if res is None:
            raise HTTPException(404, "no result for session")
        return res

    # ── static (bundle + SPA), only if present ──────────────────
    if BUNDLE_DIR.exists():
        app.mount("/bundle", StaticFiles(directory=str(BUNDLE_DIR)), name="bundle")
    if DIST_DIR.exists():
        # SPA catch-all LAST so /api/* and /bundle win.
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="spa")

    @app.exception_handler(HTTPException)
    async def _http_exc(_req, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    return app


app = create_app()
