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

import json
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from . import security
from .db import Database

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent                       # cortex_web/
BUNDLE_DIR = WEB_ROOT / "public" / "bundle"
DIST_DIR = WEB_ROOT / "dist"

# Default bundle the SPA pulls (overridable via env for S3/CloudFront).
DEFAULT_BUNDLE_URL = os.environ.get("CORTEX_BUNDLE_URL", "/bundle/v1.5-k7")
DEFAULT_SESSION_SAMPLE = int(os.environ.get("CORTEX_SESSION_SAMPLE", "500"))
TOKEN_TTL = int(os.environ.get("CORTEX_TOKEN_TTL", str(6 * 3600)))


# ───────────────────────── request models ─────────────────────────

class AuthIn(BaseModel):
    email: str
    password: str


class RegisterIn(BaseModel):
    """Public signup payload. `honeypot` should be empty — it's a hidden form
    field most bots auto-fill. Any non-empty value gets a 200 OK with no
    side-effects so the bot moves on without learning it was blocked."""
    email: str
    password: str
    displayName: str
    expertise: str = ""        # optional self-reported expertise dropdown
    honeypot: str = ""         # bot trap; must be empty


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


# ───────────────────────── validation helpers ─────────────────
# RFC 5322 is overkill; this matches what every real email service accepts
# and rejects obvious garbage. We DON'T verify deliverability — the policy
# tonight is "anyone with the URL can sign up; we trust the email at face
# value." Add MX-record / SES verification later if spam shows up.
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_MIN_PASSWORD_LEN = 8
_MAX_FIELD_LEN = 200       # tame oversized payloads (display_name etc.)


def _norm_email(s: str) -> str:
    return s.strip().lower()


def _is_email(s: str) -> bool:
    return bool(_EMAIL_RE.match(s)) and len(s) <= _MAX_FIELD_LEN


# ───────────────────────── rate limiter ────────────────────────
# In-memory sliding-window counter per (route, IP). Resets on restart, which
# is fine at this scale — restarts are rare and a determined attacker can do
# damage with a single window anyway. Two routes are protected:
#   /api/register : 5 attempts / hour / IP
#   /api/auth     : 20 attempts / hour / IP

_RATE_LIMITS = {
    "register": (5,  3600),
    "auth":     (20, 3600),
}


class _RateLimiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[tuple[str, str], deque[float]] = {}

    def hit(self, bucket: str, ip: str) -> bool:
        """Return True if allowed, False if over the per-window cap."""
        max_n, window_s = _RATE_LIMITS[bucket]
        now = time.time()
        with self._lock:
            q = self._hits.setdefault((bucket, ip), deque())
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= max_n:
                return False
            q.append(now)
        return True


# ───────────────────────── app factory ─────────────────────────

def _client_ip(req: Request) -> str:
    """Pull the originating IP. Caddy forwards real-ip via X-Forwarded-For;
    --proxy-headers makes uvicorn populate req.client. Falls back to direct
    socket IP if neither is set."""
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "unknown"


def create_app(db_path: Optional[str | Path] = None) -> FastAPI:
    app = FastAPI(title="CORTEX Web API", version="1.0")
    db = Database(db_path)
    app.state.db = db
    limiter = _RateLimiter()

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

    @app.post("/api/register")
    def register(body: RegisterIn, req: Request):
        ip = _client_ip(req)
        # Honeypot: silently 200 a bot that filled the trap field. The
        # response is intentionally indistinguishable from success so it
        # doesn't tip off scanners; nothing is actually written.
        if body.honeypot:
            return {"ok": True}
        if not limiter.hit("register", ip):
            raise HTTPException(429, "too many signups from this IP — try again later")
        email = _norm_email(body.email)
        if not _is_email(email):
            raise HTTPException(400, "invalid email")
        if len(body.password) < _MIN_PASSWORD_LEN:
            raise HTTPException(400, f"password must be at least {_MIN_PASSWORD_LEN} characters")
        display = body.displayName.strip()[:_MAX_FIELD_LEN]
        if not display:
            raise HTTPException(400, "displayName required")
        if db.get_participant_by_email(email) is not None:
            raise HTTPException(409, "an account with this email already exists")
        code = "u-" + secrets.token_urlsafe(12)
        try:
            db.register_participant(
                code=code,
                password_hash=security.hash_password(body.password),
                email=email,
                display_name=display,
                signup_ip=ip,
            )
        except Exception as e:
            # UNIQUE-index violation race (two concurrent signups, same email).
            # Translate to 409 instead of a 500.
            if "UNIQUE" in str(e) or "unique" in str(e) or "duplicate" in str(e):
                raise HTTPException(409, "an account with this email already exists")
            raise
        token = security.issue_token(code, ttl_seconds=TOKEN_TTL,
                                     extra={"email": email,
                                            "expertise": body.expertise[:80]})
        return {"token": token, "expiresIn": TOKEN_TTL, "code": code,
                "email": email, "displayName": display}

    @app.post("/api/auth")
    def auth(body: AuthIn, req: Request):
        ip = _client_ip(req)
        if not limiter.hit("auth", ip):
            raise HTTPException(429, "too many login attempts — try again later")
        email = _norm_email(body.email)
        if not _is_email(email):
            raise HTTPException(401, "invalid credentials")
        row = db.get_participant_by_email(email)
        if row is None or not row["active"]:
            raise HTTPException(401, "invalid credentials")
        if not security.verify_password(body.password, row["password_hash"]):
            raise HTTPException(401, "invalid credentials")
        code = row["code"]
        token = security.issue_token(code, ttl_seconds=TOKEN_TTL,
                                     extra={"email": email})
        return {"token": token, "expiresIn": TOKEN_TTL, "code": code,
                "email": email, "displayName": row.get("display_name") or ""}

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

    # ── visualization videos (#8) ───────────────────────────────
    # The browser posts the particle-cloud trajectory (t/l/w Float32 blobs + a
    # JSON meta) captured during the session; we reconstruct the desktop
    # session-dir schema (trajectory.npz + trials.jsonl + certificate.json) and
    # run the EXACT desktop renderers (scripts/cortex_render_videos.py +
    # render_engine_explainer.py), then stream back a zip of the 4 MP4s.
    @app.post("/api/videos")
    async def videos(meta: str = Form(...), t: UploadFile = File(...),
                     l: UploadFile = File(...), w: UploadFile = File(...),
                     _code: str = Depends(require_auth)):
        try:
            import numpy as np
        except Exception as e:  # heavy render deps are optional at boot
            raise HTTPException(503, f"render deps unavailable: {e}")
        info = json.loads(meta)
        T, N, K = (int(x) for x in info["shape"])
        tb = np.frombuffer(await t.read(), dtype="<f4").reshape(T, N, K).astype(np.float64)
        lb = np.frombuffer(await l.read(), dtype="<f4").reshape(T, N, K).astype(np.float64)
        wb = np.frombuffer(await w.read(), dtype="<f4").reshape(T, N).astype(np.float64)

        sd = Path(tempfile.mkdtemp(prefix="cortex_viz_"))
        np.savez_compressed(
            sd / "trajectory.npz", t_traj=tb, l_traj=lb, w_traj=wb,
            task_codes=np.array(list(info["taskCodes"])),
            seg_ids=np.array(list(info["segIds"]), dtype=np.int64),
            delta_auroc=np.float64("nan"), n_questions=int(T))
        with open(sd / "trials.jsonl", "w") as fh:
            for tr in info["trials"]:
                fh.write(json.dumps(tr) + "\n")
        (sd / "certificate.json").write_text(json.dumps(info["certificate"]))
        (sd / "participant.json").write_text(
            json.dumps({"identity": {"name": info.get("participantName", "Anonymous")}}))

        def _render():
            scripts = str((WEB_ROOT.parent / "scripts").resolve())
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            from cortex_render_videos import render_all
            from render_engine_explainer import render_engine_explainer
            outs = list(render_all(sd).values()) + [render_engine_explainer(sd)]
            zpath = sd / "cortex_visualizations.zip"
            import zipfile
            with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
                for p in outs:
                    z.write(p, arcname=Path(p).name)
            return zpath

        try:
            zpath = await run_in_threadpool(_render)
        except Exception as e:
            shutil.rmtree(sd, ignore_errors=True)
            raise HTTPException(500, f"render failed: {e}")
        return FileResponse(
            zpath, media_type="application/zip", filename="cortex_visualizations.zip",
            background=BackgroundTask(shutil.rmtree, sd, True))

    # ── admin ───────────────────────────────────────────────────
    @app.get("/api/admin/participants")
    def admin_list(_: bool = Depends(require_admin)):
        return [dict(r) for r in db.list_participants()]

    @app.post("/api/admin/participants")
    def admin_gen(body: AdminGenIn, _: bool = Depends(require_admin)):
        """Seed test accounts. Since accounts are email+password since the
        public-signup change, this auto-generates synthetic emails so the
        seeded accounts can still authenticate via /api/auth — distinguishable
        from real signups by the @cortex.seed suffix."""
        created = []
        for _i in range(body.count):
            code = f"{body.prefix}-{secrets.token_hex(4)}"
            password = secrets.token_urlsafe(9)
            email = f"{code}@cortex.seed"
            db.register_participant(
                code=code, password_hash=security.hash_password(password),
                email=email, display_name=code, signup_ip="admin",
            )
            created.append({"code": code, "email": email, "password": password})
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
