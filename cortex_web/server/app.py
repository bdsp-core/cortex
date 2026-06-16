"""CORTEX web backend — FastAPI.

Tiny by design (PLAN §3, §8): the engine runs entirely in the participant's
browser, so the server only (a) authenticates, (b) hands out the bundle URL,
and (c) ingests results. No per-question traffic. A single small instance
serves 100 concurrent participants because there is no shared compute.

Endpoints (all JSON, prefix /api):
    GET  /api/health                      liveness
    POST /api/register  {email, password, displayName} → {needsVerification, email}
    POST /api/verify/confirm {email, code} → {ok}        (marks email verified)
    POST /api/verify/resend  {email}       → {ok}        (re-issue verify code)
    POST /api/auth      {email, password}  → {token, expiresIn}  (403 if unverified)
    POST /api/forgot    {email}            → {ok}        (issue reset code)
    POST /api/reset     {email, code, newPassword} → {ok}
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

from . import email as email_mod
from . import sample_data
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


class VerifyIn(BaseModel):
    email: str
    code: str


class EmailIn(BaseModel):
    email: str


class ResetIn(BaseModel):
    email: str
    code: str
    newPassword: str


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


class TrainingStartIn(BaseModel):
    taskFocus: Optional[str] = None


class TrainingFinalizeIn(BaseModel):
    trainingId: str
    nItems: Optional[int] = None
    summary: Optional[dict[str, Any]] = None


class TrajectoryIn(BaseModel):
    points: list[dict[str, Any]] = Field(default_factory=list)


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
    "verify":   (20, 3600),   # confirm a verification code
    "resend":   (5,  3600),   # re-send a verification code
    "forgot":   (5,  3600),   # request a password-reset code
    "reset":    (20, 3600),   # submit a reset code + new password
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


def _future_utc(seconds: int) -> str:
    """An ISO-Z timestamp `seconds` in the future. Same fixed format as
    db.utc_now() so lexicographic compare == chronological compare."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + seconds))


def _expose_codes() -> bool:
    """Dev/CI only: echo issued codes in API responses so headless smoke tests
    can complete the verify/reset flow. Never set in production (SES sends the
    real email). Off unless CORTEX_EMAIL_EXPOSE_CODE=1."""
    return os.environ.get("CORTEX_EMAIL_EXPOSE_CODE") == "1"


def _issue_code(db: Database, participant_code: str, email: str, purpose: str) -> str:
    """Generate + store + email a fresh 6-digit code for (account, purpose)."""
    code = security.gen_numeric_code()
    db.put_auth_code(participant_code, purpose, security.hash_code(code),
                     _future_utc(security.CODE_TTL_SECONDS))
    email_mod.send_auth_code(email, code, purpose)
    return code


def _check_code(db: Database, participant_code: str, purpose: str, presented: str) -> bool:
    """Validate a presented code: not consumed, not expired, under the attempt
    cap, and matching. Bumps the attempt counter on a mismatch. The caller is
    responsible for consuming the code on success."""
    row = db.get_auth_code(participant_code, purpose)
    if row is None or row.get("consumed_utc"):
        return False
    from .db import utc_now
    if utc_now() >= row["expires_utc"]:
        return False
    if int(row.get("attempts") or 0) >= security.CODE_MAX_ATTEMPTS:
        return False
    if not security.verify_code(presented.strip(), row["code_hash"]):
        db.increment_auth_attempts(participant_code, purpose)
        return False
    return True


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
        # No session is issued at signup: the account must verify its email
        # before it can sign in. Email a 6-digit code; the SPA advances to the
        # verify screen.
        dev_code = _issue_code(db, code, email, "verify")
        resp = {"needsVerification": True, "email": email, "displayName": display}
        if _expose_codes():
            resp["devCode"] = dev_code
        return resp

    @app.post("/api/verify/confirm")
    def verify_confirm(body: VerifyIn, req: Request):
        ip = _client_ip(req)
        if not limiter.hit("verify", ip):
            raise HTTPException(429, "too many attempts — try again later")
        email = _norm_email(body.email)
        row = db.get_participant_by_email(email)
        if row is None:
            raise HTTPException(400, "invalid or expired code")
        if row.get("email_verified_utc"):
            return {"ok": True, "alreadyVerified": True}
        if not _check_code(db, row["code"], "verify", body.code):
            raise HTTPException(400, "invalid or expired code")
        db.consume_auth_code(row["code"], "verify")
        db.mark_email_verified(row["code"])
        return {"ok": True}

    @app.post("/api/verify/resend")
    def verify_resend(body: EmailIn, req: Request):
        ip = _client_ip(req)
        if not limiter.hit("resend", ip):
            raise HTTPException(429, "too many requests — try again later")
        email = _norm_email(body.email)
        row = db.get_participant_by_email(email)
        resp: dict[str, Any] = {"ok": True}
        # Only (re)issue for an existing, still-unverified account; respond 200
        # either way so the endpoint doesn't reveal which emails exist.
        if row is not None and not row.get("email_verified_utc"):
            dev_code = _issue_code(db, row["code"], email, "verify")
            if _expose_codes():
                resp["devCode"] = dev_code
        return resp

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
        # Credentials are valid but the email isn't verified yet: signal the SPA
        # to route to the verify screen (detail is a stable machine code).
        if not row.get("email_verified_utc"):
            raise HTTPException(403, "email_not_verified")
        code = row["code"]
        token = security.issue_token(code, ttl_seconds=TOKEN_TTL,
                                     extra={"email": email})
        return {"token": token, "expiresIn": TOKEN_TTL, "code": code,
                "email": email, "displayName": row.get("display_name") or ""}

    @app.post("/api/forgot")
    def forgot(body: EmailIn, req: Request):
        ip = _client_ip(req)
        if not limiter.hit("forgot", ip):
            raise HTTPException(429, "too many requests — try again later")
        email = _norm_email(body.email)
        row = db.get_participant_by_email(email)
        resp: dict[str, Any] = {"ok": True}
        # Respond 200 regardless so the endpoint doesn't reveal which emails
        # have accounts; only actually issue a code for a real account.
        if row is not None and row["active"]:
            dev_code = _issue_code(db, row["code"], email, "reset")
            if _expose_codes():
                resp["devCode"] = dev_code
        return resp

    @app.post("/api/reset")
    def reset(body: ResetIn, req: Request):
        ip = _client_ip(req)
        if not limiter.hit("reset", ip):
            raise HTTPException(429, "too many attempts — try again later")
        email = _norm_email(body.email)
        if len(body.newPassword) < _MIN_PASSWORD_LEN:
            raise HTTPException(400, f"password must be at least {_MIN_PASSWORD_LEN} characters")
        row = db.get_participant_by_email(email)
        if row is None or not _check_code(db, row["code"], "reset", body.code):
            raise HTTPException(400, "invalid or expired code")
        db.consume_auth_code(row["code"], "reset")
        db.set_password_hash(row["code"], security.hash_password(body.newPassword))
        # A successful reset also confirms control of the email address.
        if not row.get("email_verified_utc"):
            db.mark_email_verified(row["code"])
        return {"ok": True}

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

    # ── dashboard / learning-protocol (Phase 2) ─────────────────
    # The dashboard surfaces. Real certification results (verdicts/AUROC) come
    # from the results table; ℓ/θ/RT TRAINING trajectories + the protocol plan
    # are sample data (flagged `sample: true`) until the trainer is ported.
    @app.get("/api/dashboard")
    def dashboard(code: str = Depends(require_auth)):
        result = db.latest_result_for_code(code)
        return {
            "result": result,                    # latest real cert result, or null
            "hasResult": result is not None,
            "tasks": sample_data.tasks(),         # sample mastery-grid summaries
            "kpis": sample_data.kpis(),
            "sample": True,                       # KPIs + task ℓ/θ/RT are illustrative
        }

    @app.get("/api/regimen")
    def regimen(code: str = Depends(require_auth)):
        reg = db.get_active_regimen(code)
        if reg is not None:
            return {"regimen": reg["plan"], "sample": False}
        return {"regimen": sample_data.regimen_plan(), "sample": True}

    @app.get("/api/trajectories")
    def trajectories(code: str = Depends(require_auth)):
        rows = db.get_trajectories(code)
        if rows:
            pts = [{"taskK": r["task_k"], "phase": r["phase"], "ell": r["ell"],
                    "theta": r["theta"], "sd": r["sd"], "rt": r["rt"], "ts": r["ts"]}
                   for r in rows]
            return {"trajectories": pts, "sample": False}
        return {"trajectories": sample_data.trajectories(), "sample": True}

    @app.get("/api/training-sessions")
    def training_list(code: str = Depends(require_auth)):
        return {"sessions": db.list_training_sessions(code)}

    @app.post("/api/training-sessions")
    def training_start(body: TrainingStartIn, code: str = Depends(require_auth)):
        training_id = uuid.uuid4().hex
        db.create_training_session(training_id, code, body.taskFocus)
        return {"trainingId": training_id}

    @app.post("/api/training-sessions/finalize")
    def training_finalize(body: TrainingFinalizeIn, code: str = Depends(require_auth)):
        ok = db.finalize_training_session(body.trainingId, code, body.nItems, body.summary)
        if not ok:
            raise HTTPException(404, "unknown training session")
        return {"ok": True}

    @app.post("/api/trajectories")
    def trajectories_append(body: TrajectoryIn, code: str = Depends(require_auth)):
        db.append_trajectory_points(code, body.points)
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
            db.mark_email_verified(code)   # seeded accounts skip email verification
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
