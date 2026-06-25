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
import random
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
from statistics import median
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from . import email as email_mod
from . import security
from .db import Database, utc_now
from .session_bank import SessionBank

HERE = Path(__file__).resolve().parent       # services/api/
CORTEX_WEB = HERE.parents[1]                  # cortex_web/
REPO_ROOT = HERE.parents[2]                   # repo root

# Static serving is OPTIONAL. In prod, Caddy file-serves the SPA + EEG bundle
# and uvicorn is a pure API (leave CORTEX_SERVE_STATIC unset). Set it (dev /
# ui-smoke / single-process runs) to have uvicorn also serve the built SPA +
# bundle. Dirs are env-overridable for split-deploy / CDN cases.
SERVE_STATIC = os.environ.get("CORTEX_SERVE_STATIC", "") not in ("", "0", "false")
DIST_DIR = Path(os.environ.get("CORTEX_DIST_DIR", str(CORTEX_WEB / "apps" / "web" / "dist")))
BUNDLE_DIR = Path(os.environ.get("CORTEX_BUNDLE_DIR", str(CORTEX_WEB / "apps" / "web" / "public" / "bundle")))
# Renderers for POST /api/videos. Prefer the repo-root scripts/ (dev/desktop);
# fall back to the vendored copies under services/api/render_assets/ when the
# repo root isn't present (the prod web box deploys only cortex_web/).
_REPO_SCRIPTS = REPO_ROOT / "scripts"
_VENDORED_SCRIPTS = HERE / "render_assets"
SCRIPTS_DIR = os.environ.get(
    "CORTEX_SCRIPTS_DIR",
    str(_REPO_SCRIPTS if _REPO_SCRIPTS.exists() else _VENDORED_SCRIPTS),
)

# Default bundle the SPA pulls (overridable via env for S3/CloudFront).
DEFAULT_BUNDLE_URL = os.environ.get("CORTEX_BUNDLE_URL", "/bundle/v1.5-k7")
# Per-session candidate-pool size (the server-drawn subset the client engine
# selects within). Latency at N=1200 scales ~linearly with this: ~300ms/q at
# 250, ~530ms at 400, ~1s at 700 (engine/_latency_bench). 400 balances
# between-question speed against per-task resolution headroom; OC sets the final
# value (with per_domain_cap). Raise it once speculative precompute lands.
DEFAULT_SESSION_SAMPLE = int(os.environ.get("CORTEX_SESSION_SAMPLE", "400"))
TOKEN_TTL = int(os.environ.get("CORTEX_TOKEN_TTL", str(6 * 3600)))

# Where user reports (the /report page) are emailed. Overridable via env.
REPORT_TO = os.environ.get("CORTEX_REPORT_TO", "elikeldsen@icloud.com")


# ───────────────────────── request models ─────────────────────────

class AuthIn(BaseModel):
    email: str
    password: str
    tzOffset: Optional[int] = None   # getTimezoneOffset() — for the local login day


class AuthGoogleIn(BaseModel):
    credential: str   # Google ID token (JWT) returned by Sign in with Google
    tzOffset: Optional[int] = None


class RegisterIn(BaseModel):
    """Public signup payload. `honeypot` should be empty — it's a hidden form
    field most bots auto-fill. Any non-empty value gets a 200 OK with no
    side-effects so the bot moves on without learning it was blocked."""
    email: str
    password: str
    displayName: str
    expertise: str = ""        # optional self-reported expertise dropdown
    profile: dict[str, Any] = Field(default_factory=dict)  # demographic/clinical fields collected at signup
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


class ReportIn(BaseModel):
    """Public support/feedback report. `client` carries browser-collected
    diagnostics (OS, timezone, screen, etc.) for troubleshooting."""
    username: str = ""
    email: str = ""
    message: str
    client: dict[str, Any] = Field(default_factory=dict)


class ConsentIn(BaseModel):
    """Record a participant's consent acceptance (Phase O1)."""
    consentType: str = "research_irb"
    consentVersion: str
    irbProtocolId: Optional[str] = None


class ConsentWithdrawIn(BaseModel):
    consentType: Optional[str] = None    # None → withdraw all consent types


class ProfileIn(BaseModel):
    """Edit account/profile details (Settings page). All fields optional so a
    partial update is fine; None leaves a field unchanged."""
    displayName: Optional[str] = None
    expertise: Optional[str] = None
    profile: dict[str, Any] = Field(default_factory=dict)


class PasswordChangeIn(BaseModel):
    currentPassword: str
    newPassword: str


class EmailChangeIn(BaseModel):
    newEmail: str
    password: str


# Whitelisted demographic/clinical profile keys (collected at signup, editable
# in Settings). Anything else in a submitted profile dict is dropped.
PROFILE_FIELDS = (
    "expertise", "institution", "practice_setting", "years_reading_eeg",
    "eeg_volume_per_month", "self_rated_confidence", "color_vision",
    "prior_test_taken", "sex", "age", "location", "country", "race_ethnicity",
)


def _clean_profile(d: dict[str, Any]) -> dict[str, str]:
    """Keep only whitelisted profile keys, string-coerced and length-capped."""
    out: dict[str, str] = {}
    for k in PROFILE_FIELDS:
        v = d.get(k)
        if v is None:
            continue
        s = str(v).strip()[:_MAX_FIELD_LEN]
        if s:
            out[k] = s
    return out


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


# ──────────────────── dashboard derivation ─────────────────────
# The fixed 7-task ontology (D5: spike + the 6 IIIC patterns; K=7). Used to
# render every task tile in canonical order, including for legacy results that
# stored only a bare `verdicts` list (pre-Step-1, no per-task ℓ/θ/AUROC).
_CANONICAL_TASKS = [
    (0, "spike", "Spike"),
    (1, "sz",    "Seizure"),
    (2, "lpd",   "LPD"),
    (3, "gpd",   "GPD"),
    (4, "lrda",  "LRDA"),
    (5, "grda",  "GRDA"),
    (6, "iic",   "Other"),
]


def _dashboard_tasks(result: dict, latest_traj: Optional[dict] = None) -> list[dict]:
    """Per-task mastery summary derived from a real cert result. Reads the
    persisted `perTask` block (real ℓ/θ/ℓ*/AUROC + verdict) when present; for a
    legacy result (verdicts only) ℓ/ℓ*/AUROC come back None and just the verdict
    is shown. Always returns all 7 canonical tasks in engine-index order.

    `latest_traj` (task_k → latest real trajectory point) overrides ℓ/θ with the
    most recent MEASUREMENT — a training/re-cert point once those exist, else the
    cert eval point (same value). ℓ* (the threshold), AUROC and verdict stay from
    the certification result."""
    latest_traj = latest_traj or {}
    per = result.get("perTask")
    by_k: dict[int, dict] = {}
    if isinstance(per, list):
        for p in per:
            if isinstance(p, dict) and p.get("taskK") is not None:
                by_k[int(p["taskK"])] = p
    verdicts = result.get("verdicts") or []
    out = []
    for k, code, label in _CANONICAL_TASKS:
        p = by_k.get(k)
        lt = latest_traj.get(k)
        legacy_verdict = verdicts[k] if k < len(verdicts) else "PENDING"
        # Latest measurement wins for ℓ/θ; fall back to the cert perTask values.
        ell = p.get("ell") if p else None
        theta = p.get("theta") if p else None
        if lt is not None:
            if lt.get("ell") is not None:
                ell = lt["ell"]
            if lt.get("theta") is not None:
                theta = lt["theta"]
        if p is not None:
            out.append({
                "taskK": k,
                "code": p.get("code") or code,
                "label": p.get("label") or label,
                "ell": ell,
                "ellStar": p.get("ellStar"),
                "theta": theta,
                "auroc": p.get("auroc"),
                "verdict": p.get("verdict") or legacy_verdict,
            })
        else:
            out.append({
                "taskK": k, "code": code, "label": label,
                "ell": ell, "ellStar": None, "theta": theta, "auroc": None,
                "verdict": legacy_verdict,
            })
    return out


def _eval_points_from_result(result: dict, trials: list[dict]) -> list[dict]:
    """Build the per-domain EVAL operating point from a finished cert result:
    ℓ/θ/σ from the persisted `perTask` block + the per-task MEDIAN reaction time
    from that session's trials. Returns [] for a legacy result with no perTask."""
    per = result.get("perTask")
    if not isinstance(per, list):
        return []
    rts_by_task: dict[int, list[float]] = {}
    for t in trials:
        tk, rm = t.get("task_k"), t.get("reaction_ms")
        if tk is not None and rm is not None:
            rts_by_task.setdefault(int(tk), []).append(float(rm))
    pts = []
    for p in per:
        if not isinstance(p, dict) or p.get("taskK") is None:
            continue
        k = int(p["taskK"])
        rts = rts_by_task.get(k)
        pts.append({"taskK": k, "ell": p.get("ell"), "theta": p.get("theta"),
                    "sd": p.get("sd"), "rt": (median(rts) if rts else None)})
    return pts


def _dashboard_kpis(tasks: list[dict], last_assessed: Optional[str]) -> dict:
    """Real certification-summary KPIs (no learning-protocol data, which doesn't
    exist until the trainer ships): tasks certified, date last assessed, and the
    mean per-task AUROC over the tasks that have one."""
    aurocs = [t["auroc"] for t in tasks if isinstance(t.get("auroc"), (int, float))]
    certified = sum(1 for t in tasks if str(t.get("verdict") or "").upper() == "PASS")
    return {
        "tasksCertified": certified,
        "tasksTotal": len(tasks),
        "lastAssessed": last_assessed,
        "meanAuroc": round(sum(aurocs) / len(aurocs), 3) if aurocs else None,
    }


# ──────────────── per-question breakdown (history) ──────────────
# The bundle manifest is the ground-truth source: segId → patternClass, the
# per-task pattern words, and which task is the (binary) spike task. Loaded once
# and cached — it's the same file the SPA + the ℓ* backfill read.
_TRUTH_CACHE: dict | None = None


def _load_truth_map() -> dict:
    """{'seg': {segId: patternClass}, 'words': [...], 'classes': [...],
    'labels': [...]} from the bundle manifest, or empty maps if absent."""
    global _TRUTH_CACHE
    if _TRUTH_CACHE is not None:
        return _TRUTH_CACHE
    empty = {"seg": {}, "words": [], "classes": [], "labels": []}
    try:
        manifests = sorted(BUNDLE_DIR.glob("*/manifest.json"))
        if not manifests:
            _TRUTH_CACHE = empty
            return _TRUTH_CACHE
        m = json.loads(manifests[0].read_text())
        _TRUTH_CACHE = {
            "seg": {int(s["segId"]): s.get("patternClass")
                    for s in m.get("segments", []) if "segId" in s},
            "words": m.get("taskPatternWords", []),
            "classes": m.get("taskClasses", []),
            "labels": m.get("taskLabels", []),
        }
    except Exception:
        _TRUTH_CACHE = empty
    return _TRUTH_CACHE


def _question_breakdown(trials: list[dict], truth: dict) -> list[dict]:
    """Per-question rows for one session: the examinee's answer, the correct
    answer (spike → s>0; IIIC → segment class == task word), reaction time, the
    per-question contribution to skill-parameter uncertainty (ΔR, the increment
    in normalized info gain for the targeted domain), and the running posterior
    (ℓ/θ), pass-mass π, and cumulative R. Reconstructed from the stored diag, so
    legacy sessions work too."""
    seg, words, classes, labels = (truth["seg"], truth["words"],
                                   truth["classes"], truth["labels"])
    prev_R: dict[int, float] = {}
    out = []
    for row in trials:
        diag = row.get("diag")
        if isinstance(diag, str):
            try:
                diag = json.loads(diag)
            except Exception:
                diag = None
        k = row.get("task_k")
        k = int(k) if k is not None else (int(diag["taskK"]) if diag and diag.get("taskK") is not None else None)

        def _at(key, idx):
            v = diag.get(key) if diag else None
            return v[idx] if isinstance(v, list) and idx is not None and idx < len(v) else None

        # Answer + correct answer. Spike is binary (Yes/No). IIIC is a 6-way
        # classification: `pick` is the engine task index of the pattern the
        # examinee chose, so their answer is that pattern's label and the correct
        # answer is the segment's true pattern (NOT a Yes/No carried from spike).
        y = diag.get("y") if diag else None
        pick = row.get("pick")
        pick = int(pick) if pick is not None else None
        # Spike phase vs IIIC phase. Use the manifest's taskClasses when present;
        # otherwise fall back to the canonical spike index (0).
        is_spike = k is not None and (
            classes[k] == "spike" if (classes and k < len(classes)) else k == 0)
        answer = correct = None
        is_correct = None
        if is_spike:
            answer = ("Yes" if y == 1 else "No") if y is not None else None
            s = diag.get("s") if diag else None
            truth_yes = (s > 0) if isinstance(s, (int, float)) else None
            correct = ("Yes" if truth_yes else "No") if truth_yes is not None else None
            is_correct = None if (answer is None or correct is None) else (answer == correct)
        else:
            answer = labels[pick] if (pick is not None and 0 <= pick < len(labels)) else None
            pc = seg.get(int(row["seg_id"])) if row.get("seg_id") is not None else None
            if pc is not None:
                ci = words.index(pc) if pc in words else None
                correct = labels[ci] if (ci is not None and ci < len(labels)) else pc.upper()
            if pick is not None and 0 <= pick < len(words) and pc is not None:
                is_correct = (words[pick] == pc)
        R_k = _at("R", k)
        d_R = None
        if R_k is not None and k is not None:
            d_R = max(0.0, R_k - prev_R.get(k, 0.0))
            prev_R[k] = R_k
        out.append({
            "q": (row.get("trial_index", 0) or 0) + 1,
            "taskK": k,
            # The test phase, not the engine-probed sub-domain (which reads as if
            # it were the correct answer). The correct pattern is its own column.
            "domain": ("—" if k is None else ("Spike" if is_spike else "IIIC")),
            "answer": answer,
            "correct": correct,
            "isCorrect": is_correct,
            "rt": row.get("reaction_ms"),
            "deltaR": d_R,
            "R": R_k,
            "pi": _at("pi", k),
            "ell": _at("lMean", k),
            "theta": _at("tMean", k),
        })
    return out


# ───────────────────────── rate limiter ────────────────────────
# In-memory sliding-window counter per (route, IP). Resets on restart, which
# is fine at this scale — restarts are rare and a determined attacker can do
# damage with a single window anyway. Two routes are protected:
#   /api/register : 5 attempts / hour / IP
#   /api/auth     : 20 attempts / hour / IP

_RATE_LIMITS = {
    "register": (5,  3600),
    "auth":     (20, 3600),
    "auth_google": (20, 3600),  # Sign in with Google
    "verify":   (20, 3600),   # confirm a verification code
    "resend":   (5,  3600),   # re-send a verification code
    "forgot":   (5,  3600),   # request a password-reset code
    "reset":    (20, 3600),   # submit a reset code + new password
    "report":   (5,  3600),   # submit a support/feedback report
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


def _verify_google_credential(credential: str, client_id: str) -> dict:
    """Verify a Google ID token against `client_id` (the OAuth audience) and
    return its claims (sub, email, email_verified, name, …), or raise. google-
    auth is imported lazily so dev/test installs without it still load; tests
    monkeypatch this function. Verification checks issuer, audience, signature,
    and expiry."""
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    return google_id_token.verify_oauth2_token(
        credential, google_requests.Request(), client_id)


def _build_report_email(username: str, email: str, message: str,
                        client: dict[str, Any], ip: str) -> tuple[str, str]:
    """Render the support-report email (plain text) sent to REPORT_TO. The
    receiver template: reporter identity, the message, submission time, and the
    browser-collected diagnostics for troubleshooting."""
    def g(k: str) -> str:
        v = client.get(k)
        s = str(v).strip() if v is not None else ""
        return s[:500] if s else "(not provided)"
    who = username or email or "anonymous"
    subject = f"[CORTEX Report] {who}"
    lines = [
        "A new report was submitted through CORTEX (app.cortexeeg.org).",
        "",
        "── Reporter ──",
        f"Username: {username or '(not provided)'}",
        f"Email:    {email or '(not provided)'}",
        "",
        "── Message ──",
        message,
        "",
        "── Submitted ──",
        f"Server time (UTC):   {utc_now()}",
        f"Reporter local time: {g('clientTime')}",
        f"App language:        {g('appLang')}",
        "",
        "── Diagnostics (for troubleshooting) ──",
        f"Operating system: {g('os')}",
        f"Browser:          {g('browser')}",
        f"Device platform:  {g('platform')}",
        f"Screen:           {g('screen')}",
        f"Viewport:         {g('viewport')}",
        f"Timezone:         {g('timezone')}",
        f"Locale:           {g('language')}",
        f"Page URL:         {g('url')}",
        f"Referrer:         {g('referrer')}",
        f"Online:           {g('online')}",
        f"User agent:       {g('userAgent')}",
        f"IP address:       {ip}  (approximate location can be looked up from this)",
        "",
        "— Sent automatically by CORTEX. Reply to this email to reach the reporter.",
    ]
    return subject, "\n".join(lines)


def create_app(db_path: Optional[str | Path] = None) -> FastAPI:
    app = FastAPI(title="CORTEX Web API", version="1.0")
    db = Database(db_path)
    app.state.db = db
    limiter = _RateLimiter()

    # ── per-session question bank (goal 3: server-side balanced draw) ──
    # Bundle config is read here (not at import) so tests/dev can point it at a
    # fixture bundle via env. The bank (the full signal index = the bundle
    # manifest) loads LAZILY on first /api/session and is cached for the app's
    # lifetime; a missing manifest (CI / fresh box) leaves it None and yields a
    # clear 503 instead of breaking app construction or unrelated endpoints.
    bundle_url = os.environ.get("CORTEX_BUNDLE_URL", DEFAULT_BUNDLE_URL)
    bundle_dir = Path(os.environ.get("CORTEX_BUNDLE_DIR", str(BUNDLE_DIR)))
    session_sample = int(os.environ.get("CORTEX_SESSION_SAMPLE", str(DEFAULT_SESSION_SAMPLE)))
    spacing_days = int(os.environ.get("CORTEX_SPACING_DAYS", "30"))
    spacing_sessions = int(os.environ.get("CORTEX_SPACING_SESSIONS", "3"))
    _bank_cache: dict[str, Optional[SessionBank]] = {}

    def get_session_bank() -> Optional[SessionBank]:
        if "bank" not in _bank_cache:
            version = bundle_url.rstrip("/").split("/")[-1]
            mpath = bundle_dir / version / "manifest.json"
            _bank_cache["bank"] = (
                SessionBank(mpath, bundle_url) if mpath.exists() else None)
        return _bank_cache["bank"]

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
        prof = _clean_profile(body.profile)
        expertise = (body.expertise.strip() or prof.get("expertise", "")).strip()[:_MAX_FIELD_LEN] or None
        try:
            db.register_participant(
                code=code,
                password_hash=security.hash_password(body.password),
                email=email,
                display_name=display,
                signup_ip=ip,
                signup_expertise=expertise,
                profile=(json.dumps(prof) if prof else None),
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
        db.record_login_day(code, body.tzOffset or 0)
        token = security.issue_token(code, ttl_seconds=TOKEN_TTL,
                                     extra={"email": email})
        return {"token": token, "expiresIn": TOKEN_TTL, "code": code,
                "email": email, "displayName": row.get("display_name") or ""}

    @app.post("/api/auth/google")
    def auth_google(body: AuthGoogleIn, req: Request):
        ip = _client_ip(req)
        if not limiter.hit("auth_google", ip):
            raise HTTPException(429, "too many sign-in attempts — try again later")
        client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        if not client_id:
            raise HTTPException(503, "Google sign-in is not configured")
        try:
            info = _verify_google_credential(body.credential, client_id)
        except Exception:
            raise HTTPException(401, "invalid Google credential")
        if not info.get("email_verified"):
            raise HTTPException(401, "Google account email is not verified")
        sub = str(info.get("sub") or "")
        email = _norm_email(info.get("email") or "")
        name = (info.get("name") or "").strip()[:_MAX_FIELD_LEN]
        if not sub or not _is_email(email):
            raise HTTPException(401, "incomplete Google profile")
        # Find by Google id → else link to an existing same-email account
        # (Google asserts the email) → else create a fresh OAuth account.
        row = db.get_participant_by_google_sub(sub)
        if row is None:
            existing = db.get_participant_by_email(email)
            if existing is not None:
                db.link_google_sub(existing["code"], sub)
                if not existing.get("email_verified_utc"):
                    db.mark_email_verified(existing["code"])
                row = db.get_participant_by_email(email)
            else:
                code = "u-" + secrets.token_urlsafe(12)
                try:
                    db.register_oauth_participant(
                        code=code, email=email, display_name=name or email,
                        google_sub=sub, signup_ip=ip)
                except Exception as e:
                    # UNIQUE race (concurrent first sign-in): fall back to lookup.
                    if "UNIQUE" in str(e) or "unique" in str(e) or "duplicate" in str(e):
                        row = db.get_participant_by_google_sub(sub) or db.get_participant_by_email(email)
                    else:
                        raise
                row = row or db.get_participant_by_email(email)
        if row is None or not row.get("active"):
            raise HTTPException(403, "account is disabled")
        db.record_login_day(row["code"], body.tzOffset or 0)
        token = security.issue_token(row["code"], ttl_seconds=TOKEN_TTL,
                                     extra={"email": email})
        return {"token": token, "expiresIn": TOKEN_TTL, "code": row["code"],
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

    @app.post("/api/report")
    def report(body: ReportIn, req: Request):
        ip = _client_ip(req)
        if not limiter.hit("report", ip):
            raise HTTPException(429, "too many reports from this IP — try again later")
        message = body.message.strip()
        if not message:
            raise HTTPException(400, "please describe the issue or suggestion")
        message = message[:5000]
        username = body.username.strip()[:_MAX_FIELD_LEN]
        email = _norm_email(body.email)
        reply_to = email if _is_email(email) else None
        subject, text = _build_report_email(username, email, message, body.client or {}, ip)
        email_mod.send_email(REPORT_TO, subject, text, reply_to=reply_to)
        return {"ok": True}

    # ── gated ───────────────────────────────────────────────────
    @app.get("/api/manifest")
    def manifest(_code: str = Depends(require_auth)):
        # Real bundle identity (replaces the hardcoded "v1.1-local"). The SPA
        # uses bundleUrl as the base for lazily-fetched EEG/spec blobs; the
        # per-session question SET now comes from POST /api/session (the
        # server-side draw), so sessionSample is advisory only.
        bank = get_session_bank()
        return {
            "bundleUrl": bundle_url,
            "version": bank.version if bank else None,
            "sessionSample": session_sample,
        }

    @app.get("/api/tutorial-example")
    def tutorial_example(_code: str = Depends(require_auth)):
        # One IIIC example segment for the tutorial walkthrough (goal 3: no
        # full-manifest fetch to the browser). 503 if the bank isn't configured.
        bank = get_session_bank()
        if bank is None:
            raise HTTPException(503, "question bank unavailable")
        return bank.example()

    @app.post("/api/session")
    def new_session(body: SessionIn, code: str = Depends(require_auth)):
        # Server-side balanced draw (goal 3): pick this sitting's ~session_sample
        # questions from the full bank, excluding the participant's recently-seen
        # segments (goal 5 spacing), and stamp the bundle version (provenance O3).
        # The browser SMC engine runs over `bank.segments` exactly as before.
        bank = get_session_bank()
        if bank is None:
            raise HTTPException(503, "question bank unavailable")
        seed = (body.sampleSeed if body.sampleSeed is not None
                else random.randint(0, 2**31 - 1))
        exclude = db.get_exposure_exclusion(code, spacing_days, spacing_sessions)
        drawn = bank.draw(seed, session_sample, exclude)
        session_id = uuid.uuid4().hex
        db.create_session(session_id, code, body.participant, seed,
                          bundle_version=bank.version)
        return {"sessionId": session_id, "sampleSeed": seed, "bank": drawn}

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
        # Seed the per-domain evolution charts: record this test's EVAL operating
        # point (ℓ/θ/σ + median RT) as one real trajectory point per task.
        eval_pts = _eval_points_from_result(body.result, db.session_trials(body.sessionId))
        if eval_pts:
            db.write_eval_trajectory(code, body.sessionId, eval_pts)
        return {"ok": True}

    # ── dashboard (cert-result surfaces) ────────────────────────
    # Real certification data only: per-task ℓ/θ/ℓ*/AUROC + verdict from the
    # latest result, plus cert-summary KPIs (tasks certified / last assessed /
    # mean AUROC). Learning-protocol surfaces (streak, deck, trajectories) carry
    # NO data until the trainer is ported — see /api/regimen + /api/trajectories.
    @app.get("/api/dashboard")
    def dashboard(code: str = Depends(require_auth)):
        sessions = db.list_results_for_code(code)
        if not sessions:
            return {"result": None, "hasResult": False, "tasks": [],
                    "kpis": None, "sample": False}
        latest = sessions[0]
        result = latest["result"]
        # Latest real measurement per domain (rows come ordered by task_k, ts
        # ascending, so the last one seen per task is the most recent).
        latest_traj: dict[int, dict] = {}
        for r in db.get_trajectories(code):
            latest_traj[int(r["task_k"])] = r
        tasks = _dashboard_tasks(result, latest_traj)
        return {
            "result": result,
            "hasResult": True,
            "tasks": tasks,
            "kpis": _dashboard_kpis(tasks, latest.get("finished_utc")),
            "sample": False,
        }

    @app.get("/api/activity")
    def activity(tz: int = 0, code: str = Depends(require_auth)):
        # Per-day activity levels for the consistency heatmap (sign-in / cert /
        # training), keyed by the user's LOCAL date. `tz` = getTimezoneOffset().
        return {"days": db.activity_levels(code, tz)}

    @app.get("/api/history")
    def history(code: str = Depends(require_auth)):
        # Completed certification attempts for this participant, newest first.
        # Real data only (no sample); scoped strictly by the authed code.
        return {"sessions": db.list_results_for_code(code)}

    @app.get("/api/history/{session_id}/questions")
    def history_questions(session_id: str, code: str = Depends(require_auth)):
        # Per-question breakdown for one of THIS participant's tests. Lazy —
        # the history UI fetches it only when a test's dropdown is expanded.
        sess = db.get_session(session_id)
        if sess is None or sess["code"] != code:
            raise HTTPException(404, "unknown session")
        questions = _question_breakdown(db.session_trials(session_id), _load_truth_map())
        return {"sessionId": session_id, "nQuestions": len(questions),
                "questions": questions}

    # Learning-protocol surfaces (regimen + ℓ/θ/RT trajectories). These carry
    # REAL data only — the active regimen if the trainer has generated one, and
    # real (is_real=1) trajectory points. Empty until the L1 trainer ships; no
    # sample fallback. `sample` stays in the contract (always False now) so the
    # client keeps a single response shape.
    @app.get("/api/regimen")
    def regimen(code: str = Depends(require_auth)):
        reg = db.get_active_regimen(code)
        return {"regimen": reg["plan"] if reg is not None else None, "sample": False}

    @app.get("/api/trajectories")
    def trajectories(code: str = Depends(require_auth)):
        rows = db.get_trajectories(code)
        pts = [{"taskK": r["task_k"], "phase": r["phase"], "ell": r["ell"],
                "theta": r["theta"], "sd": r["sd"], "rt": r["rt"], "ts": r["ts"]}
               for r in rows]
        return {"trajectories": pts, "sample": False}

    @app.get("/api/training-sessions")
    def training_list(code: str = Depends(require_auth)):
        return {"sessions": db.list_training_sessions(code)}

    @app.post("/api/training-sessions")
    def training_start(body: TrainingStartIn, code: str = Depends(require_auth)):
        training_id = uuid.uuid4().hex
        reg = db.get_active_regimen(code)   # link the sitting to its regimen (Phase O2)
        db.create_training_session(
            training_id, code, body.taskFocus,
            regimen_id=(reg["regimen_id"] if reg else None),
            source_session_id=(reg.get("source_session_id") if reg else None))
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

    # ── consent ledger (Phase O1) ───────────────────────────────
    @app.post("/api/consent")
    def consent_record(body: ConsentIn, req: Request, code: str = Depends(require_auth)):
        cver = (body.consentVersion or "").strip()[:_MAX_FIELD_LEN]
        if not cver:
            raise HTTPException(400, "consentVersion required")
        ctype = (body.consentType or "").strip()[:_MAX_FIELD_LEN] or "research_irb"
        irb = (body.irbProtocolId or "").strip()[:_MAX_FIELD_LEN] or None
        db.record_consent(code, ctype, cver, irb_protocol_id=irb, consent_ip=_client_ip(req))
        return {"ok": True}

    @app.post("/api/consent/withdraw")
    def consent_withdraw(body: ConsentWithdrawIn, code: str = Depends(require_auth)):
        n = db.withdraw_consent(code, (body.consentType or "").strip() or None)
        return {"ok": True, "withdrawn": n}

    @app.get("/api/consent")
    def consent_list(code: str = Depends(require_auth)):
        return {"events": db.get_consent_events(code)}

    # ── account / profile (Settings page) ───────────────────────
    @app.get("/api/profile")
    def get_profile(code: str = Depends(require_auth)):
        row = db.get_participant(code)
        if row is None:
            raise HTTPException(404, "account not found")
        prof: dict[str, Any] = {}
        if row.get("profile"):
            try:
                prof = json.loads(row["profile"])
            except Exception:
                prof = {}
        return {
            "email": row.get("email") or "",
            "displayName": row.get("display_name") or "",
            "expertise": row.get("signup_expertise") or "",
            "authProvider": row.get("auth_provider") or "local",
            "profile": prof,
        }

    @app.put("/api/profile")
    def put_profile(body: ProfileIn, code: str = Depends(require_auth)):
        dn = body.displayName.strip()[:_MAX_FIELD_LEN] if body.displayName is not None else None
        if dn is not None and not dn:
            raise HTTPException(400, "display name cannot be empty")
        prof = _clean_profile(body.profile) if body.profile is not None else None
        # expertise lives both in its own column and in the profile blob; keep them aligned
        exp = body.expertise.strip()[:_MAX_FIELD_LEN] if body.expertise is not None else (
            prof.get("expertise") if prof else None)
        db.update_profile(code, display_name=dn,
                          profile=(json.dumps(prof) if prof is not None else None),
                          signup_expertise=exp)
        return {"ok": True}

    @app.post("/api/account/password")
    def change_password(body: PasswordChangeIn, code: str = Depends(require_auth)):
        row = db.get_participant(code)
        if row is None or not security.verify_password(body.currentPassword, row["password_hash"]):
            raise HTTPException(403, "current password is incorrect")
        if len(body.newPassword) < _MIN_PASSWORD_LEN:
            raise HTTPException(400, f"password must be at least {_MIN_PASSWORD_LEN} characters")
        db.set_password_hash(code, security.hash_password(body.newPassword))
        return {"ok": True}

    @app.post("/api/account/email")
    def change_email(body: EmailChangeIn, code: str = Depends(require_auth)):
        row = db.get_participant(code)
        if row is None or not security.verify_password(body.password, row["password_hash"]):
            raise HTTPException(403, "password is incorrect")
        new_email = _norm_email(body.newEmail)
        if not _is_email(new_email):
            raise HTTPException(400, "invalid email")
        existing = db.get_participant_by_email(new_email)
        if existing is not None and existing["code"] != code:
            raise HTTPException(409, "an account with this email already exists")
        try:
            db.update_email(code, new_email)
        except Exception as e:
            if "UNIQUE" in str(e) or "unique" in str(e) or "duplicate" in str(e):
                raise HTTPException(409, "an account with this email already exists")
            raise
        return {"ok": True, "email": new_email}

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
            scripts = SCRIPTS_DIR
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

    # ── static (bundle + SPA): OPTIONAL — prod serves these via Caddy ──
    if SERVE_STATIC and BUNDLE_DIR.exists():
        app.mount("/bundle", StaticFiles(directory=str(BUNDLE_DIR)), name="bundle")
    if SERVE_STATIC and DIST_DIR.exists():
        # SPA catch-all LAST so /api/* and /bundle win.
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="spa")

    @app.exception_handler(HTTPException)
    async def _http_exc(_req, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    return app


app = create_app()
