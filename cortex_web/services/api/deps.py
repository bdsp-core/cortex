"""Request-level dependencies + the in-process rate limiter.

Routers get shared state (Database, RateLimiter, SessionBank getter, config)
from `request.app.state` — created once in create_app(). The dependencies
here are stateless: they read headers/env only, so any router can import them
without a circular import back into the app factory.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from collections import deque

from fastapi import Header, HTTPException, Request

from . import security


def client_ip(req: Request) -> str:
    """Pull the originating IP. Caddy forwards real-ip via X-Forwarded-For;
    --proxy-headers makes uvicorn populate req.client. Falls back to direct
    socket IP if neither is set."""
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "unknown"


def require_auth(authorization: str = Header(default="")) -> str:
    """Bearer-token dependency: returns the participant code (JWT subject)."""
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


# ───────────────────────── rate limiter ────────────────────────
# In-memory sliding-window counter per (route, IP). Resets on restart, which
# is fine at this scale — restarts are rare and a determined attacker can do
# damage with a single window anyway. Single-process by design (see run.py).

RATE_LIMITS = {
    "register": (5,  3600),
    "auth":     (20, 3600),
    "auth_google": (20, 3600),  # Sign in with Google
    "verify":   (20, 3600),   # confirm a verification code
    "verify_status": (120, 3600),  # verify-screen bounce poll (~5s cadence for 2 min)
    "resend":   (5,  3600),   # re-send a verification code
    "forgot":   (5,  3600),   # request a password-reset code
    "reset":    (20, 3600),   # submit a reset code + new password
    "report":   (5,  3600),   # submit a support/feedback report
    "cohort_create": (10, 3600),   # create a cohort
    "cohort_invite": (60, 3600),   # invite by 9-digit id (also bounds id probing)
}


class RateLimiter:
    # Every SWEEP_EVERY hits, drop (bucket, ip) keys whose window has fully
    # expired — otherwise the map grows monotonically with distinct client IPs
    # for the life of the process.
    _SWEEP_EVERY = 512

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._n = 0

    def _sweep(self, now: float) -> None:
        for key, q in list(self._hits.items()):
            window_s = RATE_LIMITS[key[0]][1]
            while q and now - q[0] > window_s:
                q.popleft()
            if not q:
                del self._hits[key]

    def hit(self, bucket: str, ip: str) -> bool:
        """Return True if allowed, False if over the per-window cap."""
        max_n, window_s = RATE_LIMITS[bucket]
        now = time.time()
        with self._lock:
            self._n += 1
            if self._n % self._SWEEP_EVERY == 0:
                self._sweep(now)
            q = self._hits.setdefault((bucket, ip), deque())
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= max_n:
                return False
            q.append(now)
        return True
