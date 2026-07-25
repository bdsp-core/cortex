"""Policy boundary for public SPA crash telemetry.

The endpoint is intentionally available before sign-in, but public input must
not be able to page the operator. This module normalizes unstable React
internals, classifies the known injected-DOM circular-JSON signature, derives
privacy-limited metadata, verifies optional bearer authentication, and applies
durable global/per-fingerprint budgets before the router journals an event.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

from . import security, timeutil
from .models import ClientErrorIn

_REACT_TOKEN_RE = re.compile(r"__reactFiber\$[A-Za-z0-9_]+")
_SPACE_RE = re.compile(r"\s+")
_BROWSER_PATTERNS = (
    ("Edge", re.compile(r"\bEdg/(\d+)")),
    ("Opera", re.compile(r"\bOPR/(\d+)")),
    ("Chrome", re.compile(r"\b(?:Chrome|CriOS)/(\d+)")),
    ("Firefox", re.compile(r"\b(?:Firefox|FxiOS)/(\d+)")),
    ("Safari", re.compile(r"\bVersion/(\d+).*\bSafari/")),
)
_FETCH_SITE_VALUES = frozenset({"same-origin", "same-site", "cross-site", "none"})


@dataclass(frozen=True)
class ClientErrorObservation:
    fingerprint: str
    classification: str
    authenticated: bool
    message: str
    stack: str
    url: str
    surface: str
    ua_family: str
    origin_class: str
    sec_fetch_site: str
    request_fingerprint: str
    day: str
    now_utc: str


def normalize_react_tokens(value: str) -> str:
    """Remove per-page React expando randomness from an error signature."""
    return _REACT_TOKEN_RE.sub("__reactFiber$<id>", value or "")


def _one_line(value: str) -> str:
    return _SPACE_RE.sub(" ", normalize_react_tokens(value)).strip()


def normalize_message(message: str) -> str:
    return _one_line(message)[:500]


def normalize_stack(stack: str) -> str:
    frames = [_one_line(line) for line in (stack or "").splitlines()[:8]]
    return " | ".join(frame for frame in frames if frame)[:4000]


def classify_error(message: str, stack: str) -> str:
    """Identify the observed injected-DOM circular-serialization failure.

    The conjunction is deliberately narrow: an ordinary circular-data bug is
    not silently classified merely because it also mentions JSON.stringify.
    """
    msg = message or ""
    message_signature = (
        "Converting circular structure to JSON" in msg
        and "HTMLAnchorElement" in msg
        and "__reactFiber$" in msg
        and "stateNode" in msg
        and "closes the circle" in msg
    )
    # The stack normally corroborates this with JSON.stringify + appendChild,
    # but some browser/privacy modes omit stacks. The five-part message
    # signature is already specific enough to classify safely on its own.
    return "injected-dom" if message_signature else "application"


def error_fingerprint(message: str, stack: str) -> str:
    material = f"{normalize_message(message)}\n{normalize_stack(stack)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def normalized_user_agent(raw: str) -> str:
    """Return browser-major + OS family, never the identifying raw UA."""
    raw = raw or ""
    browser = "Other"
    for label, pattern in _BROWSER_PATTERNS:
        match = pattern.search(raw)
        if match:
            browser = f"{label}/{match.group(1)}"
            break
    if "Windows" in raw:
        os_family = "Windows"
    elif "Android" in raw:
        os_family = "Android"
    elif "iPhone" in raw or "iPad" in raw:
        os_family = "iOS"
    elif "Macintosh" in raw:
        os_family = "macOS"
    elif "Linux" in raw:
        os_family = "Linux"
    else:
        os_family = "Other"
    return f"{browser} {os_family}"


def sanitized_path(raw: str) -> str:
    """Keep only a pathname; discard query strings, fragments, and origins."""
    try:
        path = urlsplit(raw or "/").path or "/"
    except ValueError:
        return "/invalid"
    return (path if path.startswith("/") else "/invalid")[:300]


def origin_class(origin: str, host: str) -> str:
    """Reduce Origin to a relationship category instead of retaining it."""
    if not origin:
        return "missing"
    try:
        parsed = urlsplit(origin)
        if not parsed.scheme or not parsed.netloc:
            return "invalid"
        return "same-origin" if parsed.netloc.lower() == host.lower() else "cross-origin"
    except ValueError:
        return "invalid"


def fetch_site_class(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return "missing"
    return value if value in _FETCH_SITE_VALUES else "other"


def authenticated_participant(request) -> str | None:
    """Return a live participant code only for a valid bearer token."""
    authorization = request.headers.get("authorization", "")
    if not authorization.lower().startswith("bearer "):
        return None
    try:
        claims = security.decode_token(authorization[7:].strip())
        code = str(claims["sub"])
        row = request.app.state.db.get_participant(code)
    except (KeyError, TypeError, security.TokenError):
        return None
    return code if row is not None and row.get("active") else None


def _request_fingerprint(day: str, ip: str, ua_family: str) -> str:
    # A dedicated key can be independently rotated. Falling back to the JWT
    # key avoids a new required production secret while remaining an HMAC, not
    # a reversible or raw network identifier. Including day prevents tracking
    # a browser across digest periods.
    key = os.environ.get("CORTEX_TELEMETRY_FINGERPRINT_SECRET")
    secret = key.encode("utf-8") if key else security._jwt_secret()
    material = f"{day}\0{ip}\0{ua_family}".encode("utf-8")
    return hmac.new(secret, material, hashlib.sha256).hexdigest()[:16]


class ClientErrorPolicy:
    """Build observations and enforce durable aggregate budgets."""

    def __init__(self, db) -> None:
        self.db = db
        self.global_limit = max(
            1, int(os.environ.get("CORTEX_CLIENT_ERROR_GLOBAL_LIMIT", "120")))
        self.fingerprint_limit = max(
            1, int(os.environ.get("CORTEX_CLIENT_ERROR_FINGERPRINT_LIMIT", "20")))
        self.window_s = max(
            60, int(os.environ.get("CORTEX_CLIENT_ERROR_LIMIT_WINDOW_S", "3600")))
        # The app runs one uvicorn process, but sync FastAPI routes execute in
        # a thread pool. Serialize the select/update persistence units.
        self._lock = threading.Lock()

    def observe(self, body: ClientErrorIn, request, ip: str,
                now_s: float | None = None) -> ClientErrorObservation:
        if now_s is None:
            now_s = time.time()
        day = time.strftime("%Y-%m-%d", time.gmtime(now_s))
        ua = normalized_user_agent(
            request.headers.get("user-agent") or body.ua)
        code = authenticated_participant(request)
        return ClientErrorObservation(
            fingerprint=error_fingerprint(body.message, body.stack),
            classification=classify_error(body.message, body.stack),
            authenticated=code is not None,
            message=normalize_message(body.message),
            stack=normalize_stack(body.stack),
            url=sanitized_path(body.url),
            surface=body.surface if body.surface in {"desktop", "mobile"} else "unknown",
            ua_family=ua,
            origin_class=origin_class(
                request.headers.get("origin", ""), request.headers.get("host", "")),
            sec_fetch_site=fetch_site_class(
                request.headers.get("sec-fetch-site", "")),
            request_fingerprint=_request_fingerprint(day, ip, ua),
            day=day,
            now_utc=timeutil.iso_at(now_s),
        )

    def admit_and_record(self, observation: ClientErrorObservation,
                         now_s: float | None = None) -> bool:
        """Apply both budgets, aggregate the event, and return journalability."""
        if now_s is None:
            now_s = timeutil.parse_iso(observation.now_utc)
        with self._lock:
            # Anonymous traffic must not be able to consume the trusted
            # authenticated channel's alert budget (or vice versa).
            scope = "authenticated" if observation.authenticated else "anonymous"
            global_ok = self.db.consume_client_error_budget(
                f"global:{scope}", self.global_limit, self.window_s, now_s)
            fingerprint_ok = self.db.consume_client_error_budget(
                f"fingerprint:{scope}:{observation.fingerprint}",
                self.fingerprint_limit, self.window_s, now_s)
            accepted = global_ok and fingerprint_ok
            self.db.record_client_error_event(
                day=observation.day,
                fingerprint=observation.fingerprint,
                classification=observation.classification,
                authenticated=observation.authenticated,
                suppressed=not accepted,
                now_utc=observation.now_utc,
                surface=observation.surface,
                url=observation.url,
                message=observation.message,
                ua_family=observation.ua_family,
                origin_class=observation.origin_class,
                sec_fetch_site=observation.sec_fetch_site,
                request_fingerprint=observation.request_fingerprint,
            )
        return accepted
