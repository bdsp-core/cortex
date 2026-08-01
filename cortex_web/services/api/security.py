"""Dependency-light auth primitives — PBKDF2 password hashing + HS256 JWT.

We deliberately avoid PyJWT / bcrypt / passlib so the backend runs on a bare
Python 3.11+ with only FastAPI installed. Both primitives are standard,
well-understood constructions built on the stdlib `hashlib` / `hmac`:

  * Passwords: PBKDF2-HMAC-SHA256, 200k iterations, 16-byte per-user salt.
    Stored as ``pbkdf2_sha256$<iters>$<salt_b64>$<hash_b64>``.
  * Tokens: compact JWT (HS256) — base64url(header).base64url(payload).sig.
    Signed with a server secret; `exp` enforced on decode.

If the deployment later wants argon2id, swap `hash_password` /
`verify_password` — the stored-format prefix lets old hashes keep verifying.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

# ───────────────────────── password hashing ─────────────────────────

_PBKDF2_ITERS = 200_000
_PBKDF2_ALGO = "sha256"
_SALT_BYTES = 16


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def hash_password(password: str, *, iters: int = _PBKDF2_ITERS) -> str:
    """Return a self-describing PBKDF2 hash string for `password`."""
    salt = secrets.token_bytes(_SALT_BYTES)
    dk = hashlib.pbkdf2_hmac(_PBKDF2_ALGO, password.encode("utf-8"), salt, iters)
    return f"pbkdf2_{_PBKDF2_ALGO}${iters}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify `password` against a stored PBKDF2 hash."""
    try:
        algo_tag, iters_s, salt_b64, hash_b64 = stored.split("$")
        if not algo_tag.startswith("pbkdf2_"):
            return False
        algo = algo_tag.removeprefix("pbkdf2_")
        iters = int(iters_s)
        salt = _b64d(salt_b64)
        expected = _b64d(hash_b64)
    except (ValueError, AttributeError):
        return False
    dk = hashlib.pbkdf2_hmac(algo, password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(dk, expected)


# ───────────────────────── JWT (HS256) ─────────────────────────


class TokenError(Exception):
    """Raised when a token is malformed, mis-signed, or expired."""


# Dev-secret cache, keyed by resolved file path. The env var is deliberately
# NOT cached (tests swap it per-case); the file path is — this avoids a disk
# read on every token/code operation AND keeps the in-process secret stable
# when the file can't be written (read-only FS): the uncached version minted a
# fresh secret per call there, so no token could ever verify.
_FILE_SECRET_CACHE: dict[str, bytes] = {}


def _jwt_secrets() -> list[bytes]:
    """Resolve the secret list: first entry SIGNS, every entry VERIFIES.

    CORTEX_JWT_SECRET may hold a comma-separated list to support zero-logout
    key rotation: put the fresh secret first, keep the outgoing one behind it
    for at least the token TTL (and the 15-minute email-code TTL), then drop
    it. A single-value env behaves exactly as before. Falls back to the
    persisted dev secret file so a server restart doesn't invalidate live
    sessions during local testing.
    """
    env = os.environ.get("CORTEX_JWT_SECRET")
    if env:
        parts = [p.strip() for p in env.split(",") if p.strip()]
        if parts:
            return [p.encode("utf-8") for p in parts]
    path = Path(os.environ.get("CORTEX_JWT_SECRET_FILE",
                               Path(__file__).with_name(".jwt_secret")))
    key = str(path)
    cached = _FILE_SECRET_CACHE.get(key)
    if cached is not None:
        return [cached]
    if path.exists():
        secret = path.read_bytes()
    else:
        secret = secrets.token_bytes(32)
        try:
            path.write_bytes(secret)
            path.chmod(0o600)
        except OSError:
            pass  # read-only FS (e.g. Lambda) — keep the in-process secret
    _FILE_SECRET_CACHE[key] = secret
    return [secret]


def _jwt_secret() -> bytes:
    """The SIGNING secret (first of the list)."""
    return _jwt_secrets()[0]


def issue_token(subject: str, *, ttl_seconds: int = 6 * 3600,
                extra: dict | None = None, now: int | None = None) -> str:
    """Issue a signed HS256 JWT for `subject`, valid for `ttl_seconds`."""
    iat = int(now if now is not None else time.time())
    payload = {"sub": subject, "iat": iat, "exp": iat + ttl_seconds}
    if extra:
        payload.update(extra)
    header = {"alg": "HS256", "typ": "JWT"}
    seg_h = _b64e(json.dumps(header, separators=(",", ":")).encode())
    seg_p = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{seg_h}.{seg_p}".encode("ascii")
    sig = hmac.new(_jwt_secret(), signing_input, hashlib.sha256).digest()
    return f"{seg_h}.{seg_p}.{_b64e(sig)}"


def decode_token(token: str, *, now: int | None = None) -> dict:
    """Verify signature + expiry; return the claims dict or raise TokenError."""
    try:
        seg_h, seg_p, seg_s = token.split(".")
    except ValueError:
        raise TokenError("malformed token")
    signing_input = f"{seg_h}.{seg_p}".encode("ascii")
    try:
        got = _b64d(seg_s)
    except Exception:
        raise TokenError("bad signature encoding")
    # Any secret in the rotation list may have signed this token. Every
    # candidate is checked with a constant-time compare; the loop bound is
    # the configured key count, not attacker-controlled input.
    if not any(
        hmac.compare_digest(
            got, hmac.new(secret, signing_input, hashlib.sha256).digest())
        for secret in _jwt_secrets()
    ):
        raise TokenError("bad signature")
    try:
        claims = json.loads(_b64d(seg_p))
    except Exception:
        raise TokenError("bad payload")
    exp = claims.get("exp", 0)
    if int(now if now is not None else time.time()) >= int(exp):
        raise TokenError("expired")
    return claims


# ───────────────────── short-lived email codes ─────────────────────
# 6-digit codes for email verification + password reset. A 6-digit space is
# only 1e6, so the protections are: short expiry, a hard per-code attempt cap
# (enforced at the DB layer), and a keyed HMAC-SHA256 at rest (peppered with
# the same server secret as the JWT) so a DB leak doesn't expose live codes.

CODE_TTL_SECONDS = 15 * 60
CODE_MAX_ATTEMPTS = 5


def gen_numeric_code(n_digits: int = 6) -> str:
    """A cryptographically-random zero-padded numeric code."""
    return "".join(str(secrets.randbelow(10)) for _ in range(n_digits))


def hash_code(code: str) -> str:
    """Keyed HMAC-SHA256 of a short code, for storage at rest."""
    return _b64e(hmac.new(_jwt_secret(), code.encode("utf-8"), hashlib.sha256).digest())


def verify_code(code: str, stored_hash: str) -> bool:
    """Constant-time compare of a presented code against a stored hash.

    Codes are peppered with the signing secret; during a key rotation an
    in-flight code may have been hashed under the outgoing secret, so every
    secret in the rotation list is a candidate pepper."""
    try:
        return any(
            hmac.compare_digest(
                _b64e(hmac.new(secret, code.encode("utf-8"),
                               hashlib.sha256).digest()),
                stored_hash)
            for secret in _jwt_secrets()
        )
    except Exception:
        return False
