"""SNS message-signature verification (SigV1 SHA1withRSA / SigV2 SHA256withRSA).

Closes the documented tradeoff in routers/ses_events.py: the webhook now
verifies that every envelope was actually signed by AWS SNS before acting.
The canonical string is rebuilt exactly per the SNS spec (fixed field order
by message type, "Name\nValue\n" concatenation, absent optional fields
skipped), the signing certificate must come from an
https://sns.<region>.amazonaws.com/ URL, and the RSA signature is checked
with the certificate's public key (`cryptography`, already in the lock).

Certificates are cached per URL for the process lifetime — SNS rotates cert
URLs when it rotates certs, so a cache never serves a stale key. The fetch
is injectable for tests.
"""
from __future__ import annotations

import base64
import re
import urllib.request

from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# Notification vs. (Un)SubscribeConfirmation sign different field sets, in
# this exact order (SNS spec).
_FIELDS_BY_TYPE = {
    "Notification": ("Message", "MessageId", "Subject", "Timestamp",
                     "TopicArn", "Type"),
    "SubscriptionConfirmation": ("Message", "MessageId", "SubscribeURL",
                                 "Timestamp", "Token", "TopicArn", "Type"),
    "UnsubscribeConfirmation": ("Message", "MessageId", "SubscribeURL",
                                "Timestamp", "Token", "TopicArn", "Type"),
}

_CERT_URL_PATTERN = re.compile(
    r"^https://sns\.[a-z0-9-]+\.amazonaws\.com(\.cn)?/", re.IGNORECASE,
)

_CERT_CACHE: dict[str, bytes] = {}


class SnsVerifyError(Exception):
    """The envelope's signature could not be verified."""


def _fetch_cert(url: str) -> bytes:
    """Module-level so tests can monkeypatch (like ses_events._http_get)."""
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (https pattern-enforced by caller)
        return resp.read()


def _public_key(cert_url: str):
    pem = _CERT_CACHE.get(cert_url)
    if pem is None:
        pem = _fetch_cert(cert_url)
        _CERT_CACHE[cert_url] = pem
    key = x509.load_pem_x509_certificate(pem).public_key()
    if not isinstance(key, rsa.RSAPublicKey):
        raise SnsVerifyError("signing certificate does not carry an RSA key")
    return key


def canonical_string(envelope: dict) -> bytes:
    kind = envelope.get("Type")
    fields = _FIELDS_BY_TYPE.get(kind or "")
    if fields is None:
        raise SnsVerifyError(f"unsupported message Type {kind!r}")
    parts: list[str] = []
    for name in fields:
        value = envelope.get(name)
        if value is None:
            if name == "Subject":       # the only optional signed field
                continue
            raise SnsVerifyError(f"envelope is missing signed field {name}")
        parts.append(f"{name}\n{value}\n")
    return "".join(parts).encode("utf-8")


def verify_envelope(envelope: dict) -> None:
    """Raise SnsVerifyError unless the envelope carries a valid AWS signature."""
    cert_url = str(envelope.get("SigningCertURL", ""))
    if not _CERT_URL_PATTERN.match(cert_url):
        raise SnsVerifyError(f"refusing signing cert from {cert_url!r}")
    version = str(envelope.get("SignatureVersion", ""))
    if version == "1":
        digest = hashes.SHA1()  # noqa: S303 — mandated by the SNS SigV1 spec
    elif version == "2":
        digest = hashes.SHA256()
    else:
        raise SnsVerifyError(f"unsupported SignatureVersion {version!r}")
    try:
        signature = base64.b64decode(str(envelope.get("Signature", "")),
                                     validate=True)
    except Exception as e:
        raise SnsVerifyError(f"bad signature encoding: {e}")
    message = canonical_string(envelope)
    try:
        _public_key(cert_url).verify(signature, message,
                                     padding.PKCS1v15(), digest)
    except SnsVerifyError:
        raise
    except Exception:
        raise SnsVerifyError("signature does not verify")
