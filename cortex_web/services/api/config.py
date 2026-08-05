"""Environment-derived configuration, resolved once at import.

Paths and defaults only — no side effects beyond reading env + the RELEASE
stamp. Values that tests/dev need to re-read per-app (bundle dir/url, session
sample, spacing) are ALSO re-read from env inside create_app(), so pointing a
test client at a fixture bundle via env keeps working; the constants here are
the process-wide defaults.
"""
from __future__ import annotations

import os
from pathlib import Path

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

# Default bundle the SPA pulls (overridable via env for S3/CloudFront).
DEFAULT_BUNDLE_URL = os.environ.get("CORTEX_BUNDLE_URL", "/bundle/v1.5-k7")
# The frozen Precision profile was qualified on the full 35k served bank. It
# is a separate pilot input so the public AD6 account population keeps its
# existing bundle/sample profile during the account-scoped rollout.
DEFAULT_PRECISION_BUNDLE_URL = os.environ.get(
    "CORTEX_PRECISION_BUNDLE_URL", "/bundle/v1.6-k7-35k")
# Per-session candidate-pool size (the server-drawn subset the client engine
# selects within). The prod value is env-driven from /etc/cortex/cortex.env
# (provision.sh writes 500); speculative precompute hides the between-question
# latency that made smaller pools attractive pre-v1.6 (engine/_latency_bench
# has the pool-size numbers).
DEFAULT_SESSION_SAMPLE = int(os.environ.get("CORTEX_SESSION_SAMPLE", "400"))

# Deployed-commit stamp. deploy_app.sh writes cortex_web/RELEASE on the box at
# deploy time; absent (dev/CI checkouts) → None. Surfaced by /api/health so
# "what SHA is live?" is answerable from the outside.
_RELEASE_FILE = CORTEX_WEB / "RELEASE"
RELEASE = (_RELEASE_FILE.read_text().strip()[:200] or None) if _RELEASE_FILE.exists() else None

TOKEN_TTL = int(os.environ.get("CORTEX_TOKEN_TTL", str(6 * 3600)))

# Where user reports (the /report page) are emailed. Overridable via env.
REPORT_TO = os.environ.get("CORTEX_REPORT_TO", "elikeldsen@icloud.com")

# Public site origin used to build links in outbound email (verify/reset
# deep links, the digest button, cohort invites). Read per call, not at
# import, because tests and dev set it per case. Returns "" when unset;
# each caller decides what that means — mailer and the digest omit the link
# entirely, cohort invites fall back to the canonical production origin.
DEFAULT_PUBLIC_ORIGIN = "https://app.cortexeeg.org"


def public_origin() -> str:
    """The configured origin with any trailing slash removed, or ""."""
    return os.environ.get("CORTEX_PUBLIC_ORIGIN", "").strip().rstrip("/")

# Training-protocol exposure gate (the adaptive trainer went live for all users
# 2026-07-07). A soft, flag-based control so exposure is reversible WITHOUT a
# revert-and-redeploy:
#   "all"    — every authenticated user (the current prod posture)
#   "cohort" — only participants in CORTEX_TRAINING_ALLOWLIST (a pilot cohort)
#   "off"    — nobody (kill-switch: hides the entry + 403s the start endpoints;
#              in-flight sessions still finalize)
# The allowlist matches a participant's code, 9-digit public_id, or email
# (case-insensitive). These are ALSO re-read per-app in create_app() so tests
# can flip the mode via env.
TRAINING_MODE = os.environ.get("CORTEX_TRAINING_MODE", "all").strip().lower()
TRAINING_ALLOWLIST = frozenset(
    x.strip().lower()
    for x in os.environ.get("CORTEX_TRAINING_ALLOWLIST", "").split(",")
    if x.strip()
)

# Certification stopping-policy rollout:
#   "all"             — every new authenticated sitting uses precision_v1
#   "email_allowlist" — only normalized emails in PRECISION_POLICY_EMAILS
#   "off"             — every new sitting uses AD6 (immediate rollback)
# Existing sittings always retain their persisted policy stamp. Unknown modes
# fail closed to AD6 in routers/testing.py.
PRECISION_POLICY_ROLLOUT = os.environ.get(
    "CORTEX_PRECISION_POLICY_ROLLOUT", "all").strip().lower()
PRECISION_POLICY_EMAILS = frozenset(
    x.strip().lower()
    for x in os.environ.get(
        "CORTEX_PRECISION_POLICY_EMAILS", "elikeldsen@icloud.com").split(",")
    if x.strip()
)

# Precision c1 stopping-recalibration rollout (n_min 20->0, persistence
# 2->3; qualified by the 2026-08 nmin-stopping-study). OFF by default —
# unknown modes fail closed to the shipped 20/2 configuration. Stamped per
# sitting (sessions.precision_recalibration) so resume and server-side
# replay verification always use the sitting's own stopping constants.
PRECISION_C1_ROLLOUT = os.environ.get(
    "CORTEX_PRECISION_C1_ROLLOUT", "off").strip().lower()
PRECISION_C1_EMAILS = frozenset(
    x.strip().lower()
    for x in os.environ.get("CORTEX_PRECISION_C1_EMAILS", "").split(",")
    if x.strip()
)

# Graded bias-flag reporting rollout (bias-flag-tiers-v1; report-only, never
# read by stopping/selection/verdicts). "all" serves the graded WATCH/EXTREME/
# EXTREME_CONFIRMED flags + withheld reasons; LEGACY by default — unknown
# modes fail closed to the historical interval-clears flags. Stamped per
# sitting (sessions.bias_flag_tiers) so resume and server-side replay
# verification always use the sitting's own reporting mode.
BIAS_FLAG_TIERS = os.environ.get(
    "CORTEX_BIAS_FLAG_TIERS", "legacy").strip().lower()
BIAS_FLAG_TIERS_EMAILS = frozenset(
    x.strip().lower()
    for x in os.environ.get("CORTEX_BIAS_FLAG_TIERS_EMAILS", "").split(",")
    if x.strip()
)

# Precision browser-compute rollout. This is deliberately independent of the
# stopping-policy rollout: it changes execution placement only, is OFF by
# default, and is stamped per sitting so resume never changes execution mode
# halfway through an assessment.
PRECISION_COMPUTE_ROLLOUT = os.environ.get(
    "CORTEX_PRECISION_COMPUTE_ROLLOUT", "off").strip().lower()
PRECISION_COMPUTE_EMAILS = frozenset(
    x.strip().lower()
    for x in os.environ.get("CORTEX_PRECISION_COMPUTE_EMAILS", "").split(",")
    if x.strip()
)

# N-way response-model rollout: which response model a NEW precision sitting
# is stamped with.
#   "all"             — every new precision sitting gets the qualified
#                       draw-latent profile
#   "email_allowlist" — only normalized emails in NWAY_RESPONSE_EMAILS
#   "off"             — every new sitting keeps the floor015 mixture stamp
# OFF is deliberately the default; unknown values fail closed to the mixture
# in nway_profile.py. Existing sittings always retain their persisted stamp,
# so flipping this changes only new-session assignment.
NWAY_RESPONSE_ROLLOUT = os.environ.get(
    "CORTEX_NWAY_RESPONSE_ROLLOUT", "off").strip().lower()
NWAY_RESPONSE_EMAILS = frozenset(
    x.strip().lower()
    for x in os.environ.get("CORTEX_NWAY_RESPONSE_EMAILS", "").split(",")
    if x.strip()
)

# Provisional historical percentile rollout. OFF is deliberately the default.
# Public exposure (cohort/all) also requires CORTEX_PERCENTILE_RELEASE_SHA256
# to equal the verified runtime binary hash; shadow calculates and stores
# scores but never returns display=true.
PERCENTILE_MODE = os.environ.get("CORTEX_PERCENTILE_MODE", "off").strip().lower()
PERCENTILE_ALLOWLIST = frozenset(
    x.strip().lower()
    for x in os.environ.get("CORTEX_PERCENTILE_ALLOWLIST", "").split(",")
    if x.strip()
)
PERCENTILE_RELEASE_SHA256 = os.environ.get(
    "CORTEX_PERCENTILE_RELEASE_SHA256", ""
).strip().lower()
PERCENTILE_NORM_METADATA = Path(os.environ.get(
    "CORTEX_PERCENTILE_NORM_METADATA",
    str(CORTEX_WEB / "apps" / "web" / "public" / "norms"
        / "historical-calibration-k7-provisional-v1.json"),
))
