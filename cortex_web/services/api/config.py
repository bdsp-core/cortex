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
# selects within). Prod runs 700 (env): speculative precompute hides the
# between-question latency that made smaller pools attractive pre-v1.6
# (engine/_latency_bench has the pool-size numbers).
DEFAULT_SESSION_SAMPLE = int(os.environ.get("CORTEX_SESSION_SAMPLE", "400"))

# Deployed-commit stamp. deploy_app.sh writes cortex_web/RELEASE on the box at
# deploy time; absent (dev/CI checkouts) → None. Surfaced by /api/health so
# "what SHA is live?" is answerable from the outside.
_RELEASE_FILE = CORTEX_WEB / "RELEASE"
RELEASE = (_RELEASE_FILE.read_text().strip()[:200] or None) if _RELEASE_FILE.exists() else None

TOKEN_TTL = int(os.environ.get("CORTEX_TOKEN_TTL", str(6 * 3600)))

# Where user reports (the /report page) are emailed. Overridable via env.
REPORT_TO = os.environ.get("CORTEX_REPORT_TO", "elikeldsen@icloud.com")

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
