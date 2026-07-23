#!/usr/bin/env bash
# Boot the app (one uvicorn serving SPA + bundle + API via CORTEX_SERVE_STATIC),
# run the headless-Chrome UI smoke against it (public signup, no pre-minted
# creds), then tear it down.
# Requires: a production build in apps/web/dist, the `playwright` dev dep +
# system Chrome, and the bundle-build deps (h5py/numpy/…).
set -euo pipefail
WEB="$(cd "$(dirname "$0")/.." && pwd)"          # cortex_web/apps/web
SERVICES="$(cd "$WEB/../../services" && pwd)"     # cortex_web/services

PORT="${CORTEX_PORT:-8083}"
export CORTEX_JWT_SECRET="ui-smoke-secret"
export CORTEX_DB="/tmp/cortex_ui_smoke.db"
# Dev email backend that echoes the 6-digit code back in the API response so
# the headless flow can read it (register → verify). Never enabled in prod.
export CORTEX_EMAIL_BACKEND="dev"
export CORTEX_EMAIL_EXPOSE_CODE="1"
# The tiny smoke bank cannot satisfy Precision's full-bank content contract;
# Precision worker execution has its own real-browser gate (worker-smoke).
export CORTEX_PRECISION_POLICY_ROLLOUT="off"
export CORTEX_PRECISION_COMPUTE_ROLLOUT="off"
# Default smoke is the noninterference posture. A dedicated percentile smoke
# may explicitly override this to all/cohort plus the release SHA.
export CORTEX_PERCENTILE_MODE="${CORTEX_PERCENTILE_MODE:-off}"
# Single-process smoke: uvicorn serves the SPA + bundle as well as /api.
export CORTEX_SERVE_STATIC=1
rm -f /tmp/cortex_ui_smoke.db* 2>/dev/null || true

# Build unless the caller just built. Reusing whatever happens to sit in dist/
# means the smoke can pass against a PREVIOUS revision while you believe you
# are testing your working tree — the gate builds immediately before calling
# this, so it opts out; a standalone run always rebuilds.
if [ "${CORTEX_SMOKE_REUSE_DIST:-0}" = "1" ] && [ -d "$WEB/dist" ]; then
  echo "▸ reusing the existing build in apps/web/dist"
else
  ( cd "$WEB" && npm run build )
fi

# Prefer the repository environment so the smoke uses the same pinned FastAPI,
# NumPy, and bundle-builder dependencies as the backend suite.
PY="${CORTEX_PY:-python3}"
[ -x "$WEB/../../../.venv/bin/python" ] && PY="${CORTEX_PY:-$WEB/../../../.venv/bin/python}"

# Small K=7 smoke bundle: few spike segs so spike-first exhausts quickly and the
# smoke reaches the IIIC phase fast. Skip with CORTEX_SMOKE_NOBUILD=1 to reuse.
if [ -z "${CORTEX_SMOKE_NOBUILD:-}" ]; then
  if [ "${CORTEX_SMOKE_SYNTHETIC:-0}" = "1" ]; then
    BUILD_COMMAND=("$PY" scripts/build_smoke_bundle.py)
  else
    BUILD_COMMAND=("$PY" scripts/prepare_web_bundle.py --version _smoke --max 14 --max-spike 6)
  fi
  ( cd "$WEB" && "${BUILD_COMMAND[@]}" ) \
    >/tmp/cortex_smoke_bundle.log 2>&1 \
    && echo "▸ built small K=7 smoke bundle (apps/web/public/bundle/_smoke)" \
    || { echo "smoke bundle build failed:"; tail -5 /tmp/cortex_smoke_bundle.log; exit 1; }
fi
export CORTEX_BUNDLE_URL="/bundle/_smoke"

# Refuse to run when something already owns the port. Otherwise uvicorn fails
# to bind, the readiness probe below is answered by the LEFTOVER server, and
# the smoke silently exercises whatever code that process is running. Observed:
# a leaked server kept serving a hours-old revision, and its accumulated
# per-IP rate-limit state (register is 5/hour) eventually failed signup —
# which reads as a code regression, not a stale process.
if curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; then
  echo "port $PORT is already serving — refusing to smoke against it." >&2
  echo "find and stop the leftover process:  ss -lptn 'sport = :$PORT'" >&2
  exit 1
fi

# `exec` so uvicorn REPLACES the subshell. Without it $! is the subshell's pid
# and uvicorn is its child, so the cleanup below killed the wrapper and left
# uvicorn orphaned — still holding the port and still serving that revision.
# That is how stale servers accumulated and quietly answered later runs.
( cd "$SERVICES" && exec "$PY" -m uvicorn api.app:app --port "$PORT" --log-level warning ) \
    >/tmp/cortex_ui_smoke.log 2>&1 &
UV=$!
cleanup() {
  kill "$UV" 2>/dev/null || true
  wait "$UV" 2>/dev/null || true      # reap it before the port check next run
  rm -f /tmp/cortex_ui_smoke.db* 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in $(seq 1 40); do curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1 && break; sleep 0.3; done

# The readiness probe above cannot tell OUR server from someone else's, so
# confirm the process we started is the one still running.
if ! kill -0 "$UV" 2>/dev/null; then
  echo "the smoke server exited before becoming ready:" >&2
  tail -5 /tmp/cortex_ui_smoke.log >&2
  exit 1
fi

echo "▸ UI smoke against http://localhost:$PORT"
node "$WEB/scripts/ui_smoke.mjs" "http://localhost:$PORT"
