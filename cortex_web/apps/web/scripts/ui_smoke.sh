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
# Single-process smoke: uvicorn serves the SPA + bundle as well as /api.
export CORTEX_SERVE_STATIC=1
rm -f /tmp/cortex_ui_smoke.db* 2>/dev/null || true

[ -d "$WEB/dist" ] || ( cd "$WEB" && npm run build )

# Small K=7 smoke bundle: few spike segs so spike-first exhausts quickly and the
# smoke reaches the IIIC phase fast. Skip with CORTEX_SMOKE_NOBUILD=1 to reuse.
if [ -z "${CORTEX_SMOKE_NOBUILD:-}" ]; then
  ( cd "$WEB" && python3 scripts/prepare_web_bundle.py --version _smoke --max 14 --max-spike 6 ) \
    >/tmp/cortex_smoke_bundle.log 2>&1 \
    && echo "▸ built small K=7 smoke bundle (apps/web/public/bundle/_smoke)" \
    || { echo "smoke bundle build failed:"; tail -5 /tmp/cortex_smoke_bundle.log; exit 1; }
fi
export CORTEX_BUNDLE_URL="/bundle/_smoke"

( cd "$SERVICES" && python3 -m uvicorn api.app:app --port "$PORT" --log-level warning ) \
    >/tmp/cortex_ui_smoke.log 2>&1 &
UV=$!
cleanup() { kill $UV 2>/dev/null || true; rm -f /tmp/cortex_ui_smoke.db* 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for i in $(seq 1 40); do curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1 && break; sleep 0.3; done

echo "▸ UI smoke against http://localhost:$PORT"
node "$WEB/scripts/ui_smoke.mjs" "http://localhost:$PORT"
