#!/usr/bin/env bash
# Boot the app (one uvicorn serving SPA + bundle + API), mint a participant,
# run the headless-Chrome UI smoke against it, then tear everything down.
# Requires: a production build in dist/ (run `npm run build` first) and the
# `playwright` dev dep + system Chrome.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${CORTEX_PORT:-8083}"
export CORTEX_ADMIN_TOKEN="ui-smoke-admin"
export CORTEX_JWT_SECRET="ui-smoke-secret"
export CORTEX_DB="/tmp/cortex_ui_smoke.db"
rm -f /tmp/cortex_ui_smoke.db* 2>/dev/null || true

[ -d dist ] || npm run build

# Build a SMALL K=7 smoke bundle (few spike segs) so the smoke can walk the
# spike phase to exhaustion and reach the IIIC phase quickly, then serve it.
if [ -z "${CORTEX_SMOKE_NOBUILD:-}" ]; then
  python3 scripts/prepare_web_bundle.py --version _smoke --max 14 >/tmp/cortex_smoke_bundle.log 2>&1 \
    && echo "▸ built small K=7 smoke bundle (public/bundle/_smoke)" \
    || { echo "smoke bundle build failed:"; tail -5 /tmp/cortex_smoke_bundle.log; exit 1; }
fi
export CORTEX_BUNDLE_URL="/bundle/_smoke"

python3 -m uvicorn server.app:app --port "$PORT" --log-level warning >/tmp/cortex_ui_smoke.log 2>&1 &
UV=$!
cleanup() { kill $UV 2>/dev/null || true; rm -f /tmp/cortex_ui_smoke.db* 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for i in $(seq 1 40); do curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1 && break; sleep 0.3; done

CRED=$(curl -s -X POST "http://localhost:$PORT/api/admin/participants" \
  -H "X-Admin-Token: ui-smoke-admin" -H "Content-Type: application/json" \
  -d '{"count":1,"prefix":"qa"}')
CODE=$(echo "$CRED" | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["code"])')
PW=$(echo "$CRED"   | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["password"])')

echo "▸ UI smoke against http://localhost:$PORT  (participant $CODE)"
node scripts/ui_smoke.mjs "http://localhost:$PORT" "$CODE" "$PW"
