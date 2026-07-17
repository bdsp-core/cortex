#!/usr/bin/env bash
# Boot a throwaway app server (same pattern as ui_smoke.sh, but no EEG bundle:
# only the public auth/report screens render) and run the visual regression
# smoke against it. Pass --update to (re)write the committed baselines after
# an intentional visual change.
# Requires: a production build in apps/web/dist + the playwright dev dep.
set -euo pipefail
WEB="$(cd "$(dirname "$0")/.." && pwd)"          # cortex_web/apps/web
SERVICES="$(cd "$WEB/../../services" && pwd)"     # cortex_web/services

PORT="${CORTEX_PORT:-8084}"
export CORTEX_JWT_SECRET="ui-smoke-secret"
export CORTEX_DB="/tmp/cortex_visual_smoke.db"
export CORTEX_EMAIL_BACKEND="dev"
export CORTEX_SERVE_STATIC=1
rm -f /tmp/cortex_visual_smoke.db* 2>/dev/null || true

[ -d "$WEB/dist" ] || ( cd "$WEB" && npm run build )

# Prefer the repo venv (has uvicorn/fastapi) when the shell doesn't.
PY="${CORTEX_PY:-python3}"
[ -x "$WEB/../../../.venv/bin/python" ] && PY="${CORTEX_PY:-$WEB/../../../.venv/bin/python}"

( cd "$SERVICES" && "$PY" -m uvicorn api.app:app --port "$PORT" --log-level warning ) \
    >/tmp/cortex_visual_smoke.log 2>&1 &
UV=$!
cleanup() { kill $UV 2>/dev/null || true; rm -f /tmp/cortex_visual_smoke.db* 2>/dev/null || true; }
trap cleanup EXIT INT TERM

for i in $(seq 1 40); do curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1 && break; sleep 0.3; done

echo "▸ visual smoke against http://localhost:$PORT"
node "$WEB/scripts/visual_smoke.mjs" "http://localhost:$PORT" "$@"
