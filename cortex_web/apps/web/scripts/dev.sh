#!/usr/bin/env bash
# Dev mode with hot reload: FastAPI on :8000 + Vite dev server on :5173 (which
# proxies /api → :8000, see vite.config.ts). Open http://localhost:5173.
# Vite serves the SPA in dev, so the API stays pure (no CORTEX_SERVE_STATIC).
#
#   apps/web/scripts/dev.sh
#
# Ctrl-C stops both.
set -euo pipefail
WEB="$(cd "$(dirname "$0")/.." && pwd)"          # cortex_web/apps/web
SERVICES="$(cd "$WEB/../../services" && pwd)"     # cortex_web/services

export CORTEX_ADMIN_TOKEN="${CORTEX_ADMIN_TOKEN:-local-admin}"
export CORTEX_RELOAD=1

if ! ( cd "$SERVICES" && python3 -m api.admin list 2>/dev/null | grep -q . ); then
  ( cd "$SERVICES" && python3 -m api.admin gen --count 1 --prefix demo --out "$WEB/demo_codes.csv" )
  echo "demo credentials ($WEB/demo_codes.csv):"; cat "$WEB/demo_codes.csv"
fi

echo "▸ backend  → http://localhost:8000"
( cd "$SERVICES" && python3 -m api.run ) &
BACK=$!
trap 'kill $BACK 2>/dev/null || true' EXIT INT TERM

echo "▸ frontend → http://localhost:5173 (proxies /api → :8000)"
cd "$WEB" && npm run dev
