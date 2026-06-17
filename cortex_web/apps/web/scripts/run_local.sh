#!/usr/bin/env bash
# Build the SPA and serve the WHOLE app (SPA + EEG bundle + API) from one
# uvicorn process on http://localhost:8000 — the simplest way to test the full
# flow locally before AWS. Single-process, so we opt uvicorn into static
# serving (CORTEX_SERVE_STATIC=1); prod uses Caddy for static instead.
#
#   apps/web/scripts/run_local.sh
#
# First run mints a demo participant and prints its credentials.
set -euo pipefail
WEB="$(cd "$(dirname "$0")/.." && pwd)"          # cortex_web/apps/web
SERVICES="$(cd "$WEB/../../services" && pwd)"     # cortex_web/services

export CORTEX_ADMIN_TOKEN="${CORTEX_ADMIN_TOKEN:-local-admin}"
export CORTEX_SERVE_STATIC=1     # uvicorn also serves the SPA + bundle here
PORT="${CORTEX_PORT:-8000}"

echo "▸ building SPA (tsc + vite) …"
( cd "$WEB" && npm run build )

echo "▸ ensuring a demo participant exists …"
if ! ( cd "$SERVICES" && python3 -m api.admin list 2>/dev/null | grep -q . ); then
  ( cd "$SERVICES" && python3 -m api.admin gen --count 1 --prefix demo --out "$WEB/demo_codes.csv" )
  echo "  demo credentials written to $WEB/demo_codes.csv:"; cat "$WEB/demo_codes.csv"
fi

echo "▸ serving on http://localhost:${PORT}  (SPA + /bundle + /api)"
echo "  admin token: ${CORTEX_ADMIN_TOKEN}"
cd "$SERVICES" && exec python3 -m uvicorn api.app:app --host 0.0.0.0 --port "${PORT}"
