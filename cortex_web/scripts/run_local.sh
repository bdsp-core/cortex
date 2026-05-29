#!/usr/bin/env bash
# Build the SPA and serve the WHOLE app (SPA + EEG bundle + API) from one
# uvicorn process on http://localhost:8000 — the simplest way to test the full
# flow locally before AWS. Same-origin, so no CORS or proxy needed.
#
#   ./scripts/run_local.sh
#
# First run mints a demo participant and prints its credentials. Set
# CORTEX_ADMIN_TOKEN to enable the admin API; CORTEX_PORT to change the port.
set -euo pipefail
cd "$(dirname "$0")/.."

export CORTEX_ADMIN_TOKEN="${CORTEX_ADMIN_TOKEN:-local-admin}"
PORT="${CORTEX_PORT:-8000}"

echo "▸ building SPA (tsc + vite) …"
npm run build

echo "▸ ensuring a demo participant exists …"
if ! python3 -m server.admin list 2>/dev/null | grep -q .; then
  python3 -m server.admin gen --count 1 --prefix demo --out demo_codes.csv
  echo "  demo credentials written to demo_codes.csv:"
  cat demo_codes.csv
fi

echo "▸ serving on http://localhost:${PORT}  (SPA + /bundle + /api)"
echo "  admin token: ${CORTEX_ADMIN_TOKEN}"
exec python3 -m uvicorn server.app:app --host 0.0.0.0 --port "${PORT}"
