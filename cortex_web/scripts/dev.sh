#!/usr/bin/env bash
# Dev mode with hot reload: FastAPI on :8000 + Vite dev server on :5173 (which
# proxies /api → :8000, see vite.config.ts). Open http://localhost:5173.
#
#   ./scripts/dev.sh
#
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

export CORTEX_ADMIN_TOKEN="${CORTEX_ADMIN_TOKEN:-local-admin}"
export CORTEX_RELOAD=1

if ! python3 -m server.admin list 2>/dev/null | grep -q .; then
  python3 -m server.admin gen --count 1 --prefix demo --out demo_codes.csv
  echo "demo credentials (demo_codes.csv):"; cat demo_codes.csv
fi

echo "▸ backend  → http://localhost:8000"
python3 -m server.run &
BACK=$!
trap 'kill $BACK 2>/dev/null || true' EXIT INT TERM

echo "▸ frontend → http://localhost:5173 (proxies /api → :8000)"
npm run dev
