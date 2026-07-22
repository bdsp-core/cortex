#!/usr/bin/env bash
# Reproducible local/CI gate for the canonical CORTEX web replacement.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${CORTEX_QUALITY_PYTHON:-python3}"

cd "$ROOT"

npm run lint
npm run typecheck
npm test
"$PYTHON_BIN" -m pytest -q
npm run build
node apps/web/scripts/csp_verify.mjs
npm audit --audit-level=high

if [ "${CORTEX_BROWSER_GATES:-0}" = "1" ]; then
  npm run -w cortex-web worker-smoke
  # `npm run build` above just produced dist/, so the smoke reuses it instead
  # of building the same tree twice. A standalone ui-smoke rebuilds.
  CORTEX_SMOKE_REUSE_DIST=1 npm run -w cortex-web ui-smoke
fi
