#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"

POLICY_PATHS="$WORKSPACE/termination-policy/src:$WORKSPACE/ad6-policy/src:$WORKSPACE/precision-policy/src"
export PYTHONPATH="$POLICY_PATHS${PYTHONPATH:+:$PYTHONPATH}"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

for project in termination-policy ad6-policy precision-policy; do
  (
    cd "$WORKSPACE/$project"
    python -m pytest
  )
done

python -m pytest "$ROOT/tests"

(
  cd "$WORKSPACE/cortex_web"
  npx vitest run cortex_web_python_reference/parity/cortex_web_parity.test.ts \
    --root "$WORKSPACE"
  npx vitest run \
    apps/web/engine/mathfns.test.ts \
    apps/web/engine/pipeline.test.ts \
    apps/web/engine/resume_replay.test.ts \
    apps/web/engine/termination.test.ts \
    apps/web/engine/v15_drift.test.ts \
    --config apps/web/vite.config.ts
)
