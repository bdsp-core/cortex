#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

"$ROOT/cortex_web_python_reference/run_tests.sh"

python -m pytest -m "not slow" \
  "$ROOT/tests/test_precision_policy_local_integration.py" \
  "$ROOT/tests/test_choose_item_subsample.py" \
  "$ROOT/tests/test_cortex_ad6_integration.py" \
  "$ROOT/tests/test_cortex_engine_inputs.py" \
  "$ROOT/tests/test_cortex_session_controller.py" \
  "$ROOT/tests/test_coverage_k7_corrt_drift.py" \
  "$ROOT/tests/test_instrument_freeze.py" \
  "$ROOT/tests/test_phase9_scaffolding.py" \
  "$ROOT/tests/test_ell_star_v15.py" \
  "$ROOT/tests/test_trainer_g0_ad6_accessors.py"

if [[ "${CORTEX_RUN_POLICY_GOLDENS:-0}" == "1" ]]; then
  python -m pytest -m slow "$ROOT/tests/test_choose_item_subsample.py"
  (
    cd "$ROOT/cortex_web_python_reference"
    CORTEX_RUN_POLICY_GOLDENS=1 \
      python -m pytest tests/test_policy_packages.py
  )
fi
