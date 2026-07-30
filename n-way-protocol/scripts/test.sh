#!/usr/bin/env bash
set -euo pipefail

protocol_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cpu_total="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)"
worker_budget=$((cpu_total > 4 ? cpu_total - 2 : cpu_total))

cd "$protocol_dir"

../cortex_web/node_modules/.bin/tsc --noEmit -p tsconfig.json &
typecheck_pid=$!

test_files="$(find tests -maxdepth 1 -name '*.test.ts' | wc -l)"
test_workers=$((test_files < worker_budget ? test_files : worker_budget))
../cortex_web/node_modules/.bin/vitest run tests \
  --pool threads \
  --poolOptions.threads.minThreads 1 \
  --poolOptions.threads.maxThreads "$test_workers" &
typescript_pid=$!

python3 -m pytest -q python/tests &
python_pid=$!

status=0
wait "$typecheck_pid" || status=1
wait "$typescript_pid" || status=1
wait "$python_pid" || status=1
exit "$status"
