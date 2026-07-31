#!/usr/bin/env bash
# Build and start the fully local draw-latent session console.
#
# Build follows the established sidecar pattern (scripts/build_precision_cli.sh):
# the toolchain's rolldown bundles the node entry point, erasing the engine's
# type-only imports; type safety comes from the tsc --noEmit pass on
# local_server/tsconfig.json run here first.
#
# Usage:
#   scripts/local_server.sh                start on http://127.0.0.1:8734/
#   PORT=9000 scripts/local_server.sh      alternative port
#   scripts/local_server.sh --build-only   build the bundle and exit
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

out_dir=".local-server-dist"
mkdir -p "$out_dir"
tmp_file="$out_dir/.local_server.$$.mjs"
trap 'rm -f "$tmp_file"' EXIT

../cortex_web/node_modules/.bin/tsc --noEmit -p local_server/tsconfig.json
../cortex_web/node_modules/.bin/rolldown local_server/server.ts \
  --format esm --platform node \
  --file "$tmp_file" >/dev/null
mv -f "$tmp_file" "$out_dir/local_server.mjs"

if [[ "${1:-}" == "--build-only" ]]; then
  echo "built $out_dir/local_server.mjs"
  exit 0
fi

PORT="${PORT:-8734}"
echo "Starting the draw-latent session console on http://127.0.0.1:${PORT}/"
exec env PORT="$PORT" node "$out_dir/local_server.mjs"
