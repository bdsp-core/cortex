#!/usr/bin/env bash
# Bundle the Precision-stopping sidecar (src/precision_cli.ts) for node.
#
# The repo toolchain (cortex_web/node_modules/.bin) carries no `tsx`, and a
# plain `tsc` CommonJS emit is impossible: the type-only import closure of the
# unchanged engine reaches nway_selector_executor.ts, whose `import.meta`
# worker URL only compiles under ESM. `rolldown` (the same toolchain dir) is
# therefore the build path: it erases type-only imports before bundling, so
# the emitted bundle contains exactly the runtime closure of the sidecar.
# Type-safety still comes from `npm run typecheck` (tsc --noEmit), which
# covers src/precision_cli.ts.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

out_dir=".precision-cli-dist"
mkdir -p "$out_dir"
tmp_file="$out_dir/.precision_cli.$$.mjs"
trap 'rm -f "$tmp_file"' EXIT

../cortex_web/node_modules/.bin/rolldown src/precision_cli.ts \
  --format esm --platform node \
  --file "$tmp_file" >/dev/null

# Atomic publish so concurrently starting workers never see a partial bundle.
mv -f "$tmp_file" "$out_dir/precision_cli.mjs"
