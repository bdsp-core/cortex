#!/usr/bin/env bash
# Bundle the deterministic session replayer (scripts/replay_precision_session
# .ts) for plain node — the server-side result-verification runner
# (services/api/result_verify.py) executes the bundle to re-derive a sitting
# from its stored picks and compare against what the client submitted.
#
# Same rationale and mechanics as n-way-protocol/scripts/build_precision_cli
# .sh: the toolchain carries no tsx, and the engine's import closure only
# compiles under ESM, so rolldown (already in cortex_web/node_modules/.bin)
# erases type-only imports and emits the exact runtime closure. Type safety
# comes from `tsc --noEmit`, which covers this script's sources.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

out_dir=".replay-verifier-dist"
mkdir -p "$out_dir"
tmp_file="$out_dir/.replay_precision_session.$$.mjs"
trap 'rm -f "$tmp_file"' EXIT

../../node_modules/.bin/rolldown scripts/replay_precision_session.ts \
  --format esm --platform node \
  --file "$tmp_file" >/dev/null

# Atomic publish so a concurrently starting verifier never sees a partial
# bundle.
mv -f "$tmp_file" "$out_dir/replay_precision_session.mjs"
