#!/usr/bin/env bash
set -euo pipefail

protocol_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-smoke}"
shift || true

cd "$protocol_dir"
export PYTHONPATH="$protocol_dir/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -m nway_protocol.selector_regret_real \
  .artifacts/categorical_bank_axes.csv --mode "$mode" "$@"
