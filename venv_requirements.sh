#!/usr/bin/env bash
# CORTEX internal test — one-time environment setup.
#
# Creates a local Python virtual environment (.venv) and installs every
# package the test needs. Run once, from this folder:
#
#     bash venv_requirements.sh
#
# This installs Python *packages* — it requires Python 3.11 to already
# be installed on the machine.
set -euo pipefail
cd "$(dirname "$0")"

echo "CORTEX internal test — environment setup"
echo

# --- locate a Python 3.11 interpreter --------------------------------------
PY=""
for cand in python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver="$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' \
               2>/dev/null || echo "")"
        if [ "$ver" = "3.11" ]; then PY="$cand"; break; fi
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: Python 3.11 was not found on this machine."
    echo
    echo "The test requires Python 3.11 specifically. Install it from"
    echo "  https://www.python.org/downloads/release/python-3119/"
    echo "then run  bash venv_requirements.sh  again."
    exit 1
fi
echo "Found $("$PY" --version)"

# --- create the venv and install -------------------------------------------
echo "Creating the virtual environment (.venv) ..."
"$PY" -m venv .venv
echo "Installing packages (this takes a few minutes) ..."
./.venv/bin/python -m pip install --quiet --upgrade pip
./.venv/bin/python -m pip install --quiet -r requirements-cortex.txt

echo
echo "Setup complete. To start the test, run:"
echo "    bash run_cortex_test.sh"
