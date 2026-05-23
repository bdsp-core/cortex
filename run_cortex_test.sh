#!/usr/bin/env bash
# CORTEX internal test — launcher.
#
# This is the one command to take the test:
#
#     bash run_cortex_test.sh
#
# On the first run it sets up the environment (a few minutes); after that
# it launches straight away. It always uses the bundled virtual
# environment, so there is no "wrong Python" to worry about.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x "./.venv/bin/python" ]; then
    echo "First run — setting up the environment ..."
    echo
    bash venv_requirements.sh
    echo
fi

echo "Starting the CORTEX test ..."
exec ./.venv/bin/python scripts/eeg_bank_viewer.py
