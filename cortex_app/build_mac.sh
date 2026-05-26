#!/usr/bin/env bash
# Build the CORTEX test-taker as a standalone macOS .app + .dmg.
#
# Result lands in:
#     cortex_app/dist/CORTEX.app   <- double-clickable
#     cortex_app/dist/CORTEX.dmg   <- distributable disk image
#
# Distribution: attach the .dmg as a GitHub Release asset. First-time
# macOS users see a Gatekeeper warning because the .app is unsigned —
# they right-click -> Open once and macOS remembers the approval. The
# README points them to this workaround.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== CORTEX build (macOS) ==="

# --- preflight ------------------------------------------------------------
# Python 3.11 specifically — matches what the bundled environment ran on,
# avoids surprises across stdlib changes.
PY=""
for cand in python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        v="$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")"
        if [ "$v" = "3.11" ]; then PY="$cand"; break; fi
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: need Python 3.11 to build CORTEX (matches the runtime env)."
    echo "Install from https://www.python.org/downloads/release/python-3119/"
    exit 1
fi
echo "Using $($PY --version)"

if [ ! -f cortex_config.yaml ]; then
    echo "ERROR: cortex_app/cortex_config.yaml is missing."
    echo "Copy cortex_config.yaml.template to cortex_config.yaml and fill"
    echo "in the Dropbox credentials, then re-run this script."
    exit 1
fi
if [ ! -f data/eeg_bank.h5 ]; then
    echo "data/eeg_bank.h5 not found — fetching from S3 ..."
    bash fetch_test_bank.sh
fi

# --- build venv -----------------------------------------------------------
if [ ! -d build_venv ]; then
    echo "Creating build_venv ..."
    "$PY" -m venv build_venv
fi
# shellcheck disable=SC1091
source build_venv/bin/activate
echo "Installing build deps ..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements-cortex.txt
pip install --quiet pyinstaller

# --- build ----------------------------------------------------------------
echo "Running PyInstaller ..."
pyinstaller --clean --noconfirm cortex.spec

echo
echo "=== build done ==="
echo "App bundle: dist/CORTEX.app"

# --- DMG -------------------------------------------------------------------
if command -v hdiutil >/dev/null 2>&1; then
    rm -f dist/CORTEX.dmg
    echo "Creating DMG ..."
    hdiutil create -volname "CORTEX" \
        -srcfolder dist/CORTEX.app \
        -ov -format UDZO \
        dist/CORTEX.dmg
    echo "DMG ready:  dist/CORTEX.dmg"
fi

du -sh dist/CORTEX.app dist/CORTEX.dmg 2>/dev/null || true
