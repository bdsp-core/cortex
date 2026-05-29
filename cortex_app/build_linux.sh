#!/usr/bin/env bash
# Build the CORTEX test-taker as a standalone Linux directory bundle.
#
# Result lands in:
#     cortex_app/dist/CORTEX/cortex   <- run from a terminal
#     cortex_app/dist/CORTEX.tar.gz   <- distributable tarball
#
# Distribution: attach the .tar.gz as a GitHub Release asset. Linux
# users extract and run cortex_app/dist/CORTEX/cortex — no Gatekeeper
# / SmartScreen warnings (and no .desktop file shipped; the binary is
# launched directly from a terminal or via the file manager).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

echo "=== CORTEX build (Linux) ==="

# --- preflight ------------------------------------------------------------
# Python 3.11 specifically — matches what the bundled runtime targets.
PY=""
for cand in python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        v="$("$cand" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")"
        if [ "$v" = "3.11" ]; then PY="$cand"; break; fi
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: need Python 3.11 to build CORTEX (matches the runtime env)."
    echo "Ubuntu/Debian:   sudo apt install python3.11 python3.11-venv"
    echo "Fedora/RHEL:     sudo dnf install python3.11"
    echo "Arch:            sudo pacman -S python python-pip"
    echo "Or from source:  https://www.python.org/downloads/release/python-3119/"
    exit 1
fi
echo "Using $($PY --version)"

if [ ! -f "$REPO/cortex_config.yaml" ]; then
    echo "ERROR: $REPO/cortex_config.yaml is missing."
    echo "Copy cortex_config.example.yaml to cortex_config.yaml at the repo"
    echo "root and fill in the Dropbox credentials, then re-run this script."
    exit 1
fi
if [ ! -f "$REPO/data/eeg_bank.h5" ]; then
    echo "data/eeg_bank.h5 not found at the repo root — fetching ..."
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
pip install --quiet -r "$REPO/requirements-cortex.txt"
pip install --quiet pyinstaller

# --- build ----------------------------------------------------------------
echo "Running PyInstaller ..."
pyinstaller --clean --noconfirm cortex.spec

echo
echo "=== build done ==="
echo "Binary directory: dist/CORTEX/"
echo "Launch with:      ./dist/CORTEX/cortex"

# --- tarball --------------------------------------------------------------
# Tarball the entire COLLECT directory so internal testers can extract
# and run with no extra setup. xz compression for size (~30-40% smaller
# than gzip on PyInstaller bundles dominated by .so + .py files).
if [ -d dist/CORTEX ]; then
    echo "Creating tarball ..."
    rm -f dist/CORTEX.tar.gz dist/CORTEX.tar.xz
    tar -C dist -czf dist/CORTEX.tar.gz CORTEX
    echo "Tarball ready: dist/CORTEX.tar.gz"
fi

du -sh dist/CORTEX dist/CORTEX.tar.gz 2>/dev/null || true
