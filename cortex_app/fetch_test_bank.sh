#!/usr/bin/env bash
# Download the curated test bank (~170 MB) for local CORTEX builds and
# stage it where the spec expects it: <repo>/data/eeg_bank.h5
#
# Source: the `build-data-v1` release on this repo. CI uses the same
# release; we keep them in sync so a local build matches what CI ships.
# No AWS interaction — just gh CLI with your normal GitHub auth.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$REPO/data"
DEST="$REPO/data/eeg_bank.h5"

if [ -f "$DEST" ]; then
    echo "Already present: $DEST"
    ls -lh "$DEST"
    exit 0
fi

echo "Downloading eeg_bank.h5 from build-data-v1 release -> $DEST"
gh release download build-data-v1 \
    --repo bdsp-core/ilae-skill-certification-test-multi \
    --pattern eeg_bank.h5 \
    --dir "$REPO/data"
ls -lh "$DEST"
