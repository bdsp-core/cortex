#!/usr/bin/env bash
# Download the curated test bank (~170 MB) from S3 and stage it where the
# spec expects it: cortex_app/data/eeg_bank.h5
#
# The bank is the same byte-for-byte file as the in-repo test_h5.h5 (100
# IIIC + 100 spike cases, /iiic/<seg_id>/{eeg30s,sdata,sfreqs,stimes} and
# /spike/<seg_id>/{eeg30s}). It's gitignored because it exceeds GitHub's
# 100 MB hard per-file limit.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data
DEST="data/eeg_bank.h5"
URI="s3://bdsp-opendata-credentialed/eeg-test/test_h5.h5"
PROFILE="${AWS_PROFILE:-opendata}"   # read-only is fine

if [ -f "$DEST" ]; then
    echo "Already present: $DEST"
    ls -lh "$DEST"
    exit 0
fi

echo "Downloading $URI -> $DEST  (~170 MB)"
aws --profile "$PROFILE" s3 cp "$URI" "$DEST"
ls -lh "$DEST"
