#!/usr/bin/env bash
# Clean, staged deployment of the canonical cortex_web tree.
#
# Run from a clean checkout whose HEAD exactly matches origin/main:
#
#   bash cortex_web/deploy/scripts/deploy_app.sh
#
# The release is built under /opt/cortex/releases before the stable
# /opt/cortex/cortex_web path is switched. Persistent Postgres data, secrets,
# and the external EEG bank are never copied into or removed with a release.
set -euo pipefail

SSH_HOST="${CORTEX_SSH:-cortex-prod}"
CORTEX_USER="${CORTEX_USER:-cortex}"

say() { printf "▸ %s\n" "$*"; }

readiness_gate() {
  local base="$1" i body=""
  for i in $(seq 1 20); do
    body=$(curl -fsS "$base/api/health?deep=1" 2>/dev/null || true)
    if printf '%s' "$body" | grep -q '"ok": *true'; then
      say "readiness OK — $body"
      return 0
    fi
    sleep 1
  done
  printf 'readiness failed for %s; last body: %s\n' "$base" "${body:-<none>}" >&2
  return 1
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WEB_LOCAL=$(cd "$SCRIPT_DIR/../.." && pwd)
REPO_ROOT=$(git -C "$WEB_LOCAL" rev-parse --show-toplevel)

if [ ! -d "$WEB_LOCAL/apps/web" ] || [ ! -d "$WEB_LOCAL/services/api" ]; then
  printf 'could not locate canonical cortex_web from %s\n' "$SCRIPT_DIR" >&2
  exit 2
fi

if [ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]; then
  printf '%s\n' "refusing to deploy a dirty repository; commit or discard every change first" >&2
  exit 2
fi

git -C "$REPO_ROOT" fetch --quiet origin main
HEAD_SHA=$(git -C "$REPO_ROOT" rev-parse HEAD)
MAIN_SHA=$(git -C "$REPO_ROOT" rev-parse origin/main)
if [ "$HEAD_SHA" != "$MAIN_SHA" ]; then
  printf 'refusing to deploy %s: origin/main is %s\n' "$HEAD_SHA" "$MAIN_SHA" >&2
  exit 2
fi

RELEASE_ID="$(date -u +%Y%m%dT%H%M%SZ)-${HEAD_SHA:0:12}"
REMOTE_STAGE="/tmp/cortex-release-$RELEASE_ID"
REMOTE_RELEASE="/opt/cortex/releases/$RELEASE_ID"

rollback_remote() {
  say "rolling back to the previously active release"
  ssh "$SSH_HOST" \
    "sudo '$REMOTE_RELEASE/deploy/scripts/release_switch.sh' rollback"
}

say "creating a fresh pre-deploy backup"
ssh "$SSH_HOST" 'sudo systemctl start cortex-backup.service'

say "staging clean release $RELEASE_ID"
ssh "$SSH_HOST" "find '$REMOTE_STAGE' -depth -delete 2>/dev/null || true; install -d '$REMOTE_STAGE'"
rsync -az --delete \
  --exclude 'node_modules' \
  --exclude 'dist' \
  --exclude 'apps/web/public/bundle' \
  --exclude '.vite' \
  --exclude '*.tsbuildinfo' \
  --exclude 'services/api/cortex.db*' \
  --exclude 'services/api/.jwt_secret' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'test_*.py' \
  --exclude 'conftest.py' \
  --exclude '*.test.ts' \
  --exclude '*.test.tsx' \
  "$WEB_LOCAL/" "$SSH_HOST:$REMOTE_STAGE/"
printf '%s %s\n' "$HEAD_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | ssh "$SSH_HOST" "cat > '$REMOTE_STAGE/RELEASE'"

say "building and validating the staged release"
ssh "$SSH_HOST" "sudo bash -s -- '$RELEASE_ID' '$CORTEX_USER'" <<'REMOTE'
set -euo pipefail
RELEASE_ID="$1"
CORTEX_USER="$2"
STAGED="/tmp/cortex-release-$RELEASE_ID"
RELEASE="/opt/cortex/releases/$RELEASE_ID"

test -f "$STAGED/services/api/app.py"
test -f "$STAGED/apps/web/package.json"
install -d -o "$CORTEX_USER" -g "$CORTEX_USER" /opt/cortex/releases
test ! -e "$RELEASE"
mv "$STAGED" "$RELEASE"
chown -R "$CORTEX_USER:$CORTEX_USER" "$RELEASE"

REQ="$RELEASE/services/api/requirements.lock"
test -f "$REQ" || REQ="$RELEASE/services/api/requirements.txt"
sudo -u "$CORTEX_USER" /opt/cortex/.venv/bin/pip install -r "$REQ" --quiet
sudo -u "$CORTEX_USER" bash -c "cd '$RELEASE' && npm ci --no-audit --no-fund"
sudo -u "$CORTEX_USER" bash -c "cd '$RELEASE' && npm run build"
sudo -u "$CORTEX_USER" /opt/cortex/.venv/bin/python -m compileall -q \
  "$RELEASE/services/api"

test -f "$RELEASE/apps/web/dist/index.html"
test -f "$RELEASE/RELEASE"
test -f "$RELEASE/apps/web/dist/norms/historical-calibration-k7-provisional-v1.json"
test -f "$RELEASE/apps/web/dist/norms/historical-calibration-k7-provisional-v1.bin"
printf '%s  %s\n' \
  "c0ca140310b2410d12d61733478c0b8d4999e25e9f77c204a2caa318a7ee4776" \
  "$RELEASE/apps/web/dist/norms/historical-calibration-k7-provisional-v1.bin" \
  | sha256sum -c -
"$RELEASE/deploy/scripts/publish_assets.sh" "$RELEASE"

# Apply a governed Caddy change only when the live file still byte-matches the
# currently active release's template. Unexpected hand-maintained drift fails
# closed instead of being overwritten. Validate before switching application
# code and retain an exact config rollback until reload succeeds.
CURRENT=$(readlink -f /opt/cortex/cortex_web)
LIVE_CADDY=/etc/caddy/Caddyfile
CURRENT_CADDY="$CURRENT/deploy/Caddyfile.template"
NEXT_CADDY="$RELEASE/deploy/Caddyfile.template"
CADDY_BACKUP=""
if ! cmp -s "$NEXT_CADDY" "$LIVE_CADDY"; then
  if [ ! -f "$CURRENT_CADDY" ] || ! cmp -s "$CURRENT_CADDY" "$LIVE_CADDY"; then
    printf '%s\n' \
      "refusing to overwrite a live Caddyfile that drifted from the active release" >&2
    exit 2
  fi
  caddy validate --adapter caddyfile --config "$NEXT_CADDY"
  # The service runs as `caddy`; validation does not open file writers, so
  # establish the runtime sink with explicit ownership before reload. Preserve
  # any existing log contents, and refuse a symlink target.
  CADDY_ACCESS_LOG=/var/log/caddy/cortex-telemetry-access.json
  if [ -L "$CADDY_ACCESS_LOG" ]; then
    printf 'refusing symlinked Caddy access log: %s\n' "$CADDY_ACCESS_LOG" >&2
    exit 2
  fi
  install -d -o caddy -g caddy -m 0750 /var/log/caddy
  touch "$CADDY_ACCESS_LOG"
  chown caddy:caddy "$CADDY_ACCESS_LOG"
  chmod 0640 "$CADDY_ACCESS_LOG"
  CADDY_BACKUP=$(mktemp /tmp/cortex-Caddyfile.XXXXXX)
  cp "$LIVE_CADDY" "$CADDY_BACKUP"
fi

"$RELEASE/deploy/scripts/release_switch.sh" activate "$RELEASE"
if [ -n "$CADDY_BACKUP" ]; then
  install -m 0644 "$NEXT_CADDY" "$LIVE_CADDY"
fi
if ! systemctl reload caddy; then
  if [ -n "$CADDY_BACKUP" ]; then
    install -m 0644 "$CADDY_BACKUP" "$LIVE_CADDY"
    systemctl reload caddy || true
  fi
  "$RELEASE/deploy/scripts/release_switch.sh" rollback
  [ -z "$CADDY_BACKUP" ] || rm -f "$CADDY_BACKUP"
  exit 1
fi
[ -z "$CADDY_BACKUP" ] || rm -f "$CADDY_BACKUP"
REMOTE

DOMAIN=$(ssh "$SSH_HOST" 'sudo grep ^CORTEX_DOMAIN= /etc/cortex/cortex.env | cut -d= -f2-')

say "checking public database-backed health"
if ! readiness_gate "https://$DOMAIN"; then
  rollback_remote
  exit 1
fi

say "checking the live phone surface"
if (cd "$WEB_LOCAL/apps/web" && node scripts/phone_smoke.mjs "https://$DOMAIN"); then
  say "phone smoke OK"
else
  rollback_remote
  exit 1
fi

say "deployed $RELEASE_ID successfully"
