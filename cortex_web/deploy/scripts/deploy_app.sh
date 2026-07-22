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

caddy_drift_warn() {
  local template_path="$1" template live
  template=$(ssh "$SSH_HOST" "cat '$template_path'" 2>/dev/null || true)
  live=$(ssh "$SSH_HOST" 'sudo cat /etc/caddy/Caddyfile' 2>/dev/null || true)
  if [ -n "$template" ] && [ -n "$live" ] && [ "$template" != "$live" ]; then
    say "NOTE: the release Caddy template differs from the live configuration."
    printf '%s\n' "  The deploy does not overwrite hand-maintained Caddy settings." >&2
  fi
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
"$RELEASE/deploy/scripts/release_switch.sh" activate "$RELEASE"
if ! systemctl reload caddy; then
  "$RELEASE/deploy/scripts/release_switch.sh" rollback
  exit 1
fi
REMOTE

caddy_drift_warn "$REMOTE_RELEASE/deploy/Caddyfile.template"
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
