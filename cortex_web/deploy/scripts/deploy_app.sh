#!/usr/bin/env bash
# One-command deploy from your laptop. Run from the repo root on your laptop:
#
#   bash cortex_web/deploy/scripts/deploy_app.sh
#
# Two source modes — auto-detected from where you run it:
#   - **Laptop mode** (default; what you'll use). Rsyncs the working tree to
#     the deploy box, runs the on-box build step over SSH, restarts the
#     service. No GitHub credentials needed on the box. Bundle is preserved
#     (rsync's --delete is scoped so it doesn't touch public/bundle/).
#   - **On-box mode** (when run directly on the deploy box with sudo). Does
#     a `git pull` instead — meant for boxes you've wired up with a deploy
#     key / PAT. Not used today but kept for forward compatibility.
#
# Environment (laptop mode):
#   CORTEX_SSH      SSH host alias for the deploy box (default: cortex-prod)
#   CORTEX_USER     remote system user (default: cortex)
set -euo pipefail

SSH_HOST="${CORTEX_SSH:-cortex-prod}"
CORTEX_USER="${CORTEX_USER:-cortex}"

say() { printf "▸ %s\n" "$*"; }

# ── on-box mode? ──────────────────────────────────────────────────
if [ "${EUID:-$(id -u)}" -eq 0 ] && [ -d /opt/cortex/cortex_web ]; then
  BRANCH="${1:-main}"
  APP=/opt/cortex
  WEB="$APP/cortex_web"

  if [ -d "$APP/.git" ]; then
    say "on-box mode — git pull ($BRANCH)…"
    sudo -u "$CORTEX_USER" git -C "$APP" fetch --depth=1 origin "$BRANCH"
    sudo -u "$CORTEX_USER" git -C "$APP" reset --hard "origin/$BRANCH"
  else
    say "on-box mode but no /opt/cortex/.git — refusing to run."
    echo "  Use laptop mode (run from your workstation): " >&2
    echo "  bash cortex_web/deploy/scripts/deploy_app.sh" >&2
    exit 2
  fi

  say "refreshing Python deps…"
  sudo -u "$CORTEX_USER" "$APP/.venv/bin/pip" install -r "$WEB/server/requirements.txt" >/dev/null

  say "rebuilding the SPA…"
  sudo -u "$CORTEX_USER" bash -c "cd $WEB && npm ci && npm run build"

  say "restarting cortex.service…"
  systemctl reload caddy || true
  systemctl restart cortex.service
  sleep 2
  systemctl --no-pager status cortex.service | head -8

  DOMAIN=$(grep '^CORTEX_DOMAIN=' /etc/cortex/cortex.env | cut -d= -f2-)
  echo; say "health check:"; curl -sS "https://$DOMAIN/api/health" || true; echo
  exit 0
fi

# ── laptop mode ───────────────────────────────────────────────────
# Resolve where this script lives so we can rsync the right tree no matter
# where the user invokes it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_LOCAL="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -f "$WEB_LOCAL/package.json" ] || [ ! -d "$WEB_LOCAL/server" ]; then
  echo "could not locate cortex_web/ from $SCRIPT_DIR" >&2; exit 1
fi

say "rsync $WEB_LOCAL → $SSH_HOST:/tmp/cortex_web/"
# --delete is INSIDE the rsync, scoped to the tree we send (no node_modules /
# dist / bundle / venv / dev DBs). Crucially we keep public/bundle on the
# *remote* by not propagating its absence, so the in-place rsync below won't
# touch it either.
rsync -avz --delete \
  --exclude 'node_modules' \
  --exclude 'dist' \
  --exclude 'public/bundle' \
  --exclude '.vite' \
  --exclude '*.tsbuildinfo' \
  --exclude 'server/cortex.db*' \
  --exclude 'server/.jwt_secret' \
  --exclude '.venv' \
  --exclude 'demo_codes.csv' \
  --exclude '__pycache__' \
  "$WEB_LOCAL/" "$SSH_HOST:/tmp/cortex_web/" >/dev/null

say "stage on box (chown cortex), refresh deps, rebuild SPA, restart"
ssh "$SSH_HOST" "sudo bash -s <<'REMOTE'
set -euo pipefail
APP=/opt/cortex
WEB=\$APP/cortex_web

# Layer the rsynced tree onto /opt/cortex/cortex_web without touching
# public/bundle/ or node_modules/ (--exclude on the staging rsync), then
# force ownership back to the cortex user so npm + the venv can write.
sudo rsync -a --delete \
  --exclude public/bundle \
  --exclude node_modules \
  /tmp/cortex_web/ \$WEB/
chown -R ${CORTEX_USER}:${CORTEX_USER} \$APP

sudo -u ${CORTEX_USER} \$APP/.venv/bin/pip install -r \$WEB/server/requirements.txt --quiet
sudo -u ${CORTEX_USER} bash -c \"cd \$WEB && npm ci --no-audit --no-fund\" 2>&1 | tail -3
sudo -u ${CORTEX_USER} bash -c \"cd \$WEB && npm run build\" 2>&1 | tail -3

systemctl restart cortex.service
sleep 2
systemctl --no-pager is-active cortex.service
REMOTE"

say "health check from your laptop"
DOMAIN=$(ssh "$SSH_HOST" 'sudo grep ^CORTEX_DOMAIN /etc/cortex/cortex.env | cut -d= -f2-')
curl -sS "https://$DOMAIN/api/health"; echo
say "done."
