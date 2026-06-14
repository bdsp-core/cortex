#!/usr/bin/env bash
# Pull the latest source on the deploy box, reinstall Python deps, rebuild
# the SPA, and restart the systemd service. Run after provision.sh on every
# subsequent deploy.
#
#   sudo bash deploy/scripts/deploy_app.sh [<branch>]
#
# Idempotent. Safe to run while the service is up — restart is the last
# step, so the participant gets a brief 502 from Caddy (~1–2 s) at the swap.
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "must run as root (use sudo)" >&2; exit 1
fi
CORTEX_USER="${CORTEX_USER:-cortex}"
BRANCH="${1:-main}"
APP=/opt/cortex
WEB="$APP/cortex_web"

say() { printf "▸ %s\n" "$*"; }

say "pulling source ($BRANCH)…"
sudo -u "$CORTEX_USER" git -C "$APP" fetch --depth=1 origin "$BRANCH"
sudo -u "$CORTEX_USER" git -C "$APP" reset --hard "origin/$BRANCH"

say "refreshing Python deps…"
sudo -u "$CORTEX_USER" "$APP/.venv/bin/pip" install -r "$WEB/server/requirements.txt" >/dev/null

say "rebuilding the SPA…"
sudo -u "$CORTEX_USER" bash -c "cd $WEB && npm ci && npm run build"

say "reloading Caddy + restarting cortex.service…"
systemctl reload caddy || true
systemctl restart cortex.service

sleep 2
systemctl --no-pager status cortex.service | head -8
echo
say "done. health check:"
DOMAIN=$(grep '^CORTEX_DOMAIN=' /etc/cortex/cortex.env | cut -d= -f2-)
curl -sS "https://$DOMAIN/api/health" || true
echo
