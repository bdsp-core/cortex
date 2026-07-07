#!/usr/bin/env bash
# One-command deploy from your laptop. Run from anywhere in the repo:
#
#   bash cortex_web/deploy/scripts/deploy_app.sh
#
# Monorepo layout (Phase B): cortex_web/{apps/web (SPA), services/api (FastAPI)}.
# Caddy file-serves apps/web/dist + the EEG bundle; uvicorn is a pure API.
#
# Two source modes — auto-detected from where you run it:
#   - Laptop mode (default): rsync the working tree to the box, run the on-box
#     build over SSH, restart the service. The EEG bundle on the box
#     (apps/web/public/bundle) is preserved (rsync --delete is scoped to skip it).
#   - On-box mode (run on the box as root): git pull instead. Not used today.
#
# Env (laptop mode): CORTEX_SSH (default cortex-prod), CORTEX_USER (default cortex).
#
# NOTE: the FIRST cutover from the old flat layout needs a one-time on-box
# bundle move (public/bundle -> apps/web/public/bundle) before this runs — see
# deploy/README.md / the Phase B cutover steps.
set -euo pipefail

SSH_HOST="${CORTEX_SSH:-cortex-prod}"
CORTEX_USER="${CORTEX_USER:-cortex}"

say() { printf "▸ %s\n" "$*"; }

# Poll GET /api/health?deep=1 until it reports ok (DB round-trip included), or
# fail the deploy. The DEEP probe is the one that would have caught the
# 2026-06-24 "process up but DB unreachable" outage — a bare `curl … || true`
# reports success on a boot that 500s. $1 = base origin (https://domain or
# http://127.0.0.1:8000). Returns non-zero (aborting `set -e`) on failure.
readiness_gate() {
  local base="$1" i body
  for i in $(seq 1 20); do
    body=$(curl -fsS "$base/api/health?deep=1" 2>/dev/null || true)
    if printf '%s' "$body" | grep -q '"ok": *true'; then
      say "readiness OK — $body"
      return 0
    fi
    sleep 1
  done
  echo "✗ readiness gate FAILED: /api/health?deep=1 never reported ok." >&2
  echo "  Last body: ${body:-<none>}" >&2
  echo "  The new process is up but not healthy (DB? bundle?). Investigate;" >&2
  echo "  roll back with: git revert <sha> && bash cortex_web/deploy/scripts/deploy_app.sh" >&2
  return 1
}

# Warn (do NOT auto-overwrite) if the repo Caddyfile.template has drifted from
# the live /etc/caddy/Caddyfile. The live file is hand-maintained on the box
# (CSP/body-caps were applied manually), so silently clobbering it is unsafe —
# but silent drift is how the template's security headers rot. Surfacing it is
# the right middle ground. Runs over SSH in laptop mode; local in on-box mode.
caddy_drift_warn() {
  local runner="$1"   # "" for local, or an ssh prefix
  local tmpl live
  tmpl=$($runner cat "$2" 2>/dev/null || true)
  live=$($runner sudo cat /etc/caddy/Caddyfile 2>/dev/null || true)
  if [ -n "$tmpl" ] && [ -n "$live" ] && [ "$tmpl" != "$live" ]; then
    say "NOTE: deploy/Caddyfile.template differs from the live /etc/caddy/Caddyfile."
    echo "  deploy_app.sh does NOT touch Caddy config (it's hand-maintained on the" >&2
    echo "  box). If the template carries intended header/CSP changes, apply them" >&2
    echo "  manually and 'systemctl reload caddy'." >&2
  fi
}

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
    echo "  Use laptop mode (run from your workstation)." >&2
    exit 2
  fi

  say "stamping RELEASE…"
  git -C "$APP" rev-parse --short=12 HEAD | sudo -u "$CORTEX_USER" tee "$WEB/RELEASE" >/dev/null

  say "refreshing Python deps (lock)…"
  # requirements.lock pins the exact prod closure (reproducible redeploys);
  # requirements.txt keeps the floor constraints (CI + regenerating the lock).
  REQ="$WEB/services/api/requirements.lock"
  [ -f "$REQ" ] || REQ="$WEB/services/api/requirements.txt"
  sudo -u "$CORTEX_USER" "$APP/.venv/bin/pip" install -r "$REQ" >/dev/null

  say "building the SPA (workspace)…"
  sudo -u "$CORTEX_USER" bash -c "cd $WEB && npm ci && npm run build"

  caddy_drift_warn "" "$WEB/deploy/Caddyfile.template"

  say "restarting cortex.service…"
  systemctl reload caddy || true
  systemctl restart cortex.service
  sleep 2
  systemctl --no-pager status cortex.service | head -8

  # Readiness gate against the loopback API (before Caddy/TLS) so a bad boot
  # aborts the deploy loudly instead of reporting a green checkmark.
  say "readiness gate (deep health)…"
  readiness_gate "http://127.0.0.1:8000"
  DOMAIN=$(grep '^CORTEX_DOMAIN=' /etc/cortex/cortex.env | cut -d= -f2-)
  echo; say "public health:"; curl -sS "https://$DOMAIN/api/health" || true; echo
  exit 0
fi

# ── laptop mode ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_LOCAL="$(cd "$SCRIPT_DIR/../.." && pwd)"     # cortex_web/

if [ ! -d "$WEB_LOCAL/apps/web" ] || [ ! -d "$WEB_LOCAL/services/api" ]; then
  echo "could not locate the cortex_web monorepo (apps/web + services/api) from $SCRIPT_DIR" >&2
  exit 1
fi

# Record what is being shipped (surfaced by /api/health as "release"). A
# dirty tree still deploys — this is the lab deploy flow — but it is warned
# about and stamped as +dirty so "what SHA is live?" is always answerable.
GIT_SHA=$(git -C "$WEB_LOCAL" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
DIRTY=""
if [ -n "$(git -C "$WEB_LOCAL" status --porcelain -- . 2>/dev/null)" ]; then
  DIRTY="+dirty"
  say "WARNING: cortex_web/ working tree is DIRTY — uncommitted changes will deploy (stamped ${GIT_SHA}${DIRTY})"
fi

say "rsync $WEB_LOCAL → $SSH_HOST:/tmp/cortex_web/"
# --delete is scoped to the sent tree; the EEG bundle + node_modules + build
# output + dev artifacts are excluded so the on-box copies are untouched.
rsync -avz --delete \
  --exclude 'node_modules' \
  --exclude 'dist' \
  --exclude 'apps/web/public/bundle' \
  --exclude '.vite' \
  --exclude '*.tsbuildinfo' \
  --exclude 'services/api/cortex.db*' \
  --exclude 'services/api/.jwt_secret' \
  --exclude '.venv' \
  --exclude 'demo_codes.csv' \
  --exclude '__pycache__' \
  "$WEB_LOCAL/" "$SSH_HOST:/tmp/cortex_web/" >/dev/null

# Stamp the release into the staged tree; the on-box rsync layers it into
# /opt/cortex/cortex_web/RELEASE, where /api/health reads it at boot.
printf '%s%s %s\n' "$GIT_SHA" "$DIRTY" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | ssh "$SSH_HOST" 'cat > /tmp/cortex_web/RELEASE'

say "stage on box (chown cortex), refresh deps, rebuild SPA, restart"
ssh "$SSH_HOST" "sudo bash -s <<'REMOTE'
set -euo pipefail
APP=/opt/cortex
WEB=\$APP/cortex_web

# Layer the rsynced tree onto /opt/cortex/cortex_web without touching the EEG
# bundle (apps/web/public/bundle) or node_modules, then restore ownership.
sudo rsync -a --delete \
  --exclude apps/web/public/bundle \
  --exclude node_modules \
  /tmp/cortex_web/ \$WEB/
chown -R ${CORTEX_USER}:${CORTEX_USER} \$APP

# Install from the lock (exact prod pins) when present; fall back to the
# floor-constraint requirements.txt on a box that predates the lock.
REQ=\$WEB/services/api/requirements.lock
[ -f \$REQ ] || REQ=\$WEB/services/api/requirements.txt
sudo -u ${CORTEX_USER} \$APP/.venv/bin/pip install -r \$REQ --quiet
sudo -u ${CORTEX_USER} bash -c \"cd \$WEB && npm ci --no-audit --no-fund\" 2>&1 | tail -3
sudo -u ${CORTEX_USER} bash -c \"cd \$WEB && npm run build\" 2>&1 | tail -3

systemctl reload caddy || true
systemctl restart cortex.service
sleep 2
systemctl --no-pager is-active cortex.service
REMOTE"

caddy_drift_warn "ssh $SSH_HOST" "/opt/cortex/cortex_web/deploy/Caddyfile.template"

say "readiness gate from your laptop (deep health)"
DOMAIN=$(ssh "$SSH_HOST" 'sudo grep ^CORTEX_DOMAIN /etc/cortex/cortex.env | cut -d= -f2-')
# Fails the deploy (non-zero exit) if the new process is up but unhealthy.
readiness_gate "https://$DOMAIN"
say "done."
