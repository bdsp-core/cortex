#!/usr/bin/env bash
# One-shot provisioner for a fresh Ubuntu 24.04 LTS Lightsail instance.
# Installs Caddy + Postgres + Python + rclone, creates the cortex system
# user, drops the systemd units in place, generates secrets, sets up the
# Postgres role + database, and enables the services.
#
#   sudo bash deploy/scripts/provision.sh <your-domain.example.org>
#
# Re-runnable — every step is idempotent so a retry after a transient
# failure is safe. Reads optional env overrides:
#   CORTEX_USER     system user (default: cortex)
#   CORTEX_REPO     git clone URL (default: bdsp-core/ilae-skill-certification-test-multi)
#   CORTEX_BRANCH   default: main
set -euo pipefail

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "must run as root (use sudo)" >&2; exit 1
fi
if [ $# -lt 1 ]; then
  echo "usage: $0 <domain>" >&2; exit 2
fi
DOMAIN="$1"
CORTEX_USER="${CORTEX_USER:-cortex}"
CORTEX_REPO="${CORTEX_REPO:-https://github.com/bdsp-core/ilae-skill-certification-test-multi.git}"
CORTEX_BRANCH="${CORTEX_BRANCH:-main}"

APP=/opt/cortex
DATA=/var/lib/cortex
LOG=/var/log/cortex
CONF=/etc/cortex

say() { printf "▸ %s\n" "$*"; }

# ── packages ───────────────────────────────────────────────────────
say "installing packages (caddy, postgres, python, nodejs, rclone)…"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y \
    ca-certificates curl gnupg debian-keyring debian-archive-keyring \
    apt-transport-https lsb-release \
    git build-essential pkg-config \
    python3 python3-venv python3-dev \
    postgresql postgresql-contrib \
    rclone sqlite3 jq

# Caddy from official repo (Ubuntu's caddy is several versions old)
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
      | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
      | tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  apt-get update -y
  apt-get install -y caddy
fi

# Node (for the SPA build). Use NodeSource LTS.
if ! command -v node >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi

# ── user + dirs ────────────────────────────────────────────────────
say "creating system user + directories…"
id -u "$CORTEX_USER" &>/dev/null \
  || useradd --system --create-home --shell /usr/sbin/nologin "$CORTEX_USER"
mkdir -p "$APP" "$DATA" "$LOG" "$CONF"
chown -R "$CORTEX_USER:$CORTEX_USER" "$APP" "$DATA" "$LOG"
chmod 750 "$CONF"

# ── source ─────────────────────────────────────────────────────────
# Three modes are supported, picked automatically:
#   (a) repo already cloned at /opt/cortex with .git  → fetch + reset
#   (b) only /opt/cortex/cortex_web is present (operator rsynced the tree
#       from their workstation; the simple first-deploy path when the repo
#       is private and the box has no GitHub creds) → keep what's there
#   (c) nothing present → git clone from CORTEX_REPO
WEB="$APP/cortex_web"
if [ -d "$APP/.git" ]; then
  say "fetching source (branch: $CORTEX_BRANCH)…"
  sudo -u "$CORTEX_USER" git -C "$APP" fetch --depth=1 origin "$CORTEX_BRANCH"
  sudo -u "$CORTEX_USER" git -C "$APP" reset --hard "origin/$CORTEX_BRANCH"
elif [ -d "$WEB" ]; then
  say "using pre-populated /opt/cortex/cortex_web (rsync deploy) — skipping git clone."
else
  say "cloning source (branch: $CORTEX_BRANCH)…"
  sudo -u "$CORTEX_USER" git clone --depth=1 --branch "$CORTEX_BRANCH" "$CORTEX_REPO" "$APP"
fi

# ── python venv ────────────────────────────────────────────────────
say "creating Python venv + installing requirements…"
sudo -u "$CORTEX_USER" python3 -m venv "$APP/.venv"
sudo -u "$CORTEX_USER" "$APP/.venv/bin/pip" install --upgrade pip wheel >/dev/null
sudo -u "$CORTEX_USER" "$APP/.venv/bin/pip" install -r "$WEB/services/api/requirements.txt"

# ── SPA build ──────────────────────────────────────────────────────
say "building the SPA (npm ci + vite build)…"
sudo -u "$CORTEX_USER" bash -c "cd $WEB && npm ci && npm run build"

# Bundle is gitignored — operator drops it in or rsyncs it manually after
# provision (see README §"After provisioning"). We don't fetch the bank
# from inside provision.sh (it's ~330 MB + needs auth).
mkdir -p "$WEB/public/bundle"
chown -R "$CORTEX_USER:$CORTEX_USER" "$WEB"

# ── Postgres role + database ───────────────────────────────────────
say "configuring Postgres role + database…"
PG_PASS=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='cortex'" | grep -q 1 \
  || sudo -u postgres psql -c "CREATE ROLE cortex LOGIN PASSWORD '$PG_PASS'"
sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='cortex'" | grep -q 1 \
  || sudo -u postgres createdb -O cortex cortex
# Keep the password we just minted in the env file (below); refresh it on
# every run so leaked old envs become inert.
sudo -u postgres psql -c "ALTER ROLE cortex WITH PASSWORD '$PG_PASS'"

# ── env file ───────────────────────────────────────────────────────
say "writing /etc/cortex/cortex.env (preserving any existing secrets)…"
JWT_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')"
ADMIN_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
# If an env file already exists, preserve the old JWT secret + admin token
# (rotating those invalidates live JWTs + breaks the admin tooling).
if [ -f "$CONF/cortex.env" ]; then
  OLD_JWT=$(grep '^CORTEX_JWT_SECRET=' "$CONF/cortex.env" | cut -d= -f2- || true)
  OLD_ADM=$(grep '^CORTEX_ADMIN_TOKEN=' "$CONF/cortex.env" | cut -d= -f2- || true)
  [ -n "$OLD_JWT" ] && JWT_SECRET="$OLD_JWT"
  [ -n "$OLD_ADM" ] && ADMIN_TOKEN="$OLD_ADM"
fi
install -m 640 -o "$CORTEX_USER" -g "$CORTEX_USER" /dev/null "$CONF/cortex.env"
cat >"$CONF/cortex.env" <<EOF
CORTEX_DOMAIN=$DOMAIN
CORTEX_CORS_ORIGINS=https://$DOMAIN
CORTEX_BUNDLE_URL=/bundle/v1.5-k7
CORTEX_SESSION_SAMPLE=500
CORTEX_TOKEN_TTL=21600
CORTEX_JWT_SECRET=$JWT_SECRET
CORTEX_ADMIN_TOKEN=$ADMIN_TOKEN
CORTEX_DB=postgresql://cortex:$PG_PASS@127.0.0.1:5432/cortex
CORTEX_HOST=127.0.0.1
CORTEX_PORT=8000
CORTEX_RCLONE_REMOTE=box
CORTEX_BOX_PATH=CORTEX/backups
CORTEX_BACKUP_RETENTION_DAYS=30
# ── FILL THESE IN (see deploy/.env.example for full docs) ──
# Without the email vars, signup email-verification falls back to the dev stub
# (logs codes to stderr, sends nothing) and NEW users can't complete signup.
# CORTEX_EMAIL_BACKEND=smtp
# CORTEX_SMTP_HOST=email-smtp.us-west-2.amazonaws.com
# CORTEX_SMTP_PORT=587
# CORTEX_SMTP_SECURITY=starttls
# CORTEX_SMTP_USER=
# CORTEX_SMTP_PASSWORD=
# CORTEX_EMAIL_FROM=CORTEX <no-reply@$DOMAIN>
# Without GOOGLE_CLIENT_ID the /api/auth/google endpoint 503s + the button hides.
# GOOGLE_CLIENT_ID=
# Where /report submissions are emailed (defaults to elikeldsen@icloud.com).
# CORTEX_REPORT_TO=
EOF

# ── Caddy ──────────────────────────────────────────────────────────
say "installing Caddyfile (auto-TLS; domains fixed to cortexeeg.org in the template)…"
cp "$WEB/deploy/Caddyfile.template" /etc/caddy/Caddyfile
systemctl enable --now caddy
systemctl reload caddy || systemctl restart caddy

# ── systemd units ──────────────────────────────────────────────────
say "installing systemd units…"
cp "$WEB/deploy/systemd/cortex.service" /etc/systemd/system/
cp "$WEB/deploy/systemd/cortex-backup.service" /etc/systemd/system/
cp "$WEB/deploy/systemd/cortex-backup.timer"   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now cortex.service
systemctl enable --now cortex-backup.timer

# ── status ─────────────────────────────────────────────────────────
sleep 2
say "provision complete. status:"
systemctl --no-pager status cortex.service | head -10
echo
say "next steps:"
cat <<NEXT
  1. Point DNS  $DOMAIN  →  this instance's static IP.
  2. Open Lightsail firewall: TCP 80 + 443 (HTTP-01 + HTTPS).
  3. Drop the EEG bundle into $WEB/apps/web/public/bundle/<version>/ on this box
     (rsync from your workstation; ~330 MB for v1.5-k7).
  4. Set up rclone for the Box backup:
        sudo -u $CORTEX_USER bash
        cd /home/$CORTEX_USER && rclone config        # add a remote named "box"
        # see deploy/scripts/rclone-box-setup.md for the JWT-app alternative.
  5. Mint participant credentials:
        cd /opt/cortex/cortex_web/services && sudo -u $CORTEX_USER \\
          /opt/cortex/.venv/bin/python -m api.admin \\
            --db "\$(grep ^CORTEX_DB /etc/cortex/cortex.env | cut -d= -f2-)" \\
            gen --count 50 --prefix cortex --out /tmp/codes.csv
  6. Fill in email (SMTP/SES) + GOOGLE_CLIENT_ID in $CONF/cortex.env, then
        systemctl restart cortex.service
     Until then signup email-verification uses the dev stub and Google login 503s.
  7. Sanity check:    curl -s "https://$DOMAIN/api/health?deep=1"   # expect "ok": true
NEXT
