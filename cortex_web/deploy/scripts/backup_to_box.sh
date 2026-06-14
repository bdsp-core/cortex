#!/usr/bin/env bash
# CORTEX backup → Box. Runs every 6 hours via the systemd timer.
# Snapshots the DB + admin exports, uploads to Box via rclone, prunes old
# snapshots beyond the retention window.
#
#   /opt/cortex/deploy/scripts/backup_to_box.sh
#
# Env (from /etc/cortex/cortex.env, loaded by systemd):
#   CORTEX_DB                  sqlite path OR postgresql:// URL
#   CORTEX_RCLONE_REMOTE       rclone remote name (default: box)
#   CORTEX_BOX_PATH            folder inside Box (default: CORTEX/backups)
#   CORTEX_BACKUP_RETENTION_DAYS   keep snapshots newer than this (default: 30)
#   CORTEX_ADMIN_TOKEN         used by the admin exporter
set -euo pipefail

REMOTE="${CORTEX_RCLONE_REMOTE:-box}"
BOX_PATH="${CORTEX_BOX_PATH:-CORTEX/backups}"
RETAIN_DAYS="${CORTEX_BACKUP_RETENTION_DAYS:-30}"
APP=/opt/cortex
WEB="$APP/cortex_web"
PY="$APP/.venv/bin/python"
LOG=/var/log/cortex
mkdir -p "$LOG"

STAMP=$(date -u +%Y-%m-%dT%H%M%SZ)
YYMM=$(date -u +%Y/%m)
WORK="$(mktemp -d -t cortex-backup-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

say() { printf "[%s] %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }

# ── 1. DB snapshot ─────────────────────────────────────────────────
if [[ "$CORTEX_DB" == postgres://* || "$CORTEX_DB" == postgresql://* ]]; then
  say "pg_dump → $WORK/db.sql.gz"
  PGPASSFILE="$(mktemp)"; trap 'rm -f "$PGPASSFILE"' RETURN
  # rfc-style postgres://user:pw@host:port/db → parse for PGPASSFILE
  PG_URL="$CORTEX_DB"
  USER=$(printf '%s' "$PG_URL" | sed -E 's|^[a-z]+://([^:@]+).*|\1|')
  PASS=$(printf '%s' "$PG_URL" | sed -E 's|^[a-z]+://[^:@]+:([^@]+)@.*|\1|')
  HOST=$(printf '%s' "$PG_URL" | sed -E 's|^[a-z]+://[^@]+@([^:/]+).*|\1|')
  PORT=$(printf '%s' "$PG_URL" | sed -nE 's|^[a-z]+://[^@]+@[^:]+:([0-9]+)/.*|\1|p'); PORT=${PORT:-5432}
  DB=$(printf '%s' "$PG_URL"   | sed -E 's|.*/([^?]+).*|\1|')
  echo "$HOST:$PORT:$DB:$USER:$PASS" > "$PGPASSFILE"
  chmod 600 "$PGPASSFILE"
  PGPASSFILE="$PGPASSFILE" pg_dump --no-owner --no-privileges \
      -h "$HOST" -p "$PORT" -U "$USER" "$DB" | gzip -9 > "$WORK/db.sql.gz"
else
  say "sqlite snapshot → $WORK/db.sqlite.gz"
  # .backup gives a consistent online copy even mid-write.
  sqlite3 "$CORTEX_DB" ".backup $WORK/db.sqlite"
  gzip -9 "$WORK/db.sqlite"
fi

# ── 2. mineable exports (CSV sessions + per-result JSON) ──────────
say "admin exports"
( cd "$WEB" && CORTEX_DB="$CORTEX_DB" \
    "$PY" -m server.admin export-sessions --out "$WORK/sessions.csv" >/dev/null )
( cd "$WEB" && CORTEX_DB="$CORTEX_DB" \
    "$PY" -m server.admin export-results --out "$WORK/results" >/dev/null )
( cd "$WORK" && tar czf results.tgz results && rm -rf results )

# ── 3. heartbeat (lets you spot a silent failure on Box) ──────────
N_SESS=$(grep -c "^[^,]*," "$WORK/sessions.csv" || true)
N_SESS=$(( N_SESS > 0 ? N_SESS - 1 : 0 ))   # minus header
echo "stamp=$STAMP sessions=$N_SESS host=$(hostname)" > "$WORK/HEARTBEAT.txt"

# ── 4. upload ─────────────────────────────────────────────────────
DEST="$REMOTE:$BOX_PATH/$YYMM/$STAMP"
say "uploading → $DEST"
rclone --quiet copy "$WORK" "$DEST"

# ── 5. prune old snapshots (Box-side) ─────────────────────────────
say "pruning snapshots older than $RETAIN_DAYS days"
rclone --quiet delete --min-age "${RETAIN_DAYS}d" "$REMOTE:$BOX_PATH"
# rclone delete leaves empty dirs; rmdirs cleans them up.
rclone --quiet rmdirs --leave-root "$REMOTE:$BOX_PATH"

say "done"
