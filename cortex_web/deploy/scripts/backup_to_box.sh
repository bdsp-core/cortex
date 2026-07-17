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

# Single-instance lock: the timer fires every 6h; a slow upload must not overlap
# the next tick (double pg_dump + racing prunes). Re-exec under flock once.
LOCK=/var/lock/cortex-backup.lock
if [ "${_CORTEX_BACKUP_LOCKED:-}" != "1" ]; then
  exec env _CORTEX_BACKUP_LOCKED=1 flock -n "$LOCK" "$0" "$@" || {
    echo "another cortex backup is already running — skipping this tick" >&2
    exit 0
  }
fi

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
  # Password file lives in $WORK (0700 mktemp dir, removed by the EXIT trap on
  # every path incl. errors) and is deleted explicitly right after pg_dump,
  # BEFORE section 4 uploads $WORK — so the credential never reaches Box.
  # (Earlier a bare mktemp guarded by `trap ... RETURN` leaked the password to
  # /tmp: RETURN traps don't fire at top-level script scope.)
  PGPASSFILE="$WORK/.pgpass"
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
  rm -f "$PGPASSFILE"   # drop the credential before $WORK is uploaded (section 4)
else
  say "sqlite snapshot → $WORK/db.sqlite.gz"
  # .backup gives a consistent online copy even mid-write.
  sqlite3 "$CORTEX_DB" ".backup $WORK/db.sqlite"
  gzip -9 "$WORK/db.sqlite"
fi

# ── 2. mineable exports (CSV sessions + per-result JSON) ──────────
say "admin exports"
( cd "$WEB/services" && CORTEX_DB="$CORTEX_DB" \
    "$PY" -m api.admin export-sessions --out "$WORK/sessions.csv" >/dev/null )
( cd "$WEB/services" && CORTEX_DB="$CORTEX_DB" \
    "$PY" -m api.admin export-results --out "$WORK/results" >/dev/null )
( cd "$WORK" && tar czf results.tgz results && rm -rf results )

# ── 3. heartbeat (lets you spot a silent failure on Box) ──────────
N_SESS=$(grep -c "^[^,]*," "$WORK/sessions.csv" || true)
N_SESS=$(( N_SESS > 0 ? N_SESS - 1 : 0 ))   # minus header
echo "stamp=$STAMP sessions=$N_SESS host=$(hostname)" > "$WORK/HEARTBEAT.txt"

# ── 4. upload ─────────────────────────────────────────────────────
# Size sanity: a truncated-but-exit-0 dump must NOT then trigger the prune and
# evict older good snapshots. A real dump is comfortably over this floor; abort
# (leaving snapshots intact) if it isn't.
# (find, NOT `ls a b | head -1`: under `set -euo pipefail` ls exits 2 on the
# flavor's missing filename and silently killed every backup here — the
# 2026-07 ten-day backup gap found by the restore drill.)
DUMP=$(find "$WORK" -maxdepth 1 \( -name db.sql.gz -o -name db.sqlite.gz \) | head -1)
[ -n "$DUMP" ] || { echo "✗ no DB dump produced in $WORK" >&2; exit 1; }
DUMP_BYTES=$(stat -c %s "$DUMP" 2>/dev/null || echo 0)
if [ "$DUMP_BYTES" -lt 1024 ]; then
  echo "✗ DB dump is only ${DUMP_BYTES} bytes (< 1 KiB) — aborting BEFORE prune so" >&2
  echo "  a bad dump can't evict good snapshots. Investigate pg_dump/sqlite." >&2
  exit 1
fi

DEST="$REMOTE:$BOX_PATH/$YYMM/$STAMP"
say "uploading → $DEST (dump ${DUMP_BYTES} bytes)"
rclone --quiet copy "$WORK" "$DEST"

# ── 5. prune old snapshots (Box-side) ─────────────────────────────
say "pruning snapshots older than $RETAIN_DAYS days"
rclone --quiet delete --min-age "${RETAIN_DAYS}d" "$REMOTE:$BOX_PATH"
# rclone delete leaves empty dirs; rmdirs cleans them up.
rclone --quiet rmdirs --leave-root "$REMOTE:$BOX_PATH"

say "done"
