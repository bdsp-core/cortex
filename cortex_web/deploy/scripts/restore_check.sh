#!/usr/bin/env bash
# CORTEX restore check — prove a Box backup actually restores. An untested
# backup is hope, not a backup: this rehearses the restore end-to-end into a
# SCRATCH database, sanity-checks the contents, then drops the scratch. It
# never touches the live database or the running service.
#
#   sudo /opt/cortex/cortex_web/deploy/scripts/restore_check.sh
#       → newest snapshot on Box
#   sudo .../restore_check.sh 'box:CORTEX/backups/2026/07/<stamp>'
#       → that snapshot
#   sudo .../restore_check.sh /path/to/db.sql.gz          (or db.sqlite.gz)
#       → a local dump file (no rclone needed)
#
# Env (loaded from /etc/cortex/cortex.env when run interactively):
#   CORTEX_RCLONE_REMOTE   rclone remote name (default: box)
#   CORTEX_BOX_PATH        folder inside Box (default: CORTEX/backups)
#
# PASS requires: the dump restores cleanly, the participants/sessions/
# training_trials tables are queryable, and at least one participant row
# exists. The snapshot's HEARTBEAT.txt session count is printed alongside the
# restored count as a freshness cross-check (informational: the heartbeat
# counts the admin export's rows, which can lag the raw table).
set -euo pipefail

REMOTE="${CORTEX_RCLONE_REMOTE:-box}"
BOX_PATH="${CORTEX_BOX_PATH:-CORTEX/backups}"
SCRATCH_DB=cortex_restore_check
ENV_FILE=/etc/cortex/cortex.env

say() { printf "[%s] %s\n" "$(date -u +%H:%M:%SZ)" "$*"; }
fail() { echo "✗ RESTORE CHECK FAIL: $*" >&2; exit 1; }

# systemd loads the env for the backup unit; interactive runs load it here.
if [ -z "${CORTEX_RCLONE_REMOTE:-}" ] && [ -r "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
  REMOTE="${CORTEX_RCLONE_REMOTE:-box}"
  BOX_PATH="${CORTEX_BOX_PATH:-CORTEX/backups}"
fi

WORK="$(mktemp -d -t cortex-restore-check-XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

# ── 1. locate + fetch the dump ─────────────────────────────────────
SRC="${1:-}"
if [ -n "$SRC" ] && [ -f "$SRC" ]; then
  say "using local dump $SRC"
  cp "$SRC" "$WORK/"
else
  if [ -z "$SRC" ]; then
    say "finding newest snapshot under $REMOTE:$BOX_PATH"
    # snapshot dirs are YYYY/MM/<stamp>/ — lexicographic max = newest
    NEWEST=$(rclone lsf --dirs-only -R "$REMOTE:$BOX_PATH" \
             | grep -E '^[0-9]{4}/[0-9]{2}/[^/]+/$' | sort | tail -1)
    [ -n "$NEWEST" ] || fail "no snapshots found under $REMOTE:$BOX_PATH"
    SRC="$REMOTE:$BOX_PATH/${NEWEST%/}"
  fi
  say "fetching $SRC"
  rclone copy --include 'db.*' --include 'HEARTBEAT.txt' "$SRC" "$WORK/"
fi
# (not `ls a b | head -1`: under pipefail one missing name fails the pipeline
# even when the other matched)
DUMP=$(find "$WORK" -maxdepth 1 \( -name db.sql.gz -o -name db.sqlite.gz \) | head -1)
[ -n "$DUMP" ] || fail "snapshot has no db.sql.gz / db.sqlite.gz"
gzip -t "$DUMP" || fail "dump is not a valid gzip: $DUMP"
say "dump: $(basename "$DUMP") ($(stat -c %s "$DUMP") bytes)"

# ── 2. restore into a scratch DB ───────────────────────────────────
q() { :; }   # per-flavor scalar query, defined below
if [[ "$DUMP" == *db.sql.gz ]]; then
  command -v psql >/dev/null || fail "psql not installed"
  say "restoring into scratch Postgres DB '$SCRATCH_DB'"
  sudo -u postgres dropdb --if-exists "$SCRATCH_DB"
  sudo -u postgres createdb "$SCRATCH_DB"
  # ON_ERROR_STOP so a truncated/corrupt dump fails loudly, not silently.
  gunzip -c "$DUMP" | sudo -u postgres psql -q -v ON_ERROR_STOP=1 "$SCRATCH_DB" \
    || fail "psql restore errored"
  q() { sudo -u postgres psql -tAc "$1" "$SCRATCH_DB"; }
  cleanup_scratch() { sudo -u postgres dropdb --if-exists "$SCRATCH_DB"; }
else
  command -v sqlite3 >/dev/null || fail "sqlite3 not installed"
  say "restoring into scratch SQLite file"
  gunzip -c "$DUMP" > "$WORK/restored.db"
  [ "$(sqlite3 "$WORK/restored.db" 'PRAGMA integrity_check;')" = "ok" ] \
    || fail "sqlite integrity_check failed"
  q() { sqlite3 "$WORK/restored.db" "$1"; }
  cleanup_scratch() { :; }
fi

# ── 3. sanity: the tables a restore exists to bring back ───────────
N_PART=$(q "SELECT count(*) FROM participants")   || fail "participants table unreadable"
N_SESS=$(q "SELECT count(*) FROM sessions")       || fail "sessions table unreadable"
N_TRIALS=$(q "SELECT count(*) FROM training_trials") || fail "training_trials table unreadable"
cleanup_scratch
[ "$N_PART" -ge 1 ] || fail "restored DB has zero participants"

if [ -f "$WORK/HEARTBEAT.txt" ]; then
  say "snapshot heartbeat: $(cat "$WORK/HEARTBEAT.txt")"
fi
say "✓ RESTORE CHECK PASS participants=$N_PART sessions=$N_SESS training_trials=$N_TRIALS"
