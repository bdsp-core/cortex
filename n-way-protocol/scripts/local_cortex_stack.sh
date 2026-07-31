#!/usr/bin/env bash
# Local cortex_web stack for testing the draw-latent (atoms17) n-way engine
# through the EXACT production UI, fully isolated from prod.
#
#   scripts/local_cortex_stack.sh          # (re)start the stack
#   scripts/local_cortex_stack.sh stop     # stop the stack it started
#
# One uvicorn serves SPA + /bundle + /api (the apps/web/scripts/run_local.sh
# idiom, CORTEX_SERVE_STATIC=1) on http://127.0.0.1:8788 with a DEDICATED
# SQLite DB under n-way-protocol/local_server/data/ — the checked-out dev
# cortex.db and the prod server are never touched. The research draw-latent
# stamp requires BOTH CORTEX_NWAY_RESEARCH_UNQUALIFIED=1 AND no
# cortex_web/RELEASE file (services/api/nway_profile.py fails closed
# otherwise), and is scoped to ONE account by pinning the precision_v1
# policy rollout to that account's email.
#
# Idempotent: kills only its own previous instance (pidfile + cmdline match
# on this stack's unique port — never a broad pkill), refuses a port squatted
# by anything else, and rebuilds apps/web/dist when sources are newer.
# See n-way-protocol/local_server/CORTEX_STACK.md for the full runbook.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"            # n-way-protocol/
REPO="$(cd "$HERE/.." && pwd)"                       # repo root
WEB="$REPO/cortex_web/apps/web"
SERVICES="$REPO/cortex_web/services"
PY="$REPO/.venv/bin/python"
[ -x "$PY" ] || { echo "missing $PY (repo venv)"; exit 1; }

PORT="${CORTEX_STACK_PORT:-8788}"                    # 8734 is another rig — do not use it
DATA_DIR="$HERE/local_server/data"
DB="$DATA_DIR/cortex_ui_test.db"
PIDFILE="$DATA_DIR/.cortex_stack_${PORT}.pid"
LOG="$DATA_DIR/cortex_stack_${PORT}.log"
ACCOUNT_EMAIL="elikeldsen+nway-local@gmail.com"
ACCOUNT_PASSWORD="nway-local-test-1"
PROFILE_ID="precision_nway_f1_engine_frame_atoms17_draw_latent_rd_v1"

mkdir -p "$DATA_DIR"

# ── stop any previous instance THIS script started (marker = pidfile + the
#    stack's own port in the recorded process's cmdline; nothing else is
#    ever killed) ──────────────────────────────────────────────────────────
stop_own() {
  if [ -f "$PIDFILE" ]; then
    local pid; pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] \
       && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "uvicorn api.app:app .*--port $PORT"; then
      echo "▸ stopping previous stack instance (pid $pid)"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
}

stop_own
if [ "${1:-}" = "stop" ]; then echo "▸ stack stopped"; exit 0; fi

# ── leaked/foreign-server guard (the trap that made earlier smokes lie):
#    after stopping our own instance the port must be silent ───────────────
if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
  echo "port $PORT is already serving and it is NOT this stack's process." >&2
  echo "find and stop it yourself:  ss -lptn 'sport = :$PORT'" >&2
  exit 1
fi

# ── non-production guard: a RELEASE stamp would silently disable the
#    draw-latent escape (sessions would stamp the production mixture) ──────
if [ -f "$REPO/cortex_web/RELEASE" ]; then
  echo "cortex_web/RELEASE exists — this checkout looks production-deployed;" >&2
  echo "the draw-latent escape refuses. Remove the file only if you are sure" >&2
  echo "this is a dev checkout." >&2
  exit 1
fi

# ── fresh dist (stale-dist trap): rebuild when any source is newer than the
#    built index.html, then prove the bundle carries the atoms17 profile ───
need_build=0
if [ ! -f "$WEB/dist/index.html" ]; then
  need_build=1
elif [ -n "$(find "$WEB/src" "$WEB/engine" "$WEB/index.html" "$WEB/package.json" \
              -newer "$WEB/dist/index.html" -print -quit 2>/dev/null)" ]; then
  need_build=1
fi
if [ "$need_build" = 1 ]; then
  echo "▸ sources newer than dist — rebuilding SPA (tsc + vite; takes minutes)"
  ( cd "$WEB" && npm run build )
else
  echo "▸ dist is up to date (built $(date -r "$WEB/dist/index.html" '+%F %T'))"
fi
if ! grep -rq "$PROFILE_ID" "$WEB/dist/assets"; then
  echo "built dist does NOT contain the draw-latent profile id ($PROFILE_ID);" >&2
  echo "wrong branch or stale build — refusing to serve it." >&2
  exit 1
fi
echo "▸ dist contains the atoms17 draw-latent profile id ✓"

# ── start the API (single process serving SPA + bundle + API) ─────────────
echo "▸ starting API on http://127.0.0.1:$PORT (db: $DB)"
( cd "$SERVICES" && \
  CORTEX_DB="$DB" \
  CORTEX_SERVE_STATIC=1 \
  CORTEX_NWAY_RESEARCH_UNQUALIFIED=1 \
  CORTEX_PRECISION_POLICY_ROLLOUT=email_allowlist \
  CORTEX_PRECISION_POLICY_EMAILS="$ACCOUNT_EMAIL" \
  CORTEX_EMAIL_BACKEND=dev \
  CORTEX_EMAIL_EXPOSE_CODE=1 \
  CORTEX_ADMIN_TOKEN="${CORTEX_ADMIN_TOKEN:-local-admin}" \
  CORTEX_DIGEST_DISABLED=1 \
  CORTEX_CLIENT_ERROR_DIGEST_DISABLED=1 \
  CORTEX_OPS_ALERT_TO= \
  exec "$PY" -m uvicorn api.app:app --host 127.0.0.1 --port "$PORT" \
       --log-level warning ) >"$LOG" 2>&1 &
UV=$!
echo "$UV" > "$PIDFILE"

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1 && break
  if ! kill -0 "$UV" 2>/dev/null; then
    echo "API exited before becoming ready:" >&2; tail -8 "$LOG" >&2
    rm -f "$PIDFILE"; exit 1
  fi
  sleep 0.5
done
health="$(curl -sf "http://127.0.0.1:$PORT/api/health")"
case "$health" in *'"release":null'*) ;; *)
  echo "health reports a RELEASE stamp — the draw-latent escape is OFF: $health" >&2
  stop_own; exit 1;;
esac
echo "▸ API healthy ($health)"

# ── local test account (idempotent: only registers when absent; the register
#    endpoint is limited to 5/hour/IP and restarting the API resets it) ─────
have_account="$("$PY" - "$DB" "$ACCOUNT_EMAIL" <<'EOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
row = db.execute("SELECT email_verified_utc FROM participants WHERE email=?",
                 (sys.argv[2],)).fetchone()
print("verified" if row and row[0] else ("unverified" if row else "absent"))
EOF
)"
if [ "$have_account" != "verified" ]; then
  echo "▸ creating + verifying the test account ($ACCOUNT_EMAIL)"
  reg="$(curl -sf -X POST "http://127.0.0.1:$PORT/api/register" \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$ACCOUNT_EMAIL\",\"password\":\"$ACCOUNT_PASSWORD\",\"displayName\":\"NWay Local Tester\",\"expertise\":\"attending\"}")"
  code="$(printf '%s' "$reg" | "$PY" -c 'import sys,json;print(json.load(sys.stdin).get("devCode",""))')"
  [ -n "$code" ] || { echo "no devCode in register response: $reg" >&2; stop_own; exit 1; }
  curl -sf -X POST "http://127.0.0.1:$PORT/api/verify/confirm" \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$ACCOUNT_EMAIL\",\"code\":\"$code\"}" >/dev/null
else
  echo "▸ test account already present + verified"
fi

cat <<DONE

────────────────────────────────────────────────────────────────────
  CORTEX local draw-latent stack is UP

  URL       http://127.0.0.1:$PORT
  account   $ACCOUNT_EMAIL  /  $ACCOUNT_PASSWORD
  database  $DB
  log       $LOG
  engine    $PROFILE_ID
            (stamped on every NEW certification session of this account)

  stop      $(basename "$0") stop
────────────────────────────────────────────────────────────────────
DONE
