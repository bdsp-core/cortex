#!/usr/bin/env bash
# Atomically switch /opt/cortex/cortex_web between pre-built release trees.
# The first invocation converts the legacy physical directory into a retained
# release; subsequent switches are atomic symlink replacements.
set -euo pipefail

ACTION="${1:-}"
RELEASE_ARG="${2:-}"
APP_ROOT="${CORTEX_APP_ROOT:-/opt/cortex}"
LIVE="$APP_ROOT/cortex_web"
RELEASES="$APP_ROOT/releases"
STATE="$APP_ROOT/release-state"
SERVICE="${CORTEX_SERVICE_NAME:-cortex.service}"
HEALTH_URL="${CORTEX_HEALTH_URL:-http://127.0.0.1:8000/api/health?deep=1}"

fail() { printf 'release switch: %s\n' "$*" >&2; exit 2; }

test_mode() {
  [ "${CORTEX_RELEASE_TEST_MODE:-0}" = "1" ] && [ "$APP_ROOT" != "/opt/cortex" ]
}

validate_release() {
  local candidate release_root
  candidate=$(realpath -e "$1") || fail "release does not exist: $1"
  release_root=$(realpath -e "$RELEASES") || fail "release root does not exist"
  case "$candidate/" in
    "$release_root/"*) ;;
    *) fail "release is outside $RELEASES" ;;
  esac
  [ -f "$candidate/services/api/app.py" ] || fail "release has no API"
  [ -f "$candidate/apps/web/dist/index.html" ] || fail "release has no built SPA"
  [ -f "$candidate/RELEASE" ] || fail "release has no provenance stamp"
  printf '%s' "$candidate"
}

write_state() {
  local current="$1" previous="$2" tmp
  install -d "$STATE"
  tmp="$STATE/.state.$$"
  printf '%s\n' "$current" > "$tmp.current"
  printf '%s\n' "$previous" > "$tmp.previous"
  mv -f "$tmp.current" "$STATE/current"
  mv -f "$tmp.previous" "$STATE/previous"
}

atomic_link() {
  local target="$1" next="$APP_ROOT/.cortex_web.next.$$"
  ln -s "$target" "$next"
  mv -Tf "$next" "$LIVE"
}

restart_and_probe() {
  local attempt body=""
  if test_mode; then
    return 0
  fi
  systemctl restart "$SERVICE"
  for attempt in $(seq 1 20); do
    body=$(curl -fsS "$HEALTH_URL" 2>/dev/null || true)
    if printf '%s' "$body" | grep -q '"ok": *true'; then
      return 0
    fi
    sleep 1
  done
  printf 'release switch: health failed; last body: %s\n' "${body:-<none>}" >&2
  return 1
}

activate() {
  local candidate previous legacy stamp
  candidate=$(validate_release "$RELEASE_ARG")

  if [ -L "$LIVE" ]; then
    previous=$(readlink -f "$LIVE")
  elif [ -d "$LIVE" ]; then
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    legacy="$RELEASES/legacy-$stamp"
    test ! -e "$legacy" || fail "legacy target already exists: $legacy"
    mv "$LIVE" "$legacy"
    previous="$legacy"
  else
    fail "live application path is missing: $LIVE"
  fi

  atomic_link "$candidate"
  write_state "$candidate" "$previous"
  if ! restart_and_probe; then
    atomic_link "$previous"
    restart_and_probe || true
    write_state "$previous" "$candidate"
    fail "activation failed and previous release was restored"
  fi
  printf 'activated %s; previous %s\n' "$candidate" "$previous"
}

rollback() {
  local current previous
  [ -L "$LIVE" ] || fail "live application is not release-managed"
  [ -f "$STATE/previous" ] || fail "no previous release is recorded"
  current=$(readlink -f "$LIVE")
  previous=$(validate_release "$(cat "$STATE/previous")")
  atomic_link "$previous"
  if ! restart_and_probe; then
    atomic_link "$current"
    restart_and_probe || true
    fail "rollback target failed health; current release was restored"
  fi
  write_state "$previous" "$current"
  printf 'rolled back to %s\n' "$previous"
}

install -d "$RELEASES"
case "$ACTION" in
  activate) [ -n "$RELEASE_ARG" ] || fail "activate requires a release path"; activate ;;
  rollback) rollback ;;
  *) fail "usage: $0 activate RELEASE_DIR | rollback" ;;
esac
