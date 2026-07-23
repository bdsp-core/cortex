#!/usr/bin/env bash
# Publish Vite's content-hashed assets into the stable, append-only asset
# store served by Caddy. Long-lived tabs retain immutable index/chunk URLs
# across a release activation; deleting those files turns a lazy worker or
# dynamic import into an HTML-fallback load failure.
set -euo pipefail

APP_ROOT="${CORTEX_APP_ROOT:-/opt/cortex}"
RELEASES="$APP_ROOT/releases"
STORE="$APP_ROOT/assets"
CANDIDATE="${1:-}"

fail() { printf 'asset publish: %s\n' "$*" >&2; exit 2; }

publish_tree() {
  local tree="$1" source name destination
  source="$tree/apps/web/dist/assets"
  [ -d "$source" ] || return 0
  while IFS= read -r -d '' file; do
    name="${file##*/}"
    # Historical releases may contain public source maps. They are not
    # runtime assets and must not become reachable through the stable store.
    case "$name" in
      *.map|*.map.br|*.map.gz) continue ;;
    esac
    destination="$STORE/$name"
    if [ -e "$destination" ]; then
      cmp -s "$file" "$destination" \
        || fail "content-hash collision for $name"
    else
      install -m 0644 "$file" "$destination"
    fi
  done < <(find "$source" -maxdepth 1 -type f -print0)
}

install -d -m 0755 "$STORE"

# Backfill retained releases on the first run so tabs opened before this
# mechanism was deployed recover without a manual refresh.
if [ -d "$RELEASES" ]; then
  for release in "$RELEASES"/*; do
    [ -d "$release" ] && publish_tree "$release"
  done
fi

if [ -n "$CANDIDATE" ]; then
  [ -d "$CANDIDATE" ] || fail "candidate does not exist: $CANDIDATE"
  publish_tree "$CANDIDATE"
fi
