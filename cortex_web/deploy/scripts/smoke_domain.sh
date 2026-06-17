#!/usr/bin/env bash
# Phase-A domain smoke test for the cortexeeg.org migration.
#
#   bash deploy/scripts/smoke_domain.sh
#
# Uses curl --resolve so it tests the box + Caddy + TLS cert directly, even
# before public DNS has propagated. Also checks authoritative DNS. Exits 0 only
# if every check passes. Mid-migration, some checks are expected to fail until
# their step is done — the labels say which step each belongs to.
set -uo pipefail
IP=44.233.29.150
AUTH_NS=ns-339.awsdns-42.com          # one of the cortexeeg.org authoritative NS
APP=app.cortexeeg.org
APEX=cortexeeg.org
WWW=www.cortexeeg.org
NIP=cortex-44-233-29-150.nip.io

fail=0
pass(){ printf '  \033[32m✓\033[0m %s\n' "$*"; }
bad(){ printf '  \033[31m✗\033[0m %s\n' "$*"; fail=1; }

dns_auth(){ # host — authoritative lookup, bypasses negative caching
  local h=$1 got
  got=$(dig +short @"$AUTH_NS" A "$h" 2>/dev/null | tail -1)
  [ "$got" = "$IP" ] && pass "DNS(auth) $h → $IP" || bad "DNS(auth) $h → '${got:-<none>}' (want $IP)"
}
health(){ # host
  local h=$1 code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 --resolve "$h:443:$IP" "https://$h/api/health" 2>/dev/null)
  [ "$code" = 200 ] && pass "$h /api/health → 200 (TLS ok)" || bad "$h /api/health → ${code:-conn-fail}"
}
spa(){ # host
  local h=$1
  curl -s --max-time 15 --resolve "$h:443:$IP" "https://$h/" 2>/dev/null | grep -q 'assets/index-' \
    && pass "$h serves SPA bundle" || bad "$h SPA bundle missing"
}
redirect(){ # host wantprefix
  local h=$1 want=$2 code loc
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 --resolve "$h:443:$IP" "https://$h/" 2>/dev/null)
  loc=$(curl -s -o /dev/null -w '%{redirect_url}' --max-time 15 --resolve "$h:443:$IP" "https://$h/" 2>/dev/null)
  case "$code:$loc" in
    30[12]:"$want"*) pass "$h → $code $loc" ;;
    *) bad "$h → ${code} '${loc:-<none>}' (want 301→$want*)" ;;
  esac
}

echo "── DNS (authoritative) ──";        dns_auth "$APP"; dns_auth "$APEX"; dns_auth "$WWW"
echo "── app.cortexeeg.org (serves app) ──"; health "$APP"; spa "$APP"
echo "── apex/www (redirect → app) ──";   redirect "$APEX" "https://$APP"; redirect "$WWW" "https://$APP"
echo "── nip.io fallback (still serves) ──"; health "$NIP"; spa "$NIP"

echo
[ "$fail" = 0 ] && echo "SMOKE: ALL PASS" || echo "SMOKE: some checks failed (expected mid-migration; see labels)"
exit $fail
