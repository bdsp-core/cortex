#!/usr/bin/env bash
# CORTEX deep-health watchdog.
#
# Guards the 2026-08-18 failure class: the app process stays alive (systemd
# sees nothing wrong) while every DB-touching request wedges silently, so
# only an end-to-end deep-health probe notices. Three consecutive failed
# probes (~15 min at the 5-min timer cadence) restart cortex.service, with a
# cooldown so a fault that survives restarts cannot cause a restart loop.
#
# Install (as root):
#   cp cortex_health_watchdog.sh /opt/cortex/watchdog/
#   cp cortex-health-watchdog.service cortex-health-watchdog.timer /etc/systemd/system/
#   systemctl daemon-reload && systemctl enable --now cortex-health-watchdog.timer
set -u

STATE_DIR=/run/cortex-watchdog
FAILS_F="$STATE_DIR/consecutive_failures"
LAST_RESTART_F="$STATE_DIR/last_restart_epoch"
THRESHOLD=3
COOLDOWN_S=1800
# Probe uvicorn directly: the wedge lives in the app, and Caddy staying up
# must not mask it.
URL="http://127.0.0.1:8000/api/health?deep=1"

mkdir -p "$STATE_DIR"

if curl -fsS --max-time 15 "$URL" | grep -q '"ok": *true'; then
    echo 0 > "$FAILS_F"
    exit 0
fi

fails=$(( $(cat "$FAILS_F" 2>/dev/null || echo 0) + 1 ))
echo "$fails" > "$FAILS_F"
logger -t cortex-watchdog "deep health probe failed ($fails consecutive)"

if [ "$fails" -lt "$THRESHOLD" ]; then
    exit 0
fi

now=$(date +%s)
last=$(cat "$LAST_RESTART_F" 2>/dev/null || echo 0)
if [ $(( now - last )) -lt "$COOLDOWN_S" ]; then
    logger -t cortex-watchdog "still failing but inside restart cooldown; leaving service alone"
    exit 0
fi

echo "$now" > "$LAST_RESTART_F"
echo 0 > "$FAILS_F"
logger -t cortex-watchdog "RESTARTING cortex.service after $THRESHOLD consecutive deep-health failures"
systemctl restart cortex.service
