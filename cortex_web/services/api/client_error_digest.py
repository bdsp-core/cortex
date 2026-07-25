"""Once-daily summary for unauthenticated SPA crash telemetry.

Anonymous POSTs never call the immediate OpsAlerter. They are aggregated by
normalized fingerprint and delivered after the UTC day closes, with only the
coarsened request metadata produced by client_error_policy.py.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

from . import config, mailer, timeutil

log = logging.getLogger("cortex.client_error_digest")

TICK_S = 900
INITIAL_DELAY_S = 120
DEFAULT_SEND_HOUR_UTC = 15
RETENTION_DAYS = 30


def _day_at(now_s: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now_s))


def _build_body(day: str, rows: list[dict]) -> str:
    total = sum(int(row.get("count") or 0) for row in rows)
    suppressed = sum(int(row.get("suppressed") or 0) for row in rows)
    lines = [
        f"Unauthenticated CORTEX client-error digest for {day} UTC",
        f"Events: {total}; fingerprints: {len(rows)}; rate-suppressed: {suppressed}",
        "",
        "No raw IP, Origin, user agent, token, account, or query string is retained.",
        "",
    ]
    for row in rows[:25]:
        lines.extend([
            f"fingerprint={row['fingerprint']} "
            f"class={row['classification']} count={row['count']} "
            f"suppressed={row['suppressed']}",
            f"route={row.get('sample_url') or '/'} "
            f"surface={row.get('sample_surface') or 'unknown'} "
            f"ua={row.get('ua_family') or 'Other Other'} "
            f"origin={row.get('origin_class') or 'missing'} "
            f"fetch_site={row.get('sec_fetch_site') or 'missing'} "
            f"request_fp={row.get('request_fingerprint') or 'none'}",
            f"sample={row.get('sample_message') or ''}",
            "",
        ])
    if len(rows) > 25:
        lines.append(f"{len(rows) - 25} additional fingerprint(s) omitted.")
    lines.extend([
        "Full stream:",
        "journalctl -u cortex | grep cortex.clienterr",
    ])
    return "\n".join(lines)


def run_client_error_digest(db, now_s: float | None = None,
                            to_addr: str | None = None) -> int:
    """Send the preceding UTC day's anonymous aggregate at most once."""
    if os.environ.get("CORTEX_CLIENT_ERROR_DIGEST_DISABLED"):
        return 0
    if now_s is None:
        now_s = time.time()
    send_hour = int(os.environ.get(
        "CORTEX_CLIENT_ERROR_DIGEST_HOUR_UTC",
        str(DEFAULT_SEND_HOUR_UTC)))
    if time.gmtime(now_s).tm_hour < send_hour:
        return 0
    # Cleanup is independent of whether yesterday produced anonymous mail;
    # authenticated aggregates and expired limiter buckets must remain bounded
    # during quiet periods too.
    db.prune_client_error_telemetry(
        _day_at(now_s - RETENTION_DAYS * 86400),
        timeutil.iso_at(now_s - 2 * 86400))
    destination = (
        os.environ.get("CORTEX_OPS_ALERT_TO", config.REPORT_TO)
        if to_addr is None else to_addr
    ).strip()
    if not destination:
        return 0
    completed_day = _day_at(now_s - 86400)
    rows = db.client_error_digest_rows(completed_day)
    if not rows or not db.claim_client_error_digest_day(completed_day):
        return 0
    subject = f"[cortex ops] daily client-error digest: {completed_day}"
    mailer.send_email(destination, subject, _build_body(completed_day, rows))
    return 1


async def scheduler_loop(db, tick_s: float = TICK_S,
                         initial_delay_s: float = INITIAL_DELAY_S) -> None:
    await asyncio.sleep(initial_delay_s)
    while True:
        try:
            sent = run_client_error_digest(db)
            if sent:
                log.info("[cortex.clientdigest] sent daily anonymous digest")
        except Exception:
            log.exception("[cortex.clientdigest] pass failed")
        await asyncio.sleep(tick_s)
