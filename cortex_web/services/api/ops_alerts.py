"""Ops alerting — turn error signals into email so they get NOTICED.

Before this module, backend 500s and SPA crash reports only landed in the
systemd journal: visible if someone greps, invisible otherwise. Two signals
now also raise an email to the operator:

  * "server-error"  — any unhandled exception in an API handler (the
                      app-level Exception handler in app.py)
  * "client-error"  — SPA crash telemetry (POST /api/client-error)

Destination: CORTEX_OPS_ALERT_TO (default: config.REPORT_TO, the same inbox
as user support reports). Set it to an empty string to disable alerting.

Storm control: a per-kind cooldown (CORTEX_OPS_ALERT_COOLDOWN_S, default 6 h)
collapses repeats into one email per window; the next email after a quiet
window carries the count suppressed meanwhile. The journal keeps the full
stream either way — this is the "wake a human" channel, not the log.

Sends run on a daemon thread so the request path never blocks on SMTP/SES,
and a send failure can never take down the request that triggered it.
"""
from __future__ import annotations

import sys
import threading
import time

from . import mailer


class OpsAlerter:
    def __init__(self, to_addr: str, cooldown_s: int = 6 * 3600) -> None:
        self.to_addr = (to_addr or "").strip()
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._last_sent: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}
        # Last dispatched send thread — tests join() it for determinism.
        self.last_send_thread: threading.Thread | None = None

    def notify(self, kind: str, summary: str) -> threading.Thread | None:
        """Record one error event; email unless `kind` is inside its cooldown
        window. Returns the send thread when an email was dispatched."""
        if not self.to_addr:
            return None
        now = time.time()
        with self._lock:
            last = self._last_sent.get(kind)
            if last is not None and now - last < self.cooldown_s:
                self._suppressed[kind] = self._suppressed.get(kind, 0) + 1
                return None
            suppressed = self._suppressed.pop(kind, 0)
            self._last_sent[kind] = now
        subject = f"[cortex ops] {kind}"
        lines = [summary]
        if suppressed:
            lines.append(f"(+{suppressed} earlier {kind} event(s) suppressed "
                         f"during the previous cooldown window)")
        lines.append("")
        lines.append("Full stream: journalctl -u cortex | "
                     f"grep cortex.{'clienterr' if kind == 'client-error' else 'servererr'}")
        t = threading.Thread(target=self._send, args=(subject, "\n".join(lines)),
                             daemon=True)
        t.start()
        self.last_send_thread = t
        return t

    def _send(self, subject: str, body: str) -> None:
        try:
            mailer.send_email(self.to_addr, subject, body)
        except Exception as e:   # alerting must never raise into a request
            print(f"[cortex.opsalert] send failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
