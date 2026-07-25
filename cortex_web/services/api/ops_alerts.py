"""Ops alerting — turn error signals into email so they get NOTICED.

Backend failures and trusted browser failures raise an immediate email:

  * "server-error"  — any unhandled exception in an API handler (the
                      app-level Exception handler in app.py)
  * "client-error-authenticated" — an application crash accompanied by a
                      valid bearer token. Anonymous crashes go only to a
                      daily digest, and the injected-DOM signature is always
                      journal-only.

Destination: CORTEX_OPS_ALERT_TO (default: config.REPORT_TO, the same inbox
as user support reports). Set it to an empty string to disable alerting.

Storm control: a durable per-kind cooldown (CORTEX_OPS_ALERT_COOLDOWN_S,
default 6 h) collapses repeats into one email per window; the next email after
a quiet window carries the count suppressed meanwhile. The state lives in the
application database, so restarting the service does not reopen the gate.

Sends run on a daemon thread so the request path never blocks on SMTP/SES,
and a send failure can never take down the request that triggered it.
"""
from __future__ import annotations

import sys
import threading
import time

from . import mailer


class OpsAlerter:
    def __init__(self, to_addr: str, db,
                 cooldown_s: int = 6 * 3600) -> None:
        self.to_addr = (to_addr or "").strip()
        self.db = db
        self.cooldown_s = cooldown_s
        self._lock = threading.Lock()
        # Emergency fallback if the persistence backend itself is the failed
        # component. Normal operation always uses the durable DB state.
        self._fallback_last_sent: dict[str, float] = {}
        self._fallback_suppressed: dict[str, int] = {}
        # Last dispatched send thread — tests join() it for determinism.
        self.last_send_thread: threading.Thread | None = None

    def notify(self, kind: str, summary: str,
               now_s: float | None = None) -> threading.Thread | None:
        """Record one error event; email unless `kind` is inside its cooldown
        window. Returns the send thread when an email was dispatched."""
        if not self.to_addr:
            return None
        now = time.time() if now_s is None else now_s
        with self._lock:
            try:
                suppressed = self.db.claim_ops_alert(
                    kind, now, self.cooldown_s)
            except Exception as e:
                # A database outage is itself a common source of server 500s.
                # Alerting must still work and must not raise recursively from
                # the application's exception handler.
                print(f"[cortex.opsalert] state failed: "
                      f"{type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                last = self._fallback_last_sent.get(kind)
                if last is not None and now - last < self.cooldown_s:
                    self._fallback_suppressed[kind] = (
                        self._fallback_suppressed.get(kind, 0) + 1)
                    suppressed = None
                else:
                    suppressed = self._fallback_suppressed.pop(kind, 0)
                    self._fallback_last_sent[kind] = now
            if suppressed is None:
                return None
        subject = f"[cortex ops] {kind}"
        lines = [summary]
        if suppressed:
            lines.append(f"(+{suppressed} earlier {kind} event(s) suppressed "
                         f"during the previous cooldown window)")
        lines.append("")
        lines.append("Full stream: journalctl -u cortex | "
                     f"grep cortex.{'clienterr' if kind.startswith('client-error') else 'servererr'}")
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
