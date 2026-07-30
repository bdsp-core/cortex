"""Client-side error telemetry. The SPA's global error hook
(apps/web/src/telemetry.ts: window error + unhandledrejection + engine-worker
failures) posts here, so UI crashes in the field surface in the journal like
every other incident signal instead of waiting for a user to file a report:

    journalctl -u cortex | grep cortex.clienterr

The endpoint is public because crashes can happen before sign-in. Public input
can therefore never directly trigger email: anonymous reports are journaled
and aggregated into a daily digest. A valid bearer token upgrades an ordinary
application crash to immediate alerting; the known injected-DOM signature
remains journal-only even then.

Three storm controls apply: the existing per-IP request limit plus durable
global and per-normalized-fingerprint budgets. Random React expando suffixes
are removed before fingerprinting, so ``__reactFiber$abc`` and
``__reactFiber$xyz`` share one bucket."""
from __future__ import annotations

import sys

from fastapi import APIRouter, HTTPException, Request

from ..client_error_policy import ClientErrorPolicy
from ..deps import client_ip
from ..models import ClientErrorIn

router = APIRouter(prefix="/api")


@router.post("/client-error")
def client_error(body: ClientErrorIn, req: Request):
    limiter = req.app.state.limiter
    ip = client_ip(req)
    if not limiter.hit("client_error", ip):
        raise HTTPException(429, "too many reports; try again later")
    policy: ClientErrorPolicy = req.app.state.client_error_policy
    try:
        observation = policy.observe(body, req, ip)
        accepted = policy.admit_and_record(observation)
    except Exception as exc:
        # Telemetry is best-effort. In particular, an unauthenticated report
        # during a DB outage must not become a server 500 and thereby reach the
        # immediate server-error email channel.
        print(f"[cortex.clienterr] dropped=policy-unavailable "
              f"err={type(exc).__name__}",
              file=sys.stderr, flush=True)
        return {"ok": True, "accepted": False}
    if not accepted:
        # A successful response prevents telemetry clients/proxies from
        # retrying. The durable aggregate already counted this suppression.
        return {"ok": True, "accepted": False}
    print(
          f"[cortex.clienterr] auth={int(observation.authenticated)} "
          f"class={observation.classification} fp={observation.fingerprint} "
          f"request_fp={observation.request_fingerprint} "
          f"origin={observation.origin_class} "
          f"fetch_site={observation.sec_fetch_site} "
          f"surface={observation.surface} url={observation.url!r} "
          f"msg={observation.message!r} ua={observation.ua_family!r} "
          f"stack={observation.stack!r}",
          file=sys.stderr, flush=True)
    if observation.authenticated and observation.classification != "injected-dom":
        req.app.state.alerts.notify(
            "client-error-authenticated",
            f"fp={observation.fingerprint} surface={observation.surface} "
            f"url={observation.url!r} msg={observation.message!r}")
    return {"ok": True, "accepted": True}
