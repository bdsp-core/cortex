"""Client-side error telemetry. The SPA's global error hook
(apps/web/src/telemetry.ts: window error + unhandledrejection + engine-worker
failures) posts here, so UI crashes in the field surface in the journal like
every other incident signal instead of waiting for a user to file a report:

    journalctl -u cortex | grep cortex.clienterr

Log-only by design (no DB table): the volume is tiny, the journal is already
the ops query surface, and nothing here is worth a schema. The endpoint is
public (crashes can happen pre-auth), so payloads are length-capped by the
model and rate-limited per IP; the client additionally dedupes and caps
reports per page load."""
from __future__ import annotations

import sys

from fastapi import APIRouter, HTTPException, Request

from ..deps import client_ip
from ..models import ClientErrorIn

router = APIRouter(prefix="/api")


@router.post("/client-error")
def client_error(body: ClientErrorIn, req: Request):
    limiter = req.app.state.limiter
    if not limiter.hit("client_error", client_ip(req)):
        raise HTTPException(429, "too many reports; try again later")
    # One journal line per error: newlines collapsed, frames capped.
    stack = " | ".join(body.stack.splitlines()[:8])
    print(f"[cortex.clienterr] surface={body.surface or '?'} url={body.url!r} "
          f"msg={body.message!r} ua={body.ua!r} stack={stack!r}",
          file=sys.stderr, flush=True)
    return {"ok": True}
