"""Dev entry point: `python -m api.run` (from cortex_web/services/) → uvicorn on :8000.

Honors env: CORTEX_PORT, CORTEX_HOST, CORTEX_ADMIN_TOKEN, CORTEX_BUNDLE_URL,
CORTEX_DB, CORTEX_JWT_SECRET. For production use a process manager running
exactly ONE worker (`uvicorn api.app:app`, cwd cortex_web/services/ — see
deploy/systemd/cortex.service). Single-process is a hard invariant, NOT a
tuning knob: `--workers N` would split the in-memory rate limiter across
workers, load N copies of the ~150 MB SessionBank, and race the idempotent
boot migrations.
"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("CORTEX_HOST", "127.0.0.1")
    port = int(os.environ.get("CORTEX_PORT", "8000"))
    uvicorn.run("api.app:app", host=host, port=port,
                reload=bool(os.environ.get("CORTEX_RELOAD")))


if __name__ == "__main__":
    main()
