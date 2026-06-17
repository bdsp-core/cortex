"""Dev entry point: `python -m api.run` (from cortex_web/services/) → uvicorn on :8000.

Honors env: CORTEX_PORT, CORTEX_HOST, CORTEX_ADMIN_TOKEN, CORTEX_BUNDLE_URL,
CORTEX_DB, CORTEX_JWT_SECRET. For production use a process manager + a real
ASGI server invocation (`uvicorn api.app:app --workers N`, cwd cortex_web/services/).
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
