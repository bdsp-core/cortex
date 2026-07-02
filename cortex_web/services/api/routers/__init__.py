"""Endpoint routers, one module per surface. Wired together in api.app.

Shared state (Database, RateLimiter, SessionBank getter, per-app config)
lives on `request.app.state`; stateless header deps live in api.deps; pure
helpers in api.helpers / api.dashboard_logic. Helper calls go through the
module attribute (`helpers.x(...)`) so test monkeypatching reaches them.
"""
from . import account, admin, auth, dashboard, report, testing, videos  # noqa: F401

ALL_ROUTERS = [
    auth.router,
    account.router,
    testing.router,
    dashboard.router,
    report.router,
    videos.router,
    admin.router,
]
