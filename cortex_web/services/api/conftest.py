"""Shared API test fixtures.

The register-time email deliverability check (helpers.email_domain_deliverable)
does a live MX/DNS lookup — REAL in production and on any box with dnspython
installed. Test accounts sign up with @example.test (NXDOMAIN in real DNS),
so EVERY test module that builds participants must bypass it, not just
test_server.py. Hosting the autouse bypass here lets modules like
test_engine_trainer.py (which imports _make_participant) inherit it; without
it those tests pass only where dnspython is absent (dev), then fail on a
server where the check actually resolves. A same-named module fixture still
overrides this where a module needs the real function (test_server.py keeps
its own for the dedicated deliverability tests).
"""
import pytest

from . import helpers


@pytest.fixture(autouse=True)
def _dns_check_open(monkeypatch):
    monkeypatch.setattr(helpers, "email_domain_deliverable",
                        lambda domain: True)
