"""Canonical UTC timestamp helpers.

Every timestamp this service stores or compares uses ONE fixed ISO-Z format.
That is load-bearing, not cosmetic: timestamps live in TEXT columns and
several queries select on `>=` over them, so a lexicographic compare is only
equal to a chronological compare while the format never varies (fixed width,
zero-padded, always UTC, always the trailing Z).

The format string previously appeared in seven independent strftime calls
across db, helpers, engine_trainer, and two routers. One definition here
means a change cannot land in six places and miss the seventh.
"""
from __future__ import annotations

import calendar
import time

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def iso_at(epoch: float) -> str:
    """The canonical ISO-Z rendering of an epoch."""
    return time.strftime(ISO_FMT, time.gmtime(epoch))


def utc_now() -> str:
    """Now, canonically formatted."""
    return time.strftime(ISO_FMT, time.gmtime())


def iso_in(seconds: float) -> str:
    """`seconds` from now; negative reaches into the past (window cutoffs)."""
    return iso_at(time.time() + seconds)


def parse_iso(value: str) -> float:
    """Epoch seconds for a canonical ISO-Z string.

    Raises ValueError on anything else. Callers treat that as "no usable
    timestamp" rather than guessing — a malformed legacy value must never
    silently become a real instant.
    """
    return calendar.timegm(time.strptime(value, ISO_FMT))
