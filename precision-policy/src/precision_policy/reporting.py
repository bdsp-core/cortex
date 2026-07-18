from __future__ import annotations

from .policy import DETERMINED, ESTIMATE_COMPLETE

ABOVE_CUT = "ABOVE_CUT"
BELOW_CUT = "BELOW_CUT"
INDETERMINATE_AT_CUT = "INDETERMINATE_AT_CUT"


def classify_interval_against_cut(interval, cut: float) -> str:
    """Pure downstream interval classification; never called by stopping."""

    low, high = (float(interval[0]), float(interval[1]))
    cut = float(cut)
    if low > cut:
        return ABOVE_CUT
    if high < cut:
        return BELOW_CUT
    return INDETERMINATE_AT_CUT


def classify_determined_interval_against_cut(
    status: str, interval, cut: float
) -> str:
    """Classify only a determined domain, enforcing the reporting boundary."""

    if status not in (ESTIMATE_COMPLETE, DETERMINED):
        raise ValueError("certification cuts may be applied only to DETERMINED domains")
    return classify_interval_against_cut(interval, cut)
