"""Exposure Ledger — the repeat-prevention data layer (D-INT-7, D-INT-8).

One canonical authority for "what has this participant been shown, when, in
what phase, for which domain," and the single function both the exam draw and
the training draw call: *which candidate items are eligible to serve now?*
See docs/TRAINER_INTEGRATION_PLAN.md §2.5.

Design (G0 lands the interface + policy + an in-memory backend; the SQLite
backend is G2 and the Postgres backend is G4 — both implement `ExposureLedger`
and reuse this exact `NoRepeatPolicy`):

  * Record (append-only): one `ExposureRecord` per served question, written by
    BOTH the exam and the trainer. `domain_ordinal` is the 1-indexed running
    count of within-domain items served to the participant at write time, so
    "intervening questions since last seen" is a single subtraction.

  * Eligibility (D-INT-7): for a candidate last seen at ordinal `o_x`, time
    `t_x`, with the participant's current per-domain ordinal `O` and clock
    `now`,
        eligible(x) = (O - o_x >= C)            # count-primary
                      AND (now - t_x >= T)      # same-session / 24 h floor
    A never-seen candidate is always eligible. Scope is UNIFIED test + train
    and per-participant (an item seen in a test is excluded from training and
    vice versa — what makes the D-INT-4 fresh retest an independent
    re-measurement).

  * Exhaustion (D-INT-8): when no candidate is eligible, fall back to serving
    the least-recently-seen item (largest intervening count, then oldest
    timestamp, then lowest seg_id). The loop never stalls.

Determinism: `eligible()` is a pure function of ledger state + policy + `now`,
with a stable seg_id tie-break, so G2/G3 parity tests reproduce bitwise.
"""
from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureRecord:
    participant_id: str
    seg_id: int
    domain: str          # production task code (trainer.domains.CODES)
    phase: str           # "test" | "train"
    served_at: float     # epoch seconds
    domain_ordinal: int  # 1-indexed within-domain count for (participant, domain)


@dataclass(frozen=True)
class NoRepeatPolicy:
    """Count-primary no-repeat rule with a same-session time floor (D-INT-7)
    and a least-recently-seen exhaustion fallback (D-INT-8).

    count_threshold : C — intervening within-domain items before re-eligibility.
    time_floor_s    : T — minimum wall-clock gap (default 24 h) to kill the
                          marathon-session hole; the ONLY load-bearing use of
                          time.
    exhaustion      : "least_recently_seen" (default) or "block".
    """
    count_threshold: int = 600
    time_floor_s: float = 24 * 3600.0
    exhaustion: str = "least_recently_seen"

    def __post_init__(self):
        if self.exhaustion not in ("least_recently_seen", "block"):
            raise ValueError(f"unknown exhaustion mode {self.exhaustion!r}")


@dataclass(frozen=True)
class EligibilityResult:
    """`eligible` = candidates that satisfy the no-repeat predicate, in a
    deterministic order (ascending seg_id). `used_fallback` is True when the
    eligible set was empty and the D-INT-8 exhaustion ranking was returned
    instead."""
    eligible: list[int]
    used_fallback: bool


class ExposureLedger(ABC):
    """Abstract exposure store. Backends: in-memory (here / G2 unit tests),
    SQLite (G2 closed-loop), Postgres (G4 web app)."""

    @abstractmethod
    def record(self, participant_id: str, seg_id: int, domain: str,
               phase: str, served_at: float) -> ExposureRecord:
        """Append one served-question exposure; assigns and returns the
        1-indexed within-domain ordinal."""

    @abstractmethod
    def domain_ordinal(self, participant_id: str, domain: str) -> int:
        """Current within-domain served count for (participant, domain)."""

    @abstractmethod
    def last_seen(self, participant_id: str, domain: str, seg_id: int):
        """(ordinal, served_at) of the most recent exposure of `seg_id` for
        this participant+domain, or None if never seen."""

    # ── concrete eligibility logic (shared by all backends) ────────────────
    def _staleness(self, participant_id: str, domain: str, seg_id: int, now: float):
        """(intervening_count, seconds_since) for a seen item; None if unseen."""
        ls = self.last_seen(participant_id, domain, seg_id)
        if ls is None:
            return None
        o_x, t_x = ls
        return self.domain_ordinal(participant_id, domain) - o_x, now - t_x

    def eligible(self, participant_id: str, domain: str,
                 candidate_seg_ids, policy: NoRepeatPolicy,
                 now: float) -> EligibilityResult:
        """Return the eligible candidates (deterministic order), or the
        exhaustion fallback ranking when none qualify. Pure function of ledger
        state + policy + now."""
        eligible: list[int] = []
        seen: list[tuple] = []  # (neg_count, served_since_older_first, seg) for fallback
        for seg in candidate_seg_ids:
            st = self._staleness(participant_id, domain, int(seg), now)
            if st is None:
                eligible.append(int(seg))          # never seen → always eligible
                continue
            intervening, secs_since = st
            if intervening >= policy.count_threshold and secs_since >= policy.time_floor_s:
                eligible.append(int(seg))
            else:
                # rank key for fallback: most-stale first → largest intervening,
                # then oldest (largest secs_since), then lowest seg_id.
                seen.append((-intervening, -secs_since, int(seg)))

        if eligible:
            return EligibilityResult(sorted(eligible), used_fallback=False)

        if policy.exhaustion == "block":
            return EligibilityResult([], used_fallback=True)

        # D-INT-8: least-recently-seen first.
        seen.sort()
        return EligibilityResult([s for *_, s in seen], used_fallback=True)


class InMemoryExposureLedger(ExposureLedger):
    """Reference in-memory backend. Sufficient for the G0 policy unit tests
    and the G2 local closed-loop; the SQLite/Postgres backends mirror its
    semantics."""

    def __init__(self):
        self._count: dict[tuple[str, str], int] = {}
        # (participant, domain, seg) -> (ordinal, served_at) of the latest exposure
        self._last: dict[tuple[str, str, int], tuple[int, float]] = {}

    def record(self, participant_id: str, seg_id: int, domain: str,
               phase: str, served_at: float) -> ExposureRecord:
        key = (participant_id, domain)
        ordinal = self._count.get(key, 0) + 1
        self._count[key] = ordinal
        self._last[(participant_id, domain, int(seg_id))] = (ordinal, float(served_at))
        return ExposureRecord(participant_id, int(seg_id), domain, phase,
                              float(served_at), ordinal)

    def domain_ordinal(self, participant_id: str, domain: str) -> int:
        return self._count.get((participant_id, domain), 0)

    def last_seen(self, participant_id: str, domain: str, seg_id: int):
        return self._last.get((participant_id, domain, int(seg_id)))

    def seen_segids(self, participant_id: str) -> set:
        return {seg for (p, _d, seg) in self._last if p == participant_id}


class SqliteExposureLedger(ExposureLedger):
    """SQLite-backed exposure ledger (G2 local closed-loop). Same eligibility
    semantics as `InMemoryExposureLedger` (the shared `ExposureLedger.eligible`
    logic); persists across process restarts. The G4 Postgres backend mirrors
    this schema. `path=':memory:'` (default) is ephemeral."""

    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS exposure(participant_id TEXT, "
            "seg_id INTEGER, domain TEXT, phase TEXT, served_at REAL, "
            "domain_ordinal INTEGER)")
        self._db.execute("CREATE INDEX IF NOT EXISTS ix_pd "
                         "ON exposure(participant_id, domain)")
        self._db.execute("CREATE INDEX IF NOT EXISTS ix_pds "
                         "ON exposure(participant_id, domain, seg_id)")
        self._db.commit()

    def record(self, participant_id: str, seg_id: int, domain: str,
               phase: str, served_at: float) -> ExposureRecord:
        ordinal = self.domain_ordinal(participant_id, domain) + 1
        self._db.execute("INSERT INTO exposure VALUES (?,?,?,?,?,?)",
                         (participant_id, int(seg_id), domain, phase,
                          float(served_at), ordinal))
        self._db.commit()
        return ExposureRecord(participant_id, int(seg_id), domain, phase,
                              float(served_at), ordinal)

    def domain_ordinal(self, participant_id: str, domain: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM exposure WHERE participant_id=? AND domain=?",
            (participant_id, domain)).fetchone()
        return int(row[0])

    def last_seen(self, participant_id: str, domain: str, seg_id: int):
        row = self._db.execute(
            "SELECT domain_ordinal, served_at FROM exposure WHERE "
            "participant_id=? AND domain=? AND seg_id=? "
            "ORDER BY domain_ordinal DESC LIMIT 1",
            (participant_id, domain, int(seg_id))).fetchone()
        return (int(row[0]), float(row[1])) if row else None

    def seen_segids(self, participant_id: str) -> set:
        return {int(r[0]) for r in self._db.execute(
            "SELECT DISTINCT seg_id FROM exposure WHERE participant_id=?",
            (participant_id,))}
