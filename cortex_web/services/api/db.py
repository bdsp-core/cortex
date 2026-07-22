"""Backend persistence — same API, two backends (SQLite for dev, Postgres for prod).

Tables (schema is SQL-92, identical on both engines):
  participants        accounts (code → PBKDF2 hash, email, profile, OAuth link).
  sessions            one row per started test; status + provenance stamps.
  trials              append-only per-question checkpoints (crash-safety).
  results             one row per finished test; the full session JSON.
  regimens            training protocol per participant (plan JSON).
  training_sessions   one row per training sitting.
  training_trials     per-question training exposure (L1 forward hook).
  param_trajectories  (task, ℓ, θ, sd, rt) time series for evolution charts.
  login_days          one row per (participant, local day) signed in.
  auth_codes          short-lived 6-digit verify/reset codes.
  consent_events      auditable consent ledger (Phase O1).
  cohorts             manager-run peer groups (isolated pods).
  cohort_members      cohort membership + pending invites.

`CORTEX_DB` selects the backend:
  unset / a filesystem path  → SQLite (WAL).
  "postgres://..." / "postgresql://..."  → Postgres via psycopg3.

Concurrency model — one `_connection()` unit of work per public method:

  * Postgres: a small psycopg connection POOL. Each operation checks a
    connection out, runs its statements, and the pool commits on clean exit /
    rolls back on exception / health-checks and replaces dead connections at
    checkout. This is the load-bearing design change from the 2026-06 shared-
    single-connection incidents: rollback-on-error (06-21), commit-after-read
    so nothing lingers idle-in-transaction (06-22), and reconnect after a
    Postgres restart severs the socket (06-24) are all the POOL's contract
    now, not hand-rolled recovery code here.
  * SQLite (dev/tests): the classic single shared connection under a process
    lock, committed per unit of work. SQLite has none of the failure modes
    above, so it keeps the simple shape.

The only places that branch on backend are (a) connection setup, (b) `?` vs
`%s` placeholders (`_q`), (c) the upserts (INSERT OR REPLACE vs ON CONFLICT).
"""
from __future__ import annotations

import calendar
import json
import os
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from . import timeutil
from .persistence import migrations

_DEFAULT_DB = Path(__file__).with_name("cortex.db")

# Statements split so we can execute them one-by-one on Postgres (no
# executescript). All standard SQL-92 — both engines accept verbatim.
_SCHEMA_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS participants (
        code           TEXT PRIMARY KEY,
        password_hash  TEXT NOT NULL,
        label          TEXT DEFAULT '',
        active         INTEGER NOT NULL DEFAULT 1,
        created_utc    TEXT NOT NULL,
        email          TEXT,
        display_name   TEXT,
        signup_ip      TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id     TEXT PRIMARY KEY,
        code           TEXT NOT NULL,
        participant    TEXT NOT NULL,
        sample_seed    BIGINT,
        started_utc    TEXT NOT NULL,
        finished_utc   TEXT,
        status         TEXT NOT NULL DEFAULT 'in_progress',
        stop_reason    TEXT,
        n_questions    INTEGER,
        termination_policy TEXT NOT NULL DEFAULT 'ad6',
        compute_mode       TEXT NOT NULL DEFAULT 'serial',
        candidate_exclusion TEXT,
        candidate_bank_sha256 TEXT,
        nway_profile      TEXT,
        FOREIGN KEY (code) REFERENCES participants(code)
    )""",
    """CREATE TABLE IF NOT EXISTS trials (
        session_id     TEXT NOT NULL,
        trial_index    INTEGER NOT NULL,
        seg_id         INTEGER,
        task_k         INTEGER,
        pick           INTEGER,
        is_correct     INTEGER,
        reaction_ms    REAL,
        diag           TEXT,
        received_utc   TEXT NOT NULL,
        shown_client_utc    TEXT,   -- client wall-clock at item render (L2)
        answered_client_utc TEXT,   -- client wall-clock at answer (L2)
        PRIMARY KEY (session_id, trial_index)
    )""",
    """CREATE TABLE IF NOT EXISTS results (
        session_id     TEXT PRIMARY KEY,
        result         TEXT NOT NULL,
        received_utc   TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_sessions_code ON sessions(code)",
    # ── dashboard / learning-protocol tables (Phase 2) ──
    # A participant's training protocol, generated from a certification result:
    # per-task target ℓ* + a scheduled deck plan (stored as JSON in `plan`).
    """CREATE TABLE IF NOT EXISTS regimens (
        regimen_id        TEXT PRIMARY KEY,
        code              TEXT NOT NULL,
        source_session_id TEXT,
        plan              TEXT NOT NULL,
        active            INTEGER NOT NULL DEFAULT 1,
        created_utc       TEXT NOT NULL
    )""",
    # One row per daily training sitting (training-mode analogue of sessions).
    """CREATE TABLE IF NOT EXISTS training_sessions (
        training_id       TEXT PRIMARY KEY,
        code              TEXT NOT NULL,
        task_focus        TEXT,
        started_utc       TEXT NOT NULL,
        finished_utc      TEXT,
        status            TEXT NOT NULL DEFAULT 'in_progress',
        n_items           INTEGER,
        summary           TEXT,
        regimen_id        TEXT,        -- FK→regimens (Phase O2)
        source_session_id TEXT         -- FK→sessions (the seeding cert; Phase O2)
    )""",
    # Append-only time series of (task, ℓ, θ, sd, rt) for the evolution charts.
    """CREATE TABLE IF NOT EXISTS param_trajectories (
        code              TEXT NOT NULL,
        task_k            INTEGER NOT NULL,
        phase             TEXT NOT NULL,
        ell               REAL,
        theta             REAL,
        sd                REAL,
        rt                REAL,
        ts                TEXT NOT NULL,
        training_id       TEXT,         -- FK→training_sessions (Phase O2)
        source_session_id TEXT,         -- FK→sessions (Phase O2)
        seq_in_session    INTEGER,      -- monotonic order within a sitting (Phase O2)
        is_real           INTEGER NOT NULL DEFAULT 0  -- 1=real trainer output (L1); 0=synthetic/quarantined
    )""",
    "CREATE INDEX IF NOT EXISTS idx_regimens_code ON regimens(code)",
    "CREATE INDEX IF NOT EXISTS idx_training_code ON training_sessions(code)",
    "CREATE INDEX IF NOT EXISTS idx_param_traj_code ON param_trajectories(code)",
    # record_training_progress scans by training_id on every checkpoint POST.
    "CREATE INDEX IF NOT EXISTS idx_param_traj_tid ON param_trajectories(training_id)",
    # One row per (participant, UTC day) the participant signed in — the
    # lightest tier of the dashboard activity heatmap (cert/training days are
    # derived from sessions/training_sessions).
    """CREATE TABLE IF NOT EXISTS login_days (
        code  TEXT NOT NULL,
        day   TEXT NOT NULL,
        PRIMARY KEY (code, day)
    )""",
    # Short-lived 6-digit codes for email verification + password reset.
    # One unconsumed (code, purpose) row is kept at a time (put replaces).
    """CREATE TABLE IF NOT EXISTS auth_codes (
        participant_code  TEXT NOT NULL,
        purpose           TEXT NOT NULL,
        code_hash         TEXT NOT NULL,
        created_utc       TEXT NOT NULL,
        expires_utc       TEXT NOT NULL,
        consumed_utc      TEXT,
        attempts          INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (participant_code, purpose)
    )""",
    # Auditable consent ledger (Phase O1). One row per (participant, consent
    # type, version); re-accepting re-affirms (clears withdrawal); withdrawal
    # sets withdrawn_utc and is honoured at research-export time.
    """CREATE TABLE IF NOT EXISTS consent_events (
        code            TEXT NOT NULL,
        consent_type    TEXT NOT NULL,
        consent_version TEXT NOT NULL,
        irb_protocol_id TEXT,
        accepted_utc    TEXT NOT NULL,
        consent_ip      TEXT,
        withdrawn_utc   TEXT,
        PRIMARY KEY (code, consent_type, consent_version)
    )""",
    # Per-question TRAINING exposure (forward hook for the L1 trainer). Empty
    # until the real trainer ships; get_exposure_exclusion already unions it
    # with test `trials` so test↔train re-exposure spacing works the day
    # training data starts landing. (Test exposure lives in `trials`.)
    """CREATE TABLE IF NOT EXISTS training_trials (
        training_id   TEXT NOT NULL,
        code          TEXT NOT NULL,
        seg_id        INTEGER NOT NULL,
        task_k        INTEGER,
        shown_utc     TEXT NOT NULL,
        pick                INTEGER,  -- response record (Phase L2): raw response index
        y_star              INTEGER,  -- gold label for the asked task
        is_correct          INTEGER,
        feedback_shown      TEXT,     -- the reveal text actually rendered
        rt_ms               REAL,     -- client reaction time delta
        shown_client_utc    TEXT,
        answered_client_utc TEXT,
        seq_in_session      INTEGER,  -- answer order (engine rebuild key, L3)
        mode                TEXT,     -- serving mode (skill/bias/review; L4)
        link                TEXT,     -- 'binary' one-vs-rest | 'nway' (L4)
        quality_flag        TEXT,     -- NULL ok | 'burst' sub-500ms run (L4)
        PRIMARY KEY (training_id, seg_id)
    )""",
    # Cross-sitting spaced-repetition state (Phase L4 retention layer): the
    # incumbent's expanding-interval pattern (interval x ease on a correct
    # review retrieval, reset on a miss), persisted per (participant, domain).
    """CREATE TABLE IF NOT EXISTS training_retention (
        code          TEXT NOT NULL,
        domain        TEXT NOT NULL,
        next_due_utc  TEXT NOT NULL,
        interval_s    REAL NOT NULL,
        PRIMARY KEY (code, domain)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_training_trials_code ON training_trials(code)",
    # record_training_progress scans by training_id on every checkpoint POST.
    "CREATE INDEX IF NOT EXISTS idx_training_trials_tid ON training_trials(training_id)",
    # ── cohorts: manager-run peer groups ──
    # Each cohort is an isolated pod: membership is the ONLY grant that lets a
    # participant see other members' performance, and payloads are keyed by
    # public_id (display names resolve for the manager alone; internal codes
    # never leave the API). The creator is the manager and is also an active
    # member (their line plots too).
    """CREATE TABLE IF NOT EXISTS cohorts (
        cohort_id     TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        manager_code  TEXT NOT NULL,
        created_utc   TEXT NOT NULL
    )""",
    # status: 'invited' (manager added the public id; no data visible either
    # direction until the user accepts) | 'active'. Decline/leave/remove all
    # DELETE the row.
    """CREATE TABLE IF NOT EXISTS cohort_members (
        cohort_id     TEXT NOT NULL,
        code          TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT 'invited',
        invited_utc   TEXT NOT NULL,
        joined_utc    TEXT,
        PRIMARY KEY (cohort_id, code)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cohort_members_code ON cohort_members(code)",
    "CREATE INDEX IF NOT EXISTS idx_cohorts_manager ON cohorts(manager_code)",
    # Email invites to addresses with no account yet: attached (converted to a
    # normal 'invited' cohort_members row) when that email finishes signup
    # (attach_email_invites). The invitee still accepts in-app — the consent
    # gate is identical to the public-id invite flow.
    """CREATE TABLE IF NOT EXISTS cohort_email_invites (
        cohort_id     TEXT NOT NULL,
        email         TEXT NOT NULL,
        invited_utc   TEXT NOT NULL,
        PRIMARY KEY (cohort_id, email)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cohort_email_invites_email "
    "ON cohort_email_invites(email)",
    # Ripeness-digest ledger: one row per (participant, LOCAL day) a training
    # digest letter was sent. The INSERT is the atomic once-per-day claim
    # (claim_digest_day), taken BEFORE the send so a crash can never double-mail.
    """CREATE TABLE IF NOT EXISTS digest_log (
        code      TEXT NOT NULL,
        day       TEXT NOT NULL,
        sent_utc  TEXT NOT NULL,
        PRIMARY KEY (code, day)
    )""",
    # Recognition ledger (awards.py): domain badges + milestones. Badges are
    # per-domain certification credentials; losing one sets revoked_utc and a
    # re-earn APPENDS a new row (the earned/lost history is the point).
    # Milestones are once-ever rows; acknowledged_utc records the banner
    # dismissal so a milestone is announced exactly once.
    """CREATE TABLE IF NOT EXISTS awards (
        award_id         TEXT PRIMARY KEY,
        code             TEXT NOT NULL,
        kind             TEXT NOT NULL,
        key              TEXT NOT NULL,
        label            TEXT NOT NULL,
        detail           TEXT,
        awarded_utc      TEXT NOT NULL,
        revoked_utc      TEXT,
        acknowledged_utc TEXT
    )""",
    "CREATE INDEX IF NOT EXISTS idx_awards_code ON awards(code)",
    # The email-unique index is created in _migrate_participants AFTER the
    # ALTER TABLE that ensures the column exists (an older schema may not
    # have it yet on an existing DB).
]


# Columns added to participants after the original schema. Idempotent
# ALTERs run at startup so an existing database upgrades in place.
# Re-exported: `from .db import utc_now` is the historical spelling used
# across the package. The definition lives in timeutil (see its module
# docstring on why the format must never vary).
utc_now = timeutil.utc_now


# User-facing 9-digit account ids (`participants.public_id`): uniformly random
# in [100000000, 999999999] — always exactly 9 digits, no leading zero. ~9e8
# values, so collisions are vanishingly rare; the unique index + the retry in
# _allocate_public_id are the correctness backstop, not this generator. Stored
# as TEXT: it is an identifier, never arithmetic.
_PUBLIC_ID_MAX_TRIES = 20

# Never-accepted cohort invites (pending member invites + email invites)
# expire after this many days — hygiene for typo'd addresses and forgotten
# invitations. Enforced lazily by purge_expired_invites().
INVITE_TTL_DAYS = 30


def _random_public_id() -> str:
    return str(100_000_000 + secrets.randbelow(900_000_000))


def _is_public_id_collision(e: Exception) -> bool:
    """True when a write failed on the public_id unique index specifically
    (sqlite3.IntegrityError names the column; psycopg UniqueViolation names
    the index — both messages contain 'public_id')."""
    s = str(e)
    return ("UNIQUE" in s or "unique" in s or "duplicate" in s) and "public_id" in s


def _local_day(utc_iso: Optional[str], tz_offset_min: int) -> Optional[str]:
    """The local calendar day (YYYY-MM-DD) for a UTC ISO timestamp, given the
    browser's getTimezoneOffset (minutes UTC is AHEAD of local). VPN-safe: the
    offset comes from the device clock, not IP geolocation."""
    if not utc_iso or len(utc_iso) < 19:
        return utc_iso[:10] if utc_iso else None
    try:
        secs = calendar.timegm(time.strptime(utc_iso[:19], "%Y-%m-%dT%H:%M:%S"))
        return time.strftime("%Y-%m-%d", time.gmtime(secs - (tz_offset_min or 0) * 60))
    except Exception:
        return utc_iso[:10]


def _is_pg_url(s: str) -> bool:
    return s.startswith("postgres://") or s.startswith("postgresql://")


class Database:
    """Thread-safe wrapper. Identical API for SQLite and Postgres backends."""

    def __init__(self, path_or_url: Optional[Path | str] = None):
        raw = str(path_or_url or os.environ.get("CORTEX_DB", _DEFAULT_DB))
        self._raw = raw
        self._pg = _is_pg_url(raw)
        # Lazy invite-purge throttle (see purge_expired_invites): 0.0 means
        # "never ran", so the first cohort read after boot always purges.
        self._last_invite_purge = 0.0

        if self._pg:
            self._pool = self._make_pool()
            self.path = None
        else:
            self._lock = threading.Lock()
            self.path = Path(raw)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

        # Boot schema + idempotent additive migrations, one transaction.
        with self._connection() as conn:
            for stmt in _SCHEMA_STATEMENTS:
                conn.execute(stmt)
            self._migrate_participants(conn)
            for table, columns in migrations.BY_TABLE:
                self._add_missing_columns(conn, table, columns)
            # Quarantine backfill: every pre-existing trajectory row is synthetic
            # (the only writer before the L1 trainer was the removed Shell.tsx
            # placeholder). NOT NULL DEFAULT 0 already fills new column; this is
            # belt-and-suspenders for any backend that left it NULL.
            conn.execute("UPDATE param_trajectories SET is_real=0 WHERE is_real IS NULL")

        # Grandfather accounts created before the public_id column: assign a
        # 9-digit id to every row missing one. Own units of work (the retry on
        # a collision must not abort the boot-migration transaction above).
        self._backfill_public_ids()

    # ── connection management ─────────────────────────────────────
    def _make_pool(self):
        """Open the psycopg3 connection pool (lazy import so SQLite-only dev
        installs don't need psycopg). min_size=1 keeps one warm connection;
        max_size defaults to 10 (env CORTEX_PG_POOL_MAX): sync routes run on
        Starlette's ~40-thread pool, so a burst of >max_size DB-touching
        requests queues on checkout — 4 was a needlessly low ceiling for the
        "100 concurrent participants" target on local Postgres. `check`
        re-validates a connection at checkout and silently replaces a dead
        one — a Postgres restart (the 2026-06-24 outage) self-heals on the
        next request instead of failing until a manual service restart; the
        extra SELECT 1 round-trip is sub-ms against the on-box DB and is the
        deliberate robustness>latency trade."""
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        return ConnectionPool(
            self._raw, min_size=1,
            max_size=int(os.environ.get("CORTEX_PG_POOL_MAX", "10")),
            open=True,
            check=ConnectionPool.check_connection,
            kwargs={"row_factory": dict_row},
        )

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        """One transactional unit of work on one connection.

        Postgres: a pooled connection — psycopg_pool commits on clean exit,
        rolls back on exception, and never leaves it idle-in-transaction
        (the 2026-06-21/22 failure modes, by construction). SQLite: the shared
        connection under the process lock, committed on clean exit."""
        if self._pg:
            with self._pool.connection() as conn:
                yield conn
        else:
            with self._lock:
                try:
                    yield self._conn
                    self._conn.commit()
                except Exception:
                    self._conn.rollback()
                    raise

    def _q(self, sql: str) -> str:
        """Translate `?` placeholders to `%s` when on Postgres."""
        return sql.replace("?", "%s") if self._pg else sql

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        with self._connection() as conn:
            row = conn.execute(self._q(sql), params).fetchone()
            return dict(row) if row is not None else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connection() as conn:
            return [dict(r) for r in conn.execute(self._q(sql), params).fetchall()]

    def _write(self, sql: str, params: tuple = ()) -> None:
        """Execute one write statement as its own unit of work."""
        with self._connection() as conn:
            conn.execute(self._q(sql), params)

    def close(self) -> None:
        if self._pg:
            self._pool.close()
        else:
            with self._lock:
                self._conn.close()

    def ping(self) -> None:
        """Round-trip the backend (SELECT 1) — raises when the DB is unusable.
        Powers GET /api/health?deep=1 so an external monitor sees a DB outage
        as a 503 instead of the shallow probe's evergreen 200."""
        with self._connection() as conn:
            conn.execute("SELECT 1")

    # ── boot migrations (run inside __init__'s transaction) ──────
    def _migrate_participants(self, conn) -> None:
        """Add columns to participants that didn't exist in earlier schemas.
        Idempotent on both engines."""
        if self._pg:
            for col, typ in migrations.PARTICIPANTS:
                conn.execute(f"ALTER TABLE participants ADD COLUMN IF NOT EXISTS {col} {typ}")
        else:
            existing = {dict(r)["name"] for r in
                        conn.execute("PRAGMA table_info(participants)").fetchall()}
            for col, typ in migrations.PARTICIPANTS:
                if col not in existing:
                    conn.execute(f"ALTER TABLE participants ADD COLUMN {col} {typ}")
        # Now that the email column is guaranteed to exist, the unique-when-
        # set index is safe to create on both engines.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_email "
            "ON participants(email) WHERE email IS NOT NULL"
        )
        # One CORTEX account per Google identity.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_google_sub "
            "ON participants(google_sub) WHERE google_sub IS NOT NULL"
        )
        # The user-facing 9-digit id must never repeat across accounts.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_public_id "
            "ON participants(public_id) WHERE public_id IS NOT NULL"
        )

    def _add_missing_columns(self, conn, table: str,
                             columns: list[tuple[str, str]]) -> None:
        """Idempotent additive migration for `table` on both backends — the
        generic form of _migrate_participants, used for the learning tables."""
        if self._pg:
            for col, typ in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
        else:
            existing = {dict(r)["name"] for r in
                        conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col, typ in columns:
                if col not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    # ── participants ──────────────────────────────────────────────
    def _allocate_public_id(self, write_with_pid) -> str:
        """Run a write that stores a candidate 9-digit public id, retrying
        with a fresh candidate if the unique index rejects it (two accounts
        drawing the same number — ~1e-9 per signup). Any other failure
        (email collision, etc.) propagates to the caller unchanged."""
        for _ in range(_PUBLIC_ID_MAX_TRIES):
            pid = _random_public_id()
            try:
                write_with_pid(pid)
                return pid
            except Exception as e:
                if _is_public_id_collision(e):
                    continue
                raise
        raise RuntimeError("could not allocate a unique public id")

    def ensure_public_id(self, code: str) -> Optional[str]:
        """Return the account's 9-digit public id, assigning one first if the
        row predates the public_id column (grandfathering + lazy repair)."""
        row = self._fetchone(
            "SELECT public_id FROM participants WHERE code=?", (code,))
        if row is None:
            return None
        if row.get("public_id"):
            return str(row["public_id"])
        # The IS NULL guard makes a concurrent double-assign a no-op for the
        # loser; the re-read below returns whichever id actually landed.
        self._allocate_public_id(lambda pid: self._write(
            "UPDATE participants SET public_id=? WHERE code=? AND public_id IS NULL",
            (pid, code)))
        row = self._fetchone(
            "SELECT public_id FROM participants WHERE code=?", (code,))
        return str(row["public_id"]) if row and row.get("public_id") else None

    def _backfill_public_ids(self) -> None:
        """Assign a public id to every account missing one. Idempotent; runs
        at every boot so pre-column rows (and any row a failed assignment
        left behind) are repaired without a manual migration."""
        for r in self._fetchall(
                "SELECT code FROM participants WHERE public_id IS NULL"):
            self.ensure_public_id(r["code"])

    def add_participant(self, code: str, password_hash: str, label: str = "") -> str:
        return self._allocate_public_id(lambda pid: self._write(
            "INSERT INTO participants(code, password_hash, label, public_id, "
            "created_utc) VALUES (?,?,?,?,?)",
            (code, password_hash, label, pid, utc_now()),
        ))

    def get_participant(self, code: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM participants WHERE code=?", (code,))

    def list_participants(self) -> list[dict]:
        return self._fetchall(
            "SELECT code,label,active,created_utc FROM participants "
            "ORDER BY created_utc")

    def set_participant_active(self, code: str, active: bool) -> None:
        self._write("UPDATE participants SET active=? WHERE code=?",
                    (1 if active else 0, code))

    # ── public-signup helpers ─────────────────────────────────────
    def register_participant(self, *, code: str, password_hash: str,
                              email: str, display_name: str,
                              signup_ip: Optional[str] = None,
                              signup_expertise: Optional[str] = None,
                              profile: Optional[str] = None) -> None:
        """Insert a new email-based account. Caller is responsible for
        generating `code` (the stable internal id stored in JWT subjects);
        UNIQUE constraint on email surfaces as a backend-specific IntegrityError
        the caller catches to return a clean 409. `signup_expertise` is the
        optional self-reported role from the signup form (Phase O1; defaults
        None so existing callers are unchanged). `profile` is the JSON
        demographic/clinical record collected during signup (editable later in
        Settings). A fresh 9-digit public id is allocated with the row."""
        self._allocate_public_id(lambda pid: self._write(
            "INSERT INTO participants(code, password_hash, email, "
            "display_name, signup_ip, signup_expertise, profile, public_id, "
            "created_utc) VALUES (?,?,?,?,?,?,?,?,?)",
            (code, password_hash, email, display_name, signup_ip,
             signup_expertise, profile, pid, utc_now()),
        ))

    def update_pending_registration(self, *, code: str, password_hash: str,
                                    display_name: str,
                                    signup_ip: Optional[str] = None,
                                    signup_expertise: Optional[str] = None,
                                    profile: Optional[str] = None) -> None:
        """Overwrite the mutable signup fields of an UNVERIFIED account when
        the same email registers again (see routers/auth.py). The internal
        code and public_id stay stable; created_utc refreshes to the retake."""
        self._write(
            "UPDATE participants SET password_hash=?, display_name=?, "
            "signup_ip=?, signup_expertise=?, profile=?, created_utc=? "
            "WHERE code=?",
            (password_hash, display_name, signup_ip, signup_expertise,
             profile, utc_now(), code))

    def update_profile(self, code: str, *, display_name: Optional[str],
                       profile: Optional[str], signup_expertise: Optional[str]) -> None:
        """Update the editable account fields (Settings page). display_name and
        signup_expertise are stored columns; profile is the JSON demographic
        record. None args are left unchanged."""
        sets, params = [], []
        if display_name is not None:
            sets.append("display_name=?"); params.append(display_name)
        if profile is not None:
            sets.append("profile=?"); params.append(profile)
        if signup_expertise is not None:
            sets.append("signup_expertise=?"); params.append(signup_expertise)
        if not sets:
            return
        params.append(code)
        self._write(f"UPDATE participants SET {', '.join(sets)} WHERE code=?",
                    tuple(params))

    def update_email(self, code: str, new_email: str) -> None:
        """Change an account's email. The UNIQUE-when-set index surfaces a
        collision as a backend IntegrityError the caller maps to a 409."""
        self._write("UPDATE participants SET email=? WHERE code=?", (new_email, code))

    def get_participant_by_email(self, email: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM participants WHERE email=?", (email,))

    def mark_email_verified(self, code: str) -> None:
        self._write(
            "UPDATE participants SET email_verified_utc=? WHERE code=?",
            (utc_now(), code))

    def mark_email_undeliverable(self, code: str) -> None:
        """Record that mail to this account's address hard-bounced (SES bounce
        event). Surfaced to pending signups via POST /api/verify/status."""
        self._write(
            "UPDATE participants SET email_undeliverable_utc=? WHERE code=?",
            (utc_now(), code))

    def clear_email_undeliverable(self, code: str) -> None:
        """A confirmed verify/reset code proves the mailbox receives our mail."""
        self._write(
            "UPDATE participants SET email_undeliverable_utc=NULL WHERE code=?",
            (code,))

    def set_password_hash(self, code: str, password_hash: str) -> None:
        self._write(
            "UPDATE participants SET password_hash=? WHERE code=?",
            (password_hash, code))

    # ── Google OAuth accounts ─────────────────────────────────────
    def get_participant_by_google_sub(self, sub: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM participants WHERE google_sub=?", (sub,))

    def register_oauth_participant(self, *, code: str, email: str, display_name: str,
                                    google_sub: str, signup_ip: Optional[str] = None) -> None:
        """Insert a new Google-authenticated account. password_hash holds a
        non-PBKDF2 sentinel (so password login is impossible); the email is
        verified by Google, so email_verified_utc is set immediately. A fresh
        9-digit public id is allocated with the row."""
        self._allocate_public_id(lambda pid: self._write(
            "INSERT INTO participants(code, password_hash, email, display_name, "
            "signup_ip, created_utc, auth_provider, google_sub, "
            "email_verified_utc, public_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (code, "google-oauth:no-password", email, display_name, signup_ip,
             utc_now(), "google", google_sub, utc_now(), pid)))

    def link_google_sub(self, code: str, sub: str) -> None:
        """Attach a Google identity to an existing (e.g. password) account."""
        self._write(
            "UPDATE participants SET google_sub=? WHERE code=?", (sub, code))

    # ── auth codes (email verify + password reset) ────────────────
    def put_auth_code(self, participant_code: str, purpose: str,
                      code_hash: str, expires_utc: str) -> None:
        """Store (replacing any existing) the active code for (account, purpose)."""
        if self._pg:
            sql = ("INSERT INTO auth_codes(participant_code, purpose, code_hash, "
                   "created_utc, expires_utc, consumed_utc, attempts) "
                   "VALUES (?,?,?,?,?,NULL,0) "
                   "ON CONFLICT (participant_code, purpose) DO UPDATE SET "
                   "code_hash=EXCLUDED.code_hash, created_utc=EXCLUDED.created_utc, "
                   "expires_utc=EXCLUDED.expires_utc, consumed_utc=NULL, attempts=0")
        else:
            sql = ("INSERT OR REPLACE INTO auth_codes(participant_code, purpose, "
                   "code_hash, created_utc, expires_utc, consumed_utc, attempts) "
                   "VALUES (?,?,?,?,?,NULL,0)")
        self._write(sql, (participant_code, purpose, code_hash,
                          utc_now(), expires_utc))

    def get_auth_code(self, participant_code: str, purpose: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM auth_codes WHERE participant_code=? AND purpose=?",
            (participant_code, purpose))

    def increment_auth_attempts(self, participant_code: str, purpose: str) -> None:
        self._write(
            "UPDATE auth_codes SET attempts=attempts+1 "
            "WHERE participant_code=? AND purpose=?",
            (participant_code, purpose))

    def consume_auth_code(self, participant_code: str, purpose: str) -> None:
        self._write(
            "UPDATE auth_codes SET consumed_utc=? "
            "WHERE participant_code=? AND purpose=?",
            (utc_now(), participant_code, purpose))

    # ── consent ledger (Phase O1) ─────────────────────────────────
    def record_consent(self, code: str, consent_type: str, consent_version: str,
                        *, irb_protocol_id: Optional[str] = None,
                        consent_ip: Optional[str] = None) -> None:
        """Record (or re-affirm) a participant's consent. Re-accepting the same
        (type, version) clears any prior withdrawal."""
        if self._pg:
            sql = ("INSERT INTO consent_events(code, consent_type, consent_version, "
                   "irb_protocol_id, accepted_utc, consent_ip, withdrawn_utc) "
                   "VALUES (?,?,?,?,?,?,NULL) "
                   "ON CONFLICT (code, consent_type, consent_version) DO UPDATE SET "
                   "accepted_utc=EXCLUDED.accepted_utc, "
                   "irb_protocol_id=EXCLUDED.irb_protocol_id, "
                   "consent_ip=EXCLUDED.consent_ip, withdrawn_utc=NULL")
        else:
            sql = ("INSERT OR REPLACE INTO consent_events(code, consent_type, "
                   "consent_version, irb_protocol_id, accepted_utc, consent_ip, "
                   "withdrawn_utc) VALUES (?,?,?,?,?,?,NULL)")
        self._write(sql, (code, consent_type, consent_version,
                          irb_protocol_id, utc_now(), consent_ip))

    def withdraw_consent(self, code: str, consent_type: Optional[str] = None) -> int:
        """Mark consent withdrawn (all types, or one). Honoured at export time
        (future releases only — already-released DOIs are irrevocable)."""
        with self._connection() as conn:
            if consent_type:
                cur = conn.execute(self._q(
                    "UPDATE consent_events SET withdrawn_utc=? "
                    "WHERE code=? AND consent_type=?"),
                    (utc_now(), code, consent_type))
            else:
                cur = conn.execute(self._q(
                    "UPDATE consent_events SET withdrawn_utc=? WHERE code=?"),
                    (utc_now(), code))
            return (cur.rowcount or 0) if hasattr(cur, "rowcount") else 0

    def get_consent_events(self, code: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM consent_events WHERE code=? "
            "ORDER BY consent_type, consent_version", (code,))

    # ── sessions ──────────────────────────────────────────────────
    def create_session(self, session_id: str, code: str, participant: dict,
                        sample_seed: Optional[int],
                        bundle_version: Optional[str] = None,
                        drawn_seg_ids: Optional[str] = None,
                        termination_policy: str = "ad6",
                        compute_mode: str = "serial",
                        candidate_exclusion: Optional[str] = None,
                        candidate_bank_sha256: Optional[str] = None,
                        nway_profile: Optional[dict] = None) -> None:
        # A distinct open-state is a rollback interlock. Older releases search
        # only status='in_progress', so after an application rollback they
        # cannot resume and reinterpret a native n-way answer stream as binary.
        # The new release treats both open states normally; result ingest on an
        # already-open browser remains accepted by either release.
        open_status = "in_progress_nway" if nway_profile else "in_progress"
        self._write(
            "INSERT INTO sessions(session_id, code, participant, sample_seed, "
            "bundle_version, drawn_seg_ids, termination_policy, "
            "compute_mode, candidate_exclusion, candidate_bank_sha256, nway_profile, "
            "started_utc, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session_id, code, json.dumps(participant), sample_seed,
             bundle_version, drawn_seg_ids, termination_policy, compute_mode,
             candidate_exclusion, candidate_bank_sha256,
             json.dumps(nway_profile, sort_keys=True) if nway_profile else None,
             utc_now(), open_status),
        )

    def get_session(self, session_id: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,))

    def get_active_session(self, code: str,
                           max_age_hours: int = 24) -> Optional[dict]:
        """The participant's most recent still-open sitting, for resume.
        Age-gated: past the window a fresh draw serves them better than a
        test they've lost the context of."""
        row = self._fetchone(
            "SELECT * FROM sessions WHERE code=? "
            "AND status IN ('in_progress','in_progress_nway') "
            "ORDER BY started_utc DESC LIMIT 1", (code,))
        if row is None:
            return None
        cutoff = timeutil.iso_in(-max_age_hours * 3600)
        return row if row["started_utc"] >= cutoff else None

    def supersede_open_sessions(self, code: str) -> None:
        """Starting a new sitting retires any still-open one, so at most one
        session per account is ever offered for resume. A superseded session
        keeps accepting trail-end progress/results posts (ownership is the
        only gate there) — it just stops being resumable."""
        self._write(
            "UPDATE sessions SET status='superseded' "
            "WHERE code=? AND status IN ('in_progress','in_progress_nway')", (code,))

    def _finalize_session_stmt(self, conn, session_id: str,
                               stop_reason: Optional[str],
                               n_questions: Optional[int]) -> None:
        conn.execute(self._q(
            "UPDATE sessions SET status='complete', finished_utc=?, "
            "stop_reason=?, n_questions=? WHERE session_id=?"),
            (utc_now(), stop_reason, n_questions, session_id))

    def finalize_session(self, session_id: str, stop_reason: Optional[str],
                         n_questions: Optional[int]) -> None:
        with self._connection() as conn:
            self._finalize_session_stmt(conn, session_id, stop_reason, n_questions)

    def has_training_history(self, code: str) -> bool:
        """Any training sitting exists (drives the dashboard CTA label:
        Resume vs Start training). PROGRAM-level semantics: beliefs continue
        across sittings (the regimen prior seeds from the latest trajectory
        point) and exposure spacing prevents repeats, so once training has
        ever started, the truthful label is Resume."""
        return self._fetchone(
            "SELECT 1 AS x FROM training_sessions WHERE code=? LIMIT 1",
            (code,)) is not None

    def latest_training_finished(self, code: str) -> Optional[str]:
        """finished_utc of the participant's most recent COMPLETED training
        sitting, or None. Powers the exam washout gate (routers/testing.py):
        an exam started too soon after training measures a practice boost,
        not stable skill."""
        row = self._fetchone(
            "SELECT MAX(finished_utc) AS f FROM training_sessions "
            "WHERE code=? AND status='complete'", (code,))
        return row["f"] if row and row["f"] else None

    # ── trials ────────────────────────────────────────────────────
    def upsert_trial(self, session_id: str, trial: dict) -> None:
        # Upsert syntax differs: SQLite's INSERT OR REPLACE vs Postgres's
        # ON CONFLICT … DO UPDATE on the (session_id, trial_index) PK.
        if self._pg:
            sql = ("INSERT INTO trials(session_id, trial_index, seg_id, "
                   "task_k, pick, is_correct, reaction_ms, diag, received_utc, "
                   "shown_client_utc, answered_client_utc) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                   "ON CONFLICT (session_id, trial_index) DO UPDATE SET "
                   "seg_id=EXCLUDED.seg_id, task_k=EXCLUDED.task_k, "
                   "pick=EXCLUDED.pick, is_correct=EXCLUDED.is_correct, "
                   "reaction_ms=EXCLUDED.reaction_ms, diag=EXCLUDED.diag, "
                   "received_utc=EXCLUDED.received_utc, "
                   "shown_client_utc=EXCLUDED.shown_client_utc, "
                   "answered_client_utc=EXCLUDED.answered_client_utc")
        else:
            sql = ("INSERT OR REPLACE INTO trials(session_id, trial_index, "
                   "seg_id, task_k, pick, is_correct, reaction_ms, diag, "
                   "received_utc, shown_client_utc, answered_client_utc) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?)")
        self._write(sql, (
            session_id,
            int(trial.get("trialIndex", trial.get("trial_index", 0))),
            trial.get("segId", trial.get("seg_id")),
            trial.get("taskK", trial.get("task_k")),
            trial.get("pick"),
            1 if trial.get("isCorrect") else 0 if "isCorrect" in trial else None,
            trial.get("reactionMs", trial.get("reaction_ms")),
            json.dumps(trial.get("diag")) if trial.get("diag") is not None else None,
            utc_now(),
            trial.get("shownClientUtc"),
            trial.get("answeredClientUtc"),
        ))

    def session_trials(self, session_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM trials WHERE session_id=? ORDER BY trial_index",
            (session_id,))

    def get_exposure_exclusion(self, code: str, days_window: int,
                               session_window: int) -> set:
        """Seg_ids this participant saw recently — the UNION of their last
        `session_window` sittings AND everything within `days_window` days
        (whichever is stricter), across BOTH test (`trials`) and training
        (`training_trials`). Drives question-reuse spacing at draw time so a
        segment seen in testing or training isn't re-shown too soon."""
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - max(0, days_window) * 86400))
        excl: set = set()
        with self._connection() as conn:
            # (sitting table, its id column, the per-question table keyed by it)
            for sess_tbl, id_col, trial_tbl in (
                ("sessions", "session_id", "trials"),
                ("training_sessions", "training_id", "training_trials"),
            ):
                sittings = [dict(r) for r in conn.execute(self._q(
                    f"SELECT {id_col} AS sid, started_utc AS started_utc "
                    f"FROM {sess_tbl} WHERE code=? ORDER BY started_utc DESC"),
                    (code,)).fetchall()]
                keep = [r["sid"] for i, r in enumerate(sittings)
                        if i < session_window or (r["started_utc"] or "") >= cutoff]
                if not keep:
                    continue
                qmarks = ",".join("?" for _ in keep)
                rows = conn.execute(self._q(
                    f"SELECT DISTINCT seg_id FROM {trial_tbl} "
                    f"WHERE seg_id IS NOT NULL AND {id_col} IN ({qmarks})"),
                    tuple(keep)).fetchall()
                excl.update(int(dict(r)["seg_id"]) for r in rows
                            if dict(r)["seg_id"] is not None)
        return excl

    # ── results ───────────────────────────────────────────────────
    def _store_result_stmt(self, conn, session_id: str, result: dict) -> None:
        if self._pg:
            sql = ("INSERT INTO results(session_id, result, received_utc) "
                   "VALUES (?,?,?) "
                   "ON CONFLICT (session_id) DO UPDATE SET "
                   "result=EXCLUDED.result, received_utc=EXCLUDED.received_utc")
        else:
            sql = ("INSERT OR REPLACE INTO results(session_id, result, "
                   "received_utc) VALUES (?,?,?)")
        conn.execute(self._q(sql), (session_id, json.dumps(result), utc_now()))

    def store_result(self, session_id: str, result: dict) -> None:
        with self._connection() as conn:
            self._store_result_stmt(conn, session_id, result)

    def get_result(self, session_id: str) -> Optional[dict]:
        row = self._fetchone(
            "SELECT result FROM results WHERE session_id=?", (session_id,))
        return json.loads(row["result"]) if row else None

    def all_sessions(self) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM sessions ORDER BY started_utc")

    # ── dashboard: latest certification result for a participant ──
    # NULLS LAST: on Postgres a NULL finished_utc sorts FIRST under DESC, so a
    # half-finalized session (result stored, finalize interrupted) would shadow
    # the true latest attempt. SQLite ≥3.30 accepts the same syntax.
    def latest_result_for_code(self, code: str) -> Optional[dict]:
        """The most recent completed-session attempt for `code` (same row shape
        as one list_results_for_code entry), or None. LIMIT 1 — the dashboard
        must not pay O(attempts × blob) just to read the newest result."""
        row = self._fetchone(
            "SELECT s.session_id AS session_id, s.finished_utc AS finished_utc, "
            "s.n_questions AS n_questions, s.stop_reason AS stop_reason, "
            "r.result AS result FROM results r "
            "JOIN sessions s ON s.session_id = r.session_id "
            "WHERE s.code=? "
            "ORDER BY s.finished_utc DESC NULLS LAST, s.started_utc DESC "
            "LIMIT 1",
            (code,))
        if row is None:
            return None
        return {"session_id": row["session_id"], "finished_utc": row["finished_utc"],
                "n_questions": row["n_questions"], "stop_reason": row["stop_reason"],
                "result": json.loads(row["result"])}

    def list_results_for_code(self, code: str) -> list[dict]:
        """All COMPLETED sessions for `code` joined with their result JSON,
        newest first. Mirrors latest_result_for_code's JOIN but returns every
        attempt with its session metadata. Scoped strictly by sessions.code."""
        rows = self._fetchall(
            "SELECT s.session_id AS session_id, s.finished_utc AS finished_utc, "
            "s.n_questions AS n_questions, s.stop_reason AS stop_reason, "
            "r.result AS result FROM results r "
            "JOIN sessions s ON s.session_id = r.session_id "
            "WHERE s.code=? "
            "ORDER BY s.finished_utc DESC NULLS LAST, s.started_utc DESC",
            (code,))
        return [
            {"session_id": r["session_id"], "finished_utc": r["finished_utc"],
             "n_questions": r["n_questions"], "stop_reason": r["stop_reason"],
             "result": json.loads(r["result"])}
            for r in rows
        ]

    # ── regimens (training protocol) ──────────────────────────────
    def create_regimen(self, regimen_id: str, code: str,
                        source_session_id: Optional[str], plan: dict) -> None:
        # Deactivate-then-insert is one transaction: there is never a moment
        # with zero (or two) active regimens for a participant.
        with self._connection() as conn:
            conn.execute(self._q("UPDATE regimens SET active=0 WHERE code=?"),
                         (code,))
            conn.execute(self._q(
                "INSERT INTO regimens(regimen_id, code, source_session_id, plan, "
                "active, created_utc) VALUES (?,?,?,?,1,?)"),
                (regimen_id, code, source_session_id, json.dumps(plan), utc_now()))

    def get_active_regimen(self, code: str) -> Optional[dict]:
        row = self._fetchone(
            "SELECT * FROM regimens WHERE code=? AND active=1 "
            "ORDER BY created_utc DESC LIMIT 1", (code,))
        if not row:
            return None
        row["plan"] = json.loads(row["plan"])
        return row

    # ── training sessions ─────────────────────────────────────────
    def create_training_session(self, training_id: str, code: str,
                                task_focus: Optional[str], *,
                                regimen_id: Optional[str] = None,
                                source_session_id: Optional[str] = None) -> None:
        self._write(
            "INSERT INTO training_sessions(training_id, code, task_focus, "
            "regimen_id, source_session_id, started_utc, status) "
            "VALUES (?,?,?,?,?,?, 'in_progress')",
            (training_id, code, task_focus, regimen_id, source_session_id, utc_now()))

    def finalize_training_session(self, training_id: str, code: str,
                                  n_items: Optional[int], summary: Optional[dict]) -> bool:
        with self._connection() as conn:
            cur = conn.execute(self._q(
                "UPDATE training_sessions SET status='complete', finished_utc=?, "
                "n_items=?, summary=? WHERE training_id=? AND code=?"),
                (utc_now(), n_items,
                 json.dumps(summary) if summary is not None else None,
                 training_id, code))
            return (cur.rowcount or 0) > 0 if hasattr(cur, "rowcount") else True

    def list_training_sessions(self, code: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM training_sessions WHERE code=? "
            "ORDER BY started_utc DESC", (code,))

    # ── parameter trajectories ────────────────────────────────────
    def append_trajectory_points(self, code: str, points: list[dict]) -> None:
        if not points:
            return
        with self._connection() as conn:   # all points in one transaction
            for p in points:
                conn.execute(self._q(
                    "INSERT INTO param_trajectories(code, task_k, phase, ell, "
                    "theta, sd, rt, ts, training_id, source_session_id, "
                    "seq_in_session, is_real) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"),
                    (code, int(p["taskK"]), str(p.get("phase", "train")),
                     p.get("ell"), p.get("theta"), p.get("sd"), p.get("rt"),
                     p.get("ts") or utc_now(),
                     p.get("trainingId"), p.get("sourceSessionId"),
                     p.get("seqInSession"), 1 if p.get("isReal") else 0))

    def _write_eval_trajectory_stmts(self, conn, code: str,
                                     source_session_id: str,
                                     points: list[dict]) -> None:
        conn.execute(self._q(
            "DELETE FROM param_trajectories "
            "WHERE source_session_id=? AND phase='eval'"),
            (source_session_id,))
        for p in points:
            conn.execute(self._q(
                "INSERT INTO param_trajectories(code, task_k, phase, ell, "
                "theta, sd, rt, ts, training_id, source_session_id, "
                "seq_in_session, is_real) VALUES (?,?,'eval',?,?,?,?,?,?,?,?,1)"),
                (code, int(p["taskK"]), p.get("ell"), p.get("theta"),
                 p.get("sd"), p.get("rt"), utc_now(), None,
                 source_session_id, 0))

    def write_eval_trajectory(self, code: str, source_session_id: str,
                              points: list[dict]) -> None:
        """Record the per-domain EVAL operating point for a certification
        session — one real (is_real=1) trajectory point per task. Idempotent on
        re-post: any prior eval rows for this session are replaced (the DELETE
        + INSERTs are one transaction). These seed the evolution charts;
        training/re-cert points append to the same series."""
        if not points:
            return
        with self._connection() as conn:
            self._write_eval_trajectory_stmts(conn, code, source_session_id, points)

    def store_result_finalized(self, session_id: str, code: str, result: dict,
                               stop_reason: Optional[str],
                               n_questions: Optional[int],
                               eval_points: list[dict]) -> None:
        """POST /api/results as ONE transaction: store the result blob,
        finalize the session, and replace its eval trajectory points together.
        The previous three separate units of work could be interrupted between
        statements, leaving a results row whose session was still
        'in_progress' (finished_utc NULL) — a half-state that shadowed the
        participant's true latest attempt on Postgres (NULLs sort first under
        ORDER BY … DESC)."""
        with self._connection() as conn:
            self._store_result_stmt(conn, session_id, result)
            self._finalize_session_stmt(conn, session_id, stop_reason, n_questions)
            if eval_points:
                self._write_eval_trajectory_stmts(conn, code, session_id, eval_points)

    def get_trajectories(self, code: str) -> list[dict]:
        # Dashboard reads REAL trainer output only (is_real=1). Synthetic/
        # quarantined rows (is_real=0) are excluded so no fabricated curve can
        # reach the UI before the L1 trainer ships.
        return self._fetchall(
            "SELECT task_k, phase, ell, theta, sd, rt, ts "
            "FROM param_trajectories WHERE code=? AND is_real=1 "
            "ORDER BY task_k, ts", (code,))

    def training_session_owner(self, training_id: str) -> Optional[str]:
        """The participant `code` that owns a training session (None if absent).
        The anti-tamper gate: only the owner may record its progress."""
        row = self._fetchone(
            "SELECT code FROM training_sessions WHERE training_id=?",
            (training_id,))
        return row["code"] if row else None

    def record_training_progress(self, code: str, training_id: str,
                                 points: list[dict]) -> None:
        """Server-authoritative per-trial training record (L1). For each point:
        (1) an idempotent `training_trials` exposure row (its seg_id then feeds
        `get_exposure_exclusion`, so retests never re-show a trained segment —
        D-INT-4/7), which since Phase L2 also carries the RESPONSE RECORD
        (pick, y_star, is_correct, feedback_shown, rt_ms, client timestamps) —
        the longitudinal training ledger the learning-engine dynamics refit
        consumes; and (2) a REAL `param_trajectories` row. `is_real`, `phase`,
        and `code` are set HERE from the authenticated session — never trusted
        from the client (anti-tamper). Caller must have verified ownership."""
        if not points:
            return
        cols = ("training_id, code, seg_id, task_k, shown_utc, pick, y_star, "
                "is_correct, feedback_shown, rt_ms, shown_client_utc, "
                "answered_client_utc, seq_in_session, mode, link, "
                "quality_flag")
        vals = "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
        if self._pg:
            excl = (f"INSERT INTO training_trials({cols}) VALUES ({vals}) "
                    "ON CONFLICT (training_id, seg_id) DO NOTHING")
        else:
            excl = (f"INSERT OR IGNORE INTO training_trials({cols}) "
                    f"VALUES ({vals})")
        with self._connection() as conn:
            # Idempotency for the per-answer checkpoint outbox: a point whose
            # (training_id, seq_in_session) already landed is skipped, so a
            # retry after an ambiguous network failure can never duplicate
            # trajectory rows. (Exposure rows were already ON CONFLICT-safe.)
            seen = {r["seq_in_session"] for r in conn.execute(self._q(
                "SELECT seq_in_session FROM param_trajectories "
                "WHERE training_id=?"), (training_id,)).fetchall()}
            # Burst detection (L4 data hygiene): a response is flagged when
            # it AND its immediate predecessor are both sub-500 ms — runs of
            # speed-clicking become mechanically queryable at refit time.
            rt_by_seq = {r["seq_in_session"]: r["rt_ms"] for r in conn.execute(
                self._q("SELECT seq_in_session, rt_ms FROM training_trials "
                        "WHERE training_id=?"), (training_id,)).fetchall()}
            for p in points:
                seq = p.get("seqInSession")
                if seq is not None and seq in seen:
                    continue
                if seq is not None:
                    seen.add(seq)   # in-batch duplicates too
                seg = p.get("segId")
                tk = int(p["taskK"])
                if seg is not None:
                    rt = p.get("rtMs", p.get("rt"))
                    if seq is not None and rt is not None:
                        rt_by_seq[seq] = rt
                    prev_rt = (rt_by_seq.get(seq - 1)
                               if seq is not None else None)
                    burst = (rt is not None and float(rt) < 500
                             and prev_rt is not None
                             and float(prev_rt) < 500)
                    conn.execute(self._q(excl), (
                        training_id, code, int(seg), tk, utc_now(),
                        p.get("pick"), p.get("yStar"),
                        (1 if p.get("isCorrect") else 0
                         if "isCorrect" in p else None),
                        p.get("feedbackShown"), rt,
                        p.get("shownClientUtc"), p.get("answeredClientUtc"),
                        seq, p.get("mode"), p.get("link"),
                        "burst" if burst else None))
                conn.execute(self._q(
                    "INSERT INTO param_trajectories(code, task_k, phase, ell, "
                    "theta, sd, rt, ts, training_id, seq_in_session, is_real) "
                    "VALUES (?,?,'train',?,?,?,?,?,?,?,1)"),
                    (code, tk, p.get("ell"), p.get("theta"), p.get("sd"),
                     p.get("rt"), utc_now(), training_id, seq))

    # ── engine-trainer metadata + retention state (Phase L4) ─────
    def set_training_engine_meta(self, training_id: str, meta: dict) -> None:
        """Sitting-level engine metadata (seeding, attainability report,
        config), written server-side at /training-engine/start."""
        self._write("UPDATE training_sessions SET engine_meta=? "
                    "WHERE training_id=?", (json.dumps(meta), training_id))

    def get_retention(self, code: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM training_retention WHERE code=?", (code,))

    def upsert_retention(self, code: str, domain: str,
                         next_due_utc: str, interval_s: float) -> None:
        if self._pg:
            sql = ("INSERT INTO training_retention(code, domain, "
                   "next_due_utc, interval_s) VALUES (?,?,?,?) "
                   "ON CONFLICT (code, domain) DO UPDATE SET "
                   "next_due_utc=EXCLUDED.next_due_utc, "
                   "interval_s=EXCLUDED.interval_s")
        else:
            sql = ("INSERT OR REPLACE INTO training_retention(code, domain, "
                   "next_due_utc, interval_s) VALUES (?,?,?,?)")
        self._write(sql, (code, domain, next_due_utc, float(interval_s)))

    # ── training pilot monitor (admin) ────────────────────────────
    def training_monitor(self) -> dict:
        """Aggregate safety + volume telemetry for the deployed trainer. Volume
        metrics are SQL; the SAFETY metric — confirmed false-graduation (the
        trainer declared a domain mastered, but the participant's next fresh
        retest did NOT pass it) — is cross-referenced in Python by linking each
        training session's final per-task ℓ against ℓ* and the subsequent cert
        verdict. See docs/LIVE_TRAINER.md for the active trainer architecture."""
        def _n(sql: str) -> int:
            row = self._fetchone(sql)
            return int((row["n"] if row else 0) or 0)

        learners = _n("SELECT COUNT(DISTINCT code) AS n FROM param_trajectories "
                      "WHERE is_real=1 AND phase='train'")
        started = _n("SELECT COUNT(*) AS n FROM training_sessions")
        completed = _n("SELECT COUNT(*) AS n FROM training_sessions "
                       "WHERE status='complete'")
        trials = _n("SELECT COUNT(*) AS n FROM param_trajectories "
                    "WHERE is_real=1 AND phase='train'")
        exposure = _n("SELECT COUNT(*) AS n FROM training_trials")
        per_domain = [
            {"taskK": int(r["task_k"]), "trials": int(r["n"])}
            for r in self._fetchall(
                "SELECT task_k, COUNT(*) AS n FROM param_trajectories "
                "WHERE is_real=1 AND phase='train' GROUP BY task_k ORDER BY task_k")]
        return {
            "learners": learners,
            "sessionsStarted": started,
            "sessionsCompleted": completed,
            "trainingTrials": trials,
            "exposureRows": exposure,
            "perDomainTrials": per_domain,
            **self._graduation_safety(),
        }

    def _graduation_safety(self) -> dict:
        """Graduation count + CONFIRMED false-graduation rate (the SAP primary
        safety endpoint). A (participant, domain) 'graduated' if its final
        trained ℓ ≥ ℓ*; it's a 'confirmed retest' if a fresh cert finished after
        that training; a 'false graduation' if that retest's verdict ≠ PASS."""
        def _per_task(result: dict, task: int, field: str):
            for p in (result or {}).get("perTask") or []:
                if isinstance(p, dict) and p.get("taskK") == task:
                    return p.get(field)
            return None

        rows = self._fetchall(
            "SELECT code, training_id, task_k, ell FROM param_trajectories "
            "WHERE is_real=1 AND phase='train' AND training_id IS NOT NULL "
            "AND ell IS NOT NULL ORDER BY ts")
        final_ell: dict = {}   # (code, training_id, task) -> final ℓ (ts-ordered, last wins)
        for r in rows:
            final_ell[(r["code"], r["training_id"], int(r["task_k"]))] = float(r["ell"])
        empty = {"graduatedDomains": 0, "confirmedRetests": 0,
                 "falseGraduations": 0, "falseGraduationRate": None}
        if not final_ell:
            return empty
        tfin = {t["training_id"]: t.get("finished_utc")
                for t in self._fetchall(
                    "SELECT training_id, finished_utc FROM training_sessions")}
        # Cert results per participant, ascending by finish time (ISO ⇒
        # sortable) — ONE query across all learner codes, not one per code
        # (the old per-code list_results_for_code loop was an N+1 that also
        # re-parsed every learner's full result blobs once per learner).
        codes = sorted({c for (c, _t, _k) in final_ell})
        ph = ",".join("?" * len(codes))
        results: dict = {}
        for r in self._fetchall(
                f"SELECT s.code AS code, s.finished_utc AS finished_utc, "
                f"r.result AS result FROM results r "
                f"JOIN sessions s ON s.session_id = r.session_id "
                f"WHERE s.code IN ({ph}) AND s.finished_utc IS NOT NULL "
                f"ORDER BY s.code, s.finished_utc",
                tuple(codes)):
            results.setdefault(r["code"], []).append(
                {"finished_utc": r["finished_utc"],
                 "result": json.loads(r["result"])})
        graduated = confirmed = false_grad = 0
        for (code, tid, task), ell in final_ell.items():
            rs = results.get(code, [])
            # ℓ* is bundle-wide: take it from any of the participant's results
            ellstar = next((v for r in rs
                            if (v := _per_task(r["result"], task, "ellStar")) is not None), None)
            if ellstar is None or ell < float(ellstar):
                continue
            graduated += 1
            tf = tfin.get(tid)
            if not tf:
                continue
            nxt = next((r for r in rs if r["finished_utc"] > tf), None)
            if nxt is None:
                continue
            verdict = _per_task(nxt["result"], task, "verdict")
            if verdict is None:
                continue
            confirmed += 1
            from .dashboard_logic import is_certified_verdict
            if not is_certified_verdict(verdict):
                false_grad += 1
        return {
            "graduatedDomains": graduated,
            "confirmedRetests": confirmed,
            "falseGraduations": false_grad,
            "falseGraduationRate": (false_grad / confirmed) if confirmed else None,
        }

    # ── cohorts ───────────────────────────────────────────────────
    def create_cohort(self, cohort_id: str, name: str, manager_code: str) -> None:
        """Create a cohort; the manager joins as an active member in the same
        transaction (their performance line is part of the pod)."""
        now = utc_now()
        with self._connection() as conn:
            conn.execute(self._q(
                "INSERT INTO cohorts(cohort_id, name, manager_code, created_utc) "
                "VALUES (?,?,?,?)"), (cohort_id, name, manager_code, now))
            conn.execute(self._q(
                "INSERT INTO cohort_members(cohort_id, code, status, "
                "invited_utc, joined_utc) VALUES (?,?,'active',?,?)"),
                (cohort_id, manager_code, now, now))

    def get_cohort(self, cohort_id: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM cohorts WHERE cohort_id=?", (cohort_id,))

    def cohorts_for(self, code: str) -> list[dict]:
        """Every cohort `code` belongs to (any status), with their member row."""
        return self._fetchall(
            "SELECT c.cohort_id, c.name, c.manager_code, c.created_utc, "
            "m.status, m.invited_utc, m.joined_utc "
            "FROM cohort_members m JOIN cohorts c ON c.cohort_id=m.cohort_id "
            "WHERE m.code=? ORDER BY c.created_utc, c.cohort_id", (code,))

    def count_cohorts_managed(self, code: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM cohorts WHERE manager_code=?", (code,))
        return int(row["n"]) if row else 0

    def cohort_membership(self, cohort_id: str, code: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM cohort_members WHERE cohort_id=? AND code=?",
            (cohort_id, code))

    def cohort_member_rows(self, cohort_id: str) -> list[dict]:
        """Member rows joined to their participant identity. Stable order =
        invite time (append-only), so client-side color slots don't reshuffle
        when someone new joins."""
        return self._fetchall(
            "SELECT m.code, m.status, m.invited_utc, m.joined_utc, "
            "p.public_id, p.display_name "
            "FROM cohort_members m JOIN participants p ON p.code=m.code "
            "WHERE m.cohort_id=? ORDER BY m.invited_utc, m.code", (cohort_id,))

    def count_cohort_members(self, cohort_id: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM cohort_members WHERE cohort_id=?",
            (cohort_id,))
        return int(row["n"]) if row else 0

    def active_member_counts(self, cohort_ids: list[str]) -> dict[str, int]:
        """cohort_id → active-member count, one grouped query for the whole
        list (the cohort list view needs only the counts, not member rows)."""
        if not cohort_ids:
            return {}
        ph = ",".join("?" * len(cohort_ids))
        rows = self._fetchall(
            f"SELECT cohort_id, COUNT(*) AS n FROM cohort_members "
            f"WHERE status='active' AND cohort_id IN ({ph}) GROUP BY cohort_id",
            tuple(cohort_ids))
        return {r["cohort_id"]: int(r["n"]) for r in rows}

    def invite_cohort_member(self, cohort_id: str, code: str) -> None:
        """Add a pending invite. The (cohort_id, code) PK surfaces a re-invite
        of an existing member/invitee as an IntegrityError → caller 409s."""
        self._write(
            "INSERT INTO cohort_members(cohort_id, code, status, invited_utc, "
            "joined_utc) VALUES (?,?,'invited',?,NULL)",
            (cohort_id, code, utc_now()))

    def accept_cohort_invite(self, cohort_id: str, code: str) -> bool:
        """Flip the caller's own invite to active. False when there is no
        pending invite (never touches other statuses/rows)."""
        with self._connection() as conn:
            cur = conn.execute(self._q(
                "UPDATE cohort_members SET status='active', joined_utc=? "
                "WHERE cohort_id=? AND code=? AND status='invited'"),
                (utc_now(), cohort_id, code))
            return cur.rowcount > 0

    # ── invite lifecycle ──────────────────────────────────────────
    def purge_expired_invites(self) -> None:
        """Drop never-accepted invites older than INVITE_TTL_DAYS: pending
        member invites (status='invited'; ACTIVE memberships are never
        touched) and email invites. Called lazily from the cohort read paths
        and from attach_email_invites, so no cron is needed: an invite that
        nobody ever looks at again simply stops existing the next time
        anything cohort-shaped is read.

        Throttled to once per minute per process (single-worker deploy, same
        assumption as the rate limiter): expiry has day granularity, and a
        dashboard entry shouldn't pay two DELETE statements per read. Benign
        race under the threadpool — two concurrent first-reads both purge."""
        now = time.monotonic()
        if now - self._last_invite_purge < 60.0:
            return
        self._last_invite_purge = now
        cutoff = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(time.time() - INVITE_TTL_DAYS * 86400))
        with self._connection() as conn:
            conn.execute(self._q(
                "DELETE FROM cohort_members WHERE status='invited' "
                "AND invited_utc < ?"), (cutoff,))
            conn.execute(self._q(
                "DELETE FROM cohort_email_invites WHERE invited_utc < ?"),
                (cutoff,))

    # ── email invites (addresses without an account yet) ─────────
    def put_cohort_email_invite(self, cohort_id: str, email: str) -> None:
        """Store (or refresh) a pending email invite. Upsert: re-inviting the
        same address just updates the timestamp."""
        if self._pg:
            sql = ("INSERT INTO cohort_email_invites(cohort_id, email, invited_utc) "
                   "VALUES (?,?,?) ON CONFLICT (cohort_id, email) "
                   "DO UPDATE SET invited_utc=EXCLUDED.invited_utc")
        else:
            sql = ("INSERT OR REPLACE INTO cohort_email_invites"
                   "(cohort_id, email, invited_utc) VALUES (?,?,?)")
        self._write(sql, (cohort_id, email, utc_now()))

    def delete_cohort_email_invite(self, cohort_id: str, email: str) -> None:
        self._write(
            "DELETE FROM cohort_email_invites WHERE cohort_id=? AND email=?",
            (cohort_id, email))

    def list_cohort_email_invites(self, cohort_id: str) -> list[dict]:
        return self._fetchall(
            "SELECT * FROM cohort_email_invites WHERE cohort_id=? "
            "ORDER BY invited_utc", (cohort_id,))

    def count_cohort_email_invites(self, cohort_id: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM cohort_email_invites WHERE cohort_id=?",
            (cohort_id,))
        return int(row["n"]) if row else 0

    def attach_email_invites(self, email: str, code: str) -> int:
        """Convert any pending email invites for `email` into normal 'invited'
        cohort memberships for the (freshly verified) account `code`, then
        drop the email rows. Called when an address finishes signup
        (routers/auth.py). Returns how many cohorts were attached."""
        self.purge_expired_invites()   # a stale invite must never attach
        attached = 0
        for row in self._fetchall(
                "SELECT cohort_id FROM cohort_email_invites WHERE email=?",
                (email,)):
            try:
                self.invite_cohort_member(row["cohort_id"], code)
                attached += 1
            except Exception as e:
                # Already a member/invitee of that cohort: nothing to attach.
                if "UNIQUE" not in str(e) and "unique" not in str(e) \
                        and "duplicate" not in str(e):
                    raise
            self._write(
                "DELETE FROM cohort_email_invites WHERE cohort_id=? AND email=?",
                (row["cohort_id"], email))
        return attached

    def remove_cohort_member(self, cohort_id: str, code: str) -> None:
        """Decline / leave / manager-remove are all the same row delete."""
        self._write(
            "DELETE FROM cohort_members WHERE cohort_id=? AND code=?",
            (cohort_id, code))

    def delete_cohort(self, cohort_id: str) -> None:
        with self._connection() as conn:
            conn.execute(self._q(
                "DELETE FROM cohort_members WHERE cohort_id=?"), (cohort_id,))
            conn.execute(self._q(
                "DELETE FROM cohort_email_invites WHERE cohort_id=?"), (cohort_id,))
            conn.execute(self._q(
                "DELETE FROM cohorts WHERE cohort_id=?"), (cohort_id,))

    def get_participant_by_public_id(self, public_id: str) -> Optional[dict]:
        return self._fetchone(
            "SELECT * FROM participants WHERE public_id=?", (public_id,))

    def eval_trajectories_for_codes_since(self, codes: list[str],
                                          since_ts: str) -> list[dict]:
        """REAL (is_real=1) trajectory points for a set of members within the
        lookback window — certification evals today; real training points join
        the same series the day the L1 trainer ships. Synthetic rows stay
        quarantined (same is_real=1 contract as get_trajectories)."""
        if not codes:
            return []
        ph = ",".join("?" * len(codes))
        return self._fetchall(
            f"SELECT code, task_k, phase, ell, theta, sd, ts "
            f"FROM param_trajectories "
            f"WHERE is_real=1 AND ts>=? AND code IN ({ph}) "
            f"ORDER BY code, task_k, ts",
            (since_ts, *codes))

    # ── activity heatmap ──────────────────────────────────────────
    def record_login_day(self, code: str, tz_offset_min: int = 0) -> None:
        """Mark today (the user's LOCAL day, per their tz offset) as a sign-in
        day for `code` (idempotent per day)."""
        day = _local_day(utc_now(), tz_offset_min)
        if self._pg:
            sql = ("INSERT INTO login_days(code, day) VALUES (?,?) "
                   "ON CONFLICT (code, day) DO NOTHING")
        else:
            sql = "INSERT OR IGNORE INTO login_days(code, day) VALUES (?,?)"
        self._write(sql, (code, day))

    # ── ripeness digest (see digest.py) ───────────────────────────
    def set_tz_offset(self, code: str, tz_offset_min: int) -> None:
        """Remember the device's getTimezoneOffset (refreshed at bootstrap) so
        the digest scheduler can aim letters at the learner's local morning."""
        self._write("UPDATE participants SET tz_offset_min=? WHERE code=?",
                    (int(tz_offset_min), code))

    def set_digest_opt_out(self, code: str, opt_out: bool) -> None:
        self._write("UPDATE participants SET digest_opt_out=? WHERE code=?",
                    (1 if opt_out else 0, code))

    def digest_recipients(self) -> list[dict]:
        """Participants eligible for the training digest: active account,
        verified + deliverable email, reminders not turned off, and an active
        regimen to train against. Cheap enough to run every scheduler tick."""
        return self._fetchall(
            "SELECT p.code AS code, p.email AS email, "
            "p.tz_offset_min AS tz_offset_min FROM participants p "
            "WHERE p.active=1 AND p.email IS NOT NULL "
            "AND p.email_verified_utc IS NOT NULL "
            "AND p.email_undeliverable_utc IS NULL "
            "AND COALESCE(p.digest_opt_out, 0)=0 "
            "AND EXISTS (SELECT 1 FROM regimens r "
            "            WHERE r.code=p.code AND r.active=1)")

    def claim_digest_day(self, code: str, day: str) -> bool:
        """Atomically claim (code, local day) in the digest ledger; True for
        exactly one caller per day. Claimed BEFORE the send: a failed send
        costs one day's letter instead of ever risking a double-send."""
        if self._pg:
            sql = ("INSERT INTO digest_log(code, day, sent_utc) VALUES (?,?,?) "
                   "ON CONFLICT (code, day) DO NOTHING")
        else:
            sql = "INSERT OR IGNORE INTO digest_log(code, day, sent_utc) VALUES (?,?,?)"
        with self._connection() as conn:
            cur = conn.execute(self._q(sql), (code, day, utc_now()))
            return cur.rowcount > 0

    # ── awards: domain badges + milestones (awards.py) ────────────
    def award_badge(self, code: str, key: str, label: str,
                    detail: Optional[str] = None) -> bool:
        """Award the domain badge unless it is currently held. A re-earn after
        a loss APPENDS a row, preserving the earned/lost history."""
        held = self._fetchone(
            "SELECT award_id AS a FROM awards WHERE code=? AND kind='badge' "
            "AND key=? AND revoked_utc IS NULL LIMIT 1", (code, key))
        if held:
            return False
        self._write(
            "INSERT INTO awards(award_id, code, kind, key, label, detail, "
            "awarded_utc) VALUES (?,?,'badge',?,?,?,?)",
            (secrets.token_hex(16), code, key, label, detail, utc_now()))
        return True

    def revoke_badge(self, code: str, key: str) -> bool:
        """Mark the currently-held badge for `key` as lost (exam
        underperformance). No-op when none is held."""
        row = self._fetchone(
            "SELECT award_id AS a FROM awards WHERE code=? AND kind='badge' "
            "AND key=? AND revoked_utc IS NULL LIMIT 1", (code, key))
        if not row:
            return False
        self._write("UPDATE awards SET revoked_utc=? WHERE award_id=?",
                    (utc_now(), row["a"]))
        return True

    def award_milestone(self, code: str, key: str, label: str,
                        detail: Optional[str] = None) -> bool:
        """Once-ever recognition row; False when `key` was already awarded."""
        if self._fetchone(
                "SELECT 1 AS x FROM awards WHERE code=? AND kind='milestone' "
                "AND key=? LIMIT 1", (code, key)):
            return False
        self._write(
            "INSERT INTO awards(award_id, code, kind, key, label, detail, "
            "awarded_utc) VALUES (?,?,'milestone',?,?,?,?)",
            (secrets.token_hex(16), code, key, label, detail, utc_now()))
        return True

    def list_awards(self, code: str) -> list[dict]:
        return self._fetchall(
            "SELECT award_id, kind, key, label, detail, awarded_utc, "
            "revoked_utc, acknowledged_utc FROM awards WHERE code=? "
            "ORDER BY awarded_utc DESC, award_id DESC", (code,))

    def pending_milestones(self, code: str) -> list[dict]:
        """Milestones not yet announced (the dashboard banner's feed)."""
        return self._fetchall(
            "SELECT award_id, key, label, detail, awarded_utc FROM awards "
            "WHERE code=? AND kind='milestone' AND acknowledged_utc IS NULL "
            "ORDER BY awarded_utc, award_id", (code,))

    def acknowledge_award(self, code: str, award_id: str) -> bool:
        """Owner-scoped banner dismissal; True exactly once per award."""
        with self._connection() as conn:
            cur = conn.execute(self._q(
                "UPDATE awards SET acknowledged_utc=? WHERE award_id=? "
                "AND code=? AND acknowledged_utc IS NULL"),
                (utc_now(), award_id, code))
            return cur.rowcount > 0

    def count_completed_results(self, code: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM results r "
            "JOIN sessions s ON s.session_id = r.session_id WHERE s.code=?",
            (code,))
        return int(row["n"]) if row else 0

    def count_completed_trainings(self, code: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS n FROM training_sessions "
            "WHERE code=? AND status='complete'", (code,))
        return int(row["n"]) if row else 0

    def activity_levels(self, code: str, tz_offset_min: int = 0) -> dict[str, int]:
        """Per-day activity level (LOCAL YYYY-MM-DD → level) for the heatmap:
        1 = signed in, 2 = certification test, 3 = training completed. Highest
        level wins. Cert/training UTC timestamps are shifted to the user's local
        day; login_days are already stored local."""
        levels: dict[str, int] = {}

        def bump(day: Optional[str], lvl: int) -> None:
            if day and levels.get(day, 0) < lvl:
                levels[day] = lvl

        with self._connection() as conn:
            for r in conn.execute(self._q(
                    "SELECT day FROM login_days WHERE code=?"), (code,)).fetchall():
                bump(dict(r)["day"], 1)
            for r in conn.execute(self._q(
                    "SELECT s.finished_utc AS f FROM results r "
                    "JOIN sessions s ON s.session_id=r.session_id WHERE s.code=?"),
                    (code,)).fetchall():
                bump(_local_day(dict(r)["f"], tz_offset_min), 2)
            for r in conn.execute(self._q(
                    "SELECT finished_utc AS f FROM training_sessions "
                    "WHERE code=? AND status='complete'"), (code,)).fetchall():
                bump(_local_day(dict(r)["f"], tz_offset_min), 3)
        return levels
