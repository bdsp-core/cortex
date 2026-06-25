"""Backend persistence — same API, two backends (SQLite for dev, Postgres for prod).

Four tables (schema is SQL-92, identical on both engines):
  participants  credential rows (code → PBKDF2 hash). Created by admin CLI.
  sessions      one row per started test; participant demographics + status.
  trials        append-only per-question checkpoints (crash-safety).
  results       one row per finished test; the full serialized session JSON.

`CORTEX_DB` selects the backend:
  unset / a filesystem path  → SQLite (WAL).
  "postgres://..." / "postgresql://..."  → Postgres via psycopg3.

The methods (add_participant, create_session, upsert_trial, store_result, …)
are unchanged from the original SQLite-only version, so app.py / admin.py /
test_server.py don't need touching. The only places that branch on backend are
(a) connection setup, (b) placeholder syntax (? vs %s), (c) the two upserts
that used "INSERT OR REPLACE" (Postgres needs ON CONFLICT … DO UPDATE).
"""
from __future__ import annotations

import calendar
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

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
        PRIMARY KEY (training_id, seg_id)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_training_trials_code ON training_trials(code)",
    # The email-unique index is created in _migrate_participants AFTER the
    # ALTER TABLE that ensures the column exists (an older schema may not
    # have it yet on an existing DB).
]


# Columns added to participants after the original schema. Idempotent
# ALTERs run at startup so an existing database upgrades in place.
_PARTICIPANTS_MIGRATION_COLUMNS = [
    ("email",             "TEXT"),
    ("display_name",      "TEXT"),
    ("signup_ip",         "TEXT"),
    ("email_verified_utc", "TEXT"),
    ("auth_provider",     "TEXT"),   # NULL/'local' for password accounts, 'google' for OAuth
    ("google_sub",        "TEXT"),   # Google account id ('sub' claim); unique when set
    ("signup_expertise",  "TEXT"),   # self-reported role from the signup form (Phase O1)
    ("profile",           "TEXT"),   # JSON demographic/clinical profile collected at signup, editable in Settings
]

# Columns added to the learning tables after their original schema (Phase O2),
# so an existing DB upgrades in place. Mirror the CREATE TABLE additions above.
_PARAM_TRAJ_MIGRATION_COLUMNS = [
    ("training_id",       "TEXT"),
    ("source_session_id", "TEXT"),
    ("seq_in_session",    "INTEGER"),
    ("is_real",           "INTEGER NOT NULL DEFAULT 0"),
]
_TRAINING_SESSIONS_MIGRATION_COLUMNS = [
    ("regimen_id",        "TEXT"),
    ("source_session_id", "TEXT"),
]
# Provenance stamp (O3): which bundle/bank version produced a session, so a
# 35k/v15 session stays attributable + reproducible after a bundle change.
_SESSIONS_MIGRATION_COLUMNS = [
    ("bundle_version",    "TEXT"),
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
        self._pg = _is_pg_url(raw)
        self._lock = threading.Lock()

        if self._pg:
            # psycopg3 — opt-in dep. Lazy-imported so SQLite-only dev installs
            # don't need it.
            self._raw = raw                  # retained so a severed connection
            self._conn = self._pg_connect()  # can be re-established (see _reconnect)
            self.path = None
        else:
            self.path = Path(raw)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

        with self._lock:
            for stmt in _SCHEMA_STATEMENTS:
                self._exec(stmt)
            self._migrate_participants()
            self._add_missing_columns("param_trajectories", _PARAM_TRAJ_MIGRATION_COLUMNS)
            self._add_missing_columns("training_sessions", _TRAINING_SESSIONS_MIGRATION_COLUMNS)
            self._add_missing_columns("sessions", _SESSIONS_MIGRATION_COLUMNS)
            # Quarantine backfill: every pre-existing trajectory row is synthetic
            # (the only writer before the L1 trainer was the removed Shell.tsx
            # placeholder). NOT NULL DEFAULT 0 already fills new column; this is
            # belt-and-suspenders for any backend that left it NULL.
            self._exec("UPDATE param_trajectories SET is_real=0 WHERE is_real IS NULL")
            self._conn.commit()

    def _migrate_participants(self) -> None:
        """Add columns to participants that didn't exist in earlier schemas
        (email, display_name, signup_ip). Idempotent on both engines."""
        if self._pg:
            for col, typ in _PARTICIPANTS_MIGRATION_COLUMNS:
                self._exec(f"ALTER TABLE participants ADD COLUMN IF NOT EXISTS {col} {typ}")
        else:
            cur = self._exec("PRAGMA table_info(participants)")
            existing = {dict(r)["name"] for r in cur.fetchall()}
            for col, typ in _PARTICIPANTS_MIGRATION_COLUMNS:
                if col not in existing:
                    self._exec(f"ALTER TABLE participants ADD COLUMN {col} {typ}")
        # Now that the email column is guaranteed to exist, the unique-when-
        # set index is safe to create on both engines.
        self._exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_email "
            "ON participants(email) WHERE email IS NOT NULL"
        )
        # One CORTEX account per Google identity.
        self._exec(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_participants_google_sub "
            "ON participants(google_sub) WHERE google_sub IS NOT NULL"
        )

    def _add_missing_columns(self, table: str, columns: list[tuple[str, str]]) -> None:
        """Idempotent additive migration for `table` on both backends — the
        generic form of _migrate_participants, used for the learning tables."""
        if self._pg:
            for col, typ in columns:
                self._exec(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
        else:
            cur = self._exec(f"PRAGMA table_info({table})")
            existing = {dict(r)["name"] for r in cur.fetchall()}
            for col, typ in columns:
                if col not in existing:
                    self._exec(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    # ── low-level helpers (backend-aware) ─────────────────────────
    def _q(self, sql: str) -> str:
        """Translate `?` placeholders to `%s` when on Postgres."""
        return sql.replace("?", "%s") if self._pg else sql

    def _pg_connect(self):
        """Open the shared psycopg3 connection (autocommit=False, dict rows).
        Single source of truth for both the initial connect and _reconnect, so
        the connection parameters can't drift between them."""
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(self._raw, autocommit=False, row_factory=dict_row)

    def _reconnect(self) -> None:
        """Re-establish the shared PG connection after it was severed (Postgres
        restart, terminated backend, dropped socket). The schema/migrations
        already ran against the database, so only the connection is recreated.
        The caller holds self._lock, so this is serialized with all other DB
        access — the single-connection model is preserved, just made durable."""
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = self._pg_connect()

    def _pg_conn_dead(self) -> bool:
        """True when the shared PG connection can no longer be used — the server
        closed it (restart / pg_terminate_backend) or the socket broke. Lets a
        *severed* connection (→ reconnect) be told apart from a merely *aborted*
        transaction on a live connection (→ rollback)."""
        try:
            return bool(self._conn.closed) or bool(self._conn.broken)
        except Exception:
            return True   # can't tell → assume dead and reconnect (a fresh
                          # connection is always usable; never loops in prod
                          # because a real psycopg conn always exposes both flags)

    def _pg_tx_idle(self) -> bool:
        """True if the PG connection has no open transaction right now, i.e. a
        statement that fails can be safely retried on a fresh connection without
        losing earlier work. libpq's transaction status is read WITHOUT any I/O,
        so it reflects the last known state even for a connection the server
        killed while idle. Compared numerically (PQTRANS_IDLE == 0) so the helper
        doesn't require psycopg to be importable, e.g. in SQLite-only test runs."""
        try:
            return int(self._conn.info.transaction_status) == 0   # 0 == PQTRANS_IDLE
        except Exception:
            return False   # unknown → assume mid-transaction → don't retry

    def _exec(self, sql: str, params: tuple = ()) -> Any:
        """Execute one statement, return a cursor (or sqlite3's conn.execute).
        Wraps the placeholder translation so callers stay backend-agnostic.

        On Postgres the single shared connection (autocommit=False) has two
        independent failure modes this guards against:

        1. **Statement error on a live connection** (deadlock victim, statement
           timeout, an uncaught UNIQUE violation, …) aborts the transaction. If we
           don't roll back, EVERY later statement raises InFailedSqlTransaction
           until the process restarts — a single transient error becomes a total
           login outage (the 2026-06-21 incident). So roll back before re-raising.

        2. **The connection itself is severed** — Postgres restarted (an
           unattended-upgrade of a libpq dependency bounces the service), the
           backend was terminated, or the socket dropped. The held connection is
           dead and, with no recovery, every later request fails forever until a
           manual restart (the 2026-06-24 incident: an idle connection killed by
           the Jun-23 06:21 Postgres restart surfaced as AdminShutdown on the next
           login and stayed down ~all day). So detect the dead connection,
           reconnect, and — only when we were at a clean transaction boundary —
           retry the statement once. The app self-heals instead of going dark.

        (SQLite neither poisons later statements nor gets bounced this way; the
        rollback/reconnect machinery is Postgres-only, leaving SQLite as-is.)
        """
        if self._pg:
            idle_before = self._pg_tx_idle()   # retry is safe only at a txn boundary
            try:
                cur = self._conn.cursor()
                cur.execute(self._q(sql), params)
                return cur
            except Exception:
                if self._pg_conn_dead():
                    # Connection severed. Reconnect so the app recovers; retry this
                    # statement only if nothing earlier in this operation's
                    # transaction was lost with it (a bare read, or the first
                    # statement of a write — the common case, incl. a connection
                    # killed while idle between requests). If we were mid
                    # multi-statement write, re-raise instead of persisting a
                    # partial write; the next request starts clean on the new conn.
                    self._reconnect()
                    if idle_before:
                        cur = self._conn.cursor()
                        cur.execute(self._q(sql), params)
                        return cur
                    raise
                # Live connection, aborted transaction — roll back so later
                # statements aren't poisoned. If the rollback itself finds the
                # connection gone, reconnect before re-raising.
                try:
                    self._conn.rollback()
                except Exception:
                    self._reconnect()
                raise
        return self._conn.execute(self._q(sql), params)

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        cur = self._exec(sql, params)
        row = cur.fetchone()
        if self._pg and cur:
            cur.close()
            self._end_read()
        return dict(row) if row is not None else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._exec(sql, params)
        rows = cur.fetchall()
        if self._pg and cur:
            cur.close()
            self._end_read()
        return [dict(r) for r in rows]

    def _end_read(self) -> None:
        """Commit the implicit transaction a read opened. On Postgres the single
        shared connection runs autocommit=False (db.py __init__), so even a bare
        SELECT begins a transaction. The write helpers commit; the read helpers
        historically did NOT — so after any read the connection sat
        `idle in transaction` indefinitely, holding AccessShare on the queried
        table(s). That AccessShare blocks the AccessExclusive a *second*
        Database() instance's boot migrations take (`ALTER TABLE … ADD COLUMN IF
        NOT EXISTS`, `CREATE UNIQUE INDEX …`) — which is exactly how the nightly
        backup's `api.admin export-sessions` hung for hours (2026-06-22) — and it
        also pins the VACUUM xmin horizon, bloating the DB. Committing after each
        read returns the shared connection to a clean idle state. (This is the
        read-side complement to the error-path rollback in `_exec`, the
        2026-06-21 fix.) SQLite opens no transaction for a SELECT, so the read
        helpers skip this — their write paths already commit."""
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── participants ──────────────────────────────────────────────
    def add_participant(self, code: str, password_hash: str, label: str = "") -> None:
        with self._lock:
            self._exec(
                "INSERT INTO participants(code, password_hash, label, created_utc) "
                "VALUES (?,?,?,?)",
                (code, password_hash, label, utc_now()),
            )
            self._conn.commit()

    def get_participant(self, code: str) -> Optional[dict]:
        with self._lock:
            return self._fetchone(
                "SELECT * FROM participants WHERE code=?", (code,))

    def list_participants(self) -> list[dict]:
        with self._lock:
            return self._fetchall(
                "SELECT code,label,active,created_utc FROM participants "
                "ORDER BY created_utc")

    def set_participant_active(self, code: str, active: bool) -> None:
        with self._lock:
            self._exec("UPDATE participants SET active=? WHERE code=?",
                       (1 if active else 0, code))
            self._conn.commit()

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
        Settings)."""
        with self._lock:
            self._exec(
                "INSERT INTO participants(code, password_hash, email, "
                "display_name, signup_ip, signup_expertise, profile, created_utc) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (code, password_hash, email, display_name, signup_ip,
                 signup_expertise, profile, utc_now()),
            )
            self._conn.commit()

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
        with self._lock:
            self._exec(f"UPDATE participants SET {', '.join(sets)} WHERE code=?", tuple(params))
            self._conn.commit()

    def update_email(self, code: str, new_email: str) -> None:
        """Change an account's email. The UNIQUE-when-set index surfaces a
        collision as a backend IntegrityError the caller maps to a 409."""
        with self._lock:
            self._exec("UPDATE participants SET email=? WHERE code=?", (new_email, code))
            self._conn.commit()

    def get_participant_by_email(self, email: str) -> Optional[dict]:
        with self._lock:
            return self._fetchone(
                "SELECT * FROM participants WHERE email=?", (email,))

    def mark_email_verified(self, code: str) -> None:
        with self._lock:
            self._exec(
                "UPDATE participants SET email_verified_utc=? WHERE code=?",
                (utc_now(), code))
            self._conn.commit()

    def set_password_hash(self, code: str, password_hash: str) -> None:
        with self._lock:
            self._exec(
                "UPDATE participants SET password_hash=? WHERE code=?",
                (password_hash, code))
            self._conn.commit()

    # ── Google OAuth accounts ─────────────────────────────────────
    def get_participant_by_google_sub(self, sub: str) -> Optional[dict]:
        with self._lock:
            return self._fetchone(
                "SELECT * FROM participants WHERE google_sub=?", (sub,))

    def register_oauth_participant(self, *, code: str, email: str, display_name: str,
                                    google_sub: str, signup_ip: Optional[str] = None) -> None:
        """Insert a new Google-authenticated account. password_hash holds a
        non-PBKDF2 sentinel (so password login is impossible); the email is
        verified by Google, so email_verified_utc is set immediately."""
        with self._lock:
            self._exec(
                "INSERT INTO participants(code, password_hash, email, display_name, "
                "signup_ip, created_utc, auth_provider, google_sub, email_verified_utc) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (code, "google-oauth:no-password", email, display_name, signup_ip,
                 utc_now(), "google", google_sub, utc_now()))
            self._conn.commit()

    def link_google_sub(self, code: str, sub: str) -> None:
        """Attach a Google identity to an existing (e.g. password) account."""
        with self._lock:
            self._exec(
                "UPDATE participants SET google_sub=? WHERE code=?", (sub, code))
            self._conn.commit()

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
        with self._lock:
            self._exec(sql, (participant_code, purpose, code_hash,
                             utc_now(), expires_utc))
            self._conn.commit()

    def get_auth_code(self, participant_code: str, purpose: str) -> Optional[dict]:
        with self._lock:
            return self._fetchone(
                "SELECT * FROM auth_codes WHERE participant_code=? AND purpose=?",
                (participant_code, purpose))

    def increment_auth_attempts(self, participant_code: str, purpose: str) -> None:
        with self._lock:
            self._exec(
                "UPDATE auth_codes SET attempts=attempts+1 "
                "WHERE participant_code=? AND purpose=?",
                (participant_code, purpose))
            self._conn.commit()

    def consume_auth_code(self, participant_code: str, purpose: str) -> None:
        with self._lock:
            self._exec(
                "UPDATE auth_codes SET consumed_utc=? "
                "WHERE participant_code=? AND purpose=?",
                (utc_now(), participant_code, purpose))
            self._conn.commit()

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
        with self._lock:
            self._exec(sql, (code, consent_type, consent_version,
                             irb_protocol_id, utc_now(), consent_ip))
            self._conn.commit()

    def withdraw_consent(self, code: str, consent_type: Optional[str] = None) -> int:
        """Mark consent withdrawn (all types, or one). Honoured at export time
        (future releases only — already-released DOIs are irrevocable)."""
        with self._lock:
            if consent_type:
                cur = self._exec(
                    "UPDATE consent_events SET withdrawn_utc=? "
                    "WHERE code=? AND consent_type=?",
                    (utc_now(), code, consent_type))
            else:
                cur = self._exec(
                    "UPDATE consent_events SET withdrawn_utc=? WHERE code=?",
                    (utc_now(), code))
            self._conn.commit()
            return (cur.rowcount or 0) if hasattr(cur, "rowcount") else 0

    def get_consent_events(self, code: str) -> list[dict]:
        with self._lock:
            return self._fetchall(
                "SELECT * FROM consent_events WHERE code=? "
                "ORDER BY consent_type, consent_version", (code,))

    # ── sessions ──────────────────────────────────────────────────
    def create_session(self, session_id: str, code: str, participant: dict,
                        sample_seed: Optional[int],
                        bundle_version: Optional[str] = None) -> None:
        with self._lock:
            self._exec(
                "INSERT INTO sessions(session_id, code, participant, sample_seed, "
                "bundle_version, started_utc, status) VALUES (?,?,?,?,?,?, 'in_progress')",
                (session_id, code, json.dumps(participant), sample_seed,
                 bundle_version, utc_now()),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._lock:
            return self._fetchone(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,))

    def finalize_session(self, session_id: str, stop_reason: Optional[str],
                         n_questions: Optional[int]) -> None:
        with self._lock:
            self._exec(
                "UPDATE sessions SET status='complete', finished_utc=?, "
                "stop_reason=?, n_questions=? WHERE session_id=?",
                (utc_now(), stop_reason, n_questions, session_id),
            )
            self._conn.commit()

    # ── trials ────────────────────────────────────────────────────
    def upsert_trial(self, session_id: str, trial: dict) -> None:
        # Upsert syntax differs: SQLite's INSERT OR REPLACE vs Postgres's
        # ON CONFLICT … DO UPDATE on the (session_id, trial_index) PK.
        if self._pg:
            sql = ("INSERT INTO trials(session_id, trial_index, seg_id, "
                   "task_k, pick, is_correct, reaction_ms, diag, received_utc) "
                   "VALUES (?,?,?,?,?,?,?,?,?) "
                   "ON CONFLICT (session_id, trial_index) DO UPDATE SET "
                   "seg_id=EXCLUDED.seg_id, task_k=EXCLUDED.task_k, "
                   "pick=EXCLUDED.pick, is_correct=EXCLUDED.is_correct, "
                   "reaction_ms=EXCLUDED.reaction_ms, diag=EXCLUDED.diag, "
                   "received_utc=EXCLUDED.received_utc")
        else:
            sql = ("INSERT OR REPLACE INTO trials(session_id, trial_index, "
                   "seg_id, task_k, pick, is_correct, reaction_ms, diag, "
                   "received_utc) VALUES (?,?,?,?,?,?,?,?,?)")
        with self._lock:
            self._exec(sql, (
                session_id,
                int(trial.get("trialIndex", trial.get("trial_index", 0))),
                trial.get("segId", trial.get("seg_id")),
                trial.get("taskK", trial.get("task_k")),
                trial.get("pick"),
                1 if trial.get("isCorrect") else 0 if "isCorrect" in trial else None,
                trial.get("reactionMs", trial.get("reaction_ms")),
                json.dumps(trial.get("diag")) if trial.get("diag") is not None else None,
                utc_now(),
            ))
            self._conn.commit()

    def session_trials(self, session_id: str) -> list[dict]:
        with self._lock:
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
        with self._lock:
            # (sitting table, its id column, the per-question table keyed by it)
            for sess_tbl, id_col, trial_tbl in (
                ("sessions", "session_id", "trials"),
                ("training_sessions", "training_id", "training_trials"),
            ):
                sittings = self._fetchall(
                    f"SELECT {id_col} AS sid, started_utc AS started_utc "
                    f"FROM {sess_tbl} WHERE code=? ORDER BY started_utc DESC",
                    (code,))
                keep = [r["sid"] for i, r in enumerate(sittings)
                        if i < session_window or (r["started_utc"] or "") >= cutoff]
                if not keep:
                    continue
                qmarks = ",".join("?" for _ in keep)
                rows = self._fetchall(
                    f"SELECT DISTINCT seg_id FROM {trial_tbl} "
                    f"WHERE seg_id IS NOT NULL AND {id_col} IN ({qmarks})",
                    tuple(keep))
                excl.update(int(r["seg_id"]) for r in rows
                            if r["seg_id"] is not None)
        return excl

    # ── results ───────────────────────────────────────────────────
    def store_result(self, session_id: str, result: dict) -> None:
        if self._pg:
            sql = ("INSERT INTO results(session_id, result, received_utc) "
                   "VALUES (?,?,?) "
                   "ON CONFLICT (session_id) DO UPDATE SET "
                   "result=EXCLUDED.result, received_utc=EXCLUDED.received_utc")
        else:
            sql = ("INSERT OR REPLACE INTO results(session_id, result, "
                   "received_utc) VALUES (?,?,?)")
        with self._lock:
            self._exec(sql, (session_id, json.dumps(result), utc_now()))
            self._conn.commit()

    def get_result(self, session_id: str) -> Optional[dict]:
        with self._lock:
            row = self._fetchone(
                "SELECT result FROM results WHERE session_id=?", (session_id,))
        return json.loads(row["result"]) if row else None

    def all_sessions(self) -> list[dict]:
        with self._lock:
            return self._fetchall(
                "SELECT * FROM sessions ORDER BY started_utc")

    # ── dashboard: latest certification result for a participant ──
    def latest_result_for_code(self, code: str) -> Optional[dict]:
        """The most recent completed-session result JSON for `code`, or None."""
        with self._lock:
            row = self._fetchone(
                "SELECT r.result AS result FROM results r "
                "JOIN sessions s ON s.session_id = r.session_id "
                "WHERE s.code=? ORDER BY s.finished_utc DESC, s.started_utc DESC "
                "LIMIT 1",
                (code,))
        return json.loads(row["result"]) if row else None

    def list_results_for_code(self, code: str) -> list[dict]:
        """All COMPLETED sessions for `code` joined with their result JSON,
        newest first. Mirrors latest_result_for_code's JOIN but returns every
        attempt with its session metadata. Scoped strictly by sessions.code."""
        with self._lock:
            rows = self._fetchall(
                "SELECT s.session_id AS session_id, s.finished_utc AS finished_utc, "
                "s.n_questions AS n_questions, s.stop_reason AS stop_reason, "
                "r.result AS result FROM results r "
                "JOIN sessions s ON s.session_id = r.session_id "
                "WHERE s.code=? ORDER BY s.finished_utc DESC, s.started_utc DESC",
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
        with self._lock:
            self._exec("UPDATE regimens SET active=0 WHERE code=?", (code,))
            self._exec(
                "INSERT INTO regimens(regimen_id, code, source_session_id, plan, "
                "active, created_utc) VALUES (?,?,?,?,1,?)",
                (regimen_id, code, source_session_id, json.dumps(plan), utc_now()))
            self._conn.commit()

    def get_active_regimen(self, code: str) -> Optional[dict]:
        with self._lock:
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
        with self._lock:
            self._exec(
                "INSERT INTO training_sessions(training_id, code, task_focus, "
                "regimen_id, source_session_id, started_utc, status) "
                "VALUES (?,?,?,?,?,?, 'in_progress')",
                (training_id, code, task_focus, regimen_id, source_session_id, utc_now()))
            self._conn.commit()

    def finalize_training_session(self, training_id: str, code: str,
                                  n_items: Optional[int], summary: Optional[dict]) -> bool:
        with self._lock:
            cur = self._exec(
                "UPDATE training_sessions SET status='complete', finished_utc=?, "
                "n_items=?, summary=? WHERE training_id=? AND code=?",
                (utc_now(), n_items,
                 json.dumps(summary) if summary is not None else None,
                 training_id, code))
            self._conn.commit()
            return (cur.rowcount or 0) > 0 if hasattr(cur, "rowcount") else True

    def list_training_sessions(self, code: str) -> list[dict]:
        with self._lock:
            return self._fetchall(
                "SELECT * FROM training_sessions WHERE code=? "
                "ORDER BY started_utc DESC", (code,))

    # ── parameter trajectories ────────────────────────────────────
    def append_trajectory_points(self, code: str, points: list[dict]) -> None:
        if not points:
            return
        with self._lock:
            for p in points:
                self._exec(
                    "INSERT INTO param_trajectories(code, task_k, phase, ell, "
                    "theta, sd, rt, ts, training_id, source_session_id, "
                    "seq_in_session, is_real) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (code, int(p["taskK"]), str(p.get("phase", "train")),
                     p.get("ell"), p.get("theta"), p.get("sd"), p.get("rt"),
                     p.get("ts") or utc_now(),
                     p.get("trainingId"), p.get("sourceSessionId"),
                     p.get("seqInSession"), 1 if p.get("isReal") else 0))
            self._conn.commit()

    def write_eval_trajectory(self, code: str, source_session_id: str,
                              points: list[dict]) -> None:
        """Record the per-domain EVAL operating point for a certification
        session — one real (is_real=1) trajectory point per task. Idempotent on
        re-post: any prior eval rows for this session are replaced. These seed
        the evolution charts; training/re-cert points append to the same series."""
        if not points:
            return
        with self._lock:
            self._exec("DELETE FROM param_trajectories "
                       "WHERE source_session_id=? AND phase='eval'",
                       (source_session_id,))
            for p in points:
                self._exec(
                    "INSERT INTO param_trajectories(code, task_k, phase, ell, "
                    "theta, sd, rt, ts, training_id, source_session_id, "
                    "seq_in_session, is_real) VALUES (?,?,'eval',?,?,?,?,?,?,?,?,1)",
                    (code, int(p["taskK"]), p.get("ell"), p.get("theta"),
                     p.get("sd"), p.get("rt"), utc_now(), None,
                     source_session_id, 0))
            self._conn.commit()

    def get_trajectories(self, code: str) -> list[dict]:
        # Dashboard reads REAL trainer output only (is_real=1). Synthetic/
        # quarantined rows (is_real=0) are excluded so no fabricated curve can
        # reach the UI before the L1 trainer ships.
        with self._lock:
            return self._fetchall(
                "SELECT task_k, phase, ell, theta, sd, rt, ts "
                "FROM param_trajectories WHERE code=? AND is_real=1 "
                "ORDER BY task_k, ts", (code,))

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
        with self._lock:
            self._exec(sql, (code, day))
            self._conn.commit()

    def activity_levels(self, code: str, tz_offset_min: int = 0) -> dict[str, int]:
        """Per-day activity level (LOCAL YYYY-MM-DD → level) for the heatmap:
        1 = signed in, 2 = certification test, 3 = training completed. Highest
        level wins. Cert/training UTC timestamps are shifted to the user's local
        day; login_days are already stored local."""
        levels: dict[str, int] = {}

        def bump(day: Optional[str], lvl: int) -> None:
            if day and levels.get(day, 0) < lvl:
                levels[day] = lvl

        with self._lock:
            for r in self._fetchall("SELECT day FROM login_days WHERE code=?", (code,)):
                bump(r["day"], 1)
            for r in self._fetchall(
                "SELECT s.finished_utc AS f FROM results r "
                "JOIN sessions s ON s.session_id=r.session_id WHERE s.code=?", (code,)):
                bump(_local_day(r["f"], tz_offset_min), 2)
            for r in self._fetchall(
                "SELECT finished_utc AS f FROM training_sessions "
                "WHERE code=? AND status='complete'", (code,)):
                bump(_local_day(r["f"], tz_offset_min), 3)
        return levels
