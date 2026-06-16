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
        training_id   TEXT PRIMARY KEY,
        code          TEXT NOT NULL,
        task_focus    TEXT,
        started_utc   TEXT NOT NULL,
        finished_utc  TEXT,
        status        TEXT NOT NULL DEFAULT 'in_progress',
        n_items       INTEGER,
        summary       TEXT
    )""",
    # Append-only time series of (task, ℓ, θ, sd, rt) for the evolution charts.
    """CREATE TABLE IF NOT EXISTS param_trajectories (
        code        TEXT NOT NULL,
        task_k      INTEGER NOT NULL,
        phase       TEXT NOT NULL,
        ell         REAL,
        theta       REAL,
        sd          REAL,
        rt          REAL,
        ts          TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_regimens_code ON regimens(code)",
    "CREATE INDEX IF NOT EXISTS idx_training_code ON training_sessions(code)",
    "CREATE INDEX IF NOT EXISTS idx_param_traj_code ON param_trajectories(code)",
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
]


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
            import psycopg
            from psycopg.rows import dict_row
            self._conn = psycopg.connect(raw, autocommit=False, row_factory=dict_row)
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

    # ── low-level helpers (backend-aware) ─────────────────────────
    def _q(self, sql: str) -> str:
        """Translate `?` placeholders to `%s` when on Postgres."""
        return sql.replace("?", "%s") if self._pg else sql

    def _exec(self, sql: str, params: tuple = ()) -> Any:
        """Execute one statement, return a cursor (or sqlite3's conn.execute).
        Wraps the placeholder translation so callers stay backend-agnostic."""
        if self._pg:
            cur = self._conn.cursor()
            cur.execute(self._q(sql), params)
            return cur
        return self._conn.execute(self._q(sql), params)

    def _fetchone(self, sql: str, params: tuple = ()) -> Optional[dict]:
        cur = self._exec(sql, params)
        row = cur.fetchone()
        if self._pg and cur:
            cur.close()
        return dict(row) if row is not None else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._exec(sql, params)
        rows = cur.fetchall()
        if self._pg and cur:
            cur.close()
        return [dict(r) for r in rows]

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
                              signup_ip: Optional[str] = None) -> None:
        """Insert a new email-based account. Caller is responsible for
        generating `code` (the stable internal id stored in JWT subjects);
        UNIQUE constraint on email surfaces as a backend-specific IntegrityError
        the caller catches to return a clean 409."""
        with self._lock:
            self._exec(
                "INSERT INTO participants(code, password_hash, email, "
                "display_name, signup_ip, created_utc) "
                "VALUES (?,?,?,?,?,?)",
                (code, password_hash, email, display_name, signup_ip, utc_now()),
            )
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

    # ── sessions ──────────────────────────────────────────────────
    def create_session(self, session_id: str, code: str, participant: dict,
                        sample_seed: Optional[int]) -> None:
        with self._lock:
            self._exec(
                "INSERT INTO sessions(session_id, code, participant, sample_seed, "
                "started_utc, status) VALUES (?,?,?,?,?, 'in_progress')",
                (session_id, code, json.dumps(participant), sample_seed, utc_now()),
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
                                task_focus: Optional[str]) -> None:
        with self._lock:
            self._exec(
                "INSERT INTO training_sessions(training_id, code, task_focus, "
                "started_utc, status) VALUES (?,?,?,?, 'in_progress')",
                (training_id, code, task_focus, utc_now()))
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
                    "theta, sd, rt, ts) VALUES (?,?,?,?,?,?,?,?)",
                    (code, int(p["taskK"]), str(p.get("phase", "train")),
                     p.get("ell"), p.get("theta"), p.get("sd"), p.get("rt"),
                     p.get("ts") or utc_now()))
            self._conn.commit()

    def get_trajectories(self, code: str) -> list[dict]:
        with self._lock:
            return self._fetchall(
                "SELECT task_k, phase, ell, theta, sd, rt, ts "
                "FROM param_trajectories WHERE code=? ORDER BY task_k, ts", (code,))
