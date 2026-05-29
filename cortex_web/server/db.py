"""SQLite persistence for the CORTEX web backend.

One file, four tables. SQLite is plenty for the pilot (≤100 concurrent users,
two writes per session — register + finalize — plus optional per-trial
checkpoints). WAL mode lets readers (admin export) run while writers commit.

  participants  credential rows (code → PBKDF2 hash). Created by the admin CLI.
  sessions      one row per started test; participant demographics + status.
  trials        append-only per-question checkpoints (crash-safety).
  results       one row per finished test; the full serialized session JSON.

In production the results would also mirror to S3 (PLAN §8); here the SQLite
row is the source of truth and the admin CLI exports CSV/JSON for analysis.
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

_SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    code           TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    label          TEXT DEFAULT '',
    active         INTEGER NOT NULL DEFAULT 1,
    created_utc    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    code           TEXT NOT NULL,
    participant    TEXT NOT NULL,           -- JSON: registration demographics
    sample_seed    INTEGER,
    started_utc    TEXT NOT NULL,
    finished_utc   TEXT,
    status         TEXT NOT NULL DEFAULT 'in_progress',
    stop_reason    TEXT,
    n_questions    INTEGER,
    FOREIGN KEY (code) REFERENCES participants(code)
);

CREATE TABLE IF NOT EXISTS trials (
    session_id     TEXT NOT NULL,
    trial_index    INTEGER NOT NULL,
    seg_id         INTEGER,
    task_k         INTEGER,
    pick           INTEGER,
    is_correct     INTEGER,
    reaction_ms    REAL,
    diag           TEXT,                     -- JSON: per-trial diagnostic
    received_utc   TEXT NOT NULL,
    PRIMARY KEY (session_id, trial_index)
);

CREATE TABLE IF NOT EXISTS results (
    session_id     TEXT PRIMARY KEY,
    result         TEXT NOT NULL,            -- JSON: full final result
    received_utc   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_code ON sessions(code);
"""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Database:
    """Thin thread-safe wrapper around a SQLite connection.

    A single connection guarded by a lock is simplest and correct for our
    write volume; WAL keeps the admin reader from blocking. FastAPI runs the
    handlers in a threadpool, so the lock matters.
    """

    def __init__(self, path: Optional[Path | str] = None):
        self.path = Path(path or os.environ.get("CORTEX_DB", _DEFAULT_DB))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ── participants ──────────────────────────────────────────────
    def add_participant(self, code: str, password_hash: str, label: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO participants(code, password_hash, label, created_utc) "
                "VALUES (?,?,?,?)",
                (code, password_hash, label, utc_now()),
            )
            self._conn.commit()

    def get_participant(self, code: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM participants WHERE code=?", (code,))
            return cur.fetchone()

    def list_participants(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT code,label,active,created_utc FROM participants "
                "ORDER BY created_utc").fetchall()

    def set_participant_active(self, code: str, active: bool) -> None:
        with self._lock:
            self._conn.execute("UPDATE participants SET active=? WHERE code=?",
                               (1 if active else 0, code))
            self._conn.commit()

    # ── sessions ──────────────────────────────────────────────────
    def create_session(self, session_id: str, code: str, participant: dict,
                        sample_seed: Optional[int]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(session_id, code, participant, sample_seed, "
                "started_utc, status) VALUES (?,?,?,?,?, 'in_progress')",
                (session_id, code, json.dumps(participant), sample_seed, utc_now()),
            )
            self._conn.commit()

    def get_session(self, session_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()

    def finalize_session(self, session_id: str, stop_reason: Optional[str],
                         n_questions: Optional[int]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET status='complete', finished_utc=?, "
                "stop_reason=?, n_questions=? WHERE session_id=?",
                (utc_now(), stop_reason, n_questions, session_id),
            )
            self._conn.commit()

    # ── trials ────────────────────────────────────────────────────
    def upsert_trial(self, session_id: str, trial: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO trials(session_id, trial_index, seg_id, "
                "task_k, pick, is_correct, reaction_ms, diag, received_utc) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    session_id,
                    int(trial.get("trialIndex", trial.get("trial_index", 0))),
                    trial.get("segId", trial.get("seg_id")),
                    trial.get("taskK", trial.get("task_k")),
                    trial.get("pick"),
                    1 if trial.get("isCorrect") else 0 if "isCorrect" in trial else None,
                    trial.get("reactionMs", trial.get("reaction_ms")),
                    json.dumps(trial.get("diag")) if trial.get("diag") is not None else None,
                    utc_now(),
                ),
            )
            self._conn.commit()

    def session_trials(self, session_id: str) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM trials WHERE session_id=? ORDER BY trial_index",
                (session_id,)).fetchall()

    # ── results ───────────────────────────────────────────────────
    def store_result(self, session_id: str, result: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO results(session_id, result, received_utc) "
                "VALUES (?,?,?)",
                (session_id, json.dumps(result), utc_now()),
            )
            self._conn.commit()

    def get_result(self, session_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT result FROM results WHERE session_id=?", (session_id,)).fetchone()
        return json.loads(row["result"]) if row else None

    def all_sessions(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM sessions ORDER BY started_utc").fetchall()
