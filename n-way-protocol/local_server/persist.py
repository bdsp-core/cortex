#!/usr/bin/env python3
"""Materialize local test sessions into the production-layout SQLite file.

The node local server (v20: no node:sqlite; better-sqlite3 not vendored)
journals every session as newline-JSON rows in the exact two-table production
shape under <data-dir>/ndjson/. This script — python3 stdlib only — creates
<db> from local_server/schema.sql (idempotent) and upserts the requested
sessions' rows into it. The server invokes it synchronously at session stop;
it can also be run by hand with --all to rebuild the .db from the journal.

Usage:
  python3 local_server/persist.py --db local_server/data/local_test.db \
      --data-dir local_server/data (--session SESSION_ID | --all)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# Exact production column order (mirrors schema.sql; see README parity note).
SESSION_COLUMNS = [
    "session_id", "code", "participant", "sample_seed", "started_utc",
    "finished_utc", "status", "stop_reason", "n_questions", "bundle_version",
    "drawn_seg_ids", "termination_policy", "candidate_exclusion",
    "candidate_bank_sha256", "compute_mode", "nway_profile", "norm_id",
    "norm_sha256", "score_schema_version", "norm_profile",
    "engine_profile_id", "response_model", "response_artifact_id",
    "response_artifact_sha256", "selector_version", "engine_algorithm_version",
]
TRIAL_COLUMNS = [
    "session_id", "trial_index", "seg_id", "task_k", "pick", "is_correct",
    "reaction_ms", "diag", "received_utc", "shown_client_utc",
    "answered_client_utc",
]


def _encode(value, sort_keys: bool = False):
    """JSON-serialize composite values into TEXT columns.

    nway_profile uses sort_keys=True to byte-match production's
    json.dumps(nway_profile, sort_keys=True); diag matches production's plain
    json.dumps(diag); lists (drawn_seg_ids) keep their order either way.
    """
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=sort_keys)
    return value


def _upsert(conn: sqlite3.Connection, table: str, columns: list[str],
            values: list) -> None:
    placeholders = ",".join("?" * len(columns))
    conn.execute(
        f"INSERT OR REPLACE INTO {table}({','.join(columns)}) "
        f"VALUES ({placeholders})", values)


def persist_session(conn: sqlite3.Connection, ndjson_dir: Path,
                    session_id: str) -> int:
    session_path = ndjson_dir / f"{session_id}.session.json"
    row = json.loads(session_path.read_text())
    _upsert(conn, "sessions", SESSION_COLUMNS, [
        _encode(row.get(col), sort_keys=(col == "nway_profile"))
        for col in SESSION_COLUMNS
    ])
    n_trials = 0
    trials_path = ndjson_dir / f"{session_id}.trials.jsonl"
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if not line.strip():
                continue
            trial = json.loads(line)
            _upsert(conn, "trials", TRIAL_COLUMNS,
                    [_encode(trial.get(col)) for col in TRIAL_COLUMNS])
            n_trials += 1
    return n_trials


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--session", action="append", default=[],
                        help="session id to persist (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="persist every journaled session")
    args = parser.parse_args()

    ndjson_dir = args.data_dir / "ndjson"
    session_ids = list(args.session)
    if args.all:
        session_ids += sorted(
            p.name[: -len(".session.json")]
            for p in ndjson_dir.glob("*.session.json")
            if p.name[: -len(".session.json")] not in session_ids)
    if not session_ids:
        parser.error("nothing to persist: pass --session or --all")

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        for session_id in session_ids:
            n_trials = persist_session(conn, ndjson_dir, session_id)
            print(f"persisted {session_id}: sessions row + {n_trials} trials "
                  f"-> {args.db}")
        conn.commit()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
