import sqlite3

from api.db import Database
from api.persistence import migrations


def _create_pre_compute_mode_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE participants (
            code TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            label TEXT DEFAULT '',
            active INTEGER NOT NULL DEFAULT 1,
            created_utc TEXT NOT NULL,
            email TEXT,
            display_name TEXT,
            signup_ip TEXT
        );
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            code TEXT NOT NULL,
            participant TEXT NOT NULL,
            sample_seed BIGINT,
            started_utc TEXT NOT NULL,
            finished_utc TEXT,
            status TEXT NOT NULL DEFAULT 'in_progress',
            stop_reason TEXT,
            n_questions INTEGER,
            bundle_version TEXT,
            drawn_seg_ids TEXT,
            termination_policy TEXT NOT NULL DEFAULT 'ad6',
            candidate_exclusion TEXT,
            candidate_bank_sha256 TEXT
        );
        INSERT INTO participants(code, password_hash, created_utc, email)
        VALUES ('legacy-reader', 'hash', '2026-01-01T00:00:00Z', 'legacy@example.test');
        INSERT INTO sessions(
            session_id, code, participant, sample_seed, started_utc, status,
            n_questions, bundle_version, termination_policy
        ) VALUES (
            'legacy-session', 'legacy-reader', '{}', 7,
            '2026-01-01T00:00:00Z', 'complete', 42, 'legacy-bank', 'ad6'
        );
        """
    )
    connection.commit()
    connection.close()


def test_additive_session_migration_preserves_legacy_rows(tmp_path):
    path = tmp_path / "legacy.db"
    _create_pre_compute_mode_database(path)

    database = Database(path)
    row = database.get_session("legacy-session")
    assert row is not None
    assert row["code"] == "legacy-reader"
    assert row["n_questions"] == 42
    assert row["bundle_version"] == "legacy-bank"
    assert row["termination_policy"] == "ad6"
    assert row["compute_mode"] == "serial"
    assert row["nway_profile"] is None
    database.close()

    # Reopening proves the declarations and executor are idempotent.
    reopened = Database(path)
    assert reopened.get_session("legacy-session")["compute_mode"] == "serial"
    reopened.close()


def test_migration_registry_has_unique_columns_per_table_pass():
    for table, columns in migrations.BY_TABLE:
        names = [name for name, _sql_type in columns]
        assert len(names) == len(set(names)), table
    assert ("compute_mode", "TEXT NOT NULL DEFAULT 'serial'") in migrations.SESSIONS
    assert ("nway_profile", "TEXT") in migrations.SESSIONS
