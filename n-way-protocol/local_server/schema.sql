-- Production-layout tables for LOCAL test capture.
--
-- Column names and types mirror cortex_web EXACTLY so analysis written
-- against this file later runs against the real production tables unchanged:
--   sessions  = cortex_web/services/api/persistence/schema.py (base columns)
--             + persistence/migrations.py SESSIONS (additive columns)
--             + n-way-protocol/server/migration.sql (approved additive columns)
--   trials    = persistence/schema.py + migrations.py TRIALS
-- Local relaxation (documented in local_server/README.md): NOT NULL / FK
-- constraints tied to the production accounts system (code, participants)
-- are dropped because local sessions have no participant account; the
-- column set and primary keys are unchanged.

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    code           TEXT,
    participant    TEXT,
    sample_seed    INTEGER,
    started_utc    TEXT,
    finished_utc   TEXT,
    status         TEXT,
    stop_reason    TEXT,
    n_questions    INTEGER,
    bundle_version TEXT,
    drawn_seg_ids  TEXT,
    termination_policy TEXT,
    candidate_exclusion TEXT,
    candidate_bank_sha256 TEXT,
    compute_mode   TEXT,
    nway_profile   TEXT,
    norm_id        TEXT,
    norm_sha256    TEXT,
    score_schema_version TEXT,
    norm_profile   TEXT,
    -- approved additive columns from server/migration.sql
    engine_profile_id TEXT,
    response_model TEXT,
    response_artifact_id TEXT,
    response_artifact_sha256 TEXT,
    selector_version TEXT,
    engine_algorithm_version TEXT
);

CREATE TABLE IF NOT EXISTS trials (
    session_id     TEXT NOT NULL,
    trial_index    INTEGER NOT NULL,
    seg_id         INTEGER,
    task_k         INTEGER,
    pick           INTEGER,
    is_correct     INTEGER,
    reaction_ms    REAL,
    diag           TEXT,
    received_utc   TEXT NOT NULL,
    shown_client_utc    TEXT,
    answered_client_utc TEXT,
    PRIMARY KEY (session_id, trial_index)
);
