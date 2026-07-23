"""Canonical table definitions (DDL).

Split from db.py so the persistence layer has schema in one place, beside
the additive column migrations in migrations.py: db.py holds connection
management and queries, this file holds what the tables ARE.

Statements are listed individually rather than as one script so they can be
executed one-by-one on Postgres (which has no executescript). All are
standard SQL-92 that both engines accept verbatim, and all are IF NOT EXISTS
so boot is idempotent on an existing database.
"""
from __future__ import annotations

SCHEMA_STATEMENTS = [
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
        norm_id           TEXT,
        norm_sha256       TEXT,
        score_schema_version TEXT,
        norm_profile      TEXT,
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
        source_session_id TEXT,        -- FK→sessions (the seeding cert; Phase O2)
        norm_id           TEXT,
        norm_sha256       TEXT,
        score_schema_version TEXT,
        norm_profile      TEXT
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
