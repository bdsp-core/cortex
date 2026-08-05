"""Idempotent additive migration declarations.

The runtime migration executor remains in :mod:`api.db` because it owns the
active transaction and backend-specific column introspection. Keeping the
declarations here makes schema evolution reviewable without mixing it with
account, session, cohort, and training repository methods.
"""

Column = tuple[str, str]

PARTICIPANTS: list[Column] = [
    ("email", "TEXT"),
    ("display_name", "TEXT"),
    ("signup_ip", "TEXT"),
    ("email_verified_utc", "TEXT"),
    ("auth_provider", "TEXT"),
    ("google_sub", "TEXT"),
    ("signup_expertise", "TEXT"),
    ("profile", "TEXT"),
    ("public_id", "TEXT"),
    ("email_undeliverable_utc", "TEXT"),
    ("tz_offset_min", "INTEGER"),
    ("digest_opt_out", "INTEGER"),
]

PARAM_TRAJECTORIES: list[Column] = [
    ("training_id", "TEXT"),
    ("source_session_id", "TEXT"),
    ("seq_in_session", "INTEGER"),
    ("is_real", "INTEGER NOT NULL DEFAULT 0"),
]

TRAINING_SESSIONS: list[Column] = [
    ("regimen_id", "TEXT"),
    ("source_session_id", "TEXT"),
    ("norm_id", "TEXT"),
    ("norm_sha256", "TEXT"),
    ("score_schema_version", "TEXT"),
    ("norm_profile", "TEXT"),
]

# Precision sessions reconstruct their manifest-ordered pool from the compact
# exclusion list and exact bank hash; AD6 retains its explicit drawn id list.
SESSIONS: list[Column] = [
    ("bundle_version", "TEXT"),
    ("drawn_seg_ids", "TEXT"),
    ("termination_policy", "TEXT NOT NULL DEFAULT 'ad6'"),
    ("compute_mode", "TEXT NOT NULL DEFAULT 'serial'"),
    ("candidate_exclusion", "TEXT"),
    ("candidate_bank_sha256", "TEXT"),
    ("nway_profile", "TEXT"),
    ("precision_recalibration", "TEXT"),
    ("bias_flag_tiers", "TEXT"),
    ("norm_id", "TEXT"),
    ("norm_sha256", "TEXT"),
    ("score_schema_version", "TEXT"),
    ("norm_profile", "TEXT"),
]

TRAINING_TRIALS: list[Column] = [
    ("pick", "INTEGER"),
    ("y_star", "INTEGER"),
    ("is_correct", "INTEGER"),
    ("feedback_shown", "TEXT"),
    ("rt_ms", "REAL"),
    ("shown_client_utc", "TEXT"),
    ("answered_client_utc", "TEXT"),
    ("seq_in_session", "INTEGER"),
    ("mode", "TEXT"),
    ("link", "TEXT"),
    ("quality_flag", "TEXT"),
]

TRAINING_SESSIONS_ENGINE: list[Column] = [
    ("engine_meta", "TEXT"),
]

TRIALS: list[Column] = [
    ("shown_client_utc", "TEXT"),
    ("answered_client_utc", "TEXT"),
]

# Server-side deterministic replay verification of ingested results
# (api.result_verify): the client computes verdicts, so the server re-derives
# each complete sitting from its stored picks and records agreement here.
RESULTS: list[Column] = [
    ("verify_status", "TEXT"),      # NULL=pending · pass · divergent · error
    ("verify_detail", "TEXT"),
    ("verified_utc", "TEXT"),
]

BY_TABLE: tuple[tuple[str, list[Column]], ...] = (
    ("param_trajectories", PARAM_TRAJECTORIES),
    ("training_sessions", TRAINING_SESSIONS),
    ("sessions", SESSIONS),
    ("training_trials", TRAINING_TRIALS),
    ("trials", TRIALS),
    ("training_sessions", TRAINING_SESSIONS_ENGINE),
    ("results", RESULTS),
)
