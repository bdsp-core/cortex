"""Exact-text locks on the generated upsert SQL.

Why this file exists: the suite runs on SQLite, so the Postgres half of every
upsert only ever executes in production. Before these statements were
generated they were hand-written pairs at nine call sites, and nothing checked
that the two halves agreed. These tests pin the generated text for BOTH
dialects, so a change to _upsert_sql that would alter a production write has
to be an explicit, reviewed edit to the expected string here.

The Postgres expectations below are the pre-refactor statements with one
deliberate difference: where the old text assigned a literal in the DO UPDATE
clause (`consumed_utc=NULL`, `attempts=0`, `withdrawn_utc=NULL`), the
generated form assigns `EXCLUDED.<col>`. Those are the same value —
EXCLUDED.<col> is the value the statement proposed, and for these columns the
proposed value IS that literal (see the VALUES clause in the same statement).
"""
from __future__ import annotations

from . import db as db_mod


def gen(table, columns, conflict, *, pg, on_conflict="update"):
    return db_mod._upsert_sql(table, columns, conflict, pg=pg,
                              on_conflict=on_conflict)


def test_auth_code_upsert_sql():
    args = ("auth_codes", db_mod.AUTH_CODE_COLUMNS,
            ("participant_code", "purpose"))
    assert gen(*args, pg=False) == (
        "INSERT OR REPLACE INTO auth_codes(participant_code, purpose, "
        "code_hash, created_utc, expires_utc, consumed_utc, attempts) "
        "VALUES (?,?,?,?,?,NULL,0)")
    assert gen(*args, pg=True) == (
        "INSERT INTO auth_codes(participant_code, purpose, code_hash, "
        "created_utc, expires_utc, consumed_utc, attempts) "
        "VALUES (?,?,?,?,?,NULL,0) "
        "ON CONFLICT (participant_code, purpose) DO UPDATE SET "
        "code_hash=EXCLUDED.code_hash, created_utc=EXCLUDED.created_utc, "
        "expires_utc=EXCLUDED.expires_utc, "
        "consumed_utc=EXCLUDED.consumed_utc, attempts=EXCLUDED.attempts")


def test_consent_event_upsert_sql():
    args = ("consent_events", db_mod.CONSENT_EVENT_COLUMNS,
            ("code", "consent_type", "consent_version"))
    assert gen(*args, pg=False) == (
        "INSERT OR REPLACE INTO consent_events(code, consent_type, "
        "consent_version, irb_protocol_id, accepted_utc, consent_ip, "
        "withdrawn_utc) VALUES (?,?,?,?,?,?,NULL)")
    assert gen(*args, pg=True) == (
        "INSERT INTO consent_events(code, consent_type, consent_version, "
        "irb_protocol_id, accepted_utc, consent_ip, withdrawn_utc) "
        "VALUES (?,?,?,?,?,?,NULL) "
        "ON CONFLICT (code, consent_type, consent_version) DO UPDATE SET "
        "irb_protocol_id=EXCLUDED.irb_protocol_id, "
        "accepted_utc=EXCLUDED.accepted_utc, "
        "consent_ip=EXCLUDED.consent_ip, "
        "withdrawn_utc=EXCLUDED.withdrawn_utc")


def test_trial_upsert_sql():
    args = ("trials", db_mod.TRIAL_COLUMNS, ("session_id", "trial_index"))
    assert gen(*args, pg=False) == (
        "INSERT OR REPLACE INTO trials(session_id, trial_index, seg_id, "
        "task_k, pick, is_correct, reaction_ms, diag, received_utc, "
        "shown_client_utc, answered_client_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?)")
    assert gen(*args, pg=True) == (
        "INSERT INTO trials(session_id, trial_index, seg_id, task_k, pick, "
        "is_correct, reaction_ms, diag, received_utc, shown_client_utc, "
        "answered_client_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT (session_id, trial_index) DO UPDATE SET "
        "seg_id=EXCLUDED.seg_id, task_k=EXCLUDED.task_k, "
        "pick=EXCLUDED.pick, is_correct=EXCLUDED.is_correct, "
        "reaction_ms=EXCLUDED.reaction_ms, diag=EXCLUDED.diag, "
        "received_utc=EXCLUDED.received_utc, "
        "shown_client_utc=EXCLUDED.shown_client_utc, "
        "answered_client_utc=EXCLUDED.answered_client_utc")


def test_result_upsert_sql():
    args = ("results", db_mod.RESULT_COLUMNS, ("session_id",))
    assert gen(*args, pg=False) == (
        "INSERT OR REPLACE INTO results(session_id, result, received_utc) "
        "VALUES (?,?,?)")
    assert gen(*args, pg=True) == (
        "INSERT INTO results(session_id, result, received_utc) "
        "VALUES (?,?,?) ON CONFLICT (session_id) DO UPDATE SET "
        "result=EXCLUDED.result, received_utc=EXCLUDED.received_utc")


def test_training_trial_insert_ignores_conflicts():
    args = ("training_trials", db_mod.TRAINING_TRIAL_COLUMNS,
            ("training_id", "seg_id"))
    cols = ("training_id, code, seg_id, task_k, shown_utc, pick, y_star, "
            "is_correct, feedback_shown, rt_ms, shown_client_utc, "
            "answered_client_utc, seq_in_session, mode, link, quality_flag")
    vals = "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?"
    assert gen(*args, pg=False, on_conflict="nothing") == (
        f"INSERT OR IGNORE INTO training_trials({cols}) VALUES ({vals})")
    assert gen(*args, pg=True, on_conflict="nothing") == (
        f"INSERT INTO training_trials({cols}) VALUES ({vals}) "
        "ON CONFLICT (training_id, seg_id) DO NOTHING")


def test_retention_upsert_sql():
    args = ("training_retention", db_mod.RETENTION_COLUMNS, ("code", "domain"))
    assert gen(*args, pg=False) == (
        "INSERT OR REPLACE INTO training_retention(code, domain, "
        "next_due_utc, interval_s) VALUES (?,?,?,?)")
    assert gen(*args, pg=True) == (
        "INSERT INTO training_retention(code, domain, next_due_utc, "
        "interval_s) VALUES (?,?,?,?) ON CONFLICT (code, domain) "
        "DO UPDATE SET next_due_utc=EXCLUDED.next_due_utc, "
        "interval_s=EXCLUDED.interval_s")


def test_email_invite_upsert_sql():
    args = ("cohort_email_invites", db_mod.EMAIL_INVITE_COLUMNS,
            ("cohort_id", "email"))
    assert gen(*args, pg=False) == (
        "INSERT OR REPLACE INTO cohort_email_invites(cohort_id, email, "
        "invited_utc) VALUES (?,?,?)")
    assert gen(*args, pg=True) == (
        "INSERT INTO cohort_email_invites(cohort_id, email, invited_utc) "
        "VALUES (?,?,?) ON CONFLICT (cohort_id, email) DO UPDATE SET "
        "invited_utc=EXCLUDED.invited_utc")


def test_ledger_inserts_ignore_conflicts():
    login = ("login_days", db_mod.LOGIN_DAY_COLUMNS, ("code", "day"))
    assert gen(*login, pg=False, on_conflict="nothing") == (
        "INSERT OR IGNORE INTO login_days(code, day) VALUES (?,?)")
    assert gen(*login, pg=True, on_conflict="nothing") == (
        "INSERT INTO login_days(code, day) VALUES (?,?) "
        "ON CONFLICT (code, day) DO NOTHING")

    digest = ("digest_log", db_mod.DIGEST_LOG_COLUMNS, ("code", "day"))
    assert gen(*digest, pg=False, on_conflict="nothing") == (
        "INSERT OR IGNORE INTO digest_log(code, day, sent_utc) VALUES (?,?,?)")
    assert gen(*digest, pg=True, on_conflict="nothing") == (
        "INSERT INTO digest_log(code, day, sent_utc) VALUES (?,?,?) "
        "ON CONFLICT (code, day) DO NOTHING")


def test_conflict_keys_are_never_reassigned():
    """A DO UPDATE must not reassign the columns it conflicted on."""
    for table, columns, conflict in (
        ("auth_codes", db_mod.AUTH_CODE_COLUMNS, ("participant_code", "purpose")),
        ("consent_events", db_mod.CONSENT_EVENT_COLUMNS,
         ("code", "consent_type", "consent_version")),
        ("trials", db_mod.TRIAL_COLUMNS, ("session_id", "trial_index")),
        ("results", db_mod.RESULT_COLUMNS, ("session_id",)),
        ("training_retention", db_mod.RETENTION_COLUMNS, ("code", "domain")),
        ("cohort_email_invites", db_mod.EMAIL_INVITE_COLUMNS,
         ("cohort_id", "email")),
    ):
        sets = gen(table, columns, conflict, pg=True).split("DO UPDATE SET ")[1]
        for key in conflict:
            assert f"{key}=EXCLUDED." not in sets, (table, key)


def test_every_bound_column_has_a_placeholder():
    """The parameter count must match the '?' count, or writes bind wrong."""
    for columns in (db_mod.AUTH_CODE_COLUMNS, db_mod.CONSENT_EVENT_COLUMNS,
                    db_mod.TRIAL_COLUMNS, db_mod.RESULT_COLUMNS,
                    db_mod.TRAINING_TRIAL_COLUMNS, db_mod.RETENTION_COLUMNS,
                    db_mod.EMAIL_INVITE_COLUMNS, db_mod.LOGIN_DAY_COLUMNS,
                    db_mod.DIGEST_LOG_COLUMNS):
        sql = gen("t", columns, (columns[0][0],), pg=False)
        assert sql.count("?") == sum(1 for _, expr in columns if expr == "?")
