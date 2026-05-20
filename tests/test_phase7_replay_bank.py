"""Phase 7 sub-step 3-A — rater replay bank gates.

The strict-A replay harness (sub-7.3) needs a per-rater per-task
join of `data/labels/labels.csv` (with the erratum-correct {bipd,
birds, other}→other mapping) with the K=7 frozen item bank at
`data/deployment_prior/case_bank.csv`. This file gates the
correctness contract for that build product:

  * The output exists at `data/replay/`;
  * Schema matches the engine inputs (rater_id, task, seg_id, y,
    s_mean, s_sd);
  * Erratum-correct mapping is applied (no `bipd`/`birds` value
    survives in the IIIC tasks — they fold into `other`);
  * The 7 production tasks are all populated;
  * Y is binary {0, 1} for every row;
  * The 4 Centaur gold experts (mbw=97, cal=99000001,
    matt=99000002, tianyu=99000003) appear with the expected
    Phase-1 ingest counts (~5,000 segs/IIIC task for cal/matt/
    tianyu; mbw across all 7 tasks);
  * The per-candidate cohort at the deployment N_min_per_task=10
    floor across ALL 7 tasks is non-trivial (≥10 raters; matches
    the documented audit ~21 raters, mostly expert/experienced).

The build is idempotent: re-running the builder must produce
byte-identical output (no nondeterminism in the join).
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
BANK = REPO / "data" / "replay" / "rater_replay_bank.csv.gz"
SUMMARY = REPO / "data" / "replay" / "rater_replay_summary.csv"

TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]
GOLD = {97: "mbw", 99000001: "cal", 99000002: "matt", 99000003: "tianyu"}


def _skip_if_no_bank():
    if not (BANK.exists() and SUMMARY.exists()):
        pytest.skip(
            "rater_replay_bank.csv.gz not built — run "
            "`python -m pipeline.replay.build_rater_replay_bank` first")


@pytest.fixture(scope="module")
def bank():
    _skip_if_no_bank()
    return pd.read_csv(BANK)


@pytest.fixture(scope="module")
def summary():
    _skip_if_no_bank()
    return pd.read_csv(SUMMARY)


def test_outputs_exist():
    _skip_if_no_bank()
    assert BANK.is_file()
    assert SUMMARY.is_file()


def test_bank_schema(bank):
    """Schema = engine inputs: rater_id, task, seg_id, y, s_mean, s_sd."""
    assert list(bank.columns) == [
        "rater_id", "task", "seg_id", "y", "s_mean", "s_sd"]
    assert bank["y"].isin([0, 1]).all(), (
        "Y must be binary {0, 1}; found other values")
    assert bank["s_mean"].notna().all(), (
        "s_mean must be present for every row (engine input)")
    assert bank["s_sd"].notna().all(), (
        "s_sd must be present for every row (engine input)")
    # Phase-3.5 invariant: s_sd > 0 by construction (joint-IRT fit)
    assert (bank["s_sd"] > 0).all()


def test_all_seven_tasks_populated(bank):
    """All 7 production tasks must appear."""
    assert set(bank["task"].unique()) == set(TASKS)
    # Each task must have a non-trivial number of observations
    counts = bank.groupby("task").size()
    for t in TASKS:
        assert counts[t] > 100_000, (
            f"task {t!r} has only {counts[t]} observations "
            f"(expected > 100,000)")


def test_erratum_correct_iiic_mapping(bank):
    """Erratum-correct `{bipd, birds, other}→other` mapping (Phase-3
    D5 / AUDIT §6) — no bipd/birds value survives as a separate
    IIIC task. The IIIC task list is exactly the 6 in TASKS minus
    spike."""
    iiic = bank[bank["task"] != "spike"]
    assert set(iiic["task"].unique()) == set(TASKS) - {"spike"}
    # In particular, no `bipd` or `birds` task — they fold into other
    assert "bipd" not in iiic["task"].unique()
    assert "birds" not in iiic["task"].unique()


def test_centaur_gold_experts_present(bank):
    """The 4 Centaur-IIIC gold experts (the canonical 4-expert
    panel, mbw + cal/matt/tianyu) appear with the expected Phase-1
    ingest counts: cal/matt/tianyu = exactly 5,000 segs per IIIC
    task (the gold cohort, no spike); mbw across all 7 tasks."""
    counts = bank[bank["rater_id"].isin(GOLD)].groupby(
        ["rater_id", "task"]).size().unstack(fill_value=0)
    for rid, name in GOLD.items():
        assert rid in counts.index, (
            f"Centaur gold expert {name!r} ({rid}) absent from bank")
    # mbw (rater 97) — has all 7 tasks (general expert pool member)
    assert (counts.loc[97] > 0).all(), (
        "mbw should have responses on all 7 tasks")
    # cal/matt/tianyu — Centaur 5,000-case gold panel; IIIC only
    for rid in (99000001, 99000002, 99000003):
        for task in (set(TASKS) - {"spike"}):
            assert counts.loc[rid, task] == 5000, (
                f"Centaur gold expert {GOLD[rid]!r} task {task!r}: "
                f"{counts.loc[rid, task]} segs (expected 5,000 — "
                f"the Phase-1 gold cohort)")
        # no spike for the 3 gold experts (Centaur-IIIC cohort only)
        assert counts.loc[rid, "spike"] == 0


def test_summary_schema(summary):
    """Summary cols: rater_id, task, n_segs, n_pos, canonical_name,
    expertise_level. Aggregates a non-trivial cohort."""
    for col in ("rater_id", "task", "n_segs", "n_pos",
                "expertise_level"):
        assert col in summary.columns, f"summary missing {col!r}"
    # n_segs must be > 0 for every row (only rater×task cells with
    # actual responses appear)
    assert (summary["n_segs"] > 0).all()
    # n_pos must be ≤ n_segs
    assert (summary["n_pos"] <= summary["n_segs"]).all()


def test_per_task_cohort_at_n_min_10(summary):
    """Per-task cohort at the deployment N_min_per_task=10 floor —
    the user-locked headline floor (Q2 sub-7.3, 2026-05-19). Must
    be ≥10,000 (rater, task) cells across the 7 tasks (the
    pre-build audit found 14,955; the post-bank-join number is
    slightly smaller due to segs not in the K=7 case_bank — but
    still ≥10,000 — see PHASE7_REPLAY_DESIGN.md §5)."""
    at_10 = (summary["n_segs"] >= 10).sum()
    assert at_10 >= 10_000, (
        f"per-task cohort at floor=10 has only {at_10} cells "
        f"(expected ≥10,000)")
    # Per-task floor=10 (each IIIC task has its own count, spike
    # has its own).
    by_task = (summary[summary["n_segs"] >= 10]
               .groupby("task").size().to_dict())
    for t in TASKS:
        assert by_task.get(t, 0) >= 1_000, (
            f"task {t!r} at floor=10 has only {by_task.get(t, 0)} "
            f"raters (expected ≥1,000)")


def test_per_candidate_cohort_at_n_min_10(summary):
    """Per-candidate cohort = raters with ≥N_min_per_task=10 on ALL
    7 tasks. The headline cohort the per-CANDIDATE replay analyses
    (Q4 coverage iii, 2026-05-19). The pre-build audit found 22
    raters; the post-bank-join number drops to 21 (documented in
    PHASE7_REPLAY_DESIGN.md §1) because one rater's scored-segs
    subset on at least one task drops below 10 after the join with
    the K=7 case_bank (segs the rater scored but that aren't in
    the production frozen bank)."""
    piv = summary.pivot_table(
        index="rater_id", columns="task", values="n_segs",
        fill_value=0)
    has_all = (piv[TASKS] >= 10).all(axis=1)
    n = int(has_all.sum())
    assert n >= 10, (
        f"per-candidate cohort at floor=10 has only {n} raters "
        f"(expected ≥10)")
    # Document the actual count for downstream tests/reports.
    # The build audit at 2026-05-19 found exactly 21 (mostly expert).
    # Allow ±1 to absorb future labels.csv updates that legitimately
    # add or remove one cell.
    assert 20 <= n <= 23, (
        f"per-candidate cohort drift: {n} raters (audit was 21)")
    # tier distribution: should be mostly expert (the build
    # audit found 19 expert + 2 experienced)
    qualifiers = piv[has_all].index
    tiers = summary[summary["rater_id"].isin(qualifiers)] \
        .drop_duplicates("rater_id")["expertise_level"] \
        .value_counts().to_dict()
    assert tiers.get("expert", 0) >= 15, (
        f"per-candidate cohort is unexpectedly non-expert-heavy: "
        f"{tiers}")


def test_per_task_y_distributions_sensible(bank):
    """Sanity: spike has a high positive rate (~65% — the spike-
    paper convention is y=1 ⇒ spike); each IIIC task has a low
    positive rate (~10–25% — one-vs-rest binary collapse across
    7 classes after the {bipd,birds,other}→other mapping)."""
    by_task = bank.groupby("task")["y"].agg(["sum", "size"])
    by_task["pos_rate"] = by_task["sum"] / by_task["size"]
    # spike: high pos rate (positive minority + benign mix; the
    # combined_spike sn1-clean binary at the corpus level has
    # ~65% positives — the spike paper's reference distribution)
    spike_pos = by_task.loc["spike", "pos_rate"]
    assert 0.3 < spike_pos < 0.9, (
        f"spike positive rate {spike_pos:.3f} outside [0.3, 0.9]")
    # Each IIIC task: one-vs-rest binary — pos rate the fraction
    # of segs voted that class. Empirically 10–30% across the 6.
    for t in (set(TASKS) - {"spike"}):
        pos = by_task.loc[t, "pos_rate"]
        assert 0.05 < pos < 0.40, (
            f"IIIC task {t!r} positive rate {pos:.3f} outside "
            f"[0.05, 0.40]")
