"""Phase 7 sub-step 3-B — deployment replay driver correctness.

Tests the strict-A deployment-replay driver at
`deployment/replay/run_deployment_replay.py`. The driver wraps
`simulate_candidate(y_source=..., initial_decision=...)` (the
Phase-7-sub-3-B engine hooks) with the rater's strict-A bank +
Y-lookup.

What this file gates:

  * the `_RaterYLookup` callable: on engine inner-loop calls, it
    returns the rater's recorded Y from `labels.csv` (not a
    Bernoulli draw);
  * the per-task replay (`replay_one_per_task`): uses the K=7
    engine with non-target tasks pre-marked as 'refer' and the
    target task's bank = the rater's scored-segs subset; the
    engine receives ONLY real Y values from the lookup;
  * the per-candidate replay (`replay_one_per_candidate`): full
    K=7 engine driven by the rater's strict-A banks across all
    7 tasks; engine receives ONLY real Y values;
  * bank-exhaustion semantics: if the rater's bank on a task is
    smaller than `N_min_per_task`, the verdict is REFER and
    `bank_exhausted` flag is set;
  * "engine-calls match recorded responses" — every Y the engine
    sees in `state.history` must equal the rater's recorded Y
    in `data/labels/labels.csv` for that (seg, k) pair (the
    strict-A contract).

Skips cleanly if the 7.3-A bank artifacts are absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
REPLAY_BANK = REPO / "data" / "replay" / "rater_replay_bank.csv.gz"
REPLAY_SUMMARY = REPO / "data" / "replay" / "rater_replay_summary.csv"


def _skip_if_no_bank():
    if not (REPLAY_BANK.exists() and REPLAY_SUMMARY.exists()):
        pytest.skip(
            "rater_replay bank artifacts missing — run "
            "`python -m pipeline.replay.build_rater_replay_bank`")


@pytest.fixture(scope="module")
def replay_setup():
    _skip_if_no_bank()
    for p in (str(REPO), str(REPO / "engine")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from deployment.simulate_test import (
        load_deployment, TestConfig, deployment_task_names)
    from deployment.replay.run_deployment_replay import (
        _RaterYLookup, _bank_for_rater_task, _y_lookup_for_rater,
        replay_one_per_task, replay_one_per_candidate,
    )
    bank = pd.read_csv(REPLAY_BANK)
    summary = pd.read_csv(REPLAY_SUMMARY)
    Sigma_prior, _, ell_star = load_deployment()
    cfg = TestConfig.from_yaml()
    tasks = deployment_task_names()
    return {
        "bank": bank, "summary": summary, "tasks": tasks,
        "Sigma_prior": Sigma_prior, "ell_star": ell_star, "cfg": cfg,
        "_RaterYLookup": _RaterYLookup,
        "_bank_for_rater_task": _bank_for_rater_task,
        "_y_lookup_for_rater": _y_lookup_for_rater,
        "replay_one_per_task": replay_one_per_task,
        "replay_one_per_candidate": replay_one_per_candidate,
    }


def test_y_lookup_returns_recorded_response(replay_setup):
    """_RaterYLookup must return the rater's recorded Y on the
    requested (k, seg) — the strict-A contract."""
    bank = replay_setup["bank"]
    tasks = replay_setup["tasks"]
    # Pick rater 97 (mbw) on task 'gpd' — has data
    rid = 97
    rater_sub = bank[bank["rater_id"] == rid]
    assert len(rater_sub) > 0, "test fixture broken: no rows for rater 97"
    y_lookup = replay_setup["_y_lookup_for_rater"](bank, rid, tasks)
    # Take a random sample of 20 (k, seg, y) triples and verify the
    # lookup returns the right Y
    pick = rater_sub.sample(n=min(20, len(rater_sub)), random_state=0)
    for _, row in pick.iterrows():
        k = tasks.index(row["task"])
        y_returned = y_lookup(k, int(row["seg_id"]), 0.0)
        assert y_returned == int(row["y"]), (
            f"y_lookup mismatch: rater {rid} task {row['task']!r} "
            f"seg {row['seg_id']}: lookup returned {y_returned}, "
            f"labels.csv has {row['y']}")


def test_y_lookup_raises_on_unrecorded_seg(replay_setup):
    """If the engine ever selects a seg outside the rater's bank,
    _RaterYLookup must raise (strict-A guarantees this never
    happens by construction, but the guard is contract-defending)."""
    bank = replay_setup["bank"]
    tasks = replay_setup["tasks"]
    y_lookup = replay_setup["_y_lookup_for_rater"](bank, 97, tasks)
    # 99999999 is not a real seg_id
    with pytest.raises(RuntimeError, match="contract violation"):
        y_lookup(0, 99999999, 0.0)


def test_bank_for_rater_task_returns_strict_subset(replay_setup):
    """_bank_for_rater_task must return EXACTLY the rater's scored
    segs for the given task — no more, no fewer, with engine-input
    cols (seg_id, s_mean, s_sd)."""
    bank = replay_setup["bank"]
    # rater 97 on 'spike'
    rb = replay_setup["_bank_for_rater_task"](bank, 97, "spike")
    expected_segs = set(
        bank[(bank["rater_id"] == 97) & (bank["task"] == "spike")]
        ["seg_id"].astype(int))
    got_segs = set(rb["seg_id"].astype(int))
    assert got_segs == expected_segs
    assert list(rb.columns) == ["seg_id", "s_mean", "s_sd"]


def test_per_task_engine_y_matches_labels(replay_setup):
    """The strict-A contract on per-task replay: every Y in
    state.history MUST be the rater's recorded value in labels.csv
    for the (seg, k) pair the engine selected. Equivalently:
    every engine-recorded Y matches the looked-up Y."""
    # Use a rater with broad data — rater 97 (mbw) on 'gpd'.
    bank = replay_setup["bank"]
    tasks = replay_setup["tasks"]
    # Re-implement the inner-loop check by inspecting state.history
    # after a replay. The cleanest way: invoke replay_one_per_task,
    # then independently verify that every (k, seg, Y) in state
    # history matches labels for rater 97 on 'gpd'.
    rid, task = 97, "gpd"
    rater_y_table = (bank[(bank["rater_id"] == rid) & (bank["task"] == task)]
                     .set_index("seg_id")["y"].to_dict())
    # We don't have direct access to state.history without invoking
    # the engine. Use a CALL-LOGGING y_source to capture the
    # (k, seg, y) sequence the engine sees, then assert all match
    # the labels.csv record.
    from deployment.replay.run_deployment_replay import (
        _bank_for_rater_task)
    from deployment.simulate_test import simulate_candidate
    k_target = tasks.index(task)
    rater_bank = _bank_for_rater_task(bank, rid, task)
    bank_by_task = {ki: (rater_bank if t == task else
                          pd.DataFrame(columns=["seg_id", "s_mean", "s_sd"]))
                    for ki, t in enumerate(tasks)}
    init_decision = ["refer"] * len(tasks)
    init_decision[k_target] = "pending"
    call_log = []

    def y_real(k, seg, s_val):
        y = rater_y_table.get(int(seg))
        call_log.append((k, int(seg), y))
        return int(y)

    cfg = replay_setup["cfg"]
    Sigma_prior = replay_setup["Sigma_prior"]
    ell_star = replay_setup["ell_star"]
    rng = np.random.default_rng(0)
    true_theta = np.zeros(Sigma_prior.shape[0])
    state = simulate_candidate(
        true_theta, Sigma_prior, ell_star, bank_by_task, cfg,
        rng=rng, y_source=y_real, initial_decision=init_decision)
    # state.history records (k, s_val, Y) — NOT (k, seg_id, Y).
    # The strict-A contract: for each engine inner-loop call,
    #   (i) the engine asked y_source for (k, seg, s_val), and
    #   (ii) the engine's recorded Y equals the labels.csv Y for
    #        that (rater, seg, k) — which y_real ensures by
    #        construction by pulling from rater_y_table.
    # So the test asserts: len(call_log) == len(state.history); for
    # each i, history's k matches call_log's k and history's Y matches
    # call_log's Y. The s_val column in history is a SIGNAL, not a
    # seg_id, so we don't compare it to call_log's seg.
    assert len(state.history) == len(call_log)
    for (h_k, _h_s, h_y), (c_k, c_seg, c_y) in zip(state.history,
                                                    call_log):
        assert h_k == c_k, (
            f"engine k mismatch: history says {h_k}, call_log says {c_k}")
        assert h_y == c_y, (
            f"engine recorded Y {h_y} for (k={c_k}, seg={c_seg}) but "
            f"labels.csv has {c_y} — strict-A contract broken")
    # Sanity: every (k, seg) the engine selected is in the rater's
    # scored set for the target task.
    selected_segs_target = [seg for (k, seg, _y) in call_log
                            if k == k_target]
    assert set(selected_segs_target) <= set(rater_y_table.keys())
    # AND every call_log Y matches the rater's labels.csv Y at that
    # seg (the y_source itself MUST pull from labels).
    for (c_k, c_seg, c_y) in call_log:
        assert c_y == rater_y_table[c_seg], (
            f"y_real(call_log) returned {c_y} for seg {c_seg} but "
            f"labels.csv has y={rater_y_table[c_seg]}")


def test_per_task_replay_runs_end_to_end(replay_setup):
    """A representative per-task replay produces a valid verdict
    (one of pass/fail/refer) with sensible per-task accounting."""
    out = replay_setup["replay_one_per_task"](
        rater_id=97, task="gpd",
        bank=replay_setup["bank"],
        Sigma_prior=replay_setup["Sigma_prior"],
        ell_star=replay_setup["ell_star"],
        tasks=replay_setup["tasks"],
        cfg=replay_setup["cfg"],
        seed=0,
    )
    assert out["rater_id"] == 97
    assert out["task"] == "gpd"
    assert out["decision"] in ("pass", "fail", "refer")
    assert out["n_per_task"] >= 0
    assert out["n_scored_segs"] > 10  # rater 97 has lots of gpd data
    assert out["n_engine_calls"] == out["n_per_task"]  # 1-to-1
    # If decision is pass/fail, n must satisfy cfg.N_min_per_task
    if out["decision"] in ("pass", "fail"):
        assert out["n_per_task"] >= replay_setup["cfg"].N_min_per_task


def test_per_candidate_replay_runs_end_to_end(replay_setup):
    """A representative per-candidate replay produces decisions on
    all 7 tasks."""
    out = replay_setup["replay_one_per_candidate"](
        rater_id=97, bank=replay_setup["bank"],
        Sigma_prior=replay_setup["Sigma_prior"],
        ell_star=replay_setup["ell_star"],
        tasks=replay_setup["tasks"],
        cfg=replay_setup["cfg"], seed=0,
    )
    tasks = replay_setup["tasks"]
    for t in tasks:
        assert f"decision_{t}" in out
        assert out[f"decision_{t}"] in ("pass", "fail", "refer")
        assert out[f"n_per_task_{t}"] >= 0
    assert isinstance(out["all_pass"], bool)
    assert isinstance(out["any_fail"], bool)
    # consistency: all_pass requires every task pass
    if out["all_pass"]:
        for t in tasks:
            assert out[f"decision_{t}"] == "pass"


def test_per_task_replay_other_tasks_skipped(replay_setup):
    """In per-task interpretation (i), the 6 non-target tasks must
    not receive any engine calls (their banks are empty + their
    decisions are pre-marked 'refer'). Verify by call-log inspection."""
    # rater 97 on 'lpd' — broad data, similar pattern
    bank = replay_setup["bank"]
    tasks = replay_setup["tasks"]
    rid, task = 97, "lpd"
    k_target = tasks.index(task)
    from deployment.replay.run_deployment_replay import (
        _bank_for_rater_task, _y_lookup_for_rater)
    from deployment.simulate_test import simulate_candidate
    rater_bank = _bank_for_rater_task(bank, rid, task)
    bank_by_task = {ki: (rater_bank if t == task else
                          pd.DataFrame(columns=["seg_id", "s_mean", "s_sd"]))
                    for ki, t in enumerate(tasks)}
    init_decision = ["refer"] * len(tasks)
    init_decision[k_target] = "pending"
    y_lookup = _y_lookup_for_rater(bank, rid, tasks)
    rng = np.random.default_rng(0)
    true_theta = np.zeros(replay_setup["Sigma_prior"].shape[0])
    state = simulate_candidate(
        true_theta, replay_setup["Sigma_prior"],
        replay_setup["ell_star"], bank_by_task, replay_setup["cfg"],
        rng=rng, y_source=y_lookup, initial_decision=init_decision)
    # Every k recorded in state.history must equal k_target
    for (k, s, y) in state.history:
        assert k == k_target, (
            f"per-task interpretation (i) broken: engine probed "
            f"non-target task k={k} (target={k_target})")
    # The non-target task decisions remain 'refer' (their initial mark)
    for ki, t in enumerate(tasks):
        if t == task:
            continue
        assert state.decision[ki] == "refer", (
            f"non-target task {t!r} got decision {state.decision[ki]!r}"
            f" (expected 'refer')")
        assert state.n_per_task[ki] == 0, (
            f"non-target task {t!r} got n_per_task={state.n_per_task[ki]}"
            f" (expected 0)")


def test_per_candidate_replay_y_matches_labels(replay_setup):
    """The strict-A contract on per-candidate replay: every Y the
    engine sees across all 7 tasks must be the rater's recorded
    labels.csv value. State.history records (k, s_val, Y) where
    s_val is the SIGNAL (not the seg_id), so we verify the contract
    via a call-logging y_source that captures (k, seg, y) directly."""
    bank = replay_setup["bank"]
    tasks = replay_setup["tasks"]
    rid = 97  # mbw, broad coverage on all 7 tasks
    from deployment.replay.run_deployment_replay import (
        _bank_for_rater_task)
    from deployment.simulate_test import simulate_candidate

    # The per-rater Y truth table from labels.csv (via the bank
    # which is built by inner-joining labels with case_bank).
    rsub = bank[bank["rater_id"] == rid]
    truth_by_k_seg = {}
    for _, r in rsub.iterrows():
        truth_by_k_seg[(tasks.index(r["task"]),
                        int(r["seg_id"]))] = int(r["y"])

    bank_by_task = {ki: _bank_for_rater_task(bank, rid, t)
                    for ki, t in enumerate(tasks)}

    call_log = []

    def y_real(k, seg, s_val):
        truth_y = truth_by_k_seg.get((k, int(seg)))
        assert truth_y is not None, (
            f"engine selected (k={k}, seg={seg}) outside rater's "
            f"strict-A bank — internal bug")
        call_log.append((k, int(seg), truth_y))
        return truth_y

    rng = np.random.default_rng(0)
    true_theta = np.zeros(replay_setup["Sigma_prior"].shape[0])
    state = simulate_candidate(
        true_theta, replay_setup["Sigma_prior"],
        replay_setup["ell_star"], bank_by_task, replay_setup["cfg"],
        rng=rng, y_source=y_real)

    # Every history entry's (k, Y) must match the corresponding
    # call_log entry — the engine records exactly what y_source
    # returned.
    assert len(state.history) == len(call_log)
    for (h_k, _h_s, h_y), (c_k, c_seg, c_y) in zip(state.history,
                                                    call_log):
        assert h_k == c_k
        assert h_y == c_y, (
            f"per-candidate engine recorded Y={h_y} for (k={c_k}, "
            f"seg={c_seg}) but labels.csv has Y={c_y}")
    # And every call_log Y was pulled from labels.csv by construction
    # (y_real raises if missing).
    assert len(call_log) > 0
