"""CORTEX internal test — adaptive-session controller (Phase B).

Drives a K=6 IIIC adaptive certification session on the SMC + MCMC
particle-cloud engine (engine/core_mcmc.py). Two layers in one file:

  * ``CortexSession`` — the pure, Qt-free adaptive loop. Built entirely
    from the engine's public primitives (make_state_hier, choose_item,
    update, ess, resample_and_rejuvenate) — the engine-modification-free
    pattern of scripts/viz_smc_collapse.py. It de-duplicates served
    segments (choose_item has no de-dup of its own), captures per-trial
    telemetry, and delegates termination to an injectable
    ``TerminationPolicy`` (AD6 by default; the legacy AUROC half-width
    delta-rule and a never-stop variant remain available via the
    ``cortex_policy.default_policy_for`` precedence ladder). Testable
    headless with any ``y_source`` callable.

  * ``EngineWorker`` + ``SessionController`` — the Qt wrapper.
    EngineWorker runs a CortexSession inside a background QThread with a
    queue-backed y_source; SessionController owns the session_id, the
    GUI->engine queue, and the one-vs-rest binary-Y reduction. Defined
    only when PyQt6 is importable, so the pure core is usable without Qt.

The engine numerics are never modified. BLAS is pinned single-thread
(the engine reproducibility contract) before numpy is imported.
"""
from __future__ import annotations

import os

# Engine reproducibility contract — pin BLAS single-thread BEFORE numpy.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import queue  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402

import numpy as np  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "engine"), os.path.join(_REPO, "scripts"),
           _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_mcmc import (  # noqa: E402
    make_state_hier, choose_item, update, ess, resample_and_rejuvenate,
    simulate_response, _expected_loss_vec)
from auroc import (  # noqa: E402
    auroc_mean_from_particles_hier, auroc_halfwidths_hier)
from cortex_engine_inputs import build_iiic_engine_inputs  # noqa: E402
from cortex_policy import default_policy_for  # noqa: E402

# ── engine hyperparameters (the methodology config — viz_smc_collapse.py) ──
N_PARTICLES = 600
# delta=0.15 is the CORTEX internal-test stop target, PI-confirmed
# 2026-05-22. The Paper-1 methodology delta=0.05 is unreachable in the
# 100-segment IIIC bank (audit: 0/13 raters) — see results/audit/.
DELTA_AUROC = 0.15
N_MH_STEPS = 15
ESS_THRESHOLD_FRAC = 0.5
ALPHA = 0.05               # 95% CI for the AUROC half-width
# Q1 varies per examinee: the opening question is drawn uniformly (session
# rng) from the FIRST_ITEM_TOPN most-informative items rather than the
# single deterministic argmin (which would be identical for everyone).
FIRST_ITEM_TOPN = 10
R_ASSUMED = 0.378          # compound-symmetry fallback only — unused once
                           # the fitted Corr_l is supplied to make_state_hier

_SENTINEL_ABORT = object()  # pushed onto the answer queue to end a session


class SessionAborted(Exception):
    """Raised inside the engine loop when the session is aborted."""


@dataclass
class SessionResult:
    """Outcome of one adaptive session."""

    session_id: str
    seed: int
    selection: str                 # "adaptive" | "random"
    n_questions: int
    stop_reason: str               # "all_resolved" (AD6) | "delta_reached"
                                   # (legacy delta) | "bank_exhausted" | "aborted"
    delta_auroc: float | None
    n_particles: int
    task_codes: list
    trials: list                   # per-trial telemetry dicts
    served_seg_ids: list
    final_auroc_mean: np.ndarray
    final_auroc_hw: np.ndarray
    final_t_mean: np.ndarray
    final_l_mean: np.ndarray
    # optional particle-cloud trajectory (Phase D trajectory.npz / Phase 6)
    t_traj: np.ndarray = None
    l_traj: np.ndarray = None
    w_traj: np.ndarray = None
    aborted: bool = False
    # AD6 termination-policy output: final per-task verdict labels (PASS /
    # FAIL / REFER_BORDERLINE / REFER_UNINFORMATIVE) and the last call's
    # diagnostics (π_k, mcse_k, R_k, ESS, n_per_task). Both None for the
    # NoStop / Delta legacy paths — those policies do not render verdicts.
    verdicts: list = None
    policy_diagnostics: dict = None


def _posterior_summary(state):
    """Weighted posterior total variance + per-task means from a hier state.

    Returns (var_t_total, var_l_total, t_mean[K], l_mean[K]) — var_*_total
    is the sum over tasks, the same quantity choose_item's _expected_loss_vec
    minimises in expectation.
    """
    w = state["w"] / state["w"].sum()
    K = state["t"].shape[1]
    t_mean = np.array([float((w * state["t"][:, k]).sum()) for k in range(K)])
    l_mean = np.array([float((w * state["l"][:, k]).sum()) for k in range(K)])
    var_t = sum(float((w * (state["t"][:, k] - t_mean[k]) ** 2).sum())
                for k in range(K))
    var_l = sum(float((w * (state["l"][:, k] - l_mean[k]) ** 2).sum())
                for k in range(K))
    return var_t, var_l, t_mean, l_mean


class CortexSession:
    """Pure, Qt-free K=6 IIIC adaptive session over the SMC engine."""

    def __init__(self, inputs, *, session_id="adhoc",
                 n_particles=N_PARTICLES, delta_auroc=None,
                 max_questions=None, ess_threshold_frac=ESS_THRESHOLD_FRAC,
                 n_mh_steps=N_MH_STEPS, seed=None, selection="adaptive",
                 policy=None, capture_clouds=False,
                 first_item_topn=FIRST_ITEM_TOPN):
        self.inputs = inputs
        self.session_id = str(session_id)
        self.N = int(n_particles)
        self.delta_auroc = float(delta_auroc) if delta_auroc is not None else None
        self.K = len(inputs.task_codes)
        n_bank = len(inputs.all_seg_ids)
        self.max_questions = (n_bank if max_questions is None
                              else min(int(max_questions), n_bank))
        self.ess_threshold_frac = float(ess_threshold_frac)
        self.n_mh_steps = int(n_mh_steps)
        self.capture_clouds = bool(capture_clouds)
        self.first_item_topn = max(1, int(first_item_topn))
        if selection not in ("adaptive", "random"):
            raise ValueError(f"selection must be adaptive|random, got {selection!r}")
        self.selection = selection
        if seed is None:
            seed = int(hashlib.sha256(self.session_id.encode()).hexdigest()[:8], 16)
        self.seed = int(seed)
        # AD6: the stopping check is a single injectable TerminationPolicy.
        # `default_policy_for` resolves precedence — explicit policy=, then
        # AD6Policy (delta_auroc=None, production), NoStopPolicy (==0.0,
        # audit/OC), DeltaStopPolicy (>0, legacy methods).
        self.policy = default_policy_for(
            inputs, delta_auroc=delta_auroc, policy=policy)
        self.proposal_scale = 2.38 / np.sqrt(2 * self.K)

    def _select(self, state, remaining, rng, trial_index):
        """Pick the next (task k, seg_id, s, s_sd) + the chosen item's
        expected post-answer total posterior variance."""
        inp = self.inputs
        if self.selection == "adaptive":
            bank_signals, bank_sds, bank_segids = inp.as_engine_arrays(remaining)
            if trial_index == 0 and self.first_item_topn > 1:
                # Q1 varies per examinee — draw uniformly (session rng) from
                # the top-N most-informative items, not the single argmin
                # (which would be identical for every test-taker).
                k, seg_id, s, s_sd = self._pick_top_n(
                    state, bank_signals, bank_sds, bank_segids, rng)
            else:
                k, s, s_sd, seg_id = choose_item(
                    state, bank_signals, bank_sds=bank_sds,
                    return_sd=True, bank_segids=bank_segids)
        else:  # random null baseline — the Phase-1 ablation comparator
            seg_id = int(rng.choice(remaining))
            k = int(rng.integers(self.K))
            bs, bsd, _ = inp.as_engine_arrays([seg_id])
            s, s_sd = float(bs[k][0]), float(bsd[k][0])
        k, seg_id, s, s_sd = int(k), int(seg_id), float(s), float(s_sd)
        # the engine's selection objective for the chosen item (telemetry /
        # audit only — _expected_loss_vec is read, the engine is not modified)
        expected_loss = float(_expected_loss_vec(
            state, k, np.array([s]), signal_sds=np.array([s_sd]))[0])
        return k, seg_id, s, s_sd, expected_loss

    def _pick_top_n(self, state, bank_signals, bank_sds, bank_segids, rng):
        """Trial-0 selection — uniformly pick among the `first_item_topn`
        items with the lowest expected posterior variance, so the opening
        question differs per examinee instead of being a fixed argmin."""
        scored = []                       # (expected_loss, task_k, idx)
        for k in range(self.K):
            sigs = np.asarray(bank_signals[k])
            sds = np.asarray(bank_sds[k])
            losses = _expected_loss_vec(state, k, sigs, signal_sds=sds)
            for idx in range(len(sigs)):
                scored.append((float(losses[idx]), k, idx))
        scored.sort(key=lambda row: row[0])
        n_top = min(self.first_item_topn, len(scored))
        _, k, idx = scored[int(rng.integers(n_top))]
        return (k, int(np.asarray(bank_segids[k])[idx]),
                float(np.asarray(bank_signals[k])[idx]),
                float(np.asarray(bank_sds[k])[idx]))

    def run(self, y_source, on_item=None, on_trial=None):
        """Run the adaptive loop. ``y_source(k, seg_id, s) -> int`` returns
        the binary response; it may block (the live GUI path). ``on_item`` /
        ``on_trial`` are optional callbacks fired before/after each answer.
        """
        rng = np.random.default_rng(self.seed)
        inp = self.inputs
        state = make_state_hier(self.N, self.K, R_ASSUMED, rng,
                                Sigma_l=inp.Corr_l, Sigma_t=inp.Corr_l)
        remaining = list(inp.all_seg_ids)
        trials, served = [], []
        t_traj, l_traj, w_traj = [], [], []
        stop_reason = "bank_exhausted"
        aborted = False
        self.policy.reset(self.K)
        n_per_task = [0] * self.K
        decision = None

        try:
            for trial_index in range(self.max_questions):
                if not remaining:
                    break
                t0 = time.perf_counter()
                k, seg_id, s, s_sd, expected_loss = self._select(
                    state, remaining, rng, trial_index)
                select_ms = (time.perf_counter() - t0) * 1000.0

                if on_item is not None:
                    on_item({"trial_index": trial_index, "task_k": k,
                             "task_code": inp.task_codes[k], "seg_id": seg_id,
                             "s_mean": s, "s_sd": s_sd})

                y = int(y_source(k, seg_id, s))   # blocks on the live path

                update(state, k, s, y, s_sd=s_sd)
                rejuv = False
                if ess(state["w"]) < self.ess_threshold_frac * self.N:
                    resample_and_rejuvenate(state, rng, self.n_mh_steps,
                                            self.proposal_scale)
                    rejuv = True

                var_t, var_l, t_mean, l_mean = _posterior_summary(state)
                auroc_mean = auroc_mean_from_particles_hier(state)
                auroc_hw = auroc_halfwidths_hier(state, alpha=ALPHA)
                served.append(seg_id)
                remaining.remove(seg_id)
                n_per_task[k] += 1

                tel = {
                    "trial_index": trial_index, "task_k": k,
                    "task_code": inp.task_codes[k], "seg_id": seg_id,
                    "pattern_class_true": inp.pattern_class(seg_id),
                    "s_mean": s, "s_sd": s_sd, "response_y": y,
                    "select_ms": select_ms,
                    "expected_loss_chosen": expected_loss,
                    "total_var_t": var_t, "total_var_l": var_l,
                    "total_var": var_t + var_l,
                    "auroc_mean": auroc_mean.tolist(),
                    "auroc_hw": auroc_hw.tolist(),
                    "max_hw": float(auroc_hw.max()),
                    "t_post_mean": t_mean.tolist(),
                    "l_post_mean": l_mean.tolist(),
                    "ess": float(ess(state["w"])), "rejuv": rejuv,
                    "n_per_task": list(n_per_task),
                }

                # AD6: policy sees post-update state, post-increment n_per_task.
                # Diagnostics + verdicts attach to `tel` BEFORE on_trial fires so
                # the viewer / trials.jsonl see verdict-gate evolution in real
                # time. NoStop / Delta paths return None — additive keys only,
                # no existing telemetry contract is broken.
                decision = self.policy(state, tel, n_per_task, self.K)
                if decision.diagnostics is not None:
                    tel["policy_diag"] = decision.diagnostics
                if decision.verdicts is not None:
                    tel["verdicts"] = list(decision.verdicts)

                trials.append(tel)
                if self.capture_clouds:
                    t_traj.append(state["t"].copy())
                    l_traj.append(state["l"].copy())
                    w_traj.append(state["w"].copy())
                if on_trial is not None:
                    on_trial(tel)

                if decision.stop:
                    stop_reason = decision.stop_reason
                    break
        except SessionAborted:
            aborted = True
            stop_reason = "aborted"

        # Always finalize — partial PASS/FAIL verdicts already locked in stay,
        # PENDING tasks become REFER_BORDERLINE / REFER_UNINFORMATIVE. Safe to
        # call on aborted or zero-trial sessions; returns None on NoStop/Delta.
        verdicts = self.policy.finalize_verdicts()
        policy_diagnostics = (decision.diagnostics if decision is not None
                              else None)

        var_t, var_l, t_mean, l_mean = (
            _posterior_summary(state) if trials
            else (0.0, 0.0, np.zeros(self.K), np.zeros(self.K)))
        return SessionResult(
            session_id=self.session_id, seed=self.seed, selection=self.selection,
            n_questions=len(trials), stop_reason=stop_reason,
            delta_auroc=self.delta_auroc, n_particles=self.N,
            task_codes=list(inp.task_codes), trials=trials,
            served_seg_ids=served,
            final_auroc_mean=auroc_mean_from_particles_hier(state),
            final_auroc_hw=auroc_halfwidths_hier(state, alpha=ALPHA),
            final_t_mean=t_mean, final_l_mean=l_mean,
            t_traj=np.array(t_traj) if t_traj else None,
            l_traj=np.array(l_traj) if l_traj else None,
            w_traj=np.array(w_traj) if w_traj else None,
            aborted=aborted,
            verdicts=verdicts, policy_diagnostics=policy_diagnostics)


def make_simulated_y_source(true_t, true_l, seed=0):
    """A y_source backed by a simulated rater of known (t, l) per task — the
    engine's own generative form (core_mcmc.simulate_response). Used by the
    selection audit and the delta-reachability check."""
    rng = np.random.default_rng(seed)
    true_t = np.asarray(true_t, float)
    true_l = np.asarray(true_l, float)

    def _y(k, seg_id, s):
        return simulate_response(s, true_t[k], true_l[k], rng)

    return _y


# ──────────────────────── Qt wrapper (optional) ────────────────────────
try:
    from PyQt6.QtCore import QObject, QThread, pyqtSignal
    _HAVE_QT = True
except ImportError:                                    # pragma: no cover
    _HAVE_QT = False


if _HAVE_QT:

    class EngineWorker(QObject):
        """Runs one CortexSession inside a background QThread. The
        queue-backed y_source blocks the worker thread only — the GUI
        thread stays live. Engine numerics are untouched."""

        itemReady = pyqtSignal(dict)        # before each answer
        trialDone = pyqtSignal(dict)        # after each answer (telemetry)
        sessionComplete = pyqtSignal(object)   # SessionResult
        sessionFailed = pyqtSignal(str)

        def __init__(self, session: CortexSession, answer_queue: queue.Queue):
            super().__init__()
            self._session = session
            self._queue = answer_queue

        def _y_source(self, k, seg_id, s):
            """Block on the GUI->engine queue. The GUI puts the rater's raw
            0-based 6-way IIIC pick; the one-vs-rest reduction happens here
            with the exact task k the engine asked: Y = 1 iff pick == k."""
            raw = self._queue.get()
            if raw is _SENTINEL_ABORT:
                raise SessionAborted()
            return int(int(raw) == int(k))

        def run(self):
            try:
                result = self._session.run(
                    self._y_source,
                    on_item=self.itemReady.emit,
                    on_trial=self.trialDone.emit)
                self.sessionComplete.emit(result)
            except Exception as exc:                   # pragma: no cover
                self.sessionFailed.emit(repr(exc))

    class SessionController(QObject):
        """Owns the session_id, the Phase-A engine inputs, the EngineWorker
        + its QThread, and the GUI->engine answer queue. The viewer connects
        to itemReady / trialDone / sessionComplete and calls submit_answer.
        """

        itemReady = pyqtSignal(dict)
        trialDone = pyqtSignal(dict)
        sessionComplete = pyqtSignal(object)
        sessionFailed = pyqtSignal(str)

        def __init__(self, inputs, session_id, parent=None, **session_kw):
            super().__init__(parent)
            self.inputs = inputs
            self.session_id = str(session_id)
            self.session = CortexSession(inputs, session_id=self.session_id,
                                         **session_kw)
            self._queue = queue.Queue(maxsize=1)
            self._thread = QThread()
            self._worker = EngineWorker(self.session, self._queue)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.itemReady.connect(self.itemReady)
            self._worker.trialDone.connect(self.trialDone)
            self._worker.sessionComplete.connect(self._on_complete)
            self._worker.sessionFailed.connect(self.sessionFailed)

        def start(self):
            """Start the engine thread; the first itemReady follows shortly."""
            self._thread.start()

        def submit_answer(self, raw_choice: int):
            """Hand the rater's raw 0-based 6-way IIIC pick to the engine."""
            self._queue.put(int(raw_choice))

        def abort(self):
            """End the session early (e.g. the window closed mid-test)."""
            self._queue.put(_SENTINEL_ABORT)

        def _on_complete(self, result):
            self._thread.quit()
            self._thread.wait()
            self.sessionComplete.emit(result)


if __name__ == "__main__":
    # Smoke run: one simulated clearly-pass rater, adaptive selection.
    inp = build_iiic_engine_inputs()
    K = len(inp.task_codes)
    sess = CortexSession(inp, session_id="smoke", n_particles=600)
    res = sess.run(make_simulated_y_source(np.zeros(K), np.full(K, 0.6)))
    print(f"smoke: n_q={res.n_questions}  stop={res.stop_reason}  "
          f"max_hw={float(res.final_auroc_hw.max()):.4f}  "
          f"auroc_mean={np.round(res.final_auroc_mean, 3).tolist()}")
