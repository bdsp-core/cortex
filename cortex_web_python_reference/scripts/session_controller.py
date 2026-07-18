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

BLAS is pinned single-thread (the engine reproducibility contract) before
numpy is imported. The isolated vNext path adds fail-closed update handling and
diagnostics; the shipped selector objective remains unchanged.
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

# Frozen PyInstaller bundles: __file__ for a PYZ-loaded module does not
# resolve to a real filesystem path whose parent.parent is the data unpack
# root. sys._MEIPASS is set by the bootloader to that root.
if getattr(sys, "frozen", False):
    _REPO = sys._MEIPASS
else:
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_REPO, "engine"), os.path.join(_REPO, "scripts"),
           _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_mcmc import (  # noqa: E402
    make_state_hier, choose_item, update, ess, resample_and_rejuvenate,
    simulate_response, _expected_loss_vec, _p_response_yes,
    _coarse_to_fine_top_indices,
    _uncertainty_aware_coarse_to_fine_top_indices,
    PosteriorUpdateError)
from auroc import (  # noqa: E402
    auroc_mean_from_particles_hier, auroc_halfwidths_hier)
from cortex_engine_inputs_k7 import build_k7_engine_inputs  # noqa: E402
from cortex_policy_k7 import (  # noqa: E402
    AD6_POLICY_NAME,
    FROZEN_PRECISION_PROFILE,
    PRECISION_POLICY_NAME,
    default_policy_for_k7,
    policy_for_name_k7,
    resolve_termination_policy_name,
)
from cortex_policy import (  # noqa: E402
    PENDING, ACTIVE, ESTIMATE_COMPLETE)

# Phase-9 K=7: K=7 wins by default. Module remains K-agnostic via
# inputs.task_codes; the build_k7 loader supplies 7 tasks (spike + 6 IIIC).
# Back-compat aliases preserve the old names for callers that haven't
# migrated; the underlying loader is K=7 either way.
build_iiic_engine_inputs = build_k7_engine_inputs   # back-compat alias
default_policy_for = default_policy_for_k7          # back-compat alias

# ── engine hyperparameters (the methodology config — viz_smc_collapse.py) ──
# vNext reference profile: match the served v1.6/v15 browser manifest. The
# previous Python full-session default of 600 was a pre-existing parity defect.
N_PARTICLES = 1200
# v1.6 production termination: each unresolved domain may receive at most 60
# of its own questions. On reaching that budget it is removed from selection
# and finalized as REFER. This replaces the old fixed 500-question session
# sweep; `max_questions` remains only as an optional legacy/audit ceiling.
PER_DOMAIN_CAP_DEFAULT = 60
# v1.3.5: live-test default for the consecutive-same-domain cap. After this
# many questions in a row on one IIIC task the selector is forced to switch
# domains, to break up long single-domain runs (one rater hit 99 seizure in a
# row, session e46cc793). Chosen from sim_v1_3_5/run_consec_sweep.py (300
# heterogeneous-rater sessions): cap=12 capped the worst IIIC run from ~158 to
# 12 while IMPROVING all_resolved (93.3% -> 98.3%) and shortening worst-case
# sessions (p95 300 -> 213), with verdicts unchanged. Only the live test path
# sets this; CortexSession's own default is None (off) for tests/sims/audit.
# v1.3.6: lowered 12 -> 5 (Eli's call) for more variety and less exploitability
# in the late single-domain tail. Note: 5 is below the run_consec_sweep grid that
# validated 12; more frequent forced switches add cross-task evidence (which the
# sweep found HELPS resolution) at the cost of slightly more domain-hopping. Re-run
# sim_v1_3_5/run_consec_sweep.py at cap=5 if a fresh OC characterization is wanted.
MAX_CONSEC_SAME_DOMAIN_DEFAULT = 5
# Lab internal-test release (2026-06-04): keep asking questions PAST the point
# the AD6 rule would have stopped the session, up to the explicit audit ceiling,
# to gather per-domain data across expertise levels for retrospective recalibration
# of the PASS/FAIL/REFER thresholds (the MBW pilot showed they are likely too
# strict — corpus §11). The OFFICIAL verdict/AUROC/visualizations stay FROZEN at
# the v1.4.0 would-have-stopped point; the extra questions are logged + flagged
# post_decision in the report CSVs only. Only the live viewer path turns this on;
# CortexSession defaults it OFF so every test / sim / audit is byte-identical to
# v1.4.0. Set to False here to ship the public version (early-stop restored).
EXTENDED_DATA_COLLECTION_DEFAULT = True
# delta=0.15 is the legacy DeltaStop target (Mode-A / methodology path).
# AD6Policy (production) does not use delta. See docs/AD6_RESOLUTION.md.
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


@dataclass(frozen=True)
class ResponseRecord:
    """Additive raw-response envelope accepted by ``CortexSession.run``."""

    y: int
    raw_choice: int | None = None
    response_time_ms: float | None = None
    quality_metadata: dict | None = None


def _normalize_response(value, task_k: int) -> ResponseRecord:
    if isinstance(value, ResponseRecord):
        record = value
    elif isinstance(value, dict):
        raw_choice = value.get("raw_choice")
        y = value.get("y")
        if y is None and raw_choice is not None:
            y = int(int(raw_choice) == int(task_k))
        if y is None:
            raise ValueError("response mapping requires y or raw_choice")
        record = ResponseRecord(
            y=int(y),
            raw_choice=(None if raw_choice is None else int(raw_choice)),
            response_time_ms=(None if value.get("response_time_ms") is None
                              else float(value["response_time_ms"])),
            quality_metadata=value.get("quality_metadata"),
        )
    else:
        record = ResponseRecord(y=int(value))
    if record.y not in (0, 1):
        raise ValueError(f"binary response must be 0 or 1, got {record.y!r}")
    return record


@dataclass
class SessionResult:
    """Outcome of one adaptive session."""

    session_id: str
    seed: int
    selection: str                 # "adaptive" | "random"
    n_questions: int
    stop_reason: str               # "all_resolved" | "resolved_or_referred"
                                   # | "delta_reached" (legacy delta)
                                   # | "bank_exhausted" | "aborted"
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
    # Extended data-collection mode (lab internal-test release). When the
    # session keeps asking past the would-have-stopped point to gather more
    # per-domain calibration data, EVERY field above is FROZEN at the v1.4.0
    # decision point (so ResultsScreen / certificate / the AUROC+ROC
    # visualizations are byte-identical to v1.4.0); the additional questions
    # exist only in the recorder's trials.jsonl/CSV, flagged post_decision.
    # These two record the actual extended run for retrospective analysis.
    n_questions_total: int = None      # questions actually answered
    extended_stop_reason: str = None   # what ended the extended run
    # Additive PrecisionPolicy/reporting contract. AD6 leaves these None.
    domain_statuses: list = None
    determinations: list = None
    terminal_reasons: list = None
    skill_intervals: list = None
    bias_intervals: list = None
    invalid_reason: str = None
    # Optional campaign-only final cloud for stopping-time SBC ranks. Disabled
    # by default so the production/AD6 result contract remains unchanged.
    final_l_particles: np.ndarray = None
    final_weights: np.ndarray = None


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


def _predictive_yes_probability(state, k: int, s: float, s_sd: float) -> float:
    """Posterior-predictive P(Y=1) for model-fit telemetry."""
    w = np.asarray(state["w"], dtype=float)
    w = w / w.sum()
    el = np.exp(state["l"][:, int(k)])
    z = el * (float(s) + state["t"][:, int(k)])
    if float(s_sd) != 0.0:
        z = z / np.sqrt(1.0 + (el * float(s_sd)) ** 2)
    return float((w * _p_response_yes(z)).sum())


class CortexSession:
    """Pure, Qt-free K=6 IIIC adaptive session over the SMC engine."""

    def __init__(self, inputs, *, session_id="adhoc",
                 n_particles=N_PARTICLES, delta_auroc=None,
                 max_questions=None, ess_threshold_frac=ESS_THRESHOLD_FRAC,
                 n_mh_steps=N_MH_STEPS, seed=None, selection="adaptive",
                 policy=None, capture_clouds=False,
                 first_item_topn=FIRST_ITEM_TOPN,
                 max_consecutive_same_domain=None,
                 extended_data_collection=False, objective="total_var",
                 per_domain_cap=PER_DOMAIN_CAP_DEFAULT,
                 n_subsample=None, floor_progress_deadline=True,
                 uncertainty_aware_subsample=False,
                 capture_final_cloud=False):
        self.inputs = inputs
        self.session_id = str(session_id)
        self.N = int(n_particles)
        self.delta_auroc = float(delta_auroc) if delta_auroc is not None else None
        self.K = len(inputs.task_codes)
        n_bank = len(inputs.all_seg_ids)
        self.max_questions = (n_bank if max_questions is None
                              else min(int(max_questions), n_bank))
        self.per_domain_cap = int(per_domain_cap)
        if self.per_domain_cap <= 0:
            raise ValueError("per_domain_cap must be a positive integer")
        self.ess_threshold_frac = float(ess_threshold_frac)
        self.n_mh_steps = int(n_mh_steps)
        self.capture_clouds = bool(capture_clouds)
        self.capture_final_cloud = bool(capture_final_cloud)
        self.first_item_topn = max(1, int(first_item_topn))
        if selection not in ("adaptive", "random"):
            raise ValueError(f"selection must be adaptive|random, got {selection!r}")
        self.selection = selection
        # discrimination_rd Lever A: item-selection objective threaded to
        # choose_item. "total_var" (default) ⇒ byte-identical shipped behavior.
        self.objective = str(objective)
        self.n_subsample = (None if n_subsample is None
                            else int(n_subsample))
        if self.n_subsample is not None and self.n_subsample <= 0:
            raise ValueError("n_subsample must be a positive integer")
        self.floor_progress_deadline = bool(floor_progress_deadline)
        self.uncertainty_aware_subsample = bool(
            uncertainty_aware_subsample)
        if self.uncertainty_aware_subsample and self.n_subsample is None:
            raise ValueError(
                "uncertainty_aware_subsample requires n_subsample")
        if seed is None:
            seed = int(hashlib.sha256(self.session_id.encode()).hexdigest()[:8], 16)
        self.seed = int(seed)
        # AD6: the stopping check is a single injectable TerminationPolicy.
        # `default_policy_for` resolves precedence — explicit policy=, then
        # AD6Policy (delta_auroc=None, production), NoStopPolicy (==0.0,
        # audit/OC), DeltaStopPolicy (>0, legacy methods).
        self.policy = default_policy_for(
            inputs, delta_auroc=delta_auroc, policy=policy)
        self._uses_precision_status = bool(
            getattr(self.policy, "uses_domain_status", False))
        if self._uses_precision_status and self.objective != "total_var":
            raise ValueError(
                "PrecisionPolicy v1 requires the shipped total_var selector")
        if (self._uses_precision_status
                and int(getattr(self.policy, "per_domain_cap"))
                != self.per_domain_cap):
            raise ValueError(
                "PrecisionPolicy and CortexSession per_domain_cap must match")

        # v1.3.5 (default OFF): after this many consecutive questions on one
        # task, the selector is forced onto a different domain for the next
        # question, to break up long single-domain runs (UX). None preserves
        # the original behavior exactly. State reset at the start of run().
        self._max_consec = (int(max_consecutive_same_domain)
                            if max_consecutive_same_domain else None)
        self._last_task = None
        self._consec_count = 0
        # Extended data-collection (default OFF — see EXTENDED_DATA_COLLECTION_
        # DEFAULT). When ON, run() captures a v1.4.0 snapshot at the would-have-
        # stopped point then keeps asking (all domains active) to max_questions.
        self._extended = bool(extended_data_collection)
        if self._uses_precision_status and self._extended:
            raise ValueError(
                "extended_data_collection is disabled on the hard-cap "
                "PrecisionPolicy path")
        self._post_decision = False           # True once past the snapshot
        self._decision_snapshot = None        # frozen v1.4.0 result fields
        self._capped_tasks = set()            # unresolved domains at question budget
        self._capped_referrals = {}           # terminal REFER label fixed at cap
        self._n_per_task = [0] * self.K
        self._band_administered = [[0, 0, 0] for _ in range(self.K)]
        self._last_bank_telemetry = None
        self._last_floor_progress_domains = []
        self.proposal_scale = 2.38 / np.sqrt(2 * self.K)

    def _precision_bank_telemetry(self, bank_signals):
        """Current own-domain supply and band counts before reservations."""
        band_remaining = []
        for k, signals in enumerate(bank_signals):
            values = np.asarray(signals, dtype=float)
            if self.policy.band_edges is None:
                counts = [int(len(values)), 0, 0]
            else:
                bands = np.searchsorted(
                    self.policy.band_edges[k], values, side="right")
                counts = np.bincount(bands, minlength=3).astype(int).tolist()
            band_remaining.append(counts)
        deficits = [
            [max(0, int(self.policy.band_min) - int(value))
             for value in row]
            for row in self._band_administered
        ]
        return {
            "remaining_bank_counts": [int(len(row)) for row in bank_signals],
            "band_administered": [list(row) for row in self._band_administered],
            "band_remaining": band_remaining,
            "band_deficits": deficits,
        }

    def _prepare_precision_candidates(self, bank_signals, bank_sds,
                                      bank_segids):
        """Apply floor feasibility, reservation, and floor-progress filters."""
        telemetry = self._precision_bank_telemetry(bank_signals)
        self._last_bank_telemetry = telemetry
        self.policy.observe_bank_feasibility(
            telemetry, self._n_per_task, self.K)
        statuses = self.policy.domain_statuses
        deficits = telemetry["band_deficits"]
        supplies = telemetry["band_remaining"]

        # When supply exactly equals a still-required deficit, every such item
        # is reserved from other domains. Reservations are recomputed each turn.
        reserved_for: dict[int, set[int]] = {}
        if int(self.policy.band_min) > 0:
            for k, status in enumerate(statuses):
                if status.startswith("UNDETERMINABLE_"):
                    continue
                ids = np.asarray(bank_segids[k], dtype=int)
                signals = np.asarray(bank_signals[k], dtype=float)
                for band in range(3):
                    if deficits[k][band] <= 0 or supplies[k][band] != deficits[k][band]:
                        continue
                    for seg_id, signal in zip(ids, signals):
                        if self.policy.band_index(k, float(signal)) == band:
                            reserved_for.setdefault(int(seg_id), set()).add(k)

        floor_only = set(self.policy.floor_only_blocked_domains)
        # Do not wait until precision has already persisted to begin filling
        # content bands. Once outstanding deficit units consume all questions
        # remaining before N_min, every subsequent own-domain question must
        # reduce a deficit. This guarantees a feasible floor by the evidence
        # boundary without steering the selector earlier than necessary.
        if self.floor_progress_deadline:
            for k, status in enumerate(statuses):
                if status != ACTIVE or not any(deficits[k]):
                    continue
                questions_to_evidence = max(
                    0, int(self.policy.n_min) - int(self._n_per_task[k]))
                if (sum(int(value) for value in deficits[k])
                        >= questions_to_evidence):
                    floor_only.add(k)
        self._last_floor_progress_domains = sorted(floor_only)
        filtered_signals, filtered_sds, filtered_ids = [], [], []
        for k in range(self.K):
            signals = np.asarray(bank_signals[k], dtype=float)
            sds = np.asarray(bank_sds[k], dtype=float)
            ids = np.asarray(bank_segids[k], dtype=int)
            keep = np.ones(len(ids), dtype=bool)
            blocked_ids = [seg_id for seg_id, owners in reserved_for.items()
                           if k not in owners]
            if blocked_ids:
                keep &= ~np.isin(ids, np.asarray(blocked_ids, dtype=int))
            if k in floor_only:
                bands = np.searchsorted(
                    self.policy.band_edges[k], signals, side="right")
                keep &= np.asarray(deficits[k], dtype=int)[bands] > 0
            filtered_signals.append(signals[keep])
            filtered_sds.append(sds[keep])
            filtered_ids.append(ids[keep])
        return filtered_signals, filtered_sds, filtered_ids

    def _compute_active_domains(self, bank_signals):
        """K=7 phase-aware active-domains for the engine selector. Eli's
        v1.2.4 sectioning rule + Issue-4 empty-bank guard:

          Phase A (spike-first sectioning): if K=7 AND spike (k=0) verdict
            is still PENDING AND the spike bank is non-empty, return
            [0] only — the engine asks spike questions until AD6 locks
            the spike verdict OR the spike bank exhausts.

          Phase B (IIIC sectioning): once the spike phase has ended (verdict
            locked or pool empty), return all IIIC task indices (1..6)
            with non-empty banks AND PENDING verdicts.

          K=6 (legacy) / non-AD6 policies: return all k with non-empty
            banks (no phase awareness needed — IIIC banks are uniform).

        Also fixes Issue 4 — `engine/core_mcmc.py:choose_item` IndexError
        when a per-task bank empties (at N_MIN=20 the engine can ask all
        50 spike segs by ~trial 140 and crash on the 51st pick because
        spike's bank_signals[0] was shape (0,)). Phase-aware
        active_domains prevents the empty-bank pick.
        """
        K = self.K
        if self._uses_precision_status:
            statuses = self.policy.domain_statuses or [ACTIVE] * K
            # Preserve spike-first sectioning with status rather than AD6 cuts.
            if (K == 7 and statuses[0] == ACTIVE
                    and self._n_per_task[0] < self.per_domain_cap
                    and len(bank_signals[0]) > 0):
                return [0]
            base = [
                k for k in range(K)
                if (statuses[k] == ACTIVE
                    and self._n_per_task[k] < self.per_domain_cap
                    and len(bank_signals[k]) > 0)
            ]
            if (self._max_consec and self._last_task is not None
                    and self._consec_count >= self._max_consec
                    and self._last_task in base):
                others = [k for k in base if k != self._last_task]
                if others:
                    return others
                # Hard-cap precision semantics: only a reversible, currently
                # complete domain with remaining budget may absorb the variety
                # question. If none exists, relax variety and continue the lone
                # ACTIVE domain. Thus no domain ever exceeds its budget.
                variety = [
                    k for k in range(1, K)
                    if (k != self._last_task
                        and statuses[k] == ESTIMATE_COMPLETE
                        and self._n_per_task[k] < self.per_domain_cap
                        and len(bank_signals[k]) > 0)
                ]
                if variety:
                    return variety
            return base

        verdicts = (self.policy._verdicts if (hasattr(self.policy, "_verdicts")
                                               and self.policy._verdicts is not None)
                    else [PENDING] * K)
        # Extended data-collection (post-decision): the v1.4.0 stopping point has
        # already passed and its result is frozen. Keep ALL 7 domains active
        # (ignore verdict locks + spike-first sectioning) so the engine keeps
        # adaptively probing every domain for more calibration data. The
        # consecutive-cap variety rule below still applies.
        if self._post_decision:
            base = [k for k in range(K) if len(bank_signals[k]) > 0]
        else:
            # Phase A — K=7 spike sectioning (the spike block is deliberate; the
            # consecutive-cap does NOT apply here).
            if (K == 7 and verdicts[0] == PENDING
                    and 0 not in self._capped_tasks
                    and len(bank_signals[0]) > 0):
                return [0]
            # Phase B / K=6 — all unresolved tasks with non-empty banks
            base = [k for k in range(K)
                    if (len(bank_signals[k]) > 0
                        and verdicts[k] == PENDING
                        and k not in self._capped_tasks)]
        # v1.3.5 consecutive-same-domain cap (default OFF). After `_max_consec`
        # questions in a row on one task, force the NEXT question onto a
        # different domain. Prefer other UNRESOLVED tasks; if the dominant task
        # is the ONLY unresolved one, fall back to other IIIC tasks with
        # non-empty banks (even already-resolved) so a long single-domain run
        # is broken up. k>=1 keeps the variety question a 6-way IIIC (not a
        # spike yes/no). The forced question still updates the joint posterior.
        if (self._max_consec and self._last_task is not None
                and self._consec_count >= self._max_consec
                and self._last_task in base):
            others = [k for k in base if k != self._last_task]
            if others:
                return others
            variety = [k for k in range(1, K)
                       if k != self._last_task and len(bank_signals[k]) > 0]
            if variety:
                return variety
        return base

    def _select(self, state, remaining, rng, trial_index):
        """Pick the next (task k, seg_id, s, s_sd) + the chosen item's
        expected post-answer total posterior variance. Returns None if
        no domain is selectable (all resolved or all banks empty) — the
        run loop catches this and stops the session with
        stop_reason="all_active_resolved"."""
        inp = self.inputs
        self._last_best_expected_loss = None
        bank_signals, bank_sds, bank_segids = inp.as_engine_arrays(remaining)
        if self._uses_precision_status:
            bank_signals, bank_sds, bank_segids = (
                self._prepare_precision_candidates(
                    bank_signals, bank_sds, bank_segids))
        if self.selection == "adaptive":
            # v1.2.4: phase-aware active domains (spike-first sectioning +
            # empty-bank skip). See _compute_active_domains docstring.
            active_domains = self._compute_active_domains(bank_signals)
            if not active_domains:
                return None
            if trial_index == 0 and self.first_item_topn > 1:
                # Q1 varies per examinee — draw uniformly (session rng) from
                # the top-N most-informative items, not the single argmin
                # (which would be identical for every test-taker).
                k, seg_id, s, s_sd = self._pick_top_n(
                    state, bank_signals, bank_sds, bank_segids, rng,
                    active_domains=active_domains)
            else:
                k, s, s_sd, seg_id = choose_item(
                    state, bank_signals, bank_sds=bank_sds,
                    active_domains=active_domains,
                    n_subsample=self.n_subsample,
                    uncertainty_aware_subsample=(
                        self.uncertainty_aware_subsample),
                    return_sd=True, bank_segids=bank_segids,
                    objective=self.objective,
                    ell_star=getattr(self.policy, "ell_star", None))
        else:  # random null baseline — the Phase-1 ablation comparator
            bs, bsd, bseg = bank_signals, bank_sds, bank_segids
            active_domains = self._compute_active_domains(bs)
            if not active_domains:
                return None
            k = int(rng.choice(active_domains))
            idx = int(rng.integers(len(bs[k])))
            seg_id = int(bseg[k][idx])
            s, s_sd = float(bs[k][idx]), float(bsd[k][idx])
        k, seg_id, s, s_sd = int(k), int(seg_id), float(s), float(s_sd)
        # the engine's selection objective for the chosen item (telemetry /
        # audit only — _expected_loss_vec is read, the engine is not modified)
        expected_loss = float(_expected_loss_vec(
            state, k, np.array([s]), signal_sds=np.array([s_sd]),
            objective=self.objective,
            ell_star=getattr(self.policy, "ell_star", None))[0])
        if self.selection == "adaptive" and self._last_best_expected_loss is None:
            # choose_item is the exact argmin over the constrained candidate set.
            self._last_best_expected_loss = expected_loss
        return k, seg_id, s, s_sd, expected_loss

    def _pick_top_n(self, state, bank_signals, bank_sds, bank_segids, rng,
                     active_domains=None):
        """Trial-0 selection — uniformly pick among the `first_item_topn`
        items with the lowest expected posterior variance, so the opening
        question differs per examinee instead of being a fixed argmin.

        v1.2.4: respects `active_domains` (the K=7 phase-A spike-first
        sectioning rule). Defaults to all K domains if not supplied
        (back-compat with non-K=7 callers)."""
        scored = []                       # (expected_loss, task_k, idx)
        domains = active_domains if active_domains is not None else range(self.K)
        for k in domains:
            sigs = np.asarray(bank_signals[k])
            sds = np.asarray(bank_sds[k])
            loss_of = lambda ii: _expected_loss_vec(
                state, k, sigs[ii], signal_sds=sds[ii],
                objective=self.objective,
                ell_star=getattr(self.policy, "ell_star", None))
            if self.uncertainty_aware_subsample:
                top_indices = _uncertainty_aware_coarse_to_fine_top_indices(
                    loss_of, len(sigs), self.first_item_topn,
                    self.n_subsample, sds)
            else:
                top_indices = _coarse_to_fine_top_indices(
                    loss_of, len(sigs), self.first_item_topn,
                    self.n_subsample)
            losses = _expected_loss_vec(
                state, k, sigs[top_indices], signal_sds=sds[top_indices],
                objective=self.objective,
                ell_star=getattr(self.policy, "ell_star", None))
            for idx, loss in zip(top_indices, losses):
                scored.append((float(loss), k, int(idx)))
        scored.sort(key=lambda row: row[0])
        self._last_best_expected_loss = float(scored[0][0])
        n_top = min(self.first_item_topn, len(scored))
        _, k, idx = scored[int(rng.integers(n_top))]
        return (k, int(np.asarray(bank_segids[k])[idx]),
                float(np.asarray(bank_signals[k])[idx]),
                float(np.asarray(bank_sds[k])[idx]))

    def _make_decision_snapshot(self, state, trials, served, t_traj_len,
                                stop_reason, decision):
        """Freeze the result fields exactly as v1.4.0 would have produced them
        at the would-have-stopped point. Called ONCE (extended mode), the first
        time the session would have terminated. finalize_verdicts() returns a
        fresh list, so later (post-decision) policy updates can't leak in."""
        var_t, var_l, t_mean, l_mean = _posterior_summary(state)
        return {
            "n_questions": len(trials),
            "stop_reason": stop_reason,
            "verdicts": self._finalize_verdicts(),
            "policy_diagnostics": (decision.diagnostics
                                   if decision is not None else None),
            "final_auroc_mean": auroc_mean_from_particles_hier(state),
            "final_auroc_hw": auroc_halfwidths_hier(state, alpha=ALPHA),
            "final_l_mean": l_mean,
            "final_t_mean": t_mean,
            "trials": list(trials),
            "served_seg_ids": list(served),
            "t_traj_len": int(t_traj_len),
            **self._precision_result_fields(),
        }

    def _finalize_verdicts(self):
        """Finalize policy verdicts while preserving cap-triggered referrals.

        A domain that was still unresolved at its per-domain budget remains a
        referral even if later cross-domain evidence moves the joint cloud.
        Its estimates may continue updating, but the terminal status records
        that the domain itself exhausted its allowed question budget.
        """
        verdicts = self.policy.finalize_verdicts()
        if verdicts is None:
            return None
        out = list(verdicts)
        for task_k, referral in self._capped_referrals.items():
            out[task_k] = referral
        return out

    def _precision_result_fields(self):
        if not self._uses_precision_status:
            return {
                "domain_statuses": None, "determinations": None,
                "terminal_reasons": None, "skill_intervals": None,
                "bias_intervals": None,
            }
        diagnostics = self.policy.last_diagnostics or {}
        return {
            "domain_statuses": self.policy.finalize_statuses(),
            "determinations": self.policy.finalize_determinations(),
            "terminal_reasons": self.policy.terminal_reasons,
            "skill_intervals": diagnostics.get("skill_intervals"),
            "bias_intervals": diagnostics.get("bias_intervals"),
        }

    def run(self, y_source, on_item=None, on_trial=None):
        """Run the adaptive loop. ``y_source(k, seg_id, s) -> int`` returns
        the binary response; it may block (the live GUI path). ``on_item`` /
        ``on_trial`` are optional callbacks fired before/after each answer.
        """
        rng = np.random.default_rng(self.seed)
        inp = self.inputs
        bias_correlation = getattr(inp, "Corr_t", None)
        if bias_correlation is None:
            if self._uses_precision_status:
                raise ValueError("PrecisionPolicy requires a distinct Corr_t input")
            # Historical K=6 fixtures predate Corr_t. Production K=7 inputs and
            # every Precision session must provide it explicitly.
            bias_correlation = inp.Corr_l
        state = make_state_hier(self.N, self.K, R_ASSUMED, rng,
                                Sigma_l=inp.Corr_l, Sigma_t=bias_correlation)
        remaining = list(inp.all_seg_ids)
        trials, served = [], []
        t_traj, l_traj, w_traj = [], [], []
        stop_reason = "bank_exhausted"
        aborted = False
        invalid_reason = None
        cumulative_log_score = 0.0
        self.policy.reset(self.K)
        self._last_task = None        # v1.3.5: reset consecutive-cap state
        self._consec_count = 0
        self._post_decision = False   # extended mode: reset per run
        self._decision_snapshot = None
        self._capped_tasks = set()
        self._capped_referrals = {}
        n_per_task = [0] * self.K
        self._n_per_task = n_per_task
        self._band_administered = [[0, 0, 0] for _ in range(self.K)]
        self._last_bank_telemetry = None
        decision = None

        try:
            for trial_index in range(self.max_questions):
                if not remaining:
                    break
                t0 = time.perf_counter()
                _selected = self._select(state, remaining, rng, trial_index)
                # v1.2.4: _select returns None when no active domain has a
                # candidate (all banks empty for unresolved tasks OR all
                # tasks already resolved). Stop the session cleanly.
                if _selected is None:
                    # Extended mode: v1.4.0 would stop here (all_active_resolved).
                    # Freeze the snapshot, then re-select with ALL domains active
                    # so data-collection continues even past resolution.
                    if self._extended and self._decision_snapshot is None:
                        self._decision_snapshot = self._make_decision_snapshot(
                            state, trials, served, len(t_traj),
                            "all_active_resolved", decision)
                        self._post_decision = True
                        _selected = self._select(state, remaining, rng,
                                                 trial_index)
                    if _selected is None:
                        if self._uses_precision_status:
                            decision = self.policy.current_decision()
                            stop_reason = (decision.stop_reason if decision.stop
                                           else "all_active_resolved")
                        else:
                            stop_reason = ("all_active_resolved"
                                           if self._decision_snapshot is None
                                           else "bank_exhausted")
                        break
                k, seg_id, s, s_sd, expected_loss = _selected
                select_ms = (time.perf_counter() - t0) * 1000.0

                # v1.3.5: track consecutive same-domain run length so the next
                # _compute_active_domains can force a switch once it hits the
                # cap. (No-op for the default max_consec=None path.)
                if k == self._last_task:
                    self._consec_count += 1
                else:
                    self._last_task = k
                    self._consec_count = 1

                if on_item is not None:
                    on_item({"trial_index": trial_index, "task_k": k,
                             "task_code": inp.task_codes[k], "seg_id": seg_id,
                             "s_mean": s, "s_sd": s_sd})

                response = _normalize_response(
                    y_source(k, seg_id, s), k)   # blocks on the live path
                y = response.y
                predictive_p_yes = _predictive_yes_probability(
                    state, k, s, s_sd)
                observed_probability = (predictive_p_yes if y == 1
                                        else 1.0 - predictive_p_yes)
                log_score = float(np.log(max(observed_probability, 1e-300)))
                cumulative_log_score += log_score
                pre_var_t, pre_var_l, _, _ = _posterior_summary(state)
                expected_gain = float(pre_var_t + pre_var_l - expected_loss)
                remaining_best_gain = (
                    None if self._last_best_expected_loss is None
                    else float(pre_var_t + pre_var_l
                               - self._last_best_expected_loss))

                try:
                    update(state, k, s, y, s_sd=s_sd)
                except PosteriorUpdateError as exc:
                    # The cloud is still exactly the pre-answer cloud. Record
                    # the administered response and terminate fail-closed.
                    served.append(seg_id)
                    remaining.remove(seg_id)
                    n_per_task[k] += 1
                    if self._uses_precision_status:
                        band = self.policy.band_index(k, s)
                        self._band_administered[k][band] += 1
                    invalid_reason = str(exc)
                    stop_reason = "invalid_posterior_update"
                    failure_tel = {
                        "trial_index": trial_index, "task_k": k,
                        "task_code": inp.task_codes[k], "seg_id": seg_id,
                        "s_mean": s, "s_sd": s_sd, "response_y": y,
                        "raw_choice": response.raw_choice,
                        "response_time_ms": response.response_time_ms,
                        "quality_metadata": response.quality_metadata,
                        "update_applied": False,
                        "update_error": invalid_reason,
                        "predictive_p_yes": predictive_p_yes,
                        "log_score": log_score,
                        "cumulative_log_score": cumulative_log_score,
                        "n_per_task": list(n_per_task),
                        "post_decision": self._post_decision,
                    }
                    trials.append(failure_tel)
                    if on_trial is not None:
                        on_trial(failure_tel)
                    break

                rejuv = False
                rejuvenation_acceptance = None
                if ess(state["w"]) < self.ess_threshold_frac * self.N:
                    rejuvenation_acceptance = resample_and_rejuvenate(
                        state, rng, self.n_mh_steps, self.proposal_scale,
                        q_index=trial_index)
                    rejuv = True

                var_t, var_l, t_mean, l_mean = _posterior_summary(state)
                auroc_mean = auroc_mean_from_particles_hier(state)
                auroc_hw = auroc_halfwidths_hier(state, alpha=ALPHA)
                served.append(seg_id)
                remaining.remove(seg_id)
                n_per_task[k] += 1
                if self._uses_precision_status:
                    band = self.policy.band_index(k, s)
                    self._band_administered[k][band] += 1

                tel = {
                    "trial_index": trial_index, "task_k": k,
                    "task_code": inp.task_codes[k], "seg_id": seg_id,
                    "pattern_class_true": inp.pattern_class(seg_id),
                    "s_mean": s, "s_sd": s_sd, "response_y": y,
                    "raw_choice": response.raw_choice,
                    "response_time_ms": response.response_time_ms,
                    "quality_metadata": response.quality_metadata,
                    "update_applied": True,
                    "select_ms": select_ms,
                    "expected_loss_chosen": expected_loss,
                    "expected_gain_chosen": expected_gain,
                    "remaining_bank_best_gain": remaining_best_gain,
                    "predictive_p_yes": predictive_p_yes,
                    "log_score": log_score,
                    "cumulative_log_score": cumulative_log_score,
                    "total_var_t": var_t, "total_var_l": var_l,
                    "total_var": var_t + var_l,
                    "auroc_mean": auroc_mean.tolist(),
                    "auroc_hw": auroc_hw.tolist(),
                    "max_hw": float(auroc_hw.max()),
                    "t_post_mean": t_mean.tolist(),
                    "l_post_mean": l_mean.tolist(),
                    "ess": float(ess(state["w"])), "rejuv": rejuv,
                    "rejuvenation_acceptance": rejuvenation_acceptance,
                    "last_rejuvenation": state.get("_last_rejuvenation"),
                    "n_per_task": list(n_per_task),
                    # Extended data-collection: True once the session has passed
                    # the v1.4.0 would-have-stopped point. Always False in normal
                    # (non-extended) sessions. The recorder uses it to flag the
                    # extra calibration questions and to keep the OFFICIAL
                    # summary/certificate stats on the pre-decision trials only.
                    "post_decision": self._post_decision,
                }
                if self._uses_precision_status:
                    post_bank_signals, _, _ = inp.as_engine_arrays(remaining)
                    bank_telemetry = self._precision_bank_telemetry(
                        post_bank_signals)
                    self._last_bank_telemetry = bank_telemetry
                    tel.update(bank_telemetry)
                    tel["floor_progress_domains"] = list(
                        self._last_floor_progress_domains)

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
                if decision.domain_statuses is not None:
                    tel["domain_statuses"] = list(decision.domain_statuses)
                    tel["precision_streak_counts"] = list(
                        decision.streak_counts)
                    tel["terminal_reasons"] = list(decision.terminal_reasons)

                # v1.6 per-domain budget. Only verdict-producing policies use
                # this production behavior; NoStop/Delta remain full-trajectory
                # research comparators. A still-PENDING domain is capped after
                # its 60th own question, excluded from future selection, and
                # converted to REFER by finalize_verdicts() at session end.
                if decision.verdicts is not None:
                    newly_capped = []
                    for task_k, verdict in enumerate(decision.verdicts):
                        if (verdict == PENDING
                                and n_per_task[task_k] >= self.per_domain_cap):
                            if task_k not in self._capped_tasks:
                                newly_capped.append(task_k)
                                self._capped_tasks.add(task_k)
                    if newly_capped:
                        cap_labels = self.policy.finalize_verdicts()
                        for task_k in newly_capped:
                            self._capped_referrals[task_k] = cap_labels[task_k]
                    tel["capped_tasks"] = sorted(self._capped_tasks)

                trials.append(tel)
                if self.capture_clouds:
                    t_traj.append(state["t"].copy())
                    l_traj.append(state["l"].copy())
                    w_traj.append(state["w"].copy())
                if on_trial is not None:
                    on_trial(tel)

                all_resolved_or_capped = (
                    decision.verdicts is not None
                    and all(verdict != PENDING or task_k in self._capped_tasks
                            for task_k, verdict in enumerate(decision.verdicts)))
                should_stop = decision.stop or all_resolved_or_capped
                resolved_stop_reason = (
                    decision.stop_reason if decision.stop
                    else "resolved_or_referred")

                if should_stop:
                    if not self._extended:
                        stop_reason = resolved_stop_reason
                        break
                    # Extended mode: freeze the v1.4.0 result the first time the
                    # session would have stopped, then keep collecting. Once past
                    # the snapshot, decision.stop is ignored (verdicts are locked
                    # monotonically, so the official result never changes) — the
                    # loop now ends only at max_questions or bank exhaustion.
                    if self._decision_snapshot is None:
                        self._decision_snapshot = self._make_decision_snapshot(
                            state, trials, served, len(t_traj),
                            resolved_stop_reason, decision)
                        self._post_decision = True
        except SessionAborted:
            aborted = True
            stop_reason = "aborted"

        # A trial-0 bank failure still receives a posterior interval report from
        # the untouched initial cloud. Terminal bank states remain sticky.
        if (self._uses_precision_status and invalid_reason is None
                and self.policy.last_diagnostics is None):
            if self._last_bank_telemetry is None:
                bank_signals, _, _ = inp.as_engine_arrays(remaining)
                self._last_bank_telemetry = self._precision_bank_telemetry(
                    bank_signals)
            decision = self.policy(
                state, self._last_bank_telemetry, n_per_task, self.K)

        # Always finalize — partial PASS/FAIL verdicts already locked in stay,
        # PENDING tasks become REFER_BORDERLINE / REFER_UNINFORMATIVE. Safe to
        # call on aborted or zero-trial sessions; returns None on NoStop/Delta.
        n_total = len(trials)
        snap = self._decision_snapshot
        if snap is not None:
            # Extended data-collection: the OFFICIAL result is FROZEN at the
            # v1.4.0 would-have-stopped point. The extra (post_decision) trials
            # were streamed to on_trial and live only in the recorder's
            # trials.jsonl/CSV; here we surface only the frozen result + counts.
            clen = snap["t_traj_len"]
            result = SessionResult(
                session_id=self.session_id, seed=self.seed,
                selection=self.selection,
                n_questions=snap["n_questions"], stop_reason=snap["stop_reason"],
                delta_auroc=self.delta_auroc, n_particles=self.N,
                task_codes=list(inp.task_codes), trials=snap["trials"],
                served_seg_ids=snap["served_seg_ids"],
                final_auroc_mean=snap["final_auroc_mean"],
                final_auroc_hw=snap["final_auroc_hw"],
                final_t_mean=snap["final_t_mean"],
                final_l_mean=snap["final_l_mean"],
                t_traj=np.array(t_traj[:clen]) if t_traj else None,
                l_traj=np.array(l_traj[:clen]) if l_traj else None,
                w_traj=np.array(w_traj[:clen]) if w_traj else None,
                aborted=aborted,
                verdicts=snap["verdicts"],
                policy_diagnostics=snap["policy_diagnostics"],
                n_questions_total=n_total, extended_stop_reason=stop_reason,
                domain_statuses=snap["domain_statuses"],
                determinations=snap["determinations"],
                terminal_reasons=snap["terminal_reasons"],
                skill_intervals=snap["skill_intervals"],
                bias_intervals=snap["bias_intervals"],
                invalid_reason=invalid_reason)
            return result

        verdicts = self._finalize_verdicts()
        policy_diagnostics = (decision.diagnostics if decision is not None
                              else None)
        if self._uses_precision_status:
            policy_diagnostics = self.policy.last_diagnostics
        precision_fields = self._precision_result_fields()

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
            verdicts=verdicts, policy_diagnostics=policy_diagnostics,
            n_questions_total=n_total, extended_stop_reason=stop_reason,
            invalid_reason=invalid_reason,
            final_l_particles=(
                np.asarray(state["l"], float).copy()
                if self.capture_final_cloud else None),
            final_weights=(
                np.asarray(state["w"], float).copy()
                if self.capture_final_cloud else None),
            **precision_fields)


def build_cortex_session(inputs, *, termination_policy=None, **kwargs):
    """Build a locally configured session through the policy swap seam.

    ``CORTEX_TERMINATION_POLICY`` is read only here. Its safe default is AD6.
    Precision sessions receive the complete frozen controller profile; a
    caller cannot silently override one of those load-bearing settings.
    Direct ``CortexSession(policy=...)`` injection remains available to small
    unit tests and historical development replay.
    """

    selected = resolve_termination_policy_name(termination_policy)
    if "policy" in kwargs or "delta_auroc" in kwargs:
        raise ValueError(
            "build_cortex_session owns policy selection; use CortexSession "
            "directly for explicit research-policy injection"
        )
    if selected == PRECISION_POLICY_NAME:
        for key, frozen_value in FROZEN_PRECISION_PROFILE.session_kwargs().items():
            if key in kwargs and kwargs[key] != frozen_value:
                raise ValueError(
                    f"{PRECISION_POLICY_NAME} freezes {key}={frozen_value!r}; "
                    f"got {kwargs[key]!r}"
                )
            kwargs[key] = frozen_value
    elif selected != AD6_POLICY_NAME:  # defensive; resolver already validates
        raise ValueError(f"unsupported termination policy {selected!r}")
    policy = policy_for_name_k7(inputs, selected)
    return CortexSession(inputs, policy=policy, **kwargs)


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
            wait_started = time.perf_counter()
            payload = self._queue.get()
            if payload is _SENTINEL_ABORT:
                raise SessionAborted()
            elapsed_ms = (time.perf_counter() - wait_started) * 1000.0
            if isinstance(payload, dict):
                raw = int(payload["raw_choice"])
                response_time_ms = payload.get("response_time_ms")
                if response_time_ms is None:
                    response_time_ms = elapsed_ms
                quality_metadata = payload.get("quality_metadata")
            else:
                raw = int(payload)
                response_time_ms = elapsed_ms
                quality_metadata = None
            return {
                "y": int(raw == int(k)),
                "raw_choice": raw,
                "response_time_ms": float(response_time_ms),
                "quality_metadata": quality_metadata,
            }

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
            self.session = build_cortex_session(
                inputs, session_id=self.session_id, **session_kw)
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

        def submit_answer(self, raw_choice: int, *, response_time_ms=None,
                          quality_metadata=None):
            """Preserve the raw pick and optional response-quality metadata."""
            self._queue.put({
                "raw_choice": int(raw_choice),
                "response_time_ms": response_time_ms,
                "quality_metadata": quality_metadata,
            })

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
    sess = CortexSession(inp, session_id="smoke", n_particles=N_PARTICLES)
    res = sess.run(make_simulated_y_source(np.zeros(K), np.full(K, 0.6)))
    print(f"smoke: n_q={res.n_questions}  stop={res.stop_reason}  "
          f"max_hw={float(res.final_auroc_hw.max()):.4f}  "
          f"auroc_mean={np.round(res.final_auroc_mean, 3).tolist()}")
