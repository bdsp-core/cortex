"""M20 — sandbox state persistence (npz + json; NO pickle, repo rule).

Persists everything that matters STATISTICALLY across sessions:
  per task   mixture strata clouds (θ/ℓ/w/learn per stratum) + log-weights;
             e-gate (log_e, n, fired, e_now); probe cadence counters; mode-
             policy continuity bits (label balance, mirror side/partner);
             GapAnchor stability (S, LS accumulators, gap history).
  global     served seg_ids (serve-once), retention-scheduler bins, global
             trial/session counters, last-session-end epoch, mastery
             lifecycle (provisional/confirmed epochs, D33).
Deliberately NOT persisted: RNG streams (fresh seeds per session — MC noise,
not statistics; documented) and per-session ephemera (deficiency-scheduler
finish/cooldown state).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from sandbox import config as C
from training.gap_anchor import MixtureGapAnchor
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import EProcessGate

STATE_NPZ = os.path.join(C.STATE_DIR, "state.npz")
STATE_JSON = os.path.join(C.STATE_DIR, "state.json")


def fresh_state(seed=20260701):
    """First-run initialization: fresh-learner beliefs per task."""
    rng = np.random.default_rng(seed)
    filters, gates, anchors = {}, {}, {}
    for t in C.TASKS:
        th0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
        el0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
        filters[t] = SigmaInfMixtureFilter(
            th0, el0, C.assumed_params(t), ell_inf_mean=C.EXPERT_ELL[t],
            tau=C.MIX_TAU, J=C.MIX_J, p_static_stratum=C.P_STATIC_STRATUM,
            ell_star=C.ELL_STAR[t], seed=seed + 31 * t)
        gates[t] = EProcessGate(alpha=0.05)
        anchors[t] = MixtureGapAnchor()
    meta = {
        "created": time.time(), "session_no": 0, "global_trial": 0,
        "last_session_end": None, "served": [],
        "provisional": {}, "confirmed": {}, "ever_declared": [],
        "retention_next": {}, "retention_ivl": {},
        "probe_ctr": {str(t): 0 for t in C.TASKS},
        "probe_flip": {str(t): 1 for t in C.TASKS},
        "bias_balance": {str(t): 0 for t in C.TASKS},
        "skill_side": {str(t): 1 for t in C.TASKS},
        "last_skill_s": {str(t): None for t in C.TASKS},
        "tasks": list(C.TASKS),
        # M21 (F73): σ units depend on the render physics — block resume
        # across a stimulus recalibration (like a task rescope).
        "render": [C.RENDER_GAIN, C.RENDER_NOISE, C.RENDER_N],
        "user": C.USER,
    }
    return filters, gates, anchors, meta


def save_state(filters, gates, anchors, meta):
    os.makedirs(C.STATE_DIR, exist_ok=True)
    arrs = {}
    for t in C.TASKS:
        f = filters[t]
        arrs[f"t{t}_logw"] = f.log_w
        arrs[f"t{t}_grid"] = f.grid
        for j, s in enumerate(f.strata):
            arrs[f"t{t}_s{j}_theta"] = s.theta
            arrs[f"t{t}_s{j}_ell"] = s.ell
            arrs[f"t{t}_s{j}_w"] = s.w
            arrs[f"t{t}_s{j}_learn"] = s.learn
        g = gates[t]
        arrs[f"t{t}_egate"] = np.array([*g.log_e, g.n, float(g.fired),
                                        g.e_now])
        a = anchors[t]
        arrs[f"t{t}_anchor"] = np.array([a.S, a._ss_num, a._ss_den])
        arrs[f"t{t}_anchor_hist"] = (np.array(a.history)
                                     if a.history else np.zeros((0, 2)))
    np.savez(STATE_NPZ, **arrs)
    with open(STATE_JSON, "w") as fh:
        json.dump(meta, fh, indent=1)


def load_state():
    if not (os.path.exists(STATE_NPZ) and os.path.exists(STATE_JSON)):
        return None
    with open(STATE_JSON) as fh:
        meta = json.load(fh)
    if meta.get("tasks") != list(C.TASKS):
        raise RuntimeError(
            f"state was created for tasks {meta.get('tasks')} but config now "
            f"says {list(C.TASKS)} — run with --reset to archive and rescope")
    render_now = [C.RENDER_GAIN, C.RENDER_NOISE, C.RENDER_N]
    # states older than M21 carry no render stamp — they were measured at
    # the M20 constants (gain 0.55)
    meta.setdefault("render", [0.55, 1.0, 48])
    if list(meta["render"]) != render_now:
        raise RuntimeError(
            f"state was measured under render params {meta['render']} but "
            f"config now says {render_now} — σ units are not comparable "
            f"across a stimulus recalibration; run with --reset (M21/F73)")
    z = np.load(STATE_NPZ)
    filters, gates, anchors = {}, {}, {}
    seed = int(meta.get("session_no", 0)) * 7919 + 11
    for t in C.TASKS:
        f = SigmaInfMixtureFilter(
            np.zeros(C.N_PARTICLES), np.zeros(C.N_PARTICLES),
            C.assumed_params(t), ell_inf_mean=C.EXPERT_ELL[t],
            tau=C.MIX_TAU, J=C.MIX_J, p_static_stratum=C.P_STATIC_STRATUM,
            ell_star=C.ELL_STAR[t], seed=seed + 31 * t)
        f.log_w = z[f"t{t}_logw"].copy()
        assert np.allclose(f.grid, z[f"t{t}_grid"]), "grid drift vs config"
        for j, s in enumerate(f.strata):
            s.theta = z[f"t{t}_s{j}_theta"].copy()
            s.ell = z[f"t{t}_s{j}_ell"].copy()
            s.w = z[f"t{t}_s{j}_w"].copy()
            s.learn = z[f"t{t}_s{j}_learn"].copy()
        filters[t] = f
        g = EProcessGate(alpha=0.05)
        e = z[f"t{t}_egate"]
        g.log_e = e[:len(g.LAMBDAS)].copy()
        g.n = int(e[len(g.LAMBDAS)])
        g.fired = bool(e[len(g.LAMBDAS) + 1])
        g.e_now = float(e[len(g.LAMBDAS) + 2])
        gates[t] = g
        a = MixtureGapAnchor()
        a.S, a._ss_num, a._ss_den = (float(x) for x in z[f"t{t}_anchor"])
        a.history = [tuple(r) for r in z[f"t{t}_anchor_hist"]]
        anchors[t] = a
    return filters, gates, anchors, meta


def archive_state(tag="archive"):
    """--reset: move current state+logs aside (never delete user data)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = os.path.join(C.STATE_DIR, f"{tag}-{stamp}")
    os.makedirs(dst, exist_ok=True)
    moved = []
    for p in (STATE_NPZ, STATE_JSON, C.TRIALS_JSONL, C.SESSIONS_JSONL):
        if os.path.exists(p):
            os.rename(p, os.path.join(dst, os.path.basename(p)))
            moved.append(os.path.basename(p))
    return dst, moved
