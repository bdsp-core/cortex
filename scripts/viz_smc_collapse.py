"""SMC particle-collapse visualisation: capture per-question snapshots.

Mirrors the inner loop of `engine/core_mcmc.run_session_mcmc_auroc` (method
="hier") exactly, but additionally records the (t, l, w, hw_per_domain,
rejuv_flag) state after every observation so the cloud's collapse can be
animated.  No engine modifications — uses the public primitives:

    make_state_hier, choose_item, update, ess, resample_and_rejuvenate,
    simulate_response  (all from engine/core_mcmc.py)
    auroc_quantiles_from_particles_hier  (from engine/auroc.py)
    load_fitted_Sigma                    (engine/core_mcmc.py)
    load_raters / build_true_params / load_bank_signals  (bridge/_common.py)

Output: `results/viz/snapshots_<slug>.npz` per rater with keys:
    t_traj   shape (T, N, K)   — t-particles at each question (post-update,
                                 post-rejuvenation if it fired this q)
    l_traj   shape (T, N, K)   — l-particles
    w_traj   shape (T, N)      — weights (uniform on rejuv quanta)
    hw_traj  shape (T, K)      — per-domain AUROC halfwidths
    rejuv    shape (T,) bool   — True if a rejuvenation fired this q
    q_idx    shape (T,)        — question index 1..n_q (T = n_q)
    true_t   shape (K,)        — rater's fitted t per domain
    true_l   shape (K,)        — rater's fitted l per domain
    n_q      int               — final question count at stop
    stopped  bool              — δ-stop reached
    rater_name str

The snapshots also include a single PRIOR frame (q_idx = 0) so the animation
can show the initial uninformative cloud before any questions.

Run:
    .venv/bin/python scripts/viz_smc_collapse.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
# engine/core_mcmc.py uses flat sibling imports (`from auroc import ...`),
# so the `engine/` subdir must be on sys.path. ENGINE_REPO is also needed
# so `from bridge._common import ...` resolves as a package.
_ENGINE_SUBDIR = os.path.join(ENGINE_REPO, "engine")
for p in (_ENGINE_SUBDIR, ENGINE_REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from core_mcmc import (
    make_state_hier,
    choose_item,
    update,
    ess,
    resample_and_rejuvenate,
    simulate_response,
    load_fitted_Sigma,
)
from auroc import auroc_quantiles_from_particles_hier
from bridge._common import (
    _autodetect_banks_dir,
    _default_rater_matrix_path,
    load_bank_signals,
    load_raters,
    build_true_params,
)


# ───────────── configuration (matches scripts/run_mbw_trajectory.py) ─────────────
DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
K = 6
N_PARTICLES = 600
MAX_Q = 1500
DELTA_AUROC = 0.05            # user choice — 12 fps will yield ~10-20 s video
ALPHA = 0.05                  # 95% CI
SEED = 0
ESS_THRESHOLD_FRAC = 0.5
N_MH_STEPS = 15
PROPOSAL_SCALE = 2.38 / np.sqrt(2 * K)
R_ASSUMED = 0.378             # unused when Sigma_l is supplied (CS fallback only)

RATERS_TO_VIZ = [
    "M. Brandon Westover",
    "Marcus Ng",
    "Aaron F. Struck",
    "Aline Herlopian",
]

OUT_DIR = os.path.join(ENGINE_REPO, "results", "viz")


def _slugify(name: str) -> str:
    return name.lower().replace(".", "").replace(" ", "_")


def _hw_per_domain(state, alpha=ALPHA):
    """Per-domain AUROC halfwidth from the live state — see engine/auroc.py:87."""
    q = auroc_quantiles_from_particles_hier(state, alphas=(alpha / 2, 1 - alpha / 2))
    return (q[:, 1] - q[:, 0]) / 2.0


def run_session_with_snapshots(rater_name, true_params, Corr_l, bank_signals,
                                seed=SEED, max_q=MAX_Q, delta_auroc=DELTA_AUROC):
    """One hier SMC session for `rater_name` recording (t,l,w,hw,rejuv) per q.

    Returns a dict with all trajectory arrays (see module docstring).
    """
    rng = np.random.default_rng(seed)
    state = make_state_hier(N_PARTICLES, K, R_ASSUMED, rng,
                            Sigma_l=Corr_l, Sigma_t=Corr_l)

    true_t = np.array([true_params[2 * k]     for k in range(K)])
    true_l = np.array([true_params[2 * k + 1] for k in range(K)])

    # ── prior frame (q=0) ────────────────────────────────────────────────
    t_traj = [state["t"].copy()]
    l_traj = [state["l"].copy()]
    w_traj = [state["w"].copy()]
    hw_traj = [_hw_per_domain(state)]
    rejuv = [False]
    q_idx = [0]

    stop_step = None
    n_rejuv = 0

    for q in range(max_q):
        # ── choose item (full-bank EV, no subsample to keep BIT-IDENTICAL to scripts/run_mbw_trajectory) ──
        k_q, s_q = choose_item(state, bank_signals)
        # ── simulate y from this rater's fitted true (t_k, l_k) ──────────
        y_q = simulate_response(s_q, true_t[k_q], true_l[k_q], rng)
        # ── reweight ─────────────────────────────────────────────────────
        update(state, k_q, s_q, y_q)
        # ── rejuvenate if ESS drops below threshold ──────────────────────
        rj = False
        if ess(state["w"]) < ESS_THRESHOLD_FRAC * N_PARTICLES:
            resample_and_rejuvenate(state, rng, N_MH_STEPS, PROPOSAL_SCALE)
            rj = True
            n_rejuv += 1
        # ── snapshot AFTER reweight + optional rejuv ────────────────────
        t_traj.append(state["t"].copy())
        l_traj.append(state["l"].copy())
        w_traj.append(state["w"].copy())
        hw = _hw_per_domain(state)
        hw_traj.append(hw)
        rejuv.append(rj)
        q_idx.append(q + 1)
        # ── stopping ────────────────────────────────────────────────────
        if stop_step is None and float(np.max(hw)) < delta_auroc:
            stop_step = q + 1
            break

    return {
        "rater_name":   rater_name,
        "n_q":          stop_step if stop_step is not None else len(q_idx) - 1,
        "stopped":      stop_step is not None,
        "t_traj":       np.array(t_traj,  dtype=np.float32),
        "l_traj":       np.array(l_traj,  dtype=np.float32),
        "w_traj":       np.array(w_traj,  dtype=np.float32),
        "hw_traj":      np.array(hw_traj, dtype=np.float32),
        "rejuv":        np.array(rejuv,   dtype=bool),
        "q_idx":        np.array(q_idx,   dtype=np.int32),
        "true_t":       true_t.astype(np.float32),
        "true_l":       true_l.astype(np.float32),
        "n_rejuv":      n_rejuv,
        "delta_auroc":  float(delta_auroc),
        "N_particles":  N_PARTICLES,
        "domains":      np.array(DOMAINS),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    Corr_l = np.asarray(
        load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))["Corr_l"],
        dtype=float,
    )
    bank_signals = load_bank_signals(_autodetect_banks_dir(), DOMAINS)
    raters_df = load_raters(_default_rater_matrix_path())

    needed = [f"sigma_{d}" for d in DOMAINS] + [f"theta_{d}" for d in DOMAINS]
    raters_df = raters_df.dropna(subset=needed).reset_index(drop=True)

    for name in RATERS_TO_VIZ:
        sub = raters_df[raters_df["confirmed_canonical_name"] == name]
        if sub.empty:
            print(f"ERROR: rater {name!r} not found (or has NaN). Skipping.",
                  file=sys.stderr)
            continue
        row = sub.iloc[0]
        true_params = build_true_params(row, DOMAINS)
        print(f"\n▶ {name}: true_l = "
              f"{[round(true_params[2*k+1], 3) for k in range(K)]}")

        t0 = time.time()
        result = run_session_with_snapshots(
            name, true_params, Corr_l, bank_signals,
            seed=SEED, max_q=MAX_Q, delta_auroc=DELTA_AUROC,
        )
        dt = time.time() - t0
        print(f"   stopped={result['stopped']}  n_q={result['n_q']}  "
              f"n_rejuv={result['n_rejuv']}  ({dt:.1f}s)")

        slug = _slugify(name)
        out_path = os.path.join(OUT_DIR, f"snapshots_{slug}.npz")
        np.savez_compressed(out_path, **result)
        sz_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"   → {out_path}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
