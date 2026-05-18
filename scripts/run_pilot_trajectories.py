"""Preliminary F4.2 visual: Mode-A precision trajectory by skill tier.

The Multi-AUROC Precision Protocol has no binary pass/fail (that is the
deprecated Mode-B, kept for Paper 2).  Its native quantity is the
stopping statistic itself — max_k HW_{0.95}(AUROC_k) — which the test
drives below δ.  This plots that statistic vs question number for three
skill tiers, the directly interpretable "expert certifies fast, novice
slow" picture, with the δ ∈ {0.10, 0.05, 0.025} thresholds and the
question at which each tier's mean trajectory crosses them.

Quick dedicated mini-run (the F4.2 pilot stored only summary stop-steps,
not per-question trajectories): hier (production method), K=6 (SPARCNET
production), Σ_l=empirical, 3 tiers × N_SEEDS seeds, run_until_max so all
curves are full-length for a clean overlay.

  Novice       ℓ=-1.5  AUROC≈0.62
  Experienced  ℓ= 0.0  AUROC≈0.84
  Expert       ℓ=+1.0  AUROC≈0.91

Output: results/phase4_simstudy/fig_pilot_trajectory.{pdf,png}
        results/phase4_simstudy/pilot_trajectory_crossings.json
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
from _parallel import configure_blas_single_thread, parallel_map  # noqa: E402
configure_blas_single_thread()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ENGINE_REPO = os.path.dirname(_THIS)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)
from core_mcmc import run_session_mcmc_auroc, load_fitted_Sigma  # noqa: E402
from auroc import auroc_from_l  # noqa: E402

OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase4_simstudy")
os.makedirs(OUT_DIR, exist_ok=True)

K = 6
N_PARTICLES = 1000
MAX_Q = 1200
N_SEEDS = 5
DELTAS = [0.10, 0.05, 0.025]

TIERS = [
    {"name": "Novice",      "l": -1.5, "color": "#E69F00"},
    {"name": "Experienced", "l":  0.0, "color": "#0072B2"},
    {"name": "Expert",      "l":  1.0, "color": "#009E73"},
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

_CORR = np.asarray(
    load_fitted_Sigma(os.path.join(ENGINE_REPO, "Sigma_l_fitted.npy"))["Corr_l"],
    dtype=float)


def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    l_val = task["l"]
    true_params: List[float] = []
    for _ in range(K):
        true_params += [0.0, l_val]
    out = run_session_mcmc_auroc(
        method="hier", true_params=true_params, K=K, r_assumed=0.378,
        max_q=MAX_Q, delta_auroc=min(DELTAS), N=N_PARTICLES,
        seed=task["seed"], run_until_max=True, log_trajectory=True,
        ess_threshold_frac=0.5, Sigma_l=_CORR, Sigma_t=_CORR,
        proposal_scale=2.38 / np.sqrt(2 * K))
    hw_max = ((out["hi_traj"] - out["lo_traj"]) / 2.0).max(axis=1)  # (T,)
    return {"tier": task["tier"], "seed": task["seed"],
            "hw_max": hw_max.tolist()}


def first_crossing(curve: np.ndarray, d: float):
    """First q≥1 with max-HW < d (matches the production stop rule)."""
    idx = np.where(curve[1:] < d)[0]
    return int(idx[0]) + 1 if idx.size else None


def main():
    tasks = [{"tier": t["name"], "l": t["l"], "seed": s}
             for t in TIERS for s in range(N_SEEDS)]
    print(f"=== pilot trajectories: {len(tasks)} hier K={K} sessions "
          f"(run_until_max={MAX_Q}) ===", flush=True)
    res = parallel_map(_worker, tasks, desc="traj sessions",
                       ordered=True, progress_every=5)

    by_tier: Dict[str, List[np.ndarray]] = {t["name"]: [] for t in TIERS}
    for r in res:
        if isinstance(r, dict) and "__error__" in r:
            raise RuntimeError(f"worker failed: {r['__error__']}")
        by_tier[r["tier"]].append(np.asarray(r["hw_max"], dtype=float))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    qs = np.arange(MAX_Q + 1)
    crossings: Dict[str, Any] = {}

    for t in TIERS:
        stack = np.vstack(by_tier[t["name"]])          # (n_seeds, T)
        mean = stack.mean(axis=0)
        q25, q75 = np.percentile(stack, [25, 75], axis=0)
        auroc = float(auroc_from_l(np.array([t["l"]]))[0])
        lbl = (f"{t['name']}  ($\\ell={t['l']:+.1f}$, "
               f"AUROC$\\approx${auroc:.2f})")
        ax.plot(qs, mean, color=t["color"], label=lbl, zorder=3)
        ax.fill_between(qs, q25, q75, color=t["color"], alpha=0.15,
                        linewidth=0, zorder=2)
        crossings[t["name"]] = {}
        for d in DELTAS:
            qc = first_crossing(mean, d)
            crossings[t["name"]][str(d)] = qc
            if qc is not None:
                ax.scatter([qc], [d], s=26, color=t["color"],
                           edgecolor="white", linewidth=0.7, zorder=5)

    for d in DELTAS:
        ax.axhline(d, color="#bdbdbd", ls=":", lw=1.0, zorder=1)
        ax.text(MAX_Q * 0.995, d, f" $\\delta$={d:g}", va="center",
                ha="right",
                fontsize=7.5, color="#666666",
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.8))

    ax.set_xlabel("Questions administered")
    ax.set_ylabel("max$_k$  half-width$_{0.95}$(AUROC$_k$)")
    ax.set_title("Mode-A precision trajectory by skill tier\n"
                 f"(hierarchical, K={K} SPARCNET domains, "
                 f"N={N_PARTICLES}, mean $\\pm$ IQR over {N_SEEDS} seeds)",
                 loc="left", fontweight="bold", pad=10)
    ax.set_xlim(0, MAX_Q)
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", frameon=False)
    ax.margins(x=0)

    cap = ("Markers: question at which each tier's mean trajectory first "
           "crosses $\\delta$ (the Mode-A stopping event — the "
           "precision-certification analogue of 'pass'). Curves are "
           "full-length (run-to-cap) for overlay; a tier not reaching a "
           f"$\\delta$ within {MAX_Q} q means certification at that "
           "precision needs a larger budget — itself an operating "
           "characteristic.")
    fig.text(0.012, -0.06, cap, fontsize=7.0, color="#444444", wrap=True)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_pilot_trajectory.{ext}"))
    with open(os.path.join(OUT_DIR,
                           "pilot_trajectory_crossings.json"), "w") as f:
        json.dump({"K": K, "N_PARTICLES": N_PARTICLES, "MAX_Q": MAX_Q,
                   "N_SEEDS": N_SEEDS, "tiers": TIERS,
                   "crossings_qc": crossings}, f, indent=2)
    print("  → fig_pilot_trajectory.{pdf,png} + crossings JSON", flush=True)
    print("  crossings (q at first δ cross, mean curve):", flush=True)
    for tn, dd in crossings.items():
        print(f"    {tn:<12} {dd}", flush=True)


if __name__ == "__main__":
    main()
