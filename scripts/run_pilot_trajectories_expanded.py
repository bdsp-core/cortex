"""Expanded preliminary F4.2 visuals (pre-full-run exploration).

Fig A — within-group stochastic variability (spaghetti):
  100 independent stochastic realizations each of Expert / Experienced /
  Novice (same true skill ℓ per tier; variability is pure response +
  SMC-seed stochasticity), hier, K=6.  Thin translucent per-session
  trajectories + a bold tier-mean overlay → shows how widely individual
  raters of the *same* competence spread in questions-to-precision.

Fig B — method comparison across all three skill tiers (small multiples):
  One panel per tier (Expert / Experienced / Novice); within each,
  Hierarchical / Independent (brute) / Random as mean ± IQR in the same
  questions-vs-max-HW view.  random→brute isolates adaptive (Global-EV)
  item selection; brute→hier isolates hierarchical cross-domain pooling;
  the panels show how those gains scale with rater skill.

Both use the production engine (hier: fitted Corr_l prior; brute/random:
K independent N(0,1) posteriors; random = uniform item-selection null).
Output: results/phase4_simstudy/fig_pilot_spaghetti.{pdf,png}
        results/phase4_simstudy/fig_pilot_methods.{pdf,png}
"""
from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D  # noqa: E402

ENGINE_REPO = os.path.dirname(_THIS)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)
from core_mcmc import run_session_mcmc_auroc, load_fitted_Sigma  # noqa: E402
from auroc import auroc_from_l  # noqa: E402

OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase4_simstudy")
os.makedirs(OUT_DIR, exist_ok=True)

K = 6
N_PARTICLES = 1000
DELTAS = [0.10, 0.05, 0.025]

# Tiers (shared by both figures).  Colors match the fig_pilot_trajectory
# scheme (Okabe-Ito): Expert=green, Experienced=blue, Novice=orange.
# Experienced ℓ=0.50 ≈ AUROC 0.89.
TIERS = [
    {"name": "Expert",      "l":  1.00, "color": "#009E73"},  # green
    {"name": "Experienced", "l":  0.50, "color": "#0072B2"},  # blue
    {"name": "Novice",      "l": -1.50, "color": "#E69F00"},  # orange
]

# Fig A
N_RATERS = 100
MAX_Q_A = 3000

# Fig B (canonical project method palette; small multiples per tier).
# Per-tier question cap, sized from the F4.2 pilot (K=6, single-seed
# δ=0.025 stop: Expert slowest=random@928; Experienced(ℓ=0)@3161;
# Novice@5164) with ~1.5–1.8× margin so every method's 6-seed *mean*
# trajectory reaches δ=0.025 within budget.  Independent per-panel
# x-axis (shared y) auto-scales each panel to its own δ=0.025 reach.
N_SEEDS_B = 6
MAX_Q_B_BY_TIER = {"Expert": 2000, "Experienced": 4000, "Novice": 9000}
METHOD_STYLE = {
    "random": {"color": "#7f7f7f", "label": "Random (null)"},
    "brute":  {"color": "#D55E00", "label": "Independent (adaptive)"},
    "hier":   {"color": "#0072B2", "label": "Hierarchical (Corr$_l$)"},
}
METHOD_ORDER = ["random", "brute", "hier"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.0,
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
    method = task["method"]
    true_params: List[float] = []
    for _ in range(K):
        true_params += [0.0, l_val]
    kw = dict(method=method, true_params=true_params, K=K, r_assumed=0.378,
              max_q=task["max_q"], delta_auroc=min(DELTAS), N=N_PARTICLES,
              seed=task["seed"], run_until_max=True, log_trajectory=True,
              ess_threshold_frac=0.5)
    if method == "hier":
        kw.update(Sigma_l=_CORR, Sigma_t=_CORR,
                  proposal_scale=2.38 / np.sqrt(2 * K))
    out = run_session_mcmc_auroc(**kw)
    hw_max = ((out["hi_traj"] - out["lo_traj"]) / 2.0).max(axis=1)
    return {"group": task["group"], "tier": task["tier"],
            "method": method, "seed": task["seed"],
            "hw_max": hw_max.tolist()}


def first_crossing(curve: np.ndarray, d: float):
    idx = np.where(curve[1:] < d)[0]
    return int(idx[0]) + 1 if idx.size else None


def _delta_lines(ax, x_right, label=True):
    for d in DELTAS:
        ax.axhline(d, color="#bdbdbd", ls=":", lw=1.0, zorder=1)
        if label:
            ax.text(x_right * 0.995, d, f" $\\delta$={d:g}", va="center",
                    ha="right", fontsize=7.0, color="#666666",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.8))


def make_fig_a(by_tier):
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    handles = []
    n_raters = T = None
    for t in TIERS:
        stack = np.vstack(by_tier[t["name"]])           # (n_raters, T)
        n_raters, T = stack.shape
        qs = np.arange(T)
        for row in stack:
            ax.plot(qs, row, color=t["color"], lw=1.0, alpha=1.0,
                    zorder=3, solid_capstyle="round")
        auroc = float(auroc_from_l(np.array([t["l"]]))[0])
        handles.append(Line2D([0], [0], color=t["color"], lw=2.2,
                              label=f"{t['name']}  ($\\ell={t['l']:+.2f}$, "
                                    f"AUROC$\\approx${auroc:.2f})"))
    x_right = T - 1
    _delta_lines(ax, x_right)
    ax.set_xlabel("Questions administered")
    ax.set_ylabel("max$_k$  half-width$_{0.95}$(AUROC$_k$)")
    ax.set_title(f"Within-group stochastic variability: {n_raters} "
                 f"raters per tier\n(hierarchical, K={K} SPARCNET "
                 f"domains, N={N_PARTICLES}; each line = one rater)",
                 loc="left", fontweight="bold", pad=10)
    ax.set_xlim(0, x_right); ax.set_ylim(0, None); ax.margins(x=0)
    ax.legend(handles=handles, loc="upper right", frameon=False)
    cap = ("All raters within a tier have identical true skill; the "
           "spread is pure response + SMC stochasticity. It shows how "
           "much two equally-skilled raters can differ in "
           "questions-to-precision — the operating-characteristic "
           "variability the certification budget must absorb.")
    fig.text(0.012, -0.055, cap, fontsize=7.0, color="#444444", wrap=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_pilot_spaghetti.{ext}"))
    plt.close(fig)


def make_fig_b(by_tier_method):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharey=True)
    crossings: Dict[str, Any] = {}
    for ax, t in zip(axes, TIERS):
        tier = t["name"]
        crossings[tier] = {}
        means = {}
        for m in METHOD_ORDER:
            stack = np.vstack(by_tier_method[tier][m])   # (N_SEEDS_B, T)
            mean = stack.mean(axis=0)
            means[m] = mean
            qs = np.arange(mean.shape[0])
            q25, q75 = np.percentile(stack, [25, 75], axis=0)
            c = METHOD_STYLE[m]["color"]
            ax.plot(qs, mean, color=c, label=METHOD_STYLE[m]["label"],
                    zorder=4)
            ax.fill_between(qs, q25, q75, color=c, alpha=0.15, lw=0,
                            zorder=2)
            crossings[tier][m] = {}
            for d in DELTAS:
                qc = first_crossing(mean, d)
                crossings[tier][m][str(d)] = qc
                if qc is not None:
                    ax.scatter([qc], [d], s=22, color=c,
                               edgecolor="white", linewidth=0.6, zorder=5)
        # Independent x-axis: scale this panel to its own δ=0.025 reach
        # (the slowest method that reaches it), with headroom.  Fall back
        # to the full trajectory length only if some method never does.
        d025 = [crossings[tier][m]["0.025"] for m in METHOD_ORDER]
        full_len = max(means[m].shape[0] for m in METHOD_ORDER) - 1
        if all(v is not None for v in d025):
            x_right = int(min(full_len, max(d025) * 1.12))
        else:
            x_right = full_len
        _delta_lines(ax, x_right, label=(ax is axes[-1]))
        auroc = float(auroc_from_l(np.array([t["l"]]))[0])
        ax.set_title(f"{tier}  ($\\ell={t['l']:+.2f}$, "
                     f"AUROC$\\approx${auroc:.2f})", fontsize=10,
                     fontweight="bold", pad=6)
        ax.set_xlabel("Questions administered")
        ax.set_xlim(0, x_right); ax.set_ylim(0, None); ax.margins(x=0)
    axes[0].set_ylabel("max$_k$  half-width$_{0.95}$(AUROC$_k$)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Method comparison across skill tiers "
                 f"(K={K} SPARCNET domains, N={N_PARTICLES}, "
                 f"mean $\\pm$ IQR over {N_SEEDS_B} seeds)",
                 x=0.012, ha="left", fontweight="bold", fontsize=11.5,
                 y=1.04)
    cap = ("Same questions-vs-max-HW view as the tier figure, faceted by "
           "skill. random$\\rightarrow$brute = value of adaptive "
           "(Global-EV) item selection; brute$\\rightarrow$hier = value "
           "of hierarchical cross-domain "
           "pooling. Markers: mean-curve δ crossings (the precision-"
           "certification 'pass' event). Shared y-axis; each panel's "
           "x-axis is independently scaled to that tier's δ=0.025 reach "
           "(note the very different question budgets per skill level).")
    fig.text(0.012, -0.04, cap, fontsize=7.0, color="#444444", wrap=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_pilot_methods.{ext}"))
    plt.close(fig)
    return crossings


def main():
    global N_RATERS, MAX_Q_A, N_SEEDS_B
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-raters", type=int, default=N_RATERS)
    ap.add_argument("--max-q-a", type=int, default=MAX_Q_A)
    ap.add_argument("--n-seeds-b", type=int, default=N_SEEDS_B)
    ap.add_argument("--q-scale", type=float, default=1.0,
                    help="Scale all per-tier Fig-B caps (smoke test).")
    ap.add_argument("--replot", action="store_true",
                    help="Skip all compute; rebuild both figures from "
                         "the cached trajectories (instant — for "
                         "layout/colour/caption tweaks).")
    cli = ap.parse_args()
    N_RATERS, MAX_Q_A, N_SEEDS_B = cli.n_raters, cli.max_q_a, cli.n_seeds_b
    cap_b = {t["name"]: max(50, int(MAX_Q_B_BY_TIER[t["name"]]
                                    * cli.q_scale)) for t in TIERS}
    cache_path = os.path.join(OUT_DIR, "pilot_expanded_cache.npz")

    if cli.replot:
        if not os.path.exists(cache_path):
            raise SystemExit(f"--replot: no cache at {cache_path}; run "
                             "once without --replot first.")
        z = np.load(cache_path)
        by_tier = {t["name"]: list(z[f"A__{t['name']}"]) for t in TIERS}
        by_tier_method = {
            t["name"]: {m: list(z[f"B__{t['name']}__{m}"])
                        for m in METHOD_ORDER} for t in TIERS}
        print(f"[replot] loaded cache {cache_path}", flush=True)
        make_fig_a(by_tier)
        crossings = make_fig_b(by_tier_method)
        print("  → regenerated fig_pilot_spaghetti.{pdf,png}, "
              "fig_pilot_methods.{pdf,png} from cache", flush=True)
        for tier in [t["name"] for t in TIERS]:
            for m in METHOD_ORDER:
                print(f"    {tier:<12} {m:<7} {crossings[tier][m]}",
                      flush=True)
        return

    tasks: List[Dict[str, Any]] = []
    for t in TIERS:
        for i in range(N_RATERS):
            tasks.append({"group": "A", "tier": t["name"], "l": t["l"],
                          "method": "hier", "seed": 2000 + i,
                          "max_q": MAX_Q_A})
    for t in TIERS:
        for m in METHOD_ORDER:
            for s in range(N_SEEDS_B):
                tasks.append({"group": "B", "tier": t["name"],
                              "l": t["l"], "method": m, "seed": 3000 + s,
                              "max_q": cap_b[t["name"]]})

    print(f"=== expanded pilot: {len(tasks)} sessions "
          f"(A: {3*N_RATERS} spaghetti @≤{MAX_Q_A}q, "
          f"B: {3*3*N_SEEDS_B} method×tier, caps={cap_b}) ===",
          flush=True)
    res = parallel_map(_worker, tasks, desc="sessions",
                       ordered=True, progress_every=25)

    by_tier: Dict[str, List[np.ndarray]] = {t["name"]: [] for t in TIERS}
    by_tier_method: Dict[str, Dict[str, List[np.ndarray]]] = {
        t["name"]: {m: [] for m in METHOD_ORDER} for t in TIERS}
    for r in res:
        if isinstance(r, dict) and "__error__" in r:
            raise RuntimeError(f"worker failed: {r['__error__']}")
        arr = np.asarray(r["hw_max"], dtype=float)
        if r["group"] == "A":
            by_tier[r["tier"]].append(arr)
        else:
            by_tier_method[r["tier"]][r["method"]].append(arr)

    cache = {f"A__{tn}": np.vstack(by_tier[tn]) for tn in by_tier}
    for tn, mm in by_tier_method.items():
        for m, lst in mm.items():
            cache[f"B__{tn}__{m}"] = np.vstack(lst)
    np.savez_compressed(cache_path, **cache)
    print(f"  cached trajectories → {cache_path} "
          f"(use --replot for instant figure tweaks)", flush=True)

    make_fig_a(by_tier)
    crossings = make_fig_b(by_tier_method)
    with open(os.path.join(OUT_DIR, "pilot_expanded_meta.json"), "w") as f:
        json.dump({"K": K, "N_PARTICLES": N_PARTICLES,
                   "N_RATERS": N_RATERS, "MAX_Q_A": MAX_Q_A,
                   "N_SEEDS_B": N_SEEDS_B, "MAX_Q_B_caps": cap_b,
                   "tiers": TIERS,
                   "method_crossings_qc": crossings}, f, indent=2)
    print("  → fig_pilot_spaghetti.{pdf,png}, "
          "fig_pilot_methods.{pdf,png}", flush=True)
    print("  method δ-crossings (mean curve, q):", flush=True)
    for tier in [t["name"] for t in TIERS]:
        for m in METHOD_ORDER:
            print(f"    {tier:<12} {m:<7} {crossings[tier][m]}",
                  flush=True)


if __name__ == "__main__":
    main()
