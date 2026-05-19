"""Deployment-test figures 1–5 (analog of the single-task manuscript).

Adapts the manuscript figures 1–5 to the multi-task certification test.
Reads, from data/deployment_prior/:
  Sigma.csv (the frozen (2K)×(2K) prior), ell_thresholds.csv,
  case_bank.csv, sim/candidates.csv, sim/trajectories.npz.
Writes PNGs to data/deployment_prior/figures/.

UNIFIED-MERGE PROVENANCE (Phase 4.7, 2026-05-19). Port of the PI
scripts/plot_deploy.py. Documented changes vs PI:
  • hardcoded absolute-ROOT → engine_paths.DEPLOYMENT_PRIOR shim
    (zero absolute paths; mirrors deployment/simulate_test.py).
  • K-AGNOSTIC (was K=6-hardcoded: "12×12 prior", 2×3=6-panel grids):
    TASKS/K are now derived from the SHIPPED artifact via
    `deployment_task_names()` (K=7 post-4.6), and the per-task figure
    grids use `_grid()` (computed nrows×ncols, unused axes hidden) so
    figures are correct at any K. Concept figures 1/5 were already
    K-generic (range(K)/TASKS); they scale once K is artifact-derived.
  • PRETTY gains the real "other"/IIC task.
This is wired BEST-EFFORT by deployment/cli.py (a matplotlib/style
failure logs a warning and does NOT fail the freeze→simulate
pipeline — figures are derived viz, not gating artifacts).
"""
from __future__ import annotations
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

# ── path shim (mirrors deployment/simulate_test.py) ──
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_DEPLOY_DIR)
for _p in (os.path.join(_REPO, "engine"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import engine_paths  # noqa: E402
from deployment.simulate_test import deployment_task_names  # noqa: E402

DEPLOY = Path(engine_paths.DEPLOYMENT_PRIOR)
SIM = DEPLOY / "sim"
FIG = DEPLOY / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# Phase 4.7: TASKS/K from the SHIPPED artifact (K=7 post-4.6), not a
# hardcoded 6-list — the figures must match what is actually deployed.
TASKS = deployment_task_names()
PRETTY = {"spike": "Spike", "seizure": "Seizure", "lpd": "LPD",
          "gpd": "GPD", "lrda": "LRDA", "grda": "GRDA",
          "other": "Other/IIC"}
K = len(TASKS)
DIM = 2 * K


def _grid(n, base=(4.6, 3.7), sharex=False, sharey=False, ncols=None):
    """K-agnostic per-task panel grid: returns (fig, flat-axes list of
    length n). Replaces the PI hardcoded plt.subplots(2,3) (=6 panels).
    Unused trailing axes are removed so K≠6 renders cleanly."""
    ncols = ncols or min(4, n)
    nrows = math.ceil(n / ncols)
    fig, axarr = plt.subplots(
        nrows, ncols, figsize=(base[0] * ncols, base[1] * nrows),
        sharex=sharex, sharey=sharey, squeeze=False)
    flat = axarr.flat
    for j in range(n, nrows * ncols):       # hide unused panels
        fig.delaxes(flat[j])
    return fig, list(flat)[:n]

TIER_COLORS = {
    "expert":      "#1f77b4",
    "experienced": "#2ca02c",
    "novice":      "#ff7f0e",
    "crowd":       "#d62728",
}
DECISION_COLORS = {"pass": "#2ca02c", "fail": "#d62728", "refer": "#bcbd22"}


# ──────────────── Figure 1: model + prior concept ────────────────

def figure_1():
    Sigma = pd.read_csv(DEPLOY / "Sigma.csv", index_col=0).values
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(13, 8))
    # Layout: (A) curves; (B) one rater's profile; (C) sampled profiles.
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Panel A: P(yes|s) curves
    axA = fig.add_subplot(gs[0, 0])
    s_grid = np.linspace(-3, 3, 200)
    for ell, color in [(0.5, "#9ecae1"), (1.0, "#4292c6"), (1.5, "#08519c")]:
        axA.plot(s_grid, norm.cdf(np.exp(ell) * (s_grid - 0.0)),
                  color=color, lw=2, label=fr"$\ell={ell}$")
    axA.axhline(0.5, color="grey", lw=0.5, ls=":")
    axA.set_xlabel(r"case signal $s$")
    axA.set_ylabel(r"$P(\mathrm{yes})$")
    axA.set_title(r"A. Response curves at varying $\ell$  ($t=0$)")
    axA.legend(frameon=False, fontsize=9)
    axA.grid(alpha=0.3)

    # Panel B: example single-rater skill profile (purely ILLUSTRATIVE
    # values). Phase 4.7: K-agnostic — the PI 6-element literals are
    # cyclically resized to length K (np.resize) so K=6 is unchanged
    # and K=7+ extends deterministically (concept figure only).
    axB = fig.add_subplot(gs[0, 1])
    profile_ell = np.resize(
        np.array([1.3, 1.6, 0.9, 0.5, 1.2, 0.7]), K)
    profile_t   = np.resize(
        np.array([0.0, 0.1, 0.3, -0.2, 0.1, 0.0]), K)
    x = np.arange(K)
    axB.bar(x - 0.18, profile_ell, width=0.36, color="#4292c6", label=r"$\ell$")
    axB.bar(x + 0.18, profile_t,   width=0.36, color="#fb6a4a", label=r"$t$")
    axB.axhline(0, color="black", lw=0.4)
    axB.axhline(0.62, color="grey", lw=0.5, ls="--", label=r"$\ell^*$")
    axB.set_xticks(x); axB.set_xticklabels([PRETTY[t] for t in TASKS], rotation=20, ha="right")
    axB.set_ylabel("parameter value")
    axB.set_title(r"B. One rater's $(\ell_k, t_k)$ profile across tasks")
    axB.legend(frameon=False, fontsize=9, loc="upper right")
    axB.grid(alpha=0.3, axis="y")

    # Panel C: ℓ-block correlation matrix of the prior
    axC = fig.add_subplot(gs[0, 2])
    ell_idx = [2 * k + 1 for k in range(K)]
    ell_block = Sigma[np.ix_(ell_idx, ell_idx)]
    im = axC.imshow(ell_block, cmap="RdBu_r", vmin=-1, vmax=1)
    axC.set_xticks(range(K)); axC.set_xticklabels([PRETTY[t] for t in TASKS], rotation=30, ha="right", fontsize=8)
    axC.set_yticks(range(K)); axC.set_yticklabels([PRETTY[t] for t in TASKS], fontsize=8)
    for i in range(K):
        for j in range(K):
            v = ell_block[i, j]
            axC.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(v) > 0.6 else "black")
    plt.colorbar(im, ax=axC, fraction=0.046)
    axC.set_title(r"C. Prior $\Sigma$: $\ell$-block correlations")

    # Panels D-F (bottom row): 3 samples from the prior — show "skill profiles"
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(DIM))
    for s_idx, sample_seed in enumerate([1, 7, 13]):
        rng2 = np.random.default_rng(sample_seed)
        z = rng2.standard_normal(DIM)
        theta = L @ z
        # add a "tier-level shift" so the bars are visually distinct
        shift = {0: 1.5, 1: 0.5, 2: -0.5}[s_idx]
        ells = theta[1::2] + shift
        ts = theta[0::2]
        ax = fig.add_subplot(gs[1, s_idx])
        ax.bar(x - 0.18, ells, width=0.36, color="#4292c6", label=r"$\ell$")
        ax.bar(x + 0.18, ts,   width=0.36, color="#fb6a4a", label=r"$t$")
        ax.axhline(0, color="black", lw=0.4)
        ax.axhline(0.62, color="grey", lw=0.5, ls="--")
        ax.set_xticks(x); ax.set_xticklabels([PRETTY[t] for t in TASKS], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("parameter value")
        tier_lbl = ["expert-ish", "novice-ish", "below-novice"][s_idx]
        ax.set_title(f"{'DEF'[s_idx]}. Prior draw, shifted ({tier_lbl})", fontsize=10)
        if s_idx == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Figure 1. Multi-task 2PL probit model and the deployment prior",
                  fontsize=12, y=0.98)
    plt.savefig(FIG / "fig1_concept.png", dpi=130, bbox_inches="tight")
    plt.close()


# ──────────────── Figure 2: single-candidate trace ────────────────

def figure_2():
    cands = pd.read_csv(SIM / "candidates.csv")
    npz = np.load(SIM / "trajectories.npz", allow_pickle=True)
    mu_traj = npz["mu_traj"]
    p_pass_traj = npz["p_pass_traj"]

    # Pick a candidate with a varied profile: a "spike-strong, IIIC-mixed" expert.
    # Use the expert with highest spike-ℓ and a clear pass/fail mix.
    exp = cands[cands.tier == "expert"].copy()
    # Score: high spike ℓ, low average ℓ on IIIC. Pick where decision is diverse.
    exp["spike_lead"] = exp["true_ell_spike"] - exp[[f"true_ell_{t}" for t in TASKS[1:]]].mean(axis=1)
    cand_row = exp.sort_values("spike_lead", ascending=False).iloc[0]
    ci = int(cand_row["cand_id"])
    print(f"  fig2 candidate: cand_id={ci}, tier=expert, "
          f"spike_lead={cand_row['spike_lead']:+.2f}")

    mu = mu_traj[ci]  # (T+1, DIM)
    pp = p_pass_traj[ci]  # (T+1, K)
    T = mu.shape[0] - 1  # number of trials

    # Per-task trials timeline (which trials touched task k)
    # We don't store this directly — reconstruct from changes in mu_per_task slot
    # Simpler: extract from candidates n_trials_<task> only the totals; for trace
    # we use the global trial axis on the x-axis.
    fig, axs = _grid(K, sharex=True)            # 4.7: K-agnostic
    for k, task in enumerate(TASKS):
        ax = axs[k]
        ell_mean = mu[:, 2 * k + 1]
        true_ell = cand_row[f"true_ell_{task}"]
        ax.plot(np.arange(T + 1), ell_mean, color=TIER_COLORS["expert"], lw=1.6,
                 label=r"$\hat\ell$")
        ax.axhline(true_ell, color="black", lw=0.7, ls=":",
                    label=fr"true $\ell = {true_ell:+.2f}$")
        ax.axhline(0.62, color="grey", lw=0.5, ls="--", label=r"$\ell^*$")
        # color pass/fail region
        ax.axhspan(0.62, 3.0, color="#2ca02c", alpha=0.08)
        ax.axhspan(-3.0, 0.62, color="#d62728", alpha=0.05)
        # p_pass on secondary axis
        ax2 = ax.twinx()
        ax2.plot(np.arange(T + 1), pp[:, k], color="purple", lw=1.0, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.axhline(0.95, color="purple", lw=0.4, ls=":")
        ax2.axhline(0.05, color="purple", lw=0.4, ls=":")
        if k % 3 == 2:
            ax2.set_ylabel(r"$P(\ell > \ell^*)$", color="purple", fontsize=9)
        ax.set_ylim(-2.5, 3.0)
        decision = cand_row[f"decision_{task}"]
        n_trials = int(cand_row[f"n_trials_{task}"])
        ax.set_title(f"{PRETTY[task]}: {decision.upper()} (n={n_trials} task trials)",
                     fontsize=10)
        if k // 3 == 1:
            ax.set_xlabel("global trial number")
        if k % 3 == 0:
            ax.set_ylabel(r"$\hat\ell$ posterior mean")
        if k == 0:
            ax.legend(frameon=False, fontsize=7, loc="lower right")

    fig.suptitle(f"Figure 2. Single-candidate trace — tier = expert, "
                  f"total trials = {T}",
                  fontsize=12, y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_single_candidate.png", dpi=130, bbox_inches="tight")
    plt.close()


# ──────────────── Figure 3: per-task skill by tier + ℓ* threshold ────────────────

def figure_3():
    cands = pd.read_csv(SIM / "candidates.csv")
    fig, axs = _grid(K, sharey=True)            # 4.7: K-agnostic
    tier_order = ["expert", "experienced", "novice", "crowd"]
    for k, task in enumerate(TASKS):
        ax = axs[k]
        for tier_idx, tier in enumerate(tier_order):
            sub = cands[cands.tier == tier]
            true_ells = sub[f"true_ell_{task}"].values
            hat_ells  = sub[f"hat_ell_{task}"].values
            # Strip plot of hat ell
            xj = tier_idx + np.random.default_rng(tier_idx).normal(0, 0.08, len(hat_ells))
            ax.scatter(xj, hat_ells, s=15, color=TIER_COLORS[tier], alpha=0.55,
                       edgecolor="none")
            # Bar at median
            med = float(np.median(hat_ells))
            ax.plot([tier_idx - 0.25, tier_idx + 0.25], [med, med],
                     color=TIER_COLORS[tier], lw=2.5)
        ax.axhline(0.62, color="grey", lw=1.0, ls="--", label=r"$\ell^* = 0.62$")
        ax.set_xticks(range(len(tier_order)))
        ax.set_xticklabels(tier_order, rotation=15)
        if k % 3 == 0:
            ax.set_ylabel(r"final $\hat\ell$")
        ax.set_title(PRETTY[task])
        ax.set_ylim(-2.5, 3.2)
        ax.grid(alpha=0.3, axis="y")
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(r"Figure 3. Final $\hat\ell$ at test stop, by tier × task. "
                 r"$\ell^* = 0.62$ is the deployment threshold (anchored to spike).",
                 fontsize=11, y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_skill_by_tier.png", dpi=130, bbox_inches="tight")
    plt.close()


# ──────────────── Figure 4: parameter recovery vs N ────────────────

def figure_4():
    cands = pd.read_csv(SIM / "candidates.csv")
    npz = np.load(SIM / "trajectories.npz", allow_pickle=True)
    mu_traj = npz["mu_traj"]

    fig, axs = _grid(K, sharex=True, sharey=False)   # 4.7: K-agnostic
    for k, task in enumerate(TASKS):
        ax = axs[k]
        # For each tier, plot |Δℓ̂| vs N for each candidate
        for tier in ["expert", "experienced", "novice", "crowd"]:
            sub = cands[cands.tier == tier]
            traces = []
            for _, row in sub.iterrows():
                ci = int(row["cand_id"])
                true_ell = row[f"true_ell_{task}"]
                ell_traj = mu_traj[ci][:, 2 * k + 1]
                err = np.abs(ell_traj - true_ell)
                traces.append(err)
            if not traces:
                continue
            # Pad to common length (trajectories vary by total trials)
            max_T = max(len(tr) for tr in traces)
            padded = np.full((len(traces), max_T), np.nan)
            for i, tr in enumerate(traces):
                padded[i, :len(tr)] = tr
            med = np.nanmedian(padded, axis=0)
            q25 = np.nanquantile(padded, 0.25, axis=0)
            q75 = np.nanquantile(padded, 0.75, axis=0)
            n_axis = np.arange(max_T)
            ax.plot(n_axis, med, color=TIER_COLORS[tier], lw=1.5, label=tier)
            ax.fill_between(n_axis, q25, q75, color=TIER_COLORS[tier], alpha=0.13)
        ax.axhline(0.1, color="black", lw=0.4, ls="--")
        ax.axhline(0.3, color="black", lw=0.4, ls=":")
        ax.set_title(PRETTY[task])
        if k // 3 == 1:
            ax.set_xlabel("total trial N")
        if k % 3 == 0:
            ax.set_ylabel(r"$|\hat\ell - \ell_\mathrm{true}|$")
        ax.set_ylim(0, 1.5)
        if k == 0:
            ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
    fig.suptitle(r"Figure 4. Per-task recovery error $|\hat\ell - \ell_\mathrm{true}|$ "
                 r"vs total trial number, by tier",
                  fontsize=11, y=1.00)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_recovery.png", dpi=130, bbox_inches="tight")
    plt.close()


# ──────────────── Figure 5: trials-to-decision + verdicts ────────────────

def figure_5():
    cands = pd.read_csv(SIM / "candidates.csv")
    tier_order = ["expert", "experienced", "novice", "crowd"]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          height_ratios=[1, 1])

    # A: ECDF of total trials by tier
    axA = fig.add_subplot(gs[0, 0])
    for tier in tier_order:
        sub = cands[cands.tier == tier]
        n_total = sub["n_trials_total"].values
        xs = np.sort(n_total)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        axA.plot(xs, ys, color=TIER_COLORS[tier], lw=2, label=tier)
    axA.axvline(60, color="grey", lw=0.5, ls=":", label="N_min=60")
    axA.axvline(150, color="grey", lw=0.5, ls="--", label="N_max=150")
    axA.set_xlabel("total trials per candidate")
    axA.set_ylabel("ECDF")
    axA.set_title("A. Total test length by tier")
    axA.legend(frameon=False, fontsize=8, loc="lower right")
    axA.grid(alpha=0.3)

    # B: stacked verdict bar per tier
    axB = fig.add_subplot(gs[0, 1])
    # Compute fraction of (pass / refer / fail) summed over all 6 tasks per tier
    width = 0.6
    for ti, tier in enumerate(tier_order):
        sub = cands[cands.tier == tier]
        counts = {"pass": 0, "refer": 0, "fail": 0}
        for t in TASKS:
            for d in counts:
                counts[d] += int((sub[f"decision_{t}"] == d).sum())
        total = sum(counts.values())
        if total == 0:
            continue
        bottom = 0
        for d in ["pass", "refer", "fail"]:
            f = counts[d] / total
            axB.bar(ti, f, bottom=bottom, width=width,
                     color=DECISION_COLORS[d], label=d if ti == 0 else None)
            bottom += f
    axB.set_xticks(range(len(tier_order)))
    axB.set_xticklabels(tier_order)
    axB.set_ylabel("fraction of (candidate × task) decisions")
    axB.set_title("B. Verdict distribution per tier (pooled across 6 tasks)")
    axB.legend(frameon=False, fontsize=8, loc="lower right")
    axB.grid(alpha=0.3, axis="y")

    # C: per-task pass rates
    axC = fig.add_subplot(gs[0, 2])
    pass_mat = np.zeros((len(tier_order), K))
    for ti, tier in enumerate(tier_order):
        sub = cands[cands.tier == tier]
        for k, t in enumerate(TASKS):
            pass_mat[ti, k] = (sub[f"decision_{t}"] == "pass").mean()
    im = axC.imshow(pass_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axC.set_xticks(range(K)); axC.set_xticklabels([PRETTY[t] for t in TASKS], rotation=30, ha="right", fontsize=8)
    axC.set_yticks(range(len(tier_order))); axC.set_yticklabels(tier_order)
    for ti in range(len(tier_order)):
        for kk in range(K):
            axC.text(kk, ti, f"{pass_mat[ti, kk]:.0%}",
                      ha="center", va="center", fontsize=8,
                      color="white" if pass_mat[ti, kk] > 0.7 or pass_mat[ti, kk] < 0.3 else "black")
    plt.colorbar(im, ax=axC, fraction=0.046, label="pass rate")
    axC.set_title("C. Per-task pass rate by tier")

    # D: candidate skill-profile heatmap (sorted within tier)
    axD = fig.add_subplot(gs[1, :2])
    sorted_cands = []
    for tier in tier_order:
        sub = cands[cands.tier == tier].copy()
        sub["mean_hat_ell"] = sub[[f"hat_ell_{t}" for t in TASKS]].mean(axis=1)
        sub = sub.sort_values("mean_hat_ell", ascending=False)
        sorted_cands.append(sub)
    all_sorted = pd.concat(sorted_cands, ignore_index=True)
    H = all_sorted[[f"hat_ell_{t}" for t in TASKS]].values
    im = axD.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axD.set_xticks(range(K)); axD.set_xticklabels([PRETTY[t] for t in TASKS], rotation=30, ha="right", fontsize=8)
    # tier separators
    cum = 0
    for tier in tier_order:
        n = (cands.tier == tier).sum()
        cum += n
        axD.axhline(cum - 0.5, color="white", lw=0.8)
        axD.text(K + 0.2, cum - n / 2, tier, va="center", fontsize=9,
                  color=TIER_COLORS[tier])
    axD.set_ylabel("candidates (sorted by mean $\\hat\\ell$ within tier)")
    plt.colorbar(im, ax=axD, fraction=0.046, label=r"$\hat\ell$")
    axD.set_title(r"D. Candidate skill-profile heatmap (one row per candidate)")

    # E: trials-to-decision vs candidate min-ℓ (test length scales with worst task)
    axE = fig.add_subplot(gs[1, 2])
    cands["min_true_ell"] = cands[[f"true_ell_{t}" for t in TASKS]].min(axis=1)
    cands["dist_from_threshold"] = np.abs(cands["min_true_ell"] - 0.62)
    for tier in tier_order:
        sub = cands[cands.tier == tier]
        axE.scatter(sub["dist_from_threshold"], sub["n_trials_total"],
                     s=18, color=TIER_COLORS[tier], alpha=0.55,
                     label=tier, edgecolor="none")
    axE.set_xlabel(r"$|\min_k \ell_{\mathrm{true},k} - \ell^*|$")
    axE.set_ylabel("total trials")
    axE.set_title("E. Test length vs worst-task distance from threshold")
    axE.axhline(150, color="grey", lw=0.4, ls="--")
    axE.legend(frameon=False, fontsize=8)
    axE.grid(alpha=0.3)

    fig.suptitle("Figure 5. Multi-task verdict outcomes and test length",
                  fontsize=12, y=0.995)
    plt.savefig(FIG / "fig5_verdicts.png", dpi=130, bbox_inches="tight")
    plt.close()


def main():
    print(f"Writing figures to {FIG}")
    print("  figure 1 (concept) ...")
    figure_1()
    print("  figure 2 (single candidate) ...")
    figure_2()
    print("  figure 3 (per-task skill) ...")
    figure_3()
    print("  figure 4 (recovery vs N) ...")
    figure_4()
    print("  figure 5 (verdicts) ...")
    figure_5()
    print("Done.")


if __name__ == "__main__":
    main()
