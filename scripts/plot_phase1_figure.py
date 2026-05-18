"""Generate Phase-1 publication-quality composite figure.

Reads results/phase1_figures/expA_real_k6.json, expB_kscaling.json,
example_trajectory.json (produced by run_phase1_experiments.py) and emits a
3-panel Nature Medicine-style composite plus a supplementary heatmap.

Conventions:
  - Okabe-Ito colorblind-safe palette (matches single-domain spike paper).
  - PDF + PNG at 300 DPI; PDF has fonttype=42 for journal TrueType embedding.
  - No top/right spines; no legend frame.
  - Direct labels where possible; minimum tick clutter.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
RESULTS_DIR = os.path.join(ENGINE_REPO, "results", "phase1_figures")


# ── style ─────────────────────────────────────────────────────────────
# F-rand: three-method ablation ladder.  random (null: no adaptive
# selection, no pooling) → brute (+adaptive item selection) → hier
# (+hierarchical pooling).  Each rung adds exactly one capability.
OKABE_ITO = {
    "random": "#7f7f7f",   # gray  — null baseline (no adaptivity)
    "brute":  "#D55E00",   # vermillion — independent + adaptive selection
    "hier":   "#0072B2",   # blue  — + hierarchical pooling (headline)
    "delta":  "#bdbdbd",   # light gray — δ reference lines (dotted)
    "accent": "#009E73",   # green — speedup annotations
}
METHOD_ORDER = ["random", "brute", "hier"]
METHOD_LABEL = {
    "random": "Random (null baseline)",
    "brute":  "Independent (adaptive)",
    "hier":   "Hierarchical (Corr_l)",
}
METHOD_LABEL_SHORT = {"random": "Random", "brute": "Independent",
                      "hier": "Hierarchical"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 110,
})


# ── data loading ──────────────────────────────────────────────────────
def load_results():
    """Prefer v2 results files when available; fall back to v1.

    The v2 run (script: run_phase1_experiments_v2.py) uses all 27 SPARCNET
    raters with complete K=6 SDT fits, vs the v1 spike-test 4-rater pilot.
    """
    def _pick(stem):
        v2 = os.path.join(RESULTS_DIR, f"{stem}_v2.json")
        v1 = os.path.join(RESULTS_DIR, f"{stem}.json")
        return v2 if os.path.exists(v2) else v1

    with open(_pick("expA_real_k6")) as f:
        expA = json.load(f)
    with open(_pick("expB_kscaling")) as f:
        expB = json.load(f)
    with open(_pick("example_trajectory")) as f:
        traj = json.load(f)
    print(f"  loaded expA from {_pick('expA_real_k6')}")
    print(f"  loaded expB from {_pick('expB_kscaling')}")
    print(f"  loaded traj from {_pick('example_trajectory')}")
    return expA, expB, traj


def _stops_by_method(rows: List[Dict[str, Any]], delta_key: str):
    """Return dict method → list of n_q values (drop None = did not reach δ)."""
    out: Dict[str, List[int]] = {m: [] for m in METHOD_ORDER}
    for r in rows:
        v = r.get(delta_key)
        if v is None:
            continue
        out.setdefault(r["method"], []).append(int(v))
    return out


# ── panel A: δ-sweep speedup at K=6 SPARCNET ──────────────────────────
def panel_a(ax, expA):
    """Median ± IQR of n_q to reach each δ — random/brute/hier, K=6 SPARCNET.

    Headline annotation is the TOTAL speedup hier ÷ random (the full method
    vs the true null baseline); the random→brute and brute→hier rungs are
    visible directly from the grouped bars.
    """
    rows = expA["rows"]
    deltas = expA["config"]["deltas"]
    delta_keys = {0.025: "stop_d0_025", 0.05: "stop_d0_05", 0.10: "stop_d0_10"}

    x = np.arange(len(deltas))
    width = 0.26
    offsets = {"random": -width, "brute": 0.0, "hier": width}
    summary = {}
    counts = {m: [] for m in METHOD_ORDER}

    for method in METHOD_ORDER:
        meds, q25, q75 = [], [], []
        for d in deltas:
            vals = _stops_by_method(rows, delta_keys[d]).get(method, [])
            counts[method].append(len(vals))
            if vals:
                arr = np.asarray(vals)
                meds.append(np.median(arr))
                q25.append(np.percentile(arr, 25))
                q75.append(np.percentile(arr, 75))
            else:
                meds.append(np.nan); q25.append(np.nan); q75.append(np.nan)
        summary[method] = (np.array(meds), np.array(q25), np.array(q75))
        err = np.vstack([np.array(meds) - np.array(q25),
                         np.array(q75) - np.array(meds)])
        ax.bar(x + offsets[method], meds, width=width,
               color=OKABE_ITO[method], edgecolor="white", linewidth=0.6,
               label=METHOD_LABEL[method])
        ax.errorbar(x + offsets[method], meds, yerr=err, fmt="none",
                    ecolor="black", elinewidth=0.7, capsize=2.0, capthick=0.7)

    # Headline = total speedup hier ÷ random where all 3 have ≥2 sessions.
    y_max_overall = 0.0
    for i, d in enumerate(deltas):
        meds_i = {m: summary[m][0][i] for m in METHOD_ORDER}
        q75_i = {m: summary[m][2][i] for m in METHOD_ORDER}
        finite = [m for m in METHOD_ORDER if not np.isnan(meds_i[m])]
        if finite:
            y_max_overall = max(y_max_overall,
                                max(q75_i[m] for m in finite))
        if (not np.isnan(meds_i["hier"]) and not np.isnan(meds_i["random"])
                and counts["hier"][i] >= 2 and counts["random"][i] >= 2):
            sp = meds_i["random"] / meds_i["hier"]
            ymax = max(q75_i[m] for m in finite)
            ax.text(x[i], ymax * 1.05,
                    f"{sp:.1f}× hier vs random"
                    if sp >= 1.0 else f"{1.0/sp:.1f}× random",
                    ha="center", va="bottom", color=OKABE_ITO["accent"],
                    fontweight="bold", fontsize=8)

    for i, d in enumerate(deltas):
        nr, nb, nh = (counts["random"][i], counts["brute"][i],
                      counts["hier"][i])
        ax.text(x[i], -0.06, f"n = {nr}|{nb}|{nh}",
                ha="center", va="top", transform=ax.get_xaxis_transform(),
                fontsize=6.5, color="#555555")

    ax.set_xticks(x)
    ax.set_xticklabels([f"δ = {d:.3f}" for d in deltas])
    ax.set_ylabel("Questions to reach precision\n(median ± IQR)")
    ax.set_title("a   K = 6 SPARCNET (real raters)",
                 loc="left", fontweight="bold", pad=8)
    ax.set_ylim(0, y_max_overall * 1.34 if y_max_overall > 0 else None)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5,
              handlelength=1.4, handletextpad=0.5,
              labelspacing=0.3, borderpad=0.4)
    ax.tick_params(axis="x", pad=14)


# ── panel B: K-scaling at δ=0.05 ──────────────────────────────────────
def panel_b(ax, expB):
    """Median n_q to reach δ=0.05 vs K — random/brute/hier."""
    rows = expB["rows"]
    Ks = expB["config"]["Ks"]
    for method in METHOD_ORDER:
        meds, q25, q75 = [], [], []
        for K in Ks:
            vals = [r["stop_d0_05"] for r in rows
                    if r["K"] == K and r["method"] == method
                    and r["stop_d0_05"] is not None]
            if vals:
                meds.append(np.median(vals))
                q25.append(np.percentile(vals, 25))
                q75.append(np.percentile(vals, 75))
            else:
                meds.append(np.nan); q25.append(np.nan); q75.append(np.nan)
        meds = np.asarray(meds, dtype=float)
        q25 = np.asarray(q25, dtype=float)
        q75 = np.asarray(q75, dtype=float)
        ls = "--" if method == "random" else "-"
        ax.plot(Ks, meds, ls, marker="o", color=OKABE_ITO[method],
                markersize=5, markeredgecolor="white", markeredgewidth=0.6,
                label=METHOD_LABEL_SHORT[method])
        ax.fill_between(Ks, q25, q75, color=OKABE_ITO[method], alpha=0.15,
                        linewidth=0)

    ax.set_xticks(Ks)
    ax.set_xlabel("Number of domains K")
    ax.set_ylabel("Questions to reach δ = 0.05\n(median, IQR shaded)")
    ax.set_title("b   Scaling with K (synthetic)",
                 loc="left", fontweight="bold", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5,
              handlelength=1.4, handletextpad=0.5,
              labelspacing=0.3, borderpad=0.4)


# ── panel C: example HW(AUROC) trajectory ─────────────────────────────
def panel_c(ax, traj_json):
    """One example rater: max-k HW(AUROC) trajectory for hier vs brute."""
    rater_id = traj_json["rater_id"]
    trajs = traj_json["trajectories"]
    for method in METHOD_ORDER:
        if method not in trajs:
            continue
        hw = np.asarray(trajs[method]["hw_max_per_q"])
        qs = np.arange(len(hw))
        ls = "--" if method == "random" else "-"
        ax.plot(qs, hw, color=OKABE_ITO[method], linewidth=1.4,
                linestyle=ls, label=METHOD_LABEL_SHORT[method])
        for dkey, dval in [("stop_d0_10", 0.10), ("stop_d0_05", 0.05),
                            ("stop_d0_025", 0.025)]:
            n_q = trajs[method].get(dkey)
            if n_q is not None and 0 < n_q < len(hw):
                ax.plot([n_q], [hw[n_q]], "o", color=OKABE_ITO[method],
                        markersize=4, markeredgecolor="white",
                        markeredgewidth=0.6, zorder=5)

    # Reference δ lines (dotted light gray so they don't read as the
    # random-method line, which is solid-ish gray)
    for d in [0.025, 0.05, 0.10]:
        ax.axhline(d, color=OKABE_ITO["delta"], linestyle=":",
                   linewidth=0.7, alpha=0.9)
        ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 400,
                d, f" δ = {d:.3f}", color="#888888",
                fontsize=7, va="center", ha="left")

    ax.set_xlabel("Question index")
    ax.set_ylabel("max$_k$ HW (AUROC$_k$)")
    ax.set_title(f"c   Example trajectory ({rater_id})",
                 loc="left", fontweight="bold", pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8,
              handlelength=1.4, handletextpad=0.5,
              labelspacing=0.3, borderpad=0.4)
    ax.set_ylim(bottom=0)


# ── supplementary heatmap: per-rater speedup at each δ ────────────────
def supplementary_heatmap(expA):
    rows = expA["rows"]
    rater_ids = sorted(set(r["rater_id"] for r in rows))
    deltas = expA["config"]["deltas"]
    delta_keys = {0.025: "stop_d0_025", 0.05: "stop_d0_05", 0.10: "stop_d0_10"}

    # Median over seeds per (rater, method, δ); then ratio brute/hier.
    ratio = np.full((len(rater_ids), len(deltas)), np.nan)
    for ri, rid in enumerate(rater_ids):
        for di, d in enumerate(deltas):
            hv = [r[delta_keys[d]] for r in rows
                  if r["rater_id"] == rid and r["method"] == "hier"
                  and r[delta_keys[d]] is not None]
            bv = [r[delta_keys[d]] for r in rows
                  if r["rater_id"] == rid and r["method"] == "brute"
                  and r[delta_keys[d]] is not None]
            if hv and bv:
                ratio[ri, di] = float(np.median(bv) / np.median(hv))

    fig, ax = plt.subplots(figsize=(4.6, 0.55 * len(rater_ids) + 1.5))
    # Centred diverging colormap at speedup=1.0
    vmax = float(np.nanmax(ratio)) if np.isfinite(ratio).any() else 2.0
    vmin = float(np.nanmin(ratio)) if np.isfinite(ratio).any() else 0.5
    vextent = max(abs(np.log2(vmax)), abs(np.log2(vmin if vmin > 0 else 0.5)))
    im = ax.imshow(np.log2(ratio), cmap="RdBu_r",
                   vmin=-vextent, vmax=vextent, aspect="auto")
    ax.set_xticks(np.arange(len(deltas)))
    ax.set_xticklabels([f"δ = {d:.3f}" for d in deltas])
    ax.set_yticks(np.arange(len(rater_ids)))
    ax.set_yticklabels(rater_ids, fontsize=8)
    # Cell text: speedup ratio
    for ri in range(len(rater_ids)):
        for di in range(len(deltas)):
            v = ratio[ri, di]
            if np.isfinite(v):
                ax.text(di, ri, f"{v:.2f}×", ha="center", va="center",
                        fontsize=7,
                        color=("white" if abs(np.log2(v)) > vextent * 0.5
                               else "black"))
            else:
                ax.text(di, ri, "n/a", ha="center", va="center",
                        fontsize=7, color="#999999")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("log₂ speedup (brute / hier)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax.set_title("Per-rater pooling speedup (brute n_q / hier n_q)\n"
                 "isolates hierarchical pooling; random null baseline is "
                 "in the main figure (panels a–c)",
                 loc="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    out_pdf = os.path.join(RESULTS_DIR, "fig_phase1_supp_speedup_heatmap.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_phase1_supp_speedup_heatmap.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  → wrote {out_pdf}\n  → wrote {out_png}")


# ── main composite figure ─────────────────────────────────────────────
def main_figure(expA, expB, traj):
    fig = plt.figure(figsize=(11.5, 3.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.0, 1.4],
                            wspace=0.40, top=0.88, bottom=0.16,
                            left=0.06, right=0.98, figure=fig)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a(ax_a, expA)
    panel_b(ax_b, expB)
    panel_c(ax_c, traj)

    # Footer / source line
    n_A = len(expA["rows"])
    K_A = expA["config"]["K"]
    n_B = len(expB["rows"])
    fig.text(0.005, -0.03,
             f"Multi-AUROC Precision Protocol • Mode-A engine v11 • "
             f"Ablation ladder: random (null: no adaptive selection) → "
             f"independent (+adaptive item selection) → hierarchical "
             f"(+Corr_l pooling).  Panel a: {n_A} sessions, K={K_A}, real "
             f"SPARCNET raters.  Panel b: {n_B} sessions, synthetic.  "
             f"Panel c: K=6, single seed.",
             fontsize=6, color="#555555", ha="left", va="top")

    out_pdf = os.path.join(RESULTS_DIR, "fig_phase1_main.pdf")
    out_png = os.path.join(RESULTS_DIR, "fig_phase1_main.png")
    fig.savefig(out_pdf)
    fig.savefig(out_png)
    plt.close(fig)
    print(f"  → wrote {out_pdf}\n  → wrote {out_png}")


if __name__ == "__main__":
    expA, expB, traj = load_results()
    print("Generating composite figure ...")
    main_figure(expA, expB, traj)
    print("Generating supplementary heatmap ...")
    supplementary_heatmap(expA)
