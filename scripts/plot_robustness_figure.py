"""F3.7 — Nature-quality 2-panel robustness figure.

  (a) Σ_l prior-covariance sensitivity — hier-vs-brute speedup by prior
      structure.  Legitimate unit-diagonal cells (corr_l, cs, diagonal)
      vs the fitted_raw / ledoit_wolf false-precision negative controls.
  (b) Lapse-rate misspecification (Wichmann-Hill) — |AUROC bias| and
      excess-over-well-specified vs the true lapse, with the 0.02
      robustness band.

Okabe-Ito, 300 DPI, PDF + PNG, fonttype 42.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(_THIS), "results", "phase2_validation")

OK = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "purple": "#CC79A7", "orange": "#E69F00", "gray": "#7f7f7f"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})


def panel_a_sigma(ax):
    s = json.load(open(os.path.join(RES, "sigma_sensitivity.json")))
    # Order: legitimate unit-diagonal cells first, then negative controls.
    legit = ["diagonal", "cs_r0.2", "cs_r0.4", "corr_l"]
    controls = ["ledoit_wolf", "fitted_raw"]
    cells = legit + controls
    labels = {"diagonal": "diagonal\n(no corr.)", "cs_r0.2": "CS r=0.2",
              "cs_r0.4": "CS r=0.4", "corr_l": "Corr_l\n(production)",
              "ledoit_wolf": "Ledoit–Wolf", "fitted_raw": "raw Σ_l"}
    x = np.arange(len(cells))
    w = 0.36
    d05 = [s["cells"][c]["speedup_d0_05"] for c in cells]
    d10 = [s["cells"][c]["speedup_d0_10"] for c in cells]
    # Split-axis trick: controls have absurd speedups (24×, 213×) that are
    # *false precision*, not real.  Cap the bar height and annotate the true
    # value so the legitimate cells stay readable.
    CAP = 3.2
    def capped(vals):
        return [min(v, CAP) for v in vals]
    b1 = ax.bar(x - w / 2, capped(d05), w, color=OK["blue"],
                edgecolor="white", linewidth=0.6, label="speedup @ δ=0.05")
    b2 = ax.bar(x + w / 2, capped(d10), w, color=OK["orange"],
                edgecolor="white", linewidth=0.6, label="speedup @ δ=0.10")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="-")
    # annotate capped (control) bars with their true value
    for xi, c in enumerate(cells):
        for off, v in ((-w / 2, d05[xi]), (w / 2, d10[xi])):
            if v > CAP:
                ax.annotate(f"{v:.0f}×", (x[xi] + off, CAP),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=7,
                            color=OK["vermillion"], fontweight="bold")
            else:
                ax.annotate(f"{v:.2f}×", (x[xi] + off, v),
                            textcoords="offset points", xytext=(0, 2),
                            ha="center", fontsize=6.8, color="#444444")
    # shade the negative-control region
    ax.axvspan(len(legit) - 0.5, len(cells) - 0.5, color=OK["vermillion"],
               alpha=0.07, linewidth=0)
    ax.text((len(legit) + len(cells) - 1) / 2, CAP * 0.92,
            "false-precision\nnegative controls", ha="center", va="top",
            fontsize=7, color=OK["vermillion"])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[c] for c in cells], fontsize=7.5)
    ax.set_ylabel("hier ÷ brute speedup (median n_q)")
    ax.set_ylim(0, CAP + 0.25)
    bl = s["brute_baseline"]
    ax.set_title(f"a   Σ_l sensitivity — speedup attributable to "
                 f"correlation\n(brute baseline median n_q: δ=0.05 "
                 f"{bl['median_d0_05']:.0f}, δ=0.10 {bl['median_d0_10']:.0f}; "
                 f"n=12)", loc="left", fontweight="bold", pad=6, fontsize=9)
    ax.legend(loc="upper left", frameon=False)


def panel_b_lapse(ax):
    l = json.load(open(os.path.join(RES, "lapse_sensitivity.json")))
    lam = sorted(float(k) for k in l["summary"].keys())
    absb = [l["summary"][f"{x:g}" if f"{x:g}" in l["summary"] else str(x)]
            ["abs_bias"] for x in lam]
    # robust key lookup (json keys are stringified floats)
    def get(x, field):
        for k, v in l["summary"].items():
            if abs(float(k) - x) < 1e-9:
                return v[field]
        raise KeyError(x)
    absb = [get(x, "abs_bias") for x in lam]
    exc = [get(x, "excess_over_wellspecified") for x in lam]
    ax.plot(lam, absb, "o-", color=OK["blue"], markersize=6,
            markeredgecolor="white", markeredgewidth=0.6,
            label="|AUROC bias| (abs)")
    ax.plot(lam, exc, "s--", color=OK["vermillion"], markersize=5,
            label="excess over well-specified")
    # 0.02 robustness band + well-specified marker
    ax.axhspan(0, 0.02, color=OK["green"], alpha=0.12, linewidth=0,
               label="robustness band (≤0.02)")
    ax.axvline(0.025, color=OK["gray"], linestyle=":", linewidth=1.0)
    ax.annotate("λ assumed\n= 0.025", (0.025, max(absb) * 0.92),
                fontsize=7, color=OK["gray"], ha="left",
                textcoords="offset points", xytext=(4, 0))
    for x, y in zip(lam, absb):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7,
                    color=OK["blue"])
    for x, y in zip(lam, exc):
        if x != 0.025:
            ax.annotate(f"{y:+.3f}", (x, y), textcoords="offset points",
                        xytext=(0, -12), ha="center", fontsize=6.8,
                        color=OK["vermillion"])
    verdict = l["verdict"]
    ax.set_xlabel("true lapse rate λ_true")
    ax.set_ylabel("AUROC bias")
    ax.set_xticks(lam)
    ax.set_xlim(0.005, 0.11)
    ax.set_ylim(bottom=-0.005)
    ax.set_title(f"b   Lapse misspecification (Wichmann-Hill) — "
                 f"{verdict}\nrobust λ∈[0.01,0.05]; bounded at λ=0.10 "
                 f"(excess {l['worst_excess']:.3f})",
                 loc="left", fontweight="bold", pad=6, fontsize=9)
    ax.legend(loc="upper left", frameon=False)


def main():
    fig = plt.figure(figsize=(12.0, 4.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1.0], wspace=0.24,
                           left=0.07, right=0.975, top=0.84, bottom=0.16,
                           figure=fig)
    panel_a_sigma(fig.add_subplot(gs[0, 0]))
    panel_b_lapse(fig.add_subplot(gs[0, 1]))
    fig.suptitle("Robustness — prior-covariance structure & lapse-rate "
                 "misspecification (Mode-A, post-F0.1)",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.text(0.07, 0.01,
             "Σ_l: 12 SPARCNET raters, K=6, ess=0.5 (uniform across cells "
             "→ ranking invariant).  Lapse: 100 synthetic raters × 6 "
             "domains, n_obs=500 fixed-design MLE; engine assumes λ=0.025.",
             fontsize=6.5, color="#555555", ha="left")
    for ext in ("pdf", "png"):
        out = os.path.join(RES, f"fig_robustness.{ext}")
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
