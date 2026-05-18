"""F3.6 — Nature-quality 4-panel inference-validation composite.

Reads the Phase-2 result JSON/CSV in results/phase2_validation/ and emits
one publication figure that visually substantiates "the SMC + MH engine
is calibrated" for the Paper-1 methods section:

  (a) SBC rank uniformity      — 12 per-parameter rank histograms vs the
                                 Bonferroni-adjusted 95% binomial band.
  (b) AUROC CI coverage        — empirical vs nominal, production config,
                                 with the ±0.03 acceptance band.
  (c) SPARCNET test-retest     — split-half AUROC h1 vs h2, per domain,
                                 with per-domain ICC(3,1).
  (d) Gold-chain agreement     — per (examinee,param) standardized mean
                                 diff vs |log sd-ratio| vs the acceptance
                                 box (the SMC-vs-exact-MH check).

Okabe-Ito palette, 300 DPI, PDF + PNG, fonttype 42.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_THIS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(_THIS), "results", "phase2_validation")

OK = {
    "blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
    "purple": "#CC79A7", "orange": "#E69F00", "sky": "#56B4E9",
    "gray": "#7f7f7f", "yellow": "#F0E442",
}
DOMAIN_COLORS = {
    "sz": OK["blue"], "lpd": OK["vermillion"], "gpd": OK["green"],
    "lrda": OK["purple"], "grda": OK["orange"], "iic": OK["sky"],
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})


def panel_a_sbc(ax):
    sbc = json.load(open(os.path.join(RES, "sbc_results.json")))
    lo, hi = sbc["config"]["bonferroni_band"]
    nb = sbc["config"]["n_bins"]
    n_draws = sbc["config"]["n_draws"]
    centers = np.arange(nb) + 0.5
    pr = sbc["param_results"]
    for name, r in pr.items():
        ax.plot(centers, r["bin_counts"], color=OK["gray"], alpha=0.35,
                linewidth=0.8)
    # Bonferroni band + uniform expectation
    ax.axhspan(lo, hi, color=OK["blue"], alpha=0.15, linewidth=0,
               label=f"95% Bonferroni band [{lo},{hi}]")
    ax.axhline(n_draws / nb, color=OK["blue"], linestyle="--",
               linewidth=0.9, label=f"uniform = {n_draws/nb:.0f}")
    n_pass = sbc["n_pass"]
    n_tot = len(pr)
    ax.set_xlabel("rank bin")
    ax.set_ylabel("count")
    ax.set_xlim(0, nb)
    ax.set_title(f"a   SBC rank uniformity — {n_pass}/{n_tot} params pass",
                 loc="left", fontweight="bold", pad=6)
    ax.legend(loc="lower center", frameon=False, ncol=1)


def panel_b_coverage(ax):
    sw = json.load(open(os.path.join(RES, "coverage_sweep.json")))
    levels = ["50%", "80%", "90%", "95%"]
    nominal = np.array([0.50, 0.80, 0.90, 0.95])
    # identity + acceptance band
    xs = np.linspace(0.45, 1.0, 50)
    ax.plot(xs, xs, color="black", linewidth=0.8, zorder=1)
    ax.fill_between(xs, xs - 0.03, xs + 0.03, color=OK["gray"],
                    alpha=0.15, linewidth=0, label="±0.03 acceptance")
    # context cells (faint) + production cell (bold)
    faint = {"N500_ess0.5": OK["gray"], "N500_ess0.9": OK["gray"],
             "N1000_ess0.5": OK["gray"]}
    for cell, col in faint.items():
        emp = [sw["summary"][cell]["per_level"][l]["empirical"]
               for l in levels]
        ax.plot(nominal, emp, "o-", color=col, alpha=0.35, markersize=3,
                linewidth=0.7)
    prod = "N1000_ess0.9"
    emp = [sw["summary"][prod]["per_level"][l]["empirical"] for l in levels]
    ax.plot(nominal, emp, "o-", color=OK["green"], markersize=6,
            markeredgecolor="white", markeredgewidth=0.6, linewidth=1.6,
            label="production (N=1000, ess=0.9)", zorder=5)
    for x, y in zip(nominal, emp):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(6, -10), fontsize=7, color=OK["green"])
    ax.set_xlabel("nominal CI level")
    ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_title("b   AUROC credible-interval coverage",
                 loc="left", fontweight="bold", pad=6)
    ax.legend(loc="upper left", frameon=False)


def panel_c_retest(ax):
    rows = list(csv.DictReader(open(os.path.join(RES,
                                                  "sparcnet_test_retest.csv"))))
    tr = json.load(open(os.path.join(RES, "sparcnet_test_retest.json")))
    by_dom = {}
    for r in rows:
        if r.get("skipped") not in (None, "", "None"):
            continue
        by_dom.setdefault(r["domain"], []).append(
            (float(r["auroc_h1"]), float(r["auroc_h2"])))
    lo, hi = 0.55, 0.95
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, zorder=1)
    order = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
    for d in order:
        pts = np.array(by_dom.get(d, []))
        if pts.size == 0:
            continue
        icc = tr["per_domain"][d]["icc_3_1_auroc"]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=DOMAIN_COLORS[d],
                   alpha=0.7, edgecolor="white", linewidth=0.3,
                   label=f"{d}  ICC={icc:.2f}")
    ax.set_xlabel("AUROC — split half 1")
    ax.set_ylabel("AUROC — split half 2")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("c   SPARCNET split-half test-retest (n=219)",
                 loc="left", fontweight="bold", pad=6)
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=0.8,
              handletextpad=0.3)


def panel_d_goldchain(ax):
    gc = json.load(open(os.path.join(RES, "gold_chain_reference.json")))
    smd_thr = 0.10
    sdr_thr = np.log(1.15)   # |log sd-ratio| acceptance bound
    ex_colors = [OK["blue"], OK["vermillion"], OK["green"]]
    for i, (ekey, s) in enumerate(sorted(gc["summary"].items())):
        smd = [c["std_mean_diff"] for c in s["cells"].values()]
        sdr = [abs(np.log(c["sd_ratio"])) for c in s["cells"].values()]
        ax.scatter(smd, sdr, s=22, color=ex_colors[i % 3], alpha=0.75,
                   edgecolor="white", linewidth=0.4,
                   label=f"{ekey} ({s['cells_pass']}/{s['cells_total']})")
    # acceptance box
    ax.axvline(smd_thr, color=OK["gray"], linestyle="--", linewidth=0.8)
    ax.axhline(sdr_thr, color=OK["gray"], linestyle="--", linewidth=0.8)
    ax.add_patch(plt.Rectangle((0, 0), smd_thr, sdr_thr, facecolor=OK["green"],
                               alpha=0.10, linewidth=0))
    ax.text(smd_thr * 0.5, sdr_thr * 0.5, "PASS\nregion", ha="center",
            va="center", fontsize=7.5, color=OK["green"])
    pc, tc = gc["pass_cells"], gc["total_cells"]
    ax.set_xlabel(r"$|\Delta\mu|/\sigma_{\rm ref}$  (standardized mean diff)")
    ax.set_ylabel(r"$|\log(\sigma_{\rm SMC}/\sigma_{\rm ref})|$")
    ax.set_title(f"d   Gold-chain agreement — {pc}/{tc} cells pass",
                 loc="left", fontweight="bold", pad=6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False)


def main():
    fig = plt.figure(figsize=(11.5, 9.0))
    gs = gridspec.GridSpec(2, 2, wspace=0.26, hspace=0.30,
                           left=0.07, right=0.975, top=0.93, bottom=0.07,
                           figure=fig)
    panel_a_sbc(fig.add_subplot(gs[0, 0]))
    panel_b_coverage(fig.add_subplot(gs[0, 1]))
    panel_c_retest(fig.add_subplot(gs[1, 0]))
    panel_d_goldchain(fig.add_subplot(gs[1, 1]))
    fig.suptitle("Inference-engine validation — Multi-AUROC Precision "
                 "Protocol (Mode-A, post-F0.1)", fontsize=12,
                 fontweight="bold", y=0.975)
    fig.text(0.07, 0.012,
             "SBC: Talts et al. 2018, non-adaptive design.  Coverage: "
             "F2.2b 2×2 sweep, 120 synthetic raters/cell.  Test-retest: "
             "split-half (seed 42), probit-lapse refit.  Gold-chain: "
             "SMC-cov-preconditioned 5×10⁵-step MH reference, moment "
             "criterion (Chopin & Papaspiliopoulos 2020).",
             fontsize=6.5, color="#555555", ha="left")
    for ext in ("pdf", "png"):
        out = os.path.join(RES, f"fig_validation_composite.{ext}")
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
