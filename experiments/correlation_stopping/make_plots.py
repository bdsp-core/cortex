"""Plots for the correlation-stopping experiment (run run_experiment.py first).

Figures (out/):
  fig1_aggregate_savings.png   aggregate questions-to-decision: correlated vs
                               independent — paired scatter + ECDF (real+pop).
  fig2_calibration.png         aggregate decision accuracy / error / over-
                               confidence by condition (speed is only fair at
                               matched correctness).
  fig3_concordance.png         per-candidate aggregate saving vs how concordant
                               the candidate is (the shared factor pays off for
                               uniformly-skilled candidates).
  fig4_pertask_vs_aggregate.png   why the aggregate is the right target:
                               per-task all-resolve gets ~no benefit at r=0.37,
                               the aggregate does.
  fig5_concordant_sweep.png    questions-to-aggregate-decision vs a common skill
                               level (mechanism: clearest at the extremes).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"
C_CORR = "#2b8cbe"      # correlated
C_INDEP = "#d95f0e"     # independent
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def paired_totals(df, col):
    p = df.groupby(["cand", "cond"])[col].mean().unstack("cond").dropna()
    return p["correlated"].to_numpy(), p["independent"].to_numpy()


def fig_aggregate_savings():
    real = pd.read_csv(OUT / "results_real.csv")
    pop = pd.read_csv(OUT / "results_pop.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # paired scatter (real)
    c, i = paired_totals(real, "agg_total")
    ax = axes[0]
    lim = max(c.max(), i.max()) * 1.02
    ax.scatter(i, c, s=14, alpha=0.45, color=C_CORR, edgecolor="none")
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("independent: questions to aggregate decision")
    ax.set_ylabel("correlated: questions")
    frac = float((c < i).mean())
    ax.set_title(f"Real raters (N={len(c)})\n"
                 f"correlated faster in {frac*100:.0f}% of raters "
                 f"(below diagonal)")

    # ECDF (real)
    ax = axes[1]
    for arr, lbl, col in ((c, "correlated", C_CORR), (i, "independent", C_INDEP)):
        xs = np.sort(arr); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=col, lw=2,
                label=f"{lbl} (median {np.median(arr):.0f})")
    ax.set_xlabel("questions to aggregate decision")
    ax.set_ylabel("fraction of raters decided")
    ax.set_title("Real raters — ECDF")
    ax.legend()

    # ECDF (pop)
    ax = axes[2]
    cp, ip = paired_totals(pop, "agg_total")
    for arr, lbl, col in ((cp, "correlated", C_CORR), (ip, "independent", C_INDEP)):
        xs = np.sort(arr); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.step(xs, ys, where="post", color=col, lw=2,
                label=f"{lbl} (median {np.median(arr):.0f})")
    ax.set_xlabel("questions to aggregate decision")
    ax.set_ylabel("fraction decided")
    ax.set_title(f"Population draws ~ N(0,Σ)  (N={len(cp)})")
    ax.legend()

    fig.suptitle("Aggregate pass/fail: at the fitted r_ℓ=0.37, modeling the "
                 "correlation does NOT reduce questions-to-decision on the "
                 "realistic population\n(ECDFs overlap; correlated faster in "
                 "only ~1/3 of real raters)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig1_aggregate_savings.png", bbox_inches="tight")
    plt.close(fig)


def fig_calibration():
    real = pd.read_csv(OUT / "results_real.csv")
    pop = pd.read_csv(OUT / "results_pop.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, df, name in ((axes[0], real, "Real raters"),
                         (axes[1], pop, "Population draws")):
        conds = ["independent", "correlated"]
        # among decidable (truth pass/fail): error rate; among gray: over-rate
        metrics = {}
        for cond in conds:
            s = df[df.cond == cond]
            dec = s[s.agg_truth.isin(["pass", "fail"])]
            err = dec["agg_wrong"].sum() / max(len(dec), 1)
            gray = s[s.agg_truth == "gray"]
            over = gray["agg_over"].sum() / max(len(gray), 1)
            refer_rate = (s["agg_decision"] == "refer").mean()
            metrics[cond] = (err, over, refer_rate)
        x = np.arange(3)
        w = 0.36
        labels = ["wrong verdict\n(truth pass/fail)",
                  "over-confident\n(truth borderline)", "referred\n(all)"]
        ax.bar(x - w / 2, metrics["independent"], w, color=C_INDEP, label="independent")
        ax.bar(x + w / 2, metrics["correlated"], w, color=C_CORR, label="correlated")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("rate")
        ax.set_title(name)
        ax.legend()
    fig.suptitle("Aggregate decision quality — aggressively exploiting "
                 "r_ℓ=0.37 (low per-task floor) slightly INCREASES wrong and "
                 "over-confident verdicts (cross-task extrapolation error)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig2_calibration.png", bbox_inches="tight")
    plt.close(fig)


def fig_concordance():
    real = pd.read_csv(OUT / "results_real.csv")
    piv = (real.groupby(["cand", "cond"])["agg_total"].mean()
               .unstack("cond").dropna())
    conc = real.groupby("cand")["concord_sd"].first().reindex(piv.index)
    sav = piv["independent"] - piv["correlated"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    sc = ax.scatter(conc, sav, s=16, alpha=0.5, c=conc, cmap="viridis_r")
    ax.axhline(0, color="k", lw=1, ls="--", alpha=0.7)
    # binned median trend
    bins = np.linspace(conc.min(), conc.max(), 9)
    bi = np.digitize(conc, bins)
    bx, by = [], []
    for b in range(1, len(bins)):
        msk = bi == b
        if msk.sum() >= 5:
            bx.append(conc[msk].mean()); by.append(np.median(sav[msk]))
    ax.plot(bx, by, "-o", color="crimson", lw=2, label="binned median saving")
    ax.set_xlabel("candidate skill spread across tasks  (SD of per-task margin;\n"
                  "low = concordant / uniformly skilled)")
    ax.set_ylabel("questions saved by correlation\n(independent − correlated)")
    ax.set_title("Saving vs concordance — near zero across the board on real "
                 "raters\n(the real population sits near the cut, where neither "
                 "approach resolves fast)")
    ax.legend()
    fig.colorbar(sc, label="skill spread")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_concordance.png", bbox_inches="tight")
    plt.close(fig)


def fig_pertask_vs_aggregate():
    real = pd.read_csv(OUT / "results_real.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, title in ((axes[0], "pt_total", "Per-task (all 6 must resolve)"),
                           (axes[1], "agg_total", "Aggregate (one overall verdict)")):
        c, i = paired_totals(real, col)
        for arr, lbl, cc in ((c, "correlated", C_CORR), (i, "independent", C_INDEP)):
            xs = np.sort(arr); ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.step(xs, ys, where="post", color=cc, lw=2,
                    label=f"{lbl} (median {np.median(arr):.0f})")
        ax.set_xlabel("total questions"); ax.set_ylabel("fraction of raters")
        ax.set_title(title); ax.legend()
    fig.suptitle("At r_ℓ=0.37 neither target benefits on the real population: "
                 "per-task all-resolve and aggregate ECDFs both ~overlap",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig4_pertask_vs_aggregate.png", bbox_inches="tight")
    plt.close(fig)


def fig_concordant_sweep():
    conc = pd.read_csv(OUT / "results_conc.csv")
    g = conc.groupby(["level", "cond"])["agg_total"].mean().unstack("cond")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(g.index, g["independent"], "-o", color=C_INDEP, label="independent")
    ax.plot(g.index, g["correlated"], "-o", color=C_CORR, label="correlated")
    ax.set_xlabel("candidate's common skill level ℓ (all tasks equal)")
    ax.set_ylabel("questions to aggregate decision")
    ax.set_title("Mechanism check (concordant candidates): correlation helps "
                 "ONLY at\nclearly pass/fail levels; in the borderline band both "
                 "hit the cap")
    ax.axvspan(0.35, 0.55, color="0.6", alpha=0.15, label="≈ℓ* (borderline)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig5_concordant_sweep.png", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_aggregate_savings()
    fig_calibration()
    fig_concordance()
    fig_pertask_vs_aggregate()
    fig_concordant_sweep()
    print("wrote figures to", OUT)
    print(json.dumps(json.loads((OUT / "summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
