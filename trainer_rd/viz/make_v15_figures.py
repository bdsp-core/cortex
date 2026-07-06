"""M11 figures — v15 operating characteristics + Tier-3 benchmark.

fig6_v15_resolution : per-arm PASS resolution at +0.3/+0.6/+1.0 (zero-bias)
                      with Wilson CIs, plus the realistic-bias +0.6 pair and
                      the A1 ablation (v15 cuts @600) — the scratch
                      replication of the v15 claim table.
fig7_tier3_benchmark: trials-to-mastery, tier3 vs tier2/tier1/staircase85/
                      measure_opt (random off-scale, annotated), soft 1x and
                      hard 2x cells, paired bootstrap CIs.

Reads the cached study npz files; re-runnable: python3 -m viz.make_v15_figures
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import viz.viz_style as vs
from studies.study_v15_oc import aggregate, wilson_ci

vs.use_style()


def fig6():
    path = os.path.join(vs.FIGDIR, "data_v15_oc.npz")
    results = json.loads(str(np.load(path, allow_pickle=False)["results_json"]))
    table = aggregate(results)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                  gridspec_kw={"width_ratios": [3, 2]})
    offs = [0.3, 0.6, 1.0]
    arms = [("v14", "shipped (v14, corr_l, 600p)", vs.TASK_FILL[1], vs.TASK_EDGE[1]),
            ("v15", "staged v15 (corr_t, 1200p)", vs.TASK_FILL[0], vs.TASK_EDGE[0])]
    width = 0.36
    for i, (arm, label, fill, edge) in enumerate(arms):
        xs = np.arange(len(offs)) + (i - 0.5) * width
        ys, los, his = [], [], []
        for off in offs:
            row = table[(arm, off, "zero")]
            lo, hi = wilson_ci(row["PASS"], row["n"])
            ys.append(100 * row["PASS"])
            los.append(100 * (row["PASS"] - lo))
            his.append(100 * (hi - row["PASS"]))
        ax.bar(xs, ys, width, label=label, color=fill, edgecolor=edge,
               linewidth=1.5, yerr=[los, his], capsize=5,
               error_kw={"ecolor": vs.ACCENT_DARK, "lw": 1.5})
    # A1 ablation marker at +0.6
    abl = table.get(("v15cuts600", 0.6, "zero"))
    if abl:
        ax.plot([1.0 + 0.5 * width], [100 * abl["PASS"]], marker="D", ms=12,
                color=vs.GOOD, mec=vs.ACCENT_DARK, zorder=5,
                label="A1: v15 cuts only (600p)")
    ax.set_xticks(np.arange(len(offs)))
    ax.set_xticklabels([f"+{o:g}" for o in offs])
    ax.set_xlabel("true skill above cut,  ℓ = ℓ* + offset")
    ax.set_ylabel("task verdicts resolved PASS  (%)")
    ax.set_ylim(0, 105)
    vs.style_ax(ax)
    vs.top_legend(ax, 2, anchor=1.16)

    # right panel: realistic-bias cost at +0.6
    pairs = [("v14", vs.TASK_FILL[1], vs.TASK_EDGE[1]),
             ("v15", vs.TASK_FILL[0], vs.TASK_EDGE[0])]
    for i, (arm, fill, edge) in enumerate(pairs):
        for j, bias in enumerate(("zero", "realistic")):
            row = table[(arm, 0.6, bias)]
            lo, hi = wilson_ci(row["PASS"], row["n"])
            x = i + (j - 0.5) * width
            hatch = "" if bias == "zero" else "//"
            ax2.bar([x], [100 * row["PASS"]], width, color=fill, edgecolor=edge,
                    linewidth=1.5, hatch=hatch,
                    yerr=[[100 * (row["PASS"] - lo)], [100 * (hi - row["PASS"])]],
                    capsize=5, error_kw={"ecolor": vs.ACCENT_DARK, "lw": 1.5})
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["v14", "v15"])
    ax2.set_ylabel("PASS at +0.6  (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("examinee bias cost (hatched =\nθ ~ N(μ_t, Σ_t) realistic arm)",
                  fontsize=18)
    vs.style_ax(ax2)
    fig.suptitle("v15 staged instrument — K=7 resolution on the real bank "
                 "(420-question regime)", y=1.04)
    return vs.save_fig(fig, "fig6_v15_resolution")


def fig7():
    path = os.path.join(vs.FIGDIR, "data_benchmark_tier3.npz")
    data = np.load(path)
    cells = [("soft", 1.0, "well-specified (soft, 1×)"),
             ("hard", 2.0, "stressed (hard rule, 2× rate misspec)")]
    pols = ["tier3", "tier2", "tier1", "staircase85", "measure_opt"]
    labels = ["tier-3\nMC rollout", "tier-2\nmode", "tier-1\ngreedy",
              "staircase\n85%", "measure-\nopt"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, (rule, rm, title) in zip(axes, cells):
        mus, errs = [], []
        for p in pols:
            v = data[f"{rule}_{rm}_{p}"].astype(float)
            bs = np.random.default_rng(7).choice(
                v, size=(2000, v.size), replace=True).mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            mus.append(v.mean())
            errs.append([v.mean() - lo, hi - v.mean()])
        errs = np.array(errs).T
        colors = [vs.GOOD] + [vs.TASK_FILL[i % 3] for i in range(len(pols) - 1)]
        edges = [vs.ACCENT_DARK] + [vs.TASK_EDGE[i % 3] for i in range(len(pols) - 1)]
        ax.bar(np.arange(len(pols)), mus, 0.62, color=colors, edgecolor=edges,
               linewidth=1.5, yerr=errs, capsize=5,
               error_kw={"ecolor": vs.ACCENT_DARK, "lw": 1.5})
        rnd = data[f"{rule}_{rm}_random"].astype(float)
        ax.annotate(f"random: {rnd.mean():.0f}", xy=(0.97, 0.95),
                    xycoords="axes fraction", ha="right", fontsize=16,
                    color=vs.NEUTRAL)
        ax.set_xticks(np.arange(len(pols)))
        ax.set_xticklabels(labels, fontsize=15)
        ax.set_title(title, fontsize=18)
        vs.style_ax(ax)
    axes[0].set_ylabel("trials to true mastery")
    fig.suptitle("Tier-3 short-horizon MC rollout vs comparators "
                 "(domain3, real bank, 30 seeds)", y=1.02)
    return vs.save_fig(fig, "fig7_tier3_benchmark")


if __name__ == "__main__":
    for fn in (fig6, fig7):
        try:
            print(fn.__name__, "→", fn())
        except FileNotFoundError as e:
            print(fn.__name__, "skipped (data not ready):", e)
