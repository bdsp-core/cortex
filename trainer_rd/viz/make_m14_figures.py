"""M14 "Nature-gap" result figures, same publication aesthetic as
`make_extset_figures` (viz_style: purple/grey/teal, gridded, PNG+PDF+SVG).

fig15 A1 — single-skill softmax (M5/M5b) vs composed marginals (held-out ΔELPD)
fig16 A2 — Bayesian posterior of the M2 globals + proper SBC rank-uniformity
fig17 A3 — specification curve (42 branches, all positive)
fig18 A4 — disattenuated validity (split-half reliabilities; P2b → 1.00)
fig19 C9 — shadow-mode placement uplift on real streams (×6)
fig20 C10 — empirical pilot power (trials-to-mastery effect + required N/arm)

Run: python3 -m viz.make_m14_figures   (skips panels with missing caches)
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz.viz_style as vs

vs.use_style(20)

DOM_FILL = ['#9671bd', '#77b5b6', '#bd8a71', '#7e7e7e', '#5a9367', '#b06060']
DOM_EDGE = ['#6a408d', '#378d94', '#8d5f44', '#4e4e4e', '#3d6b49', '#7e3d3d']
GLOBALS = [r"$\mu_\ell$", r"$\sigma_\ell$", r"$\mu_\theta$", r"$\sigma_\theta$",
           r"$\lambda_{fa}$", r"$\lambda_{miss}$"]


def fig15_model_compare():
    """A1 — neither the single-skill softmax (M5) nor +per-user-lapse (M5b)
    beats the composed marginals on equal-capacity held-out reads."""
    d = np.load(vs.FIGDIR + "/data_extset_m5.npz", allow_pickle=True)
    nt = d["n_user"].sum()
    elpd = {"composed": d["lp_composed"].sum() / nt,
            "M5": d["M5_lp"].sum() / nt, "M5b": d["M5b_lp"].sum() / nt}
    deltas = {"M5": d["delta_M5"], "M5b": d["delta_M5b"]}

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))

    ax = axes[0]
    names = ["composed", "M5b", "M5"]
    ys = np.arange(len(names))
    vals = np.array([elpd[n] for n in names])
    cols = [DOM_FILL[0], DOM_FILL[3], DOM_FILL[3]]
    edg = [DOM_EDGE[0], DOM_EDGE[3], DOM_EDGE[3]]
    xmin = vals.min() - 0.004
    ax.barh(ys, vals - xmin, left=xmin, color=cols, edgecolor=edg,
            linewidth=1.5, zorder=3, height=0.6)        # bar floor = xmin
    for y0, v in zip(ys, vals):
        ax.text(v - 0.0004, y0, f"{v:.4f}", va='center', ha='right',
                fontsize=13, color='white')
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=16)
    ax.set_xlabel("held-out label ELPD / read  (→ higher is better)")
    ax.set_xlim(xmin, vals.max() + 0.002)
    ax.set_title("Absolute fit (task2)", fontsize=18)
    vs.style_ax(ax)

    ax = axes[1]
    names = ["M5b", "M5"]
    ys = np.arange(len(names))
    for y0, nm in zip(ys, names):
        pt, lo, hi = deltas[nm]
        ax.plot([lo, hi], [y0, y0], color=vs.ACCENT_DARK, lw=2.0, zorder=2)
        ax.scatter([pt], [y0], s=150, color=vs.NEUTRAL,
                   edgecolors=vs.ACCENT_DARK, linewidths=1.4, zorder=3)
        ax.text(pt, y0 + 0.12, f"{pt:+.4f}\n[{lo:+.3f}, {hi:+.3f}]",
                ha='center', fontsize=12, color=vs.ACCENT_DARK)
    ax.axvline(0, color=vs.BAD, lw=2.0, zorder=1)
    ax.text(0.0004, 1.35, "composed\nbetter →", fontsize=12, color=vs.BAD)
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=16)
    ax.set_ylim(-0.5, 1.7)
    ax.set_xlabel("ΔELPD / read  (softmax − composed)")
    ax.set_title("Model comparison vs composed marginals", fontsize=18)
    vs.style_ax(ax)
    fig.tight_layout()
    return vs.save_fig(fig, "fig15_m14_model_compare", dpi=600)


def fig16_posterior_sbc():
    """A2 — top: posterior marginals of the M2 globals (lapses vs the engine's
    0.025); bottom: proper SBC rank-uniformity (R̂-gated), σ_θ the lone residual."""
    b = np.load(vs.FIGDIR + "/data_extset_bayes.npz", allow_pickle=True)
    s = np.load(vs.FIGDIR + "/data_extset_sbc.npz", allow_pickle=True)
    chain, lam = b["chain"], b["lam"]
    ranks, rhats = s["ranks"], s["rhats"]
    L, pvals = int(s["L"]), s["pvals"]

    fig, axes = plt.subplots(2, 6, figsize=(23, 9))
    # ── top: posterior marginals ──
    for i in range(6):
        ax = axes[0, i]
        draws = lam[:, i - 4] if i >= 4 else chain[:, i]
        col, edg = (DOM_FILL[i % 6], DOM_EDGE[i % 6])
        ax.hist(draws, bins=40, color=col, edgecolor=edg, linewidth=0.5,
                zorder=3)
        lo, md, hi = np.percentile(draws, [2.5, 50, 97.5])
        ax.axvline(md, color=vs.ACCENT_DARK, lw=2.0, zorder=4)
        if i >= 4:
            ax.axvline(0.025, color=vs.BAD, lw=2.2, ls='--', zorder=5)
        ax.set_title(f"{GLOBALS[i]}\n{md:.3f} [{lo:.3f}, {hi:.3f}]"
                     if i < 4 else
                     f"{GLOBALS[i]}\n{md:.4f} [{lo:.4f}, {hi:.4f}]",
                     fontsize=14)
        ax.set_yticks([])
        vs.style_ax(ax)
    axes[0, 0].set_ylabel("posterior", fontsize=16)
    axes[0, 5].text(0.025, axes[0, 5].get_ylim()[1] * 0.85, "engine\n0.025",
                    color=vs.BAD, fontsize=11, ha='left')

    # ── bottom: SBC rank histograms, R̂-gated ──
    for i in range(6):
        ax = axes[1, i]
        ok = rhats[:, i] < 1.1
        use = ok if ok.sum() >= 40 else np.ones(len(ranks), bool)
        h, edges = np.histogram(ranks[use, i], bins=10, range=(0, L))
        exp = use.sum() / 10.0
        passed = pvals[i] >= 0.05
        col = vs.GOOD if passed else vs.BAD
        ax.bar((edges[:-1] + edges[1:]) / 2, h, width=(L / 10) * 0.9,
               color=col, edgecolor=vs.ACCENT_DARK, linewidth=0.8, alpha=0.85,
               zorder=3)
        ax.axhspan(exp - 1.96 * np.sqrt(exp), exp + 1.96 * np.sqrt(exp),
                   color=vs.NEUTRAL, alpha=0.25, zorder=2)
        ax.axhline(exp, color=vs.NEUTRAL, lw=1.5, ls='--', zorder=2)
        ax.set_title(f"{GLOBALS[i]}  p={pvals[i]:.2f}\n"
                     f"R̂≤1.1: {int(ok.sum())}/{len(ranks)}", fontsize=13,
                     color=vs.ACCENT_DARK if passed else vs.BAD)
        ax.set_xticks([])
        ax.set_yticks([])
        vs.style_ax(ax)
    axes[1, 0].set_ylabel("SBC rank count", fontsize=16)

    fig.suptitle("A2 — Bayesian refit (top) and proper SBC calibration "
                 f"(bottom): P(both λ < 0.025 | data) = {float(b['pr_engine']):.3f}; "
                 "5/6 globals calibrated, σ_θ under-estimated", fontsize=19,
                 y=1.02)
    fig.tight_layout()
    return vs.save_fig(fig, "fig16_m14_posterior_sbc", dpi=600)


def fig17_speccurve():
    """A3 — specification curve: every one of 42 branches gives a positive ρ."""
    d = np.load(vs.FIGDIR + "/data_extset_speccurve.npz", allow_pickle=True)
    spec, cols = d["spec"], list(d["cols"])
    rho = spec[:, 5].astype(float)
    order = np.argsort(rho)
    spec, rho = spec[order], rho[order]
    frame = spec[:, 0]

    dims = ["gold", "signal", "readset", "min_reads"]
    levels = {dim: sorted(set(spec[:, cols.index(dim)]), key=str)
              for dim in dims}
    rows = [(dim, lv) for dim in dims for lv in levels[dim]]

    fig, axes = plt.subplots(2, 1, figsize=(15, 11),
                             gridspec_kw={"height_ratios": [2, 3]},
                             sharex=True)
    xs = np.arange(len(rho))

    ax = axes[0]
    fcol = np.where(frame == "t1", DOM_FILL[2], DOM_FILL[0])
    fed = np.where(frame == "t1", DOM_EDGE[2], DOM_EDGE[0])
    ax.vlines(xs, 0, rho, color=vs.NEUTRAL, lw=1.0, zorder=2)
    for x0, r0, c0, e0 in zip(xs, rho, fcol, fed):
        ax.scatter([x0], [r0], s=70, color=c0, edgecolors=e0, linewidths=1.2,
                   zorder=3)
    ax.axhline(0, color=vs.BAD, lw=2.0, zorder=1)
    ax.set_ylabel(r"validity  $\rho(\hat\ell,\ \mathrm{accuracy})$")
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(f"Specification curve — {len(rho)} branches, all positive "
                 f"(ρ ∈ [{rho.min():.2f}, {rho.max():.2f}])", fontsize=18)
    ax.scatter([], [], s=70, color=DOM_FILL[2], edgecolors=DOM_EDGE[2],
               label="task1 (binary)")
    ax.scatter([], [], s=70, color=DOM_FILL[0], edgecolors=DOM_EDGE[0],
               label="task2 (6-class)")
    ax.legend(loc='upper left', frameon=False, fontsize=14)
    vs.style_ax(ax)

    ax = axes[1]
    for yi, (dim, lv) in enumerate(rows):
        on = spec[:, cols.index(dim)] == lv
        ax.scatter(xs[on], np.full(on.sum(), yi), s=42, color=DOM_EDGE[0],
                   zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{dim}: {lv}" for dim, lv in rows], fontsize=12)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("specification (sorted by validity)")
    for b in np.cumsum([len(levels[d]) for d in dims])[:-1]:
        ax.axhline(b - 0.5, color=vs.NEUTRAL, lw=1.0, alpha=0.5)
    vs.style_ax(ax)
    fig.tight_layout()
    return vs.save_fig(fig, "fig17_m14_speccurve", dpi=600)


def fig18_disatten():
    """A4 — observed vs disattenuated validity; the binary frame's ℓ̂ and
    gold-accuracy are noise-limited measures of one construct (P2b → 1.00)."""
    d = np.load(vs.FIGDIR + "/data_extset_disatten.npz", allow_pickle=True)
    rel_ell, rel_acc, rho = d["rel_ell"], d["rel_acc"], d["rho_obs"]
    disatt = np.minimum(rho / np.sqrt(rel_ell * rel_acc), 1.0)
    tasks = ["task1 (binary)", "task2 (composite)"]

    fig, ax = plt.subplots(figsize=(10, 7))
    xs = np.arange(2)
    w = 0.36
    ax.bar(xs - w / 2, rho, width=w, color=DOM_FILL[1], edgecolor=DOM_EDGE[1],
           linewidth=1.5, zorder=3, label="observed ρ")
    ax.bar(xs + w / 2, disatt, width=w, color=DOM_FILL[0],
           edgecolor=DOM_EDGE[0], linewidth=1.5, zorder=3,
           label="disattenuated")
    for x0, ro, da in zip(xs, rho, disatt):
        ax.text(x0 - w / 2, ro + 0.02, f"{ro:.2f}", ha='center', fontsize=14)
        cap = "1.00 (cap)" if da >= 1.0 else f"{da:.2f}"
        ax.text(x0 + w / 2, da + 0.02, cap, ha='center', fontsize=14,
                color=vs.GOOD)
    ax.axhline(1.0, color=vs.NEUTRAL, lw=1.5, ls='--', zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{t}\nrel: ℓ̂ {re:.2f} · acc {ra:.2f}"
                        for t, re, ra in zip(tasks, rel_ell, rel_acc)],
                       fontsize=14)
    ax.set_ylabel("Spearman ρ (skill estimate vs gold accuracy)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Disattenuated convergent validity", fontsize=18, pad=10)
    ax.legend(loc='upper right', frameon=False, fontsize=14)
    vs.style_ax(ax)
    fig.tight_layout()
    return vs.save_fig(fig, "fig18_m14_disatten", dpi=600)


def fig19_shadow():
    """C9 — per-rater expected-learning-weight uplift of trainer placement
    over the real serving stream (shadow replay)."""
    d = np.load(vs.FIGDIR + "/data_extset_shadow.npz", allow_pickle=True)
    table = d["table"]
    uplift = table[:, 4].astype(float)
    n = len(uplift)
    med = np.median(uplift)
    q1, q3 = np.percentile(uplift, [25, 75])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(uplift, bins=30, color=DOM_FILL[1], edgecolor=DOM_EDGE[1],
            linewidth=1.0, zorder=3)
    ax.axvspan(q1, q3, color=vs.NEUTRAL, alpha=0.20, zorder=2,
               label=f"IQR [{q1:.2f}, {q3:.2f}]")
    ax.axvline(med, color=vs.GOOD, lw=2.6, zorder=4,
               label=f"median ×{med:.2f}")
    ax.axvline(1.0, color=vs.BAD, lw=2.0, ls='--', zorder=4,
               label="no improvement")
    ax.set_xlabel("expected-learning-weight uplift  (trainer ÷ real serving)")
    ax.set_ylabel(f"raters  (n = {n})")
    ax.set_title("Shadow-mode placement headroom on real streams",
                 fontsize=18, pad=10)
    ax.legend(loc='upper left', frameon=False, fontsize=13)
    vs.style_ax(ax)
    fig.tight_layout()
    return vs.save_fig(fig, "fig19_m14_shadow", dpi=600)


def fig20_pilot_power():
    """C10 — left: trials-to-true-mastery by arm (the effect size); right:
    required enrollment per arm vs staircase85 across (α, power)."""
    d = np.load(vs.FIGDIR + "/data_pilot_power.npz", allow_pickle=True)
    arms = ["tier2", "staircase85", "random"]
    true = {a: d[f"{a}_n_true"].astype(float) for a in arms}
    grid = d["grid"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    data = [true[a] for a in arms]
    cols = [vs.GOOD, DOM_FILL[2], DOM_FILL[5]]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.6,
                    showmeans=False, medianprops=dict(color=vs.ACCENT_DARK,
                    linewidth=2.2))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_edgecolor(vs.ACCENT_DARK)
        patch.set_alpha(0.85)
    for a, x0 in zip(arms, range(1, 4)):
        ax.text(x0, np.median(true[a]) + 8, f"med {np.median(true[a]):.0f}",
                ha='center', fontsize=13, color=vs.ACCENT_DARK)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(arms, fontsize=15)
    ax.set_ylabel("trials to TRUE mastery  (censored at budget)")
    ax.set_title("Effect size — training pace by arm (60 CRN seeds)",
                 fontsize=17)
    vs.style_ax(ax)

    ax = axes[1]
    sc = grid[grid[:, 0] == "staircase85"]
    labels = [f"α={r[1]}\npow={r[2]}" for r in sc]
    enroll = sc[:, 5].astype(float)
    xs = np.arange(len(sc))
    bars = ax.bar(xs, enroll, color=DOM_FILL[0], edgecolor=DOM_EDGE[0],
                  linewidth=1.5, zorder=3, width=0.62)
    strict = [i for i, r in enumerate(sc)
              if float(r[1]) == 0.005 and float(r[2]) == 0.9]
    for i in strict:
        bars[i].set_facecolor(vs.GOOD); bars[i].set_edgecolor(vs.GOOD)
    for x0, e0 in zip(xs, enroll):
        ax.text(x0, e0 + 0.5, f"{e0:.0f}", ha='center', fontsize=14)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel("enrollment required / arm  (vs staircase85)")
    ax.set_ylim(0, max(enroll) + 6)
    ax.set_title("Pilot sizing — ~33/arm at strict α (green)", fontsize=17)
    vs.style_ax(ax)
    fig.tight_layout()
    return vs.save_fig(fig, "fig20_m14_pilot_power", dpi=600)


if __name__ == "__main__":
    for fn in (fig15_model_compare, fig16_posterior_sbc, fig17_speccurve,
               fig18_disatten, fig19_shadow, fig20_pilot_power):
        try:
            print(fn.__name__, "→", fn()[0])
        except FileNotFoundError as e:
            print(fn.__name__, "skipped (data not ready):", e)
