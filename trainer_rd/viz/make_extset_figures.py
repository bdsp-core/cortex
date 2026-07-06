"""EXTSET result figures (M13), publication aesthetic per
`beautiful_figure_example_general.py` (codified in viz_style): purple/grey/
teal palette with darker edges, major+minor grids below the data, frameless
top-center legends, generous margins, PNG+PDF+SVG export to figures/.

fig9  task1 psychometric function (real reads vs engine + fitted link)
fig10 measurement validity (replay ℓ̂ vs accuracy; per-domain ρ)
fig11 cross-domain skill manifold (7×7)
fig12 link-ladder forest (held-out ΔELPD vs engine M0)
fig13 fitted population portrait in the engine state space
fig14 real-link stress campaign summary (P3)

Run: python3 -m viz.make_extset_figures   (skips panels with missing caches)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import viz.viz_style as vs
from training.extset_adapter import load_task1

vs.use_style(20)

# six domain hues: palette trio + muted companions (mode strip family)
DOM_FILL = ['#9671bd', '#77b5b6', '#bd8a71', '#7e7e7e', '#5a9367', '#b06060']
DOM_EDGE = ['#6a408d', '#378d94', '#8d5f44', '#4e4e4e', '#3d6b49', '#7e3d3d']


def _lam(v):
    return 0.49 / (1.0 + np.exp(-np.clip(v, -40, 40)))


def fig9_psychometric():
    """Real P(respond domain1 | s) vs the engine's assumed link and the
    fitted M2 link at the population mean (task1, 167,503 reads)."""
    t1 = load_task1()
    s, y = t1["s_loo"], t1["y"]
    q = np.quantile(s, np.linspace(0, 1, 21))
    mids, rates, half = [], [], []
    for a, b in zip(q[:-1], q[1:]):
        m = (s >= a) & (s < b)
        p = y[m].mean()
        mids.append(s[m].mean()); rates.append(p)
        half.append(1.96 * np.sqrt(p * (1 - p) / m.sum()))

    d = np.load(vs.FIGDIR + "/data_extset_link.npz", allow_pickle=True)
    x = d["t1_loo_M2_x"]
    mu_l, mu_t = x[0], x[2]
    lf, lm = _lam(x[4]), _lam(x[5])
    grid = np.linspace(min(mids) - 0.3, max(mids) + 0.3, 300)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.plot(grid, 0.025 + 0.95 * norm.cdf(grid), color=vs.NEUTRAL, lw=2.6,
            ls='--', zorder=2, label="engine link (λ=0.025, ℓ=θ=0)")
    ax.plot(grid, lf + (1 - lf - lm) * norm.cdf(np.exp(mu_l) * (grid + mu_t)),
            color=DOM_EDGE[0], lw=2.6, zorder=2,
            label="fitted M2 @ population mean")
    ax.errorbar(mids, rates, yerr=half, fmt='none', ecolor=vs.ACCENT_DARK,
                elinewidth=1.5, capsize=3, zorder=3)
    ax.scatter(mids, rates, s=90, color=DOM_FILL[1], edgecolors=DOM_EDGE[1],
               linewidths=1.5, zorder=4, label="real reads (vigintiles)")
    ax.set_xlabel("leave-one-user-out consensus signal  s")
    ax.set_ylabel("P(respond domain1)")
    ax.set_ylim(-0.04, 1.04)
    vs.style_ax(ax)
    vs.top_legend(ax, 2, anchor=1.16, fontsize=15)
    return vs.save_fig(fig, "fig9_extset_psychometric", dpi=600)


def fig10_replay():
    d = np.load(vs.FIGDIR + "/data_extset_replay.npz", allow_pickle=True)
    t1 = pd.DataFrame(d["t1"], columns=list(d["t1_cols"]))
    t1 = t1[t1.n >= 20].astype({"acc": float, "ell": float})
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    cf = np.polyfit(t1["acc"], t1["ell"], 1)
    gx = np.linspace(t1["acc"].min() - 0.03, t1["acc"].max() + 0.03, 50)
    ax.plot(gx, np.polyval(cf, gx), color=vs.NEUTRAL, lw=2.6, zorder=2,
            label="linear trend")
    ax.scatter(t1["acc"], t1["ell"], s=90, color=DOM_FILL[0],
               edgecolors=DOM_EDGE[0], linewidths=1.5, alpha=0.75, zorder=3,
               label="raters (n≥20 reads)")
    p2b = d["p2b"]
    ax.set_xlabel("accuracy vs source gold")
    ax.set_ylabel("replayed skill estimate  ℓ̂")
    ax.set_title(f"task1 (binary):  ρ = {float(p2b[0]):.2f},  "
                 f"n = {int(p2b[2])}", fontsize=18, pad=34)
    vs.style_ax(ax)
    vs.top_legend(ax, 2, anchor=1.085, fontsize=13)

    ax = axes[1]
    per = d["per_domain"].astype(float)
    xs = np.arange(len(per))
    ax.bar(xs, per[:, 1], color=DOM_FILL, edgecolor=DOM_EDGE, linewidth=1.5,
           zorder=3, width=0.62)
    for xi, rho in zip(xs, per[:, 1]):
        ax.text(xi, rho + 0.02, f"{rho:.2f}", ha='center', fontsize=14,
                color=vs.ACCENT_DARK)
    ax.axhline(float(d["p2"][0]), color=vs.NEUTRAL, lw=2.6, ls='--', zorder=2)
    ax.text(len(per) - 0.45, float(d["p2"][0]) + 0.02,
            f"mean-ℓ̂ composite {float(d['p2'][0]):.2f}", ha='right',
            fontsize=13, color=vs.ACCENT_DARK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"domain{int(k)}" for k in per[:, 0]], fontsize=14)
    ax.set_ylabel("Spearman ρ(ℓ̂, accuracy)")
    ax.set_ylim(0, 1.0)
    ax.set_title("task2 per-domain validity (n = 315)", fontsize=18)
    vs.style_ax(ax)
    fig.tight_layout()
    return vs.save_fig(fig, "fig10_extset_replay", dpi=600)


def fig11_manifold():
    d = np.load(vs.FIGDIR + "/data_extset_xdomain.npz", allow_pickle=True)
    corr, labels = d["corr"].astype(float), list(d["corr_labels"])
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, vmin=0, vmax=1, cmap="BuPu")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"domain{l[1:]}" for l in labels], rotation=45,
                       ha='right', fontsize=14)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([f"domain{l[1:]}" for l in labels], fontsize=14)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center',
                    fontsize=13,
                    color='white' if corr[i, j] > 0.55 else '#333333')
    cb = fig.colorbar(im, shrink=0.85, pad=0.02)
    cb.set_label("Spearman ρ (per-rater accuracy)", fontsize=15)
    p4, rel = d["p4"].astype(float), d["reliability"].astype(float)
    ax.set_title("Cross-domain skill manifold — dual-contest cohort\n"
                 f"ρ(domain1, domains2–7) = {p4[0]:.2f}  (n = {int(p4[2])}, "
                 f"disattenuated {rel[2]:.2f})", fontsize=17, pad=14)
    fig.tight_layout()
    return vs.save_fig(fig, "fig11_extset_manifold", dpi=600)


def fig12_ladder():
    d = np.load(vs.FIGDIR + "/data_extset_link.npz", allow_pickle=True)
    rows = [(r[0], r[1], float(r[2]), float(r[3]), float(r[4]))
            for r in d["summary"]]
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(10, 0.52 * len(rows) + 2.4))
    ys = np.arange(len(rows))
    for y0, (key, best, pt, lo, hi) in zip(ys, rows):
        sig = lo > 0
        ax.plot([lo, hi], [y0, y0], color=vs.ACCENT_DARK, lw=1.6, zorder=2)
        ax.scatter([pt], [y0], s=110,
                   color=vs.GOOD if sig else vs.NEUTRAL,
                   edgecolors=vs.ACCENT_DARK, linewidths=1.2, zorder=3)
        ax.text(max(hi for *_, hi in [r[2:] for r in rows]) + 0.0012, y0,
                best, va='center', fontsize=13, color=vs.ACCENT_DARK)
    ax.axvline(0, color=vs.BAD, lw=1.6, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0].replace("_", " ") for r in rows], fontsize=13)
    ax.set_xlabel("held-out ΔELPD per read  (selected link − engine M0)")
    ax.set_title("Real link vs the engine's locked likelihood — all frames",
                 fontsize=18, pad=12)
    vs.style_ax(ax)
    fig.tight_layout()
    return vs.save_fig(fig, "fig12_extset_ladder", dpi=600)


def fig13_population():
    """Fitted novice populations in the engine's (θ, ℓ) state space, vs the
    engine's standard-normal prior (LOO frames, selected models)."""
    d = np.load(vs.FIGDIR + "/data_extset_link.npz", allow_pickle=True)
    sel = {r[0]: r[1] for r in d["summary"]}
    frames = [("t1_loo", "domain1")] + [(f"t2_d{k}_loo", f"domain{k}")
                                        for k in range(2, 8)]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='datalim')
    for r in (1, 2):
        ax.add_patch(Ellipse((0, 0), 2 * r, 2 * r, fill=False,
                             color=vs.NEUTRAL, lw=2.0 if r == 1 else 1.0,
                             ls='--', zorder=2))
    ax.text(0, 2.08, "engine prior N(0,1)²  (1σ, 2σ)", ha='center',
            fontsize=13, color=vs.NEUTRAL)
    cols = ['#d4a23c'] + DOM_FILL          # gold for domain1, trio for 2–7
    edges = ['#9a7320'] + DOM_EDGE
    for i, (key, name) in enumerate(frames):
        x = d[f"{key}_{sel[key]}_x"]
        mu_t, sd_t, mu_l, sd_l = x[2], x[3], x[0], x[1]
        ax.add_patch(Ellipse((mu_t, mu_l), 2 * sd_t, 2 * sd_l,
                             facecolor=cols[i], edgecolor=edges[i],
                             linewidth=1.8, alpha=0.40, zorder=3))
        ax.scatter([mu_t], [mu_l], s=90, color=cols[i], edgecolors=edges[i],
                   linewidths=1.5, zorder=4, label=name)
    ax.axhline(0, color='#bbbbbb', lw=1.0, zorder=1)
    ax.axvline(0, color='#bbbbbb', lw=1.0, zorder=1)
    ax.set_xlabel("criterion / bias  θ")
    ax.set_ylabel("log-skill  ℓ")
    ax.set_title("Where real novices live in the engine state space\n"
                 "(fitted population, ±1σ; LOO frames)", fontsize=18, pad=58)
    vs.style_ax(ax)
    vs.top_legend(ax, 4, anchor=1.115, fontsize=13)
    return vs.save_fig(fig, "fig13_extset_population", dpi=600)


def fig14_stress():
    d = np.load(vs.FIGDIR + "/data_extset_stress.npz", allow_pickle=True)
    members = list(d["members"])
    configs = list(d["configs"])
    n_decl, ell_decl, t_decl = d["n_decl"], d["ell_decl"], d["t_decl"]
    budget = float(d["budget"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    w, xs = 0.38, np.arange(len(members))
    for j, cfg in enumerate(configs):
        med = [np.nanmedian(n_decl[i, j][n_decl[i, j] <= budget])
               for i in range(len(members))]
        ax.bar(xs + (j - 0.5) * w, med, width=w * 0.92,
               color=DOM_FILL[1] if cfg == "shipped" else DOM_FILL[4],
               edgecolor=DOM_EDGE[1] if cfg == "shipped" else DOM_EDGE[4],
               linewidth=1.5, zorder=3, label=cfg)
        for xi, m in zip(xs + (j - 0.5) * w, med):
            ax.text(xi, m + 1.5, f"{m:.0f}", ha='center', fontsize=14)
    ax.set_xticks(xs); ax.set_xticklabels(members, fontsize=15)
    ax.set_ylabel("median trials to declaration")
    ax.set_title("Training pace under the real link", fontsize=18)
    vs.style_ax(ax)
    vs.top_legend(ax, 2, anchor=1.13, fontsize=15)

    ax = axes[1]
    from scipy.stats import beta
    L_STAR, T_STAR = 0.5337430687087749, 0.30
    for j, cfg in enumerate(configs):
        fg, lo, hi = [], [], []
        for i in range(len(members)):
            decl = n_decl[i, j] <= budget
            k = int(np.nansum(decl & ((ell_decl[i, j] < L_STAR)
                                      | (np.abs(t_decl[i, j]) > T_STAR))))
            n = int(decl.sum())
            fg.append(k / n)
            lo.append(beta.ppf(0.025, k + 0.5, n - k + 0.5))
            hi.append(beta.ppf(0.975, k + 0.5, n - k + 0.5))
        ax.errorbar(xs + (j - 0.5) * 0.18, fg,
                    yerr=[np.array(fg) - np.array(lo),
                          np.array(hi) - np.array(fg)],
                    fmt='o', ms=11, capsize=4, elinewidth=1.6,
                    color=DOM_FILL[1] if cfg == "shipped" else DOM_FILL[4],
                    markeredgecolor=DOM_EDGE[1] if cfg == "shipped"
                    else DOM_EDGE[4], markeredgewidth=1.5, zorder=3,
                    label=cfg)
    ax.set_xticks(xs); ax.set_xticklabels(members, fontsize=15)
    ax.set_ylabel("false-graduation rate (latent), 95% CI")
    ax.set_ylim(0, 0.85)
    ax.set_title("Premature-declaration risk (D19 hardening helps "
                 "identically)", fontsize=18)
    vs.style_ax(ax)
    vs.top_legend(ax, 2, anchor=1.13, fontsize=15)
    fig.tight_layout()
    return vs.save_fig(fig, "fig14_extset_stress", dpi=600)


if __name__ == "__main__":
    for fn in (fig9_psychometric, fig10_replay, fig11_manifold, fig12_ladder,
               fig13_population, fig14_stress):
        try:
            print(fn.__name__, "→", fn()[0])
        except FileNotFoundError as e:
            print(fn.__name__, "skipped (data not ready):", e)
