"""M10.1 — performance figures for the learning algorithm (publication style).

Five figures into ./figures (PNG 300 dpi + PDF + SVG), beautiful_figure
aesthetic via viz_style. Simulation data is generated on first run and cached
as .npz next to the figures so the script is cheap to re-run.

  fig1_benchmark      trials-to-mastery by policy × cell, paired-seed bootstrap
                      CIs (30 seeds; production-faithful physics, M10).
  fig2_misspec        F17/D14 — coverage asymmetry under rate misspecification
                      and the believed-vs-actual-SD mechanism.
  fig3_floor          F5 — filtered posterior SD settling onto the variance
                      floor; analytic Riccati prediction; real-bank regime.
  fig4_attractor      F23 — the criterion follows the served-stream midpoint:
                      three skill-placement designs.
  fig5_designs        M10 sandbox outcomes (recorded): retention scheduling
                      A/B/C and mastery-gate G0 vs G3 (domain1 never-detect).
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from viz.viz_style import (TASK_FILL, TASK_EDGE, NEUTRAL, GOOD, BAD, ACCENT_DARK,
                       FIGDIR, use_style, style_ax, save_fig, top_legend)

use_style(20)
RNG = np.random.default_rng(7)
PURPLE, GREY, TEAL = TASK_FILL
PURPLE_D, GREY_D, TEAL_D = TASK_EDGE


def _cache(name):
    return os.path.join(FIGDIR, name)


def _boot_ci(v, n=4000, rng=RNG):
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return np.percentile(bs, [2.5, 97.5])


# ───────────────────────── fig 1: benchmark ─────────────────────────

def fig1_benchmark():
    d = np.load(_cache("data_benchmark.npz"))
    cells = [("soft", 1.0, "well-specified (soft rule, ×1 rates)"),
             ("soft", 2.0, "rates over-estimated ×2 (soft rule)"),
             ("hard", 1.0, "hard-rule learner (×1 rates)"),
             ("hard", 2.0, "hard rule + rates ×2")]
    pols = ["tier2", "tier1", "measure_opt", "staircase85", "random"]
    labels = ["Tier-2 (mode policy)", "Tier-1 (greedy)",
              "criterion placement", "85% staircase", "random"]
    colors = [PURPLE, TEAL, GREY, GREY, GREY]
    edges = [PURPLE_D, TEAL_D, GREY_D, GREY_D, GREY_D]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True, sharey=True)
    for ax, (rule, rm, title) in zip(axes.ravel(), cells):
        ypos = np.arange(len(pols))[::-1]
        for yp, p, c, e, lab in zip(ypos, pols, colors, edges, labels):
            v = d[f"{rule}_{rm}_{p}"].astype(float)
            lo, hi = _boot_ci(v)
            cens = int(np.sum(v >= 400))
            ax.plot([lo, hi], [yp, yp], color=e, lw=3.2, solid_capstyle='round',
                    zorder=2)
            ax.scatter([v.mean()], [yp], s=170, color=c, edgecolors=e,
                       linewidths=1.8, zorder=3)
            note = f"{v.mean():.0f}" + (f"  ({cens}/30 censored)" if cens else "")
            ax.text(hi * 1.12, yp, note, va='center', fontsize=15,
                    color=ACCENT_DARK)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.set_xscale('log')
        ax.set_xlim(30, 900)
        ax.set_title(title, fontsize=18)
        style_ax(ax)
    for ax in axes[1]:
        ax.set_xlabel("trials to true mastery (σ ≤ σ*, |t| ≤ t*)")
    fig.suptitle("Trials-to-mastery under production-faithful stimulus noise "
                 "(30 paired seeds, 95% bootstrap CI)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_fig(fig, "fig1_benchmark")


# ───────────────────── fig 2: misspecification asymmetry ─────────────────────

def _gen_misspec(path, n_reps=25, n_trials=120, Npart=600):
    from training.bridge_conventions import SKILL_MODE_MULTIPLIER, engine_to_plan
    from training.learner_sim import Learner, LearnerParams
    from training.training_filter import TaskFilter

    def wq(v, w, q):
        o = np.argsort(v); cw = np.cumsum(w[o]); cw /= cw[-1]
        return float(np.interp(q, cw, v[o]))

    rows = {}
    for rm in (0.5, 1.0, 2.0):
        bel, err, cover = [], [], []
        for r in range(n_reps):
            tp = dict(alpha_t=0.15, alpha_sigma=1 / 40, sigma_inf=0.5,
                      q_t=0.05, q_sigma=0.03, rho=0.5, rule="soft")
            fp = dict(tp); fp["alpha_t"] *= rm; fp["alpha_sigma"] *= rm
            lnr = Learner([1.4], [0.6], LearnerParams(**tp), seed=r)
            rg = np.random.default_rng(1000 + r)
            f = TaskFilter(-0.6 + 0.25 * rg.standard_normal(Npart),
                           -np.log(1.4) + 0.25 * rg.standard_normal(Npart),
                           LearnerParams(**fp), seed=5000 + r)
            for k in range(n_trials):
                sig_hat, t_hat = engine_to_plan(*f.mean())
                s = (t_hat + (0.3 if (k // 2) % 2 == 0 else -0.3) * sig_hat
                     if k % 2 == 0 else
                     t_hat + (1 if (k // 2) % 2 == 0 else -1)
                     * SKILL_MODE_MULTIPLIER * sig_hat)
                ys = int(s > 0)
                y = lnr.step(s, 0, y_star=ys, feedback=True)
                f.step(s, y, ys, feedback=True)
            _, sdl = f.sd(); _, ml = f.mean()
            bel.append(sdl)
            err.append(ml - (-np.log(lnr.sigma[0])))
            lo = np.exp(-wq(f.ell, f.w, 0.975)); hi = np.exp(-wq(f.ell, f.w, 0.025))
            tlo = -wq(f.theta, f.w, 0.975); thi = -wq(f.theta, f.w, 0.025)
            cover.append(int(lo <= lnr.sigma[0] <= hi and
                             tlo <= lnr.t[0] <= thi))
        rows[rm] = (np.mean(bel), np.std(err), np.mean(cover))
    np.savez(path, mults=np.array(list(rows)),
             believed=np.array([rows[m][0] for m in rows]),
             actual=np.array([rows[m][1] for m in rows]),
             coverage=np.array([rows[m][2] for m in rows]))


def fig2_misspec():
    path = _cache("data_misspec.npz")
    if not os.path.exists(path):
        print("  generating misspec data ...")
        _gen_misspec(path)
    d = np.load(path)
    mults, bel, act, cov = d["mults"], d["believed"], d["actual"], d["coverage"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.2))
    x = np.arange(len(mults))
    ax1.plot(x, cov, '-', color=PURPLE_D, lw=2.6, zorder=2)
    ax1.scatter(x, cov, s=190, color=PURPLE, edgecolors=PURPLE_D,
                linewidths=1.8, zorder=3)
    ax1.axhline(0.9025, color=NEUTRAL, lw=2.0, ls='--')
    ax1.text(0.02, 0.915, "nominal (95%×95% box)", color=NEUTRAL, fontsize=15,
             transform=ax1.get_yaxis_transform())
    ax1.set_xticks(x); ax1.set_xticklabels(["×0.5\n(under-est.)", "×1\n(correct)",
                                            "×2\n(over-est.)"])
    ax1.set_ylim(0.4, 1.05)
    ax1.set_xlabel("assumed learning rates vs truth")
    ax1.set_ylabel("credible-box coverage")
    ax1.set_title("filter coverage vs rate misspecification (F17)")
    style_ax(ax1)

    w = 0.38
    ax2.bar(x - w / 2, bel, w, color=PURPLE, edgecolor=PURPLE_D, lw=1.5,
            label="believed posterior SD(ℓ)")
    ax2.bar(x + w / 2, act, w, color=TEAL, edgecolor=TEAL_D, lw=1.5,
            label="actual error SD(ℓ)")
    ax2.set_xticks(x); ax2.set_xticklabels(["×0.5", "×1", "×2"])
    ax2.set_xlabel("assumed learning rates vs truth")
    ax2.set_ylabel("SD of skill estimate")
    ax2.set_title("the mechanism: belief vs reality")
    top_legend(ax2, 2, anchor=1.16)
    style_ax(ax2)
    fig.text(0.5, -0.02,
             "Over-estimated rates shrink believed uncertainty exactly while real error grows — "
             "the anti-conservative direction for graduation ⇒ priors lean low (D14).",
             ha='center', fontsize=15, color='#555555')
    fig.tight_layout()
    return save_fig(fig, "fig2_misspec")


# ───────────────────────── fig 3: variance floor ─────────────────────────

def _gen_floor(path, n_particles=1500, n_trials=300):
    from training.bridge_conventions import SKILL_MODE_MULTIPLIER, engine_to_plan
    from training.learner_sim import Learner, LearnerParams
    from training.training_filter import TaskFilter
    params = LearnerParams(q_t=0.05, q_sigma=0.03, rule="soft")
    out = {}
    for s_sd in (0.0, 0.85):
        trajs = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            lnr = Learner([0.5], [0.0], params, seed=seed)
            f = TaskFilter(0.25 * rng.standard_normal(n_particles),
                           -np.log(0.5) + 0.25 * rng.standard_normal(n_particles),
                           params, seed=seed + 1)
            sds, sign = [], 1
            for k in range(n_trials):
                sig_hat, t_hat = engine_to_plan(*f.mean())
                s = t_hat + sign * SKILL_MODE_MULTIPLIER * sig_hat
                sign *= -1
                ys = int(s > 0)
                s_real = s + (s_sd * rng.standard_normal() if s_sd else 0.0)
                y = lnr.step(s_real, 0, y_star=ys, feedback=True)
                f.step(s, y, ys, s_sd=s_sd, feedback=True)
                sds.append(f.sd()[1])
            trajs.append(sds)
        out[f"traj_{s_sd}"] = np.array(trajs)
    np.savez(path, **out)


def fig3_floor():
    path = _cache("data_floor.npz")
    if not os.path.exists(path):
        print("  generating floor data ...")
        _gen_floor(path)
    d = np.load(path)

    fig, ax = plt.subplots(figsize=(13, 8))
    x = np.arange(d["traj_0.0"].shape[1])
    for tr in d["traj_0.85"]:
        ax.plot(x, tr, color=TEAL, lw=1.6, alpha=0.8, zorder=2)
    for tr in d["traj_0.0"]:
        ax.plot(x, tr, color=PURPLE, lw=1.6, alpha=0.8, zorder=2)
    ax.plot([], [], color=PURPLE, lw=2.4, label="synthetic items (s_sd = 0)")
    ax.plot([], [], color=TEAL, lw=2.4, label="real-bank items (s_sd = 0.85)")
    ax.axhline(0.23, color=ACCENT_DARK, lw=2.2, ls='-')
    ax.text(297, 0.237, "mastery gate sd_floor = 0.23", ha='right', fontsize=15,
            color=ACCENT_DARK)
    ax.axhline(0.152, color=PURPLE_D, lw=1.8, ls='--')
    ax.text(297, 0.157, "measured floor 0.152 ± 0.002", ha='right',
            fontsize=14, color=PURPLE_D)
    ax.axhline(0.138, color=NEUTRAL, lw=1.8, ls=':')
    ax.text(297, 0.122, "analytic (Riccati) 0.138", ha='right', fontsize=14,
            color=NEUTRAL)
    ax.axhline(0.179, color=TEAL_D, lw=1.8, ls='--')
    ax.text(297, 0.184, "real-bank floor 0.179", ha='right', fontsize=14,
            color=TEAL_D)
    ax.set_xlim(0, 300); ax.set_ylim(0.08, 0.45)
    ax.set_xlabel("training trial")
    ax.set_ylabel("filtered posterior SD(ℓ)")
    ax.set_title("the information-balanced variance floor (F5):\n"
                 "process noise stops belief contraction — the mastery gate "
                 "must sit above the floor", fontsize=18)
    top_legend(ax, 2, anchor=1.02)
    style_ax(ax)
    fig.tight_layout()
    return save_fig(fig, "fig3_floor")


# ───────────────────────── fig 4: F23 attractor ─────────────────────────

def _gen_attractor(path, n_reps=8, n_trials=240):
    from training.bank_adapter import BankAdapter, SIGMA_STAR
    from training.bridge_conventions import engine_to_plan
    from training.learner_sim import Learner, LearnerParams
    from training.training_filter import TaskFilter
    from training.trainer_policy import expected_skill_weight

    ad = BankAdapter()
    full = ad.candidates(2, feedback_safe=True)
    sub = np.random.default_rng(0).choice(len(full), size=2000, replace=False)
    pool = full.subset(np.isin(np.arange(len(full)), sub))
    sig_inf = 0.82 * SIGMA_STAR[2]

    def run(scheme, seed):
        tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=sig_inf,
                           q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
        lnr = Learner([1.5], [0.6], tp, seed=seed)
        rng = np.random.default_rng(900 + seed)
        f = TaskFilter(0.3 * rng.standard_normal(400),
                       0.3 * rng.standard_normal(400), tp, seed=seed)
        traj, last_s = [], None
        for k in range(n_trials):
            sig_hat, t_hat = engine_to_plan(*f.mean())
            side = 1 if k % 2 == 0 else -1
            want = 1 if side > 0 else 0
            m_ = np.where(pool.y_star == want)[0]
            center = t_hat if scheme == "criterion-centred" else 0.0
            m_eff = 1.0772256
            if scheme == "mirror-paired" and last_s is not None and \
                    (last_s > 0) != (side > 0):
                m_eff = abs(last_s) / max(sig_hat, 1e-6)
            ew = expected_skill_weight(pool.s_mean[m_], pool.s_sd[m_],
                                       sig_hat, center, side=side, m=m_eff,
                                       rho=0.6)
            idx = int(m_[np.argmax(ew)])
            s, ssd, ys = (float(pool.s_mean[idx]), float(pool.s_sd[idx]),
                          int(pool.y_star[idx]))
            last_s = s
            s_real = s + ssd * rng.standard_normal()
            y = lnr.step(s_real, 0, ys, feedback=True)
            f.step(s, y, ys, s_sd=ssd, feedback=True)
            traj.append(lnr.t[0])
        return traj

    out = {}
    for scheme in ("criterion-centred", "boundary-centred", "mirror-paired"):
        out[scheme.replace("-", "_")] = np.array(
            [run(scheme, s) for s in range(n_reps)])
    np.savez(path, **out)


def fig4_attractor():
    path = _cache("data_attractor.npz")
    if not os.path.exists(path):
        print("  generating attractor data ...")
        _gen_attractor(path)
    d = np.load(path)
    schemes = [("criterion_centred", "criterion-centred placement", GREY, GREY_D),
               ("boundary_centred", "boundary-centred, per-side argmax", TEAL,
                TEAL_D),
               ("mirror_paired", "boundary-centred, mirror-paired (shipped)",
                PURPLE, PURPLE_D)]
    fig, ax = plt.subplots(figsize=(13, 8))
    x = np.arange(d["mirror_paired"].shape[1])
    for key, lab, c, e in schemes:
        tr = d[key]
        mu, sd = tr.mean(axis=0), tr.std(axis=0)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.16, lw=0)
        ax.plot(x, mu, color=e, lw=2.8, label=lab)
    ax.axhline(0, color=ACCENT_DARK, lw=1.4)
    ax.axhspan(-0.30, 0.30, color=GOOD, alpha=0.08, lw=0)
    ax.text(4, -0.26, "bias tolerance |t| ≤ t*", ha='left', fontsize=15,
            color=GOOD)
    ax.set_xlim(0, len(x) - 1); ax.set_ylim(-0.42, 0.78)
    ax.set_xlabel("skill-mode trial")
    ax.set_ylabel("learner's TRUE criterion  t")
    ax.set_title("the criterion follows the served-stream midpoint (F23):\n"
                 "item placement is the de-facto bias setpoint", fontsize=18)
    ax.legend(loc='lower right', frameon=False, fontsize=15)
    style_ax(ax)
    fig.tight_layout()
    return save_fig(fig, "fig4_attractor")


# ───────────────────── fig 5: sandbox design outcomes ─────────────────────

def fig5_designs():
    # Recorded M10 sandbox results (PROJECT_MEMORY.md §3 F18/F19); the sandbox
    # scripts were deleted after recording, per plan.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.6))

    designs = ["A\nno retention\n(status quo)", "C\nblind periodic",
               "B\ndue-driven\n(shipped, D17)"]
    deficit = [0.322, 0.256, 0.016]
    deficit_sd = [0.089, 0.129, 0.061]
    stale = [0.567, 0.521, 0.415]
    x = np.arange(3)
    w = 0.38
    ax1.bar(x - w / 2, deficit, w, yerr=deficit_sd, capsize=5, color=PURPLE,
            edgecolor=PURPLE_D, lw=1.5, label="post-gap skill deficit ℓ*−ℓ")
    ax1.bar(x + w / 2, stale, w, color=TEAL, edgecolor=TEAL_D, lw=1.5,
            label="belief staleness ℓ̂−ℓ")
    ax1.axhline(0, color=ACCENT_DARK, lw=1.2)
    ax1.set_xticks(x); ax1.set_xticklabels(designs, fontsize=15)
    ax1.set_ylim(-0.08, 0.78)
    ax1.set_ylabel("ℓ gap after 6 sessions with forgetting")
    ax1.set_title("retention scheduling (F18 sandbox)", fontsize=18, pad=14)
    ax1.legend(loc='upper right', frameon=False, fontsize=14)
    style_ax(ax1)

    doms = ["domain1\ngap 0.085", "domain3\ngap 0.231", "domain2\ngap 0.373"]
    g0 = [0.0, 1.0, 1.0]
    g3 = [1.0, 1.0, 1.0]
    g3_med = [37, 20, 20]
    x = np.arange(3)
    ax2.bar(x - w / 2, g0, w, color=GREY, edgecolor=GREY_D, lw=1.5,
            label="pass-mass gate (G0)")
    ax2.bar(x + w / 2, g3, w, color=PURPLE, edgecolor=PURPLE_D, lw=1.5,
            label="mean-skill gate (G3, shipped in hybrid D16)")
    for xi, m in zip(x, g3_med):
        ax2.text(xi + w / 2, 1.02, f"med {m} trials", ha='center', fontsize=13,
                 color=PURPLE_D)
    ax2.text(0 - w / 2, 0.04, "never\nfires", ha='center', fontsize=13,
             color=BAD, fontweight='bold')
    ax2.set_xticks(x); ax2.set_xticklabels(doms, fontsize=15)
    ax2.set_ylim(0, 1.32)
    ax2.set_ylabel("detection rate at the expert ceiling")
    ax2.set_title("mastery-gate detection (F19 sandbox)", fontsize=18, pad=14)
    ax2.legend(loc='upper left', frameon=False, fontsize=13,
               labels=["pass-mass gate (G0)", "mean-skill gate (G3, in D16)"])
    style_ax(ax2)
    fig.text(0.5, -0.02,
             "False-graduation rate was 0.00 for ALL gate variants at 0.15 below the cut "
             "(20 reps × 300 trials per cell; recorded M10 sandbox campaign).",
             ha='center', fontsize=14, color='#555555')
    fig.tight_layout()
    return save_fig(fig, "fig5_designs")


if __name__ == "__main__":
    for fn in (fig1_benchmark, fig2_misspec, fig3_floor, fig4_attractor,
               fig5_designs):
        print(fn.__name__, "->", fn()[0])
