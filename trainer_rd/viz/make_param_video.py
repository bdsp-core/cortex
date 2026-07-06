"""M10.1 — animated MP4: skill & bias parameter evolution per training question,
compared across selection policies on ONE shared axis.

For each policy {Tier-2 mode policy, Tier-1 greedy, random} the SAME simulated
learner cohort (15 paired seeds, identical start state, real domain3 bank,
production-faithful stimulus noise — the benchmark harness physics) trains for
a fixed horizon. We record the learner's TRUE parameters each question:

    skill  ℓ = −log σ   (higher = better; must clear the cert cut ℓ*)
    bias   t            (must settle inside the ±t* tolerance band)

The animation reveals mean ±1 SD trajectories question-by-question; solid
lines = skill, dashed = bias, colour = policy. Mean first-mastery (true state
crosses σ ≤ σ* AND |t| ≤ t*) is flagged per policy.

Run:  python3 -m viz.make_param_video → figures/param_evolution_demo.mp4
"""
from __future__ import annotations

import os

import numpy as np
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

from training.bank_adapter import BankAdapter, ELL_STAR, SIGMA_STAR, TaskCandidates
from training.benchmark_trainer import _select, T_STAR
from training.learner_sim import Learner, LearnerParams
from training.training_filter import TaskFilter
from training.trainer_greedy import RewardWeights
from viz.viz_style import (TASK_FILL, TASK_EDGE, NEUTRAL, GOOD, ACCENT_DARK,
                       FIGDIR, use_style, style_ax)

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

TASK = 2                       # domain3 — the benchmark task
N_SEEDS = 15
HORIZON = 300
POOL_SIZE = 1000
FPS = 12

POLICIES = [("tier2", "Tier-2 (mode policy)", TASK_FILL[0], TASK_EDGE[0]),
            ("tier1", "Tier-1 (greedy)", TASK_FILL[2], TASK_EDGE[2]),
            ("random", "random", TASK_FILL[1], TASK_EDGE[1])]


def run_traced(policy, pool, seed, horizon=HORIZON):
    """benchmark_trainer.run_one physics, fixed horizon, full trajectory."""
    sig_star, ell_star = SIGMA_STAR[TASK], ELL_STAR[TASK]
    sig_inf = 0.82 * sig_star
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=sig_inf,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    lnr = Learner([1.5], [0.8], tp, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    filt = TaskFilter(0.3 * rng.standard_normal(400),
                      0.3 * rng.standard_normal(400), tp, seed=seed)
    greedyW = RewardWeights()
    state = {"k": 0, "bal": 0}
    ell_tr, t_tr, mastered_at = [], [], None
    for k in range(horizon):
        state["k"] = k
        idx = _select(policy, filt, pool, state, rng, greedyW)
        s, s_sd, y_star = (float(pool.s_mean[idx]), float(pool.s_sd[idx]),
                           int(pool.y_star[idx]))
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        state["bal"] += 1 if y_star == 1 else -1
        ell_tr.append(-np.log(lnr.sigma[0]))
        t_tr.append(lnr.t[0])
        if mastered_at is None and (lnr.sigma[0] <= sig_star
                                    and abs(lnr.t[0]) <= T_STAR):
            mastered_at = k + 1
    return np.array(ell_tr), np.array(t_tr), mastered_at


def gen_data(path):
    ad = BankAdapter()
    full = ad.candidates(TASK, feedback_safe=True)
    sub = np.random.default_rng(0).choice(len(full), size=POOL_SIZE,
                                          replace=False)
    pool = TaskCandidates(TASK, full.seg_id[sub], full.s_mean[sub],
                          full.s_sd[sub], full.y_star[sub], full.margin[sub],
                          full.coherent[sub])
    out = {}
    for pol, _, _, _ in POLICIES:
        E, T, M = [], [], []
        for s in range(N_SEEDS):
            e, t, m = run_traced(pol, pool, seed=s)
            E.append(e); T.append(t); M.append(np.nan if m is None else m)
        out[f"ell_{pol}"] = np.array(E)
        out[f"t_{pol}"] = np.array(T)
        out[f"mast_{pol}"] = np.array(M, dtype=np.float64)
        print(f"  {pol}: mean first-mastery "
              f"{np.nanmean(out[f'mast_{pol}']):.0f} "
              f"({np.mean(~np.isnan(out[f'mast_{pol}'])) * 100:.0f}% within "
              f"{HORIZON})")
    np.savez(path, **out)
    return out


def render(d, out_path, fps=FPS):
    use_style(18)
    ell_star = ELL_STAR[TASK]
    x_all = np.arange(1, HORIZON + 1)

    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.80, bottom=0.10)

    hold = fps * 3
    seq = list(range(2, HORIZON + 1, 1)) + [HORIZON] * hold

    def draw(upto):
        ax.clear()
        x = x_all[:upto]
        # reference geometry (shared scale)
        ax.axhspan(-T_STAR, T_STAR, color=GOOD, alpha=0.08, lw=0, zorder=1)
        ax.axhline(0.0, color='#bbbbbb', lw=1.0, zorder=1)
        ax.axhline(ell_star, color=ACCENT_DARK, lw=2.0, ls='--', zorder=2)
        ax.text(HORIZON * 0.995, ell_star + 0.025, "skill cut  ℓ*",
                ha='right', fontsize=15, color=ACCENT_DARK)
        ax.text(HORIZON * 0.995, -T_STAR + 0.02, "bias tolerance  |t| ≤ t*",
                ha='right', fontsize=15, color=GOOD)

        for pi, (pol, lab, c, e) in enumerate(POLICIES):
            ell = d[f"ell_{pol}"][:, :upto]
            tt = d[f"t_{pol}"][:, :upto]
            ax.fill_between(x, ell.mean(0) - ell.std(0),
                            ell.mean(0) + ell.std(0), color=c, alpha=0.10,
                            lw=0, zorder=2)
            ax.fill_between(x, tt.mean(0) - tt.std(0),
                            tt.mean(0) + tt.std(0), color=c, alpha=0.10,
                            lw=0, zorder=2)
            ax.plot(x, ell.mean(0), color=e, lw=2.8, zorder=4)
            ax.plot(x, tt.mean(0), color=e, lw=2.4, ls='--', zorder=4)
            mast = d[f"mast_{pol}"]
            if np.all(np.isnan(mast)):
                if upto >= HORIZON:
                    ax.text(HORIZON - 4, float(tt.mean(0)[-1]) + 0.06,
                            "no mastery within horizon", ha='right',
                            fontsize=14, color=e, fontstyle='italic')
                continue
            m = np.nanmean(mast)
            if upto >= m:
                ax.axvline(m, color=e, lw=1.4, ls=':', alpha=0.9, zorder=3)
                ax.scatter([m], [ell_star], marker='*', s=420, color=c,
                           edgecolors=e, linewidths=1.5, zorder=5)
                ax.text(m + 3, ell_star - 0.13 - 0.10 * pi,
                        f"{lab.split(' ')[0]} mastery ≈ {m:.0f}",
                        fontsize=14, color=e)

        ax.set_xlim(0, HORIZON)
        ax.set_ylim(-0.75, 1.15)
        ax.set_xlabel("questions reviewed in training")
        ax.set_ylabel("parameter value  (shared scale)")
        style_ax(ax)

        # custom legend: colours = policy, linestyle = parameter
        handles = ([Line2D([], [], color=e, lw=3.4, label=lab)
                    for _, lab, _, e in POLICIES]
                   + [Line2D([], [], color=ACCENT_DARK, lw=2.6,
                             label="skill  ℓ = −log σ  (↑ better)"),
                      Line2D([], [], color=ACCENT_DARK, lw=2.4, ls='--',
                             label="bias  t  (→ 0 better)")])
        ax.legend(handles=handles, loc='upper center',
                  bbox_to_anchor=(0.5, 1.17), ncol=5, frameon=False,
                  fontsize=15)
        fig.suptitle("Skill & bias evolution per training question — policy comparison\n"
                     f"domain3, real bank, {N_SEEDS} paired seeds, mean ±1 SD, "
                     "★ mean first mastery", y=0.985, fontsize=20)
        ax.text(0.99, 0.02,
                f"question {min(upto, HORIZON)}/{HORIZON}",
                transform=ax.transAxes, ha='right', fontsize=14,
                color='#888888')
        return []

    writer = animation.FFMpegWriter(fps=fps, bitrate=6000,
                                    extra_args=['-pix_fmt', 'yuv420p'])
    anim = animation.FuncAnimation(fig, lambda i: draw(seq[i]),
                                   frames=len(seq), blit=False)
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    return out_path, len(seq)


if __name__ == "__main__":
    cache = os.path.join(FIGDIR, "data_param_evolution.npz")
    if os.path.exists(cache):
        d = dict(np.load(cache))
    else:
        print("simulating traced cohorts ...")
        d = gen_data(cache)
    out = os.path.join(FIGDIR, "param_evolution_demo.mp4")
    print("rendering ...")
    path, n = render(d, out)
    print(f"wrote {path} ({n} frames @ {FPS} fps ≈ {n / FPS:.0f}s)")
