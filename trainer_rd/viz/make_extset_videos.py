"""EXTSET result films (M13), demo-video aesthetic (viz_style conventions).

vid1  extset_measurement_replay.mp4 — the static filter watching three REAL
      task1 raters (low / median / high accuracy): posterior ℓ̂ band
      converging as their actual reads stream in, with a correctness strip.
      (Read order = file order; task1 carries no timestamps.)
vid2  extset_reallink_training.mp4 — one hardened-config training run of the
      EXTSET-fitted RealLinkLearner: true skill vs the trainer's belief,
      mastery cut, declaration and true-mastery markers.

Run: python3 -m viz.make_extset_videos      → figures/*.mp4
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import imageio_ffmpeg
import matplotlib.pyplot as plt
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

import viz.viz_style as vs
from viz.viz_style import (ACCENT_DARK, BAD, FIGDIR, GOOD, NEUTRAL,
                           TASK_EDGE, TASK_FILL, use_style)
from training.extset_adapter import load_task1
from training.learner_sim import LearnerParams
from training.training_filter import TaskFilter

FPS = 30
MU_L_POP = -0.303            # fitted task1 population mean (F43)


def _writer(fps=FPS):
    return animation.FFMpegWriter(fps=fps, bitrate=6000,
                                  extra_args=["-pix_fmt", "yuv420p"])


# ── vid1: measurement on real raters ───────────────────────────────────────

def _replay_trace(s, y, ystar, s_sd, seed, n_max):
    params = LearnerParams(alpha_t=0.0, alpha_sigma=0.0, q_t=0.0, q_sigma=0.0,
                           rho=0.6, rule="static")
    rng = np.random.default_rng(seed)
    filt = TaskFilter(rng.standard_normal(600), rng.standard_normal(600),
                      params, ess_frac=0.5, seed=seed, smear_w=True)
    mean, sd, corr = [], [], []
    for i in range(min(len(s), n_max)):
        filt.step(float(s[i]), int(y[i]), int(ystar[i]),
                  s_sd=float(s_sd[i]), feedback=False)
        mean.append(filt.mean()[1]); sd.append(filt.sd()[1])
        corr.append(int(y[i] == ystar[i]))
    return np.array(mean), np.array(sd), np.array(corr)


def vid1_measurement(n_show=300):
    t1 = load_task1()
    per = (pd.DataFrame({"u": t1["user"],
                         "c": (t1["y"] == t1["gold"]).astype(float)})
           .groupby("u")["c"].agg(["size", "mean"]))
    per = per[per["size"] >= n_show]
    picks = [per["mean"].idxmin(),
             (per["mean"] - per["mean"].median()).abs().idxmin(),
             per["mean"].idxmax()]
    names = ["lowest-accuracy rater", "median rater", "highest-accuracy rater"]

    traces = []
    for u in picks:
        idx = np.where(t1["user"] == u)[0]
        traces.append(_replay_trace(t1["s_loo"][idx], t1["y"][idx],
                                    t1["gold"][idx], t1["s_sd"][idx],
                                    seed=int(u) % 99991, n_max=n_show)
                      + (float(per.loc[u, "mean"]),))

    use_style(16)
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.8), sharex=True)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.07,
                        hspace=0.18)
    fig.suptitle("The measurement layer watching three real raters "
                 "(task1, leave-one-out signals)", fontsize=19)
    x = np.arange(1, n_show + 1)
    arts = []
    for ax, (mean, sd, corr, acc), name, col, edge in zip(
            axes, traces, names, TASK_FILL, TASK_EDGE):
        ax.set_xlim(0, n_show)
        ax.set_ylim(-2.6, 1.9)
        ax.axhline(MU_L_POP, color=NEUTRAL, lw=1.6, ls='--', zorder=2)
        ax.text(n_show * 0.995, MU_L_POP + 0.08, "cohort mean ℓ", ha='right',
                fontsize=11, color=NEUTRAL)
        ax.set_ylabel("ℓ̂")
        vs.style_ax(ax)
        band = ax.fill_between([], [], [], color=col, alpha=0.30, zorder=3)
        line, = ax.plot([], [], color=edge, lw=2.4, zorder=4)
        strip = ax.scatter([], [], s=9, marker='|', zorder=3)
        label = ax.text(0.012, 0.86, "", transform=ax.transAxes, fontsize=13,
                        color=ACCENT_DARK)
        ax.set_title(f"{name} — final accuracy {acc:.2f}", fontsize=14,
                     loc='left', pad=4)
        arts.append((band, line, strip, label, col))
    axes[-1].set_xlabel("reads (file order)")

    hold = FPS * 2
    frames = list(range(2, n_show + 1, 2)) + [n_show] * hold
    out = f"{FIGDIR}/extset_measurement_replay.mp4"
    w = _writer()
    with w.saving(fig, out, dpi=110):
        for k in frames:
            for ax, (band, line, strip, label, col), \
                    (mean, sd, corr, acc) in zip(axes, arts, traces):
                for coll in [c for c in ax.collections if c is not strip]:
                    coll.remove()
                ax.fill_between(x[:k], mean[:k] - 1.96 * sd[:k],
                                mean[:k] + 1.96 * sd[:k], color=col,
                                alpha=0.30, zorder=3, lw=0)
                line.set_data(x[:k], mean[:k])
                strip.set_offsets(np.c_[x[:k],
                                        np.full(k, ax.get_ylim()[0] + 0.13)])
                strip.set_color([GOOD if c else BAD for c in corr[:k]])
                label.set_text(f"read {k}:  ℓ̂ = {mean[k-1]:+.2f} "
                               f"± {1.96*sd[k-1]:.2f}")
            w.grab_frame()
    plt.close(fig)
    print(f"wrote {out} ({len(frames)} frames @ {FPS} fps "
          f"≈ {len(frames)/FPS:.0f}s)")
    return out


# ── vid2: real-link training trajectory ────────────────────────────────────

def _training_trace(seed=3, budget=400):
    from studies.study_misspec import (L_STAR, NPART, SIG_STAR, ZOO,
                                       assumed_params, build_pool)
    from studies.study_extset_stress import fitted_link_params, register
    from training.benchmark_trainer import T_STAR, _select
    from training.trainer_greedy import RewardWeights
    from training.trainer_policy import ModeThresholds, TaskModePolicy

    best, lf, lm, nu, pop = fitted_link_params()
    register(best, lf, lm, nu, pop)
    pool = build_pool()
    lnr = ZOO["reallink"](seed)
    fp = assumed_params()
    rng = np.random.default_rng(1000 + seed)
    filt = TaskFilter(0.3 * rng.standard_normal(NPART),
                      0.3 * rng.standard_normal(NPART), fp, seed=seed,
                      p_static=0.3, smear_w=True)
    gate = TaskModePolicy(0, L_STAR, SIG_STAR, ModeThresholds())
    state = {"k": 0, "bal": 0}
    W = RewardWeights()
    tr = {k: [] for k in ("ell_true", "ell_hat", "ell_sd", "corr")}
    n_decl = n_true = None
    for k in range(budget):
        state["k"] = k
        idx = _select("tier2", filt, pool, state, rng, W)
        s, s_sd = float(pool.s_mean[idx]), float(pool.s_sd[idx])
        y_star = int(pool.y_star[idx])
        y = lnr.step(s + s_sd * rng.standard_normal(), 0, y_star,
                     feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        gate.note_posterior(filt)
        state["bal"] += 1 if y_star == 1 else -1
        tr["ell_true"].append(-np.log(lnr.sigma[0]))
        mt, ml = filt.mean()
        tr["ell_hat"].append(ml)
        tr["ell_sd"].append(filt.sd()[1])
        tr["corr"].append(int(y == y_star))
        if n_true is None and lnr.sigma[0] <= SIG_STAR \
                and abs(lnr.t[0]) <= T_STAR:
            n_true = k + 1
        if n_decl is None and gate.is_mastered(filt):
            n_decl = k + 1
    return {k: np.array(v) for k, v in tr.items()}, n_decl, n_true, L_STAR


def vid2_training(budget=400):
    tr, n_decl, n_true, L_STAR = _training_trace(budget=budget)
    use_style(16)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.10)
    fig.suptitle("Trainer vs the EXTSET-fitted real link "
                 "(hardened config, tier-2 placement)", fontsize=19)
    x = np.arange(1, budget + 1)
    ax.set_xlim(0, budget)
    lo = min(tr["ell_hat"].min() - 0.3, tr["ell_true"].min() - 0.3)
    ax.set_ylim(lo, max(1.4, tr["ell_true"].max() + 0.4))
    ax.axhline(L_STAR, color=ACCENT_DARK, lw=2.0, ls='--', zorder=2)
    ax.text(budget * 0.995, L_STAR + 0.04, "mastery cut ℓ*", ha='right',
            fontsize=13, color=ACCENT_DARK)
    ax.set_xlabel("training trials")
    ax.set_ylabel("log-skill  ℓ")
    vs.style_ax(ax)
    line_t, = ax.plot([], [], color=ACCENT_DARK, lw=2.6, zorder=4,
                      label="true skill (real-link learner)")
    line_h, = ax.plot([], [], color=TASK_EDGE[0], lw=2.2, zorder=4,
                      label="trainer belief ℓ̂ (95% band)")
    strip = ax.scatter([], [], s=10, marker='|', zorder=3)
    vs.top_legend(ax, 2, anchor=1.10, fontsize=14)

    hold = FPS * 2
    frames = list(range(2, budget + 1, 2)) + [budget] * hold
    out = f"{FIGDIR}/extset_reallink_training.mp4"
    w = _writer()
    with w.saving(fig, out, dpi=110):
        marked = set()
        for k in frames:
            for coll in [c for c in ax.collections if c is not strip]:
                coll.remove()
            ax.fill_between(x[:k], tr["ell_hat"][:k] - 1.96 * tr["ell_sd"][:k],
                            tr["ell_hat"][:k] + 1.96 * tr["ell_sd"][:k],
                            color=TASK_FILL[0], alpha=0.30, zorder=3, lw=0)
            line_t.set_data(x[:k], tr["ell_true"][:k])
            line_h.set_data(x[:k], tr["ell_hat"][:k])
            strip.set_offsets(np.c_[x[:k],
                                    np.full(k, ax.get_ylim()[0] + 0.07)])
            strip.set_color([GOOD if c else BAD for c in tr["corr"][:k]])
            for n, col, txt in ((n_true, GOOD, "true mastery"),
                                (n_decl, TASK_EDGE[2], "trainer declares")):
                if n is not None and k >= n and n not in marked:
                    ax.axvline(n, color=col, lw=2.0, zorder=2)
                    ax.text(n + 3, ax.get_ylim()[1] - 0.18, txt, fontsize=12,
                            color=col)
                    marked.add(n)
            w.grab_frame()
    plt.close(fig)
    print(f"wrote {out} ({len(frames)} frames @ {FPS} fps "
          f"≈ {len(frames)/FPS:.0f}s)  n_decl={n_decl} n_true={n_true}")
    return out


if __name__ == "__main__":
    vid2_training()
    vid1_measurement()
