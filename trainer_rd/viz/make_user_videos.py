"""M25 — per-tester MP4: the learning algorithm in action on the REAL
sandbox data.

For each tester profile (sandbox logs-USER-*), animate the belief state the
serving stack actually held about them, question by question, straight from
the production-shaped telemetry (every trial record carries its
`belief` snapshot — nothing is re-fit):

    skill  ℓ̂ = −log σ̂  ±1 SD   vs the v15 mastery bars ℓ*_k (per task)
    bias   t̂ ±1 SD              vs 0 and the derived zero-bias band (F60)

with session boundaries (calendar gaps annotated), the D33 lifecycle events
(provisional ★ / confirmed), consistency flags, and a serving-mode strip
(bias / skill / retention / anchor / cert-probe) underneath — the
algorithm's mode decisions are as much "in action" as the belief motion.

Aesthetic: viz_style (beautiful-figure conventions); MP4 via the bundled
imageio-ffmpeg binary (no system ffmpeg on this box).

Run:  python3 -m viz.make_user_videos [USER-A ...]
Out:  figures/learning_trajectory_USER-<X>.mp4  (one per profile found)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

from sandbox import config as C
from viz.viz_style import (ACCENT_DARK, FIGDIR, GOOD, MODE_FILL, NEUTRAL,
                           TASK_EDGE, TASK_FILL, style_ax, use_style)

SB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "sandbox")
FPS = 8
HOLD_S = 3.0
# task → (fill, edge): domain2 purple, domain3 teal (viz_style order)
TASK_COL = {1: (TASK_FILL[0], TASK_EDGE[0]), 2: (TASK_FILL[2], TASK_EDGE[2])}
MODE_KEY = {"bias": "bias", "skill": "skill", "retention": "retention",
            "anchor": "eval", "skill+cert": "recert", "bias+cert": "recert",
            "retention+cert": "recert"}


def load_user(user):
    tpath = os.path.join(SB, f"logs-{user}", "trials.jsonl")
    spath = os.path.join(SB, f"logs-{user}", "sessions.jsonl")
    if not os.path.exists(tpath):
        return None
    trials = [json.loads(l) for l in open(tpath)]
    sessions = [json.loads(l) for l in open(spath)] \
        if os.path.exists(spath) else []
    return trials, sessions


def series(trials, tasks):
    """Carry-forward per-task belief traces on the global question axis
    (the belief for a task persists between its served trials; boundary
    shifts surface at the task's next served question)."""
    n = len(trials)
    out = {t: {k: np.full(n, np.nan) for k in
               ("ell", "sd_l", "t", "sd_t", "band")} for t in tasks}
    last = {t: None for t in tasks}
    for i, r in enumerate(trials):
        last[r["task"]] = r["belief"]
        for t in tasks:
            b = last[t]
            if b is None:
                continue
            out[t]["ell"][i] = b["ell_hat"]
            out[t]["sd_l"][i] = b["sd_l"]
            out[t]["t"][i] = b["t_hat"]
            out[t]["sd_t"][i] = b["sd_t"]
            out[t]["band"][i] = b["band"]
    return out


def render_user(user, trials, sessions, out_path, fps=FPS):
    use_style(16)
    tasks = sorted(set(r["task"] for r in trials))
    n = len(trials)
    x = np.arange(1, n + 1)
    tr = series(trials, tasks)
    correct = np.array([r["correct"] for r in trials], dtype=float)

    # session boundaries + gap labels
    bounds = []
    for s_i in sorted(set(r["session"] for r in trials)):
        i0 = next(i for i, r in enumerate(trials) if r["session"] == s_i)
        gap = next((s.get("gap_s", 0.0) for s in sessions
                    if s.get("session") == s_i), 0.0)
        bounds.append((i0, s_i, gap))

    # lifecycle / monitor events at (global index, task)
    g2i = {r["global_trial"]: i for i, r in enumerate(trials)}
    events = []
    for s in sessions:
        for e in s.get("events", []):
            if e.get("global_trial") is not None \
                    and e["global_trial"] in g2i:
                events.append((g2i[e["global_trial"]], e["event"],
                               e.get("task")))

    fig = plt.figure(figsize=(19.2, 10.8))
    gs = fig.add_gridspec(3, 1, height_ratios=[5.2, 3.4, 0.55],
                          hspace=0.16, left=0.06, right=0.975,
                          top=0.80, bottom=0.075)
    ax_l = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1], sharex=ax_l)
    ax_m = fig.add_subplot(gs[2], sharex=ax_l)

    ell_lo = min(np.nanmin(tr[t]["ell"] - tr[t]["sd_l"]) for t in tasks)
    ell_hi = max(np.nanmax(tr[t]["ell"] + tr[t]["sd_l"]) for t in tasks)
    ell_hi = max(ell_hi, max(C.ELL_STAR[t] for t in tasks) + 0.15)
    t_hi = max(np.nanmax(np.abs(tr[t]["t"]) + tr[t]["sd_t"]) for t in tasks)

    hold = int(fps * HOLD_S)
    seq = list(range(2, n + 1)) + [n] * hold

    def draw(upto):
        for ax in (ax_l, ax_t, ax_m):
            ax.clear()
        xi = x[:upto]
        # ── session boundaries (all panels) ──
        for i0, s_i, gap in bounds:
            for ax in (ax_l, ax_t):
                ax.axvline(i0 + 1, color=NEUTRAL, lw=1.1, ls=(0, (5, 3)),
                           alpha=0.65, zorder=1)
            if gap > 3600:
                lab = f"session {s_i}  (+{gap / 3600:.1f} h)"
            elif gap > 60:
                lab = f"session {s_i}  (+{gap / 60:.0f} min)"
            else:
                lab = f"session {s_i}" + ("  (restart)" if s_i > 1 else "")
            ax_l.text(i0 + 1.6, ell_hi + 0.03, lab, fontsize=12,
                      color=NEUTRAL, va="bottom")
        # ── skill panel ──
        for t in tasks:
            c, e = TASK_COL.get(t, (TASK_FILL[1], TASK_EDGE[1]))
            ax_l.axhline(C.ELL_STAR[t], color=e, lw=1.8, ls="--", alpha=0.8,
                         zorder=2)
            ax_l.text(n * 0.995, C.ELL_STAR[t] + 0.015,
                      f"mastery bar ℓ*  {C.TASK_NAMES[t]}", fontsize=12,
                      color=e, va="bottom", ha="right")
            m, sd = tr[t]["ell"][:upto], tr[t]["sd_l"][:upto]
            ax_l.fill_between(xi, m - sd, m + sd, color=c, alpha=0.14, lw=0,
                              zorder=2)
            ax_l.plot(xi, m, color=e, lw=2.6, zorder=4,
                      label=f"{C.TASK_NAMES[t]}  skill ℓ̂ ±1 SD")
        # lifecycle + monitor markers
        for i, ev, t in events:
            if i >= upto:
                continue
            c, e = TASK_COL.get(t, (TASK_FILL[1], TASK_EDGE[1]))
            if ev in ("provisional", "confirmed"):
                yv = tr[t]["ell"][i]
                ax_l.scatter([i + 1], [yv], marker="*", s=460, color=c,
                             edgecolors=e, linewidths=1.4, zorder=6)
                ax_l.annotate(ev, (i + 1, yv), xytext=(6, -20),
                              textcoords="offset points", fontsize=12.5,
                              color=e, fontweight="bold")
            elif ev in ("consistency_flag", "consistency_pause"):
                ax_l.scatter([i + 1], [tr[t]["ell"][i]], marker="x", s=120,
                             color="#b06060", linewidths=2.2, zorder=6)
        ax_l.set_xlim(0, n + 1)
        ax_l.set_ylim(ell_lo - 0.1, ell_hi + 0.12)
        ax_l.set_ylabel("skill  ℓ̂ = −log σ̂   (↑ better)")
        plt.setp(ax_l.get_xticklabels(), visible=False)
        style_ax(ax_l)
        # ── bias panel ──
        ax_t.axhline(0.0, color="#bbbbbb", lw=1.2, zorder=1)
        for t in tasks:
            c, e = TASK_COL.get(t, (TASK_FILL[1], TASK_EDGE[1]))
            m, sd = tr[t]["t"][:upto], tr[t]["sd_t"][:upto]
            band = tr[t]["band"][:upto]
            ax_t.fill_between(xi, m - sd, m + sd, color=c, alpha=0.14, lw=0,
                              zorder=2)
            ax_t.plot(xi, m, color=e, lw=2.4, zorder=4)
            ax_t.plot(xi, band, color=e, lw=1.0, ls=":", alpha=0.75,
                      zorder=3)
            ax_t.plot(xi, -band, color=e, lw=1.0, ls=":", alpha=0.75,
                      zorder=3)
        ax_t.set_xlim(0, n + 1)
        ax_t.set_ylim(-t_hi - 0.1, t_hi + 0.1)
        ax_t.set_ylabel("bias  t̂   (→ 0)")
        plt.setp(ax_t.get_xticklabels(), visible=False)
        style_ax(ax_t)
        ax_t.text(n * 0.995, t_hi * 0.92, "···· zero-bias band (F60)",
                  fontsize=11, color=NEUTRAL, va="top", ha="right")
        # ── mode strip ──
        for i in range(upto):
            r = trials[i]
            col = MODE_FILL.get(MODE_KEY.get(r["mode"], "skill"), "#cccccc")
            ax_m.bar(i + 1, 1.0, width=1.0, color=col,
                     edgecolor="none", alpha=0.9)
            if r["mode"].endswith("+cert"):
                ax_m.plot([i + 1], [1.25], marker="v", ms=4,
                          color=ACCENT_DARK)
        ax_m.set_xlim(0, n + 1)
        ax_m.set_ylim(0, 1.5)
        ax_m.set_yticks([])
        ax_m.set_xlabel("question served")
        ax_m.grid(False)
        # cursor + live readout
        cur = min(upto, n) - 1
        for ax in (ax_l, ax_t):
            ax.axvline(cur + 1, color=ACCENT_DARK, lw=1.0, alpha=0.55,
                       zorder=5)
        r = trials[cur]
        b = r["belief"]
        ax_l.text(0.995, 0.03,
                  f"q {cur + 1}/{n}   serving {C.TASK_NAMES[r['task']]} "
                  f"[{r['mode']}]   σ̂ {b['sig_hat']:.2f}   t̂ "
                  f"{b['t_hat']:+.2f}   "
                  f"{'✓' if r['correct'] else '✗'}   "
                  f"acc so far {np.mean(correct[:upto]):.0%}",
                  transform=ax_l.transAxes, ha="right", fontsize=13.5,
                  color="#666666", zorder=7,
                  bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
        # legends / titles
        handles = [Line2D([], [], color=TASK_COL[t][1], lw=3.2,
                          label=f"{C.TASK_NAMES[t]}") for t in tasks]
        handles += [Line2D([], [], color=ACCENT_DARK, lw=2.0, ls="--",
                           label="mastery bar ℓ* (v15)"),
                    Line2D([], [], marker="*", color="w", markersize=17,
                           markerfacecolor=TASK_FILL[0],
                           markeredgecolor=TASK_EDGE[0],
                           label="provisional declaration (D33)")]
        handles += [plt.Rectangle((0, 0), 1, 1, fc=MODE_FILL[k], alpha=0.9,
                                  label=lab) for k, lab in
                    (("bias", "mode: bias"), ("skill", "mode: skill"),
                     ("eval", "gap anchor"), ("recert", "cert probe"))]
        ax_l.legend(handles=handles, loc="lower center",
                    bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False,
                    fontsize=13)
        fig.suptitle(
            f"The learning algorithm in action — {user}  "
            f"(sandbox, real human sessions)\n"
            "posterior belief per question: skill ℓ̂ ±1 SD toward the "
            "mastery bars; bias t̂ ±1 SD toward the zero-bias band",
            y=0.99, fontsize=19)
        return []

    writer = animation.FFMpegWriter(fps=fps, bitrate=6000,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim = animation.FuncAnimation(fig, lambda i: draw(seq[i]),
                                   frames=len(seq), blit=False)
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    return len(seq)


def main():
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    users = sys.argv[1:] or [f"USER-{c}" for c in "ABCDEFG"]
    made = []
    for u in users:
        data = load_user(u)
        if data is None:
            print(f"  {u}: no logs, skipped")
            continue
        trials, sessions = data
        out = os.path.join(FIGDIR, f"learning_trajectory_{u}.mp4")
        nf = render_user(u, trials, sessions, out)
        made.append(out)
        print(f"  wrote {out} ({len(trials)} questions, {nf} frames "
              f"@ {FPS} fps ≈ {nf / FPS:.0f}s)")
    print("\nMP4 paths:")
    for p in made:
        print(f"  {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
