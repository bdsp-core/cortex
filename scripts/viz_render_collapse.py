"""Render the SMC particle-collapse MP4s from saved snapshots.

Reads `results/viz/snapshots_<slug>.npz` produced by `viz_smc_collapse.py`
and writes:

    results/viz/mbw_collapse.mp4              — solo Westover, 2×3 panel grid
    results/viz/four_examinee_collapse.mp4    — 2×2 outer grid, each cell
                                                a 2×3 (t_k, l_k) mini-grid

Each frame shows one question.  Particle alpha ∝ √(N·w) so down-weighted
particles fade out (visible *between* a reweight and the subsequent rejuv).
After a rejuvenation event the weights are reset to uniform and the cloud
re-concentrates — that's the visually dramatic "collapse" moment.

Run:
    .venv/bin/python scripts/viz_render_collapse.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
OUT_DIR = os.path.join(ENGINE_REPO, "results", "viz")

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
DOMAIN_TITLES = {
    "sz":   "spike",
    "lpd":  "LPD",
    "gpd":  "GPD",
    "lrda": "LRDA",
    "grda": "GRDA",
    "iic":  "IIC",
}

# Plot limits — N(0,1) prior has 99.7% mass in ±3; truth lies in the same range.
T_LIM = (-3.0, 3.0)
L_LIM = (-2.0, 3.0)

# 60 fps for both videos (native refresh of most displays → max smoothness).
# Solo ≤ 4 s: 173 snapshots + 67 hold @ 60 fps = 4.00 s.
# Quad ≤ 10 s: 495 snapshots + 105 hold @ 60 fps = 10.00 s.
FPS_SOLO = 60
FPS_QUAD = 60

# Examinee identifier color (used only for HUD titles in the quad video so
# the four cells are still visually distinguishable). The particles themselves
# are colored by cloud spread on the plasma colormap.
EXAMINEE_PANEL_COLORS = {
    "M. Brandon Westover":  "#1f77b4",
    "Marcus Ng":            "#d62728",
    "Aaron F. Struck":      "#2ca02c",
    "Aline Herlopian":      "#9467bd",
}

# Plasma colormap mapping: cloud RMS spread → plasma value in [0, 1].
# At the N(0, Corr_l) prior the per-panel RMS spread is ~sqrt(2) (since t and
# l each have unit marginal variance under the prior); at convergence to
# δ=0.05 it's ~0.05-0.10. Map [SPREAD_MAX, SPREAD_MIN] -> [0, 1] so:
#   wide   cloud  → 0.0 plasma value  → dark purple
#   tight  cloud  → 1.0 plasma value  → bright yellow
SPREAD_MAX = 1.20    # ~prior cloud RMS spread (saturates here for darkness)
SPREAD_MIN = 0.08    # ~converged cloud RMS spread (saturates here for brightness)
PLASMA = plt.get_cmap("plasma")


def _weighted_rms_spread(t_col, l_col, w):
    """Weighted root-mean-square spread of a 2D cloud (t, l).

    Returns sqrt(var_t + var_l), our scalar 'how spread is this cloud' metric.
    """
    s = w.sum()
    if s <= 0:
        return float("nan")
    wn = w / s
    mt = float((wn * t_col).sum())
    ml = float((wn * l_col).sum())
    vt = float((wn * (t_col - mt) ** 2).sum())
    vl = float((wn * (l_col - ml) ** 2).sum())
    return float(np.sqrt(max(vt + vl, 0.0)))


def _plasma_for_spread(spread):
    """Map a cloud RMS spread to an RGBA tuple on the plasma colormap."""
    # Normalize to [0, 1] with wide = 0 (dark), tight = 1 (light).
    v = (SPREAD_MAX - float(spread)) / (SPREAD_MAX - SPREAD_MIN)
    v = float(np.clip(v, 0.0, 1.0))
    return PLASMA(v)


# ───────────── snapshot loader ─────────────

def load_snapshots(slug: str) -> dict:
    path = os.path.join(OUT_DIR, f"snapshots_{slug}.npz")
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _slugify(name: str) -> str:
    return name.lower().replace(".", "").replace(" ", "_")


# ───────────── single-examinee 2×3 grid ─────────────

def _setup_examinee_axes(fig, outer_spec, snap, color):
    """Build 2×3 sub-axes inside a GridSpec cell. Returns (axes_list, artists).

    axes_list: 6 axes, one per domain (sz, lpd, gpd / lrda, grda, iic).
    artists:   dict of mutable artists per axis (scatter, star, hw_text).
    """
    inner = GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer_spec, wspace=0.30, hspace=0.45,
    )
    axes = []
    scatters = []
    stars = []
    hw_texts = []
    K = len(DOMAINS)
    true_t = snap["true_t"]
    true_l = snap["true_l"]
    for k in range(K):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(inner[r, c])
        ax.set_xlim(*T_LIM)
        ax.set_ylim(*L_LIM)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-1, 0, 1, 2])
        ax.tick_params(labelsize=6)
        ax.set_title(DOMAIN_TITLES[DOMAINS[k]], fontsize=8, pad=2)
        if r == 1:
            ax.set_xlabel(r"$t$", fontsize=7, labelpad=1)
        if c == 0:
            ax.set_ylabel(r"$\ell$", fontsize=7, labelpad=1)
        # zero lines
        ax.axhline(0.0, color="0.85", linewidth=0.5, zorder=0)
        ax.axvline(0.0, color="0.85", linewidth=0.5, zorder=0)
        # scatter starts empty — we'll set_offsets each frame
        sc = ax.scatter([], [], s=4, c=color, alpha=0.4,
                        edgecolors="none", zorder=2)
        scatters.append(sc)
        # ground-truth star
        star = ax.scatter([true_t[k]], [true_l[k]],
                          marker="*", s=80, c="gold",
                          edgecolors="black", linewidths=0.6, zorder=4)
        stars.append(star)
        # per-domain HW readout (top-right corner of the panel)
        txt = ax.text(0.97, 0.97, "", transform=ax.transAxes,
                      ha="right", va="top", fontsize=6,
                      family="monospace",
                      bbox=dict(boxstyle="round,pad=0.15",
                                fc="white", ec="0.7", alpha=0.85))
        hw_texts.append(txt)
        axes.append(ax)
    return axes, {"scatters": scatters, "stars": stars, "hw_texts": hw_texts}


def _frame_update_for_examinee(snap, artists, i, color):
    """Update one examinee's scatters at frame i (clipped to that snap's length).

    Plasma colormap keyed to per-panel cloud RMS spread:
        wide cloud  -> dark purple
        tight cloud -> bright yellow
    Per-particle alpha tracks weight (faded between reweight and rejuv).
    """
    T = len(snap["q_idx"])
    j = min(i, T - 1)
    t = snap["t_traj"][j]
    l = snap["l_traj"][j]
    w = snap["w_traj"][j]
    hw = snap["hw_traj"][j]
    Nw = w * w.shape[0]
    alpha = np.sqrt(np.clip(Nw, 0.0, 10.0)) * 0.18
    alpha = np.clip(alpha, 0.04, 0.75)
    K = t.shape[1]
    for k in range(K):
        artists["scatters"][k].set_offsets(np.column_stack([t[:, k], l[:, k]]))
        rgba = np.array(_plasma_for_spread(_weighted_rms_spread(
            t[:, k], l[:, k], w)))
        colors = np.tile(rgba, (t.shape[0], 1))
        colors[:, 3] = alpha
        artists["scatters"][k].set_facecolor(colors)
        artists["hw_texts"][k].set_text(f"HW={hw[k]:.3f}")


def render_solo(snap, mp4_path, fps=FPS_SOLO, max_seconds=4.0):
    """One examinee → 2×3 (t,l) panels + minimal HUD.

    Particles colored via plasma colormap keyed to cloud spread. No rejuv or
    STOP badge — just title + question counter + max-HW readout.

    `max_seconds` caps the video duration. The session-frame count + hold is
    chosen so total duration ≤ max_seconds.
    """
    name = str(snap["rater_name"])
    T = len(snap["q_idx"])
    n_q = int(snap["n_q"])

    fig = plt.figure(figsize=(8.5, 5.0), dpi=160)
    outer = GridSpec(2, 1, figure=fig,
                     height_ratios=[0.10, 0.90], hspace=0.05)
    hud_ax = fig.add_subplot(outer[0])
    hud_ax.set_axis_off()
    title = hud_ax.text(0.01, 0.65, "", transform=hud_ax.transAxes,
                        fontsize=12, fontweight="bold", family="monospace")
    sub = hud_ax.text(0.01, 0.10, "", transform=hud_ax.transAxes,
                      fontsize=9, family="monospace", color="0.3")
    grid_outer = outer[1]
    # passing a placeholder color for the star/scatter setup; per-frame
    # update replaces the scatter color via _plasma_for_spread.
    axes, artists = _setup_examinee_axes(fig, grid_outer, snap, "#888888")

    def init():
        title.set_text(f"{name}  —  SMC particle-collapse")
        sub.set_text("priors: l, t ~ N(0, Corr_l fitted from 15 SPARCNET raters)")
        return ()

    def update(i):
        _frame_update_for_examinee(snap, artists, i, None)
        j = min(i, T - 1)
        q = int(snap["q_idx"][j])
        max_hw = float(np.max(snap["hw_traj"][j]))
        title.set_text(f"{name}  —  question  {q:>4d} / {n_q}")
        sub.set_text(f"max-domain AUROC halfwidth  {max_hw:.3f}   "
                     f"(stop when < {float(snap['delta_auroc']):.3f})")
        return ()

    max_frames = int(np.floor(max_seconds * fps))
    n_hold = max(max_frames - T, 6)
    total_frames = T + n_hold
    if total_frames > max_frames:
        # Should be rare (T already > max_frames); shrink hold to 0 and trim.
        total_frames = max_frames
        n_hold = max(total_frames - T, 0)
    writer = FFMpegWriter(fps=fps, bitrate=4500,
                          codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=total_frames,
                         interval=1000 / fps, blit=False)
    t0 = time.time()
    anim.save(mp4_path, writer=writer, dpi=160)
    plt.close(fig)
    print(f"   {mp4_path}  ({(time.time()-t0):.1f}s render, "
          f"{total_frames/fps:.2f}s video, "
          f"{os.path.getsize(mp4_path)/(1024*1024):.1f} MB)")


# ───────────── four-examinee 2×2 grid ─────────────

def render_quad(snaps, mp4_path, fps=FPS_QUAD, max_seconds=10.0):
    """4 examinees → 2×2 outer grid; each cell holds a 2×3 (t,l) mini-grid.

    `snaps` is an ordered list of 4 snapshot dicts.  All sessions advance
    together; sessions that have stopped freeze on their last frame while
    longer sessions keep running.  `max_seconds` caps total duration.
    """
    assert len(snaps) == 4
    T_each = [len(s["q_idx"]) for s in snaps]
    T_max = max(T_each)
    n_q_each = [int(s["n_q"]) for s in snaps]

    fig = plt.figure(figsize=(15.0, 9.0), dpi=140)
    outer = GridSpec(3, 2, figure=fig,
                     height_ratios=[0.06, 0.47, 0.47],
                     hspace=0.20, wspace=0.08)
    hud_ax = fig.add_subplot(outer[0, :])
    hud_ax.set_axis_off()
    suptitle = hud_ax.text(0.5, 0.55, "", transform=hud_ax.transAxes,
                           ha="center", va="center",
                           fontsize=14, fontweight="bold",
                           family="monospace")
    subline = hud_ax.text(0.5, 0.08, "", transform=hud_ax.transAxes,
                          ha="center", va="center",
                          fontsize=9, family="monospace", color="0.3")

    # Four examinee cells: (1,0) (1,1) (2,0) (2,1)
    cell_specs = [outer[1, 0], outer[1, 1], outer[2, 0], outer[2, 1]]
    per_examinee = []
    for snap, spec in zip(snaps, cell_specs):
        name = str(snap["rater_name"])
        color = EXAMINEE_PANEL_COLORS.get(name, "#1f77b4")
        # outer title for each examinee
        # (achieved by adding a small spec row inside the cell)
        cell = GridSpecFromSubplotSpec(
            6, 3, subplot_spec=spec, wspace=0.30, hspace=0.55,
        )
        # rows 0       : title strip       (axis-off text)
        # rows 1-2-3  : domain row 1   (3 panels)
        # rows 3-4-5  : domain row 2   (3 panels)
        title_ax = fig.add_subplot(cell[0, :])
        title_ax.set_axis_off()
        title_text = title_ax.text(0.0, 0.5, "", transform=title_ax.transAxes,
                                   ha="left", va="center", fontsize=10,
                                   fontweight="bold", color=color,
                                   family="monospace")
        # 6 panels in rows 1-2 (top) and 3-4 (bottom), 3 cols each
        # Inline build (don't re-use _setup_examinee_axes — different gridding)
        axes = []
        scatters = []
        hw_texts = []
        K = len(DOMAINS)
        true_t = snap["true_t"]
        true_l = snap["true_l"]
        for k in range(K):
            r_within, c_within = divmod(k, 3)
            row_start = 1 + 2 * r_within
            ax = fig.add_subplot(cell[row_start:row_start + 2, c_within])
            ax.set_xlim(*T_LIM)
            ax.set_ylim(*L_LIM)
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-1, 0, 1, 2])
            ax.tick_params(labelsize=5)
            ax.set_title(DOMAIN_TITLES[DOMAINS[k]], fontsize=7, pad=1)
            ax.axhline(0.0, color="0.9", linewidth=0.5, zorder=0)
            ax.axvline(0.0, color="0.9", linewidth=0.5, zorder=0)
            sc = ax.scatter([], [], s=3, c=color, alpha=0.35,
                            edgecolors="none", zorder=2)
            scatters.append(sc)
            ax.scatter([true_t[k]], [true_l[k]],
                       marker="*", s=55, c="gold",
                       edgecolors="black", linewidths=0.5, zorder=4)
            txt = ax.text(0.96, 0.96, "", transform=ax.transAxes,
                          ha="right", va="top", fontsize=5.5,
                          family="monospace",
                          bbox=dict(boxstyle="round,pad=0.1",
                                    fc="white", ec="0.7", alpha=0.85))
            hw_texts.append(txt)
            axes.append(ax)
        per_examinee.append({
            "snap": snap, "color": color, "name": name,
            "scatters": scatters, "hw_texts": hw_texts,
            "title_text": title_text,
        })

    def init():
        suptitle.set_text("SMC particle-collapse  —  four examinees, "
                          "independent K=6 sessions")
        subline.set_text("prior: l, t ~ N(0, Corr_l fitted from 15 SPARCNET "
                         "raters)   ·   gold ★ = ground-truth (t, l) per domain")
        for pe in per_examinee:
            pe["title_text"].set_text(pe["name"])
        return ()

    def update(i):
        for pe in per_examinee:
            snap = pe["snap"]
            T = len(snap["q_idx"])
            j = min(i, T - 1)
            t = snap["t_traj"][j]
            l = snap["l_traj"][j]
            w = snap["w_traj"][j]
            hw = snap["hw_traj"][j]
            Nw = w * w.shape[0]
            alpha = np.clip(np.sqrt(np.clip(Nw, 0, 10)) * 0.16, 0.04, 0.65)
            K = t.shape[1]
            for k in range(K):
                pe["scatters"][k].set_offsets(np.column_stack([t[:, k], l[:, k]]))
                rgba = np.array(_plasma_for_spread(_weighted_rms_spread(
                    t[:, k], l[:, k], w)))
                colors = np.tile(rgba, (t.shape[0], 1))
                colors[:, 3] = alpha
                pe["scatters"][k].set_facecolor(colors)
                pe["hw_texts"][k].set_text(f"HW={hw[k]:.3f}")
            q_now = int(snap["q_idx"][j])
            stopped = (j >= T - 1) and bool(snap["stopped"])
            mark = "  ✓ STOP" if stopped else ""
            pe["title_text"].set_text(
                f"{pe['name']}    q={q_now:>4d} / {int(snap['n_q'])}{mark}"
            )

        global_q = min(i, T_max - 1)
        active = sum(1 for s, T in zip(snaps, T_each) if global_q < T - 1)
        finished = 4 - active
        suptitle.set_text(
            f"SMC particle-collapse  —  global step  {global_q:>4d}   "
            f"·   running: {active}   finished: {finished}"
        )
        return ()

    max_frames = int(np.floor(max_seconds * fps))
    n_hold = max(max_frames - T_max, int(fps * 0.5))
    total_frames = T_max + n_hold
    if total_frames > max_frames:
        total_frames = max_frames
        n_hold = max(total_frames - T_max, 0)
    writer = FFMpegWriter(fps=fps, bitrate=8000,
                          codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim = FuncAnimation(fig, update, init_func=init,
                         frames=total_frames,
                         interval=1000 / fps, blit=False)
    t0 = time.time()
    anim.save(mp4_path, writer=writer, dpi=140)
    plt.close(fig)
    print(f"   {mp4_path}  ({(time.time()-t0):.1f}s render, "
          f"{total_frames/fps:.2f}s video, "
          f"{os.path.getsize(mp4_path)/(1024*1024):.1f} MB)")


# ───────────── main ─────────────

def main():
    raters = [
        "M. Brandon Westover",
        "Marcus Ng",
        "Aaron F. Struck",
        "Aline Herlopian",
    ]
    snaps = {name: load_snapshots(_slugify(name)) for name in raters}

    print("\nRendering solo Westover MP4 …")
    render_solo(snaps["M. Brandon Westover"],
                os.path.join(OUT_DIR, "mbw_collapse.mp4"))

    print("\nRendering 4-examinee MP4 …")
    render_quad([snaps[r] for r in raters],
                os.path.join(OUT_DIR, "four_examinee_collapse.mp4"))


if __name__ == "__main__":
    main()
