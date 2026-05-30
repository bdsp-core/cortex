"""Per-session 3D explainer of the engine's item-selection decision process.

Renders ``engine_explainer.mp4`` into a session directory by replaying
the saved particle trajectory. Visualises the geometry of Bayesian
adaptive testing:

  * **Top-left (rotating 3D surface)** — posterior density manifold
    over (t_k, ℓ_k) for the current focal task. The cloud's shape +
    location IS the engine's current belief; the manifold contracts
    over trials as more data tightens the posterior.

  * **Top-right (2D overhead view)** — the same posterior cloud
    projected to (t_k, ℓ_k) with its 95% credible ellipse. The
    ellipse's principal axis is the **steepest-uncertainty direction**
    — the "Riemannian gradient" the engine descends. A trail of
    posterior-mean points shows the **geodesic** the cloud has
    traced through parameter space so far.

  * **Bottom (questions-by-domain scatter, v1.2.8)** — every question
    asked so far, one dot per question: x = question number, y = its
    signal strength (s_mean, probit-scale case difficulty), colored by
    domain (spike / sz / lpd / gpd / lrda / grda / iic). Dots reveal up
    to the trial in focus, with the most-recent question ringed so it
    ties to the animated panels above. (Replaces the earlier
    expected-posterior-variance score curve.)

The video cycles through all K tasks sequentially. Each task block
renders a representative subsample of trials so total runtime stays
in the 60-90 sec range for a typical 60-300 question session.

This module is **pure replay** — it never imports or instruments any
engine writer path. The Phase-6 reference-truth invariants (LAPSE_RATE
= 0.025, LOGIT_TO_PROBIT = 1/1.7, byte-equivalent fit_sdt_per_domain)
are untouched.

CLI::

    .venv/bin/python scripts/render_engine_explainer.py <session_dir>

Library::

    from render_engine_explainer import render_engine_explainer
    out_path = render_engine_explainer(session_dir)

Auto-invoked by ``cortex_storage.SessionRecorder.finalize()`` alongside
``cortex_render_videos`` when ``render_videos=True`` (the default).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Import cortex_render_videos FIRST — its module-import-time hook
# resolves imageio_ffmpeg and points matplotlib.rcParams at the
# bundled binary BEFORE we construct any FFMpegWriter. Cheap re-import
# if cortex_render_videos was already loaded.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import cortex_render_videos as _cv         # noqa: E402  (matplotlib rcParams)

import matplotlib                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec  # noqa: E402
from matplotlib.lines import Line2D       # noqa: E402  (legend proxies)
from mpl_toolkits.mplot3d import Axes3D    # noqa: E402, F401  (registers 3D)

# Frozen PyInstaller bundles: anchor on sys._MEIPASS for parity with the
# sibling renderers. v1.2.8 dropped the engine import this block used to set
# up for the old bottom-panel score curve; the bottom panel is now a
# session-level questions-vs-signal scatter that needs no engine code.
if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
    _HERE = _REPO / "scripts"

logger = logging.getLogger(__name__)


# ─── visual constants ─────────────────────────────────────────────────
DOMAIN_TITLES = {"spike": "Spike", "sz": "Seizure", "lpd": "LPD",
                 "gpd": "GPD", "lrda": "LRDA", "grda": "GRDA", "iic": "Other"}
# Per-domain dot colors for the questions-vs-signal panel — the Okabe-Ito
# colorblind-safe palette, all legible on the dark CORTEX background.
DOMAIN_COLORS = {"spike": "#E69F00", "sz": "#D55E00", "lpd": "#56B4E9",
                 "gpd": "#0072B2", "lrda": "#009E73", "grda": "#CC79A7",
                 "iic": "#F0E442"}
T_LIM = (-3.0, 3.0)
L_LIM = (-2.0, 3.0)
DPI = 110
# v1.1.5: slowed playback to target ~60 sec total runtime across the 6
# IIIC tasks. v1.1.3-v1.1.4 ran at 24 fps × 30 trials × 0.8 s hold for
# ~12 sec total — too quick for a viewer to absorb the manifold +
# ellipse + score-curve story. New defaults make each task block
# 10 sec at 10 fps (60 trial frames + 40 hold frames per task), and
# the camera rotation per real-time second halves (1.5°/frame × 10 fps
# = 15°/s vs 36°/s previously). Render wall-time roughly doubles.
#
# For a session of T questions:
#   - T < 60 : all trials shown 1:1; per task = T + 40 hold frames.
#              Video shorter than 60 s (proportional to T).
#   - T ≥ 60 : trials sub-sampled to 60 with step = T // 60; per task
#              = 60 + 40 = 100 frames = 10 sec; total = 60 sec.
FPS = 10
TRIALS_PER_TASK = 60        # cap per task block so total stays ~60 s
HOLD_SECONDS_PER_TASK = 4.0


# ─── pure-math helpers (no Qt, no engine instrumentation) ─────────────
def _state_from_cloud(t, l, w):
    """Reconstruct the (minimal) ``{t, l, w}`` cloud dict the frame renderer
    consumes from a saved particle trajectory frame."""
    return {"t": t, "l": l, "w": w}


def _weighted_mean_cov_2d(t_k, l_k, w):
    """Weighted posterior mean + 2×2 covariance for one task's
    (t_k, ℓ_k) marginal."""
    s = float(w.sum())
    if s <= 0.0:
        return np.zeros(2), np.eye(2) * 1e-4
    wn = w / s
    mu_t = float((wn * t_k).sum())
    mu_l = float((wn * l_k).sum())
    var_t = float((wn * (t_k - mu_t) ** 2).sum())
    var_l = float((wn * (l_k - mu_l) ** 2).sum())
    cov_tl = float((wn * (t_k - mu_t) * (l_k - mu_l)).sum())
    return np.array([mu_t, mu_l]), np.array([[var_t, cov_tl],
                                              [cov_tl, var_l]])


def _ellipse_xy(mu, cov, n_std=1.96, n_pts=128):
    """Parametric n_std-sigma credible ellipse from a 2D (mean, cov).

    n_std=1.96 ≈ 95% credible region for a Gaussian approximation.
    Adds a tiny ridge to keep the Cholesky stable for near-singular
    covariances (e.g. a fully-collapsed posterior at session end)."""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(2))
    pts = mu[:, None] + n_std * (L @ np.stack([np.cos(theta),
                                                np.sin(theta)]))
    return pts[0], pts[1]


def _weighted_kde_grid(t_k, l_k, w, n=30, t_range=None, l_range=None):
    """Weighted 2D histogram → smooth density grid for the 3D surface.

    Histogram (not true KDE) keeps render time fast — visually
    indistinguishable from a small-bandwidth KDE at the resolution
    matplotlib's surface plot uses anyway. v1.2.8: ``t_range``/``l_range``
    let the caller bin over the live cloud extent so the surface breathes
    to fit instead of clipping at fixed limits."""
    t_lo, t_hi = t_range if t_range is not None else T_LIM
    l_lo, l_hi = l_range if l_range is not None else L_LIM
    t_edges = np.linspace(t_lo, t_hi, n + 1)
    l_edges = np.linspace(l_lo, l_hi, n + 1)
    H, _, _ = np.histogram2d(t_k, l_k, bins=[t_edges, l_edges],
                             weights=w)
    H /= H.sum() + 1e-12
    T_c = 0.5 * (t_edges[:-1] + t_edges[1:])
    L_c = 0.5 * (l_edges[:-1] + l_edges[1:])
    TG, LG = np.meshgrid(T_c, L_c, indexing="ij")
    return TG, LG, H


# ─── session loading ─────────────────────────────────────────────────
def _load_session(session_dir: Path) -> dict:
    """Pull the artifacts the explainer reads. Shape contract:
      t_traj, l_traj : (T, N, K)
      w_traj         : (T, N)
      task_codes     : length-K list of strings
      seg_ids        : length-T list of int (which seg the engine
                       actually served at trial t — used for the
                       'previously asked' annotation)."""
    sd = Path(session_dir)
    traj = np.load(sd / "trajectory.npz")
    name = "Anonymous"
    part = sd / "participant.json"
    if part.is_file():
        try:
            name = json.loads(part.read_text()).get(
                "identity", {}).get("name", "Anonymous")
        except Exception:                            # noqa: BLE001
            pass
    task_codes = [str(c) for c in traj["task_codes"]]
    t_traj = np.asarray(traj["t_traj"])
    l_traj = np.asarray(traj["l_traj"])
    w_traj = np.asarray(traj["w_traj"])

    # v1.2.8: the bottom panel plots each asked question's signal strength
    # (s_mean) against its question number, colored by domain (task_code),
    # revealed up to the trial in focus. Pull it from trials.jsonl; tolerate
    # legacy/synthetic sessions that lack s_mean (those rows are skipped).
    questions = []
    trials_path = sd / "trials.jsonl"
    if trials_path.is_file():
        for line in trials_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:                            # noqa: BLE001
                continue
            s = rec.get("s_mean")
            if s is None:
                continue
            dom = rec.get("task_code")
            if dom is None:
                tk = rec.get("task_k")
                dom = (task_codes[tk] if isinstance(tk, int)
                       and 0 <= tk < len(task_codes) else "iic")
            questions.append({"idx": int(rec.get("trial_index", len(questions))),
                              "s": float(s), "domain": str(dom)})

    return {
        "session_dir": sd,
        "task_codes": task_codes,
        "t_traj": t_traj,
        "l_traj": l_traj,
        "w_traj": w_traj,
        "n_trials": int(t_traj.shape[0]),
        "n_particles": int(t_traj.shape[1]),
        "n_tasks": int(t_traj.shape[2]),
        "participant_name": name,
        "questions": questions,
    }


# ─── frame plan ──────────────────────────────────────────────────────
def _frame_plan(n_trials, n_tasks, fps=FPS,
                trials_per_task=TRIALS_PER_TASK,
                hold_seconds=HOLD_SECONDS_PER_TASK):
    """Returns (frames_per_task, total_frames, trial_idx_for_frame).

    trial_idx_for_frame(frame) → (task_k, trial_idx, advancing) where
    advancing=True means we're still walking trials, False during the
    end-of-task hold."""
    show = max(1, min(n_trials, trials_per_task))
    hold_frames = max(1, int(round(hold_seconds * fps)))
    fpt = show + hold_frames
    total = fpt * n_tasks
    step = max(1, n_trials // show)

    def _at(frame):
        task_k = frame // fpt
        within = frame % fpt
        if within < show:
            j = min(n_trials - 1, within * step)
            return task_k, j, True
        return task_k, n_trials - 1, False

    return fpt, total, _at


# ─── plotting ─────────────────────────────────────────────────────────
_BG = "#0e1015"
_GRID = "#3a3d45"
_GREEN = "#7ed391"
_AMBER = "#d4b169"
_RED = "#d8806a"
_TEXT = "#dde0e6"
_MUTED = "#9aa0ab"


def _style_axes_dark(ax, three_d=False):
    """Dark-on-CORTEX-bg axis style. 3D axes need different setters."""
    ax.set_facecolor(_BG)
    if three_d:
        # 3D axes' pane / grid styling is set via per-axis methods
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(_BG)
            axis.pane.set_edgecolor(_GRID)
            axis.pane.set_alpha(0.55)
        ax.tick_params(colors=_MUTED, labelsize=6)
        ax.xaxis.label.set_color(_MUTED)
        ax.yaxis.label.set_color(_MUTED)
        ax.zaxis.label.set_color(_MUTED)
        ax.title.set_color(_TEXT)
    else:
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_MUTED, labelsize=6)
        ax.xaxis.label.set_color(_MUTED)
        ax.yaxis.label.set_color(_MUTED)
        ax.title.set_color(_TEXT)


def _build_figure():
    """Three-panel layout (3D surface | 2D cloud | wide questions-by-domain
    scatter)."""
    # libx264 + yuv420p needs even pixel dimensions. figsize * DPI must
    # produce even (width, height). 11.0 * 110 = 1210; 7.0 * 110 = 770.
    fig = plt.figure(figsize=(11.0, 7.0), dpi=DPI)
    fig.patch.set_facecolor(_BG)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.08, 0.92],
                     hspace=0.04, left=0.06, right=0.97,
                     top=0.97, bottom=0.07)
    hud_ax = fig.add_subplot(outer[0])
    hud_ax.set_axis_off()
    title_h = hud_ax.text(0.01, 0.55, "", transform=hud_ax.transAxes,
                          fontsize=12, fontweight="bold",
                          family="monospace", color=_TEXT)
    sub_h = hud_ax.text(0.01, 0.05, "", transform=hud_ax.transAxes,
                        fontsize=9, family="monospace", color=_MUTED)
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1],
                                    height_ratios=[1.0, 0.62],
                                    width_ratios=[1.0, 1.0],
                                    wspace=0.22, hspace=0.32)
    ax_surf = fig.add_subplot(inner[0, 0], projection="3d")
    ax_cloud = fig.add_subplot(inner[0, 1])
    ax_q = fig.add_subplot(inner[1, :])
    _style_axes_dark(ax_surf, three_d=True)
    _style_axes_dark(ax_cloud)
    _style_axes_dark(ax_q)
    return fig, ax_surf, ax_cloud, ax_q, title_h, sub_h


_DOMAIN_ORDER = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]


def _draw_frame(state, *, task_k, task_code, j, n_trials, frame,
                fpt, mean_trail, questions, t_range, l_range,
                q_xlim, q_ylim, ax_surf, ax_cloud, ax_q):
    """Render one frame. Called from FuncAnimation's update closure.

    v1.3.0: ``t_range``/``l_range`` (per focal task) and ``q_xlim``/``q_ylim``
    (the question scatter) are STATIC limits precomputed once from the whole
    session, so every panel captures all of its values without moving frame
    to frame."""
    ax_surf.clear(); ax_cloud.clear(); ax_q.clear()
    _style_axes_dark(ax_surf, three_d=True)
    _style_axes_dark(ax_cloud)
    _style_axes_dark(ax_q)

    t_k = state["t"][:, task_k]
    l_k = state["l"][:, task_k]
    w = state["w"]

    t_lo, t_hi = t_range
    l_lo, l_hi = l_range

    # === 3D posterior density surface ===========================
    TG, LG, H = _weighted_kde_grid(t_k, l_k, w, n=30,
                                   t_range=(t_lo, t_hi), l_range=(l_lo, l_hi))
    ax_surf.view_init(elev=24, azim=-60 + (frame * 1.5) % 360)
    ax_surf.plot_surface(TG, LG, H.T, cmap="plasma", linewidth=0,
                         antialiased=True, edgecolor="none", alpha=0.82,
                         rstride=1, cstride=1)
    ax_surf.set_xlim(t_lo, t_hi); ax_surf.set_ylim(l_lo, l_hi)
    z_max = max(float(H.max()) * 1.2, 1e-3)
    ax_surf.set_zlim(0, z_max)
    ax_surf.set_xlabel(r"$t$  (bias)", labelpad=-4)
    ax_surf.set_ylabel(r"$\ell$  (skill)", labelpad=-4)
    ax_surf.set_zlabel("Density", labelpad=-6)
    ax_surf.set_title("Posterior density", pad=2, fontsize=9)

    # === 2D overhead: cloud + 95% credible ellipse + mean trail ===
    alphas = np.clip(w / (w.max() + 1e-12) * 0.55, 0.04, 0.55)
    ax_cloud.scatter(t_k, l_k, s=5, c=_MUTED, alpha=alphas,
                     edgecolor="none")
    mu, cov = _weighted_mean_cov_2d(t_k, l_k, w)
    try:
        ex, ey = _ellipse_xy(mu, cov)
        ax_cloud.plot(ex, ey, color=_GREEN, linewidth=1.4, alpha=0.9)
    except np.linalg.LinAlgError:
        pass                                         # singular cov; skip
    # Geodesic trail through (t, ℓ) — connect successive posterior means
    if len(mean_trail) >= 2:
        trail = np.asarray(mean_trail)
        ax_cloud.plot(trail[:, 0], trail[:, 1], color=_AMBER,
                      linewidth=0.9, alpha=0.55)
    ax_cloud.scatter([mu[0]], [mu[1]], s=42, c=_RED, marker="o",
                     edgecolor="white", linewidth=0.7, zorder=10)
    # Principal-uncertainty axis (largest eigenvector of cov) —
    # the direction the engine would descend most efficiently.
    try:
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, int(np.argmax(evals))]
        scale = 1.96 * float(np.sqrt(evals.max()))
        ax_cloud.annotate(
            "", xy=(mu[0] + scale * v[0], mu[1] + scale * v[1]),
            xytext=(mu[0] - scale * v[0], mu[1] - scale * v[1]),
            arrowprops=dict(arrowstyle="-", color=_GREEN, lw=1.0,
                            alpha=0.55))
    except np.linalg.LinAlgError:
        pass
    # v1.3.0: static limits (same frame as the 3D panel) so the cloud,
    # ellipse and mean path sit in a fixed, full-session-extent box.
    ax_cloud.set_xlim(t_lo, t_hi); ax_cloud.set_ylim(l_lo, l_hi)
    ax_cloud.set_xlabel(r"$t$  (bias)")
    ax_cloud.set_ylabel(r"$\ell$  (skill)")
    ax_cloud.set_title("Posterior cloud and mean path", pad=2, fontsize=9)
    ax_cloud.grid(True, color=_GRID, alpha=0.25)

    # === Question difficulty by domain ==========================
    # One dot per asked question: x = question number, y = its signal
    # strength (s_mean, probit case difficulty), colored by domain. Revealed
    # up to the trial in focus, the most-recent question ringed so it ties to
    # the panels above. v1.3.0: a neutral grey line connects the questions in
    # order, and the axes are static (full-session extent, precomputed).
    # v1.3.4: the question plot reveals progressively ONLY during the first
    # task block (task_k == 0); once that block has drawn the full sequence it
    # stays frozen (fully revealed) while the top panels cycle the other tasks.
    first_block = (task_k == 0)
    reveal_j = j if first_block else (n_trials - 1)
    revealed = [q for q in questions if q["idx"] <= reveal_j]
    if revealed:
        xs = np.array([q["idx"] + 1 for q in revealed], dtype=float)
        ys = np.array([q["s"] for q in revealed], dtype=float)
        cs = [DOMAIN_COLORS.get(q["domain"], _MUTED) for q in revealed]
        if len(xs) >= 2:
            ax_q.plot(xs, ys, color=_MUTED, linewidth=0.8, alpha=0.5,
                      zorder=2)
        ax_q.scatter(xs, ys, s=22, c=cs, edgecolor="none",
                     alpha=0.9, zorder=3)
        # Ring the most-recent question only while it is actively revealing
        # (first block); once frozen there is no "current" question to mark.
        if first_block:
            ax_q.scatter([xs[-1]], [ys[-1]], s=80, facecolor="none",
                         edgecolor="white", linewidth=1.1, zorder=4)
    ax_q.set_xlim(*q_xlim); ax_q.set_ylim(*q_ylim)
    ax_q.axhline(0.0, color=_GRID, lw=0.6, alpha=0.5, zorder=0)
    ax_q.set_xlabel("Question number")
    ax_q.set_ylabel("Signal strength  $s$\n(probit case difficulty)")
    ax_q.set_title("Question difficulty by domain", pad=2, fontsize=9)
    ax_q.grid(True, color=_GRID, alpha=0.25)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=DOMAIN_COLORS[d], markeredgecolor="none",
                      label=DOMAIN_TITLES.get(d, d)) for d in _DOMAIN_ORDER]
    ax_q.legend(handles=handles, loc="upper left", ncol=len(_DOMAIN_ORDER),
                fontsize=6, framealpha=0.0, handletextpad=0.2,
                columnspacing=0.8, labelcolor=_TEXT)


# ─── public API ───────────────────────────────────────────────────────
def render_engine_explainer(session_dir, out_path: Optional[Path] = None,
                            *, fps: int = FPS,
                            trials_per_task: int = TRIALS_PER_TASK,
                            hold_seconds: float = HOLD_SECONDS_PER_TASK,
                            progress_callback=None,
                            ) -> Path:
    """Render engine_explainer.mp4 in (or beside) the session directory.

    Returns the output path. Overwrites if it exists. Render time
    scales linearly with frames; a 6-task, 30-trial-per-task,
    24-fps video on a default machine runs in ~30-60 sec wall and
    produces a ~3-6 MB .mp4."""
    _cv._ensure_ffmpeg_logged()
    sd = Path(session_dir)
    out_path = Path(out_path) if out_path else (sd / "engine_explainer.mp4")
    sess = _load_session(sd)

    K = sess["n_tasks"]
    T = sess["n_trials"]
    if T == 0:
        logger.warning("engine_explainer: session has 0 trials; "
                       "skipping MP4 render.")
        return out_path                              # nothing to render

    fpt, total_frames, idx_for = _frame_plan(
        T, K, fps=fps, trials_per_task=trials_per_task,
        hold_seconds=hold_seconds)

    fig, ax_surf, ax_cloud, ax_q, title_h, sub_h = _build_figure()
    questions = sess["questions"]

    # Pre-compute the trail of posterior means per task — used by the
    # 2D cloud panel to draw the geodesic the cloud has traced through
    # parameter space so far.
    mean_trails = [[] for _ in range(K)]
    for k in range(K):
        for j in range(T):
            mu_jk, _ = _weighted_mean_cov_2d(
                sess["t_traj"][j, :, k],
                sess["l_traj"][j, :, k],
                sess["w_traj"][j])
            mean_trails[k].append(tuple(mu_jk))

    # v1.3.0: static axis limits computed ONCE from the whole session, so the
    # panels capture every value without moving frame to frame. Per-task
    # (t, ℓ) extents span all particles across all trials; the question
    # scatter spans all questions' numbers + signal strengths.
    task_tlim = [_cv._data_limits(sess["t_traj"][:, :, k].ravel())
                 for k in range(K)]
    task_llim = [_cv._data_limits(sess["l_traj"][:, :, k].ravel())
                 for k in range(K)]
    if questions:
        q_xlim = _cv._data_limits([q["idx"] + 1 for q in questions],
                                  frac=0.04, floor=2.0)
        q_ylim = _cv._data_limits([q["s"] for q in questions],
                                  frac=0.14, floor=0.6)
    else:
        q_xlim, q_ylim = (0.0, 2.0), (-1.0, 1.0)

    def update(frame):
        task_k, j, _adv = idx_for(frame)
        state = _state_from_cloud(sess["t_traj"][j],
                                  sess["l_traj"][j],
                                  sess["w_traj"][j])
        # Geodesic trail = mean trajectory up to and including trial j.
        trail = mean_trails[task_k][:j + 1]
        _draw_frame(state, task_k=task_k,
                    task_code=sess["task_codes"][task_k],
                    j=j, n_trials=T, frame=frame, fpt=fpt,
                    mean_trail=trail, questions=questions,
                    t_range=task_tlim[task_k], l_range=task_llim[task_k],
                    q_xlim=q_xlim, q_ylim=q_ylim,
                    ax_surf=ax_surf, ax_cloud=ax_cloud,
                    ax_q=ax_q)
        task_code = sess["task_codes"][task_k]
        title_h.set_text(
            f"Task {task_k + 1} of {K}: "
            f"{DOMAIN_TITLES.get(task_code, task_code)}"
            f"   ·   trial {j + 1} of {T}")
        sub_h.set_text(
            "The engine selects each question to reduce its uncertainty "
            "fastest.  Top: the posterior over this task's skill "
            r"$\ell$ and bias $t$, as a 3D density and a 2D cloud with its "
            "95% uncertainty ellipse and the path of the mean.  Bottom: "
            "every question asked so far, by difficulty and domain.")
        return ()

    writer = FFMpegWriter(fps=fps, bitrate=4000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000 / fps, blit=False)
    t0 = time.time()
    anim.save(str(out_path), writer=writer, dpi=DPI,
              savefig_kwargs={"facecolor": _BG},
              progress_callback=progress_callback)
    plt.close(fig)
    logger.info("engine_explainer.mp4 rendered in %.1fs (%.1fs video, "
                "%.1f MB)", time.time() - t0, total_frames / fps,
                os.path.getsize(out_path) / (1024 * 1024))
    return out_path


def main():
    if len(sys.argv) != 2:
        print("usage: render_engine_explainer.py <session_dir>",
              file=sys.stderr)
        sys.exit(2)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: "
                               "%(message)s")
    out = render_engine_explainer(sys.argv[1])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
