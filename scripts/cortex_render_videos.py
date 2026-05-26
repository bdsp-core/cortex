"""CORTEX per-test-taker MP4 visualizations.

Produces two short MP4s into a session directory from the artifacts the
SessionRecorder already writes:

  * ``collapse.mp4``  — SMC particle-cloud collapse in the (t, ℓ) plane,
    one panel per IIIC task. Style mirrors
    ``scripts/viz_render_collapse.py``'s plasma-colormap-keyed-to-spread
    visual, minus the gold ground-truth ★ (test-takers have no known
    truth — they ARE the unknown the engine is estimating).

  * ``passfail.mp4`` — per-task π_k pass-mass trajectory (the AD6
    policy's actual certification quantity, NOT the methodology's AUROC
    band). Each panel shades the PASS band at the top (π ≥ 1 − α) and
    the FAIL band at the bottom (π ≤ α); the running verdict from the
    monotonic AD6 rule locks each panel's badge as soon as a task
    resolves. At session end the certificate's per-task verdict is
    pinned on each panel.

Inputs the renderer reads from the session directory:

  * ``trajectory.npz``  → particle clouds (t_traj, l_traj, w_traj)
  * ``trials.jsonl``    → per-trial policy_diag (π, mcse, verdicts)
  * ``certificate.json``→ final per-task verdicts + stop_reason
  * ``participant.json``→ display name

CLI:

    .venv/bin/python scripts/cortex_render_videos.py <session_dir>

Auto-invoked by ``cortex_storage.SessionRecorder.finalize()`` unless
``render_videos=False`` was passed to the recorder.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# Frozen PyInstaller bundles: __file__ for PYZ-loaded modules does not
# resolve to a real filesystem path whose parent is the scripts dir;
# anchor on sys._MEIPASS so sys.path additions land in the data unpack
# root where sibling modules live.
if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
    _THIS_DIR = _REPO / "scripts"
else:
    _THIS_DIR = Path(__file__).resolve().parent
    _REPO = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cortex_policy import (  # noqa: E402
    DEFAULT_ALPHA, DEFAULT_Z, PASS, FAIL, REFER_BORDERLINE,
    REFER_UNINFORMATIVE, PENDING)

# ── visual constants — mirror viz_render_collapse for in-house consistency ──
DOMAIN_TITLES = {"sz": "Seizure", "lpd": "LPD", "gpd": "GPD",
                 "lrda": "LRDA", "grda": "GRDA", "iic": "Other"}
T_LIM = (-3.0, 3.0)
L_LIM = (-2.0, 3.0)
FPS = 30                          # 30 fps keeps render time + file size modest
HOLD_SECONDS = 1.5                # freeze on final frame for legibility
DPI = 130
SPREAD_MAX = 1.20
SPREAD_MIN = 0.08
PLASMA = plt.get_cmap("plasma")

# AD6 verdict palette — colorblind-safe, methodology-aligned where possible.
VERDICT_COLORS = {
    PASS: "#2e7d32",                # green
    FAIL: "#c62828",                # red
    REFER_BORDERLINE: "#ef8a3d",    # amber
    REFER_UNINFORMATIVE: "#888888", # gray
    PENDING: "#5a5a5a",             # dark gray (animation in-progress)
}
VERDICT_LABEL = {
    PASS: "PASS",
    FAIL: "FAIL",
    REFER_BORDERLINE: "REFER",
    REFER_UNINFORMATIVE: "REFER*",  # asterisk = info gate never opened
    PENDING: "…",
}


# ────────────────────────── session-dir loading ─────────────────────────────

def _load_session(session_dir: Path) -> dict:
    """Pull everything the renderers need out of a session directory.

    Returns a dict with the keys the two render functions consume; raises
    FileNotFoundError if any required artifact is missing."""
    sd = Path(session_dir)
    traj = np.load(sd / "trajectory.npz")
    with open(sd / "certificate.json") as fh:
        cert = json.load(fh)
    with open(sd / "participant.json") as fh:
        part = json.load(fh)
    # trials.jsonl carries the per-trial policy state — strict-ordered.
    trials = []
    with open(sd / "trials.jsonl") as fh:
        for line in fh:
            if line.strip():
                trials.append(json.loads(line))
    task_codes = [str(c) for c in traj["task_codes"]]
    name = part.get("identity", {}).get("name", "Anonymous")
    return {
        "session_dir": sd,
        "task_codes": task_codes,
        "n_questions": int(traj["n_questions"]),
        "t_traj": traj["t_traj"],          # (T, N, K)
        "l_traj": traj["l_traj"],          # (T, N, K)
        "w_traj": traj["w_traj"],          # (T, N)
        "seg_ids": traj["seg_ids"],        # (T,)
        "trials": trials,                  # T per-trial dicts
        "certificate": cert,
        "participant_name": name,
        "stop_reason": cert.get("stop_reason", ""),
        "final_verdicts": [pt.get("verdict") for pt in cert["per_task"]],
    }


# ───────────────────────── shared cloud-spread helpers ──────────────────────

def _weighted_rms_spread(t_col, l_col, w):
    """Weighted root-mean-square spread of a 2D cloud — same metric the
    methodology collapse video uses for plasma colormap keying."""
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
    v = (SPREAD_MAX - float(spread)) / (SPREAD_MAX - SPREAD_MIN)
    v = float(np.clip(v, 0.0, 1.0))
    return PLASMA(v)


# ────────────────────────── collapse renderer ───────────────────────────────

def render_collapse(session, out_path: Path, fps: int = FPS,
                    hold_seconds: float = HOLD_SECONDS) -> None:
    """One 2x3 panel grid, each panel showing the particle cloud in
    (t_k, ℓ_k) for IIIC task k, colored by per-panel cloud spread on the
    plasma colormap. No ground-truth ★. HUD shows participant name +
    question counter + max-AUROC-halfwidth from telemetry."""
    name = session["participant_name"]
    task_codes = session["task_codes"]
    t_traj = session["t_traj"]
    l_traj = session["l_traj"]
    w_traj = session["w_traj"]
    trials = session["trials"]
    T, N, K = t_traj.shape

    fig = plt.figure(figsize=(9.0, 5.4), dpi=DPI)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.10, 0.90],
                     hspace=0.05, left=0.06, right=0.97,
                     top=0.97, bottom=0.07)
    hud_ax = fig.add_subplot(outer[0]); hud_ax.set_axis_off()
    title_h = hud_ax.text(0.01, 0.62, "", transform=hud_ax.transAxes,
                          fontsize=12, fontweight="bold", family="monospace")
    sub_h = hud_ax.text(0.01, 0.05, "", transform=hud_ax.transAxes,
                        fontsize=8, family="monospace", color="0.35")

    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1],
                                    wspace=0.30, hspace=0.45)
    scatters = []
    hw_texts = []
    for k in range(K):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(inner[r, c])
        ax.set_xlim(*T_LIM); ax.set_ylim(*L_LIM)
        ax.set_xticks([-2, 0, 2]); ax.set_yticks([-1, 0, 1, 2])
        ax.tick_params(labelsize=6)
        ax.set_title(DOMAIN_TITLES.get(task_codes[k], task_codes[k]),
                     fontsize=9, pad=2)
        if r == 1:
            ax.set_xlabel(r"$t$ (bias)", fontsize=7, labelpad=1)
        if c == 0:
            ax.set_ylabel(r"$\ell$ (skill)", fontsize=7, labelpad=1)
        ax.axhline(0.0, color="0.85", linewidth=0.5, zorder=0)
        ax.axvline(0.0, color="0.85", linewidth=0.5, zorder=0)
        sc = ax.scatter([], [], s=4, alpha=0.4,
                        edgecolors="none", zorder=2)
        scatters.append(sc)
        txt = ax.text(0.97, 0.97, "", transform=ax.transAxes,
                      ha="right", va="top", fontsize=6, family="monospace",
                      bbox=dict(boxstyle="round,pad=0.15",
                                fc="white", ec="0.7", alpha=0.85))
        hw_texts.append(txt)

    def init():
        title_h.set_text(f"{name}  —  particle-cloud collapse")
        sub_h.set_text("brighter color = tighter cloud (more confident "
                       "posterior) ·  bottom-right of each panel: AUROC "
                       "half-width")
        return ()

    def update(i):
        j = min(i, T - 1)
        t = t_traj[j]; l = l_traj[j]; w = w_traj[j]
        Nw = w * N
        alpha = np.sqrt(np.clip(Nw, 0.0, 10.0)) * 0.18
        alpha = np.clip(alpha, 0.04, 0.75)
        for k in range(K):
            scatters[k].set_offsets(np.column_stack([t[:, k], l[:, k]]))
            rgba = np.array(_plasma_for_spread(
                _weighted_rms_spread(t[:, k], l[:, k], w)))
            colors = np.tile(rgba, (N, 1))
            colors[:, 3] = alpha
            scatters[k].set_facecolor(colors)
            hw_k = float(trials[j]["auroc_hw"][k])
            hw_texts[k].set_text(f"HW={hw_k:.3f}")
        title_h.set_text(f"{name}  —  question {j + 1:>3d} / {T}")
        return ()

    total = T + int(hold_seconds * fps)
    writer = FFMpegWriter(fps=fps, bitrate=4000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim = FuncAnimation(fig, update, init_func=init, frames=total,
                         interval=1000 / fps, blit=False)
    t0 = time.time()
    anim.save(str(out_path), writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"  collapse.mp4 rendered in {time.time() - t0:.1f}s "
          f"({total / fps:.1f}s video, "
          f"{os.path.getsize(out_path) / (1024 * 1024):.1f} MB)")


# ────────────────────────── passfail renderer ───────────────────────────────

def render_passfail(session, out_path: Path, fps: int = FPS,
                    hold_seconds: float = HOLD_SECONDS,
                    alpha: float = None, Z: float = None) -> None:
    """One 2x3 panel grid, one panel per IIIC task. Each panel:

      x-axis: question number (1 … n_questions)
      y-axis: π_k posterior pass-mass (0 … 1)
      Green band at the top: PASS region (π ≥ 1 − α)
      Red band at the bottom: FAIL region (π ≤ α)
      Center band: REFER region (the policy's three-way classifier)
      π_k trajectory line: colored by the AD6 running verdict per frame
      Top-right corner: locked verdict badge once the task resolves
    """
    alpha = float(DEFAULT_ALPHA if alpha is None else alpha)
    Z = float(DEFAULT_Z if Z is None else Z)
    task_codes = session["task_codes"]
    trials = session["trials"]
    n_q = session["n_questions"]
    name = session["participant_name"]
    final_verdicts = session["final_verdicts"]
    K = len(task_codes)
    T = len(trials)

    # Pre-extract π trajectories per task from the recorded policy_diag —
    # this is the authoritative AD6 quantity (NOT recomputed from particles).
    pi = np.full((T, K), np.nan)
    mcse = np.full((T, K), np.nan)
    verdicts_traj = [None] * T          # per-frame running verdict list
    for t, rec in enumerate(trials):
        pd = rec.get("policy_diag")
        if pd is None:
            continue
        pi[t] = pd["pi"]
        mcse[t] = pd["mcse"]
        verdicts_traj[t] = list(rec.get("verdicts") or [PENDING] * K)

    # Frames where policy_diag is missing (e.g. legacy non-AD6 sessions)
    # carry the previous frame's value to keep the trajectory continuous.
    last_v = [PENDING] * K
    for t in range(T):
        if verdicts_traj[t] is None:
            verdicts_traj[t] = list(last_v)
        last_v = verdicts_traj[t]

    fig = plt.figure(figsize=(11.0, 6.4), dpi=DPI)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.10, 0.90],
                     hspace=0.06, left=0.07, right=0.97,
                     top=0.97, bottom=0.08)
    hud_ax = fig.add_subplot(outer[0]); hud_ax.set_axis_off()
    title_h = hud_ax.text(0.01, 0.62, "", transform=hud_ax.transAxes,
                          fontsize=12, fontweight="bold", family="monospace")
    sub_h = hud_ax.text(0.01, 0.05, "", transform=hud_ax.transAxes,
                        fontsize=8, family="monospace", color="0.35")

    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1],
                                    wspace=0.28, hspace=0.42)
    lines = []; fills = [None] * K
    verdict_texts = []
    axes = []
    for k in range(K):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(inner[r, c])
        ax.set_xlim(1, max(n_q, 2))
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, alpha, 0.5, 1 - alpha, 1.0])
        ax.tick_params(labelsize=6)
        ax.set_title(DOMAIN_TITLES.get(task_codes[k], task_codes[k]),
                     fontsize=9, pad=2)
        if r == 1:
            ax.set_xlabel("question", fontsize=7, labelpad=1)
        if c == 0:
            ax.set_ylabel(r"$\pi_k$ (pass-mass)", fontsize=7, labelpad=1)
        # PASS / FAIL band shading
        ax.axhspan(1 - alpha, 1.0, color="#2e7d32", alpha=0.10, zorder=0)
        ax.axhspan(0.0, alpha, color="#c62828", alpha=0.10, zorder=0)
        ax.axhline(1 - alpha, color="#2e7d32", lw=0.7, ls=(0, (3, 2)),
                   zorder=1, alpha=0.6)
        ax.axhline(alpha, color="#c62828", lw=0.7, ls=(0, (3, 2)),
                   zorder=1, alpha=0.6)
        (line,) = ax.plot([], [], lw=1.8, color=VERDICT_COLORS[PENDING],
                          zorder=3)
        lines.append(line)
        vtxt = ax.text(0.96, 0.94, "", transform=ax.transAxes,
                       ha="right", va="top", fontsize=8.5, fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.20",
                                 fc="white", ec="0.6", alpha=0.85))
        verdict_texts.append(vtxt)
        axes.append(ax)

    def init():
        title_h.set_text(f"{name}  —  AD6 verdict evolution")
        sub_h.set_text(f"green band = PASS (π ≥ {1 - alpha:.2f})  ·  "
                       f"red band = FAIL (π ≤ {alpha:.2f})  ·  "
                       "trajectory color shows running AD6 verdict")
        return ()

    def update(i):
        j = min(i, T - 1)
        for k in range(K):
            x = np.arange(1, j + 2)
            y = pi[: j + 1, k]
            lines[k].set_data(x, y)
            # current running verdict
            vk = verdicts_traj[j][k]
            lines[k].set_color(VERDICT_COLORS.get(vk, VERDICT_COLORS[PENDING]))
            # CI band — mcse-wide ribbon around the trajectory
            if fills[k] is not None:
                try:
                    fills[k].remove()
                except Exception:
                    pass
            band_lo = np.clip(y - Z * mcse[: j + 1, k], 0.0, 1.0)
            band_hi = np.clip(y + Z * mcse[: j + 1, k], 0.0, 1.0)
            fills[k] = axes[k].fill_between(
                x, band_lo, band_hi,
                color=VERDICT_COLORS.get(vk, VERDICT_COLORS[PENDING]),
                alpha=0.18, lw=0, zorder=2)
            # final-verdict badge — locks at session end with the
            # certificate's authoritative label
            if j == T - 1:
                final_v = final_verdicts[k] or PENDING
                verdict_texts[k].set_text(VERDICT_LABEL.get(final_v, final_v))
                verdict_texts[k].set_color(
                    VERDICT_COLORS.get(final_v, VERDICT_COLORS[PENDING]))
            else:
                verdict_texts[k].set_text(VERDICT_LABEL.get(vk, "…"))
                verdict_texts[k].set_color(
                    VERDICT_COLORS.get(vk, VERDICT_COLORS[PENDING]))

        stop = session["stop_reason"].replace("_", " ")
        title_h.set_text(f"{name}  —  question {j + 1:>3d} / {n_q}"
                         + (f"   ·   {stop}" if j == T - 1 else ""))
        return ()

    total = T + int(hold_seconds * fps)
    writer = FFMpegWriter(fps=fps, bitrate=4500, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim = FuncAnimation(fig, update, init_func=init, frames=total,
                         interval=1000 / fps, blit=False)
    t0 = time.time()
    anim.save(str(out_path), writer=writer, dpi=DPI)
    plt.close(fig)
    print(f"  passfail.mp4 rendered in {time.time() - t0:.1f}s "
          f"({total / fps:.1f}s video, "
          f"{os.path.getsize(out_path) / (1024 * 1024):.1f} MB)")


# ────────────────────────── public entry points ─────────────────────────────

def render_all(session_dir) -> dict:
    """Render both videos into the session directory. Returns a dict
    mapping artifact name → Path. Idempotent — overwrites existing files."""
    sd = Path(session_dir)
    session = _load_session(sd)
    out = {
        "collapse": sd / "collapse.mp4",
        "passfail": sd / "passfail.mp4",
    }
    render_collapse(session, out["collapse"])
    render_passfail(session, out["passfail"])
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: cortex_render_videos.py <session_dir>", file=sys.stderr)
        sys.exit(2)
    out = render_all(sys.argv[1])
    for name, path in out.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
