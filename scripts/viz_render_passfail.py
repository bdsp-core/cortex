"""Render the AUROC pass/fail-evolution MP4s from saved SMC snapshots.

For each rater this animates, per domain, the posterior over AUROC_k — the
Mode-A protocol's actual certification measurement — as the median plus a 95%
credible interval that narrows question by question. The certificate is the
credible-interval rule:

    PASS    if  CI_lo  > A*_k          (>=95% confident the rater clears A*)
    FAIL    if  CI_hi  < A*_k
    REFER   if  the CI straddles A*_k  (measured precisely, genuinely on the
                                        boundary)

x-axis = question number; y-axis = AUROC_k.

Writes:
    results/viz/mbw_passfail.mp4            — solo Westover, 2x3 domain grid
    results/viz/four_examinee_passfail.mp4  — 2x2 examinees, each a 2x3 grid

NOTE — A* is PROVISIONAL / ILLUSTRATIVE. The competence standard A* is set by
the A_STAR_FLAT constant (a flat floor across all domains, currently 0.80);
set A_STAR_FLAT = None to use AUROC(l*_k) per domain instead, with l* from
cert_config.yaml's DEPRECATED `mode_b_legacy.l_star_per_domain`. The real
competence standard is a credentialing decision (the open "AD6" item in
docs/LIVE_TEST_INTEGRATION_PLAN.md).

Run:
    .venv/bin/python scripts/viz_render_passfail.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS_DIR)
OUT_DIR = os.path.join(ENGINE_REPO, "results", "viz")
CERT_CONFIG = os.path.join(ENGINE_REPO, "cert_config.yaml")

# engine/auroc.py is self-contained (numpy + scipy only) — load it directly.
sys.path.insert(0, os.path.join(ENGINE_REPO, "engine"))
from auroc import auroc_from_l, auroc_quantiles_from_particles_hier  # noqa: E402

DOMAINS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
DOMAIN_TITLES = {"sz": "spike", "lpd": "LPD", "gpd": "GPD",
                 "lrda": "LRDA", "grda": "GRDA", "iic": "IIC"}

ALPHA = 0.05                       # 1 - ALPHA credible interval (95%)
Y_LIM = (0.5, 0.95)                # AUROC axis (lapse-limited ceiling ~0.92)
FPS = 60

VERDICT_COLORS = {"pass": "#2e7d32", "fail": "#c62828", "refer": "#ef8a3d"}

# Competence standard. A_STAR_FLAT, when not None, overrides the per-domain
# AUROC(l*) default with a single illustrative floor across all domains.
# The real standard is the open AD6 decision — see the module docstring.
A_STAR_FLAT = 0.80
A_STAR_NOTE = (f"A* = {A_STAR_FLAT:.2f} illustrative competence floor"
               if A_STAR_FLAT is not None else
               "A* = AUROC(deprecated Mode-B l*) — provisional")

EXAMINEE_TITLE_COLORS = {
    "M. Brandon Westover": "#1f77b4",
    "Marcus Ng":           "#d62728",
    "Aaron F. Struck":     "#2ca02c",
    "Aline Herlopian":     "#9467bd",
}


# ───────────── data ─────────────

def load_ell_star():
    """Per-domain l* from cert_config.yaml `mode_b_legacy.l_star_per_domain`
    (the DEPRECATED Mode-B binary-cert thresholds — see module docstring)."""
    with open(CERT_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    raw = cfg["mode_b_legacy"]["l_star_per_domain"]
    return {d: float(raw[d]) for d in DOMAINS}


def auroc_star(ell_star):
    """A*_k = AUROC(l*_k) — the competence standard in AUROC space."""
    return {d: float(auroc_from_l(ell_star[d])) for d in DOMAINS}


def load_snapshots(slug):
    z = np.load(os.path.join(OUT_DIR, f"snapshots_{slug}.npz"),
                allow_pickle=False)
    return {k: z[k] for k in z.files}


def _slugify(name):
    return name.lower().replace(".", "").replace(" ", "_")


def compute_auroc_traj(snap, alpha=ALPHA):
    """Per-question, per-domain AUROC posterior from the particle cloud.

    Returns (q_idx (T,), lo (T,K), med (T,K), hi (T,K)) — the alpha/2, 0.5 and
    1-alpha/2 weighted quantiles of AUROC_k at each question."""
    l_traj = snap["l_traj"]                       # (T, N, K)
    w_traj = snap["w_traj"]                       # (T, N)
    T, _, K = l_traj.shape
    qs = (alpha / 2.0, 0.5, 1.0 - alpha / 2.0)
    lo = np.zeros((T, K)); med = np.zeros((T, K)); hi = np.zeros((T, K))
    for t in range(T):
        q = auroc_quantiles_from_particles_hier(
            {"l": l_traj[t], "w": w_traj[t]}, alphas=qs)   # (K, 3)
        lo[t], med[t], hi[t] = q[:, 0], q[:, 1], q[:, 2]
    return np.asarray(snap["q_idx"]), lo, med, hi


def verdict(lo_v, hi_v, a_star):
    """Three-way credible-interval certificate for one domain at one question."""
    if lo_v > a_star:
        return "pass"
    if hi_v < a_star:
        return "fail"
    return "refer"


# ───────────── one domain panel ─────────────

def _setup_panel(ax, title, n_q, a_star, label_size=8, show_x=True,
                 show_y=True):
    """Configure one AUROC-vs-question panel. The CI fill is created per
    frame; this returns the persistent artists (median line, verdict text)."""
    ax.set_xlim(0, max(int(n_q), 1))
    ax.set_ylim(*Y_LIM)
    ax.set_title(title, fontsize=label_size + 1, pad=3)
    ax.tick_params(labelsize=label_size - 2)
    if show_x:
        ax.set_xlabel("question", fontsize=label_size)
    if show_y:
        ax.set_ylabel("AUROC", fontsize=label_size)
    # pass / fail regions about the competence standard A*
    ax.axhspan(a_star, Y_LIM[1], color="#2e7d32", alpha=0.07, zorder=0)
    ax.axhspan(Y_LIM[0], a_star, color="#c62828", alpha=0.07, zorder=0)
    ax.axhline(a_star, color="0.35", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.985, a_star + 0.012, f"A* = {a_star:.3f}", ha="right",
            va="bottom", fontsize=label_size - 2.5, color="0.35",
            transform=ax.get_yaxis_transform())
    (line,) = ax.plot([], [], lw=2.0, color="0.5", zorder=4)
    vtext = ax.text(0.035, 0.95, "", transform=ax.transAxes, va="top",
                    ha="left", fontsize=label_size, fontweight="bold")
    return {"line": line, "fill": None, "verdict": vtext}


def _update_panel(ax, art, q, lo, med, hi, k, a_star, j):
    """Draw domain k up to question index j: median line + CI fill + verdict."""
    x = q[:j + 1]
    v = verdict(lo[j, k], hi[j, k], a_star)
    col = VERDICT_COLORS[v]
    art["line"].set_data(x, med[:j + 1, k])
    art["line"].set_color(col)
    if art["fill"] is not None:
        art["fill"].remove()
    art["fill"] = ax.fill_between(x, lo[:j + 1, k], hi[:j + 1, k],
                                  color=col, alpha=0.22, lw=0, zorder=2)
    art["verdict"].set_text(v.upper())
    art["verdict"].set_color(col)


# ───────────── solo render ─────────────

def render_solo(snap, q, lo, med, hi, a_star, mp4_path, hold_s=1.2):
    name = str(snap["rater_name"])
    n_q = int(snap["n_q"])
    T = len(q)

    fig = plt.figure(figsize=(10.5, 6.2), dpi=160)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.12, 0.88],
                     hspace=0.05, left=0.07, right=0.97, top=0.97, bottom=0.08)
    hud = fig.add_subplot(outer[0]); hud.set_axis_off()
    title = hud.text(0.0, 0.62, "", transform=hud.transAxes, fontsize=12,
                     fontweight="bold", family="monospace")
    sub = hud.text(0.0, 0.04, "", transform=hud.transAxes, fontsize=7.5,
                   family="monospace", color="0.45")
    inner = GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1],
                                    wspace=0.28, hspace=0.42)
    panels = []
    for k, d in enumerate(DOMAINS):
        r, c = divmod(k, 3)
        ax = fig.add_subplot(inner[r, c])
        art = _setup_panel(ax, DOMAIN_TITLES[d], n_q, a_star[d],
                           label_size=9, show_x=(r == 1), show_y=(c == 0))
        panels.append((ax, art, k, a_star[d]))

    def init():
        title.set_text(f"{name}  —  AUROC pass/fail evolution")
        sub.set_text("median + 95% CI of AUROC per domain  ·  PASS=CI>A*  "
                     "FAIL=CI<A*  REFER=CI straddles A*  ·  " + A_STAR_NOTE)
        return ()

    def update(i):
        j = min(i, T - 1)
        for ax, art, k, a in panels:
            _update_panel(ax, art, q, lo, med, hi, k, a, j)
        title.set_text(f"{name}  —  question {int(q[j]):>4d} / {n_q}")
        return ()

    total = T + int(hold_s * FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=5000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim = FuncAnimation(fig, update, init_func=init, frames=total,
                         interval=1000 / FPS, blit=False)
    t0 = time.time()
    anim.save(mp4_path, writer=writer, dpi=160)
    plt.close(fig)
    print(f"   {mp4_path}  ({time.time()-t0:.1f}s render, "
          f"{total/FPS:.2f}s video, "
          f"{os.path.getsize(mp4_path)/(1024*1024):.1f} MB)")


# ───────────── four-examinee render ─────────────

def render_quad(rs, mp4_path, hold_s=1.4):
    """`rs` is a list of 4 dicts: {snap, q, lo, med, hi, a_star, name}."""
    assert len(rs) == 4
    Ts = [len(r["q"]) for r in rs]
    T_max = max(Ts)

    fig = plt.figure(figsize=(15.0, 9.4), dpi=140)
    outer = GridSpec(3, 2, figure=fig, height_ratios=[0.07, 0.465, 0.465],
                     hspace=0.30, wspace=0.10,
                     left=0.05, right=0.985, top=0.98, bottom=0.045)
    hud = fig.add_subplot(outer[0, :]); hud.set_axis_off()
    suptitle = hud.text(0.5, 0.6, "", transform=hud.transAxes, ha="center",
                        va="center", fontsize=13, fontweight="bold",
                        family="monospace")
    subline = hud.text(0.5, 0.04, "", transform=hud.transAxes, ha="center",
                       va="center", fontsize=8, family="monospace",
                       color="0.45")

    cell_specs = [outer[1, 0], outer[1, 1], outer[2, 0], outer[2, 1]]
    per = []
    for r, spec in zip(rs, cell_specs):
        cell = GridSpecFromSubplotSpec(7, 3, subplot_spec=spec,
                                       wspace=0.34, hspace=0.85)
        col = EXAMINEE_TITLE_COLORS.get(r["name"], "#333333")
        tax = fig.add_subplot(cell[0, :]); tax.set_axis_off()
        ttl = tax.text(0.0, 0.4, "", transform=tax.transAxes, ha="left",
                       va="center", fontsize=10, fontweight="bold",
                       color=col, family="monospace")
        panels = []
        for k, d in enumerate(DOMAINS):
            r_within, c_within = divmod(k, 3)
            row0 = 1 + 3 * r_within
            ax = fig.add_subplot(cell[row0:row0 + 3, c_within])
            art = _setup_panel(ax, DOMAIN_TITLES[d], int(r["snap"]["n_q"]),
                               r["a_star"][d], label_size=6,
                               show_x=(r_within == 1), show_y=(c_within == 0))
            panels.append((ax, art, k, r["a_star"][d]))
        per.append({"r": r, "panels": panels, "ttl": ttl})

    def init():
        suptitle.set_text("AUROC pass/fail evolution  —  four examinees")
        subline.set_text("median + 95% CI of AUROC per domain  ·  "
                         "PASS=CI>A*  FAIL=CI<A*  REFER=CI straddles A*  ·  "
                         + A_STAR_NOTE)
        for pe in per:
            pe["ttl"].set_text(pe["r"]["name"])
        return ()

    def update(i):
        for pe in per:
            r = pe["r"]
            T = len(r["q"])
            j = min(i, T - 1)
            for ax, art, k, a in pe["panels"]:
                _update_panel(ax, art, r["q"], r["lo"], r["med"], r["hi"],
                              k, a, j)
            pe["ttl"].set_text(
                f"{r['name']}    question {int(r['q'][j])} / "
                f"{int(r['snap']['n_q'])}")
        return ()

    total = T_max + int(hold_s * FPS)
    writer = FFMpegWriter(fps=FPS, bitrate=9000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p"])
    anim = FuncAnimation(fig, update, init_func=init, frames=total,
                         interval=1000 / FPS, blit=False)
    t0 = time.time()
    anim.save(mp4_path, writer=writer, dpi=140)
    plt.close(fig)
    print(f"   {mp4_path}  ({time.time()-t0:.1f}s render, "
          f"{total/FPS:.2f}s video, "
          f"{os.path.getsize(mp4_path)/(1024*1024):.1f} MB)")


# ───────────── main ─────────────

def main():
    ell_star = load_ell_star()
    if A_STAR_FLAT is not None:
        a_star = {d: float(A_STAR_FLAT) for d in DOMAINS}
        print(f"competence standard A* = {A_STAR_FLAT:.2f} "
              f"(flat illustrative floor, all domains)")
    else:
        a_star = auroc_star(ell_star)
        print("competence standard A* = AUROC(l*) per domain "
              "(provisional — deprecated Mode-B l*):")
        for d in DOMAINS:
            print(f"   {DOMAIN_TITLES[d]:6s}  l*={ell_star[d]:.4f}  "
                  f"A*={a_star[d]:.4f}")

    raters = ["M. Brandon Westover", "Marcus Ng",
              "Aaron F. Struck", "Aline Herlopian"]
    rs = {}
    for name in raters:
        snap = load_snapshots(_slugify(name))
        q, lo, med, hi = compute_auroc_traj(snap)
        rs[name] = {"snap": snap, "q": q, "lo": lo, "med": med, "hi": hi,
                    "a_star": a_star, "name": name}

    print("\nRendering solo Westover MP4 …")
    mbw = rs["M. Brandon Westover"]
    render_solo(mbw["snap"], mbw["q"], mbw["lo"], mbw["med"], mbw["hi"],
                a_star, os.path.join(OUT_DIR, "mbw_passfail.mp4"))

    print("\nRendering 4-examinee MP4 …")
    render_quad([rs[r] for r in raters],
                os.path.join(OUT_DIR, "four_examinee_passfail.mp4"))


if __name__ == "__main__":
    main()
