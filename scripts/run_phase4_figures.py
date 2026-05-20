"""F4.4 — Paper-grade operating-characteristic figures (Paper 1 Results).

Pure post-processing of the F4.2 surface (results/phase4_simstudy/
simstudy_rows.json): no engine recompute, runs in seconds, bootstrap
CIs computed here so the expensive run is reusable (as the F4.2 runner
docstring promised).

Censoring treatment (baked in)
------------------------------
Every session is run to a per-method/per-K question cap; a session that
never drives max_k HW_0.95(AUROC_k) below δ within that cap is
right-censored at the cap (a single, common censoring time per cell).
For that exact structure the Kaplan-Meier median collapses to a closed
form: sort the per-cell n_q with censored entries placed at +inf and
take the median; it is finite (= the usual order statistic) iff the
cell's reach rate exceeds 50%, otherwise the median lies in the
censored region and is reported as ">cap".  δ=0.05 / 0.10 are
uncensored everywhere and use the ordinary median; δ=0.025 uses the
KM median and visually flags every cell with any censoring.

Figures
-------
  fig_oc_surface.{pdf,png}        2x2 (K); median n_q vs true AUROC,
                                  three methods, 95% boot CI, δ=0.05
                                  (fully uncensored → the trustworthy
                                  headline OC).
  fig_oc_delta_censored.{pdf,png} 2x2 (K); δ=0.025 KM-median surface;
                                  partially-censored cells = open
                                  markers with an up-arrow at the cap;
                                  >50%-censored cells annotated ">cap".
  fig_hier_gain.{pdf,png}         2x2 (K); ablation speedup factors
                                  random/hier and brute/hier vs AUROC
                                  (isolates adaptive selection vs
                                  hierarchical pooling), δ=0.05.

Also writes oc_summary.json: per (method, Σ_l, K, ℓ, δ) KM/plain
median, 95% bootstrap CI, reach rate, n — the table backing F4.5.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_THIS)
# Unified-repo layout (Phase 7 sub-7.4-B): the Tier-2 OC simstudy writes
# rows to results/phase2_validation/tier2_oc_simstudy_rows.json (matches
# scripts/run_tier2_oc_simstudy.py:33). Methodology reference wrote
# results/phase4_simstudy/simstudy_rows.json. Default to the Tier-2 path
# in this repo; --rows / --out-dir CLI overrides are honored in main().
OUT_DIR = os.path.join(ENGINE_REPO, "results", "phase2_validation")
ROWS_PATH = os.path.join(OUT_DIR, "tier2_oc_simstudy_rows.json")

# Method palette = canonical project scheme (matches fig_pilot_methods).
METHOD_STYLE = {
    "random": {"color": "#7f7f7f", "label": "Random (null)"},
    "brute":  {"color": "#D55E00", "label": "Independent (adaptive)"},
    "hier":   {"color": "#0072B2", "label": "Hierarchical (Corr$_l$)"},
}
HIER_COND = "empirical"   # production Σ_l for the headline curves
METHOD_ORDER = ["random", "brute", "hier"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

_RNG = np.random.default_rng(20260516)


def _dkey(d: float) -> str:
    return "d" + str(d).replace(".", "_")


def km_median_ci(
    stops: List[Optional[int]], reached: List[bool], cap: int,
    n_boot: int = 4000,
) -> Dict[str, Any]:
    """Kaplan-Meier median + 95% bootstrap CI for single-point right
    censoring at `cap`.

    Returns dict with: median (float or None if >cap), ci_lo, ci_hi
    (None if the corresponding bootstrap quantile is censored),
    reach (fraction), n, n_cens, censored (bool: any session censored).
    """
    n = len(stops)
    # +inf sentinel for censored sessions (KM median for a common
    # censoring time): finite iff >50% of the cell reached.
    vals = np.array(
        [float(s) if (rc and s is not None) else np.inf
         for s, rc in zip(stops, reached)],
        dtype=float,
    )
    n_reach = int(np.isfinite(vals).sum())
    reach = n_reach / max(1, n)

    def _med(sample: np.ndarray) -> float:
        return float(np.median(sample))   # inf if >50% censored

    m = _med(vals)
    bs = np.array([_med(_RNG.choice(vals, n, replace=True))
                   for _ in range(n_boot)])
    # nearest-rank quantiles (inverted_cdf) return an actual bootstrap
    # order statistic, so a censored (+inf) bound stays +inf instead of
    # producing inf-inf interpolation NaNs.
    lo = float(np.quantile(bs, 0.025, method="inverted_cdf"))
    hi = float(np.quantile(bs, 0.975, method="inverted_cdf"))
    return {
        "median": None if not np.isfinite(m) else m,
        "ci_lo": None if not np.isfinite(lo) else lo,
        "ci_hi": None if not np.isfinite(hi) else hi,
        "reach": reach, "n": n, "n_cens": n - n_reach,
        "censored": (n - n_reach) > 0, "cap": int(cap),
    }


def summarize(rows: List[dict], cfg: dict) -> Dict[str, Any]:
    """Per (method, Σ_l, K, ℓ, δ) KM-median table."""
    LG, KG, DELTAS = cfg["L_GRID"], cfg["K_GRID"], cfg["DELTAS"]
    caps = cfg["MAX_Q_BY_METHOD_K"]
    out: Dict[str, Any] = {"cells": []}
    specs = [("random", "na"), ("brute", "na"),
             ("hier", "empirical"), ("hier", "independent"),
             ("hier", "cs0.7")]
    for method, cond in specs:
        for K in KG:
            cap = int(caps[method][str(K)])
            for l_val in LG:
                sub = [r for r in rows
                       if r["method"] == method and r["K"] == K
                       and r["l_val"] == l_val
                       and (cond == "na"
                            or r["sigma_l_cond"] == cond)]
                if not sub:
                    continue
                au = float(sub[0]["true_auroc"])
                for d in DELTAS:
                    dk = _dkey(d)
                    s = [r[f"stop_{dk}"] for r in sub]
                    rc = [bool(r[f"reached_{dk}"]) for r in sub]
                    est = km_median_ci(s, rc, cap)
                    out["cells"].append({
                        "method": method, "sigma_l_cond": cond,
                        "K": K, "l_val": l_val, "true_auroc": au,
                        "delta": d, **est,
                    })
    out["config"] = {k: cfg[k] for k in
                     ("L_GRID", "K_GRID", "DELTAS", "n_reps",
                      "MAX_Q_BY_METHOD_K", "true_auroc_grid")}
    return out


def _cell(summary, method, cond, K, d):
    """Sorted-by-AUROC arrays (auroc, med, lo, hi, reach, censored)."""
    cs = [c for c in summary["cells"]
          if c["method"] == method and c["sigma_l_cond"] == cond
          and c["K"] == K and c["delta"] == d]
    cs.sort(key=lambda c: c["true_auroc"])
    au = np.array([c["true_auroc"] for c in cs])
    md = np.array([np.nan if c["median"] is None else c["median"]
                   for c in cs])
    lo = np.array([np.nan if c["ci_lo"] is None else c["ci_lo"]
                   for c in cs])
    hi = np.array([np.nan if c["ci_hi"] is None else c["ci_hi"]
                   for c in cs])
    rch = np.array([c["reach"] for c in cs])
    cap = cs[0]["cap"] if cs else np.nan
    return au, md, lo, hi, rch, cap


def _subplot_grid(n):
    """Return (nrows, ncols, figsize) for an n-K subplot layout."""
    # 1→(1,1) 2→(1,2) 3→(1,3) 4→(2,2) 5→(2,3) 6→(2,3) 7+→(ceil(n/3),3)
    if n <= 1:
        return 1, 1, (5.4, 4.6)
    if n == 2:
        return 1, 2, (9.6, 4.6)
    if n == 3:
        return 1, 3, (12.0, 4.6)
    if n == 4:
        return 2, 2, (9.6, 7.4)
    if n in (5, 6):
        return 2, 3, (12.6, 7.4)
    return (n + 2) // 3, 3, (12.6, 3.5 * ((n + 2) // 3))


def fig_oc_surface(summary, KG, d=0.05, out_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    nrow, ncol, figsize = _subplot_grid(len(KG))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True,
                             squeeze=False)
    flat_axes = axes.flat
    for ax, K in zip(flat_axes, KG):
        for m in METHOD_ORDER:
            cond = HIER_COND if m == "hier" else "na"
            au, md, lo, hi, _, _ = _cell(summary, m, cond, K, d)
            st = METHOD_STYLE[m]
            ax.fill_between(au, lo, hi, color=st["color"], alpha=0.16,
                            linewidth=0, zorder=2)
            ax.plot(au, md, color=st["color"], marker="o", ms=4.5,
                    label=st["label"], zorder=3)
        ax.set_yscale("log")
        ax.set_title(f"K = {K} domains", loc="left", fontweight="bold")
        ax.grid(True, which="both", axis="y", ls=":", lw=0.5,
                color="#dddddd", zorder=0)
        ax.margins(x=0.04)
    # Hide unused panels when len(KG) < nrow*ncol
    for ax in list(axes.flat)[len(KG):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("True AUROC (data-generating skill)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Questions to $\\delta$ (median, log)")
    axes[0, 0].legend(loc="upper right", frameon=False)
    n_reps_str = (str(summary.get("config", {}).get("n_reps", "?"))
                  if isinstance(summary, dict) else "?")
    fig.suptitle(
        "Operating characteristic: questions to certify at "
        f"$\\delta$={d:g}\n"
        f"(median $\\pm$ 95% bootstrap CI over {n_reps_str} replicate "
        f"seeds; fully uncensored)",
        fontweight="bold", x=0.012, ha="left", y=1.005)
    cap = ("Lower is better. random$\\rightarrow$brute isolates "
           "adaptive (Global-EV) item selection; "
           "brute$\\rightarrow$hier isolates hierarchical cross-domain "
           "pooling (fitted Corr$_l$). Every cell reached $\\delta$="
           f"{d:g} within budget, so these medians are unbiased.")
    fig.text(0.012, -0.02, cap, fontsize=7.0, color="#444444",
             wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_oc_surface.{ext}"))
    plt.close(fig)


def fig_oc_delta_censored(summary, KG, d=0.025, out_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    nrow, ncol, figsize = _subplot_grid(len(KG))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True,
                             squeeze=False)
    any_cens = False
    for ax, K in zip(axes.flat, KG):
        for m in METHOD_ORDER:
            cond = HIER_COND if m == "hier" else "na"
            au, md, lo, hi, rch, cap = _cell(summary, m, cond, K, d)
            st = METHOD_STYLE[m]
            full = rch >= 0.999          # uncensored
            part = (~full) & np.isfinite(md)   # censored but KM-defined
            gone = ~np.isfinite(md)      # >50% censored → >cap
            # connect only the KM-defined points
            ax.plot(au[np.isfinite(md)], md[np.isfinite(md)],
                    color=st["color"], lw=1.4, zorder=3,
                    label=st["label"])
            ax.errorbar(au[full], md[full],
                        yerr=[md[full] - lo[full], hi[full] - md[full]],
                        fmt="o", ms=4.5, color=st["color"],
                        ecolor=st["color"], elinewidth=1.0, capsize=2,
                        zorder=4)
            if part.any():
                any_cens = True
                ax.scatter(au[part], md[part], s=46,
                           facecolors="white", edgecolors=st["color"],
                           linewidths=1.4, zorder=5)
            if gone.any():
                any_cens = True
                ax.scatter(au[gone], np.full(gone.sum(), cap),
                           marker="^", s=44, color=st["color"],
                           zorder=5)
        ax.axhline(cap, color="#999999", ls="--", lw=0.8, zorder=1)
        ax.text(ax.get_xlim()[1], cap, " cap", va="bottom", ha="right",
                fontsize=6.5, color="#999999")
        ax.set_yscale("log")
        ax.set_title(f"K = {K} domains", loc="left", fontweight="bold")
        ax.grid(True, which="both", axis="y", ls=":", lw=0.5,
                color="#dddddd", zorder=0)
        ax.margins(x=0.04)
    for ax in list(axes.flat)[len(KG):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("True AUROC (data-generating skill)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Questions to $\\delta$ (KM median, log)")
    handles = [Line2D([0], [0], color=METHOD_STYLE[m]["color"],
                       marker="o", label=METHOD_STYLE[m]["label"])
               for m in METHOD_ORDER]
    handles += [
        Line2D([0], [0], marker="o", color="0.4", mfc="white",
               mec="0.4", ls="none",
               label="partial censoring (KM median)"),
        Line2D([0], [0], marker="^", color="0.4", ls="none",
               label=">50% censored (median $>$ cap)"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper right",
                      frameon=False)
    fig.suptitle(
        f"Tightest precision $\\delta$={d:g}: censoring-aware OC\n"
        "(Kaplan-Meier median for single-point right censoring at the "
        "per-cell question cap)",
        fontweight="bold", x=0.012, ha="left", y=1.005)
    cap_txt = (
        "At $\\delta$=0.025 the AUROC sampling-variance peak near "
        "AUROC$\\approx$0.69 makes the budget bite: open markers are "
        "cells with partial censoring (KM median still identified); "
        "triangles at the cap are cells where $>$50% of sessions never "
        "reached $\\delta$ so the median is only bounded below. "
        "Point estimates at those cells are NOT survivor-biased here "
        "because the KM construction places censored sessions above "
        "all observed stops.")
    fig.text(0.012, -0.03, cap_txt, fontsize=7.0, color="#444444",
             wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir,
                                 f"fig_oc_delta_censored.{ext}"))
    plt.close(fig)
    return any_cens


def fig_hier_gain(summary, KG, d=0.05, out_dir=None):
    if out_dir is None:
        out_dir = OUT_DIR
    nrow, ncol, figsize = _subplot_grid(len(KG))
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True,
                             squeeze=False)
    for ax, K in zip(axes.flat, KG):
        au, h_md, *_ = _cell(summary, "hier", HIER_COND, K, d)
        _, b_md, *_ = _cell(summary, "brute", "na", K, d)
        _, r_md, *_ = _cell(summary, "random", "na", K, d)
        ax.plot(au, r_md / h_md, color=METHOD_STYLE["random"]["color"],
                marker="o", ms=4.5,
                label="random / hier (total gain)")
        ax.plot(au, b_md / h_md, color=METHOD_STYLE["hier"]["color"],
                marker="s", ms=4.0,
                label="brute / hier (pooling only)")
        ax.axhline(1.0, color="#999999", ls="--", lw=0.8)
        ax.set_title(f"K = {K} domains", loc="left", fontweight="bold")
        ax.grid(True, axis="y", ls=":", lw=0.5, color="#dddddd")
        ax.margins(x=0.04)
    for ax in list(axes.flat)[len(KG):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("True AUROC (data-generating skill)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Speedup factor (median-$n_q$ ratio)")
    axes[0, 0].legend(loc="upper left", frameon=False)
    fig.suptitle(
        f"Ablation: hierarchical-protocol speedup ($\\delta$={d:g})\n"
        "(question-budget reduction vs the random null and vs "
        "independent-adaptive)",
        fontweight="bold", x=0.012, ha="left", y=1.005)
    cap = ("Values $>$1 mean the hierarchical protocol certifies in "
           "fewer questions. random/hier is the full protocol gain; "
           "brute/hier isolates the cross-domain pooling increment "
           "(fitted Corr$_l$, r$\\approx$0.378 — a weak-pooling "
           "regime, so the pooling increment is real but modest at "
           "this empirical correlation).")
    fig.text(0.012, -0.02, cap, fontsize=7.0, color="#444444",
             wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_hier_gain.{ext}"))
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=str, default=ROWS_PATH,
                    help="Tier-2 OC simstudy rows JSON "
                         "(default: results/phase2_validation/"
                         "tier2_oc_simstudy_rows.json).")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR,
                    help="Directory to write figures into.")
    cli = ap.parse_args()
    rows_path = cli.rows
    out_dir = cli.out_dir
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(rows_path):
        sys.exit(f"missing {rows_path} — run Tier-2 OC simstudy first "
                 f"(scripts/run_tier2_oc_simstudy.py).")
    payload = json.load(open(rows_path))
    rows, cfg = payload["rows"], payload["config"]
    KG = cfg["K_GRID"]
    print(f"=== OC figures from {len(rows)} rows "
          f"({cfg['n_reps']} reps; K_GRID={KG}) ===", flush=True)

    summary = summarize(rows, cfg)
    with open(os.path.join(out_dir, "oc_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → oc_summary.json ({len(summary['cells'])} cells)",
          flush=True)

    fig_oc_surface(summary, KG, d=0.05, out_dir=out_dir)
    print("  → fig_oc_surface.{pdf,png}  (δ=0.05, uncensored)",
          flush=True)
    cens = fig_oc_delta_censored(summary, KG, d=0.025, out_dir=out_dir)
    print(f"  → fig_oc_delta_censored.{{pdf,png}}  (δ=0.025, KM; "
          f"censoring present={cens})", flush=True)
    fig_hier_gain(summary, KG, d=0.05, out_dir=out_dir)
    print("  → fig_hier_gain.{pdf,png}  (ablation speedup)",
          flush=True)


if __name__ == "__main__":
    main()
