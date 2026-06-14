"""Rigor analysis for the expert-panel recalibration (items 1 & 3).

NON-DESTRUCTIVE. Adds the statistical justification the recalibration needs
before any adoption:
  R1  paired cluster-bootstrap ΔJ (credentialed panel vs shipped P0_cv14) per
      task, with 95% CI — is the J change significant or comparable?
  R1b panel stability table: n_expert, ℓ* bootstrap-CI width.
  R3  uniform-panel options (P0_cv14, Super8-all-tasks, Super8∪Bonobo-all-tasks)
      so the per-pattern (spike←Super8, IIIC←union) vs uniform choice is grounded.

Writes results/calibration_study/rigor_analysis.{json,txt}; changes nothing
shipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_youden_calibration_k7 import (  # noqa: E402
    DOMAINS_K7, PANEL_SIZE, _load_sdt_fits, _load_candidates,
    _cv_expert_panel, _run_calibration, _youden_one,
)
from expert_panel_recalibration_study import _credentialed  # noqa: E402

OUT = _REPO / "results" / "calibration_study"
B = 2000
SEED = 42


def _paired_delta_J(rater_ell, panelA, panelB, B=B, seed=SEED):
    """Paired cluster bootstrap of J(panelA) - J(panelB) on the SAME resampled
    rater pool (rater-level resampling). Returns (mean, lo, hi, p_le0)."""
    names = list(rater_ell)
    vals = np.array([rater_ell[n] for n in names], dtype=float)
    inA = np.array([n in panelA for n in names])
    inB = np.array([n in panelB for n in names])
    rng = np.random.default_rng(seed)
    dJ = np.empty(B)
    n = len(names)
    for b in range(B):
        idx = rng.integers(0, n, n)
        v = vals[idx]; a = inA[idx]; bb = inB[idx]
        if a.sum() == 0 or (~a).sum() == 0 or bb.sum() == 0 or (~bb).sum() == 0:
            dJ[b] = np.nan
            continue
        _, Ja = _youden_one(v[a], v[~a])
        _, Jb = _youden_one(v[bb], v[~bb])
        dJ[b] = Ja - Jb
    dJ = dJ[np.isfinite(dJ)]
    return (float(np.mean(dJ)), float(np.percentile(dJ, 2.5)),
            float(np.percentile(dJ, 97.5)), float(np.mean(dJ <= 0.0)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fits = {d: _load_sdt_fits(d) for d in DOMAINS_K7}
    candidates = _load_candidates()
    super8 = _credentialed("Super8")
    bonobo = _credentialed("Bonobo")
    union = super8 | bonobo

    # The recommended per-pattern panel: spike←Super8, IIIC←union.
    rec_panel = {d: (super8 if d == "spike" else union) for d in DOMAINS_K7}

    out = {"R1_delta_J": {}, "R1b_stability": {}, "R3_uniform": {}}
    lines = ["R1 — paired bootstrap ΔJ: recommended (credentialed) − P0_cv14",
             f"{'task':>6} {'rec_panel':>10} {'ΔJ':>8} {'95% CI':>20} {'P(ΔJ≤0)':>8}"]
    for d in DOMAINS_K7:
        cv_set, _ = _cv_expert_panel(d, fits, candidates, PANEL_SIZE)
        recset = {n for n in rec_panel[d] if n in fits[d]}
        m, lo, hi, ple = _paired_delta_J(fits[d], recset, cv_set)
        label = "Super8" if d == "spike" else "S8∪Bon"
        out["R1_delta_J"][d] = {"panel": label, "dJ_mean": m, "ci": [lo, hi],
                                "p_dJ_le_0": ple}
        lines.append(f"{d:>6} {label:>10} {m:>+8.4f} [{lo:>+7.4f},{hi:>+7.4f}] "
                     f"{ple:>8.3f}")

    # R1b stability — n_expert + ℓ* CI width for the recommended panel + P0
    lines += ["", "R1b — stability (recommended panel vs P0)",
              f"{'task':>6} {'panel':>8} {'n_exp':>5} {'ℓ*':>8} {'CIwidth':>8}"]
    for d in DOMAINS_K7:
        cv_set, _ = _cv_expert_panel(d, fits, candidates, PANEL_SIZE)
        for label, pset in (("rec", {n for n in rec_panel[d] if n in fits[d]}),
                            ("P0", cv_set)):
            exp = np.array([fits[d][n] for n in fits[d] if n in pset])
            non = np.array([fits[d][n] for n in fits[d] if n not in pset])
            r = _run_calibration(exp, non)
            w = r["ci_high"] - r["ci_low"]
            out["R1b_stability"].setdefault(d, {})[label] = {
                "n_expert": int(exp.size), "l_star": r["l_star"], "ci_width": w}
            lines.append(f"{d:>6} {label:>8} {exp.size:>5} {r['l_star']:>+8.4f} "
                         f"{w:>8.4f}")

    # R3 uniform options: P0, Super8-all, union-all
    lines += ["", "R3 — uniform-panel options (ℓ* / J) per task",
              f"{'task':>6} {'P0_cv14':>16} {'Super8_all':>16} {'union_all':>16}"]
    for d in DOMAINS_K7:
        cv_set, _ = _cv_expert_panel(d, fits, candidates, PANEL_SIZE)
        row = {}
        cells = []
        for label, pset in (("P0", cv_set), ("super8", super8), ("union", union)):
            exp = np.array([fits[d][n] for n in fits[d] if n in pset])
            non = np.array([fits[d][n] for n in fits[d] if n not in pset])
            if exp.size == 0:
                cells.append(f"{'n/a':>16}"); continue
            r = _run_calibration(exp, non)
            row[label] = {"l_star": r["l_star"], "J": r["youden_J"],
                          "n_expert": int(exp.size)}
            cells.append(f"{r['l_star']:>+6.3f}/J{r['youden_J']:.3f}/n{exp.size:>2}")
        out["R3_uniform"][d] = row
        lines.append(f"{d:>6} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}")

    (OUT / "rigor_analysis.json").write_text(json.dumps(out, indent=2))
    txt = "\n".join(lines)
    print(txt)
    (OUT / "rigor_analysis.txt").write_text(txt + "\n")
    print(f"\nwrote {OUT/'rigor_analysis.json'} and .txt")


if __name__ == "__main__":
    main()
