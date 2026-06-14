"""Expert-panel recalibration STUDY (engine-improvement program, item 1 & 3).

NON-DESTRUCTIVE. Investigates whether a better-selected expert panel produces a
more rigorous / more achievable per-task ℓ* than the shipped
`ell_star_unified_v14` (which selects its 14-expert panel BY DATA-DRIVEN SKILL —
top-14 by cross-task mean ℓ from a 29-rater pool). Reuses the byte-pinned Youden
machinery from `run_youden_calibration_k7.py` and only swaps the PANEL-SELECTION
rule. Writes a comparison report to results/calibration_study/; touches NOTHING
shipped (cert_config.yaml, youden_ell_star_k7.json unchanged) — so the default
is "no change", and a panel is adopted ONLY if proven better (revert = ignore
this study's output).

Panels compared (per task; expert = panel members WITH a converged fit for that
task, non-expert = all other converged raters):
  P0_cv14       shipped baseline — CV-top-14 by cross-(other-6)-task mean ℓ.
  P1_super8     credentialed: the Super-8 epileptologists (foundational IIIC).
  P2_bonobo     credentialed: the 15 Bonobo held-out experts.
  P3_s8_bonobo  credentialed union (Super8 ∪ Bonobo).
  P4_spike_data (spike only) credentialed experts (S8∪Bonobo) WITH spike fits —
                the spike remedy (the shipped spike panel has only 9/14 with
                spike data, the other 5 picked on IIIC skill alone).

Run:  python pipeline/reference_calibration/expert_panel_recalibration_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Reuse the shipped, byte-pinned Youden machinery (selection rule aside).
from run_youden_calibration_k7 import (  # noqa: E402
    DOMAINS_K7, PANEL_SIZE, _load_sdt_fits, _load_candidates,
    _cv_expert_panel, _run_calibration,
)

RATERS_CSV = _REPO / "data" / "labels" / "raters.csv"
OUT_DIR = _REPO / "results" / "calibration_study"
SHIPPED_JSON = _HERE / "youden_ell_star_k7.json"


def _credentialed(group_substr) -> set[str]:
    """Canonical names of raters whose `groups` contains any of group_substr."""
    r = pd.read_csv(RATERS_CSV)
    g = r["groups"].fillna("")
    mask = np.zeros(len(r), dtype=bool)
    for sub in ([group_substr] if isinstance(group_substr, str) else group_substr):
        mask |= g.str.contains(sub, regex=False)
    return set(r.loc[mask, "canonical_name"].str.strip())


def _panel_eval(panel_names: set[str], rater_ell: dict[str, float]) -> dict:
    """Youden ℓ*/J/CI for `panel_names` (those present in rater_ell = expert)
    vs the rest (non-expert)."""
    present = [n for n in rater_ell if n in panel_names]
    expert = np.array([rater_ell[n] for n in present], dtype=np.float64)
    non_expert = np.array([rater_ell[n] for n in rater_ell
                           if n not in panel_names], dtype=np.float64)
    if expert.size == 0 or non_expert.size == 0:
        return {"n_expert": int(expert.size), "n_non_expert": int(non_expert.size),
                "l_star": float("nan"), "youden_J": float("nan")}
    res = _run_calibration(expert, non_expert)
    res["panel_present"] = present
    res["panel_size"] = len(present)
    return res


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fits = {d: _load_sdt_fits(d) for d in DOMAINS_K7}
    candidates = _load_candidates()

    super8 = _credentialed("Super8")
    bonobo = _credentialed("Bonobo")
    s8_bonobo = super8 | bonobo
    print(f"credentialed panels: Super8={len(super8)}  Bonobo={len(bonobo)}  "
          f"union={len(s8_bonobo)}")

    shipped = json.loads(SHIPPED_JSON.read_text())["results"] if \
        SHIPPED_JSON.exists() else {}

    report = {}
    for d in DOMAINS_K7:
        rater_ell = fits[d]
        # P0 shipped baseline (reproduce the CV-top-14 panel exactly)
        cv_set, _ = _cv_expert_panel(d, fits, candidates, PANEL_SIZE)
        panels = {
            "P0_cv14":      cv_set,
            "P1_super8":    super8,
            "P2_bonobo":    bonobo,
            "P3_s8_bonobo": s8_bonobo,
        }
        if d == "spike":
            # P4: credentialed union restricted to those WITH spike fits (the
            # spike remedy). Equivalent to P3 ∩ {has spike fit} but named for clarity.
            panels["P4_spike_data"] = {n for n in s8_bonobo if n in rater_ell}
        report[d] = {}
        for name, pset in panels.items():
            report[d][name] = _panel_eval(pset, rater_ell)
        # attach the shipped json value for cross-check
        if d in shipped:
            report[d]["_shipped_json"] = {
                "l_star": shipped[d]["l_star"], "youden_J": shipped[d]["youden_J"]}

    (OUT_DIR / "panel_comparison.json").write_text(json.dumps(report, indent=2))

    # ── human-readable table ────────────────────────────────────────────────
    lines = []
    hdr = f"{'task':>6} {'panel':>14} {'n_exp':>5} {'ℓ*':>8} {'J':>7} {'95% CI':>20} {'exp_ℓ̄':>7}"
    for d in DOMAINS_K7:
        lines.append("")
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for name, r in report[d].items():
            if name.startswith("_"):
                continue
            if not np.isfinite(r.get("youden_J", float("nan"))):
                lines.append(f"{d:>6} {name:>14} {r['n_expert']:>5}  (insufficient)")
                continue
            ci = f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
            lines.append(f"{d:>6} {name:>14} {r['n_expert']:>5} {r['l_star']:>+8.4f} "
                         f"{r['youden_J']:>7.4f} {ci:>20} {r['expert_ell_mean']:>+7.3f}")
        sj = report[d].get("_shipped_json")
        if sj:
            lines.append(f"{d:>6} {'shipped(json)':>14} {'':>5} {sj['l_star']:>+8.4f} "
                         f"{sj['youden_J']:>7.4f}")
    table = "\n".join(lines)
    print(table)
    (OUT_DIR / "panel_comparison.txt").write_text(table + "\n")
    print(f"\nwrote {OUT_DIR/'panel_comparison.json'} and .txt")


if __name__ == "__main__":
    main()
