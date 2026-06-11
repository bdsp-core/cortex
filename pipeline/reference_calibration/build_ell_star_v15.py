"""Build the v15 CREDENTIALED ell* set — REPRODUCIBLE, with provenance.

Codifies what was previously a hand-assembled artifact
(`results/calibration_study/recommended_ell_star_final.json`) so the whole v15
cut-score set regenerates deterministically. Re-fits fresh from source via the
byte-pinned Youden machinery in `run_youden_calibration_k7.py`; only the
PANEL-SELECTION rule differs from v14 (credentialed panels, not CV-top-14).

Recommended panels (PI-confirmed 2026-06-10; docs/POST_PILOT_PRODUCTION_INSTRUMENT.md §Change-2):
  spike  -> Super8                      (credentialed; J 0.649->0.693, full coverage)
  IIIC   -> Super8 ∪ Bonobo             (credentialed union)
  lpd    -> Super8 ∪ Bonobo, ROBUST TRIM: drop credentialed experts whose lpd
            ell <= LPD_TRIM_ELL (0.19) — they score in the non-expert range
            (untrimmed CI width 0.588 -> 0.23, ell* 0.186 -> 0.306).

Default policy is UNCHANGED: this only writes results/calibration_study/ell_star_v15.json
(the source for emit_cert_config_v15). Nothing live is touched.

Run:  .venv/bin/python pipeline/reference_calibration/build_ell_star_v15.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_youden_calibration_k7 import (  # noqa: E402
    DOMAINS_K7, SEED, BOOTSTRAP_N, SIGMA_FLOOR,
    MATRIX_CSV_K7, SDT_FITS_CSV_K7,
    _load_sdt_fits, _run_calibration,
)
OUT_JSON = _REPO / "results" / "calibration_study" / "ell_star_v15.json"
RATERS_CSV = _REPO / "data" / "labels" / "raters.csv"


def _credentialed(group_substr):
    """Canonical names of raters whose `groups` cell contains any of group_substr.
    Inlined (formerly imported from expert_panel_recalibration_study) so this
    generator is self-contained on TRACKED modules only."""
    r = pd.read_csv(RATERS_CSV)
    g = r["groups"].fillna("")
    mask = np.zeros(len(r), dtype=bool)
    for sub in ([group_substr] if isinstance(group_substr, str) else group_substr):
        mask |= g.str.contains(sub, regex=False)
    return set(r.loc[mask, "canonical_name"].str.strip())

# PI-confirmed recommended panels + lpd robust-trim threshold.
SPIKE_PANEL = "Super8"
IIIC_PANEL_GROUPS = ("Super8", "Bonobo")   # union
LPD_TRIM_ELL = 0.19


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _panel_calibration(domain, panel_names, rater_ell, *, lpd_trim=False):
    """Youden ell*/J/CI + expert/non-expert means for a credentialed panel.
    `present` = panel members with a converged fit for this domain (= experts);
    non-expert = all other converged raters. For lpd with lpd_trim, drop experts
    with ell <= LPD_TRIM_ELL (they move out of the expert set)."""
    present = [n for n in rater_ell if n in panel_names]
    if lpd_trim:
        present = [n for n in present if rater_ell[n] > LPD_TRIM_ELL]
    keep = set(present)
    expert = np.array([rater_ell[n] for n in present], dtype=np.float64)
    non_expert = np.array([rater_ell[n] for n in rater_ell if n not in keep],
                          dtype=np.float64)
    res = _run_calibration(expert, non_expert)
    res["panel_present"] = present
    return res


def main() -> int:
    super8 = _credentialed("Super8")
    bonobo = _credentialed("Bonobo")
    s8_bonobo = super8 | bonobo
    panel_for = {"spike": super8}
    for d in DOMAINS_K7:
        if d != "spike":
            panel_for[d] = s8_bonobo

    fits = {d: _load_sdt_fits(d) for d in DOMAINS_K7}
    results = {}
    for d in DOMAINS_K7:
        results[d] = _panel_calibration(
            d, panel_for[d], fits[d], lpd_trim=(d == "lpd"))

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    payload = {
        "method": "youden_index_credentialed_panel_k7_v15",
        "description": (
            "Credentialed-panel Youden ell* (v15). spike=Super8; IIIC=Super8∪Bonobo. "
            "lpd uses a robust trim: credentialed experts with lpd ell<=%.2f are "
            "dropped (non-expert range). Reuses the byte-pinned Youden machinery "
            "from run_youden_calibration_k7 (bootstrap_n=%d, seed=%d, sigma_floor=%.3f)."
            % (LPD_TRIM_ELL, BOOTSTRAP_N, SEED, SIGMA_FLOOR)),
        "phase": 9,
        "K": 7,
        "panels": {"spike": SPIKE_PANEL, "IIIC": "Super8∪Bonobo"},
        "lpd_trim_ell": LPD_TRIM_ELL,
        "bootstrap_n": BOOTSTRAP_N,
        "seed": SEED,
        "sigma_floor": SIGMA_FLOOR,
        "candidate_pool": "credentialed groups from data/labels/raters.csv (Super8, Bonobo)",
        "provenance": {
            "generated_utc": now,
            "raters_csv_sha256": _sha256(RATERS_CSV),
            "sdt_fits_k7_sha256": _sha256(SDT_FITS_CSV_K7),
            "cross_domain_rater_matrix_k7_sha256": _sha256(MATRIX_CSV_K7),
            "super8_n": len(super8), "bonobo_n": len(bonobo),
            "pi_confirmed": "2026-06-10 (docs/POST_PILOT_PRODUCTION_INSTRUMENT.md §Change-2)",
        },
        "domains": DOMAINS_K7,
        "results": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print(f"wrote {OUT_JSON}")
    print(f"  {'task':>6}  {'ell*':>9}  {'sigma*':>7}  {'J':>7}  "
          f"{'exp_ℓ̄':>7}  n_exp  n_non")
    for d in DOMAINS_K7:
        r = results[d]
        print(f"  {d:>6}  {r['l_star']:>+9.4f}  {r['sigma_star']:>7.4f}  "
              f"{r['youden_J']:>7.4f}  {r['expert_ell_mean']:>+7.4f}  "
              f"{r['n_expert']:>5}  {r['n_non_expert']:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
