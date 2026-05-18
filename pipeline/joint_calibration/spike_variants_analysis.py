"""INVESTIGATION (not the production pipeline): quantify spike-domain Youden
under 3-4 combined_spike definitions, to decide how to improve the spike
statistics. Reuses the VERBATIM fitters (fit_main_effects.fit_one,
fit_sdt_per_domain.fit_domain) + the VERBATIM youden_sigma_star + the
EXACT step_g 70/30(expert)/50/50(non-expert) split. ONLY the
combined_spike data-entry binarisation varies per scenario. Touches NO
committed pipeline file; writes only scenario artifacts + a report.

Scenarios:
  current  : sn1 {0,1}->int ; ied->1 ; {vertex,posts,wickets,bets,other}->0
             (== cert_config v13 today; sanity ~J=0.366)
  A_no_other: sn1 {0,1}->int ; ied->1 ; {vertex,posts,wickets,bets}->0 ;
              'other' EXCLUDED  (Lever A — drop ambiguous)
  B_clean_sn1: sn1 {0,1}->int ONLY ; ALL Centaur-IED excluded
               (Lever B spike — the clean baseline)
  ied_own  : Centaur-IED ONLY ; ied->1 ; {vertex,posts,wickets,bets}->0 ;
             'other' excluded  (Lever B's standalone IED-discrimination
             task — shows the data is preserved as a distinct skill)
"""
from __future__ import annotations
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "pipeline" / "_calib_work"
PREP = WORK / "data" / "prepared"
RATERS = REPO / "data" / "labels" / "raters.csv"
LABELS = REPO / "data" / "labels" / "labels.csv"
BENIGN = {"vertex wave", "posts", "wickets", "bets"}

# use_sn1: include the sn1 binary {0,1} rows.  use_centaur: include the
# Centaur-IED subtype rows.  other_neg: map ambiguous 'other'->0 (else
# EXCLUDE it).  (Previous bug: B_clean_sn1 had no use_centaur gate, so it
# silently still included Centaur-IED == A_no_other. Fixed.)
SCENARIOS = {
    "current":     dict(use_sn1=True,  use_centaur=True,  other_neg=True),
    "A_no_other":  dict(use_sn1=True,  use_centaur=True,  other_neg=False),
    "B_clean_sn1": dict(use_sn1=True,  use_centaur=False, other_neg=False),
    "ied_own":     dict(use_sn1=False, use_centaur=True,  other_neg=False),
}


@contextlib.contextmanager
def _chdir(p):
    old = os.getcwd()
    os.chdir(p)
    try:
        yield
    finally:
        os.chdir(old)


def _build(name, cfg, spike, rid2canon):
    """Write {name}.csv + {name}_meta.json with the scenario Y rule,
    using the EXACT dense-id/canonical-name conventions of
    build_calibration_inputs.build_task."""
    rows = []
    for seg, rid, val in spike:
        if val in ("0", "1"):                   # sn1 binary spike-detection
            if not cfg["use_sn1"]:
                continue
            y = int(val)
        else:                                   # Centaur-IED subtype
            if not cfg["use_centaur"]:
                continue                         # exclude ALL Centaur-IED
            v = val.strip().lower()
            if v == "ied":
                y = 1
            elif v in BENIGN:
                y = 0
            else:                               # 'other' (ambiguous)
                if cfg["other_neg"]:
                    y = 0
                else:
                    continue                    # excluded
        rows.append((int(seg), int(rid), y))
    segu = sorted({r[0] for r in rows})
    ratu = sorted({r[1] for r in rows})
    sm = {s: i for i, s in enumerate(segu)}
    rm = {r: i for i, r in enumerate(ratu)}
    df = pd.DataFrame({
        "case_id": [sm[r[0]] for r in rows],
        "rater_id": [rm[r[1]] for r in rows],
        "Y": [r[2] for r in rows]})
    df.to_csv(PREP / f"{name}.csv", index=False)
    r2n = [str(rid2canon.get(str(r), f"rater_{r}")) for r in ratu]
    json.dump({"rater_index_to_name": r2n,
               "provenance": {"scenario": name, "n_obs": len(df),
                              "n_pos": int(df.Y.sum()),
                              "pos_rate": float(df.Y.mean())}},
              open(PREP / f"{name}_meta.json", "w"))
    return len(df), int(df.Y.sum())


def _youden(name, name2exp):
    from youden_sigma_star_ref import (EXPERT_TRAIN_FRAC, SEED,  # noqa
                                       youden_sigma_star)
    f = pd.read_csv(PREP / f"{name}_sdt_fits.csv")
    f = f[f["converged"] & (f["n_trials"] >= 20)].copy()
    f["rater_name"] = f["rater_name"].astype(str).str.strip()
    f["exp"] = f["rater_name"].map(name2exp).fillna("")
    e = f[f.exp == "expert"]["sigma"].to_numpy(float)
    ne = f[f.exp != "expert"]["sigma"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    se = rng.permutation(np.arange(len(e)))
    he = round(len(se) * (1 - EXPERT_TRAIN_FRAC))
    et = e[se[: len(se) - he]]
    sn = rng.permutation(np.arange(len(ne)))
    nt = ne[sn[: len(sn) // 2]]
    s_star, J = youden_sigma_star(et, nt)
    return dict(sigma_star=float(s_star), ell_star=float(-np.log(s_star)),
                youden_J=float(J), n_expert_train=int(len(et)),
                n_nonexpert_train=int(len(nt)),
                n_converged=int(len(f)))


def main() -> None:
    print("=== spike-variants investigation (verbatim fitters) ===",
          flush=True)
    raters = pd.read_csv(RATERS, low_memory=False)
    rid2canon = dict(zip(raters.rater_id.astype(str),
                         raters.canonical_name.astype(str)))
    name2exp = dict(zip(raters.canonical_name.astype(str).str.strip(),
                        raters.expertise_level.astype(str)))
    spike = [(d["seg_id"], d["rater_id"], d["value"])
             for d in __import__("csv").DictReader(open(LABELS))
             if d["label_type"] == "spike"]
    print(f"  loaded {len(spike):,} spike rows")

    import importlib.util

    def _load(modname, path):
        sp = importlib.util.spec_from_file_location(modname, path)
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        return m

    # fit_sdt_per_domain MUST be the STAGED _calib_work/src copy so its
    # PREPARED = __file__.parent.parent/data/prepared = _calib_work/...
    # (the exact orchestrator step_c pattern). fit_main_effects is
    # cwd-relative so the reference copy is fine under _chdir(WORK).
    fsd = _load("fsd_staged", WORK / "src" / "fit_sdt_per_domain.py")
    fit_domain = fsd.fit_domain                            # verbatim
    fme = _load(
        "fme", REPO / "pipeline/reference_calibration/fit_main_effects.py")
    # youden_sigma_star_ref import (used inside _youden)
    sys.path.insert(0, str(REPO / "pipeline" / "reference_calibration"))

    out = {}
    for name, cfg in SCENARIOS.items():
        n, npos = _build(name, cfg, spike, rid2canon)
        with _chdir(WORK):                # DATA_DIR='data/prepared' rel cwd
            fme.fit_one(name)             # verbatim Rasch
            fit_domain(name)              # verbatim probit-lapse SDT
        res = _youden(name, name2exp)
        res.update(n_obs=n, n_pos=npos, pos_rate=round(npos / n, 4))
        out[name] = res
        print(f"  {name:12s} N={n:>9,} pos={res['pos_rate']:.3f}  "
              f"σ*={res['sigma_star']:.4f} ℓ*={res['ell_star']:+.4f}  "
              f"J={res['youden_J']:.4f}  (nE={res['n_expert_train']} "
              f"nNE={res['n_nonexpert_train']})")

    rpt = REPO / "calibration" / "joint" / "spike_variants_report.json"
    rpt.write_text(json.dumps({
        "scenarios": out,
        "current_v13_spike_J": out["current"]["youden_J"],
        "interpretation": {
            "B_clean_sn1": "spike = clean sn1 only (un-fold). The clean "
            "baseline; centaur_2025_ied would become its own task "
            "(see ied_own).",
            "A_no_other": "fold but drop ambiguous 'other' reads.",
            "ied_own": "Centaur-IED as a standalone IED-vs-mimic task — "
            "its data fully preserved as a distinct clinical skill.",
        }}, indent=2))
    print(f"  -> {rpt}")


if __name__ == "__main__":
    main()
