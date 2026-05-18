"""Phase 3.5 part-2: assemble the 6 joint SVI posteriors into the canonical
artifacts the verbatim downstream + the engine consume.

Param mapping (joint model P=λ+(1−2λ)Φ(e^ℓ(s+t)) ↔ reference SDT
P=λ+(1−2λ)Φ((c−θ)/σ); c↔s, σ=e^{−ℓ}, e^ℓ(s+t)=(s−θ)/σ ⇒ θ=−t):
    sigma = exp(−ell_id_mean)      theta = −t_id_mean
    ell   = ell_id_mean = −log(sigma)   (exactly what run_youden uses)

Emits:
  1. data/engine_inputs/sdt_fits.csv          — REGENERATED per-rater fits
     (6 IIIC domains; schema = the carried run_youden_calibration's input).
     The Phase-3 file is preserved as sdt_fits.phase3.csv.
  2. calibration/joint/s_j_table.csv          — per (task, seg_id): s_mean,
     s_sd, n_raters  (the engine bank source; carries the s_sd the engine
     now marginalizes).
  3. data/labels/iiic_segment_signals.csv     — DENORMALIZED per-IIIC-segment
     analysis view: seg_id + per-task s_mean/s_sd + vote aggregates +
     source / tier (gold/expert/experienced/novice/crowd) breakdown.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "calibration" / "joint"
ENGINE_INPUTS = REPO / "data" / "engine_inputs"
LABELS = REPO / "data" / "labels" / "labels.csv"
RATERS = REPO / "data" / "labels" / "raters.csv"
TASKS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
# task -> canonical 6-class value (uniform {bipd,birds}->other collapse,
# same as prep.py — the Phase-3 erratum reconciliation)
TASK_POS = {"sz": "seizure", "lpd": "lpd", "gpd": "gpd",
            "lrda": "lrda", "grda": "grda", "iic": "other"}
COLLAPSE = {"bipd": "other", "birds": "other"}
IIIC_SRC = {"sparcnet50K", "pd_rda_profiler",
            "iiic_crowdsourcing:kong2025:crowd",
            "iiic_crowdsourcing:kong2025:expert",
            "centaur_2025_iiic", "centaur_iiic_expert"}
SDT_COLS = ["domain", "rater_id", "rater_name", "sigma", "theta",
            "se_sigma", "se_theta", "n_trials", "converged"]


def _load_post(task):
    d = np.load(JOINT / f"{task}_posterior.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def main() -> None:
    print("=== Phase 3.5 part-2: assemble joint outputs ===", flush=True)
    for t in TASKS:
        if not (JOINT / f"{t}_posterior.npz").exists():
            raise SystemExit(f"ABORT: missing posterior for {t}")

    # DECOUPLE (decision 2026-05-18): the joint fit serves s_j/s_sd ONLY.
    # Re-deriving per-rater Youden from the joint posterior produced
    # degenerate expert separation (sz Cohen-d=-0.38, J=0.089: Kong-crowd
    # likelihood dominance + expertise-tier interaction). cert_config v13's
    # per-rater ell* therefore comes from the erratum-fixed VERBATIM
    # two-stage path (run_unified_calibration.py), NOT here. This module
    # MUST NOT touch data/engine_inputs/sdt_fits.csv. It emits only the
    # joint s_j table + the denormalized per-IIIC-segment view. The
    # per-rater joint params are written to a SIDE-CAR for provenance.

    sdt_path = ENGINE_INPUTS / "sdt_fits.csv"  # canonical = two-stage (untouched)

    # per-rater n_trials per task (for the n_trials column)
    ntrials = {t: defaultdict(int) for t in TASKS}
    seg_meta = defaultdict(lambda: {"src": set(), "tier": defaultdict(int),
                                    "cls": defaultdict(int)})
    rid2tier = {}
    with open(RATERS) as f:
        for d in csv.DictReader(f):
            rid2tier[d["rater_id"]] = (d.get("expertise_level")
                                       or "untiered").strip() or "untiered"
    with open(LABELS) as f:
        for d in csv.DictReader(f):
            if d["label_type"] != "pattern_class":
                continue
            sd = d["source_dataset"]
            if sd not in IIIC_SRC:
                continue
            seg = d["seg_id"]
            v = COLLAPSE.get(d["value"].strip(), d["value"].strip())
            m = seg_meta[seg]
            m["src"].add(sd)
            m["tier"][rid2tier.get(d["rater_id"], "untiered")] += 1
            m["cls"][v] += 1
            for t in TASKS:
                ntrials[t][d["rater_id"]] += 1  # obs/rater for that task

    rows = []
    s_table = []                      # (task, seg_id, s_mean, s_sd, n_rater?)
    seg_sig = defaultdict(dict)       # seg_id -> {task: (s_mean,s_sd)}
    for t in TASKS:
        p = _load_post(t)
        ell = np.asarray(p["ell_id_mean"], float)
        tt = np.asarray(p["t_id_mean"], float)
        sig = np.exp(-ell)
        theta = -tt
        rid = [str(x) for x in p["rater_ids"]]
        rname = [str(x) for x in p["rater_canonical"]]
        for i in range(len(rid)):
            rows.append({
                "domain": t, "rater_id": rid[i], "rater_name": rname[i],
                "sigma": f"{sig[i]:.10g}", "theta": f"{theta[i]:.10g}",
                "se_sigma": "", "se_theta": "",
                "n_trials": ntrials[t].get(rid[i], 0),
                "converged": "True",   # SVI converged globally (see provenance)
            })
        sids = [str(x) for x in p["seg_ids"]]
        sm = np.asarray(p["s_id_mean"], float)
        ss = np.asarray(p["s_sd"], float)
        for j in range(len(sids)):
            s_table.append((t, sids[j], sm[j], ss[j]))
            seg_sig[sids[j]][t] = (sm[j], ss[j])

    # SIDE-CAR ONLY (decouple): joint per-rater params for provenance/
    # comparison — NOT consumed by Youden/engine (the documented sz
    # J=0.089 degeneracy is why). Canonical sdt_fits.csv stays the
    # erratum-fixed verbatim two-stage output.
    sidecar = JOINT / "joint_per_rater_params.csv"
    with open(sidecar, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SDT_COLS)
        w.writeheader()
        w.writerows(rows)
    assert sdt_path.exists(), "two-stage sdt_fits.csv must already exist"
    print(f"  [1] joint per-rater side-car: {len(rows)} rows "
          f"(NOT canonical — decouple; sdt_fits.csv untouched)")

    # ---- 2. s_j table (engine bank source) ----
    with open(JOINT / "s_j_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seg_id", "s_mean", "s_sd"])
        for t, sid, sm, ss in s_table:
            w.writerow([t, sid, f"{sm:.10g}", f"{ss:.10g}"])
    print(f"  [2] s_j_table.csv: {len(s_table)} (task,seg) rows")

    # ---- 3. denormalized per-IIIC-segment analysis CSV ----
    tier_keys = ["expert", "experienced", "novice", "crowd",
                 "other", "untiered"]
    out = REPO / "data" / "labels" / "iiic_segment_signals.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["seg_id", "n_sources", "sources"]
        for t in TASKS:
            hdr += [f"s_mean_{t}", f"s_sd_{t}"]
        hdr += [f"n_{k}" for k in tier_keys]
        hdr += [f"votes_{c}" for c in
                ["seizure", "lpd", "gpd", "lrda", "grda", "other"]]
        w.writerow(hdr)
        for sid in sorted(seg_sig, key=int):
            m = seg_meta.get(sid, {"src": set(), "tier": {}, "cls": {}})
            row = [sid, len(m["src"]), "|".join(sorted(m["src"]))]
            for t in TASKS:
                sm, ss = seg_sig[sid].get(t, ("", ""))
                row += [f"{sm:.6g}" if sm != "" else "",
                        f"{ss:.6g}" if ss != "" else ""]
            row += [m["tier"].get("expert", 0),
                    m["tier"].get("experienced", 0),
                    m["tier"].get("novice", 0),
                    m["tier"].get("crowd", 0) + m["tier"].get("other", 0),
                    0, m["tier"].get("untiered", 0)]
            row += [m["cls"].get(c, 0) for c in
                    ["seizure", "lpd", "gpd", "lrda", "grda", "other"]]
            w.writerow(row)
    print(f"  [3] data/labels/iiic_segment_signals.csv: "
          f"{len(seg_sig):,} segments x 6-task s_j (denormalized view)")

    (JOINT / "assemble_summary.json").write_text(json.dumps({
        "sdt_fits_rows": len(rows), "s_table_rows": len(s_table),
        "n_iiic_segments": len(seg_sig),
        "param_mapping": "sigma=exp(-ell), theta=-t, ell=-log(sigma)",
        "uniform_collapse": "{bipd,birds}->other (Phase-3 erratum fix)",
        "sdt_fits_phase3_preserved": "sdt_fits.phase3.csv",
    }, indent=2))
    print("  done.")


if __name__ == "__main__":
    main()
