"""Phase 9 Layer 2 — assemble K=7 joint outputs into canonical artifacts.

K=7 sibling of `assemble_outputs.py` (the K=6 IIIC predecessor). Consumes
the V_B (winning variant) per-task posteriors and emits:

  1. `calibration/joint/joint_per_rater_params_k7.csv` — SIDE-CAR ONLY;
     per-rater (sigma, theta) per task from the joint posterior. NOT
     consumed by Youden / engine (V_B's joint a_skill is documented
     degenerate; per-rater estimation comes from the byte-verbatim
     two-stage path at `data/engine_inputs/sdt_fits.csv` +
     `sdt_fits.spike.csv` — preserves D2 invariant).

  2. `calibration/joint/s_j_table_k7.csv` — per (task, seg_id) s_mean,
     s_sd from V_B joint posterior. 7 tasks instead of K=6's 6 IIIC.

  3. `data/labels/segment_signals.csv` — DENORMALIZED per-segment analysis
     view; supersedes `iiic_segment_signals.csv` at K=7. Each seg has 7-task
     s_mean/s_sd columns (NaN where the seg isn't a candidate for that task
     — spike segs have IIIC NaN, IIIC segs have spike NaN since the source
     datasets are disjoint).

  4. `calibration/joint/assemble_summary_k7.json` — provenance + row counts.

Param mapping (joint model P=λ+(1−2λ)Φ(e^ℓ(s+t)) ↔ reference SDT
P=λ+(1−2λ)Φ((c−θ)/σ); c↔s, σ=e^{−ℓ}, e^ℓ(s+t)=(s−θ)/σ ⇒ θ=−t):
    sigma = exp(−ell_id_mean)      theta = −t_id_mean
    ell   = ell_id_mean
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "calibration" / "joint"
ENGINE_INPUTS = REPO / "data" / "engine_inputs"
LABELS = REPO / "data" / "labels" / "labels.csv"
RATERS = REPO / "data" / "labels" / "raters.csv"
OUT_SEG_SIGNALS = REPO / "data" / "labels" / "segment_signals.csv"

# K=7 task table (same order as prep_k7.TASKS)
TASKS = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]

# Per-task labels.csv filter: (label_type, source_dataset_set, value_to_label_map)
# - IIIC: label_type='pattern_class'; multi-class with {bipd,birds}→other.
# - spike: label_type='spike'; binary {'0','1'}; non-binary values are
#   centaur_2025_ied multi-class (excluded by source filter, but defensively
#   handled here too).
IIIC_SOURCES = {"sparcnet50K", "pd_rda_profiler",
                "iiic_crowdsourcing:kong2025:crowd",
                "iiic_crowdsourcing:kong2025:expert",
                "centaur_2025_iiic", "centaur_iiic_expert"}
SPIKE_SOURCES = {"sn1_combined_v2"}
COLLAPSE_IIIC = {"bipd": "other", "birds": "other"}
# Maps task → (label_type, source_set, positive-class-value)
TASK_LABEL_FILTER = {
    "spike": ("spike", SPIKE_SOURCES, "1"),
    "sz": ("pattern_class", IIIC_SOURCES, "seizure"),
    "lpd": ("pattern_class", IIIC_SOURCES, "lpd"),
    "gpd": ("pattern_class", IIIC_SOURCES, "gpd"),
    "lrda": ("pattern_class", IIIC_SOURCES, "lrda"),
    "grda": ("pattern_class", IIIC_SOURCES, "grda"),
    "iic": ("pattern_class", IIIC_SOURCES, "other"),
}
SDT_COLS = ["domain", "rater_id", "rater_name", "sigma", "theta",
            "se_sigma", "se_theta", "n_trials", "converged"]


def _load_post(task: str, variant: str = "B"):
    """Load the K=7 posterior NPZ for one (task, variant) pair."""
    path = JOINT / f"{task}_k7_{variant}_posterior.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing K=7 V_{variant} posterior for {task}: {path}")
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _build_meta(variant: str = "B") -> tuple[dict, dict, dict]:
    """One pass over labels.csv to collect, per seg + per task:
      - per-rater observation counts (for n_trials column in the side-car)
      - per-seg source set + tier + vote breakdowns (for the denormalized CSV)
    """
    # rid → tier
    rid2tier: dict[str, str] = {}
    with open(RATERS) as f:
        for d in csv.DictReader(f):
            rid2tier[d["rater_id"]] = (
                (d.get("expertise_level") or "untiered").strip()
                or "untiered")

    # Per-task per-rater observation count
    ntrials: dict[str, dict[str, int]] = {t: defaultdict(int) for t in TASKS}
    # Per-seg metadata: src set + tier counts + class counts (for vote columns)
    seg_meta: dict[str, dict] = defaultdict(
        lambda: {"src": set(), "tier": defaultdict(int),
                 "cls": defaultdict(int), "spike_yes": 0, "spike_no": 0})

    with open(LABELS) as f:
        for d in csv.DictReader(f):
            sd = d["source_dataset"]
            lt = d["label_type"]
            v = d["value"].strip()
            seg = d["seg_id"]
            tier = rid2tier.get(d["rater_id"], "untiered")

            # IIIC labels — feed all 6 IIIC tasks' ntrials + IIIC vote columns
            if lt == "pattern_class" and sd in IIIC_SOURCES:
                v_canon = COLLAPSE_IIIC.get(v, v)
                m = seg_meta[seg]
                m["src"].add(sd)
                m["tier"][tier] += 1
                m["cls"][v_canon] += 1
                for t in ("sz", "lpd", "gpd", "lrda", "grda", "iic"):
                    ntrials[t][d["rater_id"]] += 1
            # Spike labels — feed spike ntrials + spike vote counts
            elif lt == "spike" and sd in SPIKE_SOURCES:
                if v not in ("0", "1"):
                    continue
                m = seg_meta[seg]
                m["src"].add(sd)
                m["tier"][tier] += 1
                if v == "1":
                    m["spike_yes"] += 1
                else:
                    m["spike_no"] += 1
                ntrials["spike"][d["rater_id"]] += 1
            # Other label types / source datasets — skip

    return ntrials, seg_meta, rid2tier


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", default="B",
                    help="winning variant to assemble (default B)")
    args = ap.parse_args(argv)
    variant = args.variant

    print(f"=== Phase 9 Layer 2: assemble K=7 outputs (variant V_{variant}) ===",
          flush=True)
    print("  pass 1/2 — streaming labels.csv for per-seg metadata + per-task ntrials",
          flush=True)
    ntrials, seg_meta, _rid2tier = _build_meta(variant)
    print(f"    {sum(len(v) for v in ntrials.values()):,} rater-task pairs; "
          f"{len(seg_meta):,} unique segments with valid labels",
          flush=True)

    print("  pass 2/2 — reading K=7 V_B posteriors per task", flush=True)
    rows: list[dict] = []
    s_table: list[tuple] = []                  # (task, seg_id, s_mean, s_sd)
    seg_sig: dict[str, dict] = defaultdict(dict)  # seg_id → {task: (s_mean, s_sd)}
    for t in TASKS:
        p = _load_post(t, variant=variant)
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
                "converged": "True",
            })
        sids = [str(x) for x in p["seg_ids"]]
        sm = np.asarray(p["s_id_mean"], float)
        ss = np.asarray(p["s_sd"], float)
        for j in range(len(sids)):
            s_table.append((t, sids[j], sm[j], ss[j]))
            seg_sig[sids[j]][t] = (float(sm[j]), float(ss[j]))
        print(f"    {t:>5}: {len(rid):>5} raters, {len(sids):>6} segs",
              flush=True)

    # ── 1. Side-car: joint per-rater params (NOT consumed by engine/Youden) ─
    sidecar = JOINT / f"joint_per_rater_params_k7_{variant}.csv"
    with open(sidecar, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SDT_COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"  [1] {sidecar.name}: {len(rows)} rows "
          f"(SIDE-CAR; not engine-consumed for V_B per Phase-3.5 decoupling)",
          flush=True)

    # ── 2. s_j table (engine bank source for K=7) ──
    s_table_path = JOINT / f"s_j_table_k7_{variant}.csv"
    with open(s_table_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seg_id", "s_mean", "s_sd"])
        for t, sid, sm, ss in s_table:
            w.writerow([t, sid, f"{sm:.10g}", f"{ss:.10g}"])
    print(f"  [2] {s_table_path.name}: {len(s_table)} (task,seg) rows",
          flush=True)

    # ── 3. Denormalized per-segment analysis CSV (replaces iiic_segment_signals.csv) ──
    tier_keys = ["expert", "experienced", "borderline", "novice",
                 "other", "unknown", "untiered"]
    with open(OUT_SEG_SIGNALS, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["seg_id", "n_sources", "sources"]
        for t in TASKS:
            hdr += [f"s_mean_{t}", f"s_sd_{t}"]
        hdr += [f"n_{k}" for k in tier_keys]
        hdr += ["votes_spike_yes", "votes_spike_no"]
        hdr += [f"votes_{c}" for c in
                ("seizure", "lpd", "gpd", "lrda", "grda", "other")]
        w.writerow(hdr)
        for sid in sorted(seg_sig, key=int):
            m = seg_meta.get(sid, {
                "src": set(), "tier": {}, "cls": {},
                "spike_yes": 0, "spike_no": 0})
            row = [sid, len(m["src"]), "|".join(sorted(m["src"]))]
            for t in TASKS:
                sm, ss = seg_sig[sid].get(t, ("", ""))
                row += [f"{sm:.6g}" if sm != "" else "",
                        f"{ss:.6g}" if ss != "" else ""]
            row += [m["tier"].get(k, 0) for k in tier_keys]
            row += [m.get("spike_yes", 0), m.get("spike_no", 0)]
            row += [m["cls"].get(c, 0) for c in
                    ("seizure", "lpd", "gpd", "lrda", "grda", "other")]
            w.writerow(row)
    print(f"  [3] {OUT_SEG_SIGNALS.name}: {len(seg_sig):,} segments × 7-task "
          f"s_j (denormalized; replaces iiic_segment_signals.csv at K=7)",
          flush=True)

    # ── 4. Provenance summary ──
    summary = {
        "phase": 9,
        "K": 7,
        "variant": variant,
        "param_mapping": "sigma=exp(-ell), theta=-t",
        "uniform_collapse": "{bipd,birds}->other (Phase-3 erratum fix; IIIC only)",
        "spike_binary_filter": "value in {'0','1'} only; centaur_2025_ied excluded",
        "joint_per_rater_decoupled": (
            "V_B per Phase-3.5: joint fit serves s_j+s_sd only; per-rater "
            "(sigma,theta) come from byte-verbatim two-stage at "
            "data/engine_inputs/sdt_fits.csv + sdt_fits.spike.csv (D2 preserved)"),
        "outputs": {
            "joint_per_rater_sidecar": str(sidecar.relative_to(REPO)),
            "s_j_table": str(s_table_path.relative_to(REPO)),
            "segment_signals": str(OUT_SEG_SIGNALS.relative_to(REPO)),
        },
        "n_sidecar_rows": len(rows),
        "n_s_table_rows": len(s_table),
        "n_segments_in_signals": len(seg_sig),
        "tasks": TASKS,
    }
    summary_path = JOINT / f"assemble_summary_k7_{variant}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  [4] {summary_path.name}: provenance summary", flush=True)
    print(f"=== Layer 2 assembly complete ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
