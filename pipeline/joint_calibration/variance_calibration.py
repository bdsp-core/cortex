"""Phase 3.5 gate: VI-vs-NUTS s_sd variance calibration (SVI defensibility).

The production s_j posterior is SVI (full corpus, AutoLowRankMVN). VI can
under-disperse posterior variance — and propagating a faithful s_sd is the
WHOLE point of Phase 3.5 — so we calibrate the SVI s_sd against NUTS
(exact Bayes) on an 80k stratified subsample, per task, on the segments
common to both fits.

Honest caveats (documented in the report):
  * NUTS ran on an 80k SUBSAMPLE → fewer obs/segment than the full-corpus
    SVI, so NUTS s_sd is inflated by the smaller per-segment n in ADDITION
    to any VI under-dispersion. The raw ratio is therefore a CONSERVATIVE
    upper bound on the inflation SVI actually needs — calibrating to it
    cannot understate s_sd (the safe direction: it will not re-inflate the
    power claim Phase 3.5 is correcting).
  * sz NUTS s_j R̂≈1.13 (mild non-mixing, hardest task) → its NUTS s_sd
    is itself slightly inflated → its calibration factor is conservative.

Outputs:
  calibration/joint/variance_calibration.json   — per-task: n shared segs,
      s_mean Pearson r + bias, s_sd ratio quantiles, calibration factor
      kappa = median(NUTS_s_sd / SVI_s_sd) (clipped at >=1.0: never
      shrink), and the post-calibration agreement.
  calibration/joint/s_j_table_calibrated.csv    — s_j_table with
      s_sd_calibrated = s_sd * kappa[task]  (the bank the engine consumes).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "calibration" / "joint"
TASKS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]


def _load_npz(npz):
    d = np.load(npz, allow_pickle=True)
    return ({int(s): i for i, s in enumerate(d["seg_ids"])},
            np.asarray(d["s_id_mean"], float), np.asarray(d["s_sd"], float))


def _load_svi_from_table(task):
    """SVI s_mean/s_sd come from the committed s_j_table.csv (the raw SVI
    posterior npz were clobbered by the NUTS-xcheck driver writing the
    same {task}_posterior.npz path; all SVI-DERIVED artifacts had already
    been assembled+persisted, so the s_sd is intact here — no loss)."""
    seg2i, mu, sd = {}, [], []
    with open(JOINT / "s_j_table.csv") as f:
        for d in csv.DictReader(f):
            if d["task"] != task:
                continue
            seg2i[int(d["seg_id"])] = len(mu)
            mu.append(float(d["s_mean"]))
            sd.append(float(d["s_sd"]))
    return seg2i, np.array(mu), np.array(sd)


def main() -> None:
    print("=== Phase 3.5: VI-vs-NUTS s_sd variance calibration ===",
          flush=True)
    report = {}
    kappa = {}
    for t in TASKS:
        nut = JOINT / f"{t}_nuts_posterior.npz"
        if not nut.exists():
            raise SystemExit(f"ABORT: missing NUTS posterior for {t}")
        s_map, s_mu, s_sd = _load_svi_from_table(t)   # SVI ex s_j_table.csv
        n_map, n_mu, n_sd = _load_npz(nut)            # NUTS exact-Bayes ref
        shared = sorted(set(s_map) & set(n_map))
        si = np.array([s_map[g] for g in shared])
        ni = np.array([n_map[g] for g in shared])
        smu, ssd = s_mu[si], s_sd[si]
        nmu, nsd = n_mu[ni], n_sd[ni]
        r = float(np.corrcoef(smu, nmu)[0, 1])
        bias = float(np.mean(smu - nmu))
        ratio = nsd / np.maximum(ssd, 1e-9)          # NUTS / SVI
        q = {p: float(np.quantile(ratio, p / 100))
             for p in (10, 25, 50, 75, 90)}
        k = max(1.0, float(np.median(ratio)))        # never shrink s_sd
        kappa[t] = k
        post = (ssd * k) / np.maximum(nsd, 1e-9)     # post-cal SVI/NUTS
        report[t] = {
            "n_shared_segs": len(shared),
            "s_mean_pearson_r": round(r, 4),
            "s_mean_bias_svi_minus_nuts": round(bias, 4),
            "s_sd_ratio_nuts_over_svi_quantiles": {str(k_): round(v, 3)
                                                   for k_, v in q.items()},
            "calibration_kappa": round(k, 4),
            "post_calibration_svi_over_nuts_median":
                round(float(np.median(post)), 4),
            "svi_s_sd_median": round(float(np.median(ssd)), 4),
            "nuts_s_sd_median": round(float(np.median(nsd)), 4),
        }
        print(f"  {t:5s} shared={len(shared):>5d}  s_mean r={r:.3f} "
              f"bias={bias:+.3f}  s_sd NUTS/SVI med={q[50]:.2f}  "
              f"kappa={k:.3f}  post-cal SVI/NUTS med="
              f"{report[t]['post_calibration_svi_over_nuts_median']:.2f}")

    # apply calibration to the engine-bank s_j table
    src = JOINT / "s_j_table.csv"
    out = JOINT / "s_j_table_calibrated.csv"
    n = 0
    with open(src) as fi, open(out, "w", newline="") as fo:
        rd = csv.DictReader(fi)
        w = csv.writer(fo)
        w.writerow(["task", "seg_id", "s_mean", "s_sd", "s_sd_calibrated"])
        for d in rd:
            k = kappa[d["task"]]
            w.writerow([d["task"], d["seg_id"], d["s_mean"], d["s_sd"],
                        f"{float(d['s_sd']) * k:.10g}"])
            n += 1
    summary = {
        "method": "kappa = max(1, median(NUTS_s_sd / SVI_s_sd)) per task, "
        "on shared segments; s_sd_calibrated = s_sd * kappa. NUTS = 80k "
        "subsample exact-Bayes reference.",
        "conservative_note": "NUTS subsample has fewer obs/segment than "
        "full-corpus SVI, so kappa is an UPPER bound on needed inflation "
        "— calibration cannot understate s_sd (safe direction).",
        "per_task": report,
        "kappa": {k: round(v, 4) for k, v in kappa.items()},
        "n_calibrated_rows": n,
    }
    (JOINT / "variance_calibration.json").write_text(
        json.dumps(summary, indent=2))
    print(f"  -> kappa {summary['kappa']}")
    print(f"  -> s_j_table_calibrated.csv ({n} rows) + "
          "variance_calibration.json")


if __name__ == "__main__":
    main()
