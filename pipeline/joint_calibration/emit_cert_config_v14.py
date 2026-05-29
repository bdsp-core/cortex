"""Phase 9 Layer 4 part 2 — emit cert_config v14 with K=7 ell_star values.

Reads `pipeline/reference_calibration/youden_ell_star_k7.json` (Layer-4
output from the K=7 CV-top-14 Youden) and appends an `ell_star_unified_v14`
block to `calibration/cert_config.yaml` with:

  - All 7 ell_star values + 95% bootstrap CIs
  - Per-task n_expert + n_non_expert (panel sizes)
  - Provenance: source path + sha256 + algorithm description
  - v13 → v14 task-name mapping (verbatim per Eli's Q2 choice — preserves
    `combined_spike` and `sparcnet_*` names for downstream consumer stability)

V_13 stays unchanged in the YAML (audit chain preserved). V_14 is the
Phase-9 ship configuration:

  v13 ell_star_unified_v13 (frozen; v1.0-rc1 K=6 + spike-special)
  v14 ell_star_unified_v14 (Phase 9; uniform CV-top-14 across K=7)

Per Eli's directive: "Pursue the rigorous defensible mathematically justified
option." Uniform methodology across all 7 tasks IS the rigorous option:
- Same non-circular CV-top-14 selection algorithm everywhere
- Same Youden index on per-rater ell distribution
- Same bootstrap 95% CI methodology
- Removes the v13 spike-special-case (70/30 TRAIN-only) → cleaner manuscript

Usage:
    .venv/bin/python -m pipeline.joint_calibration.emit_cert_config_v14
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
YOUDEN_JSON = REPO / "pipeline" / "reference_calibration" / "youden_ell_star_k7.json"
CERT_CONFIG = REPO / "calibration" / "cert_config.yaml"

# v13 → v14 task-name mapping (verbatim — Eli's Q2 choice).
# K=7 task code → cert_config v14 YAML key.
KEY_FOR_CODE = {
    "spike": "combined_spike",
    "sz":    "sparcnet_sz",
    "lpd":   "sparcnet_lpd",
    "gpd":   "sparcnet_gpd",
    "lrda":  "sparcnet_lrda",
    "grda":  "sparcnet_grda",
    "iic":   "sparcnet_iic",
}


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    if not YOUDEN_JSON.exists():
        raise FileNotFoundError(
            f"K=7 Youden output missing: {YOUDEN_JSON}. Run "
            "pipeline.reference_calibration.run_youden_calibration_k7 first.")
    payload = json.loads(YOUDEN_JSON.read_text())
    results = payload["results"]
    youden_sha = _sha256_file(YOUDEN_JSON)

    # Read existing cert_config.yaml as raw text; append v14 block at the end
    # (preserves yaml formatting + v13 structure exactly).
    cfg_text = CERT_CONFIG.read_text()
    if "ell_star_unified_v14:" in cfg_text:
        print(f"WARN: {CERT_CONFIG} already has ell_star_unified_v14 block; "
              "appending a NEW v14 block would duplicate. Aborting. Remove "
              "the old block first if you want to regenerate.")
        return 1

    # Build the v14 block text (yaml-safe).
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("")
    lines.append("# ── Phase 9 cert_config v14: K=7 uniform CV-top-14 Youden ────────")
    lines.append("# Source: pipeline/reference_calibration/youden_ell_star_k7.json")
    lines.append(f"# (sha256: {youden_sha})")
    lines.append(f"# Generated: {now}")
    lines.append("#")
    lines.append("# Phase-9 Layer 4 output. Uniform CV-top-14 Youden across all 7 tasks")
    lines.append("# (replaces v13's spike-specific 70/30 TRAIN methodology with the same")
    lines.append("# non-circular cross-task panel selection used for IIIC).")
    lines.append("#")
    lines.append(f"# Overall: min J = {min(r['youden_J'] for r in results.values()):.4f}; "
                 f"mean J = {sum(r['youden_J'] for r in results.values())/len(results):.4f}")
    lines.append("# (v13 baseline: min J = 0.372, mean J = 0.521)")
    lines.append("")
    lines.append("ell_star_unified_v14:")
    lines.append("  method: youden_index_cv_top_N_k7")
    lines.append(f"  panel_size_N: {payload['panel_size_N']}")
    lines.append(f"  bootstrap_n: {payload['bootstrap_n']}")
    lines.append(f"  seed: {payload['seed']}")
    lines.append(f"  sigma_floor: {payload['sigma_floor']}")
    lines.append(f"  candidate_pool: {payload['candidate_pool']!r}")
    lines.append(f"  source_sha256: {youden_sha}")
    lines.append(f"  phase: 9")
    lines.append(f"  K: 7")
    lines.append(f"  generated_utc: {now}")
    lines.append("  description: |")
    lines.append("    Uniform CV-top-14 Youden across spike + 6 IIIC tasks.")
    lines.append("    For each held-out task k, expert panel = top-N candidates by")
    lines.append("    mean ell over OTHER 6 tasks (5 other IIIC + spike for IIIC tasks;")
    lines.append("    6 IIIC for spike). Non-circular by construction.")
    lines.append("    Replaces v13's spike-special-case 70/30 TRAIN methodology;")
    lines.append("    matches K=6 v13 ell* exactly on 5 of 6 IIIC tasks (lpd, gpd,")
    lines.append("    lrda, grda, iic — panel byte-stable). sz panel updates due to")
    lines.append("    spike's inclusion in cross-task expert score (J 0.47 → 0.73).")
    lines.append("    spike methodology changes from 70/30 TRAIN to CV-top-14")
    lines.append("    (J 0.37 → 0.65).")
    lines.append("  tasks:")
    # Emit in K=7 canonical order (matches DOMAINS_K7 in run_youden_calibration_k7)
    order = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    for code in order:
        if code not in results:
            continue
        r = results[code]
        key = KEY_FOR_CODE[code]
        lines.append(f"    {key}:")
        lines.append(f"      task_code: {code}")
        lines.append(f"      ell_star: {r['l_star']:.16f}")
        lines.append(f"      sigma_star: {r['sigma_star']:.16f}")
        lines.append(f"      youden_J: {r['youden_J']:.16f}")
        lines.append(f"      ci_low_2_5: {r['ci_low']:.16f}")
        lines.append(f"      ci_high_97_5: {r['ci_high']:.16f}")
        lines.append(f"      n_expert: {r['n_expert']}")
        lines.append(f"      n_non_expert: {r['n_non_expert']}")
        lines.append(f"      expert_ell_mean: {r['expert_ell_mean']:.16f}")
        lines.append(f"      non_expert_ell_mean: {r['non_expert_ell_mean']:.16f}")
    lines.append("")

    new_text = cfg_text.rstrip() + "\n" + "\n".join(lines) + "\n"
    CERT_CONFIG.write_text(new_text)
    print(f"Appended ell_star_unified_v14 block to {CERT_CONFIG}")
    print(f"  v14 source sha256: {youden_sha}")
    print(f"  Tasks emitted: {sorted(results.keys())}")
    print()
    print("v14 summary table:")
    print(f"  {'task':>6}  {'v14 key':>16}  {'ell*':>9}  {'sigma*':>7}  {'J':>7}")
    for code in order:
        if code not in results:
            continue
        r = results[code]
        print(f"  {code:>6}  {KEY_FOR_CODE[code]:>16}  "
              f"{r['l_star']:>+9.4f}  {r['sigma_star']:>7.4f}  "
              f"{r['youden_J']:>7.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
