"""Emit cert_config `ell_star_unified_v15` (credentialed-panel K=7 cut-scores).

Reads `results/calibration_study/ell_star_v15.json` (from
`pipeline/reference_calibration/build_ell_star_v15.py`) and APPENDS an
`ell_star_unified_v15` block to `calibration/cert_config.yaml`, mirroring the
v14 emitter's schema so `cortex_policy_k7.load_ell_star_k7(block_name=
'ell_star_unified_v15')` resolves it.

STAGED / NON-LIVE: appends only. v13/v14 blocks are untouched, and the loader's
DEFAULT block stays `ell_star_unified_v14` — so the live frozen instrument is
unchanged. v15 is consumed explicitly (sims / scratch / deployment-prior) until
the PI-gated post-pilot re-freeze flips the default.

Usage:  .venv/bin/python -m pipeline.joint_calibration.emit_cert_config_v15
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V15_JSON = REPO / "results" / "calibration_study" / "ell_star_v15.json"
# v15 lives in a SIBLING file (not appended to the frozen cert_config.yaml) so the
# in-flight pilot's instrument-freeze hash on cert_config.yaml stays bit-identical.
CERT_CONFIG_V15 = REPO / "calibration" / "cert_config_v15.yaml"

# Same K=7 code → cert_config YAML key mapping used by v13/v14 + the loader.
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
    if not V15_JSON.exists():
        raise FileNotFoundError(
            f"v15 source missing: {V15_JSON}. Run "
            "pipeline.reference_calibration.build_ell_star_v15 first.")
    payload = json.loads(V15_JSON.read_text())
    results = payload["results"]
    prov = payload["provenance"]
    src_sha = _sha256_file(V15_JSON)

    if CERT_CONFIG_V15.exists():
        print(f"WARN: {CERT_CONFIG_V15} already exists; "
              "remove it first to regenerate. Aborting (no change).")
        return 1

    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    order = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
    L: list[str] = []
    L.append("")
    L.append("# ── v15: CREDENTIALED-panel K=7 Youden (STAGED — NOT the live default) ──")
    L.append("# Source: results/calibration_study/ell_star_v15.json")
    L.append(f"# (sha256: {src_sha})")
    L.append(f"# Generated: {now}")
    L.append("#")
    L.append("# Credentialed expert panels (spike=Super8; IIIC=Super8∪Bonobo), the")
    L.append("# exogenous gold-standard definition (raters.csv:groups) — replaces v14's")
    L.append("# data-driven CV-top-14 panel. lpd uses a robust trim (drop credentialed")
    L.append(f"# experts with lpd ell<={payload['lpd_trim_ell']}; non-expert range).")
    L.append("# PI-confirmed 2026-06-10 (docs/POST_PILOT_PRODUCTION_INSTRUMENT.md §Change-2).")
    L.append("# STAGED for the post-pilot re-freeze: the live loader default stays v14;")
    L.append("# consumers opt in via load_ell_star_k7(block_name='ell_star_unified_v15').")
    L.append(f"# Overall: min J = {min(r['youden_J'] for r in results.values()):.4f}; "
             f"mean J = {sum(r['youden_J'] for r in results.values())/len(results):.4f}")
    L.append("")
    L.append("ell_star_unified_v15:")
    L.append("  method: youden_index_credentialed_panel_k7_v15")
    L.append(f"  panels: {payload['panels']!r}")
    L.append(f"  lpd_trim_ell: {payload['lpd_trim_ell']}")
    L.append(f"  bootstrap_n: {payload['bootstrap_n']}")
    L.append(f"  seed: {payload['seed']}")
    L.append(f"  sigma_floor: {payload['sigma_floor']}")
    L.append(f"  candidate_pool: {payload['candidate_pool']!r}")
    L.append(f"  source_sha256: {src_sha}")
    L.append(f"  raters_csv_sha256: {prov['raters_csv_sha256']}")
    L.append(f"  sdt_fits_k7_sha256: {prov['sdt_fits_k7_sha256']}")
    L.append(f"  phase: 9")
    L.append(f"  K: 7")
    L.append(f"  staged: true   # NOT the live default; post-pilot re-freeze gates adoption")
    L.append(f"  generated_utc: {now}")
    L.append(f"  pi_confirmed: {prov['pi_confirmed']!r}")
    L.append("  description: |")
    L.append("    Credentialed-panel Youden ell* (v15). spike=Super8; IIIC=Super8∪Bonobo")
    L.append("    (exogenous gold-standard panels from raters.csv:groups). lpd robust-trim:")
    L.append("    credentialed experts scoring in the non-expert range (lpd ell<=0.19) are")
    L.append("    dropped, tightening the unstable lpd CI (0.588 -> 0.23). Same byte-pinned")
    L.append("    Youden + bootstrap machinery as v14; only the panel-selection rule differs.")
    L.append("  tasks:")
    for code in order:
        if code not in results:
            continue
        r = results[code]
        key = KEY_FOR_CODE[code]
        L.append(f"    {key}:")
        L.append(f"      task_code: {code}")
        L.append(f"      ell_star: {r['l_star']:.16f}")
        L.append(f"      sigma_star: {r['sigma_star']:.16f}")
        L.append(f"      youden_J: {r['youden_J']:.16f}")
        L.append(f"      ci_low_2_5: {r['ci_low']:.16f}")
        L.append(f"      ci_high_97_5: {r['ci_high']:.16f}")
        L.append(f"      n_expert: {r['n_expert']}")
        L.append(f"      n_non_expert: {r['n_non_expert']}")
        L.append(f"      expert_ell_mean: {r['expert_ell_mean']:.16f}")
        L.append(f"      non_expert_ell_mean: {r['non_expert_ell_mean']:.16f}")
    L.append("")

    CERT_CONFIG_V15.write_text("\n".join(L).lstrip("\n") + "\n")
    print(f"Wrote ell_star_unified_v15 to {CERT_CONFIG_V15} "
          "(sibling; frozen cert_config.yaml stays bit-identical)")
    print(f"  v15 source sha256: {src_sha}")
    print(f"  {'task':>6}  {'ell*':>9}  {'sigma*':>7}  {'J':>7}")
    for code in order:
        r = results[code]
        print(f"  {code:>6}  {r['l_star']:>+9.4f}  {r['sigma_star']:>7.4f}  "
              f"{r['youden_J']:>7.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
