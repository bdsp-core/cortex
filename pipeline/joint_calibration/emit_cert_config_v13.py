"""Phase 3.5 part-2: emit cert_config v13 (supersedes v12).

v13 = the erratum-fixed VERBATIM two-stage Youden ell* (6 IIIC) + spike
sigma* [from run_unified_calibration.py, re-run with the {bipd,birds}->
other uniform-collapse fix] + the joint-fit s_j/s_sd artifacts. The
per-rater ell* is NOT from the joint posterior (DECOUPLE: re-deriving
Youden from the joint fit gave degenerate sz separation, Cohen-d=-0.38,
J=0.089 — Kong-crowd likelihood dominance + expertise-tier interaction).
Numbers are NOT hand-edited: this only re-keys v12->v13 and adds the
phase35 provenance block.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CFG = REPO / "calibration" / "cert_config.yaml"
JOINT = REPO / "calibration" / "joint"


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text())
    # Re-runnable: accept v12 (first emit) OR v13 (provenance refresh, e.g.
    # after the variance calibration lands). Never hand-edit numbers.
    if "ell_star_unified_v13" in cfg:
        block = cfg.pop("ell_star_unified_v13")
    else:
        block = cfg.pop("ell_star_unified_v12")
    Js = {t: v.get("youden_j") for t, v in block["tasks"].items()
          if t.startswith("sparcnet_")}
    iic_J = block["tasks"].get("sparcnet_iic", {}).get("youden_j")

    block["provenance"]["phase35"] = {
        "decision": "DECOUPLE (2026-05-18, user-approved): the joint "
        "hierarchical fit serves s_j unification + s_sd uncertainty ONLY; "
        "per-rater Youden ell* is the erratum-fixed VERBATIM two-stage "
        "path. Re-deriving Youden from the joint posterior was REJECTED — "
        "degenerate expert separation (sz Cohen-d=-0.38, J=0.089) from "
        "Kong-crowd likelihood dominance (50.7% of IIIC obs) + the "
        "expertise-tier interaction.",
        "phase3_erratum_fixed": "build_calibration_inputs.py applied "
        "value=='other' only -> ~30,511 Centaur-novice bipd/birds reads "
        "were mislabelled NEGATIVE for `iic`. Fixed: uniform "
        "{bipd,birds}->other collapse across ALL sources (AUDIT s6 / "
        "Phase-1). Effect: iic Youden J 0.643 -> "
        f"{iic_J:.4f}; other 5 IIIC unchanged.",
        "iiic_min_J": min(Js.values()), "iiic_mean_J":
            sum(Js.values()) / len(Js),
        "joint_sj_artifacts": {
            "s_j_table": "calibration/joint/s_j_table.csv "
            "(per task,seg: s_mean,s_sd — the engine bank source)",
            "denormalized_view":
            "data/labels/iiic_segment_signals.csv (69,806 IIIC segs x "
            "6-task s_mean/s_sd + source/tier/vote breakdown)",
            "joint_per_rater_sidecar":
            "calibration/joint/joint_per_rater_params.csv (provenance "
            "only — NOT consumed; the sz J=0.089 degeneracy is why)",
            "sj_table_md5": _md5(JOINT / "s_j_table.csv"),
        },
        "engine_s_sd_propagation": "closed-form Gaussian-probit "
        "marginalization z/=sqrt(1+(e^l*s_sd)^2) in engine.core/_marg_z + "
        "core_mcmc; default s_sd=0.0 ⇒ BIT-EXACT parity. Gate PASSED: "
        "full suite 177 passed / 1 xfailed (prior 168 bit-identical).",
        "svi_inference": "AutoLowRankMultivariateNormal rank-20, 6 tasks "
        "x 962,467 obs, gauge-invariance|dp|<4e-7, anchors exact "
        "(s_gold=0,sd_weld=1), s_sd uncollapsed.",
        "lambda": 0.025,
    }
    vc_path = JOINT / "variance_calibration.json"
    if vc_path.exists():
        vc = json.loads(vc_path.read_text())
        block["provenance"]["phase35"]["variance_calibration"] = {
            "status": "DONE — SVI s_sd calibrated vs NUTS-subsample "
            "(exact Bayes) on shared segments.",
            "kappa_per_task": vc["kappa"],
            "s_sd_underdispersion": "SVI under-dispersed s_sd by ~18-42% "
            "vs NUTS; s_sd_calibrated = s_sd * kappa (post-cal SVI/NUTS "
            "median = 1.00). Engine bank consumes s_j_table_calibrated.csv.",
            "s_mean_agreement_pearson_r": {
                t: vc["per_task"][t]["s_mean_pearson_r"] for t in vc[
                    "per_task"]},
            "honest_caveat": "s_mean SVI-vs-NUTS r ~ 0.72-0.78 (MODERATE, "
            "not >0.9). NUTS ran on an 80k SUBSAMPLE (~12x fewer "
            "obs/segment than full-corpus SVI) so the NUTS reference is "
            "itself the noisier point estimate; a tighter check needs "
            "full-corpus NUTS (intractable on CPU — the reason SVI was "
            "chosen). Documented limitation, NOT a tight-equivalence "
            "claim. kappa is conservative (subsample inflates NUTS s_sd).",
        }
        block["provenance"]["phase35"]["pending"] = (
            "plug-in-vs-uncertainty held-out AUROC + delta_Centaur "
            "sensitivity + SBC (Phase-3.5 release gate).")
    else:
        block["provenance"]["phase35"]["pending"] = (
            "NUTS-subsample VI-vs-NUTS s_sd variance calibration + "
            "plug-in-vs-uncertainty AUROC + delta_Centaur + SBC.")
    cfg["config_version"] = 13
    cfg["ell_star_unified_v13"] = block
    CFG.write_text(
        "# cert_config v13 (Phase 3.5). Supersedes v12. The orchestrator "
        "run_unified_calibration.py writes a v12-schema file; this v13 is "
        "produced by pipeline/joint_calibration/emit_cert_config_v13.py "
        "(re-key + phase35 provenance only — numbers unchanged).\n"
        + yaml.safe_dump(cfg, sort_keys=False, width=100))
    print(f"  cert_config -> v13  | IIIC min J={min(Js.values()):.4f} "
          f"mean={sum(Js.values())/len(Js):.4f}  iic J={iic_J:.4f} "
          f"(was 0.643 pre-erratum-fix)  mode_b_legacy="
          f"{'mode_b_legacy' in cfg}")


if __name__ == "__main__":
    main()
