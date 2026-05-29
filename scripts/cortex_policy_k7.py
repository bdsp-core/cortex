"""CORTEX K=7 policy module — Phase 9 Layer 6a sibling of `cortex_policy.py`.

K=7 extension of the K=6 IIIC predecessor. The AD6Policy class itself is
K-agnostic — it operates on K-dim ell_star + var_prior arrays — so it lives
unchanged in cortex_policy.py and is re-exported here for K=7 use.

The K=7-specific changes:

  * `load_ell_star_k7(task_codes)` — maps each K=7 task code to its
    cert_config v13/v14 key. v13 uses verbatim names (`combined_spike` for
    spike, `sparcnet_*` for IIIC); v14 will preserve these per Eli's choice
    Q2 ("keep verbatim to reduce blast radius"). This function is K=7-aware
    where the K=6 cortex_policy.load_ell_star_iiic hardcoded `sparcnet_{code}`.

  * `default_policy_for_k7(inputs)` — constructs an AD6Policy with K=7
    ell_star + Var_prior (the diagonal of the K=7 Corr_l from
    Sigma_l_fitted_k7.npy).

When v14 cert_config ships (Layer 4 output), this module switches its
default config_path to point at `calibration/cert_config_v14.yaml`. Until
v14 lands, it reads v13 (which already carries all 7 ℓ\* values).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Re-export AD6Policy, DeltaStopPolicy, NoStopPolicy, StopDecision, verdict labels
# (all K-agnostic; defined in cortex_policy.py)
if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
else:
    _REPO = Path(__file__).resolve().parent.parent

_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cortex_policy import (  # noqa: E402  re-export
    AD6Policy, DeltaStopPolicy, NoStopPolicy, StopDecision, TerminationPolicy,
    PASS, FAIL, PENDING, REFER_BORDERLINE, REFER_UNINFORMATIVE,
    DEFAULT_N_MIN, DEFAULT_R_STAR, DEFAULT_ALPHA, DEFAULT_Z,
)

# K=7 task code → cert_config v13/v14 YAML key. v14 keeps v13 names verbatim
# per Eli's Q2 choice.
_KEY_FOR_CODE_K7 = {
    "spike": "combined_spike",
    "sz":    "sparcnet_sz",
    "lpd":   "sparcnet_lpd",
    "gpd":   "sparcnet_gpd",
    "lrda":  "sparcnet_lrda",
    "grda":  "sparcnet_grda",
    "iic":   "sparcnet_iic",
}


def load_ell_star_k7(task_codes, config_path=None,
                     block_name: str = "ell_star_unified_v14") -> list:
    """Load the Youden ℓ\*_k for K=7 (spike + 6 IIIC) from cert_config.

    Returns a list aligned to `task_codes`. Default block is
    `ell_star_unified_v14` (Phase-9 ship state — uniform CV-top-14 Youden
    across all 7 tasks). Callers can pass `block_name='ell_star_unified_v13'`
    explicitly for the legacy K=6+spike-special calibration.
    """
    import yaml
    path = Path(config_path) if config_path else (
        _REPO / "calibration" / "cert_config.yaml")
    if not path.exists():
        alt = _REPO / "cert_config.yaml"
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(
            f"cert_config.yaml not found at {path} — K=7 AD6Policy needs "
            "the Youden ℓ\\*_k cut-scores.")
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    try:
        tasks = data[block_name]["tasks"]
    except KeyError as e:
        raise KeyError(
            f"cert_config missing {block_name}.tasks: {e}")
    out = []
    for code in task_codes:
        key = _KEY_FOR_CODE_K7.get(code)
        if key is None:
            raise KeyError(
                f"K=7 code {code!r} has no cert_config key mapping "
                f"(expected one of {list(_KEY_FOR_CODE_K7)})")
        if key not in tasks:
            raise KeyError(
                f"cert_config[{block_name}].tasks has no entry for {key!r}")
        out.append(float(tasks[key]["ell_star"]))
    return out


def default_policy_for_k7(inputs, *, delta_auroc=None,
                          policy=None, config_path=None,
                          block_name: str = "ell_star_unified_v14"):
    """Resolve the policy used by a K=7 CortexSession. Precedence:
      explicit `policy=` instance         → use it
      `delta_auroc` is None (production)  → AD6Policy.from_inputs_k7(inputs)
      `delta_auroc == 0.0` (audit/tests)  → NoStopPolicy()
      `delta_auroc > 0` (legacy/methods)  → DeltaStopPolicy(delta_auroc)
    """
    if policy is not None:
        return policy
    if delta_auroc is None:
        ell_star = load_ell_star_k7(
            inputs.task_codes, config_path=config_path,
            block_name=block_name)
        var_prior = list(np.diag(np.asarray(inputs.Corr_l, dtype=float)))
        return AD6Policy(ell_star, var_prior)
    if float(delta_auroc) == 0.0:
        return NoStopPolicy()
    return DeltaStopPolicy(float(delta_auroc))


__all__ = [
    "AD6Policy", "DeltaStopPolicy", "NoStopPolicy", "StopDecision",
    "TerminationPolicy",
    "PASS", "FAIL", "PENDING", "REFER_BORDERLINE", "REFER_UNINFORMATIVE",
    "DEFAULT_N_MIN", "DEFAULT_R_STAR", "DEFAULT_ALPHA", "DEFAULT_Z",
    "load_ell_star_k7", "default_policy_for_k7",
]
