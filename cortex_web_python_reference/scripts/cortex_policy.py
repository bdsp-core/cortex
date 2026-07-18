"""Compatibility façade for the organized CORTEX policy projects.

Concrete policy math is maintained in the top-level ``ad6-policy`` and
``precision-policy`` projects. Existing Python-reference imports continue to
use this module, so the controller and historical harnesses retain a stable
surface while AD6 remains the default.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
else:
    _REPO = Path(__file__).resolve().parent.parent
    _POLICY_WORKSPACE = (
        _REPO if (_REPO / "precision-policy").is_dir() else _REPO.parent
    )
    # Source-checkout fallback. Installed distributions resolve normally; this
    # makes the monorepo runnable without mutating a developer's environment.
    for _source in (
        _POLICY_WORKSPACE / "termination-policy" / "src",
        _POLICY_WORKSPACE / "ad6-policy" / "src",
        _POLICY_WORKSPACE / "precision-policy" / "src",
    ):
        if _source.is_dir() and str(_source) not in sys.path:
            sys.path.insert(0, str(_source))

from cortex_termination_policy import StopDecision, TerminationPolicy  # noqa: E402
from ad6_policy import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_N_MIN,
    DEFAULT_R_STAR,
    DEFAULT_Z,
    FAIL,
    PASS,
    PENDING,
    REFER_BORDERLINE,
    REFER_UNINFORMATIVE,
    AD6Policy as _StandaloneAD6Policy,
)
from precision_policy import (  # noqa: E402
    ABOVE_CUT,
    ACTIVE,
    BELOW_CUT,
    DETERMINED,
    ESTIMATE_COMPLETE,
    INDETERMINATE_AT_CUT,
    PRECISION_BAND_MIN,
    PRECISION_CONTRACTION_BY_DOMAIN,
    PRECISION_CONTRACTION_FRACTION,
    PRECISION_ESS_FLOOR_FRACTION,
    PRECISION_RADIUS_MCSE_INFLATION,
    PRECISION_RADIUS_MCSE_Z,
    PRECISION_SURROGATE_ACCEPTANCE_FLOOR,
    PRECISION_SURROGATE_ANCESTRY_FLOOR,
    PRECISION_TASK_CODES,
    UNDETERMINABLE_BANK,
    UNDETERMINABLE_CAP,
    PrecisionPolicy,
    classify_determined_interval_against_cut,
    classify_interval_against_cut,
)
from precision_policy.policy import (  # noqa: E402
    _halfwidth_mcse,
    _point_centered_radius,
    _point_centered_radius_mcse,
    _quantile_density,
    _weighted_quantile,
)


def load_ell_star_iiic(task_codes, config_path=None):
    """Load the historical K=6 AD6 cuts from the reference configuration."""

    import yaml

    path = (
        Path(config_path)
        if config_path
        else _REPO / "calibration" / "cert_config.yaml"
    )
    if not path.exists():
        alternate = _REPO / "cert_config.yaml"
        if alternate.exists():
            path = alternate
    if not path.exists():
        raise FileNotFoundError(
            f"cert_config.yaml not found at {path} — AD6Policy needs the "
            "Youden cut scores"
        )
    with path.open() as stream:
        data = yaml.safe_load(stream) or {}
    try:
        tasks = data["ell_star_unified_v13"]["tasks"]
    except KeyError as exc:
        raise KeyError(f"cert_config missing ell_star_unified_v13.tasks: {exc}")
    out = []
    for code in task_codes:
        key = f"sparcnet_{code}"
        if key not in tasks:
            raise KeyError(f"cert_config has no entry for {key}")
        out.append(float(tasks[key]["ell_star"]))
    return out


class AD6Policy(_StandaloneAD6Policy):
    """Reference adapter that adds configuration-backed construction to AD6."""

    @classmethod
    def from_inputs(cls, inputs, **kwargs):
        ell_star = load_ell_star_iiic(inputs.task_codes)
        var_prior = list(np.diag(np.asarray(inputs.Corr_l, dtype=float)))
        return cls(ell_star, var_prior, **kwargs)


class NoStopPolicy(TerminationPolicy):
    """Never stop; retained for audit and operating-characteristic harnesses."""

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        return StopDecision(stop=False, stop_reason="continue")


class DeltaStopPolicy(TerminationPolicy):
    """Legacy AUROC-half-width termination retained for methodology replay."""

    def __init__(self, delta_auroc: float = 0.15):
        self.delta_auroc = float(delta_auroc)

    def __call__(self, state, telemetry, n_per_task, K) -> StopDecision:
        if telemetry["max_hw"] < self.delta_auroc:
            return StopDecision(stop=True, stop_reason="delta_reached")
        return StopDecision(stop=False, stop_reason="continue")


def default_policy_for(inputs, *, delta_auroc=None, policy=None) -> TerminationPolicy:
    """Resolve the historical Python default; AD6 remains unchanged."""

    if policy is not None:
        return policy
    if delta_auroc is None:
        return AD6Policy.from_inputs(inputs)
    if float(delta_auroc) == 0.0:
        return NoStopPolicy()
    return DeltaStopPolicy(float(delta_auroc))


__all__ = [
    "AD6Policy",
    "PrecisionPolicy",
    "DeltaStopPolicy",
    "NoStopPolicy",
    "StopDecision",
    "TerminationPolicy",
    "PASS",
    "FAIL",
    "PENDING",
    "REFER_BORDERLINE",
    "REFER_UNINFORMATIVE",
    "ACTIVE",
    "ESTIMATE_COMPLETE",
    "UNDETERMINABLE_CAP",
    "UNDETERMINABLE_BANK",
    "DETERMINED",
    "ABOVE_CUT",
    "BELOW_CUT",
    "INDETERMINATE_AT_CUT",
    "classify_interval_against_cut",
    "classify_determined_interval_against_cut",
    "DEFAULT_N_MIN",
    "DEFAULT_R_STAR",
    "DEFAULT_ALPHA",
    "DEFAULT_Z",
    "PRECISION_TASK_CODES",
    "PRECISION_CONTRACTION_FRACTION",
    "PRECISION_CONTRACTION_BY_DOMAIN",
    "PRECISION_BAND_MIN",
    "PRECISION_ESS_FLOOR_FRACTION",
    "PRECISION_SURROGATE_ACCEPTANCE_FLOOR",
    "PRECISION_SURROGATE_ANCESTRY_FLOOR",
    "PRECISION_RADIUS_MCSE_Z",
    "PRECISION_RADIUS_MCSE_INFLATION",
    "default_policy_for",
]
