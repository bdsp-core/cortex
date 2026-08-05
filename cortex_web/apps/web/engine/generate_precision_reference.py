"""Generate the fixed-cloud Python reference for PrecisionPolicy parity.

This intentionally avoids resampling and rejuvenation.  The browser and
Python engines use different production RNGs, so trajectory-level numerical
parity is meaningful only on a shared, scripted particle cloud.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
for source in (
    ROOT / "termination-policy" / "src",
    ROOT / "precision-policy" / "src",
):
    sys.path.insert(0, str(source))

from precision_policy.policy import (  # noqa: E402
    PRECISION_CONTRACTION_BY_DOMAIN,
    PRECISION_RADIUS_MCSE_INFLATION_30MH,
    PrecisionPolicy,
)


def cloud(width: float) -> dict:
    n, k_count = 96, 7
    base = np.linspace(-width, width, n, dtype=float)
    l = np.column_stack(
        [base + (k - 3) * 0.07 for k in range(k_count)]
    )
    t = np.column_stack(
        [base[::-1] * 0.4 + (k - 3) * 0.03 for k in range(k_count)]
    )
    # Deliberately non-uniform, normalized weights exercise weighted-CDF
    # crossing and ESS semantics while staying above the frozen ESS floor.
    weights = np.linspace(1.0, 2.0, n, dtype=float)
    weights /= weights.sum()
    return {"l": l, "t": t, "w": weights}


# Graded bias-flag reference (bias-reporting-policy APPLY_PLAN §4). Computed
# HERE, not in the Python policy package: precision-policy/ carries no flag
# surface by charter. Constants mirror the TS engine's PRECISION_BIAS_FLAG_*
# block (g5_operating_point.json sha256 b5b8ed5f…91bc) and the guard z is the
# policy's radius_mcse_z.
BIAS_FLAG_TAU = 1.0
BIAS_FLAG_TIERS = (
    (0.975, "EXTREME_CONFIRMED_OVERCALLER", "EXTREME_CONFIRMED_UNDERCALLER"),
    (0.575, "EXTREME_OVERCALLER", "EXTREME_UNDERCALLER"),
    (0.5, "WATCH_OVERCALLER", "WATCH_UNDERCALLER"),
)
BIAS_FLAG_MCSE_Z = 1.645


def graded_flag(tail_over, tail_under, ess, ess_pass, status, evidence, content):
    if status.startswith("UNDETERMINABLE_"):
        return {"flag": None, "withheld_reason": "undeterminable_domain"}
    if not (evidence and content and ess_pass):
        return {"flag": None, "withheld_reason": "insufficient_evidence"}

    def guarded(p):
        return p - BIAS_FLAG_MCSE_Z * np.sqrt(p * (1.0 - p) / max(ess, 1.0))

    over, under = guarded(tail_over), guarded(tail_under)
    for threshold, over_name, under_name in BIAS_FLAG_TIERS:
        if over >= threshold:
            return {"flag": over_name, "withheld_reason": None}
        if under >= threshold:
            return {"flag": under_name, "withheld_reason": None}
    return {"flag": None, "withheld_reason": None}


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def main() -> None:
    band_edges = [[-0.5, 0.5] for _ in range(7)]
    policy = PrecisionPolicy(
        np.ones(7),
        r_l=np.asarray(PRECISION_CONTRACTION_BY_DOMAIN, dtype=float),
        confidence=0.95,
        n_min=20,
        per_domain_cap=60,
        persistence=2,
        band_edges=band_edges,
        band_min=3,
        reliability_mode="quantile_mcse",
        ess_floor_fraction=0.5,
        radius_mcse_z=1.645,
        # Shipped guard inflation (mc-guard v2, default since 2026-07-18);
        # tracks the promoted default so the parity fixture never drifts.
        radius_mcse_inflation=PRECISION_RADIUS_MCSE_INFLATION_30MH,
        precision_statistic="point_centered_radius",
    )
    telemetry = {
        "remaining_bank_counts": [100] * 7,
        "band_administered": [[3, 3, 3] for _ in range(7)],
        "band_remaining": [[30, 30, 30] for _ in range(7)],
    }
    n_per_task = [20] * 7
    narrow = cloud(0.45)
    broad = cloud(2.8)

    expected = []
    for state in (narrow, narrow, broad):
        decision = policy(state, telemetry, n_per_task, 7)
        # ADDITIVE graded-flag reference: tail masses over the same normalized
        # weights the intervals consume, then the graded/gated rule fed by the
        # step's own recorded diagnostics.
        diag = decision.diagnostics
        weights = state["w"] / state["w"].sum()
        tail_over = [
            float(weights[state["t"][:, k] > BIAS_FLAG_TAU].sum()) for k in range(7)
        ]
        tail_under = [
            float(weights[state["t"][:, k] < -BIAS_FLAG_TAU].sum()) for k in range(7)
        ]
        graded = [
            graded_flag(
                tail_over[k],
                tail_under[k],
                diag["ess"],
                diag["ess_pass"],
                diag["statuses"][k],
                diag["evidence_floor_met"][k],
                diag["content_floor_met"][k],
            )
            for k in range(7)
        ]
        expected.append(
            {
                "stop": decision.stop,
                "stop_reason": decision.stop_reason,
                "domain_statuses": decision.domain_statuses,
                "streak_counts": decision.streak_counts,
                "diagnostics": decision.diagnostics,
                "bias_tail_over": tail_over,
                "bias_tail_under": tail_under,
                "graded_flags": graded,
            }
        )

    output = {
        "description": "96-particle scripted cloud; no resampling or RNG",
        "band_edges": band_edges,
        "n_per_task": n_per_task,
        "telemetry": telemetry,
        "clouds": {
            "narrow": {key: serializable(value) for key, value in narrow.items()},
            "broad": {key: serializable(value) for key, value in broad.items()},
        },
        "expected": serializable(expected),
    }
    destination = Path(__file__).with_name("__testdata__") / "precision_reference.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
