#!/usr/bin/env python3
"""Emit the promotion-candidate draw-latent atom grid as a versioned artifact.

The grid is the implementer default recorded in
docs/DRAW_LATENT_PROMOTION_PLAN.md (owner ratification pending, surfaced in
the final report): 33 population-quantile atoms at the OUT-OF-SAMPLE
heterogeneity (tau = 0.3581, oos_population_check) with the 0.15 robustness
floor, PLUS the lambda_d = 1 binary-nesting atom (battery evidence:
collapse-world coverage 0.43 -> 0.93; F1 then nests binary), uniform atom
weights. The rows are asserted byte-identical to
draw_latent_rd.harness.build_atoms("nesting34"), which produced the battery
and head-to-head evidence.

Research-only emission: promotionForbidden stays true; the Phase-2 qualified
re-emission is a separate, provenance-chained step.

Usage (from n-way-protocol/):
    PYTHONPATH="python:." python3 scripts/make_nesting34_artifact.py
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from draw_latent_rd.harness import OOS_TAU, build_atoms
from nway_protocol.artifact_engine_frame import population_draws
from nway_protocol.artifact_floor import floor_draws

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/iiic_conditional_f1_engine_frame_rd.json"
OUTPUT = ROOT / "artifacts/iiic_conditional_f1_engine_frame_nesting34_rd.json"


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    effects = dict(payload["randomEffects"])
    widened = dict(effects)
    widened["tau_log_beta"] = OOS_TAU
    rows = floor_draws(
        population_draws(widened, 0.0, 33), payload["robustnessFloor"],
    )
    rows = [dict(row) for row in rows] + [
        {"beta": 0.9796, "distractor_lapse": 1.0, "weight": 1.0}
    ]
    for row in rows:
        row["weight"] = 1.0 / len(rows)

    oracle = build_atoms("nesting34")
    emitted = tuple(
        (float(row["beta"]), float(row["distractor_lapse"]), float(row["weight"]))
        for row in rows
    )
    if emitted != oracle:
        raise SystemExit("emitted rows disagree with build_atoms('nesting34')")

    artifact = copy.deepcopy(payload)
    artifact["variant"] = "engine_frame_hierarchical_nesting34"
    artifact["bootstrap"]["draws"] = rows
    artifact["promotionForbidden"] = True
    artifact["status"] = "research_only_not_promoted"
    provenance = artifact.setdefault("provenance", {})
    provenance["derived_from"] = SOURCE.name
    provenance["derived_from_sha256"] = hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest()
    provenance["derivation"] = (
        "population_draws(count=33, tau_log_beta=OOS 0.3581 from "
        "oos_population_check) floored at 0.15, plus the lambda_d=1 "
        "binary-nesting atom at beta=exp(mu_log_beta); uniform 1/34 weights. "
        "Byte-identical to draw_latent_rd.harness.build_atoms('nesting34') "
        "(viability battery cells stress190/collapse_nest)."
    )
    provenance["atom_grid_decision"] = (
        "implementer default per DRAW_LATENT_PROMOTION_PLAN.md; owner "
        "ratification pending"
    )
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": OUTPUT.name,
        "n_atoms": len(rows),
        "beta_range": [rows[0]["beta"], rows[-2]["beta"]],
        "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
