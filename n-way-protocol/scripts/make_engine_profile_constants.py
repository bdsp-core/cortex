#!/usr/bin/env python3
"""Emit the browser-engine artifact constants from a canonical artifact JSON.

The deployed nway_profile.ts draw table is never hand-typed: the Phase-2
floor-adoption regenerated it from the Python-side artifact and validated
the generator bit-for-bit against the previous constants before switching
draws. This script re-establishes that discipline as a checked-in tool:

1. `--validate` re-derives the deployed floor015 ensemble block from
   artifacts/iiic_conditional_f1_integrated_ensemble9_rd_floor015.json and
   compares every beta, lapse, weight, and the canonical draws SHA-256
   against the constants currently in cortex_web/apps/web/engine/
   nway_profile.ts. Any mismatch is a hard failure — the generator must
   reproduce the deployed table exactly before it is trusted for a new one.
2. `--emit-atoms17` normalizes the research 17-atom engine-frame payload
   (artifacts/iiic_conditional_f1_engine_frame_atoms17_rd.json, raw fitter
   output with snake_case bootstrap draws) into the deployable camelCase
   draw table, stamps the canonical draws SHA-256 (identical
   canonicalization to artifact_floor._canonical_draws_sha256: compact
   sorted-key JSON of the camelCase draws array), and prints the TypeScript
   constant block to paste into nway_profile.ts.
3. `--emit-nesting34` does the same for the QUALIFIED nesting34 draw-latent
   payload (artifacts/iiic_conditional_f1_engine_frame_nesting34_qualified
   .json, the Phase-2 re-emission chained to the locked-campaign reports),
   refusing any payload that is not marked qualified/promotable.

The artifact id follows the deployed naming pattern
(<model-family>-<fit-variant>[-rd]-<fit-lineage-date>): both draw-latent
grids share the 2026-07-30 engine-frame refit lineage
(provenance.derived_from).

Usage (from n-way-protocol/):
    python3 scripts/make_engine_profile_constants.py --validate
    python3 scripts/make_engine_profile_constants.py --emit-atoms17
    python3 scripts/make_engine_profile_constants.py --emit-nesting34
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ENSEMBLE9_FLOOR015 = ROOT / "artifacts/iiic_conditional_f1_integrated_ensemble9_rd_floor015.json"
ATOMS17 = ROOT / "artifacts/iiic_conditional_f1_engine_frame_atoms17_rd.json"
NWAY_PROFILE_TS = REPO / "cortex_web/apps/web/engine/nway_profile.ts"

ATOMS17_ARTIFACT_ID = "iiic-f1-engine-frame-atoms17-rd-20260730"
NESTING34_QUALIFIED = ROOT / "artifacts/iiic_conditional_f1_engine_frame_nesting34_qualified.json"
NESTING34_ARTIFACT_ID = "iiic-f1-engine-frame-nesting34-20260730"


def canonical_draws_sha256(draws: list[dict]) -> str:
    """artifact_floor._canonical_draws_sha256, verbatim discipline."""
    canonical = json.dumps(draws, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def normalize_draws(raw_draws: list[dict]) -> list[dict]:
    """snake_case fitter draws -> the deployable camelCase draw table."""
    normalized = []
    for draw in raw_draws:
        lapse = draw.get("distractorLapse", draw.get("distractor_lapse"))
        beta, weight = float(draw["beta"]), float(draw["weight"])
        if not (beta > 0 and weight > 0 and 0 <= float(lapse) <= 1):
            raise SystemExit(f"invalid draw: {draw}")
        normalized.append({
            "beta": beta, "distractorLapse": float(lapse), "weight": weight,
        })
    return normalized


def ts_number(value: float) -> str:
    """Shortest round-trip literal — identical to JS Number serialization."""
    return repr(value)


def ts_draw_lines(draws: list[dict]) -> list[str]:
    return [
        "    { beta: %s, distractorLapse: %s, weight: %s },"
        % (ts_number(d["beta"]), ts_number(d["distractorLapse"]), ts_number(d["weight"]))
        for d in draws
    ]


def parsed_ts_draws(source: str, constant: str) -> tuple[list[dict], str]:
    """Extract the deployed draw table + sha256 for one artifact constant."""
    block = re.search(
        re.escape(constant) + r"\s*=\s*Object\.freeze\(\{(.*?)\n\}\s*as const\)",
        source, re.S,
    )
    if not block:
        raise SystemExit(f"cannot locate {constant} in nway_profile.ts")
    body = block.group(1)
    sha = re.search(r'sha256:\s*"([0-9a-f]{64})"', body)
    rows = re.findall(
        r"\{\s*beta:\s*([-\d.e]+),\s*distractorLapse:\s*([-\d.e]+),"
        r"\s*weight:\s*([^,}]+)\s*\}", body,
    )
    if not sha or not rows:
        raise SystemExit(f"cannot parse draws/sha out of {constant}")
    draws = [
        {"beta": float(beta), "distractorLapse": float(lapse),
         "weight": float(eval(weight, {"__builtins__": {}}))}  # "1 / 9" literals
        for beta, lapse, weight in rows
    ]
    return draws, sha.group(1)


def validate_ensemble9() -> None:
    artifact = json.loads(ENSEMBLE9_FLOOR015.read_text())
    expected = artifact["draws"]
    expected_sha = canonical_draws_sha256(expected)
    if expected_sha != artifact["sha256"]:
        raise SystemExit("floor015 artifact sha256 does not match its own draws")
    deployed, deployed_sha = parsed_ts_draws(
        NWAY_PROFILE_TS.read_text(), "NWAY_ARTIFACT",
    )
    if deployed_sha != expected_sha:
        raise SystemExit("deployed NWAY_ARTIFACT sha256 differs from the artifact")
    if len(deployed) != len(expected):
        raise SystemExit("deployed draw count differs")
    for index, (got, want) in enumerate(zip(deployed, expected)):
        for key in ("beta", "distractorLapse", "weight"):
            if repr(got[key]) != repr(float(want[key])):
                raise SystemExit(
                    f"draw {index} {key}: deployed {got[key]!r} != artifact {want[key]!r}"
                )
    print(f"OK: generator reproduces the deployed floor015 table bit-for-bit "
          f"({len(expected)} draws, sha {expected_sha})")


def emit_atoms17() -> None:
    payload = json.loads(ATOMS17.read_text())
    if payload["variant"] != "engine_frame_hierarchical_atoms17":
        raise SystemExit("unexpected atoms17 payload variant")
    if payload["status"] != "research_only_not_promoted" or not payload["promotionForbidden"]:
        raise SystemExit("atoms17 payload must remain research-only / promotion-forbidden")
    draws = normalize_draws(payload["bootstrap"]["draws"])
    floor = float(payload["robustnessFloor"])
    if min(d["distractorLapse"] for d in draws) < floor:
        raise SystemExit("a draw falls below the recorded robustness floor")
    sha = canonical_draws_sha256(draws)
    lines = "\n".join(ts_draw_lines(draws))
    print(f"""// generated by n-way-protocol/scripts/make_engine_profile_constants.py
export const NWAY_DRAW_LATENT_ARTIFACT = Object.freeze({{
  artifactId: "{ATOMS17_ARTIFACT_ID}",
  sha256: "{sha}",
  robustnessFloor: {ts_number(floor)},
  draws: Object.freeze([
{lines}
  ] satisfies readonly ArtifactDraw[]),
  approval: "research_only_not_promoted",
  sourceQualification: "dr07_gate_passed_owner_ratification_pending",
}} as const);""")


def emit_nesting34() -> None:
    payload = json.loads(NESTING34_QUALIFIED.read_text())
    if payload["variant"] != "engine_frame_hierarchical_nesting34":
        raise SystemExit("unexpected nesting34 payload variant")
    if payload.get("qualification") != "qualified" or payload["promotionForbidden"]:
        raise SystemExit(
            "nesting34 payload must be the qualified Phase-2 re-emission "
            "(qualification: qualified, promotionForbidden: false)"
        )
    draws = normalize_draws(payload["bootstrap"]["draws"])
    floor = float(payload["robustnessFloor"])
    if min(d["distractorLapse"] for d in draws) < floor:
        raise SystemExit("a draw falls below the recorded robustness floor")
    if draws[-1]["distractorLapse"] != 1.0:
        raise SystemExit("the lambda=1 binary-nesting atom must close the grid")
    sha = canonical_draws_sha256(draws)
    lines = "\n".join(ts_draw_lines(draws))
    print(f"""// generated by n-way-protocol/scripts/make_engine_profile_constants.py
export const NWAY_QUALIFIED_DRAW_LATENT_ARTIFACT = Object.freeze({{
  artifactId: "{NESTING34_ARTIFACT_ID}",
  sha256: "{sha}",
  robustnessFloor: {ts_number(floor)},
  draws: Object.freeze([
{lines}
  ] satisfies readonly ArtifactDraw[]),
  approval: "qualified_locked_campaign_20260801",
  sourceQualification: "locked_campaign_draw_latent_20260801",
}} as const);""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--emit-atoms17", action="store_true")
    parser.add_argument("--emit-nesting34", action="store_true")
    arguments = parser.parse_args()
    if not (arguments.validate or arguments.emit_atoms17
            or arguments.emit_nesting34):
        parser.error("pass --validate, --emit-atoms17 and/or --emit-nesting34")
    if arguments.validate:
        validate_ensemble9()
    if arguments.emit_atoms17:
        emit_atoms17()
    if arguments.emit_nesting34:
        emit_nesting34()


if __name__ == "__main__":
    sys.exit(main())
