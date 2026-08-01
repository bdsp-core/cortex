#!/usr/bin/env python3
"""Phase-2 qualified re-emission of the nesting34 draw-latent artifact.

Runs ONLY after the amended locked campaign passes. Reads the research
emission (`iiic_conditional_f1_engine_frame_nesting34_rd.json`), verifies
the campaign gate report says `campaign_pass: true`, and writes a NEW file
(`..._nesting34_qualified.json`; the research artifact stays in place)
with `qualification: "qualified"`, `promotionForbidden: false`, and
provenance chained to the campaign cell reports, the gate verdict, the
real-response arbitration, and the DR07 gate digest already recorded in
the fit lineage. The draws are byte-identical to the research emission —
qualification changes governance metadata, never numbers.

Usage (from n-way-protocol/):
    PYTHONPATH="python:." python3 scripts/qualify_nesting34_artifact.py
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/iiic_conditional_f1_engine_frame_nesting34_rd.json"
OUTPUT = ROOT / "artifacts/iiic_conditional_f1_engine_frame_nesting34_qualified.json"
GATES = ROOT / "draw_latent_rd/reports/campaign_draw_latent/GATE_VERDICT.json"
CELLS = {
    "cell1_continuum_sbc": ROOT / "draw_latent_rd/reports/campaign_draw_latent/cell1_continuum_2000.json",
    "cell2_served_bank_adaptive": ROOT / "draw_latent_rd/reports/campaign_draw_latent/cell2_precision_2000.json",
    "cell3_stress190": ROOT / "draw_latent_rd/reports/campaign_draw_latent/cell3_stress190_500.json",
    "cell4_uniform_collapse": ROOT / "draw_latent_rd/reports/campaign_draw_latent/cell4_collapse_500.json",
}
ARBITRATION = ROOT / "draw_latent_rd/reports/real_response_arbitration.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    verdict = json.loads(GATES.read_text())
    if verdict.get("campaign_pass") is not True:
        raise SystemExit(
            "REFUSED: the locked campaign gate report does not record a pass; "
            "the promotion is parked and no qualified artifact may be emitted."
        )
    payload = copy.deepcopy(json.loads(SOURCE.read_text()))
    payload["qualification"] = "qualified"
    payload["promotionForbidden"] = False
    payload["status"] = "qualified_for_serving"
    provenance = payload.setdefault("provenance", {})
    provenance["qualified_by"] = {
        "prereg": "docs/LOCKED_CAMPAIGN_PREREG_DRAW_LATENT.md",
        "gate_verdict": str(GATES.relative_to(ROOT)),
        "gate_verdict_sha256": _sha(GATES),
        "cells": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": _sha(path)}
            for name, path in CELLS.items()
        },
        "real_response_arbitration": str(ARBITRATION.relative_to(ROOT)),
        "real_response_arbitration_sha256": _sha(ARBITRATION),
        "research_emission_sha256": _sha(SOURCE),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": OUTPUT.name,
        "qualification": payload["qualification"],
        "promotionForbidden": payload["promotionForbidden"],
        "sha256": _sha(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
