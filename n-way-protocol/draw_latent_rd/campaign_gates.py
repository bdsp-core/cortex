"""Evaluate the locked-campaign gates for the draw-latent promotion.

Reads the four cell summaries (and the gating cells' per-seed rows) produced
by the amended locked campaign (docs/LOCKED_CAMPAIGN_PREREG_DRAW_LATENT.md)
and renders every pre-registered gate with an explicit PASS/FAIL:

Gating cells (1 continuum SBC, 2 served-bank adaptive) must BOTH pass:
- absolute coverage: skill AND bias Wilson-95 containing or above 0.95;
- paired noninferiority vs binary: coverage difference >= -0.03, point AND
  confidence (the owner house margin);
- tightening: per-seed skill width ratio nway/binary, mean < 1.0 with the
  seed-cluster 95% CI excluding 1.0.

Cells 3 (beta = 1.903 stress) and 4 (uniform collapse) are informational —
reported, never gated, per the pre-registration.

Fail any gate -> the campaign FAILS and the promotion parks (no iteration).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

NOMINAL = 0.95
MARGIN = -0.03


def _rows(path: Path) -> tuple[dict, list[dict]]:
    lines = path.read_text().splitlines()
    header = json.loads(lines[0])
    return header, [json.loads(line) for line in lines[1:]]


def _tightening(rows: list[dict]) -> dict:
    by_seed: dict[int, dict[str, dict]] = {}
    for row in rows:
        if "arm" not in row:
            continue
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    ratios = [
        pair["categorical_f1"]["skill_width"] / pair["binary"]["skill_width"]
        for pair in by_seed.values()
        if "categorical_f1" in pair and "binary" in pair
    ]
    array = np.asarray(ratios)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / math.sqrt(array.size))
    ci = [mean - 1.96 * se, mean + 1.96 * se]
    return {
        "mean_ratio": mean,
        "cluster95": ci,
        "n_seeds": int(array.size),
        "pass": bool(mean < 1.0 and ci[1] < 1.0),
    }


def _coverage_gate(summary: dict, parameter: str) -> dict:
    block = summary["arms"]["categorical_f1"][f"{parameter}_coverage_uncertainty"]
    wilson = block["wilson95"]
    return {
        "coverage": summary["arms"]["categorical_f1"][f"{parameter}_coverage"],
        "wilson95": wilson,
        "pass": bool(wilson[0] <= NOMINAL <= wilson[1] or wilson[0] > NOMINAL),
    }


def _noninferiority_gate(summary: dict, parameter: str) -> dict:
    block = summary["paired_comparisons"][f"{parameter}_coverage_nway_minus_binary"]
    return {
        "mean": block["mean"],
        "ci95": block["ci95"],
        "point_pass": bool(block["mean"] >= MARGIN),
        "confidence_pass": bool(block["ci95"][0] >= MARGIN),
        "pass": bool(block["mean"] >= MARGIN and block["ci95"][0] >= MARGIN),
    }


def evaluate_gating_cell(summary_path: Path, rows_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    _, rows = _rows(rows_path)
    gates = {
        "absolute_skill_coverage": _coverage_gate(summary, "skill"),
        "absolute_bias_coverage": _coverage_gate(summary, "bias"),
        "noninferiority_skill": _noninferiority_gate(summary, "skill"),
        "noninferiority_bias": _noninferiority_gate(summary, "bias"),
        "tightening_skill_width": _tightening(rows),
    }
    reported = {
        "questions": {
            arm: summary["arms"][arm]["questions"] for arm in summary["arms"]
        },
        "skill_rmse": {
            arm: summary["arms"][arm]["skill_rmse"] for arm in summary["arms"]
        },
        "bias_rmse": {
            arm: summary["arms"][arm]["bias_rmse"] for arm in summary["arms"]
        },
        "skill_sbc_ks": {
            arm: summary["arms"][arm]["skill_sbc"]["ks_distance_from_uniform"]
            for arm in summary["arms"]
        },
        "bias_sbc_ks": {
            arm: summary["arms"][arm]["bias_sbc"]["ks_distance_from_uniform"]
            for arm in summary["arms"]
        },
        "mean_acceptance": {
            arm: summary["arms"][arm]["mean_acceptance"] for arm in summary["arms"]
        },
        "mean_ancestry": {
            arm: summary["arms"][arm]["mean_ancestry"] for arm in summary["arms"]
        },
        "draw_latent_diagnostics": summary.get("draw_latent_diagnostics"),
    }
    return {
        "cell": summary.get("cell"),
        "gates": gates,
        "pass": all(gate["pass"] for gate in gates.values()),
        "reported": reported,
    }


def informational_cell(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    return {
        "cell": summary.get("cell"),
        "gating": False,
        "nway_skill_coverage": summary["arms"]["categorical_f1"]["skill_coverage"],
        "nway_skill_wilson95": summary["arms"]["categorical_f1"][
            "skill_coverage_uncertainty"]["wilson95"],
        "binary_skill_coverage": summary["arms"]["binary"]["skill_coverage"],
        "draw_latent_diagnostics": summary.get("draw_latent_diagnostics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell1-summary", type=Path, required=True)
    parser.add_argument("--cell1-rows", type=Path, required=True)
    parser.add_argument("--cell2-summary", type=Path, required=True)
    parser.add_argument("--cell2-rows", type=Path, required=True)
    parser.add_argument("--cell3-summary", type=Path, required=True)
    parser.add_argument("--cell4-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cell1 = evaluate_gating_cell(args.cell1_summary, args.cell1_rows)
    cell2 = evaluate_gating_cell(args.cell2_summary, args.cell2_rows)
    verdict = {
        "schema_version": 1,
        "prereg": "docs/LOCKED_CAMPAIGN_PREREG_DRAW_LATENT.md",
        "cell1_continuum_sbc": cell1,
        "cell2_served_bank_adaptive": cell2,
        "cell3_stress190": informational_cell(args.cell3_summary),
        "cell4_uniform_collapse": informational_cell(args.cell4_summary),
        "campaign_pass": bool(cell1["pass"] and cell2["pass"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "campaign_pass": verdict["campaign_pass"],
        "cell1_gates": {k: v["pass"] for k, v in cell1["gates"].items()},
        "cell2_gates": {k: v["pass"] for k, v in cell2["gates"].items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
