"""Categorical state-coverage audit of the served bank.

The Fisher audit (bank_audit.py) prices information; this tool answers the
coverage question underneath it: for every ordered confusion state
(asked class a → distractor c), does the served bank contain enough items
whose model-predicted distractor allocation actually supports that state?
Flat allocations mean the identity channel adds nothing on that item; thin
(a → c) cells mean a confusion state the categorical layer can rarely
exercise.  Reported at the unfloored likelihood and at the candidate λ_d
floors, so the floor's cost to the identity channel is explicit.
Audit-only: no bank mutation, no eligibility change.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .misspec_monitor import load_axes

IIIC_GROUP = (1, 2, 3, 4, 5, 6)
CLASS_NAMES = {1: "sz", 2: "lpd", 3: "gpd", 4: "lrda", 5: "grda", 6: "iic"}
DEFAULT_FLOORS = (0.0, 0.15, 0.20)
INFORMATIVE_KL_NATS = 0.05
STRONG_SUPPORT_PROBABILITY = 0.30


def allocation(
    axes: np.ndarray, asked_k: int, beta: float, lapse: float,
) -> tuple[list[int], np.ndarray]:
    """Model distractor allocation q per item for one asked class."""
    distractors = [k for k in IIIC_GROUP if k != asked_k]
    logits = beta * axes[:, distractors]
    logits = logits - logits.max(axis=1, keepdims=True)
    softmax = np.exp(logits)
    softmax /= softmax.sum(axis=1, keepdims=True)
    return distractors, lapse / len(distractors) + (1 - lapse) * softmax


def coverage(
    axes: np.ndarray,
    beta: float = 0.9912,
    floors: tuple[float, ...] = DEFAULT_FLOORS,
) -> dict:
    n_items = axes.shape[0]
    by_floor = {}
    for lapse in floors:
        asked_rows = {}
        pair_rows = {}
        for asked_k in IIIC_GROUP:
            distractors, q = allocation(axes, asked_k, beta, lapse)
            kl_uniform = np.sum(
                q * np.log(np.maximum(q * len(distractors), 1e-300)), axis=1,
            )
            asked_rows[CLASS_NAMES[asked_k]] = {
                "items": n_items,
                "median_kl_vs_uniform_nats": float(np.median(kl_uniform)),
                "informative_items": int(np.sum(kl_uniform >= INFORMATIVE_KL_NATS)),
                "informative_fraction": float(np.mean(kl_uniform >= INFORMATIVE_KL_NATS)),
            }
            modal = np.asarray(distractors)[np.argmax(q, axis=1)]
            for position, c in enumerate(distractors):
                pair_rows[f"{CLASS_NAMES[asked_k]}->{CLASS_NAMES[c]}"] = {
                    "modal_items": int(np.sum(modal == c)),
                    "strong_support_items": int(
                        np.sum(q[:, position] >= STRONG_SUPPORT_PROBABILITY)
                    ),
                }
        empty_modal = [pair for pair, row in pair_rows.items() if row["modal_items"] == 0]
        thin_strong = {
            pair: row["strong_support_items"]
            for pair, row in pair_rows.items()
            if row["strong_support_items"] < 100
        }
        by_floor[str(lapse)] = {
            "by_asked_class": asked_rows,
            "by_confusion_pair": pair_rows,
            "empty_modal_pairs": empty_modal,
            "pairs_below_100_strong_support": thin_strong,
        }
    return {
        "schema_version": 1,
        "status": "audit_only_no_bank_mutation",
        "config": {
            "beta": beta,
            "floors": list(floors),
            "informative_kl_nats": INFORMATIVE_KL_NATS,
            "strong_support_probability": STRONG_SUPPORT_PROBABILITY,
            "items": n_items,
        },
        "by_floor": by_floor,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("axes", type=Path)
    parser.add_argument("--beta", type=float, default=0.9912)
    parser.add_argument("--floors", type=float, nargs="+", default=DEFAULT_FLOORS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    axes = load_axes(args.axes)
    result = coverage(axes, beta=args.beta, floors=tuple(args.floors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    unfloored = result["by_floor"][str(args.floors[0])]
    print(json.dumps({
        "items": result["config"]["items"],
        "informative_items_by_class": {
            name: row["informative_items"]
            for name, row in unfloored["by_asked_class"].items()
        },
        "empty_modal_pairs": unfloored["empty_modal_pairs"],
        "pairs_below_100_strong_support": unfloored["pairs_below_100_strong_support"],
    }, indent=2))


if __name__ == "__main__":
    main()
