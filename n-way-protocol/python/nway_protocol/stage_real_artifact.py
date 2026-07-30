"""Stage leakage-controlled wrong-pick records from the CENTAUR source files.

All outputs are written to an explicit path under ``n-way-protocol``. Expert
reads use leave-one-expert-out plurality, avoiding self-inclusion in gold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


CLASSES = ("sz", "lpd", "gpd", "lrda", "grda", "iic")
CLASS_INDEX = {name: index + 1 for index, name in enumerate(CLASSES)}
LABEL_TO_CLASS = {
    "seizure": "sz", "seizures": "sz", "sz": "sz",
    "lpd": "lpd", "lpds": "lpd", "gpd": "gpd", "gpds": "gpd",
    "lrda": "lrda", "grda": "grda", "iic": "iic", "other": "iic",
    "bipd": "iic", "bipds": "iic", "birds": "iic", "bird": "iic",
}


def _to_class(cell) -> str | None:
    if not isinstance(cell, str):
        return None
    match = re.match(r"^\s*'?([A-Za-z/_-]+)'?\s*$", cell)
    return LABEL_TO_CLASS.get(match.group(1).lower()) if match else None


def _plurality(values: list[str | None]) -> str | None:
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None
    return ordered[0][0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(
    signals_path: Path,
    case_map_path: Path,
    expert_path: Path,
    novice_path: Path,
) -> tuple[list[dict], dict]:
    signals = pd.read_csv(signals_path)
    signal_columns = [f"s_mean_{name}" for name in CLASSES]
    signals = signals.dropna(subset=signal_columns)
    signal_lookup = {
        int(row.seg_id): tuple([0.0] + [float(getattr(row, column)) for column in signal_columns])
        for row in signals.itertuples()
    }
    case_map = pd.read_csv(case_map_path)
    segment_of = dict(zip(case_map["extset_case_id"], case_map["new_seg_id"], strict=True))
    expert = pd.read_excel(expert_path)
    expert_picks: dict[int, list[str | None]] = {}
    expert_rows: dict[int, object] = {}
    for row in expert.itertuples(index=False):
        case_id = int(row.case_id)
        expert_picks[case_id] = [
            _to_class(getattr(row, f"label_expert{index}")) for index in (1, 2, 3, 4)
        ]
        expert_rows[case_id] = row

    output: list[dict] = []
    # Expert validation is non-circular: each read is judged against only the
    # other three experts.
    for case_id, picks in expert_picks.items():
        segment_id = segment_of.get(case_id)
        if segment_id is None or int(segment_id) not in signal_lookup:
            continue
        for expert_index, pick in enumerate(picks, start=1):
            gold = _plurality([
                value for index, value in enumerate(picks, start=1)
                if index != expert_index
            ])
            if gold is None or pick is None or pick == gold:
                continue
            output.append({
                "reader_id": f"expert-{expert_index}",
                "source_id": f"case-{case_id}",
                "asked_k": CLASS_INDEX[gold],
                "raw_pick": CLASS_INDEX[pick],
                "group": "1|2|3|4|5|6",
                **{f"z_{k}": signal_lookup[int(segment_id)][k] for k in range(7)},
            })

    gold = {case_id: _plurality(picks) for case_id, picks in expert_picks.items()}
    novice = pd.read_csv(novice_path)
    novice = novice[novice["Labeling State"] == "Gold Standard"]
    for row in novice.itertuples(index=False):
        case_id = int(getattr(row, "_0"))  # pandas sanitizes ``Case ID``.
        user_id = str(getattr(row, "_1"))
        pick = _to_class(getattr(row, "_5"))
        expected = gold.get(case_id)
        segment_id = segment_of.get(case_id)
        if (
            expected is None or pick is None or pick == expected
            or segment_id is None or int(segment_id) not in signal_lookup
        ):
            continue
        output.append({
            "reader_id": f"novice-{user_id}",
            "source_id": f"case-{case_id}",
            "asked_k": CLASS_INDEX[expected],
            "raw_pick": CLASS_INDEX[pick],
            "group": "1|2|3|4|5|6",
            **{f"z_{k}": signal_lookup[int(segment_id)][k] for k in range(7)},
        })
    provenance = {
        "schema_version": 1,
        "status": "staged_research_input",
        "expert_gold": "leave_one_expert_out_for_expert_reads; full_expert_plurality_for_novices",
        "source_group": "external_case_id",
        "n_wrong_picks": len(output),
        "source_sha256": {
            str(path): _sha256(path)
            for path in (signals_path, case_map_path, expert_path, novice_path)
        },
    }
    return output, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--case-map", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--novice", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, provenance = stage(args.signals, args.case_map, args.expert, args.novice)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "reader_id", "source_id", "asked_k", "raw_pick", "group",
        *[f"z_{k}" for k in range(7)],
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
