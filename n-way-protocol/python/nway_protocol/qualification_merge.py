"""Merge sharded qualification row files into one exact summary.

Shards are produced by ``qualification.py --rows-output`` run with disjoint
``--seed-base`` ranges (for example split across hosts). The merge rebuilds
the replicate rows and calls the same ``summarize`` the single-host path
uses, so the merged report is statistically identical to one unsharded run
over the union of seeds — CRN arm pairing lives inside each replicate and
is unaffected by sharding.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .qualification import QualificationConfig, ReplicateResult, summarize


def load_shard(path: Path) -> tuple[dict, list[ReplicateResult]]:
    with path.open() as handle:
        header = json.loads(handle.readline())
        rows = [ReplicateResult(**json.loads(line)) for line in handle if line.strip()]
    return header, rows


def merge(paths: list[Path]) -> dict:
    headers, all_rows = [], []
    for path in paths:
        header, rows = load_shard(path)
        headers.append(header)
        all_rows.extend(rows)

    def comparable(header: dict) -> dict:
        return {
            key: value for key, value in header["config"].items()
            if key != "replicates"
        }

    reference = comparable(headers[0])
    for path, header in zip(paths, headers, strict=True):
        if comparable(header) != reference:
            raise ValueError(f"shard config mismatch: {path}")
    seeds = [row.seed for row in all_rows if row.arm == "binary"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("shard seed ranges overlap")

    config_fields = dict(headers[0]["config"])
    config_fields["artifact_draws"] = tuple(
        tuple(draw) for draw in config_fields["artifact_draws"]
    )
    config_fields["replicates"] = len(seeds)
    config = QualificationConfig(**config_fields)
    summary = summarize(all_rows, config)
    summary["config"] = asdict(replace(config))
    summary["shards"] = [
        {
            "path": str(path),
            "seed_base": header["seed_base"],
            "replicates": header["config"]["replicates"],
        }
        for path, header in zip(paths, headers, strict=True)
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = merge(args.shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    args.output.write_text(rendered + "\n")
    print(json.dumps({
        "replicates": summary["config"]["replicates"],
        "shards": summary["shards"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
