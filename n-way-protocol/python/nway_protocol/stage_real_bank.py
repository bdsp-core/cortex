"""Stage eligible IIIC signal axes for the isolated bank information audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


CLASSES = ("sz", "lpd", "gpd", "lrda", "grda", "iic")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.signals)
    required = [
        column
        for name in CLASSES
        for column in (f"s_mean_{name}", f"s_sd_{name}")
    ]
    vote_columns = [
        "votes_seizure", "votes_lpd", "votes_gpd", "votes_lrda",
        "votes_grda", "votes_other",
    ]
    vote_total = frame[vote_columns].fillna(0).sum(axis=1)
    eligible = frame.loc[vote_total >= 10].dropna(subset=required).copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "segment_index", "seg_id",
        *[f"s_mean_{k}" for k in range(7)],
        *[f"s_sd_{k}" for k in range(7)],
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for segment_index, row in enumerate(eligible.itertuples(index=False)):
            values = {"segment_index": segment_index, "seg_id": int(row.seg_id)}
            values["s_mean_0"] = float(getattr(row, "s_mean_spike"))
            values["s_sd_0"] = float(getattr(row, "s_sd_spike"))
            for k, name in enumerate(CLASSES, start=1):
                values[f"s_mean_{k}"] = float(getattr(row, f"s_mean_{name}"))
                values[f"s_sd_{k}"] = float(getattr(row, f"s_sd_{name}"))
            writer.writerow(values)
    provenance = {
        "schema_version": 1,
        "status": "staged_research_input",
        "source_sha256": _sha256(args.signals),
        "eligible_segments": len(eligible),
        "eligibility": "at least 10 IIIC votes and all six IIIC s_mean/s_sd coordinates finite",
    }
    args.output.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
