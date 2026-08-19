#!/usr/bin/env python3
"""Create a domain-mapped, terminology-scrubbed copy of the CENTAUR novice labels.

Maps the IIIC pattern-class taxonomy (and its iterations) to neutral domain
tokens, per the canonical K7 ordering used in pipeline code
(K7_DOMAINS = [spike, sz, lpd, gpd, lrda, grda, iic]):

    spike            -> domain1
    seizure / sz     -> domain2
    lpd              -> domain3
    gpd              -> domain4
    lrda             -> domain5
    grda             -> domain6
    iic / other / bipd / birds -> domain7   (bipd,birds collapse to other per AUDIT s6)
"""
import csv
import re
import sys

SRC = "centaur_iiic_novice_labels.csv"
DST = "centaur_iiic_novice_labels.scrubbed.csv"

# token (lowercased) -> domain. Includes every iteration we expect.
LABEL_MAP = {
    "spike": "domain1", "spikes": "domain1", "ied": "domain1",
    "seizure": "domain2", "seizures": "domain2", "sz": "domain2",
    "lpd": "domain3", "lpds": "domain3",
    "gpd": "domain4", "gpds": "domain4",
    "lrda": "domain5",
    "grda": "domain6",
    "iic": "domain7", "other": "domain7", "bipd": "domain7",
    "bipds": "domain7", "birds": "domain7", "bird": "domain7",
}


def map_label(cell: str) -> str:
    """Map a User Label cell to its domain token, preserving surrounding quotes."""
    m = re.match(r"^(\s*'?)([A-Za-z/_-]+)('?\s*)$", cell)
    if not m:
        return cell
    pre, token, post = m.groups()
    key = token.lower()
    if key in LABEL_MAP:
        return f"{pre}{LABEL_MAP[key]}{post}"
    return cell


def main() -> int:
    with open(SRC, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    li = header.index("User Label")

    unmapped = set()
    for row in rows[1:]:
        if len(row) <= li:
            continue
        before = row[li]
        row[li] = map_label(before)
        if row[li] == before and re.search(r"[A-Za-z]", before):
            unmapped.add(before)

    # write with a leading newline-free UTF-8 (no BOM) output
    with open(DST, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"wrote {DST}: {len(rows) - 1} data rows")
    if unmapped:
        print("WARNING: User Label values left unmapped:", sorted(unmapped))
        return 1
    print("all alphabetic User Label values mapped to domain tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
