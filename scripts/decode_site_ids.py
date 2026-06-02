"""Decode CORTEX de-identified `site_id` -> institution name.

`site_id` is a SALTED HASH (cortex_storage._site_id), NOT encryption — it cannot
be reversed mathematically. But it is DETERMINISTIC: the same institution maps to
the same site_id on every machine (the salt ships in the app). So you decode it
with a lookup built from the institution NAMES, which live LOCALLY in the
per-machine `registrations.csv` (PHI, never uploaded). Collect those
registrations.csv files from the test machines, point this at them, and it builds
the {site_id: institution} table and (optionally) annotates a de-identified
results CSV.

    # build the lookup from one or more local registrations.csv
    .venv/bin/python scripts/decode_site_ids.py --registrations reg1.csv reg2.csv
    # also add an institution column to a downloaded de-identified summary
    .venv/bin/python scripts/decode_site_ids.py --registrations reg.csv \
        --annotate results_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cortex_storage import _site_id  # noqa: E402  (consistent with the app coding)


def build_lookup(reg_paths):
    """{site_id: institution} from the institution names in registrations.csv(s)."""
    inst = set()
    for p in reg_paths:
        with open(p, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                v = (row.get("institution") or "").strip()
                if v:
                    inst.add(v)
    return {_site_id(i): i for i in sorted(inst)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registrations", nargs="+", required=True,
                    help="local registrations.csv file(s) with an institution column")
    ap.add_argument("--annotate",
                    help="de-id results CSV (with a site_id column) -> writes a "
                         ".decoded.csv with an added institution column")
    ap.add_argument("--out", help="write the lookup CSV here (default: stdout)")
    args = ap.parse_args()

    lut = build_lookup(args.registrations)
    rows = [{"site_id": s, "institution": i} for s, i in sorted(lut.items())]
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["site_id", "institution"])
            w.writeheader(); w.writerows(rows)
        print(f"wrote {len(rows)} site_id mappings to {args.out}")
    else:
        print("site_id,institution")
        for r in rows:
            print(f"{r['site_id']},{r['institution']}")

    if args.annotate:
        src = list(csv.DictReader(open(args.annotate, encoding="utf-8")))
        if not src or "site_id" not in src[0]:
            print(f"--annotate: {args.annotate} has no site_id column"); return
        fields = list(src[0].keys())
        if "institution" not in fields:
            fields.append("institution")
        outp = Path(args.annotate).with_suffix(".decoded.csv")
        matched = 0
        with open(outp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
            for row in src:
                sid = row.get("site_id", "")
                row["institution"] = lut.get(sid, "<unknown site>")
                matched += sid in lut
                w.writerow(row)
        print(f"annotated -> {outp} ({matched}/{len(src)} site_ids matched)")


if __name__ == "__main__":
    main()
