"""ilae-export — the de-identified CORTEX research export product (Phase R1).

Builds the gold layer (read-only), applies COHORT-AWARE disclosure control, and
writes a versioned, reproducible, de-identified dataset:

    <out>/tables/*.csv                 tidy per-grain tables
    <out>/per_participant/<sk>/*.json  the 3-bucket archival layout
    <out>/codebook.json                version-pinned data dictionary
    <out>/export_manifest.json         schema/commit/checksums/de-id policy
    <out>/DEIDENTIFICATION.md          human-readable disclosure record

Disclosure control (resolves plan C4): for a cohort below `--aggregate-below`,
individual-level quasi-identifiers are NEVER published — demographics are emitted
as single-variable marginal counts only (no cross-tabs, no per-participant rows).
Performance data (verdicts, SMC evolution) is published per pseudonym. For larger
cohorts, per-participant demographics are published with small-cell suppression
(k-anonymity).

Reproducible: identical inputs -> identical data files -> identical checksums
(only the manifest's export_utc varies, and it is not self-checksummed).

Usage:
    CORTEX_DB=postgresql://... GOLD_HMAC_SALT=<secret> \
        python -m research.export.export_cli --out ./pilot_export --export-utc 2026-06-19T00:00:00Z
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

try:  # package run
    from research.gold.build_gold import (
        DEMOGRAPHIC_ALLOWLIST, IDENTIFIER_FIELDS, _connect_ro, _loads,
        build_gold, read_operational,
    )
    from research.export.codebook import CODEBOOK, CODEBOOK_VERSION
except ModuleNotFoundError:  # flat run (files copied side-by-side)
    from build_gold import (  # type: ignore
        DEMOGRAPHIC_ALLOWLIST, IDENTIFIER_FIELDS, _connect_ro, _loads,
        build_gold, read_operational,
    )
    from codebook import CODEBOOK, CODEBOOK_VERSION  # type: ignore

SCHEMA_VERSION = "r1.0"

# Deterministic per-table sort keys (grain) so two runs are byte-identical.
SORT_KEYS = {
    "dim_participant": ["participant_sk"],
    "dim_consent": ["participant_sk"],
    "demographic_marginals": ["field", "value"],
    "fact_test_session": ["session_sk"],
    "fact_test_task_outcome": ["session_sk", "task_k"],
    "fact_trial": ["session_sk", "trial_index"],
    "fact_trial_task": ["session_sk", "trial_index", "task_k"],
    "fact_training_session": ["training_sk"],
    "fact_trajectory_point": ["participant_sk", "task_k", "ts"],
    "bridge_participant_journey": ["participant_sk"],
}

# Performance grains pass through unchanged (pseudonymous, no quasi-identifiers).
PERFORMANCE_TABLES = [
    "fact_test_session", "fact_test_task_outcome", "fact_trial", "fact_trial_task",
    "fact_training_session", "fact_trajectory_point", "bridge_participant_journey",
]


# ── disclosure control ────────────────────────────────────────────────
def disclosure_control(gold: dict, aggregate_below: int, k: int) -> tuple[dict, dict]:
    n = len(gold["dim_participant"])
    tables = {t: gold[t] for t in PERFORMANCE_TABLES}
    report: dict[str, Any] = {
        "cohort_n": n, "aggregate_below": aggregate_below, "k": k,
        "quasi_identifiers": list(DEMOGRAPHIC_ALLOWLIST),
        "consent_published": "marginals_only" if n < aggregate_below else "per_participant",
    }
    if n < aggregate_below:
        report["mode"] = "aggregate_only"
        report["rationale"] = (
            f"cohort n={n} < aggregate_below={aggregate_below}: individual-level "
            "quasi-identifiers and cross-tabulations are withheld; demographics are "
            "published as single-variable marginal counts only.")
        marginals = []
        for field in DEMOGRAPHIC_ALLOWLIST + ["consent_version", "irb_protocol_id"]:
            src = gold["dim_participant"] if field in DEMOGRAPHIC_ALLOWLIST else gold["dim_consent"]
            counts: dict[str, int] = {}
            for row in src:
                v = row.get(field)
                key = "(missing)" if v in (None, "") else str(v)
                counts[key] = counts.get(key, 0) + 1
            for value, cnt in counts.items():
                marginals.append({"field": field, "value": value, "n": cnt})
        tables["demographic_marginals"] = marginals
        report["suppressed"] = ["dim_participant (individual quasi-identifiers)", "all cross-tabulations"]
    else:
        report["mode"] = "k_anonymized"
        report["rationale"] = f"cohort n={n} >= aggregate_below: per-participant rows with cells <k={k} suppressed."
        kept, suppressed_cells = [], 0
        freq = {f: {} for f in DEMOGRAPHIC_ALLOWLIST}
        for row in gold["dim_participant"]:
            for f in DEMOGRAPHIC_ALLOWLIST:
                freq[f][row.get(f)] = freq[f].get(row.get(f), 0) + 1
        for row in gold["dim_participant"]:
            out = dict(row)
            for f in DEMOGRAPHIC_ALLOWLIST:
                if freq[f].get(row.get(f), 0) < k:
                    out[f] = "withheld_small_cell"
                    suppressed_cells += 1
            kept.append(out)
        tables["dim_participant"] = kept
        tables["dim_consent"] = gold["dim_consent"]
        report["suppressed_cells"] = suppressed_cells
    return tables, report


# ── deterministic writers ─────────────────────────────────────────────
def _sorted(table: str, rows: list[dict]) -> list[dict]:
    keys = SORT_KEYS.get(table, list(rows[0].keys()) if rows else [])
    return sorted(rows, key=lambda r: tuple(str(r.get(c)) for c in keys))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


def write_per_participant(tables: dict, outdir: Path) -> int:
    """3-bucket archival layout. Demographics omitted in aggregate_only mode."""
    by_sk_session: dict[str, dict] = {}
    for t in ["fact_test_session", "fact_test_task_outcome", "fact_trial",
              "fact_trial_task", "fact_training_session", "fact_trajectory_point"]:
        for r in tables.get(t, []):
            sk = r.get("participant_sk")
            if sk is None and "session_sk" in r:
                continue  # trial/outcome rows resolved via session below
            by_sk_session.setdefault(sk, {})
    # resolve session-grain → participant via fact_test_session
    sess_owner = {r["session_sk"]: r["participant_sk"] for r in tables.get("fact_test_session", [])}
    buckets: dict[str, dict] = {}
    for sk in {r["participant_sk"] for r in tables.get("bridge_participant_journey", [])}:
        buckets[sk] = {"testing": {"sessions": [], "task_outcomes": [], "trials": [], "trial_task": []},
                       "learning": {"training": [], "trajectory": []}}
    for r in tables.get("fact_test_session", []):
        buckets[r["participant_sk"]]["testing"]["sessions"].append(r)
    for t, key in [("fact_test_task_outcome", "task_outcomes"), ("fact_trial", "trials"),
                   ("fact_trial_task", "trial_task")]:
        for r in tables.get(t, []):
            sk = r.get("participant_sk") or sess_owner.get(r.get("session_sk"))
            if sk in buckets:
                buckets[sk]["testing"][key].append(r)
    for r in tables.get("fact_training_session", []):
        if r["participant_sk"] in buckets:
            buckets[r["participant_sk"]]["learning"]["training"].append(r)
    for r in tables.get("fact_trajectory_point", []):
        if r["participant_sk"] in buckets:
            buckets[r["participant_sk"]]["learning"]["trajectory"].append(r)
    n = 0
    for sk, b in sorted(buckets.items()):
        d = outdir / "per_participant" / str(sk)
        d.mkdir(parents=True, exist_ok=True)
        (d / "testing.json").write_text(json.dumps(b["testing"], indent=2, sort_keys=True, default=str))
        (d / "learning.json").write_text(json.dumps(b["learning"], indent=2, sort_keys=True, default=str))
        n += 1
    return n


# ── codebook coverage gate ────────────────────────────────────────────
def check_codebook(tables: dict) -> list[str]:
    missing = []
    for t, rows in tables.items():
        if not rows:
            continue
        cb = CODEBOOK.get(t, {})
        for col in rows[0].keys():
            if col not in cb:
                missing.append(f"{t}.{col}")
    return missing


def _code_commit() -> str:
    env = os.environ.get("CORTEX_CODE_COMMIT")
    if env:
        return env
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent).decode().strip()
    except Exception:
        return "unknown"


def export(raw: dict, salt: bytes, outdir: Path, export_utc: str,
           aggregate_below: int, k: int) -> dict:
    gold = build_gold(raw, salt)
    tables, disclosure = disclosure_control(gold, aggregate_below, k)

    missing = check_codebook(tables)
    if missing:
        raise SystemExit(f"codebook coverage gate FAILED — undocumented columns: {missing}")

    # anti-leak: no identifier value in any exported table
    identifiers: set[str] = set()
    for s in raw["sessions"]:
        p = _loads(s.get("participant")) or {}
        for f in IDENTIFIER_FIELDS:
            if p.get(f):
                identifiers.add(str(p[f]).strip().lower())
    for pp in raw["participants"]:
        for f in ("display_name", "email"):
            v = pp.get(f)
            if v:
                identifiers.add(str(v).split("@")[0].strip().lower())
    identifiers.discard("")

    files: dict[str, dict] = {}
    tdir = outdir / "tables"
    for t, rows in tables.items():
        rows = _sorted(t, rows)
        # anti-leak scan
        for r in rows:
            for v in r.values():
                if isinstance(v, str) and v.strip().lower() in identifiers:
                    raise SystemExit(f"anti-leak FAILED: identifier value surfaced in {t}")
        path = tdir / f"{t}.csv"
        write_csv(rows, path)
        files[f"tables/{t}.csv"] = {"sha256": _sha256(path), "bytes": path.stat().st_size, "rows": len(rows)}

    n_pp = write_per_participant(tables, outdir)

    # codebook
    cb_path = outdir / "codebook.json"
    cb_path.write_text(json.dumps({"version": CODEBOOK_VERSION, "tables": CODEBOOK},
                                  indent=2, sort_keys=True))
    files["codebook.json"] = {"sha256": _sha256(cb_path), "bytes": cb_path.stat().st_size}

    # DEIDENTIFICATION.md
    deid = outdir / "DEIDENTIFICATION.md"
    deid.write_text(
        f"# CORTEX export — de-identification record\n\n"
        f"- schema_version: {SCHEMA_VERSION}\n- cohort_n: {disclosure['cohort_n']}\n"
        f"- mode: **{disclosure['mode']}**\n- rationale: {disclosure['rationale']}\n\n"
        f"De-identification is by ALLOWLIST: direct identifiers (name, institution, "
        f"email, IP, internal code) are never selected into the dataset. Participant "
        f"identity is an HMAC pseudonym (`participant_sk`); the salt is sealed and not "
        f"published. Timestamps are generalised to month where used as an identifier.\n")
    files["DEIDENTIFICATION.md"] = {"sha256": _sha256(deid), "bytes": deid.stat().st_size}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "codebook_version": CODEBOOK_VERSION,
        "export_utc": export_utc,
        "code_commit": _code_commit(),
        "disclosure": disclosure,
        "row_counts": {t: len(rows) for t, rows in tables.items()},
        "per_participant_folders": n_pp,
        "files": dict(sorted(files.items())),
        "reproducible": True,
        "note": "Two runs over the same source produce identical data-file checksums; only export_utc varies.",
    }
    (outdir / "export_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="ilae-export — de-identified CORTEX export (R1)")
    ap.add_argument("--db", default=os.environ.get("CORTEX_DB", ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--salt", default=os.environ.get("GOLD_HMAC_SALT", "DEV-INSECURE-SALT"))
    ap.add_argument("--export-utc", default=os.environ.get("CORTEX_EXPORT_UTC", "1970-01-01T00:00:00Z"),
                    help="stamp recorded in the manifest (pass a fixed value for reproducible runs)")
    ap.add_argument("--aggregate-below", type=int, default=20,
                    help="cohorts below this size publish demographic marginals only (no individual quasi-identifiers)")
    ap.add_argument("--k", type=int, default=5, help="k-anonymity small-cell threshold for larger cohorts")
    args = ap.parse_args(argv)
    if not args.db:
        ap.error("no DB: set CORTEX_DB or pass --db")

    handle = _connect_ro(args.db)
    try:
        raw = read_operational(handle)
    finally:
        handle[1].close()

    manifest = export(raw, args.salt.encode("utf-8"), Path(args.out),
                      args.export_utc, args.aggregate_below, args.k)
    print(json.dumps({"out": args.out, "mode": manifest["disclosure"]["mode"],
                      "row_counts": manifest["row_counts"],
                      "n_files": len(manifest["files"]),
                      "per_participant_folders": manifest["per_participant_folders"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
