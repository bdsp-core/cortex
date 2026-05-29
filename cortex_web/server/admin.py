"""Admin CLI for the CORTEX web backend — credentials + results export.

Talks to the SQLite DB directly (no running server needed), so it's the tool
you use to mint participant codes before a study and pull results after.

Usage (from cortex_web/):
    python -m server.admin gen --count 100 --prefix cortex --out codes.csv
    python -m server.admin add --code cortex-alice --password s3cret
    python -m server.admin list
    python -m server.admin disable --code cortex-alice
    python -m server.admin export-sessions --out sessions.csv
    python -m server.admin export-results  --out results/        # one JSON each

The `gen` command is the bulk path: it writes a CSV of (code,password) you
distribute to participants. Passwords are shown ONCE — they're stored hashed.
"""
from __future__ import annotations

import argparse
import csv
import json
import secrets
import sys
from pathlib import Path

from .db import Database
from .security import hash_password


def _db(args) -> Database:
    return Database(args.db)


def cmd_gen(args) -> int:
    db = _db(args)
    rows = []
    for _ in range(args.count):
        code = f"{args.prefix}-{secrets.token_hex(4)}"
        password = secrets.token_urlsafe(9)
        db.add_participant(code, hash_password(password), args.label)
        rows.append({"code": code, "password": password})
    if args.out:
        out = Path(args.out)
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["code", "password"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} credentials to {out}")
    else:
        for r in rows:
            print(f"{r['code']}\t{r['password']}")
    return 0


def cmd_add(args) -> int:
    db = _db(args)
    db.add_participant(args.code, hash_password(args.password), args.label)
    print(f"added {args.code}")
    return 0


def cmd_list(args) -> int:
    db = _db(args)
    for r in db.list_participants():
        flag = "" if r["active"] else "  (disabled)"
        print(f"{r['code']}\t{r['created_utc']}\t{r['label']}{flag}")
    return 0


def cmd_disable(args) -> int:
    db = _db(args)
    db.set_participant_active(args.code, False)
    print(f"disabled {args.code}")
    return 0


def cmd_enable(args) -> int:
    db = _db(args)
    db.set_participant_active(args.code, True)
    print(f"enabled {args.code}")
    return 0


def cmd_export_sessions(args) -> int:
    db = _db(args)
    sessions = [dict(r) for r in db.all_sessions()]
    if not sessions:
        print("no sessions yet")
        return 0
    fields = list(sessions[0].keys())
    out = Path(args.out) if args.out else None
    fh = out.open("w", newline="") if out else sys.stdout
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    w.writerows(sessions)
    if out:
        fh.close()
        print(f"wrote {len(sessions)} sessions to {out}")
    return 0


def cmd_export_results(args) -> int:
    db = _db(args)
    outdir = Path(args.out or "results")
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in db.all_sessions():
        sid = r["session_id"]
        res = db.get_result(sid)
        if res is None:
            continue
        payload = {
            "session": dict(r),
            "result": res,
            "trials": [dict(t) for t in db.session_trials(sid)],
        }
        (outdir / f"{sid}.json").write_text(json.dumps(payload, indent=2))
        n += 1
    print(f"wrote {n} result files to {outdir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CORTEX web admin")
    p.add_argument("--db", default=None, help="SQLite path (default server/cortex.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="bulk-generate credentials")
    g.add_argument("--count", type=int, required=True)
    g.add_argument("--prefix", default="cortex")
    g.add_argument("--label", default="")
    g.add_argument("--out", default=None, help="CSV path (else stdout)")
    g.set_defaults(func=cmd_gen)

    a = sub.add_parser("add", help="add one credential")
    a.add_argument("--code", required=True)
    a.add_argument("--password", required=True)
    a.add_argument("--label", default="")
    a.set_defaults(func=cmd_add)

    l = sub.add_parser("list", help="list participants")
    l.set_defaults(func=cmd_list)

    d = sub.add_parser("disable", help="disable a code")
    d.add_argument("--code", required=True)
    d.set_defaults(func=cmd_disable)

    e = sub.add_parser("enable", help="re-enable a code")
    e.add_argument("--code", required=True)
    e.set_defaults(func=cmd_enable)

    es = sub.add_parser("export-sessions", help="export sessions to CSV")
    es.add_argument("--out", default=None)
    es.set_defaults(func=cmd_export_sessions)

    er = sub.add_parser("export-results", help="export per-session result JSON")
    er.add_argument("--out", default=None)
    er.set_defaults(func=cmd_export_results)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
