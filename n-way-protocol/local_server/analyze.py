#!/usr/bin/env python3
"""Per-session and cross-session report over the local prod-layout SQLite DB.

python3 stdlib only. Reads the production-shaped `sessions` + `trials` tables
written by the local test server (see local_server/schema.sql) and prints,
per session: per-domain burden and accuracy, reaction-time p50/p90 over
manual (browser-timed) trials, the stop reason, and an atom-posterior
trajectory summary (posterior-mean beta after trials 10/25/50/end plus the
final MAP atom) — then a cross-session table. Because the layout is the
production layout, this same script runs against real prod tables unchanged.

Usage:
  python3 local_server/analyze.py [local_server/data/local_test.db]
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

TASK_LABELS = ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "IIC/Other"]
CHECKPOINTS = (10, 25, 50)
RULE = "=" * 72


def percentile(values: list[float], q: float) -> float:
    """NumPy-default linear-interpolation quantile."""
    ordered = sorted(values)
    h = (len(ordered) - 1) * q
    lo, hi = int(h), min(int(h) + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def parse_diag(row) -> dict:
    return json.loads(row["diag"]) if row["diag"] else {}


def session_report(session, trials) -> dict:
    print(RULE)
    print(f"session {session['session_id']}  "
          f"({session['participant']}, {session['status']})")
    print(f"  started {session['started_utc']}  "
          f"finished {session['finished_utc'] or '-'}  "
          f"seed {session['sample_seed']}")
    print(f"  bundle {session['bundle_version']}  "
          f"policy {session['termination_policy']}  "
          f"stop: {session['stop_reason'] or '-'}  "
          f"n_questions {session['n_questions']}")

    final_diag = parse_diag(trials[-1]) if trials else {}
    statuses = final_diag.get("precision_statuses", [])

    print("  domain burden / accuracy:")
    print(f"    {'domain':<10} {'n':>4} {'correct':>8} {'acc':>7}  final status")
    total_n = total_correct = 0
    for k in range(len(TASK_LABELS)):
        rows = [t for t in trials if t["task_k"] == k]
        if not rows and k == 0:
            continue  # spike is never askable on the IIIC-only bank
        n = len(rows)
        n_correct = sum(1 for t in rows if t["is_correct"])
        total_n += n
        total_correct += n_correct
        acc = f"{n_correct / n:7.3f}" if n else f"{'-':>7}"
        status = statuses[k] if k < len(statuses) else "-"
        print(f"    {TASK_LABELS[k]:<10} {n:>4} {n_correct:>8} {acc}  {status}")
    overall = total_correct / total_n if total_n else float("nan")
    print(f"    {'total':<10} {total_n:>4} {total_correct:>8} {overall:7.3f}")

    reaction = [t["reaction_ms"] for t in trials if t["reaction_ms"] is not None]
    if reaction:
        print(f"  reaction time (manual trials, n={len(reaction)}): "
              f"p50 {percentile(reaction, 0.50):.0f} ms  "
              f"p90 {percentile(reaction, 0.90):.0f} ms")
    else:
        print("  reaction time (manual trials): none "
              "(all trials simulated; reaction_ms NULL)")

    mean_beta_end = float("nan")
    if trials:
        print("  atom posterior trajectory (posterior-mean beta):")
        for checkpoint in CHECKPOINTS:
            if checkpoint > len(trials):
                continue
            diag = parse_diag(trials[checkpoint - 1])
            print(f"    after trial {checkpoint:>3}: "
                  f"{diag.get('atom_mean_beta', float('nan')):.3f}")
        mean_beta_end = final_diag.get("atom_mean_beta", float("nan"))
        print(f"    after trial {len(trials):>3} (end): {mean_beta_end:.3f}")
        mass = final_diag.get("atom_posterior", [])
        if mass:
            map_index = mass.index(max(mass))
            print(f"    final MAP atom: index {map_index} "
                  f"(beta {final_diag.get('atom_map_beta', float('nan')):.3f}, "
                  f"mass {100 * mass[map_index]:.1f}%)")

    return {
        "session_id": session["session_id"],
        "participant": session["participant"],
        "status": session["status"],
        "n_questions": session["n_questions"]
        if session["n_questions"] is not None else len(trials),
        "stop_reason": session["stop_reason"] or "-",
        "accuracy": overall,
        "mean_beta_end": mean_beta_end,
    }


def main() -> int:
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        Path(__file__).resolve().parent / "data" / "local_test.db")
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sessions = conn.execute(
            "SELECT * FROM sessions ORDER BY started_utc, session_id"
        ).fetchall()
        if not sessions:
            print(f"{db_path}: no sessions")
            return 0
        print(f"{db_path}: {len(sessions)} session(s)")
        summaries = []
        for session in sessions:
            trials = conn.execute(
                "SELECT * FROM trials WHERE session_id=? ORDER BY trial_index",
                (session["session_id"],)).fetchall()
            summaries.append(session_report(session, trials))
    finally:
        conn.close()

    print(RULE)
    print("cross-session:")
    print(f"  {'session_id':<26} {'participant':<18} {'status':<10} "
          f"{'n_q':>4} {'acc':>6} {'beta_end':>8}  stop_reason")
    for s in summaries:
        print(f"  {s['session_id']:<26} {s['participant']:<18} "
              f"{s['status']:<10} {s['n_questions']:>4} "
              f"{s['accuracy']:>6.3f} {s['mean_beta_end']:>8.3f}  "
              f"{s['stop_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
