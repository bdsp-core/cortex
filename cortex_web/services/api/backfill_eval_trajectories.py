"""One-time backfill: reconstruct the eval operating point for LEGACY cert
results stored before per-task persistence shipped.

A legacy result blob has only `verdicts` (no `perTask`), so the dashboard shows
ℓ as "—" and the per-domain evolution curves are empty. But the per-domain
posterior means survive in the blob's FINAL trial (`lMean`/`tMean`), reaction
times survive in the `trials` table, and ℓ* lives in the bundle manifest. From
those we rebuild a `perTask` block (so the mastery grid + metric row show real
ℓ/θ) and write the per-domain eval trajectory points (so the curves render).

NOT recoverable (left null): AUROC (the per-task ROC was never stored) and σ
(the ±band came from the particle cloud, which was never persisted).

Idempotent: a result that already has `perTask` is skipped, and
`write_eval_trajectory` replaces a session's eval rows, so re-running is safe.

Run from cortex_web/services (same cwd as `uvicorn api.app:app`):

    python -m api.backfill_eval_trajectories --manifest <path/to/manifest.json>          # dry run
    python -m api.backfill_eval_trajectories --manifest <path/to/manifest.json> --apply   # write
"""
from __future__ import annotations

import argparse
import json
from statistics import median
from typing import Optional

from .db import Database

# Fixed 7-domain ontology (engine task index → code/label), matching app.py.
CANONICAL = [(0, "spike", "Spike"), (1, "sz", "Seizure"), (2, "lpd", "LPD"),
             (3, "gpd", "GPD"), (4, "lrda", "LRDA"), (5, "grda", "GRDA"),
             (6, "iic", "Other")]


def load_ell_star(manifest_path: str) -> list[float]:
    """The per-domain Youden cut-scores (ℓ*) from the bundle manifest."""
    with open(manifest_path) as fh:
        m = json.load(fh)
    es = m.get("ellStar")
    if not isinstance(es, list):
        raise ValueError(f"manifest {manifest_path} has no ellStar list")
    return es


def _median_rt_by_task(trials: list[dict]) -> dict[int, float]:
    by: dict[int, list[float]] = {}
    for t in trials:
        tk, rm = t.get("task_k"), t.get("reaction_ms")
        if tk is not None and rm is not None:
            by.setdefault(int(tk), []).append(float(rm))
    return {k: median(v) for k, v in by.items()}


def reconstruct(result: dict, trials: list[dict],
                ell_star: list[float]) -> Optional[tuple[list[dict], list[dict]]]:
    """Rebuild (perTask, eval_points) from a legacy result's final trial +
    trials reaction times + ℓ*. Returns None if the blob can't be reconstructed
    (no final-trial lMean/tMean)."""
    tr = result.get("trials")
    if not isinstance(tr, list) or not tr:
        return None
    last = tr[-1]
    lmean, tmean = last.get("lMean"), last.get("tMean")
    if not (isinstance(lmean, list) and isinstance(tmean, list)):
        return None
    verdicts = result.get("verdicts") or []
    rts = _median_rt_by_task(trials)
    per_task, points = [], []
    for k, code, label in CANONICAL:
        if k >= len(lmean) or k >= len(tmean):
            break
        ell, theta = lmean[k], tmean[k]
        es = ell_star[k] if k < len(ell_star) else None
        verdict = verdicts[k] if k < len(verdicts) else "PENDING"
        per_task.append({"taskK": k, "code": code, "label": label,
                         "ell": ell, "theta": theta, "sd": None,
                         "ellStar": es, "auroc": None, "verdict": verdict})
        points.append({"taskK": k, "ell": ell, "theta": theta,
                       "sd": None, "rt": rts.get(k)})
    return per_task, points


def run(db: Database, ell_star: list[float], *, apply: bool = False) -> dict:
    """Backfill every completed result that lacks a perTask block. Returns a
    summary dict. With apply=False, computes but writes nothing (dry run)."""
    summary = {"scanned": 0, "already_ok": 0, "reconstructed": 0,
               "unrecoverable": 0, "sessions": []}
    for sess in db.all_sessions():
        sid, code = sess["session_id"], sess["code"]
        result = db.get_result(sid)
        if result is None:
            continue
        summary["scanned"] += 1
        if isinstance(result.get("perTask"), list):
            summary["already_ok"] += 1
            continue
        out = reconstruct(result, db.session_trials(sid), ell_star)
        if out is None:
            summary["unrecoverable"] += 1
            summary["sessions"].append({"session_id": sid, "status": "unrecoverable"})
            continue
        per_task, points = out
        summary["reconstructed"] += 1
        summary["sessions"].append({"session_id": sid, "code": code,
                                    "status": "reconstructed", "n_points": len(points)})
        if apply:
            db.store_result(sid, {**result, "perTask": per_task,
                                  "perTaskReconstructed": True})
            db.write_eval_trajectory(code, sid, points)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="bundle manifest.json (for ℓ*)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = ap.parse_args()

    ell_star = load_ell_star(args.manifest)
    db = Database()
    summary = run(db, ell_star, apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN (no writes)"
    print(f"=== eval-trajectory backfill — {mode} ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
