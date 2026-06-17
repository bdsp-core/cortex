"""Illustrative sample data for the dashboard/learning-protocol surfaces.

Per the migration decision (training mechanism = UI + tables, sample data), the
ℓ/θ/RT *training* trajectories and the protocol plan are SAMPLE data until the
Python trainer (scratch/training/pipeline_demo.py) is ported. The dashboard
endpoints serve these (flagged `sample: true`) when a participant has no real
training rows yet; real certification results (verdicts/AUROC) come from the
results table and are NOT faked. Numbers here are deterministic + illustrative,
mirroring the locked-v1 mockup shape (eval -> training -> re-cert): ℓ rises
toward ℓ*, θ drifts toward 0, RT falls.

The 7 fixed tasks (engine task index k): spike, sz, lpd, gpd, lrda, grda, iic.
"""
from __future__ import annotations

import time

# (taskK, code, label, ell_eval, ell_now, ell_star, theta_eval, rt_eval_ms, auroc, verdict)
_TASKS = [
    (0, "spike", "Spike",   0.62, 1.18, 0.24, 0.42, 2600, 0.91, "PASS"),
    (1, "sz",    "Seizure", 0.40, 0.72, 0.15, -0.55, 3100, 0.87, "PASS"),
    (2, "lpd",   "LPD",     0.55, 0.98, 0.31, 0.38, 2900, 0.83, "IN_TRAINING"),
    (3, "gpd",   "GPD",     0.48, 0.98, 0.26, 0.50, 3000, 0.85, "IN_TRAINING"),
    (4, "lrda",  "LRDA",    0.30, 0.66, 0.32, -0.40, 3300, 0.79, "IN_TRAINING"),
    (5, "grda",  "GRDA",    0.28, 0.58, 0.35, 0.30, 3400, 0.76, "FAIL"),
    (6, "iic",   "Other",   0.52, 1.10, 0.30, -0.28, 2800, 0.84, "PASS"),
]

_N_TRAIN = 7   # eval (1) + training (7) + re-cert (1) = 9 points


def tasks() -> list[dict]:
    """Per-task summary for the mastery grid (sample)."""
    out = []
    for (k, code, label, _e, ell, star, _t, _rt, auroc, verdict) in _TASKS:
        out.append({"taskK": k, "code": code, "label": label,
                    "ell": ell, "ellStar": star, "auroc": auroc,
                    "verdict": verdict})
    return out


def _series(start: float, end: float, n: int) -> list[float]:
    """A gently saturating ramp from start to end over n points."""
    out = []
    for i in range(n):
        f = i / (n - 1)
        f = 1 - (1 - f) ** 2          # ease-out so most gain is early
        out.append(round(start + (end - start) * f, 4))
    return out


def trajectories() -> list[dict]:
    """Flat list of trajectory points across eval -> training -> re-cert."""
    pts: list[dict] = []
    n = _N_TRAIN + 2
    base = time.time() - n * 86400
    for (k, _code, _label, e_ell, n_ell, star, t0, rt0, _au, _v) in _TASKS:
        ells = _series(e_ell, n_ell, n)
        thetas = _series(t0, t0 * 0.25, n)        # bias drifts toward 0
        rts = _series(rt0, rt0 * 0.72, n)         # reaction time falls
        sds = _series(0.22, 0.10, n)              # posterior tightens
        for i in range(n):
            phase = "eval" if i == 0 else ("recert" if i == n - 1 else "train")
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(base + i * 86400))
            pts.append({"taskK": k, "phase": phase, "ell": ells[i],
                        "theta": thetas[i], "sd": sds[i], "rt": rts[i], "ts": ts})
    return pts


def regimen_plan() -> dict:
    """Sample protocol: per-task target ℓ* + scheduled deck counts."""
    deck = []
    for (k, code, label, _e, ell, star, _t, _rt, _au, _v) in _TASKS:
        deck.append({"taskK": k, "code": code, "label": label,
                     "ell": ell, "ellStar": star,
                     "new": 12 - k, "learning": 8 - (k % 4), "due": 21 - k})
    return {"weeks": 6, "weekOf": 3, "deck": deck}


def kpis() -> dict:
    return {"streak": 18,
            "dueToday": {"new": 25, "learning": 26, "review": 9},
            "nextRecertDays": 24}
