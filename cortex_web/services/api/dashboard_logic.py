"""Dashboard/history derivation logic — pure functions over stored results.

Everything here is deterministic data-shaping: no FastAPI, no Database
handles (callers pass rows in). The truth-map cache is module state, keyed by
bundle version/manifest path.
"""
from __future__ import annotations

import json
from statistics import median
from typing import Optional

from . import config
from .session_bank import SessionBank

# Result-blob keys that are NEVER sent to the dashboard/history listing
# surfaces. A stored result averages ~370 KB (measured in prod, 2026-07-01)
# and ~95% of it is the raw per-trial array + the ~700-int served-seg list;
# the listing UIs read only verdicts/roc/perTask + metadata, and per-question
# detail has its own lazy endpoint backed by the trials TABLE
# (GET /api/history/{id}/questions). The FULL blob still flows everywhere it
# is actually needed: the DB row itself, /api/admin/results/{id}, the backfill
# script, and the research export all bypass this.
_HEAVY_RESULT_KEYS = ("trials", "servedSegIds")


def training_enabled(cfg: dict, code: str, participant: Optional[dict]) -> bool:
    """Whether the training-protocol entry is available to this participant,
    per the `training_mode` flag ("all" | "cohort" | "off"). Pure: the caller
    passes the participant row (only consulted for cohort-allowlist matching).
    Default posture is "all" (every authed user), preserving the 2026-07-07
    live-for-everyone deploy."""
    mode = (cfg or {}).get("training_mode", "all")
    if mode == "off":
        return False
    if mode == "all":
        return True
    # cohort: match code / 9-digit public_id / email against the allowlist
    allow = (cfg or {}).get("training_allowlist") or frozenset()
    if not allow:
        return False
    if code.lower() in allow:
        return True
    if participant:
        pub = str(participant.get("public_id") or "").lower()
        email = str(participant.get("email") or "").lower()
        if pub in allow or email in allow:
            return True
    return False


def slim_result(result: dict) -> dict:
    """A listing-surface copy of a stored result, minus the per-question bulk."""
    return {k: v for k, v in result.items() if k not in _HEAVY_RESULT_KEYS}


# The fixed 7-task ontology (D5: spike + the 6 IIIC patterns; K=7). Used to
# render every task tile in canonical order, including for legacy results that
# stored only a bare `verdicts` list (pre-Step-1, no per-task ℓ/θ/AUROC).
CANONICAL_TASKS = [
    (0, "spike", "Spike"),
    (1, "sz",    "Seizure"),
    (2, "lpd",   "LPD"),
    (3, "gpd",   "GPD"),
    (4, "lrda",  "LRDA"),
    (5, "grda",  "GRDA"),
    (6, "iic",   "Other"),
]


def dashboard_tasks(result: dict, latest_traj: Optional[dict] = None) -> list[dict]:
    """Per-task mastery summary derived from a real cert result. Reads the
    persisted `perTask` block (real ℓ/θ/ℓ*/AUROC + verdict) when present; for a
    legacy result (verdicts only) ℓ/ℓ*/AUROC come back None and just the verdict
    is shown. Always returns all 7 canonical tasks in engine-index order.

    `latest_traj` (task_k → latest real trajectory point) overrides ℓ/θ with the
    most recent MEASUREMENT — a training/re-cert point once those exist, else the
    cert eval point (same value). ℓ* (the threshold), AUROC and verdict stay from
    the certification result."""
    latest_traj = latest_traj or {}
    per = result.get("perTask")
    by_k: dict[int, dict] = {}
    if isinstance(per, list):
        for p in per:
            if isinstance(p, dict) and p.get("taskK") is not None:
                by_k[int(p["taskK"])] = p
    verdicts = result.get("verdicts") or []
    out = []
    for k, code, label in CANONICAL_TASKS:
        p = by_k.get(k)
        lt = latest_traj.get(k)
        legacy_verdict = verdicts[k] if k < len(verdicts) else "PENDING"
        # Latest measurement wins for ℓ/θ; fall back to the cert perTask values.
        ell = p.get("ell") if p else None
        theta = p.get("theta") if p else None
        if lt is not None:
            if lt.get("ell") is not None:
                ell = lt["ell"]
            if lt.get("theta") is not None:
                theta = lt["theta"]
        if p is not None:
            out.append({
                "taskK": k,
                "code": p.get("code") or code,
                "label": p.get("label") or label,
                "ell": ell,
                "ellStar": p.get("ellStar"),
                "theta": theta,
                "auroc": p.get("auroc"),
                "verdict": p.get("verdict") or legacy_verdict,
            })
        else:
            out.append({
                "taskK": k, "code": code, "label": label,
                "ell": ell, "ellStar": None, "theta": theta, "auroc": None,
                "verdict": legacy_verdict,
            })
    return out


def eval_points_from_result(result: dict, trials: list[dict]) -> list[dict]:
    """Build the per-domain EVAL operating point from a finished cert result:
    ℓ/θ/σ from the persisted `perTask` block + the per-task MEDIAN reaction time
    from that session's trials. Returns [] for a legacy result with no perTask."""
    per = result.get("perTask")
    if not isinstance(per, list):
        return []
    rts_by_task: dict[int, list[float]] = {}
    for t in trials:
        tk, rm = t.get("task_k"), t.get("reaction_ms")
        if tk is not None and rm is not None:
            rts_by_task.setdefault(int(tk), []).append(float(rm))
    pts = []
    for p in per:
        if not isinstance(p, dict) or p.get("taskK") is None:
            continue
        k = int(p["taskK"])
        rts = rts_by_task.get(k)
        pts.append({"taskK": k, "ell": p.get("ell"), "theta": p.get("theta"),
                    "sd": p.get("sd"), "rt": (median(rts) if rts else None)})
    return pts


def dashboard_kpis(tasks: list[dict], last_assessed: Optional[str]) -> dict:
    """Real certification-summary KPIs (no learning-protocol data, which doesn't
    exist until the trainer ships): tasks certified, date last assessed, and the
    mean per-task AUROC over the tasks that have one."""
    aurocs = [t["auroc"] for t in tasks if isinstance(t.get("auroc"), (int, float))]
    certified = sum(1 for t in tasks if str(t.get("verdict") or "").upper() == "PASS")
    return {
        "tasksCertified": certified,
        "tasksTotal": len(tasks),
        "lastAssessed": last_assessed,
        "meanAuroc": round(sum(aurocs) / len(aurocs), 3) if aurocs else None,
    }


# ──────────────── per-question breakdown (history) ──────────────
# The bundle manifest is the ground-truth source: segId → patternClass, the
# per-task pattern words, and which task is the (binary) spike task.
# Truth maps cached PER bundle manifest path. The segId space differs across
# bundles (the 35k v1.6 segIds are absent from the 700-seg pilot bundles), so the
# per-question "correct answer" MUST be resolved against the bundle that actually
# produced the session — keyed by sessions.bundle_version (the O3 provenance
# stamp). A single global "first manifest" map mislabelled IIIC questions for
# every session not on that bundle.
_TRUTH_CACHE: dict[str, dict] = {}
_EMPTY_TRUTH = {"seg": {}, "words": [], "classes": [], "labels": []}


def _truth_map_for(version: Optional[str] = None,
                   bank: Optional[SessionBank] = None) -> dict:
    """{'seg': {segId: patternClass}, 'words', 'classes', 'labels'} for the
    bundle `version` (sessions.bundle_version). Falls back to the alphabetically
    -first bundle for legacy sessions stamped before provenance (the pilot
    bundles predate it, and that is the bundle they ran on).

    When the live SessionBank IS that bundle (every new session), build the map
    from its already-parsed segments instead of re-reading the ~35 MB manifest
    from disk — the re-parse was a transient few-hundred-MB spike on a small box."""
    if bank is not None and version and bank.version == version:
        key = f"bank:{version}"
        if key not in _TRUTH_CACHE:
            _TRUTH_CACHE[key] = {
                "seg": {int(s["segId"]): s.get("patternClass")
                        for s in bank.segments if "segId" in s},
                "words": bank.engine.get("taskPatternWords", []),
                "classes": bank.engine.get("taskClasses", []),
                "labels": bank.engine.get("taskLabels", []),
            }
        return _TRUTH_CACHE[key]
    path = None
    if version:
        p = config.BUNDLE_DIR / version / "manifest.json"
        if p.exists():
            path = p
    if path is None:                       # legacy / unknown → pilot bundle
        manifests = sorted(config.BUNDLE_DIR.glob("*/manifest.json"))
        path = manifests[0] if manifests else None
    if path is None:
        return _EMPTY_TRUTH
    key = str(path)
    if key not in _TRUTH_CACHE:
        try:
            m = json.loads(path.read_text())
            _TRUTH_CACHE[key] = {
                "seg": {int(s["segId"]): s.get("patternClass")
                        for s in m.get("segments", []) if "segId" in s},
                "words": m.get("taskPatternWords", []),
                "classes": m.get("taskClasses", []),
                "labels": m.get("taskLabels", []),
            }
        except Exception:
            _TRUTH_CACHE[key] = _EMPTY_TRUTH
    return _TRUTH_CACHE[key]


def question_breakdown(trials: list[dict], truth: dict) -> list[dict]:
    """Per-question rows for one session: the examinee's answer, the correct
    answer (spike → s>0; IIIC → segment class == task word), reaction time, the
    per-question contribution to skill-parameter uncertainty (ΔR, the increment
    in normalized info gain for the targeted domain), and the running posterior
    (ℓ/θ), pass-mass π, and cumulative R. Reconstructed from the stored diag, so
    legacy sessions work too."""
    seg, words, classes, labels = (truth["seg"], truth["words"],
                                   truth["classes"], truth["labels"])
    prev_R: dict[int, float] = {}
    out = []
    for row in trials:
        diag = row.get("diag")
        if isinstance(diag, str):
            try:
                diag = json.loads(diag)
            except Exception:
                diag = None
        k = row.get("task_k")
        k = int(k) if k is not None else (int(diag["taskK"]) if diag and diag.get("taskK") is not None else None)

        def _at(key, idx):
            v = diag.get(key) if diag else None
            return v[idx] if isinstance(v, list) and idx is not None and idx < len(v) else None

        # Answer + correct answer. Spike is binary (Yes/No). IIIC is a 6-way
        # classification: `pick` is the engine task index of the pattern the
        # examinee chose, so their answer is that pattern's label and the correct
        # answer is the segment's true pattern (NOT a Yes/No carried from spike).
        y = diag.get("y") if diag else None
        pick = row.get("pick")
        pick = int(pick) if pick is not None else None
        # Spike phase vs IIIC phase. Use the manifest's taskClasses when present;
        # otherwise fall back to the canonical spike index (0).
        is_spike = k is not None and (
            classes[k] == "spike" if (classes and k < len(classes)) else k == 0)
        answer = correct = None
        is_correct = None
        if is_spike:
            answer = ("Yes" if y == 1 else "No") if y is not None else None
            s = diag.get("s") if diag else None
            truth_yes = (s > 0) if isinstance(s, (int, float)) else None
            correct = ("Yes" if truth_yes else "No") if truth_yes is not None else None
            is_correct = None if (answer is None or correct is None) else (answer == correct)
        else:
            answer = labels[pick] if (pick is not None and 0 <= pick < len(labels)) else None
            pc = seg.get(int(row["seg_id"])) if row.get("seg_id") is not None else None
            if pc is not None:
                ci = words.index(pc) if pc in words else None
                correct = labels[ci] if (ci is not None and ci < len(labels)) else pc.upper()
            if pick is not None and 0 <= pick < len(words) and pc is not None:
                is_correct = (words[pick] == pc)
        R_k = _at("R", k)
        d_R = None
        if R_k is not None and k is not None:
            d_R = max(0.0, R_k - prev_R.get(k, 0.0))
            prev_R[k] = R_k
        out.append({
            "q": (row.get("trial_index", 0) or 0) + 1,
            "taskK": k,
            # The test phase, not the engine-probed sub-domain (which reads as if
            # it were the correct answer). The correct pattern is its own column.
            "domain": ("—" if k is None else ("Spike" if is_spike else "IIIC")),
            "answer": answer,
            "correct": correct,
            "isCorrect": is_correct,
            "rt": row.get("reaction_ms"),
            "deltaR": d_R,
            "R": R_k,
            "pi": _at("pi", k),
            "ell": _at("lMean", k),
            "theta": _at("tMean", k),
        })
    return out
