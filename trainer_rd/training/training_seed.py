"""Step 1 — `TrainingSeed`: the eval→trainer handoff artifact.

This is the seam that makes the pipeline seamless (PROJECT_MEMORY.md F9):
the certification session ends with a joint SMC particle cloud, AD6 verdicts,
and a question history; the trainer starts from exactly that, with variance
inflation (Decision D1) and a per-task marginal view (Decision D2).

Conventions: all latent parameters are stored in ENGINE coordinates (θ, ℓ).
Convert with `bridge_conventions.engine_to_plan` — never inline (F3).

D9 hooks carried by this schema (memory §3A — LT1/LT2):
  * absolute wall-clock epoch timestamps + session_id on every trial record
    and on the seed itself;
  * `LearnerLedger` — an append-only cross-session sequence of seed
    references, the data structure the future two-timescale consolidated-
    trait / forgetting model is fitted from.

Serialization is a single `.npz` (no pickle): arrays stored natively,
non-array metadata as a JSON string under `meta_json`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
DEFAULT_INFLATE = 1.5          # D1 starting value; revisit at checkpoint M3 (OQ5)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ───────────── trainer-side per-trial record (used from Step 5 on) ─────────────

@dataclass
class TrialRecord:
    """One training (or eval) trial — the unit the Phase-3 dynamics fits read.

    rt_ms / answer_changes are NaN/-1 when the presentation layer cannot
    supply them; they must still be logged (F10).
    """
    session_id: str
    trial_index: int
    task_k: int
    seg_id: int
    s: float
    s_sd: float
    y: int                      # learner's response
    y_star: int                 # ground-truth label (−1 if not revealed)
    feedback_given: bool
    mode: str                   # "eval" | "bias" | "skill" | "retention"
    rt_ms: float
    answer_changes: int
    t_epoch: float              # absolute wall-clock, seconds since epoch (D9)
    post_decision: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrialRecord":
        return cls(**d)


# ───────────── variance inflation (D1) ─────────────

def inflate_cloud(x, w, factor):
    """Rescale particle deviations about the weighted mean: x' = μ + c·(x − μ).

    Multiplies every weighted central moment's scale by exactly `factor`
    (weighted SD ×= factor; weighted mean unchanged). x: (N,) or (N, K);
    w: (N,) normalized weights.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    mu = (w[:, None] * x).sum(axis=0) if x.ndim == 2 else float((w * x).sum())
    return mu + float(factor) * (x - mu)


def _weighted_summary(x_col, w):
    mu = float((w * x_col).sum())
    sd = float(np.sqrt(max(((w * (x_col - mu) ** 2).sum()), 0.0)))
    return mu, sd


# ───────────── the seed ─────────────

@dataclass
class TrainingSeed:
    """Everything the trainer needs from a finished eval session.

    Array fields are parallel where named hist_*; (N, K) clouds are the joint
    posterior — the per-task marginal for task k is (theta[:, k], ell[:, k], w)
    (D2), exposed via `task_cloud`.
    """
    # posterior clouds (engine coords)
    theta_raw: np.ndarray       # (N, K) — eval posterior, uninflated (audit)
    ell_raw: np.ndarray         # (N, K)
    theta_infl: np.ndarray      # (N, K) — trainer's initial belief (D1)
    ell_infl: np.ndarray        # (N, K)
    w: np.ndarray               # (N,) normalized weights (shared)
    # question history
    hist_k: np.ndarray          # (n,) int — task index per trial
    hist_s: np.ndarray          # (n,) float — signal
    hist_y: np.ndarray          # (n,) int — response
    hist_s_sd: np.ndarray       # (n,) float
    hist_seg_id: np.ndarray     # (n,) int — −1 if unknown
    hist_rt_ms: np.ndarray      # (n,) float — NaN placeholder when absent (F10)
    hist_t_epoch: np.ndarray    # (n,) float — absolute wall-clock (D9)
    hist_post_decision: np.ndarray  # (n,) bool — extended-collection trials
    # non-array metadata (JSON-serializable only)
    meta: dict = field(default_factory=dict)

    # ── accessors ──
    @property
    def K(self) -> int:
        return int(self.theta_raw.shape[1])

    @property
    def N(self) -> int:
        return int(self.theta_raw.shape[0])

    def task_cloud(self, k: int, inflated: bool = True):
        """Per-task marginal (θ_k, ℓ_k, w) — the trainer filter's init (D2)."""
        th = self.theta_infl if inflated else self.theta_raw
        el = self.ell_infl if inflated else self.ell_raw
        return th[:, k].copy(), el[:, k].copy(), self.w.copy()

    def eval_seen_segids(self):
        """Seg-ids served during the eval (OQ2 — reuse policy pending)."""
        return self.hist_seg_id[self.hist_seg_id >= 0]

    # ── persistence ──
    _ARRAY_FIELDS = ("theta_raw", "ell_raw", "theta_infl", "ell_infl", "w",
                     "hist_k", "hist_s", "hist_y", "hist_s_sd", "hist_seg_id",
                     "hist_rt_ms", "hist_t_epoch", "hist_post_decision")

    def save(self, path) -> str:
        path = str(path)
        arrays = {name: getattr(self, name) for name in self._ARRAY_FIELDS}
        np.savez_compressed(path, meta_json=json.dumps(self.meta), **arrays)
        return path if path.endswith(".npz") else path + ".npz"

    @classmethod
    def load(cls, path) -> "TrainingSeed":
        with np.load(str(path), allow_pickle=False) as data:
            kwargs = {name: data[name].copy() for name in cls._ARRAY_FIELDS}
            meta = json.loads(str(data["meta_json"]))
        return cls(meta=meta, **kwargs)


# ───────────── builder ─────────────

def build_seed_from_state(state, *, session_id: str, ell_star,
                          verdicts=None, diagnostics=None,
                          inflate: float = DEFAULT_INFLATE,
                          seg_ids=None, rt_ms=None, trial_epochs=None,
                          post_decision=None,
                          session_type: str = "eval",
                          created_at: str | None = None,
                          notes: str = "") -> TrainingSeed:
    """Build a TrainingSeed from a finished engine state dict.

    state: the core_mcmc_general state ("t", "l", "w", "history",
           optional "history_s_sd").
    ell_star: per-task cut-scores ℓ*_k (mastery targets, F14).
    verdicts/diagnostics: AD6 outputs (StopDecision.verdicts /
           AD6Policy diagnostics dict). Optional for non-cert sessions.
    seg_ids / rt_ms / trial_epochs / post_decision: per-trial parallel lists;
           None ⇒ −1 / NaN / NaN / False placeholders (fields exist
           regardless — F10/D9).
    Extended-data-collection note (F9): pass the FULL-data state here (better
    measurement); `verdicts` stay the byte-frozen official snapshot.
    """
    w = np.asarray(state["w"], dtype=np.float64)
    w = w / w.sum()
    theta = np.asarray(state["t"], dtype=np.float64)
    ell = np.asarray(state["l"], dtype=np.float64)
    N, K = theta.shape
    ell_star = [float(v) for v in np.asarray(ell_star, dtype=float)]
    if len(ell_star) != K:
        raise ValueError(f"ell_star length {len(ell_star)} != K={K}")

    hist = state["history"]
    n = len(hist)
    hist_k = np.array([h[0] for h in hist], dtype=np.int64)
    hist_s = np.array([h[1] for h in hist], dtype=np.float64)
    hist_y = np.array([h[2] for h in hist], dtype=np.int64)
    s_sds = state.get("history_s_sd") or [0.0] * n
    hist_s_sd = np.asarray(s_sds, dtype=np.float64)

    def _parallel(vals, fill, dtype):
        if vals is None:
            return np.full(n, fill, dtype=dtype)
        out = np.asarray(vals, dtype=dtype)
        if out.shape != (n,):
            raise ValueError(f"parallel list shape {out.shape} != ({n},)")
        return out

    hist_seg_id = _parallel(seg_ids, -1, np.int64)
    hist_rt_ms = _parallel(rt_ms, np.nan, np.float64)
    hist_t_epoch = _parallel(trial_epochs, np.nan, np.float64)
    hist_post_decision = _parallel(post_decision, False, bool)

    posterior_summary = {
        "ell_mean": [], "ell_sd": [], "theta_mean": [], "theta_sd": [],
    }
    for k in range(K):
        ml, sl = _weighted_summary(ell[:, k], w)
        mt, st = _weighted_summary(theta[:, k], w)
        posterior_summary["ell_mean"].append(ml)
        posterior_summary["ell_sd"].append(sl)
        posterior_summary["theta_mean"].append(mt)
        posterior_summary["theta_sd"].append(st)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "session_id": str(session_id),
        "session_type": str(session_type),
        "created_at": created_at if created_at is not None else _utc_now_iso(),
        "K": K, "N": N,
        "n_trials": n,
        "inflate": float(inflate),
        "ell_star": ell_star,
        "verdicts": list(verdicts) if verdicts is not None else None,
        "ad6_diagnostics": diagnostics,
        "posterior_summary": posterior_summary,
        "param_convention": "engine (theta, ell); sigma=exp(-ell), t=-theta",
        "notes": notes,
    }
    return TrainingSeed(
        theta_raw=theta.copy(), ell_raw=ell.copy(),
        theta_infl=inflate_cloud(theta, w, inflate),
        ell_infl=inflate_cloud(ell, w, inflate),
        w=w.copy(),
        hist_k=hist_k, hist_s=hist_s, hist_y=hist_y, hist_s_sd=hist_s_sd,
        hist_seg_id=hist_seg_id, hist_rt_ms=hist_rt_ms,
        hist_t_epoch=hist_t_epoch, hist_post_decision=hist_post_decision,
        meta=meta,
    )


# ───────────── learner ledger (D9 — LT1/LT2 data structure) ─────────────

class LearnerLedger:
    """Append-only sequence of session/seed references for ONE learner.

    The slow-timescale models (consolidated trait, forgetting kernel —
    memory §3A) are fitted from this. JSON on disk; entries are never
    mutated or removed, only appended.
    """

    def __init__(self, learner_id: str, entries=None):
        self.learner_id = str(learner_id)
        self._entries = list(entries) if entries else []

    @property
    def entries(self):
        return list(self._entries)        # copy — append via append_entry only

    def append_entry(self, *, seed_path: str, session_id: str,
                     session_type: str, created_at: str | None = None,
                     notes: str = "") -> dict:
        entry = {
            "seq": len(self._entries),
            "seed_path": str(seed_path),
            "session_id": str(session_id),
            "session_type": str(session_type),
            "created_at": created_at if created_at is not None else _utc_now_iso(),
            "notes": notes,
        }
        self._entries.append(entry)
        return entry

    def save(self, path) -> None:
        payload = {"schema_version": SCHEMA_VERSION,
                   "learner_id": self.learner_id,
                   "entries": self._entries}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path) -> "LearnerLedger":
        payload = json.loads(Path(path).read_text())
        return cls(payload["learner_id"], payload["entries"])
