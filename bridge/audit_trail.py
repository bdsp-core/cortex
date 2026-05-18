"""Per-question / per-session JSONL audit log for clinical traceability.

Implements the W2-C audit trail required by reviewer R5 §D2 and Tier-1.5 of the
Wave-2 plan.  The design follows IEC 62304 §5.5 (verification) and §5.7
(traceability) for software of unknown provenance: every certification session
must produce a durable, plain-text record of (a) the input context, (b) the
final state of the Bayesian posterior, and (c) the resulting decision, in a
form a clinical reviewer can audit without re-running the engine.

Design choice: Option A (post-hoc adapter)
------------------------------------------
The certification engine `run_session_mcmc_certification` (in
`core_mcmc.py`) is treated as a black box.  We do NOT modify its signature
to inject an `audit_log_path` parameter (that would be Option B and is
deferred per the W2-C task spec).  Instead, this module wraps the call
site: the bridge runs a session, then `AuditTrail.record_session` writes
the relevant fields from the return dict to a JSONL file.

What is recorded per session
----------------------------
Each line is a single JSON object with the schema below.  Per-question
state (item index `q_idx`, signal `s`, response `y`, MCSE per question)
is NOT currently exposed by `run_session_mcmc_certification`'s return
dict; only the final session summary is recorded.  When core_mcmc.py is
extended in a future wave to expose per-question history, the same
record-format can be expanded without breaking forward compatibility
(JSON readers ignore unknown keys; the `schema_version` field allows
strict consumers to dispatch).

Schema (v1):
    {
        "schema_version":   "1",
        "timestamp_utc":    ISO-8601 UTC timestamp,
        "rater_id":         str,
        "method":           "hier" | "brute",
        "seed":             int,
        "K":                int,
        "domains":          [str, ...],            # K names
        "stop_thresh":      float,
        "z_buffer":         float,
        "lapse_rate":       float,                 # from cert_config.yaml
        "n_questions":      int,
        "per_domain_n":     [int, ...],            # K
        "decisions":        [int, ...],            # K, in {-1, 0, 1}
        "stopped_early":    bool,
        "active_at_end":    [int, ...],            # indices of undecided domains
        "pass_probs_final": [float, ...],          # K
        "mcse_final":       [float, ...],          # K
        "true_auroc":       [float, ...],          # K, ground-truth AUROC
        "true_pass":        [bool, ...],           # K, true_l > l_star_k
        "n_eff_proxy":      float | null,          # last ESS, if available
        "mean_acceptance_rate": float | null,
        "n_rejuvenations":      int | null,
    }

Usage from the bridge
---------------------
    audit = AuditTrail("audit_logs/MyRater_seed0.jsonl")
    out = run_session_mcmc_certification(...)
    audit.record_session(
        rater_id="MyRater",
        method="hier",
        seed=0,
        domains=["sz", "lpd", ...],
        l_star=l_star,
        true_params=true_params,
        lapse_rate=0.025,
        session_result=out,
    )
    audit.close()

The class is also a context manager:
    with AuditTrail(path) as audit:
        ...
        audit.record_session(...)
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

import numpy as np

SCHEMA_VERSION = "1"


def _to_jsonable(x: Any) -> Any:
    """Convert numpy / non-JSON types to plain Python so json.dumps doesn't error.

    JSON's strict mode cannot encode NaN/Inf; any non-finite float (numpy OR
    Python-native, including the python floats produced by ndarray.tolist())
    is normalised to None.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, str):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # Last resort: stringify
    return str(x)


class AuditTrail:
    """Per-session JSONL audit log for clinical traceability (IEC 62304 §5.5/§5.7).

    Each line of the output file is a JSON object describing one
    `run_session_mcmc_certification` invocation.  Append-only; safe to
    re-open and continue writing across runs.

    Parameters
    ----------
    path : str
        Path to the JSONL file.  Parent directory must exist (the bridge
        creates `audit_logs/` at startup).
    flush_each_line : bool, default True
        If True, fsync after every line (paranoid mode for clinical use).
        Set False for benchmark runs where throughput matters.
    """

    def __init__(self, path: str, flush_each_line: bool = True):
        self.path = str(path)
        self.flush_each_line = bool(flush_each_line)
        parent = os.path.dirname(self.path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        # Open in append mode so re-opening doesn't clobber prior runs.
        self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
        self._closed = False

    # ── lifecycle ──────────────────────────────────────────────────────
    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                # fsync not supported on every filesystem; best-effort only.
                pass
            self._fh.close()
        finally:
            self._closed = True

    def __enter__(self) -> "AuditTrail":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ── record one session ─────────────────────────────────────────────
    def record_session(
        self,
        *,
        rater_id: str,
        method: str,
        seed: int,
        domains: Iterable[str],
        l_star: Iterable[float],
        true_params: Iterable[float],
        lapse_rate: float,
        session_result: Dict[str, Any],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write one JSONL line summarizing a finished certification session.

        Parameters
        ----------
        rater_id : str
            Identifier for the rater (e.g. canonical name).  Goes into the
            `rater_id` field and into the suggested filename
            ``audit_logs/<rater_id>_<seed>.jsonl``.
        method : str
            "hier" or "brute" (whichever was passed to
            `run_session_mcmc_certification`).
        seed : int
            RNG seed used for the session.
        domains : iterable of str
            Domain names matching the K columns of the result arrays.
        l_star : iterable of float
            Per-domain skill thresholds (length K).
        true_params : iterable of float
            Flat (theta_0, l_0, theta_1, l_1, ..., theta_{K-1}, l_{K-1})
            list as passed to the engine.  Used to compute true_pass.
        lapse_rate : float
            Lapse rate used by the engine (recorded for full reproducibility
            even though the engine reads it from a module constant).
        session_result : dict
            The dict returned by `run_session_mcmc_certification`.
        extra : dict, optional
            Caller-supplied extra fields merged into the record under the
            top-level key ``extra``.  Useful for stamping config provenance
            (e.g. cert_config.yaml hash, git rev) without baking those
            details into the schema.

        Returns
        -------
        dict
            The JSON-serialisable record actually written (handy for tests).
        """
        if self._closed:
            raise RuntimeError("AuditTrail.record_session called after close()")

        domains = list(domains)
        K = len(domains)
        l_star_arr = np.asarray(list(l_star), dtype=float)
        true_params_arr = np.asarray(list(true_params), dtype=float)

        # Recover ground truth: true_l[k] = true_params[2k + 1].
        if true_params_arr.size != 2 * K:
            raise ValueError(
                f"true_params length {true_params_arr.size} != 2*K={2 * K} "
                f"(domains={domains})"
            )
        true_l = true_params_arr[1::2]
        true_pass = (true_l > l_star_arr).astype(bool)

        # Some fields may legitimately be missing if the engine signature
        # changes; default to None and let _to_jsonable handle it.
        decisions = session_result.get("decisions")
        per_domain_n = session_result.get("per_domain_n")
        active_at_end = session_result.get("active_at_end", [])
        pass_probs_final = session_result.get("pass_probs_final")
        mcse_final = session_result.get("mcse_final")
        true_auroc = session_result.get("true_auroc")

        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rater_id": str(rater_id),
            "method": str(method),
            "seed": int(seed),
            "K": int(K),
            "domains": list(domains),
            "stop_thresh": _to_jsonable(session_result.get("stop_thresh_used")),
            "z_buffer": _to_jsonable(session_result.get("z_buffer")),
            "lapse_rate": float(lapse_rate),
            "n_questions": _to_jsonable(session_result.get("n_questions")),
            "per_domain_n": _to_jsonable(per_domain_n),
            "decisions": _to_jsonable(decisions),
            "stopped_early": _to_jsonable(session_result.get("stopped_early")),
            "active_at_end": _to_jsonable(active_at_end),
            "pass_probs_final": _to_jsonable(pass_probs_final),
            "mcse_final": _to_jsonable(mcse_final),
            "true_auroc": _to_jsonable(true_auroc),
            "true_pass": _to_jsonable(true_pass),
            "l_star": _to_jsonable(l_star_arr),
            "n_eff_proxy": _to_jsonable(session_result.get("n_eff_proxy")),
            "mean_acceptance_rate": _to_jsonable(
                session_result.get("mean_acceptance_rate")
            ),
            "n_rejuvenations": _to_jsonable(session_result.get("n_rejuvenations")),
        }
        if extra:
            record["extra"] = _to_jsonable(extra)

        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._fh.write(line + "\n")
        if self.flush_each_line:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass
        return record

    # ── F1.3 (2026-05-15): Mode-A Multi-AUROC session record ───────────
    def record_session_mode_a(
        self,
        *,
        rater_id: str,
        method: str,
        seed: int,
        domains: Iterable[str],
        true_params: Iterable[float],
        lapse_rate: float,
        session_result: Dict[str, Any],
        delta_sweep: Iterable[float],
        sigma_note: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Write one JSONL line summarizing a finished Mode-A Multi-AUROC session.

        Mode-A schema differs from Mode-B: no l_star, no decisions, no
        pass_probs_final.  Instead records the per-domain AUROC posterior
        summary (mean ≈ CI midpoint, ci_low, ci_high), the post-hoc
        δ-sweep stops, and the trajectory shape for downstream analysis.

        Parameters
        ----------
        rater_id, method, seed, domains, true_params, lapse_rate :
            as in `record_session`.
        session_result : dict
            Output of `run_session_mcmc_auroc` (must include `final_lo`,
            `final_hi`, `n_questions`, `true_auroc`, `delta_sweep_stops`).
        delta_sweep : iterable of float
            The δ values that were evaluated post-hoc (recorded for
            provenance even though `session_result.delta_sweep_stops` is
            keyed by them).
        sigma_note : str, optional
            Free-form note about which Σ matrix was loaded (e.g.
            "Corr_l (unit-diagonal); Mode-A no boundary scaling").
        extra : dict, optional
            Caller-supplied extra fields under `extra`.
        """
        if self._closed:
            raise RuntimeError("AuditTrail.record_session_mode_a called after close()")

        domains = list(domains)
        K = len(domains)
        true_params_arr = np.asarray(list(true_params), dtype=float)
        if true_params_arr.size != 2 * K:
            raise ValueError(
                f"true_params length {true_params_arr.size} != 2*K={2 * K}"
            )
        true_t = true_params_arr[0::2]
        true_l = true_params_arr[1::2]

        final_lo = np.asarray(session_result["final_lo"], dtype=float)
        final_hi = np.asarray(session_result["final_hi"], dtype=float)
        auroc_mean = (final_lo + final_hi) / 2.0
        hw = (final_hi - final_lo) / 2.0

        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "mode": "mode_a",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rater_id": str(rater_id),
            "method": str(method),
            "seed": int(seed),
            "K": int(K),
            "domains": list(domains),
            "lapse_rate": float(lapse_rate),
            "auroc_delta": _to_jsonable(session_result.get("delta_auroc")),
            "delta_sweep": [float(d) for d in delta_sweep],
            "delta_sweep_stops": _to_jsonable(
                session_result.get("delta_sweep_stops", {})
            ),
            "n_questions_total": _to_jsonable(session_result.get("n_questions")),
            "stopped_early": _to_jsonable(session_result.get("stopped_early")),
            "auroc_mean_per_domain": _to_jsonable(auroc_mean),
            "auroc_ci_low_per_domain": _to_jsonable(final_lo),
            "auroc_ci_high_per_domain": _to_jsonable(final_hi),
            "hw_per_domain": _to_jsonable(hw),
            "hw_max_final": float(hw.max()) if hw.size else None,
            "true_auroc": _to_jsonable(session_result.get("true_auroc")),
            "true_t": _to_jsonable(true_t),
            "true_l": _to_jsonable(true_l),
            "mean_acceptance_rate": _to_jsonable(
                session_result.get("mean_acceptance_rate")
            ),
            "n_rejuvenations": _to_jsonable(session_result.get("n_rejuvenations")),
            "sigma_note": sigma_note,
        }
        if extra:
            record["extra"] = _to_jsonable(extra)

        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._fh.write(line + "\n")
        if self.flush_each_line:
            self._fh.flush()
            try:
                os.fsync(self._fh.fileno())
            except OSError:
                pass
        return record


# ── convenience helper for the bridge ──────────────────────────────────
def session_audit_path(audit_log_dir: str, rater_id: str, seed: int) -> str:
    """Return the canonical JSONL path for a given (rater, seed) pair.

    The bridge writes one JSONL file per (rater, seed) so concurrent
    processes don't have to coordinate file locks.  The directory must
    exist; create it once at bridge startup.
    """
    safe_rater = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(rater_id)
    )
    return os.path.join(audit_log_dir, f"{safe_rater}_seed{int(seed)}.jsonl")
