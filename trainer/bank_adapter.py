"""Production bank adapter — per-task `TaskCandidates` for the trainer policy
(G1 port of trainer_rd bank_adapter, retargeted to PRODUCTION data, D3/D13).

`TaskCandidates` (the array shape the policy consumes) is a verbatim port. The
`BankAdapter` is a SEMANTIC port: instead of the anonymized `*_general` CSVs it
reads the production bank `MANIFEST.json` (per-task s_mean/s_sd) and the
credentialed-panel votes (`trainer.bank.load_panel_votes`) to reconstruct
per-(seg, task) `y_star` / `margin` / `coherent` (D11) with the D-INT-6 panel as
the label authority. Not bit-parity-testable against the scratch (different data
source); gated by correctness tests instead.

Per task k (one-vs-rest):
  * s_mean/s_sd  — the segment's signal for task k (spike: the spike entry;
                   IIIC: the per-task s_mean_{code}/s_sd_{code}).
  * y_star       — 1 iff the segment IS class k (spike: panel present-fraction
                   > DOMAIN1_POS_THRESHOLD; IIIC: panel plurality == k).
  * margin       — confidence in the binary k-vs-not-k decision (D11).
  * coherent     — label agrees with the signal side (y* == (s_mean > 0)).
`candidates(feedback_safe=True)` drops conflicted + low-margin items (D11) and
any served/eval-seen seg-ids (D12).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from trainer import domains
from trainer.bank import _iiic_plurality, load_panel_votes

DOMAIN1_POS_THRESHOLD = 0.65        # spike present-fraction cut (D11)


@dataclass
class TaskCandidates:
    """Parallel per-task candidate arrays in the shape the policy consumes."""
    task: int
    seg_id: np.ndarray      # (n,) int64
    s_mean: np.ndarray      # (n,) float64
    s_sd: np.ndarray        # (n,) float64
    y_star: np.ndarray      # (n,) int64 ∈ {0,1}
    margin: np.ndarray      # (n,) float64 ∈ [0,1] — label confidence
    coherent: np.ndarray    # (n,) bool — label matches signal side

    def __len__(self):
        return int(self.seg_id.shape[0])

    def subset(self, mask) -> "TaskCandidates":
        return TaskCandidates(self.task, self.seg_id[mask], self.s_mean[mask],
                              self.s_sd[mask], self.y_star[mask],
                              self.margin[mask], self.coherent[mask])


def load_cut_scores(task_codes=domains.CODES, block="ell_star_unified_v15"):
    """(ell_star, sigma_star) arrays aligned to `task_codes`, from cert_config via
    the production loader — the SAME cut the exam uses (default = the v15 trainer
    mastery target; pass 'ell_star_unified_v14' for the eval instrument)."""
    import os
    import sys
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _scripts = os.path.join(_repo, "scripts")
    if _scripts not in sys.path:
        sys.path.append(_scripts)
    from cortex_policy_k7 import load_ell_star_k7
    ell = np.array(load_ell_star_k7(list(task_codes), block_name=block),
                   dtype=np.float64)
    return ell, np.exp(-ell)


class BankAdapter:
    """Loads production signals + credentialed-panel votes once, exposes filtered
    per-task pools. Construction scans the vote table (~seconds); do it once."""

    def __init__(self, *, manifest_in=None, labels_in=None, raters_in=None):
        kw = {k: v for k, v in (("manifest_in", manifest_in),
                                ("labels_in", labels_in),
                                ("raters_in", raters_in)) if v is not None}
        prod, spike_pos, spike_neg, iiic_votes, man = load_panel_votes(**kw)
        # per-segment per-task signal (s_mean, s_sd) from the MANIFEST
        self._sig: dict[int, dict[str, tuple]] = {}
        for s in man["segments"]:
            sid = int(s["seg_id"])
            if s["family"] == "spike":
                self._sig[sid] = {"spike": (float(s["s_mean"]), float(s["s_sd"]))}
            else:
                self._sig[sid] = {
                    d.code: (float(s[f"s_mean_{d.code}"]),
                             float(s[f"s_sd_{d.code}"]))
                    for d in domains.DOMAINS if d.family == "iiic"}
        self._prod = prod
        self._spike_pos, self._spike_neg, self._iiic_votes = (
            spike_pos, spike_neg, iiic_votes)
        self._pools = [self._build_pool(k) for k in range(domains.K)]

    def _build_pool(self, k: int) -> TaskCandidates:
        code = domains.CODES[k]
        family = domains.DOMAINS[k].family
        seg, sm, ss, y, mg, coh = [], [], [], [], [], []
        for sid, (fam, _pcl) in self._prod.items():
            if fam != family:
                continue
            sig = self._sig[sid].get(code)
            if sig is None or not (np.isfinite(sig[0]) and np.isfinite(sig[1])):
                continue
            if code == "spike":
                pos, neg = self._spike_pos.get(sid, 0), self._spike_neg.get(sid, 0)
                reads = pos + neg
                if reads == 0:
                    continue
                pf = pos / reads
                yi = int(pf > DOMAIN1_POS_THRESHOLD)
                margin = float(np.clip(abs(pf - DOMAIN1_POS_THRESHOLD)
                                       / (1.0 - DOMAIN1_POS_THRESHOLD), 0.0, 1.0))
            else:
                c = self._iiic_votes.get(sid, Counter())
                n_votes = sum(c.values())
                if n_votes == 0:
                    continue
                plur_code, _ = _iiic_plurality(c)
                yi = int(plur_code == code)
                frac = c.get(domains.by_code(code).pattern_class, 0) / n_votes
                margin = float(frac if yi == 1 else 1.0 - frac)
            seg.append(sid); sm.append(sig[0]); ss.append(sig[1])
            y.append(yi); mg.append(margin)
            coh.append((yi == 1) == (sig[0] > 0.0))
        order = np.argsort(np.asarray(seg, dtype=np.int64))   # deterministic
        arr = lambda lst, dt: np.asarray(lst, dtype=dt)[order]  # noqa: E731
        return TaskCandidates(k, arr(seg, np.int64), arr(sm, np.float64),
                              arr(ss, np.float64), arr(y, np.int64),
                              arr(mg, np.float64), arr(coh, bool))

    # ── public API (matches the scratch adapter) ──
    def task_pool(self, task: int) -> TaskCandidates:
        return self._pools[int(task)]

    def candidates(self, task: int, *, exclude_segids=None,
                   feedback_safe: bool = False,
                   min_margin: float = 0.30) -> TaskCandidates:
        pool = self._pools[int(task)]
        mask = np.ones(len(pool), dtype=bool)
        if exclude_segids:
            excl = np.fromiter((int(x) for x in exclude_segids), dtype=np.int64)
            mask &= ~np.isin(pool.seg_id, excl)
        if feedback_safe:
            mask &= pool.coherent & (pool.margin >= float(min_margin))
        return pool.subset(mask)

    def conflicted_fraction(self, task: int) -> float:
        pool = self._pools[int(task)]
        return float(1.0 - pool.coherent.mean()) if len(pool) else 0.0
