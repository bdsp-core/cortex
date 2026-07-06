"""M21 (F72) — learner-consistency monitor (shadow changepoint detector).

WHY. The two-tester sandbox sessions (2026-07-02) showed that an un-modeled
LEARNER SWITCH is nearly invisible to every shipped monitor: the e-gate
watches below-bar probe evidence, the GapAnchor only opens on ≥4 h
wall-clock gaps (the switch happened at 89.5 s), and raw prequential
surprise is flat when beliefs predict near-chance (a switch TOWARD
competence lowers surprise). What does separate the testers is the shape of
recent behavior vs the belief's prediction: windowed d′ flipped from −0.15
to +1.15 on one task and 0.0 to −0.73 on the other, and log-RT shifted by
0.6 nats (p < 1e-3).

WHAT. Per task, keep a sliding window of the last W trials
(s_mean, s_sd, y, p_pred), where p_pred is the belief's predictive
P(y=1) BEFORE that trial's update. Each trial, compute a GLR:

    Λ_W = max_{σ',t'} Σ_{i∈W} log p(y_i | σ', t', s_i, s_sd_i) − Σ_{i∈W} log p_pred,i(y_i)

with the alternative's response model the same probit-lapse observation
model the whole stack uses (s_sd-attenuated). The null term uses the
ADAPTIVE belief's own predictions, so slow drift the filter tracks is not
flagged; only structure the belief did NOT predict accumulates. 2·Λ is
compared against a threshold calibrated by simulation on stationary,
well-specified learners (repeated looks + adaptive null make analytic
χ² thresholds wrong in both directions; see studies/study_learner_switch).

An auxiliary RT statistic (Welch z of mean log-RT, window vs history) is
reported alongside — identity-relevant but person-confounded with fatigue,
so it CONTRIBUTES A REPORT, not a trigger.

SHADOW-ONLY (D34): the monitor logs; it never mutates beliefs. The
production action on a confirmed flag is an identity/context check at the
delivery-vehicle layer, not a statistical reset.
"""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.stats import norm

LAPSE = 0.025

# calibrated on 200 stationary well-specified two-session learners (400
# 40-trial single-task sessions, study_learner_switch arm A, full run
# 2026-07-02): 99th pct of the two-session max 2Λ = 11.30 (95th = 9.35,
# max observed 12.86). Single-task 40-trial sessions give the monitor MORE
# data per window than the K=2 sandbox, so this threshold is conservative
# there. Re-pin if W or the observation model changes.
DEFAULT_THRESHOLD = 11.3


def _p_yes(s, s_sd, sigma, t):
    """Attenuated probit-lapse P(y=1 | item) — the stack's observation model
    marginalized over s_real ~ N(s, s_sd²)."""
    att = np.sqrt(sigma * sigma + np.asarray(s_sd) ** 2)
    z = (np.asarray(s) - t) / att
    return LAPSE + (1.0 - 2.0 * LAPSE) * norm.cdf(z)


class ConsistencyMonitor:
    """Sliding-window GLR of 'someone else generated the recent trials'
    against the belief's own running predictions, for ONE task."""

    # alternative-hypothesis grid (σ' log-spaced, t' linear); coarse is fine —
    # the GLR needs the neighborhood maximum, not the argmax
    SIGMAS = np.exp(np.linspace(-1.1, 1.4, 26))
    TS = np.linspace(-1.6, 1.6, 33)

    def __init__(self, window=20, threshold=DEFAULT_THRESHOLD):
        self.W = int(window)
        self.threshold = float(threshold)
        self.buf = deque(maxlen=self.W)     # (s_mean, s_sd, y, p_pred)
        self.rt_hist = []                   # log-RTs before the window
        self.rt_win = deque(maxlen=self.W)
        self.glr = 0.0                      # current 2Λ_W
        self.rt_z = 0.0
        self.flagged = False                # sticky within a session

    # ── persistence (json-safe) ──
    def to_json(self):
        return {"W": self.W, "threshold": self.threshold,
                "buf": [list(map(float, b)) for b in self.buf],
                "rt_hist": [float(x) for x in self.rt_hist[-200:]],
                "rt_win": [float(x) for x in self.rt_win],
                "flagged": bool(self.flagged)}

    @classmethod
    def from_json(cls, d):
        m = cls(window=d.get("W", 20),
                threshold=d.get("threshold", DEFAULT_THRESHOLD))
        for b in d.get("buf", []):
            m.buf.append(tuple(b))
        m.rt_hist = list(d.get("rt_hist", []))
        for x in d.get("rt_win", []):
            m.rt_win.append(float(x))
        m.flagged = bool(d.get("flagged", False))
        return m

    # ── update ──
    def update(self, s_mean, s_sd, y, p_pred, rt_ms=None):
        """Record one trial (p_pred = belief predictive P(y=1) BEFORE the
        filter update) and recompute the statistics. Returns dict."""
        if len(self.rt_win) == self.rt_win.maxlen:
            self.rt_hist.append(self.rt_win[0])
        if rt_ms is not None and rt_ms > 0:
            self.rt_win.append(float(np.log(rt_ms)))
        self.buf.append((float(s_mean), float(s_sd), int(y),
                         float(min(max(p_pred, 1e-9), 1 - 1e-9))))
        self.glr = self._glr()
        self.rt_z = self._rt_shift()
        fired = self.glr > self.threshold
        self.flagged = self.flagged or fired
        return {"glr": round(self.glr, 3), "rt_z": round(self.rt_z, 2),
                "flag": bool(fired)}

    def _glr(self):
        if len(self.buf) < 8:                    # too short to say anything
            return 0.0
        s = np.array([b[0] for b in self.buf])
        sd = np.array([b[1] for b in self.buf])
        y = np.array([b[2] for b in self.buf])
        pp = np.array([b[3] for b in self.buf])
        ll_null = float(np.sum(y * np.log(pp) + (1 - y) * np.log(1 - pp)))
        # vectorized grid: (n_sig, n_t, W)
        att = np.sqrt(self.SIGMAS[:, None, None] ** 2 + sd[None, None, :] ** 2)
        z = (s[None, None, :] - self.TS[None, :, None]) / att
        p = LAPSE + (1 - 2 * LAPSE) * norm.cdf(z)
        ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p), axis=2)
        return max(2.0 * (float(ll.max()) - ll_null), 0.0)

    def _rt_shift(self):
        if len(self.rt_hist) < 10 or len(self.rt_win) < 8:
            return 0.0
        h = np.array(self.rt_hist[-120:])
        w = np.array(self.rt_win)
        se = np.sqrt(h.var(ddof=1) / len(h) + w.var(ddof=1) / len(w))
        return float((w.mean() - h.mean()) / max(se, 1e-9))
