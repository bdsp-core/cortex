"""M23 (F82/F83) — sandbox-side monitors: contact e-process + RT floor.

ContactMonitor (F83, SHADOW — log-only, D34 principle):
    Anytime-valid test of "no perceptual contact" for one task within one
    session. Null H0: the conditional probability of a CORRECT response
    stays in the chance band, P(c_t = 1 | F_{t−1}) ∈ [1−p0, p0] with
    p0 = 0.55 (the serving policy's measured label balance is 0.45–0.55).
    Statistic: E_T = Π_t  q_t(c_t) / p0, where q_t is the Krichevsky–
    Trofimov universal Bernoulli predictor q_t(1) = (n_1 + ½)/(t + 1) —
    a predictable forecaster, so under ANY law in H0,
        E[q_t(c_t)/p0 | F_{t−1}] = P(c=1)q_t(1)/p0 + P(c=0)q_t(0)/p0
                                 ≤ p0·(q_t(1) + q_t(0))/p0 = 1,
    i.e. E_T is a nonnegative supermartingale and Ville's inequality gives
    P(sup_T E_T ≥ 1/α) ≤ α. Contact is declared at E_T ≥ 20 (α = 0.05),
    valid at every trial without correction. Semantics: "responses carry
    information about the correct answer beyond the chance band" — an
    anti-correlated responder also departs the band and can (correctly)
    certify contact. Chosen over a belief-predictive numerator because the
    KT numerator is immune to the filter's own lag (F80) and to response
    bias: a μ-biased stimulus-blind guesser has marginal correctness ~½
    under label balance, which KT tracks (validated: 0 false contacts in
    200 null + 200 μ=0.65 sessions, study_m23_regime arm F).
    Restarted per session (per-session α; the monitor answers "is there
    contact NOW", not "was there ever").

rt_floor_breach (F82, ACTIVE — serving gate, D37 principle: protocol layer,
    never inference): USER-D's careless collapse ran at a 206 ms trailing
    median while every engaged tester window across 14 sessions stayed
    ≥ 1059 ms; k consecutive sub-floor RTs mark non-perceptual responding
    (scanning the 48-sample trace takes ≳ 1 s). Calibration
    (study_m23_regime arm E): floor 500 ms / k = 3 → 0 engaged-session
    false alarms, fires 3 trials before the M22 accuracy guard on D-s1.
"""
from __future__ import annotations

import numpy as np

CONTACT_P0 = 0.55
CONTACT_ALPHA = 0.05


class ContactMonitor:
    """Per-(task, session) contact e-process with a KT numerator."""

    def __init__(self, p0=CONTACT_P0, alpha=CONTACT_ALPHA):
        self.p0 = float(p0)
        self.log_thresh = float(np.log(1.0 / float(alpha)))
        self.log_e = 0.0
        self.n = 0
        self.n1 = 0
        self.contact = False                 # sticky within the session

    def update(self, correct):
        c = int(correct)
        q1 = (self.n1 + 0.5) / (self.n + 1.0)    # KT predictive for c=1
        q = q1 if c else 1.0 - q1
        self.log_e += float(np.log(q / self.p0))
        self.n += 1
        self.n1 += c
        if not self.contact and self.log_e >= self.log_thresh:
            self.contact = True
        return {"log_e": round(self.log_e, 3), "contact": self.contact}


def rt_floor_breach(rts_ms, *, floor_ms=500.0, k=3):
    """True when the last k response times are ALL below the floor."""
    if len(rts_ms) < k:
        return False
    tail = list(rts_ms)[-k:]
    return all(float(r) < float(floor_ms) for r in tail)
