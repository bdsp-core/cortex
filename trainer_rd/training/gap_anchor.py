"""M17 (D27) — measurement-based retention re-anchoring (`GapAnchor`).

DESIGN CHOICE (user-ratified): instead of committing to a parametric
forgetting law (exponential vs power — the literature is split and per-person
variation is large), the gap's effect on the state is MEASURED at each
session open. The machinery is the same Bayesian-model-averaging pattern as
the σ_∞-mixture (F54): a small grid of RETENTION-FRACTION hypotheses,
resolved by prequential evidence from a short anchor block of
information-optimal probes.

THE MATH.
  Session-open prior. Given the pre-gap posterior cloud (θ, ℓ) and personal
  baselines (θ_b, ℓ_b) (the un-trained state the learner would revert to),
  stratum j asserts retention fraction r_j ∈ grid:
      x'_j = x_b + r_j · (x − x_b) + N(0, w²)   per particle, per coord,
  i.e. r=1 ⇒ full retention, r=0 ⇒ full reversion; w is a small
  within-stratum widen (the strata carry the BETWEEN-hypothesis spread).
  Prior weights are tilted toward r̄(Δt) = exp(−Δt/S) by a WIDE kernel
  (sd 0.25 on r) — the personal stability S only tilts, evidence decides:
      log ω_j ∝ −(r_j − r̄(Δt))²/(2·0.25²).
  Anchor block. n_anchor probes (alternating criterion-info and cut-state
  ℓ-info placements, the two existing Fisher-optimal scorers) update every
  stratum and its prequential weight exactly as in mixture_filter. After the
  block the strata are COLLAPSED (ω-weighted concatenation, systematic
  resample back to N) — retention is a per-gap nuisance, not a persistent
  parameter, so no degeneracy issue arises.
  Stability update. Each resolved gap contributes (Δt_i, r̂_i = E_ω[r]);
  S is the through-origin least-squares slope of −log r̂ on Δt,
      S = (Σ Δt_i² ) / (Σ Δt_i · (−log r̂_i)),
  blended with one prior pseudo-pair (Δt = S₀, −log r̂ = 1). S feeds the
  NEXT gap's prior tilt and the anchor length — per-person adaptivity with
  a one-scalar memory, no functional-form commitment.
  Anchor length (the "how often to update estimates" answer): estimates
  update every trial while training; between sessions the anchor block IS
  the update. Its length scales with the expected forgetting the prior
  anticipates:
      n_anchor(Δt, S) = clip( n_min + n_scale · (1 − r̄(Δt)), n_min, n_max )
  (defaults 4 → 16: four probes minimum re-centers the weights — each probe
  contributes O(0.1–0.5) nats of stratum contrast; a fully-forgotten state
  needs ~a dozen to re-localize t̂ to the pre-gap precision). Long-gap
  learners automatically get longer re-anchoring; short gaps cost ~4 trials.

Phase-2/3 fit: implements the D9 `propagate_gap` slot behaviorally — the
LT2 "fitted kernel" becomes OPTIONAL (a better prior tilt), never load-
bearing, which honors the user directive to avoid committing to a law.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from training.training_filter import TaskFilter
from training.trainer_policy import bias_probe_score, cert_probe_score
from training.bridge_conventions import engine_to_plan

R_GRID = (1.0, 0.8, 0.55, 0.3, 0.0)
PRIOR_R_SD = 0.25


class GapAnchor:
    """Wraps ONE TaskFilter with gap-open / anchor / collapse mechanics."""

    def __init__(self, *, S0_days=7.0, n_min=4, n_max=16, n_scale=14,
                 widen=0.05):
        self.S = float(S0_days) * 86_400.0          # stability, seconds
        self._ss_num = self.S ** 2                  # LS accumulators (with
        self._ss_den = self.S                       #  the prior pseudo-pair)
        self.n_min, self.n_max = int(n_min), int(n_max)
        self.n_scale = float(n_scale)
        self.widen = float(widen)
        self.history = []                           # (dt_seconds, r_hat)
        self._strata = None
        self._log_w = None
        self._probe_flip = 1

    # ── session open ──
    def r_bar(self, dt_seconds):
        return float(np.exp(-dt_seconds / self.S))

    def n_anchor(self, dt_seconds):
        return int(np.clip(self.n_min
                           + self.n_scale * (1.0 - self.r_bar(dt_seconds)),
                           self.n_min, self.n_max))

    def open_session(self, filt: TaskFilter, dt_seconds, *,
                     theta_base=0.0, ell_base=0.0, seed=0):
        """Build the retention-hypothesis strata from the pre-gap belief.
        Returns the recommended anchor-block length."""
        rng = np.random.default_rng(seed)
        rb = self.r_bar(dt_seconds)
        self._dt = float(dt_seconds)
        self._strata, lw = [], []
        for j, r in enumerate(R_GRID):
            th = (theta_base + r * (filt.theta - theta_base)
                  + self.widen * rng.standard_normal(filt.N))
            el = (ell_base + r * (filt.ell - ell_base)
                  + self.widen * rng.standard_normal(filt.N))
            f = TaskFilter(th, el, filt.p, seed=seed + 31 * j + 1,
                           w=filt.w, p_static=filt.p_static,
                           smear_w=filt.smear_w,
                           exact_kernel=filt.exact_kernel)
            self._strata.append(f)
            lw.append(-((r - rb) ** 2) / (2.0 * PRIOR_R_SD ** 2))
        self._log_w = np.array(lw, dtype=np.float64)
        return self.n_anchor(dt_seconds)

    # ── anchor block ──
    def pick_probe(self, cands, sigma_star):
        """Anchor placement: alternate criterion-info (t re-anchor) and
        cut-state ℓ-info probes, label-balanced by alternation."""
        mt, ml = self.mean()
        sig_hat, t_hat = engine_to_plan(mt, ml)
        want = 1 if self._probe_flip > 0 else 0
        self._probe_flip *= -1
        m = np.where(cands.y_star == want)[0]
        if m.size == 0:
            m = np.arange(len(cands))
        if want == 1:      # criterion-info half
            sc = bias_probe_score(cands.s_mean[m], cands.s_sd[m],
                                  sig_hat, t_hat)
        else:              # cut-state ℓ-info half
            sc = cert_probe_score(cands.s_mean[m], cands.s_sd[m],
                                  sigma_star, t_hat)
        return int(m[np.argmax(sc)])

    def anchor_step(self, s, y, y_star, *, s_sd=0.0, feedback=True):
        for j, f in enumerate(self._strata):
            pred = f.reweight(s, y, s_sd)
            f.maybe_resample()
            f.propagate(s, y_star, feedback, y=y, s_sd=s_sd)
            self._log_w[j] += np.log(max(pred, 1e-300))
        self._log_w -= self._log_w.max()

    def weights(self):
        return np.exp(self._log_w - logsumexp(self._log_w))

    def mean(self):
        om = self.weights()
        ms = np.array([f.mean() for f in self._strata])
        mt, ml = (om[:, None] * ms).sum(axis=0)
        return float(mt), float(ml)

    # ── collapse + stability update ──
    def close_anchor(self, *, seed=0):
        """Collapse strata into one TaskFilter; record (Δt, r̂) and update
        the personal stability S. Returns (filter, r_hat)."""
        om = self.weights()
        r_hat = float(np.clip((om * np.array(R_GRID)).sum(), 0.05, 0.999))
        base = self._strata[0]
        theta = np.concatenate([f.theta for f in self._strata])
        ell = np.concatenate([f.ell for f in self._strata])
        w = np.concatenate([o * f.w for o, f in zip(om, self._strata)])
        w /= w.sum()
        rng = np.random.default_rng(seed + 7)
        u = (rng.random() + np.arange(base.N)) / base.N
        idx = np.minimum(np.searchsorted(np.cumsum(w), u), w.size - 1)
        merged = TaskFilter(theta[idx], ell[idx], base.p, seed=seed + 13,
                            p_static=base.p_static, smear_w=base.smear_w,
                            exact_kernel=base.exact_kernel)
        # through-origin LS of −log r̂ on Δt (prior pseudo-pair included)
        self._ss_num += self._dt ** 2
        self._ss_den += self._dt * (-np.log(r_hat))
        self.S = float(self._ss_num / max(self._ss_den, 1e-9))
        self.history.append((self._dt, r_hat))
        self._strata = None
        return merged, r_hat


class MixtureGapAnchor(GapAnchor):
    """M19-3 — gap re-anchoring for a `SigmaInfMixtureFilter`.

    At session open the belief becomes the PRODUCT mixture over
    (ceiling stratum j × retention fraction r): cell weight
    w_{j,r} = ω_j · prior_r(Δt | S), cell cloud = the r-transform of stratum
    j's cloud. The anchor block updates every cell prequentially; close
    MARGINALIZES retention back out:
        ω'_j ∝ Σ_r w_{j,r},    stratum-j cloud ← w̄_{j,·}-weighted r-merge
    (systematic resample to N, in place), and r̂ = Σ w̄_{j,r}·r feeds the
    same personal-stability update. Retention hypotheses are PER-GAP
    nuisances; ceiling hypotheses persist across the whole protocol —
    exactly the D22/D27 division of labor, and why r must be integrated out
    rather than kept as a fourth persistent dimension."""

    def open_session(self, mix, dt_seconds, *, theta_base=0.0, ell_base=0.0,
                     seed=0):
        rng = np.random.default_rng(seed)
        rb = self.r_bar(dt_seconds)
        self._dt = float(dt_seconds)
        self._mix = mix
        J = len(mix.strata)
        self._cells = []
        lw = np.zeros((J, len(R_GRID)))
        for j, f in enumerate(mix.strata):
            row = []
            for ri, r in enumerate(R_GRID):
                th = (theta_base + r * (f.theta - theta_base)
                      + self.widen * rng.standard_normal(f.N))
                el = (ell_base + r * (f.ell - ell_base)
                      + self.widen * rng.standard_normal(f.N))
                row.append(TaskFilter(th, el, f.p,
                                      seed=seed + 101 * j + 7 * ri + 1,
                                      w=f.w, p_static=f.p_static,
                                      smear_w=f.smear_w,
                                      exact_kernel=f.exact_kernel))
                lw[j, ri] = (mix.log_w[j]
                             - ((r - rb) ** 2) / (2.0 * PRIOR_R_SD ** 2))
            self._cells.append(row)
        self._cell_lw = lw
        self._strata = True                     # anchor-open marker
        return self.n_anchor(dt_seconds)

    def anchor_step(self, s, y, y_star, *, s_sd=0.0, feedback=True):
        for j, row in enumerate(self._cells):
            for ri, f in enumerate(row):
                pred = f.reweight(s, y, s_sd)
                f.maybe_resample()
                f.propagate(s, y_star, feedback, y=y, s_sd=s_sd)
                self._cell_lw[j, ri] += np.log(max(pred, 1e-300))
        self._cell_lw -= self._cell_lw.max()

    def weights(self):
        w = np.exp(self._cell_lw - logsumexp(self._cell_lw))
        return w / w.sum()

    def mean(self):
        w = self.weights()
        mt = ml = 0.0
        for j, row in enumerate(self._cells):
            for ri, f in enumerate(row):
                a, b = f.mean()
                mt += w[j, ri] * a
                ml += w[j, ri] * b
        return float(mt), float(ml)

    def close_anchor(self, *, seed=0):
        """Marginalize retention out, mutate the mixture IN PLACE, update S.
        Returns (the same mixture object, r̂)."""
        w = self.weights()
        r_hat = float(np.clip((w * np.array(R_GRID)[None, :]).sum(),
                              0.05, 0.999))
        rng = np.random.default_rng(seed + 7)
        for j, row in enumerate(self._cells):
            wj = w[j].sum()
            wr = w[j] / wj if wj > 0 else np.full(len(row), 1.0 / len(row))
            base = self._mix.strata[j]
            theta = np.concatenate([f.theta for f in row])
            ell = np.concatenate([f.ell for f in row])
            pw = np.concatenate([o * f.w for o, f in zip(wr, row)])
            pw /= pw.sum()
            u = (rng.random() + np.arange(base.N)) / base.N
            idx = np.minimum(np.searchsorted(np.cumsum(pw), u), pw.size - 1)
            base.theta = theta[idx]
            base.ell = ell[idx]
            base.w = np.full(base.N, 1.0 / base.N)
            self._mix.log_w[j] = np.log(max(wj, 1e-300))
        self._mix.log_w -= self._mix.log_w.max()
        self._ss_num += self._dt ** 2
        self._ss_den += self._dt * (-np.log(r_hat))
        self.S = float(self._ss_num / max(self._ss_den, 1e-9))
        self.history.append((self._dt, r_hat))
        self._cells = None
        self._strata = None
        return self._mix, r_hat
