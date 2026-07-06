"""M11 — Tier-3 short-horizon Monte Carlo rollout policy (plan §6.3, "method 3").

Approximates the Bellman bracket E[R] + γ·V*(b') by averaging the discounted
returns of L simulated rollouts of depth H under the tier-2 base policy π0:

  for each candidate s:  for l = 1..L:
      θ ~ b_k (the rollout's "true" learner state)
      trial k   : present s; y ~ Z(·|θ,s); θ' ~ T(θ,s,y,y*); R_k = R(θ,θ')
      belief    : b ← Bayes(b, s, y)
      trials k+1..k+H−1 : s_j = π0(b_j); same simulate/record/update
      G^(l) = Σ_j γ^j R_j
  Q̂(s) = mean_l G^(l);  pick argmax over a tier-1-shortlisted candidate set.

Receding-horizon bias decays as γ^H (γ<1) / tail-reward beyond k+H (γ=1).
Cost O(|shortlist|·L·H·n_ro) — all rollouts are run as ONE numpy batch
(R = shortlist×L parallel rollouts), so a trial costs a few ms.

Design notes (scratch findings honored):
  * Shortlist by the CONSTRAINED tier-1 greedy Q (label balance — F6b);
    rollouts then rank the shortlist by lookahead value.
  * π0 is the benchmark tier-2 rule in its M10 form: bias mode probes near
    t̂ with balanced labels; skill mode is boundary-centered (s=0) at the
    85%-point, side-alternating (F23: placement sets the served-stream
    midpoint — the rollout's base policy must not herd the criterion).
  * The rollout dynamics are the FILTER's assumed LearnerParams (the fitted
    model), exactly as the plan prescribes ("simulates ... using the fitted
    dynamics"); process noise is sampled, not dropped (the lookahead should
    see noise-driven risk, e.g. bias overshoot — F24).
  * s_sd (F20): responses are simulated at s_real ~ N(s, s_sd²); the belief
    reweight uses the attenuated likelihood — same physics as the benchmark.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from training.bridge_conventions import LAPSE_RATE, SKILL_MODE_MULTIPLIER
from training.trainer_greedy import RewardWeights, _expected_reward


# ───────────── defaults (cost knobs) ─────────────

H_DEFAULT = 6          # rollout depth (receding horizon)
L_DEFAULT = 8          # rollouts per candidate
N_RO = 96              # belief particles inside a rollout
N_SHORT = 12           # tier-1 shortlist size
GAMMA = 1.0            # undiscounted within the short horizon
POOL_CAP = 160         # π0's item pool inside rollouts


def _p_yes_engine(s, theta, ell, s_sd=0.0):
    """Lapse-mixture P(yes) in engine coords with s_sd attenuation (F20).
    Broadcasts s (…,) against theta/ell (…,)."""
    el = np.exp(ell)
    z = el * (s + theta)
    if np.any(s_sd):
        z = z / np.sqrt(1.0 + (el * s_sd) ** 2)
    return LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * ndtr(z)


def _systematic_pick(w, n, rng):
    """Systematic resampling indices: n draws from weights w."""
    u = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), u)
    return np.minimum(idx, w.size - 1)


class _RolloutPool:
    """π0's candidate pool inside rollouts: per-label sorted signal arrays
    for O(log P) vectorized nearest-target lookup."""

    def __init__(self, s, s_sd, y, cap, rng):
        if s.size > cap:
            keep = np.sort(rng.choice(s.size, size=cap, replace=False))
            s, s_sd, y = s[keep], s_sd[keep], y[keep]
        self.by_label = {}
        for lab in (0, 1):
            m = np.where(y == lab)[0]
            if m.size == 0:
                m = np.arange(s.size)
            order = m[np.argsort(s[m])]
            self.by_label[lab] = (s[order], s_sd[order])

    def nearest(self, targets, labels):
        """Vectorized nearest item per rollout. targets (R,), labels (R,)
        → s (R,), s_sd (R,)."""
        out_s = np.empty_like(targets)
        out_sd = np.empty_like(targets)
        for lab in (0, 1):
            m = labels == lab
            if not m.any():
                continue
            ss, sd = self.by_label[lab]
            j = np.searchsorted(ss, targets[m])
            j = np.clip(j, 1, ss.size - 1)
            left_closer = (targets[m] - ss[j - 1]) <= (ss[j] - targets[m])
            j = np.where(left_closer, j - 1, j)
            out_s[m], out_sd[m] = ss[j], sd[j]
        return out_s, out_sd


def _pi0_targets(th_b, el_b, w_b, step_parity, t_star):
    """Tier-2 base policy, vectorized over R rollouts: mode + placement.
    Returns (targets (R,), labels (R,)) in signal space."""
    mt = (w_b * th_b).sum(axis=1)
    ml = (w_b * el_b).sum(axis=1)
    sig_hat = np.exp(-ml)
    t_hat = -mt
    side = 1 if step_parity == 0 else -1
    # skill mode: boundary-centered 85%-point, mirror-side alternation (F23)
    skill_target = side * SKILL_MODE_MULTIPLIER * sig_hat
    skill_label = np.full(t_hat.shape, 1 if side > 0 else 0)
    # bias mode: probe near the criterion, balanced labels by parity
    bias_target = t_hat
    bias_label = np.full(t_hat.shape, 1 if step_parity == 0 else 0)
    biasy = np.abs(t_hat) > t_star
    return (np.where(biasy, bias_target, skill_target),
            np.where(biasy, bias_label, skill_label).astype(int))


def tier3_select(filt, cands, *, weights: RewardWeights = None,
                 t_star=0.30, H=H_DEFAULT, L=L_DEFAULT, n_ro=N_RO,
                 n_short=N_SHORT, gamma=GAMMA, pool_cap=POOL_CAP,
                 rng=None, label_balance=0, step_parity=0):
    """Pick the candidate index maximizing the H-step MC rollout value.

    filt: TaskFilter (current belief + assumed dynamics). cands: a
    TaskCandidates view. Returns int index into cands."""
    rng = np.random.default_rng() if rng is None else rng
    weights = weights or RewardWeights(beta_r=0.0)
    p = filt.p

    # ── 1. tier-1 shortlist (constrained greedy Q, F6b balance) ──
    Q1 = _expected_reward(filt, cands.s_mean,
                          cands.y_star.astype(np.float64), weights,
                          s_sd=cands.s_sd)
    allowed = np.ones(len(cands), dtype=bool)
    if label_balance >= 1:
        allowed &= (cands.y_star == 0)
    elif label_balance <= -1:
        allowed &= (cands.y_star == 1)
    if not allowed.any():
        allowed[:] = True
    cand_idx = np.where(allowed)[0]
    if cand_idx.size > n_short:
        cand_idx = cand_idx[np.argsort(Q1[cand_idx])[::-1][:n_short]]
    C = cand_idx.size
    R = C * L                                     # total parallel rollouts

    pool = _RolloutPool(cands.s_mean, cands.s_sd, cands.y_star, pool_cap, rng)

    # ── 2. rollout initial states ──
    # COMMON RANDOM NUMBERS across candidates: rollout l uses the SAME true
    # state and the same noise streams for every candidate, so Q̂ differences
    # reflect the candidate, not the Monte Carlo noise. Without this the
    # per-candidate signal (~one step's reward) is comparable to the rollout
    # noise SE and the argmax is a coin flip (caught by test_tier3 check 4).
    # Layout: rollout r = c*L + l ⇒ np.tile(x_L, C) aligns l across candidates.
    pick_L = _systematic_pick(filt.w, L, rng)
    pick = np.tile(pick_L, C)
    sig_true = np.exp(-filt.ell[pick])             # plan coords
    t_true = -filt.theta[pick]

    def _tile(x):
        """(L, …) noise → (R, …), same draws for every candidate (CRN)."""
        return np.tile(x, (C,) + (1,) * (x.ndim - 1)).reshape((R,) + x.shape[1:])
    # belief per rollout: one shared systematic subsample, copied R times
    base = _systematic_pick(filt.w, n_ro, rng)
    th_b = np.tile(filt.theta[base], (R, 1))       # (R, n_ro)
    el_b = np.tile(filt.ell[base], (R, 1))
    w_b = np.full((R, n_ro), 1.0 / n_ro)

    # step-0 items: each rollout presents its candidate
    s0 = np.repeat(cands.s_mean[cand_idx], L)
    sd0 = np.repeat(cands.s_sd[cand_idx], L)
    y0_star = np.repeat(cands.y_star[cand_idx], L).astype(np.float64)

    G = np.zeros(R)
    s_j, sd_j, ystar_j = s0, sd0, y0_star
    for j in range(H):
        # CRN noise for this step (shared across candidates)
        eps_s = _tile(rng.standard_normal(L))
        u_y = _tile(rng.random(L))
        eps_t = _tile(rng.standard_normal(L))
        eps_g = _tile(rng.standard_normal(L))
        # learner responds at the segment's true signal (F20)
        s_real = s_j + sd_j * eps_s
        z = (s_real - t_true) / sig_true
        p_yes = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * ndtr(z)
        y = (u_y < p_yes).astype(np.float64)
        # true-state transition through the fitted dynamics (with noise)
        delta = (p_yes - ystar_j) if p.rule == "soft" else (y - ystar_j)
        t_new = t_true + p.alpha_t * delta + p.q_t * eps_t
        d = np.abs(s_real - t_true) / sig_true
        wgt = np.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2) / (2.0 * p.rho ** 2))
        log_sig = np.log(sig_true)
        g_sig = -wgt * (log_sig - np.log(p.sigma_inf))
        sig_new = np.exp(log_sig + p.alpha_sigma * g_sig + p.q_sigma * eps_g)
        # reward = state improvement this trial (Eq. reward, β_r=0 here)
        G += (gamma ** j) * (weights.beta_t * (np.abs(t_true) - np.abs(t_new))
                             + weights.beta_sigma * (sig_true - sig_new))
        t_true, sig_true = t_new, sig_new
        if j == H - 1:
            break
        # belief update: reweight on y (attenuated likelihood), resample, propagate
        pb = _p_yes_engine(s_j[:, None], th_b, el_b, sd_j[:, None])
        like = np.where(y[:, None] == 1.0, pb, 1.0 - pb)
        w_b = w_b * like
        tot = w_b.sum(axis=1, keepdims=True)
        w_b = np.where(tot > 0, w_b / np.where(tot == 0, 1, tot), 1.0 / n_ro)
        ess = 1.0 / (w_b ** 2).sum(axis=1)
        for r in np.where(ess < 0.5 * n_ro)[0]:    # per-rollout resample
            idx = _systematic_pick(w_b[r], n_ro, rng)
            th_b[r], el_b[r] = th_b[r, idx], el_b[r, idx]
            w_b[r] = 1.0 / n_ro
        # propagate belief through T (soft: deterministic mean given p_yes)
        sig_b = np.exp(-el_b)
        t_b = -th_b
        pb_att = _p_yes_engine(s_j[:, None], th_b, el_b, sd_j[:, None])
        delta_b = (pb_att - ystar_j[:, None]) if p.rule == "soft" \
            else (y[:, None] - ystar_j[:, None])
        t_b = t_b + p.alpha_t * delta_b + p.q_t * _tile(
            rng.standard_normal((L, n_ro)))
        d_b = np.abs(s_j[:, None] - t_b) / sig_b
        w_w = np.exp(-((d_b - SKILL_MODE_MULTIPLIER) ** 2) / (2.0 * p.rho ** 2))
        log_s = np.log(sig_b)
        log_s = (log_s - p.alpha_sigma * w_w * (log_s - np.log(p.sigma_inf))
                 + p.q_sigma * _tile(rng.standard_normal((L, n_ro))))
        th_b, el_b = -t_b, -log_s
        # π0 chooses the next item per rollout
        targets, labels = _pi0_targets(th_b, el_b, w_b,
                                       (step_parity + j + 1) % 2, t_star)
        s_j, sd_j = pool.nearest(targets, labels)
        ystar_j = labels.astype(np.float64)

    Qhat = G.reshape(C, L).mean(axis=1)
    return int(cand_idx[int(np.argmax(Qhat))])


def _pooled_truth_arrays(filt):
    """Per-particle (θ, ℓ, w, σ_∞, learn) for the rollout's TRUE-state
    draws, mixture-honest (M25/F87): a SigmaInfMixtureFilter contributes
    each stratum's OWN ceiling and freezes its static stratum (the F84-ii
    lesson — the pooled shortcut credits static mass with progress it
    cannot make and trains every stratum toward the base ceiling)."""
    strata = getattr(filt, "strata", None)
    if strata is None:
        frozen = 0.0 if filt.p.rule == "static" else 1.0
        return (filt.theta, filt.ell, filt.w,
                np.full(filt.ell.shape, filt.p.sigma_inf),
                filt.learn * frozen)
    sinf, learn = [], []
    for f in strata:
        sinf.append(np.full(f.ell.shape, f.p.sigma_inf))
        learn.append(f.learn * (0.0 if f.p.rule == "static" else 1.0))
    return (filt.theta, filt.ell, filt.w,
            np.concatenate(sinf), np.concatenate(learn))


def rollout_item_q(filt, s_c, sd_c, y_c, *, pool_s, pool_sd, pool_y,
                   t_star=0.30, H=H_DEFAULT, L=L_DEFAULT, n_ro=N_RO,
                   gamma=GAMMA, pool_cap=POOL_CAP, rng=None,
                   step_parity=0, beta_t=1.0, beta_sigma=1.0):
    """M25 (F87) — item-level H-step rollout VALUE for each candidate in
    (s_c, sd_c, y_c): the non-myopic F84 successor. Returns Q̂ (n_c,).

    What the rollout prices that the myopic progress integral cannot (the
    F84 mechanism-3 finding): the INFORMATION EXTERNALITY of an item. The
    belief inside each rollout is updated on the simulated response, and
    π0's LATER placements are functions of that belief — an item that
    sharpens the belief steers every subsequent placement closer to the
    learner's true 85% point, and that future training value is credited
    into Q̂ through the simulated true-state improvements. Reward stays
    the plan's Eq. (reward) on the TRUE rollout states (no bar-referenced
    credit at the item level — the F84-iii rejected variant).

    Mixture honesty: true states are drawn from the POOLED posterior with
    per-particle (σ_∞, learn) from `_pooled_truth_arrays`, so static and
    per-stratum-ceiling mass evolves under ITS OWN dynamics; the interior
    belief uses the base-dynamics pooled cloud (it only drives π0 —
    documented approximation, same tier as tier3_select's belief).

    CRN across candidates is LOAD-BEARING (see tier3_select): rollout l
    shares its true-state draw and every noise stream across candidates.
    The caller supplies the shortlist (label- and mirror-filtered) and the
    π0 pool arrays; this function only ranks."""
    rng = np.random.default_rng() if rng is None else rng
    p = filt.p
    s_c = np.asarray(s_c, dtype=np.float64)
    C = s_c.size
    R = C * L
    theta, ell, w, sinf_all, learn_all = _pooled_truth_arrays(filt)
    pool = _RolloutPool(np.asarray(pool_s, dtype=np.float64),
                        np.asarray(pool_sd, dtype=np.float64),
                        np.asarray(pool_y), pool_cap, rng)

    pick_L = _systematic_pick(w, L, rng)
    pick = np.tile(pick_L, C)
    sig_true = np.exp(-np.clip(ell[pick], -40.0, 40.0))
    t_true = -theta[pick]
    sinf_true = sinf_all[pick]
    learn_true = learn_all[pick]

    def _tile(x):
        return np.tile(x, (C,) + (1,) * (x.ndim - 1)).reshape(
            (R,) + x.shape[1:])

    base = _systematic_pick(w, n_ro, rng)
    th_b = np.tile(theta[base], (R, 1))
    el_b = np.tile(ell[base], (R, 1))
    w_b = np.full((R, n_ro), 1.0 / n_ro)

    s_j = np.repeat(s_c, L)
    sd_j = np.repeat(np.asarray(sd_c, dtype=np.float64), L)
    ystar_j = np.repeat(np.asarray(y_c, dtype=np.float64), L)

    G = np.zeros(R)
    for j in range(H):
        eps_s = _tile(rng.standard_normal(L))
        u_y = _tile(rng.random(L))
        eps_t = _tile(rng.standard_normal(L))
        eps_g = _tile(rng.standard_normal(L))
        s_real = s_j + sd_j * eps_s
        z = (s_real - t_true) / sig_true
        p_yes = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * ndtr(z)
        y = (u_y < p_yes).astype(np.float64)
        delta = (p_yes - ystar_j) if p.rule == "soft" else (y - ystar_j)
        # static/frozen mass takes NO increment, including process noise
        t_new = t_true + learn_true * (p.alpha_t * delta + p.q_t * eps_t)
        d = np.abs(s_real - t_true) / sig_true
        wgt = np.exp(-((d - SKILL_MODE_MULTIPLIER) ** 2) / (2.0 * p.rho ** 2))
        log_sig = np.log(sig_true)
        g_sig = -wgt * (log_sig - np.log(sinf_true))
        sig_new = np.exp(log_sig + learn_true * (p.alpha_sigma * g_sig
                                                 + p.q_sigma * eps_g))
        G += (gamma ** j) * (beta_t * (np.abs(t_true) - np.abs(t_new))
                             + beta_sigma * (sig_true - sig_new))
        t_true, sig_true = t_new, sig_new
        if j == H - 1:
            break
        pb = _p_yes_engine(s_j[:, None], th_b, el_b, sd_j[:, None])
        like = np.where(y[:, None] == 1.0, pb, 1.0 - pb)
        w_b = w_b * like
        tot = w_b.sum(axis=1, keepdims=True)
        w_b = np.where(tot > 0, w_b / np.where(tot == 0, 1, tot), 1.0 / n_ro)
        ess = 1.0 / (w_b ** 2).sum(axis=1)
        for r in np.where(ess < 0.5 * n_ro)[0]:
            idx = _systematic_pick(w_b[r], n_ro, rng)
            th_b[r], el_b[r] = th_b[r, idx], el_b[r, idx]
            w_b[r] = 1.0 / n_ro
        sig_b = np.exp(-el_b)
        t_b = -th_b
        pb_att = _p_yes_engine(s_j[:, None], th_b, el_b, sd_j[:, None])
        delta_b = (pb_att - ystar_j[:, None]) if p.rule == "soft" \
            else (y[:, None] - ystar_j[:, None])
        t_b = t_b + p.alpha_t * delta_b + p.q_t * _tile(
            rng.standard_normal((L, n_ro)))
        d_b = np.abs(s_j[:, None] - t_b) / sig_b
        w_w = np.exp(-((d_b - SKILL_MODE_MULTIPLIER) ** 2) / (2.0 * p.rho ** 2))
        log_s = np.log(sig_b)
        log_s = (log_s - p.alpha_sigma * w_w * (log_s - np.log(p.sigma_inf))
                 + p.q_sigma * _tile(rng.standard_normal((L, n_ro))))
        th_b, el_b = -t_b, -log_s
        targets, labels = _pi0_targets(th_b, el_b, w_b,
                                       (step_parity + j + 1) % 2, t_star)
        s_j, sd_j = pool.nearest(targets, labels)
        ystar_j = labels.astype(np.float64)

    return G.reshape(C, L).mean(axis=1)
