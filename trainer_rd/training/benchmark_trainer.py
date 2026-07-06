"""Step 7 — benchmark campaign (plan §12). Trials-to-mastery across policies and
learner models, with misspecification stress (PROJECT_MEMORY.md F13).

Single-task harness for a clean trials-to-mastery metric: a learner who is BOTH
biased and sub-skill is trained by each policy; the filter tracks belief; we
stop when the learner's TRUE state crosses mastery (σ ≤ σ* AND |t| ≤ t*). This
measures actual learning efficiency, not the filter's self-assessment.

Policies (all read only the filter's posterior mean, as a real trainer must):
  tier2        — mode-conditional (bias if |t̂|>t*, else 85%-point skill);
  tier1        — myopic greedy (constrained), Eq. Qgreedy;
  staircase85  — always the 85%-point, alternating sides (skill-only baseline);
  measure_opt  — place at the criterion (~50% acc) — the measurement-optimal
                 placement that is WRONG for training (plan caveat §6.4);
  random       — random items (null baseline).

Learner models: soft (well-specified), hard (rule ablation). Filter rates may be
misspecified by a factor (F13/D14).
"""
from __future__ import annotations

import numpy as np

# M17 D25: benchmark mastery targets = v15 cuts (v14 numbers in M10–M16
# records are v14-conditioned; do not compare across without noting this)
from training.bank_adapter import (BankAdapter, TaskCandidates, SIGMA_INF,
                                   ELL_STAR_V15 as ELL_STAR,
                                   SIGMA_STAR_V15 as SIGMA_STAR)
from training.bridge_conventions import SKILL_MODE_MULTIPLIER, engine_to_plan
from training.learner_sim import Learner, LearnerParams
from training.training_filter import TaskFilter
from training.trainer_greedy import RewardWeights, greedy_select
from training.trainer_policy import (expected_skill_weight, bias_correction_score,
                            bias_probe_score)

POLICIES = ("tier3", "tier2", "tier1", "staircase85", "measure_opt", "random")
T_STAR = 0.30


def _nearest(s, target, y=None, want=None):
    if want is not None and y is not None:
        m = np.where(y == want)[0]
        if m.size:
            return int(m[np.argmin(np.abs(s[m] - target))])
    return int(np.argmin(np.abs(s - target)))


def _select(policy, filt, cands, state, rng, greedyW):
    sig_hat, t_hat = engine_to_plan(*filt.mean())
    s, y = cands.s_mean, cands.y_star
    if policy == "random":
        return int(rng.integers(len(cands)))
    if policy == "measure_opt":
        return _nearest(s, t_hat)                     # at the criterion
    if policy == "staircase85":
        side = 1 if (state["k"] % 2 == 0) else -1
        return _nearest(s, t_hat + side * SKILL_MODE_MULTIPLIER * sig_hat,
                        y, 1 if side > 0 else 0)
    if policy == "tier1":
        idx, _ = greedy_select(filt, cands, greedyW,
                               label_balance=state["bal"], constrain_balance=True)
        return idx
    if policy == "tier3":
        from training.trainer_rollout import tier3_select
        return tier3_select(filt, cands, weights=greedyW, t_star=T_STAR,
                            rng=rng, label_balance=state["bal"],
                            step_parity=state["k"] % 2)
    # tier2: mode-conditional, F20 noise-aware selection (the naive baselines
    # above keep nearest-to-target placement — that contrast is the point)
    if abs(t_hat) > T_STAR:                            # bias mode (dual control)
        want = 0 if state["bal"] > 0 else 1
        m = np.where(y == want)[0]
        if m.size == 0:
            m = np.arange(len(s))
        if state["k"] % 2 == 0:                        # corrective half
            sc = bias_correction_score(filt, cands)    # expected |t|-reduction
        else:                                          # probe half (track t̂)
            sc = bias_probe_score(s, cands.s_sd, sig_hat, t_hat)
        return int(m[np.argmax(sc[m])])
    side = 1 if (state["k"] % 2 == 0) else -1          # skill: max E[w], boundary-
    want = 1 if side > 0 else 0                        # centered, mirror-paired (M10)
    m = np.where(y == want)[0]
    if m.size == 0:
        m = np.arange(len(s))
    m_eff = SKILL_MODE_MULTIPLIER
    last = state.get("last_s")
    if last is not None and (last > 0) != (side > 0):
        m_eff = abs(last) / max(sig_hat, 1e-6)
    ew = expected_skill_weight(s[m], cands.s_sd[m], sig_hat, 0.0,
                               side=side, m=m_eff, rho=0.6)
    idx = int(m[np.argmax(ew)])
    state["last_s"] = float(s[idx])
    return idx


def run_one(policy, task, *, rule="soft", rate_mult=1.0, sigma0=1.5, t0=0.8,
            budget=400, Npart=400, seed=0, pool=None, filter_kwargs=None):
    """filter_kwargs (M15): opt-in TaskFilter extras (e.g. exact_kernel=True);
    None ⇒ bit-identical to the pinned M10 campaign."""
    sig_star, ell_star = SIGMA_STAR[task], ELL_STAR[task]
    sig_inf = 0.82 * sig_star                          # reachable mastery
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=sig_inf,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule=rule)
    lnr = Learner([sigma0], [t0], tp, seed=seed)
    fp = LearnerParams(alpha_t=0.2 * rate_mult, alpha_sigma=0.06 * rate_mult,
                       sigma_inf=sig_inf, q_t=0.04, q_sigma=0.02, rho=0.6,
                       rule=("soft" if rule == "soft" else "hard"))
    rng = np.random.default_rng(1000 + seed)
    filt = TaskFilter(0.0 + 0.3 * rng.standard_normal(Npart),
                      0.0 + 0.3 * rng.standard_normal(Npart), fp, seed=seed,
                      **(filter_kwargs or {}))
    greedyW = RewardWeights()
    state = {"k": 0, "bal": 0}
    for k in range(budget):
        state["k"] = k
        idx = _select(policy, filt, pool, state, rng, greedyW)
        s, s_sd, y_star = float(pool.s_mean[idx]), float(pool.s_sd[idx]), int(pool.y_star[idx])
        # learner experiences the segment's true (unknown) signal (F20);
        # the filter sees only (s_mean, s_sd) and attenuates accordingly
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        state["bal"] += 1 if y_star == 1 else -1
        if lnr.sigma[0] <= sig_star and abs(lnr.t[0]) <= T_STAR:
            return k + 1
    return budget                                      # censored at budget


def bootstrap_ci(vals, *, n_boot=2000, seed=7):
    """Percentile bootstrap 95% CI of the mean."""
    v = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def run_benchmark(*, task=2, n_seeds=20, rules=("soft", "hard"),
                  rate_mults=(1.0, 2.0), budget=400, pool_size=600,
                  return_raw=False):
    """Campaign over policies × learner rules × rate misspecification.

    Returns {(rule, rm, policy): (mean, sd, n_censored)}. Means of cells with
    n_censored > 0 are biased LOW (runs truncated at `budget`) — always report
    the censored count alongside. With return_raw=True also returns the raw
    per-seed array dict, seed-aligned across policies (common random numbers),
    for paired contrasts via `paired_diff_ci`."""
    ad = BankAdapter()
    full = ad.candidates(task, feedback_safe=True)
    sub = np.random.default_rng(0).choice(len(full), size=pool_size, replace=False)
    pool = TaskCandidates(task, full.seg_id[sub], full.s_mean[sub], full.s_sd[sub],
                          full.y_star[sub], full.margin[sub], full.coherent[sub])
    out, raw = {}, {}
    for rule in rules:
        for rm in rate_mults:
            for pol in POLICIES:
                vals = np.array([run_one(pol, task, rule=rule, rate_mult=rm,
                                         budget=budget, seed=s, pool=pool)
                                 for s in range(n_seeds)])
                raw[(rule, rm, pol)] = vals
                out[(rule, rm, pol)] = (float(vals.mean()), float(vals.std()),
                                        int(np.sum(vals >= budget)))
    return (out, raw) if return_raw else out


def paired_diff_ci(raw, key_a, key_b, **kw):
    """Mean of per-seed differences A−B with bootstrap CI. Seeds are shared
    across policies (common random numbers), so this is ~3× tighter than the
    unpaired contrast of cell means."""
    d = raw[key_a].astype(np.float64) - raw[key_b].astype(np.float64)
    lo, hi = bootstrap_ci(d, **kw)
    return float(d.mean()), lo, hi


if __name__ == "__main__":
    res, raw = run_benchmark(n_seeds=15, return_raw=True)
    print(f"{'rule':>5} {'×rt':>4} {'policy':>12} {'mean_n':>7} {'sd':>6} "
          f"{'95% CI':>16} {'cens':>4}")
    for key, (mu, sd, cens) in res.items():
        lo, hi = bootstrap_ci(raw[key])
        print(f"{key[0]:>5} {key[1]:>4.1f} {key[2]:>12} {mu:7.1f} {sd:6.1f} "
              f"[{lo:6.1f},{hi:6.1f}] {cens:4d}")
    print("\npaired tier2 − staircase85 (same seeds):")
    for rule in ("soft", "hard"):
        for rm in (1.0, 2.0):
            m, lo, hi = paired_diff_ci(raw, (rule, rm, "tier2"),
                                       (rule, rm, "staircase85"))
            print(f"  {rule} ×{rm:.1f}: {m:+6.1f}  CI [{lo:+6.1f},{hi:+6.1f}]")
