"""Mode-B binary-certification engine — DEPRECATED for Paper 1.

F3.1 (2026-05-15): relocated out of core_mcmc.py.  Under the Multi-AUROC
Precision Protocol reframe (2026-05-15), Mode-B (binary PASS/FAIL
certification with the boundary prior, Berger-IUT joint stopping, Fisher
information guard, and n_min guard) is NOT the Paper-1 production path.
Paper 1 uses Mode-A (`core_mcmc.run_session_mcmc_auroc` + `choose_item`
Global-EV item selection + per-domain AUROC HW<δ stopping).

This module is preserved verbatim (no behavioural change) as groundwork
for Paper 2 (binary credentialing with a prospectively validated expert
panel).  It imports the shared SMC/MCMC primitives from `core_mcmc`; the
boundary-prior `l_prior_mean` kwarg remains an optional, Mode-A-inert
parameter of the shared `make_state_hier`/`sample_prior_hier_K`/
`log_prior_hier` functions in core_mcmc.

Scope note: the Mode-B *brute* arm
(`core_mcmc_brute_k.run_session_mcmc_brute_k_cert`) is dispatched to from
`run_session_mcmc_certification(method="brute")` and remains a labelled
Mode-B section inside `core_mcmc_brute_k.py` (it is already an isolated
named function; relocating it would add churn without separation benefit).

Provenance: this code carries the FIX-T0.5/T1.1/T1.3/T1.6/T1.7/T1.8/
T1.9/T1.10/T1.11 + F0.1 history from the Wave-1..Phase-6 remediation.
See CHANGELOG.md.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core_mcmc import (
    LAPSE_RATE,
    Z_BUFFER,
    SIGNAL_GRID,
    make_state_hier,
    simulate_response,
    update,
    ess,
    resample_and_rejuvenate,
)
from auroc import auroc_from_l


# ───────────── certification stopping: pass probabilities ─────────────

def _pass_probs_hier(state, l_star):
    """P(l_k > l*_k) for each domain, computed from weighted particle cloud.

    Returns shape (K,) array.
    """
    w = state["w"]
    return np.array([(w * (state["l"][:, k] > l_star[k])).sum()
                     for k in range(state["l"].shape[1])])


def _pass_prob_joint(state, l_star, active_domains=None):
    """P(all active l_k > l*_k | data) from the joint particle cloud. FIX-T1.9 IUT."""
    w = state["w"]
    doms = (list(range(state["l"].shape[1]))
            if active_domains is None else list(active_domains))
    all_pass = np.ones(len(w), dtype=bool)
    for k in doms:
        all_pass &= (state["l"][:, k] > l_star[k])
    return float((w * all_pass).sum())


# ───────────── certification item selection ─────────────
# FIX-T0.5: _kl_vec and choose_item_kl have been REMOVED.  The KL criterion
# conditioned on the posterior mean theta_hat_k as a fixed point estimate;
# whenever theta_hat differed from theta_true, the selected items had near-zero
# Fisher information for l_k (P(y=1) collapsed to ~1).  See STATE_LOG §3.1.
# Item selection now goes through choose_item() (expected posterior-variance
# reduction) which marginalises over the FULL posterior in (t, l) for ALL
# domains.
#
# FIX-T1.10: certification sessions use choose_item_cert() (boundary-targeted
# Fisher information at l*_k) instead of choose_item() (global EV reduction).
# choose_item() is retained for AUROC sessions where overall posterior variance
# reduction is the right objective.


# F0.1: Fisher info for ℓ under spike-paper lapse mixture P(y=1) = λ + (1−2λ)Φ(z).
# ∂P/∂z = (1−2λ)·φ(z), and ∂z/∂ℓ = z, so I_ℓ ∝ z²(1−2λ)²φ(z)²/[P(1−P)].
_ONE_MINUS_TWO_LAPSE_SQ = (1.0 - 2.0 * LAPSE_RATE) ** 2


def choose_item_cert(state, bank_signals, active_domains, l_star):
    """Boundary-aware item selection: maximise Fisher info for l at l*_k,
    marginalised over the theta particle posterior. FIX-T1.10 + F0.1.

    Correct Fisher info for the discrimination parameter l under spike-paper
    lapse mixture P(y=1) = λ + (1 − 2λ)·Φ(z):
        I_l(s; l*, t) = (1 − 2λ)² φ(z)² z² / (P_yes · P_no)
    where z = exp(l*)(s + t).  Note z² → 0 at z=0 (the binary boundary),
    so the optimal item is at z = ±1 (i.e. s + t = ±1/exp(l*)), not at the
    50%-point.  Averaging over the weighted particle cloud avoids the
    degenerate point-estimate case where s = −theta_hat kills all l-information.
    Cross-domain selection is weighted by p_k(1−p_k) (classification uncertainty)
    so questions route to undecided domains and stop when a decision is settled.
    Falls back to SIGNAL_GRID when bank_signals is None.
    """
    candidates = (bank_signals if bank_signals is not None
                  else [SIGNAL_GRID] * state["t"].shape[1])
    w = state["w"]                   # (N,)
    pass_probs = _pass_probs_hier(state, l_star)
    best_val = -np.inf
    best_k = active_domains[0]
    best_s = float(np.asarray(candidates[best_k])[0])
    for k in active_domains:
        sigs = np.asarray(candidates[k], dtype=float)  # (S,)
        t_k = state["t"][:, k]                          # (N,)
        l_star_k = float(l_star[k])
        # z_{s,i} = exp(l*) (s + t_i),  shape (S, N)
        z = np.exp(l_star_k) * (sigs[:, None] + t_k[None, :])
        phi_z = norm.pdf(z)
        p_yes = LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z)
        p_no = 1.0 - p_yes
        # Particle-averaged Fisher info for l; z² ensures peak at |z|=1
        fi_ij = _ONE_MINUS_TWO_LAPSE_SQ * phi_z ** 2 * z ** 2 / np.maximum(p_yes * p_no, 1e-9)
        fi_k = (fi_ij * w[None, :]).sum(axis=1)  # (S,)
        p_k = float(pass_probs[k])
        val = fi_k * (p_k * (1.0 - p_k))
        idx = int(np.argmax(val))
        if val[idx] > best_val:
            best_val = val[idx]
            best_k = k
            best_s = float(sigs[idx])
    return best_k, best_s


# ───────────── certification session driver ─────────────

def run_session_mcmc_certification(
    method, true_params, K, r_assumed,
    l_star,
    max_q=400,
    stop_thresh=None,
    alpha=0.05,
    delta_indiff=0.15,
    N=500,
    seed=0,
    n_mh_steps=15,
    proposal_scale=0.5,
    ess_threshold_frac=0.5,
    bank_signals=None,
    virtual_pairs=None,        # FIX-T1.1: deprecated; raises DeprecationWarning if non-None
    brute_proposal_scale=None,
    Sigma_l=None,              # FIX-T1.6: unstructured prior covariance for l block
    Sigma_t=None,              # FIX-T1.6: unstructured prior covariance for t block
    z_buffer=Z_BUFFER,         # FIX-T1.3: MCSE-buffered stopping
    n_min_guard=20,            # FIX-T1.7: min questions per domain before FAIL can fire
    l_prior_mean=None,         # FIX-T1.8: prior mean for l block (pass l_star for boundary prior)
    use_iut_stopping=True,     # FIX-T1.9: joint IUT stopping instead of per-domain Sidak
    iut_rule="joint",          # unified-merge: "joint" (audit-endorsed default) | "berger_marginals"
    fisher_imin=0.0,           # FIX-T1.11: Fisher info guard threshold (0 → use n_min_guard only)
):
    """Binary pass/fail certification with EV-optimal items and Sidak stopping.

    FIX-T0.5/T1.1/T1.3/T1.6/T1.7: KL item selection removed; virtual cert removed;
    MCSE-buffered stopping added; unstructured prior Sigma supported;
    N_min guard and Corr_l prior scale fix applied.

    Args:
        l_star:        (K,) array of log-skill thresholds; pass iff l_k > l*_k.
        stop_thresh:   per-domain stopping probability.  Default Sidak correction
                       for K domains: (1 - alpha)^(1/K).
        alpha:         familywise error rate target (default 0.05 -> 95% confidence).
        delta_indiff:  retained for backward-compatibility; unused by EV item
                       selection.
        Sigma_l:       optional (K, K) PD matrix; unstructured prior covariance
                       on l.  If supplied, OVERRIDES r_assumed for the l block.
        Sigma_t:       optional (K, K) PD matrix for the t block.  Defaults to
                       Sigma_l when only Sigma_l is provided (matching the
                       symmetry assumption used historically).
        z_buffer:        MCSE multiplier for stopping (FIX-T1.3).  PASS requires
                         pass_probs[k] - z_buffer * mcse_k >= stop_thresh; FAIL
                         requires pass_probs[k] + z_buffer * mcse_k <= 1 - stop_thresh.
        n_min_guard:     Minimum number of questions per domain before FAIL is allowed
                         (FIX-T1.7).  PASS is still allowed before n_min_guard.
                         Set to 0 to disable (backward-compat).
        l_prior_mean:    (K,) array; prior mean for the l block (FIX-T1.8).  When set
                         to l_star, the prior is centred at the classification boundary,
                         giving P(PASS | prior) = 0.50 exactly. Requires Sigma_l to be
                         scaled to prior_sigma^2 * Corr_l (e.g. 0.25 * Corr_l for sigma=0.5).
        use_iut_stopping: If True (default), PASS fires via the IUT conjunction
                         rule; if False, per-domain Sidak-style PASS (legacy).
        iut_rule:        Which IUT conjunction statistic gates a joint PASS
                         (only when use_iut_stopping=True). Unified-merge
                         decision (2026-05-18), default "joint":
                         • "joint"            — PASS all active iff
                           P(all active l_k > l*_k | data) (from the JOINT
                           particle cloud, _pass_prob_joint) − z_buffer·mcse
                           ≥ stop_thresh. Endorsed by the Bayesian-expert
                           audit as the correct rule (more conservative).
                         • "berger_marginals" — FIX-T1.9 Berger (1982)
                           min-of-marginals: PASS iff EVERY active per-domain
                           marginal − z_buffer·mcse ≥ stop_thresh. Preserved
                           verbatim; selectable for Paper-2 comparison.
                         Paper-1 is unaffected (Mode-A has no IUT); Mode-B is
                         deprecated for Paper-1. See CHANGELOG (FIX-T1.9
                         dispute) and tests/mode_b/test_iut_rule.py.
        fisher_imin:     Fisher information guard threshold (FIX-T1.11).  FAIL for domain k
                         is blocked until domain_fi[k] >= fisher_imin.  Set to 0 (default)
                         to use n_min_guard only.  With epsilon=0.10, sigma=0.5:
                         I_min = 1/0.01 - 1/0.25 = 96.

    Returns dict with keys:
        n_questions, per_domain_n, decisions, stopped_early, active_at_end,
        true_auroc, pass_probs_final, virtual_granted, stop_thresh_used,
        mean_acceptance_rate, n_rejuvenations, mcse_final, z_buffer, n_min_guard,
        domain_fi, use_iut_stopping.

        virtual_granted is retained as an all-zero array of length K for
        backward-compat (FIX-T1.1).
    """
    if iut_rule not in ("joint", "berger_marginals"):
        raise ValueError(
            f"iut_rule must be 'joint' (default) or 'berger_marginals'; "
            f"got {iut_rule!r}.")

    # FIX-T1.1: virtual_pairs deprecation
    if virtual_pairs is not None:
        import warnings
        warnings.warn(
            "virtual_pairs is no longer supported; the joint particle posterior "
            "subsumes virtual certification via the prior covariance.  Argument "
            "ignored.",
            DeprecationWarning,
            stacklevel=2,
        )

    if method == "brute":
        from core_mcmc_brute_k import run_session_mcmc_brute_k_cert
        ps = 1.5 if brute_proposal_scale is None else float(brute_proposal_scale)
        # FIX-T1.7/T1.9/T1.11 parity: brute receives the same guards and IUT
        # stopping rule as hier, so direct comparison reflects only the prior
        # (independent vs hierarchical), not stopping-rule asymmetry.
        # l_prior_mean is hier-only; brute retains its N(0,1) baseline prior.
        return run_session_mcmc_brute_k_cert(
            true_params=true_params, K=K, l_star=l_star, max_q=max_q,
            stop_thresh=stop_thresh, alpha=alpha, delta_indiff=delta_indiff,
            N=N, seed=seed, n_mh_steps=n_mh_steps, proposal_scale=ps,
            ess_threshold_frac=ess_threshold_frac, bank_signals=bank_signals,
            virtual_pairs=None,
            n_min_guard=n_min_guard,
            use_iut_stopping=use_iut_stopping,
            iut_rule=iut_rule,
            fisher_imin=fisher_imin,
        )

    if method != "hier":
        raise ValueError(
            f"Unknown method {method!r}; valid: 'hier', 'brute'.  "
            "FIX-T0.5: 'brute_joint' has been removed."
        )

    rng = np.random.default_rng(seed)
    assert len(true_params) == 2 * K
    l_star = np.asarray(l_star, dtype=float)

    if stop_thresh is None:
        # Sidak: per-domain threshold = (1-alpha)^(1/K) so that
        # P(>= 1 false stop across K independent tests) <= alpha
        stop_thresh = (1.0 - alpha) ** (1.0 / K)

    state = make_state_hier(N, K, r_assumed, rng,
                             Sigma_l=Sigma_l, Sigma_t=Sigma_t,
                             l_prior_mean=l_prior_mean)

    true_l = np.array([true_params[2 * k + 1] for k in range(K)])
    true_auroc = auroc_from_l(true_l)

    active_domains = list(range(K))
    decisions = np.zeros(K, dtype=int)
    per_domain_n = np.zeros(K, dtype=int)
    domain_fi = np.zeros(K, dtype=float)       # FIX-T1.11: accumulated Fisher info
    virtual_granted = np.zeros(K, dtype=bool)  # FIX-T1.1: kept for backward-compat
    n_q = 0
    accept_rates = []
    mcse_final = np.zeros(K, dtype=float)

    for _ in range(max_q):
        if not active_domains:
            break

        # FIX-T1.10: boundary-targeted item selection
        k, s = choose_item_cert(state, bank_signals, active_domains, l_star)

        # FIX-T1.11 + F0.1: accumulate Fisher info for l at l*_k under spike-paper lapse.
        theta_hat_k = float(state["w"] @ state["t"][:, k])
        z_fi = float(np.exp(l_star[k]) * (s + theta_hat_k))
        phi_fi = float(norm.pdf(z_fi))
        p_fi = float(LAPSE_RATE + (1.0 - 2.0 * LAPSE_RATE) * norm.cdf(z_fi))
        domain_fi[k] += (_ONE_MINUS_TWO_LAPSE_SQ * phi_fi ** 2 * z_fi ** 2
                         / max(p_fi * (1.0 - p_fi), 1e-9))

        t_true = true_params[k * 2]
        l_true = true_params[k * 2 + 1]
        y = simulate_response(s, t_true, l_true, rng)
        update(state, k, s, y)
        per_domain_n[k] += 1
        n_q += 1

        if ess(state["w"]) < ess_threshold_frac * N:
            ar = resample_and_rejuvenate(state, rng, n_mh_steps, proposal_scale)
            accept_rates.append(ar)

        pass_probs = _pass_probs_hier(state, l_star)

        # FIX-T1.3: MCSE-buffered stopping
        n_eff = ess(state["w"])
        mcse = np.sqrt(np.maximum(pass_probs * (1.0 - pass_probs), 0.0) / max(n_eff, 1.0))
        mcse_final = mcse

        newly_decided = []

        # Unified-merge IUT toggle (2026-05-18). FIX-T1.9 dispute resolved:
        # default to the Bayesian-expert-audit-endorsed JOINT-posterior rule;
        # Berger (1982) min-of-marginals preserved verbatim as a toggle.
        # Paper-1 unaffected (Mode-A has no IUT); Mode-B is Paper-2 scope.
        if use_iut_stopping and active_domains:
            if iut_rule == "joint":
                # P(all active l_k > l*_k | data) from the JOINT particle
                # cloud (correlations retained), MCSE-buffered. This is the
                # rule the audit holds correct; it is strictly more
                # conservative than Berger's min-of-marginals.
                p_joint = _pass_prob_joint(state, l_star, active_domains)
                mcse_joint = np.sqrt(
                    max(p_joint * (1.0 - p_joint), 0.0) / max(n_eff, 1.0))
                conjunction_pass = (
                    p_joint - z_buffer * mcse_joint >= stop_thresh)
            else:  # "berger_marginals" — FIX-T1.9 verbatim
                # Berger 1982 IUT: PASS the conjunction iff EVERY active
                # per-domain marginal rejects its own H0_k (min-of-marginals).
                conjunction_pass = all(
                    pass_probs[k_act] - z_buffer * mcse[k_act] >= stop_thresh
                    for k_act in active_domains
                )
            if conjunction_pass:
                for k_act in list(active_domains):
                    decisions[k_act] = 1
                    newly_decided.append(k_act)
        else:
            # Legacy: per-domain PASS (use with Sidak-corrected stop_thresh)
            for k_act in list(active_domains):
                if pass_probs[k_act] - z_buffer * mcse[k_act] >= stop_thresh:
                    decisions[k_act] = 1
                    newly_decided.append(k_act)

        # Per-domain FAIL with Fisher info guard (FIX-T1.11)
        for k_act in list(active_domains):
            if k_act in newly_decided:
                continue
            if pass_probs[k_act] + z_buffer * mcse[k_act] <= 1.0 - stop_thresh:
                fi_ok = (domain_fi[k_act] >= fisher_imin) if fisher_imin > 0 else True
                if fi_ok and per_domain_n[k_act] >= n_min_guard:
                    decisions[k_act] = -1
                    newly_decided.append(k_act)

        for k_dec in newly_decided:
            active_domains.remove(k_dec)

    pass_probs_final = _pass_probs_hier(state, l_star)

    return {
        "n_questions": int(n_q),
        "per_domain_n": per_domain_n,
        "decisions": decisions,
        "stopped_early": len(active_domains) == 0,
        "active_at_end": list(active_domains),
        "true_auroc": true_auroc,
        "pass_probs_final": pass_probs_final,
        "virtual_granted": virtual_granted,  # FIX-T1.1: deprecated, all zeros
        "stop_thresh_used": float(stop_thresh),
        "mean_acceptance_rate": float(np.mean(accept_rates)) if accept_rates else float("nan"),
        "n_rejuvenations": len(accept_rates),
        "mcse_final": mcse_final,                    # FIX-T1.3
        "z_buffer": float(z_buffer),               # FIX-T1.3
        "n_min_guard": int(n_min_guard),           # FIX-T1.7
        "domain_fi": domain_fi,                    # FIX-T1.11
        "use_iut_stopping": bool(use_iut_stopping),  # FIX-T1.9
    }
