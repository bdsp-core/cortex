"""Phase 9 K=7 joint hierarchical model — three variants for Kong-crowd
mitigation comparison.

Three variants are run at Layer 1 (`fit_joint_k7.py --variant {A|B|C}`),
compared on the IIIC subset against the K=6 v13 joint posterior, and the
winner is promoted to the full K=7 fit. The motivation: the K=6 joint fit
documented in `calibration/cert_config.yaml:278-283` produced a degenerate
sz expert/crowd separation (Cohen-d=-0.38, J=0.089) because Kong-crowd
labels are 50.7 % of IIIC observations and the single per-rater random
effect σ_ℓ pools experts toward crowd. K=7 adds spike (sn1_combined_v2 is
46 % of the full corpus), making the source-mix more complex.

The variants:

  V_A — Per-tier σ_ℓ + σ_t. Allow the random-effect variance to differ by
        expertise tier so crowd N stops constraining expert variance. This
        is the standard mixed-effects rater-model treatment
        (Bafumi-Gelman 2007 JEBS; Patz-Junker 1999). +5 LOC vs the K=6 model.

  V_B — Two-stage decoupling at K=7. Joint fit serves s_j + s_sd ONLY;
        per-rater (σ̂, θ̂) come from the byte-verbatim two-stage
        `pipeline/reference_calibration/fit_sdt_per_domain.py` extended to
        all 7 tasks. Preserves Phase-3.5 v13 decision verbatim. D2 invariant
        (md5 byte-pin) preserved. Implemented as the same _model_B(...) as
        the K=6 _model but with K=7 task awareness — per-rater outputs
        IGNORED downstream; only s_j / s_sd consumed.

  V_C — Source-weighted likelihood. Down-weight per-label by
        w_d = 1 / sqrt(n_per_source[d]) so each source contributes ~equal
        effective N. Defensible (publication-bias correction precedent)
        but non-standard for IRT. Uses jax obs_mask weighting.

All three variants preserve:
  * LAMBDA = 0.025 (the lapse rate; reference invariant; not estimated)
  * The single likelihood definition `λ + (1−2λ)·Φ(η)` matching engine/core.py
  * Source offset δ_d (sum-to-zero) and per-source discrim scale γ_d (>0)
  * Split-anchored gauge identification (see `_gauge_fix` in fit_joint_k7.py)

Layer-1 selection criteria gate against:
  * IIIC subset s_j max-drift ≤ 0.05 per segment vs K=6 v13 joint
  * Phase-3.5 sz J ≥ 0.40 (recover from K=6 J=0.089 degeneracy)
  * NUTS R̂ < 1.05 across 7 tasks
  * Expert-vs-crowd Cohen-d > 0.5 on per-tier mean a_skill
"""
from __future__ import annotations

import math

LAMBDA = 0.025  # reference invariant — FIXED


def _shared_priors(numpyro, dist, jnp, n_seg, n_rater, n_src, n_tier):
    """Priors common to all three variants. Returns the building blocks
    (s, a_skill, a_bias, sources δ, γ) that each variant assembles into
    its own ell/t structure.

    Note (2026-05-29): Gate B (sum-to-zero contrast reparameterization of
    a_skill/a_bias) was attempted to break the tier-label-swap NUTS multi-
    modality but REGRESSED (spike R̂ 35→124 on the smoke smoke). Reverted
    here. The K=7 NUTS multi-modal posterior is a documented Phase-3.5
    corpus-structure property (`cert_config.yaml:278-283`); the V_B
    architecture's MODULAR CUT (joint fit produces only s_j; per-rater
    parameters come from byte-verbatim two-stage) is the principled response
    per Plummer 2015 *Statistics and Computing*; Jacob, Murray, Holmes,
    Robert 2017 *JRSSB*; Patz & Junker 1999 *JEBS*. SVI is the v1.0
    production posterior; NUTS validates iic only at R̂=1.02; K=6 v13 κ
    values transfer because V_B reproduces K=6 v13 essentially exactly
    (variant comparison drift_mean = 0.001 per IIIC task).
    """
    tau_s = numpyro.sample("tau_s", dist.HalfNormal(1.0))
    s_raw = numpyro.sample("s", dist.Normal(0.0, 1.0).expand([n_seg]))
    s = numpyro.deterministic("s_j", tau_s * s_raw)

    a_skill = numpyro.sample("a_skill", dist.Normal(0.0, 1.0).expand([n_tier]))
    a_bias = numpyro.sample("a_bias", dist.Normal(0.0, 1.0).expand([n_tier]))

    # Source offset δ_d sum-to-zero (non-centered); per-source γ_d log-normal.
    # Source 0 pinned (δ_0=0, γ_0=1) for identifiability.
    tau_d = numpyro.sample("tau_d", dist.HalfNormal(1.0))
    d_free = numpyro.sample("delta_free",
                            dist.Normal(0, 1).expand([n_src - 1]))
    delta = numpyro.deterministic(
        "delta", jnp.concatenate([jnp.zeros(1), tau_d * d_free]))
    g_free = numpyro.sample("loggamma_free",
                            dist.Normal(0.0, 0.5).expand([n_src - 1]))
    log_gamma = numpyro.deterministic(
        "log_gamma", jnp.concatenate([jnp.zeros(1), g_free]))

    return s, a_skill, a_bias, delta, log_gamma


def _likelihood(numpyro, dist, jnp, jnorm, lin, y):
    """Shared likelihood (Variant A and B). p = λ + (1 − 2λ)·Φ(η)."""
    p = LAMBDA + (1.0 - 2.0 * LAMBDA) * jnorm.cdf(lin)
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    numpyro.sample("y", dist.Bernoulli(probs=p), obs=y)


# ──────────────────────────────────────────────────────────────────────────
# Variant A — per-tier σ_ℓ + σ_t

def model_A(seg_idx, rater_idx, src_idx, rater_tier_idx,
            n_seg, n_rater, n_src, n_tier, y=None):
    """V_A — per-tier σ_ℓ + σ_t. Each rater's (ell_i, t_i) is centered at
    its tier mean (a_skill[g], a_bias[g]) AND scaled by its tier-specific
    standard deviation (σ_ℓ[g], σ_t[g]) — so expert tier can have a
    TIGHTER (or LOOSER) random-effect variance than crowd tier, regardless
    of n_per_tier. This is the K=7 fix for the Phase-3.5 Kong-crowd-
    dominance degeneracy (sz J=0.089) documented in
    `calibration/cert_config.yaml:278-283`.

    rater_tier_idx: (n_rater,) int-array mapping each unique rater to its
    expertise tier; built by fit_joint_k7.py from the rater-level tier
    map (vs per-observation tier_idx in prep_k7.load_task).

    Delta vs K=6 fit_joint_iiic._model: 4 lines (sig_l / sig_t expanded
    over n_tier; indexed by rater_tier_idx in the per-rater
    deterministic).
    """
    import jax.numpy as _jnp
    from jax.scipy.stats import norm as _jnorm
    import numpyro
    import numpyro.distributions as dist

    s, a_skill, a_bias, delta, log_gamma = _shared_priors(
        numpyro, dist, _jnp, n_seg, n_rater, n_src, n_tier)

    sig_l_tier = numpyro.sample(
        "sig_l_tier", dist.HalfNormal(1.0).expand([n_tier]))
    sig_t_tier = numpyro.sample(
        "sig_t_tier", dist.HalfNormal(1.0).expand([n_tier]))
    zl = numpyro.sample("zl", dist.Normal(0, 1).expand([n_rater]))
    zt = numpyro.sample("zt", dist.Normal(0, 1).expand([n_rater]))

    # Per-rater ell, t indexed by THEIR tier (rater_tier_idx is (n_rater,))
    ell = numpyro.deterministic(
        "ell",
        a_skill[rater_tier_idx] + sig_l_tier[rater_tier_idx] * zl,
    )
    t = numpyro.deterministic(
        "t",
        a_bias[rater_tier_idx] + sig_t_tier[rater_tier_idx] * zt,
    )

    lin = (_jnp.exp(ell[rater_idx] + log_gamma[src_idx])
           * (s[seg_idx] + delta[src_idx] - t[rater_idx]))
    _likelihood(numpyro, dist, _jnp, _jnorm, lin, y)


# ──────────────────────────────────────────────────────────────────────────
# Variant B — two-stage decoupling at K=7 (joint for s_j only)

def model_B(seg_idx, rater_idx, src_idx, rater_tier_idx,
            n_seg, n_rater, n_src, n_tier, y=None):
    """V_B — same structure as K=6 _model (single σ_ℓ, σ_t pooled across
    raters), extended to K=7 per task. Downstream consumes ONLY s_j + s_sd
    from this fit; per-rater (σ̂, θ̂) come from the byte-verbatim
    `fit_sdt_per_domain.py` extended to all 7 tasks (preserving D2 md5
    pin). The model body is identical to the K=6 _model with rater_tier_idx
    passed in for consistency with V_A/V_C signature."""
    import jax.numpy as _jnp
    from jax.scipy.stats import norm as _jnorm
    import numpyro
    import numpyro.distributions as dist

    s, a_skill, a_bias, delta, log_gamma = _shared_priors(
        numpyro, dist, _jnp, n_seg, n_rater, n_src, n_tier)

    # Single pooled sigma_l + sigma_t (V_B identity vs K=6)
    sig_l = numpyro.sample("sig_l", dist.HalfNormal(1.0))
    sig_t = numpyro.sample("sig_t", dist.HalfNormal(1.0))
    zl = numpyro.sample("zl", dist.Normal(0, 1).expand([n_rater]))
    zt = numpyro.sample("zt", dist.Normal(0, 1).expand([n_rater]))

    ell = numpyro.deterministic("ell", a_skill[rater_tier_idx] + sig_l * zl)
    t = numpyro.deterministic("t", a_bias[rater_tier_idx] + sig_t * zt)

    lin = (_jnp.exp(ell[rater_idx] + log_gamma[src_idx])
           * (s[seg_idx] + delta[src_idx] - t[rater_idx]))
    _likelihood(numpyro, dist, _jnp, _jnorm, lin, y)


# ──────────────────────────────────────────────────────────────────────────
# Variant C — source-weighted likelihood

def model_C(seg_idx, rater_idx, src_idx, rater_tier_idx,
            source_weight_per_obs,
            n_seg, n_rater, n_src, n_tier, y=None):
    """V_C — same prior structure as V_B, but the likelihood is weighted
    per-observation by source_weight_per_obs (typically 1 / sqrt(n_per_source[d]))
    so dominant sources (Kong crowd; sn1 spike) don't overwhelm the posterior.

    Implementation note: numpyro doesn't have a built-in per-observation
    likelihood weight. We implement via factor() — adding the weighted
    log-likelihood directly. This is mathematically equivalent to running
    Bernoulli on a re-weighted dataset.
    """
    import jax.numpy as _jnp
    from jax.scipy.stats import norm as _jnorm
    import numpyro
    import numpyro.distributions as dist

    s, a_skill, a_bias, delta, log_gamma = _shared_priors(
        numpyro, dist, _jnp, n_seg, n_rater, n_src, n_tier)

    sig_l = numpyro.sample("sig_l", dist.HalfNormal(1.0))
    sig_t = numpyro.sample("sig_t", dist.HalfNormal(1.0))
    zl = numpyro.sample("zl", dist.Normal(0, 1).expand([n_rater]))
    zt = numpyro.sample("zt", dist.Normal(0, 1).expand([n_rater]))

    ell = numpyro.deterministic("ell", a_skill[rater_tier_idx] + sig_l * zl)
    t = numpyro.deterministic("t", a_bias[rater_tier_idx] + sig_t * zt)

    lin = (_jnp.exp(ell[rater_idx] + log_gamma[src_idx])
           * (s[seg_idx] + delta[src_idx] - t[rater_idx]))
    p = LAMBDA + (1.0 - 2.0 * LAMBDA) * _jnorm.cdf(lin)
    p = _jnp.clip(p, 1e-6, 1 - 1e-6)

    if y is not None:
        # Weighted log-likelihood via factor()
        ll = (y * _jnp.log(p) + (1 - y) * _jnp.log1p(-p))
        # Multiply by per-observation source weight
        weighted_ll = (ll * source_weight_per_obs).sum()
        numpyro.factor("y_weighted", weighted_ll)


# ──────────────────────────────────────────────────────────────────────────
# Dispatcher

VARIANTS = {"A": model_A, "B": model_B, "C": model_C}


def get_model(variant: str):
    """Return the numpyro model function for the named variant."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r} not in {list(VARIANTS)}")
    return VARIANTS[variant]


def compute_source_weights(src_idx, n_src):
    """V_C helper: compute per-observation weight w_d = 1/sqrt(n_per_source[d]).
    Returns an (n_obs,) numpy array; normalized so mean weight = 1.0 (so the
    total log-likelihood scale matches an unweighted fit's; only relative
    contributions change).
    """
    import numpy as np
    counts = np.bincount(src_idx, minlength=n_src).astype(float)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w_per_obs = inv_sqrt[src_idx]
    w_per_obs = w_per_obs / w_per_obs.mean()  # normalize mean to 1.0
    return w_per_obs
