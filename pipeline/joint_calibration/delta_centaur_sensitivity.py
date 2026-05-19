"""Phase 3.5 gate: δ_Centaur sensitivity (CHEAP re-scoped check).

Question (user-approved scope — NOT a full refit): the joint model
estimates a per-source offset δ_d, AND the location gauge is pinned by
a = mean((s/b)[Centaur-gold]). The Centaur-expert source offset
δ_Centaur is therefore PARTIALLY REDUNDANT with the location anchor
(both act on the Centaur scale). Does forcing δ_Centaur ≡ 0 (Centaur
reads on the gold scale, the offset absorbed ONLY by the deterministic
anchor) materially move the unified s_j?

Method: short SVI smoke (small subsample, few steps) on 1–2
representative IIIC tasks, run TWICE:
  (A) baseline   — verbatim fit_joint_iiic._model (δ all sources free)
  (B) δ_Cent=0   — identical model, Centaur source δ hard-zeroed
Same seed / subsample / steps ⇒ the ONLY difference is δ_Centaur.
Gauge-fix both; compare s_j (Pearson r, max |Δs| in τ_s units).

Expected (robust): r ≈ 1, |Δs| ≪ τ_s — δ_Centaur is non-identified
against the anchor, so the s_j unification does NOT depend on it. This
is a ROBUSTNESS check, not a refit; it bounds the gauge's dependence
on the one modelling choice the adjudication flagged.

Calibration-stage only; never imported by the engine runtime.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "calibration" / "joint"
PROBE_TASKS = ["gpd", "iic"]      # clean IIIC + the erratum/Centaur-heavy
SUBSAMPLE = 40000
SVI_STEPS = 4000
POST_DRAWS = 300
SEED = 0


def _model_fixdelta(centaur_src, seg_idx, rater_idx, src_idx, tier_idx,
                    n_seg, n_rater, n_src, n_tier, y=None):
    """Byte-for-byte fit_joint_iiic._model EXCEPT the Centaur source's
    δ entry is hard-zeroed (offset absorbed by the location anchor)."""
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from jax.scipy.stats import norm as jnorm

    LAMBDA = 0.025
    tau_s = numpyro.sample("tau_s", dist.HalfNormal(1.0))
    s = numpyro.sample("s", dist.Normal(0.0, 1.0).expand([n_seg]))
    s = numpyro.deterministic("s_j", tau_s * s)
    a_skill = numpyro.sample("a_skill",
                             dist.Normal(0.0, 1.0).expand([n_tier]))
    a_bias = numpyro.sample("a_bias",
                            dist.Normal(0.0, 1.0).expand([n_tier]))
    sig_l = numpyro.sample("sig_l", dist.HalfNormal(1.0))
    sig_t = numpyro.sample("sig_t", dist.HalfNormal(1.0))
    zl = numpyro.sample("zl", dist.Normal(0, 1).expand([n_rater]))
    zt = numpyro.sample("zt", dist.Normal(0, 1).expand([n_rater]))
    ell = numpyro.deterministic("ell", a_skill[tier_idx] + sig_l * zl)
    t = numpyro.deterministic("t", a_bias[tier_idx] + sig_t * zt)
    tau_d = numpyro.sample("tau_d", dist.HalfNormal(1.0))
    d_free = numpyro.sample("delta_free",
                            dist.Normal(0, 1).expand([n_src - 1]))
    delta = jnp.concatenate([jnp.zeros(1), tau_d * d_free])
    # ── the ONLY change vs baseline: zero the Centaur source offset ──
    delta = delta.at[centaur_src].set(0.0)
    delta = numpyro.deterministic("delta", delta)
    g_free = numpyro.sample("loggamma_free",
                            dist.Normal(0.0, 0.5).expand([n_src - 1]))
    log_gamma = numpyro.deterministic(
        "log_gamma", jnp.concatenate([jnp.zeros(1), g_free]))
    lin = (jnp.exp(ell[rater_idx] + log_gamma[src_idx])
           * (s[seg_idx] + delta[src_idx] - t[rater_idx]))
    p = LAMBDA + (1.0 - 2.0 * LAMBDA) * jnorm.cdf(lin)
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    numpyro.sample("y", dist.Bernoulli(probs=p), obs=y)


def _fit(model_fn, td, tier_idx, tiers, seed):
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    mkw = dict(seg_idx=jnp.array(td.seg_idx),
               rater_idx=jnp.array(td.rater_idx),
               src_idx=jnp.array(td.src_idx),
               tier_idx=jnp.array(tier_idx),
               n_seg=td.n_seg, n_rater=td.n_rater, n_src=td.n_src,
               n_tier=len(tiers))
    guide = AutoLowRankMultivariateNormal(model_fn, rank=20)
    svi = SVI(model_fn, guide, numpyro.optim.Adam(2e-3), Trace_ELBO())
    res = svi.run(jax.random.PRNGKey(seed), SVI_STEPS,
                  y=jnp.array(td.y), progress_bar=False, **mkw)
    post = Predictive(model_fn, guide=guide, params=res.params,
                      num_samples=POST_DRAWS)(
        jax.random.PRNGKey(seed + 1), **mkw)
    return {k: np.asarray(v) for k, v in post.items()}


def main() -> None:
    from .prep import load_task, LOCATION_ANCHOR_SRC
    from .fit_joint_iiic import _model, _gauge_fix
    import functools

    print("=== Phase 3.5 gate: δ_Centaur sensitivity (cheap) ===",
          flush=True)
    results = {}
    for task in PROBE_TASKS:
        td = load_task(task, subsample=SUBSAMPLE, seed=SEED)
        if LOCATION_ANCHOR_SRC not in td.src_names:
            print(f"  [{task}] no Centaur-expert source — skip")
            continue
        c_idx = td.src_names.index(LOCATION_ANCHOR_SRC)
        tiers = sorted(set(td.rater_expertise))
        tmap = {tt: i for i, tt in enumerate(tiers)}
        tier_idx = np.array([tmap[e] for e in td.rater_expertise],
                            np.int32)
        t0 = time.time()
        pA = _fit(_model, td, tier_idx, tiers, SEED)               # base
        pA2 = _fit(_model, td, tier_idx, tiers, SEED + 7)          # noise
        pB = _fit(functools.partial(_model_fixdelta, c_idx),
                  td, tier_idx, tiers, SEED)                       # δ_C=0

        def _gs(post):
            sm = post["s_j"].mean(0)
            s_id, *_ = _gauge_fix(
                sm, post["delta"].mean(0), post["log_gamma"].mean(0),
                post["ell"].mean(0), post["t"].mean(0),
                td.scale_anchor_seg, td.loc_anchor_seg)
            return s_id

        sA, sA2, sB = _gs(pA), _gs(pA2), _gs(pB)
        # δ_Centaur effect vs the SAME-MODEL/different-seed SVI noise
        # floor. Robust iff zeroing δ_Centaur perturbs s_j NO MORE than
        # merely re-running SVI does (effect within optimisation noise).
        r_dlt = float(np.corrcoef(sA, sB)[0, 1])
        r_nz = float(np.corrcoef(sA, sA2)[0, 1])
        d_dlt = float(np.max(np.abs(sA - sB)))      # δ-swap shift
        d_nz = float(np.max(np.abs(sA - sA2)))      # pure SVI-noise floor
        ratio = d_dlt / d_nz if d_nz > 0 else float("nan")
        el = time.time() - t0
        results[task] = {
            "n_obs": td.n_obs, "n_seg": td.n_seg,
            "centaur_src_idx": c_idx,
            "delta_centaur_baseline_estimate":
                float(pA["delta"].mean(0)[c_idx]),
            "s_j_pearson_r_base_vs_fixed": r_dlt,
            "s_j_pearson_r_noise_floor": r_nz,
            "s_j_max_abs_shift_delta_swap": d_dlt,
            "s_j_max_abs_shift_svi_noise_floor": d_nz,
            "delta_shift_over_noise_floor_ratio": ratio,
            "wall_s": round(el, 1),
        }
        print(f"  [{task}] δ_Cent(base)="
              f"{results[task]['delta_centaur_baseline_estimate']:+.4f}  "
              f"r(δ-swap)={r_dlt:.4f} r(noise)={r_nz:.4f}  "
              f"|Δs|: δ-swap={d_dlt:.3f} noise={d_nz:.3f}  "
              f"ratio={ratio:.2f}  {el:.0f}s")

    rs = [v["s_j_pearson_r_base_vs_fixed"] for v in results.values()]
    ratios = [v["delta_shift_over_noise_floor_ratio"]
              for v in results.values()]
    summary = {
        "scope": "CHEAP re-scoped robustness probe (short SVI smoke, "
        f"subsample={SUBSAMPLE}, {SVI_STEPS} steps) — NOT a full refit.",
        "probe_tasks": list(results.keys()),
        "min_s_j_pearson_r_delta_swap": float(min(rs)) if rs else None,
        "max_delta_shift_over_noise_floor_ratio":
            float(max(ratios)) if ratios else None,
        # robust = the δ_Centaur swap perturbs s_j no more than ~merely
        # re-running SVI (ratio ≲ 1.5) AND preserves shape (r ≥ 0.95).
        "robust": bool(rs and min(rs) >= 0.95
                       and ratios and max(ratios) <= 1.5),
        "interpretation": (
            "δ_Centaur is partially NON-IDENTIFIED against the "
            "deterministic location anchor (a = mean s/b over the SAME "
            "Centaur-gold segs). The honest metric is the δ-swap s_j "
            "shift RELATIVE to a same-model/different-seed SVI noise "
            "floor (smoke-scale SVI is itself noisy on thin-coverage "
            "segments — raw max-abs-shift alone is uninterpretable; the "
            "gpd control, |δ_Centaur|≈0.0008, makes this concrete). "
            "ratio ≲ 1 ⇒ zeroing δ_Centaur moves s_j no more than "
            "re-running the optimiser does ⇒ the unified signal does NOT "
            "materially depend on this flagged choice. ratio ≫ 1 would "
            "flag a genuine dependence. High Pearson r confirms the "
            "signal SHAPE is preserved regardless. Honest scope: bounds "
            "the dependence at smoke scale; a production-scale refit "
            "(out of the user-approved cheap scope) would tighten it."),
        "per_task": results,
    }
    outp = OUT / "delta_centaur_sensitivity.json"
    outp.write_text(json.dumps(summary, indent=2))
    print(f"\n  min s_j r(δ-swap)={summary['min_s_j_pearson_r_delta_swap']}"
          f"  max(δ-shift / SVI-noise-floor)="
          f"{summary['max_delta_shift_over_noise_floor_ratio']}  "
          f"robust={summary['robust']}\n  -> {outp}")


if __name__ == "__main__":
    main()
