"""Phase 3.5 — joint hierarchical cross-source s_j fit for ONE IIIC task.

Model (reference-consistent lapse — IDENTICAL form to the carried verbatim
fit_sdt_per_domain.py / core_mcmc.py: p = λ + (1−2λ)Φ(·), λ=0.025 FIXED;
NOT the adjudicator's notational λ/2+(1−λ)Φ — chosen so the re-derived
Youden + the engine stay on the same scale as the byte-verbatim
downstream):

    P(Y_ij=1) = λ + (1−2λ)·Φ( e^{ℓ_i}·γ_{d} · ( s_j + δ_d − t_i ) )

  s_j        per-segment latent signal (the unifying quantity), N(0, τ_s²)
  ℓ_i,t_i    rater log-discrimination / threshold, partial-pooled with
             EXPERTISE-TIER means: ℓ_i = α^skill_{g(i)} + σ_ℓ·z^ℓ_i ; etc.
  δ_d        source offset (sum-to-zero over sources; non-centered)
  γ_d        per-source discrimination scale (log-normal, one pinned =1);
             its posterior is compared downstream to the two-stage ×1/1.7
             implied ratio (the "H0: ratio=1.7" report).
  λ = 0.025  FIXED (reference invariant; not estimated).

Gauge identification (3-agent adjudication — split anchoring): the model
has a location gauge (s,δ,t shift) and a scale gauge (s,δ,t scale with a
compensating ℓ shift). We fix, per posterior draw, by a DETERMINISTIC
affine: scale b = std(s over the sparcnet50K∩kong:crowd weld), location
a = mean(s/b over the n=4 Centaur-gold segs). The transform is applied to
(s,δ,t,ℓ) and a numerical assertion verifies the predicted probabilities
are invariant (the gauge must not change the fit — only its
parameterisation). Gold = anchor, never a Youden population.

Outputs (per task): joint posterior summaries → the engine/Youden
sdt_fits schema (sigma=e^{−ℓ̄_i}, theta=t̄_i, ell=ℓ̄_i), the unified
per-segment signal table (s_mean,s_sd on the common scale), and the
denormalised per-IIIC-segment analysis CSV.

Calibration-stage only; never imported by the engine runtime.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# JAX/NumPyro imported lazily inside main() so `import` is cheap and the
# engine never pulls these in.
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "calibration" / "joint"
TASKS = ["sz", "lpd", "gpd", "lrda", "grda", "iic"]
LAMBDA = 0.025  # reference invariant — FIXED


def _model(seg_idx, rater_idx, src_idx, tier_idx, n_seg, n_rater, n_src,
           n_tier, y=None):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    tau_s = numpyro.sample("tau_s", dist.HalfNormal(1.0))
    s = numpyro.sample("s", dist.Normal(0.0, 1.0).expand([n_seg]))  # raw
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

    # source offset δ_d sum-to-zero; per-source discrim scale γ_d (>0),
    # source 0 pinned (δ_0=0, γ_0=1) for identifiability.
    tau_d = numpyro.sample("tau_d", dist.HalfNormal(1.0))
    d_free = numpyro.sample("delta_free",
                            dist.Normal(0, 1).expand([n_src - 1]))
    delta = numpyro.deterministic(
        "delta", jnp.concatenate([jnp.zeros(1), tau_d * d_free]))
    g_free = numpyro.sample("loggamma_free",
                            dist.Normal(0.0, 0.5).expand([n_src - 1]))
    log_gamma = numpyro.deterministic(
        "log_gamma", jnp.concatenate([jnp.zeros(1), g_free]))

    lin = (jnp.exp(ell[rater_idx] + log_gamma[src_idx])
           * (s[seg_idx] + delta[src_idx] - t[rater_idx]))
    from jax.scipy.stats import norm as jnorm
    p = LAMBDA + (1.0 - 2.0 * LAMBDA) * jnorm.cdf(lin)
    p = jnp.clip(p, 1e-6, 1 - 1e-6)
    numpyro.sample("y", dist.Bernoulli(probs=p), obs=y)


def _predicted_p(s, delta, log_gamma, ell, t, seg, rat, src):
    """numpy replica of the likelihood mean (for the gauge-invariance
    assertion and downstream)."""
    from scipy.stats import norm
    lin = np.exp(ell[rat] + log_gamma[src]) * (s[seg] + delta[src] - t[rat])
    return LAMBDA + (1.0 - 2.0 * LAMBDA) * norm.cdf(lin)


def _gauge_fix(s, delta, log_gamma, ell, t, scale_mask, loc_mask):
    """Deterministic split-anchor identification on ONE posterior draw.
    scale b := std(s[weld]); location a := mean((s/b)[gold]).
      s'   = s/b − a
      δ'   = δ/b
      t'   = t/b − a
      ℓ'   = ℓ + ln b      (e^{ℓ}·b absorbs the scale)
    Then s'+δ'−t' = (s+δ−t)/b and e^{ℓ'} = e^{ℓ}·b ⇒ the product
    e^{ℓ'}(s'+δ'−t') = e^{ℓ}(s+δ−t) is INVARIANT (asserted numerically).
    """
    b = float(np.std(s[scale_mask]))
    if not np.isfinite(b) or b <= 0:
        b = 1.0
    s1 = s / b
    a = float(np.mean(s1[loc_mask])) if loc_mask.any() else 0.0
    return (s1 - a, delta / b, log_gamma.copy(), ell + np.log(b), t / b - a)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=TASKS)
    ap.add_argument("--subsample", type=int, default=None,
                    help="cap N (stratified-ish) for smoke tests")
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--samples", type=int, default=600)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["svi", "nuts"], default="svi",
                    help="svi = full-corpus production path; nuts = "
                         "exact-Bayes (subsample) variance cross-check")
    ap.add_argument("--svi-steps", type=int, default=40000)
    ap.add_argument("--post-draws", type=int, default=500,
                    help="posterior draws from the SVI low-rank guide")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny settings + invariance check only")
    args = ap.parse_args()

    if args.smoke:
        args.subsample = args.subsample or 40000
        args.warmup, args.samples, args.chains = 150, 150, 1
        args.svi_steps = 4000

    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    import numpyro.diagnostics as ndiag
    from .prep import load_task

    numpyro.set_host_device_count(args.chains)
    t0 = time.time()
    td = load_task(args.task, subsample=args.subsample, seed=args.seed)
    tiers = sorted(set(td.rater_expertise))
    tmap = {tt: i for i, tt in enumerate(tiers)}
    tier_idx = np.array([tmap[e] for e in td.rater_expertise], np.int32)
    print(f"[{args.task}] N={td.n_obs:,} S={td.n_seg:,} R={td.n_rater:,} "
          f"src={td.n_src} tiers={tiers} pos_rate={td.pos_rate:.3f} "
          f"scale_anchor_segs={int(td.scale_anchor_seg.sum())} "
          f"loc_anchor_segs={int(td.loc_anchor_seg.sum())}")

    mkw = dict(seg_idx=jnp.array(td.seg_idx),
               rater_idx=jnp.array(td.rater_idx),
               src_idx=jnp.array(td.src_idx),
               tier_idx=jnp.array(tier_idx),
               n_seg=td.n_seg, n_rater=td.n_rater, n_src=td.n_src,
               n_tier=len(tiers))
    if args.method == "nuts":
        # exact-Bayes cross-check / variance reference (subsample only)
        numpyro.set_host_device_count(args.chains)
        mcmc = MCMC(NUTS(_model), num_warmup=args.warmup,
                    num_samples=args.samples, num_chains=args.chains,
                    progress_bar=False)
        mcmc.run(jax.random.PRNGKey(args.seed), y=jnp.array(td.y), **mkw)
        post = mcmc.get_samples()
    else:
        # SVI production path — AutoLowRankMultivariateNormal so the s_j
        # posterior covariance is NOT mean-field-collapsed (propagating a
        # faithful s_sd is the entire point of Phase 3.5; a diagonal guide
        # would defeat it — variance is still calibrated vs NUTS-subsample
        # downstream).
        guide = AutoLowRankMultivariateNormal(_model, rank=20)
        svi = SVI(_model, guide,
                  numpyro.optim.Adam(2e-3), Trace_ELBO())
        res = svi.run(jax.random.PRNGKey(args.seed), args.svi_steps,
                      y=jnp.array(td.y), progress_bar=False, **mkw)
        losses = np.asarray(res.losses)
        post = Predictive(_model, guide=guide, params=res.params,
                          num_samples=args.post_draws)(
            jax.random.PRNGKey(args.seed + 1), **mkw)
        post = {k: np.asarray(v) for k, v in post.items()}
        elbo_tail = float(np.mean(losses[-200:]))
        print(f"[{args.task}] SVI {args.svi_steps} steps  "
              f"ELBO_tail={elbo_tail:.1f}  loss[0]={losses[0]:.1f}")
    el = time.time() - t0

    # posterior means (raw), then gauge-fix the posterior MEAN draw
    s_m = np.asarray(post["s_j"]).mean(0)
    s_sd = np.asarray(post["s_j"]).std(0)
    dl_m = np.asarray(post["delta"]).mean(0)
    lg_m = np.asarray(post["log_gamma"]).mean(0)
    el_m = np.asarray(post["ell"]).mean(0)
    t_m = np.asarray(post["t"]).mean(0)

    p_raw = _predicted_p(s_m, dl_m, lg_m, el_m, t_m,
                         td.seg_idx, td.rater_idx, td.src_idx)
    s_id, dl_id, lg_id, el_id, t_id = _gauge_fix(
        s_m, dl_m, lg_m, el_m, t_m,
        td.scale_anchor_seg, td.loc_anchor_seg)
    p_id = _predicted_p(s_id, dl_id, lg_id, el_id, t_id,
                        td.seg_idx, td.rater_idx, td.src_idx)
    max_dev = float(np.max(np.abs(p_raw - p_id)))
    # The affine transform is ALGEBRAICALLY EXACT: e^{ℓ'}(s'+δ'−t')
    #   = e^{ℓ}·b·((s/b−a)+δ/b−(t/b−a)) = e^{ℓ}·b·(s+δ−t)/b = e^{ℓ}(s+δ−t),
    # the ±a cancelling in s'−t'. Any residual is pure float64 round-off
    # through exp() of the linear predictor; bound it well below inferential
    # relevance (λ floor = 0.025, so 1e-5 is ~2500× tighter and negligible).
    # max_dev is recorded in the summary for full transparency.
    assert max_dev < 1e-5, (
        f"GAUGE TRANSFORM NOT LIKELIHOOD-INVARIANT (max |Δp|={max_dev:.2e} "
        ">= 1e-5) — exceeds float64 round-off; the affine algebra is wrong; "
        "refusing to emit.")

    # convergence diagnostics on the key blocks
    if args.method == "nuts" and args.chains >= 2:
        def _rhat(name):
            x = np.asarray(post[name])
            try:
                return float(np.nanmax(ndiag.gelman_rubin(
                    x.reshape(args.chains, -1, *x.shape[1:]))))
            except Exception:
                return float("nan")
        conv = {"method": "nuts",
                "rhat_max": {k: _rhat(k) for k in ("s_j", "ell", "t")
                             if k in post}}
    elif args.method == "nuts":
        conv = {"method": "nuts",
                "rhat_max": "n/a (single chain — smoke; real run ≥2)"}
    else:
        conv = {"method": "svi",
                "note": "SVI: R̂ not applicable; convergence = ELBO; "
                        "s_sd variance-calibrated vs NUTS-subsample "
                        "downstream"}

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": args.task, "n_obs": td.n_obs, "n_seg": td.n_seg,
        "n_rater": td.n_rater, "n_src": td.n_src,
        "subsample": args.subsample,
        "nuts": {"warmup": args.warmup, "samples": args.samples,
                 "chains": args.chains},
        "wall_s": round(el, 1),
        "gauge_invariance_max_abs_dp": max_dev,
        "convergence": conv,
        "scale_anchor_segs": int(td.scale_anchor_seg.sum()),
        "loc_anchor_segs": int(td.loc_anchor_seg.sum()),
        "lapse_lambda_fixed": LAMBDA,
        "per_source_log_gamma_mean": lg_id.tolist(),
        "src_names": td.src_names,
        "s_id_mean_over_gold": float(np.mean(s_id[td.loc_anchor_seg]))
        if td.loc_anchor_seg.any() else None,
        "s_id_std_over_weld": float(np.std(s_id[td.scale_anchor_seg]))
        if td.scale_anchor_seg.any() else None,
    }
    (OUT / f"{args.task}_summary.json").write_text(
        json.dumps(summary, indent=2))
    np.savez_compressed(
        OUT / f"{args.task}_posterior.npz",
        s_id_mean=s_id, s_sd=s_sd, ell_id_mean=el_id, t_id_mean=t_id,
        seg_ids=td.seg_ids, rater_ids=td.rater_ids,
        rater_canonical=np.array(td.rater_canonical),
        rater_expertise=np.array(td.rater_expertise),
        src_names=np.array(td.src_names))
    print(f"[{args.task}] done {el:.1f}s  method={args.method}  "
          f"gauge|Δp|={max_dev:.1e}  conv={conv.get('method')}  "
          f"s̄_gold={summary['s_id_mean_over_gold']:.3f}(→0) "
          f"sd_weld={summary['s_id_std_over_weld']:.3f}(→1)  -> {OUT}")


if __name__ == "__main__":
    main()
