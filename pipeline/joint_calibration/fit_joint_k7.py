"""Phase 9 K=7 joint hierarchical fit — driver for variant A/B/C × task ∈ K=7.

Extends `fit_joint_iiic.py` to K=7 (spike + 6 IIIC) and runs the joint fit
under one of three variants for Kong-crowd-mitigation comparison (V_A, V_B,
V_C; see `model_k7.py`).

Outputs per (task, variant) pair:
  calibration/joint/{task}_k7_{variant}_summary.json
  calibration/joint/{task}_k7_{variant}_posterior.npz

Usage:
    .venv/bin/python -m pipeline.joint_calibration.fit_joint_k7 \\
        --task iic --variant A --method svi --svi-steps 40000

    # smoke run (1 task × 1 variant, ~40K subsample, ~4000 SVI steps):
    .venv/bin/python -m pipeline.joint_calibration.fit_joint_k7 \\
        --task iic --variant A --smoke

    # NUTS exact-Bayes validation (smaller subsample, gold-standard chain):
    .venv/bin/python -m pipeline.joint_calibration.fit_joint_k7 \\
        --task sz --variant A --method nuts --subsample 80000 \\
        --warmup 1500 --samples 1500 --chains 4

Parallelization: `--gpu N` pins this process to GPU N via
CUDA_VISIBLE_DEVICES; `XLA_PYTHON_CLIENT_MEM_FRACTION` is set to 0.45 by
default so 2 processes can co-exist on the same GPU at need (one per
device fraction). The Layer-1 orchestrator (`run_variant_comparison.py`)
spawns multiple of these as subprocesses with different `--gpu` + pairs.

Variant-B caveat: this script fits the joint posterior. For V_B, downstream
consumes ONLY s_j + s_sd from the resulting posterior; the per-rater
(σ̂, θ̂) are produced separately by the byte-verbatim
`pipeline/reference_calibration/fit_sdt_per_domain.py` extended to K=7 (a
sibling driver to be added at Layer 3 assembly).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "calibration" / "joint"

LAMBDA = 0.025  # mirror; the actual constant lives in model_k7.LAMBDA


# ──────────────────────────────────────────────────────────────────────────
# GPU / BLAS environment — must be set BEFORE numpy / jax / numpyro imports

def _configure_compute_env(gpu: int | None, mem_fraction: float = 0.45) -> None:
    """Pin process to a single GPU and bound its memory footprint.

    Eli's machine: 2 × RTX A4500 (20 GB each). With mem_fraction=0.45 two
    processes can share a single GPU. The Layer-1 orchestrator runs one
    process per GPU (2 concurrent (task, variant) pairs); mem_fraction=0.45
    leaves headroom for JAX's compile-time allocator.

    BLAS single-thread mirrors the repo's bit-exact reproducibility
    contract (`workflow_conventions.md`).
    """
    for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
                "OMP_NUM_THREADS"):
        os.environ[var] = "1"
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(mem_fraction))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


# ──────────────────────────────────────────────────────────────────────────
# Gauge invariance check (mirrors fit_joint_iiic._gauge_fix + _predicted_p)

def _predicted_p(s, delta, log_gamma, ell, t, seg, rat, src):
    """numpy replica of the likelihood mean (for the gauge-invariance
    assertion). Identical to fit_joint_iiic._predicted_p."""
    from scipy.stats import norm
    lin = np.exp(ell[rat] + log_gamma[src]) * (s[seg] + delta[src] - t[rat])
    return LAMBDA + (1.0 - 2.0 * LAMBDA) * norm.cdf(lin)


def _gauge_fix(s, delta, log_gamma, ell, t, scale_mask, loc_mask):
    """Split-anchor identification — identical to fit_joint_iiic._gauge_fix.
    For spike tasks, the scale anchor is sn1_combined_v2:sn1 segs and the
    location anchor is mid-difficulty segs (see prep_k7.load_task)."""
    b = float(np.std(s[scale_mask])) if scale_mask.any() else 1.0
    if not np.isfinite(b) or b <= 0:
        b = 1.0
    s1 = s / b
    a = float(np.mean(s1[loc_mask])) if loc_mask.any() else 0.0
    return (s1 - a, delta / b, log_gamma.copy(), ell + np.log(b), t / b - a)


# ──────────────────────────────────────────────────────────────────────────
# Main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True,
                    choices=["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"])
    ap.add_argument("--variant", required=True, choices=["A", "B", "C"])
    ap.add_argument("--subsample", type=int, default=None,
                    help="cap N (stratified-ish) for smoke tests")
    ap.add_argument("--warmup", type=int, default=1500,
                    help="NUTS warmup steps (audit T2.4 target)")
    ap.add_argument("--samples", type=int, default=1500,
                    help="NUTS posterior samples (audit T2.4 target)")
    ap.add_argument("--chains", type=int, default=4,
                    help="NUTS chains (audit T2.4 target)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["svi", "nuts"], default="svi")
    ap.add_argument("--svi-steps", type=int, default=40000)
    ap.add_argument("--post-draws", type=int, default=500)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny settings + invariance check only")
    ap.add_argument("--gpu", type=int, default=None,
                    help="pin to a specific GPU (0 or 1); None = auto via CUDA_VISIBLE_DEVICES")
    ap.add_argument("--mem-fraction", type=float, default=0.45,
                    help="XLA_PYTHON_CLIENT_MEM_FRACTION (2 procs/GPU at 0.45)")
    args = ap.parse_args(argv)

    if args.smoke:
        args.subsample = args.subsample or 40000
        args.warmup, args.samples, args.chains = 150, 150, 1
        args.svi_steps = 4000

    _configure_compute_env(args.gpu, args.mem_fraction)

    # Late imports — must be AFTER environment configuration
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, Predictive
    from numpyro.infer.autoguide import AutoLowRankMultivariateNormal
    import numpyro.diagnostics as ndiag

    from .prep_k7 import load_task
    from . import model_k7

    numpyro.set_host_device_count(max(args.chains, 1))

    t0 = time.time()
    td = load_task(args.task, subsample=args.subsample, seed=args.seed)

    # Per-rater tier index (n_rater,) — required by all 3 variants
    tier_set = sorted(set(td.rater_expertise))
    tmap = {tt: i for i, tt in enumerate(tier_set)}
    rater_tier_idx = np.array([tmap[e] for e in td.rater_expertise], np.int32)
    n_tier = len(tier_set)

    print(f"[{args.task} V_{args.variant}] N={td.n_obs:,} S={td.n_seg:,} "
          f"R={td.n_rater:,} src={td.n_src} tiers={tier_set} "
          f"pos_rate={td.pos_rate:.3f} "
          f"scale_anchor={int(td.scale_anchor_seg.sum())} "
          f"loc_anchor={int(td.loc_anchor_seg.sum())} "
          f"gpu={os.environ.get('CUDA_VISIBLE_DEVICES','auto')}", flush=True)

    mkw = dict(
        seg_idx=jnp.array(td.seg_idx),
        rater_idx=jnp.array(td.rater_idx),
        src_idx=jnp.array(td.src_idx),
        rater_tier_idx=jnp.array(rater_tier_idx),
        n_seg=td.n_seg, n_rater=td.n_rater, n_src=td.n_src,
        n_tier=n_tier,
    )

    # Variant-specific arg additions
    model_fn = model_k7.get_model(args.variant)
    if args.variant == "C":
        # V_C: per-observation source weight
        weights = model_k7.compute_source_weights(td.src_idx, td.n_src)
        mkw["source_weight_per_obs"] = jnp.array(weights)

    # Inference
    if args.method == "nuts":
        # chain_method='vectorized': runs all `chains` chains in parallel on
        # ONE GPU via jax.vmap (memory scales with chain count, wall stays
        # ~constant). The numpyro default is 'sequential' which serialises
        # chains and multiplies wall time by num_chains — a 4× slowdown that
        # was caught in the first winner_full run (2026-05-28). Vectorized
        # produces identical posteriors but ~4× faster on a single GPU.
        mcmc = MCMC(NUTS(model_fn), num_warmup=args.warmup,
                    num_samples=args.samples, num_chains=args.chains,
                    chain_method="vectorized", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(args.seed), y=jnp.array(td.y), **mkw)
        post = mcmc.get_samples()
    else:
        guide = AutoLowRankMultivariateNormal(model_fn, rank=20)
        svi = SVI(model_fn, guide,
                  numpyro.optim.Adam(2e-3), Trace_ELBO())
        res = svi.run(jax.random.PRNGKey(args.seed), args.svi_steps,
                      y=jnp.array(td.y), progress_bar=False, **mkw)
        losses = np.asarray(res.losses)
        # return_sites must explicitly include un-deterministic sample sites
        # (a_skill, a_bias) — numpyro Predictive's default behavior with a
        # guide returns DETERMINISTIC sites + ALL guide samples, but the
        # per-tier latents are sample sites and need to be requested by name.
        post = Predictive(
            model_fn, guide=guide, params=res.params,
            num_samples=args.post_draws,
            return_sites=["s_j", "delta", "log_gamma", "ell", "t",
                          "a_skill", "a_bias"])(
            jax.random.PRNGKey(args.seed + 1), **mkw)
        post = {k: np.asarray(v) for k, v in post.items()}
        elbo_tail = float(np.mean(losses[-200:]))
        print(f"[{args.task} V_{args.variant}] SVI {args.svi_steps} steps "
              f"ELBO_tail={elbo_tail:.1f}  loss[0]={losses[0]:.1f}",
              flush=True)
    el = time.time() - t0

    # Posterior summaries
    s_m = np.asarray(post["s_j"]).mean(0)
    s_sd = np.asarray(post["s_j"]).std(0)
    dl_m = np.asarray(post["delta"]).mean(0)
    lg_m = np.asarray(post["log_gamma"]).mean(0)
    el_m = np.asarray(post["ell"]).mean(0)
    t_m = np.asarray(post["t"]).mean(0)
    # Per-tier means + per-rater posterior SDs (variant_selection gates need these).
    a_skill_m = np.asarray(post["a_skill"]).mean(0)
    a_skill_sd = np.asarray(post["a_skill"]).std(0)
    a_bias_m = np.asarray(post["a_bias"]).mean(0)
    a_bias_sd = np.asarray(post["a_bias"]).std(0)
    el_sd = np.asarray(post["ell"]).std(0)
    t_sd = np.asarray(post["t"]).std(0)

    # Gauge invariance check — identical to fit_joint_iiic
    p_raw = _predicted_p(s_m, dl_m, lg_m, el_m, t_m,
                         td.seg_idx, td.rater_idx, td.src_idx)
    s_id, dl_id, lg_id, el_id, t_id = _gauge_fix(
        s_m, dl_m, lg_m, el_m, t_m,
        td.scale_anchor_seg, td.loc_anchor_seg)
    p_id = _predicted_p(s_id, dl_id, lg_id, el_id, t_id,
                        td.seg_idx, td.rater_idx, td.src_idx)
    max_dev = float(np.max(np.abs(p_raw - p_id)))
    # Tolerance is a float-precision sanity check on the (mathematically exact)
    # gauge transform; default 1e-5 preserves the shipped contract. Env-override
    # exists ONLY for diagnostic NUTS re-runs whose posterior means carry more
    # float noise (e.g. spike: |Δp| ~1.03e-5 from exp(ell+γ) amplification).
    _gauge_tol = float(os.environ.get("FITJOINT_GAUGE_TOL", "1e-5"))
    assert max_dev < _gauge_tol, (
        f"GAUGE TRANSFORM NOT LIKELIHOOD-INVARIANT "
        f"(max |Δp|={max_dev:.2e} ≥ {_gauge_tol:.1e})")

    # Convergence diagnostics
    if args.method == "nuts" and args.chains >= 2:
        def _rhat(name):
            x = np.asarray(post[name])
            try:
                return float(np.nanmax(ndiag.gelman_rubin(
                    x.reshape(args.chains, -1, *x.shape[1:]))))
            except Exception:
                return float("nan")
        conv = {
            "method": "nuts",
            "rhat_max": {k: _rhat(k) for k in ("s_j", "ell", "t")
                         if k in post},
        }
    elif args.method == "nuts":
        conv = {"method": "nuts",
                "rhat_max": "n/a (single chain — smoke; real run ≥ 2)"}
    else:
        conv = {"method": "svi",
                "note": "SVI: R̂ not applicable; s_sd variance-calibrated "
                        "vs NUTS-subsample downstream"}

    # Persist
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "task": args.task,
        "variant": args.variant,
        "phase": 9,
        "K": 7,
        "n_obs": td.n_obs,
        "n_seg": td.n_seg,
        "n_rater": td.n_rater,
        "n_src": td.n_src,
        "n_tier": n_tier,
        "tiers": tier_set,
        "subsample": args.subsample,
        "method": args.method,
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
        "s_id_mean_over_loc_anchor": (
            float(np.mean(s_id[td.loc_anchor_seg]))
            if td.loc_anchor_seg.any() else None),
        "s_id_std_over_scale_anchor": (
            float(np.std(s_id[td.scale_anchor_seg]))
            if td.scale_anchor_seg.any() else None),
        "pos_rate": td.pos_rate,
    }
    sum_path = OUT / f"{args.task}_k7_{args.variant}_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2))

    np.savez_compressed(
        OUT / f"{args.task}_k7_{args.variant}_posterior.npz",
        s_id_mean=s_id, s_sd=s_sd,
        ell_id_mean=el_id, t_id_mean=t_id,
        ell_id_sd=el_sd, t_id_sd=t_sd,
        a_skill_mean=a_skill_m, a_skill_sd=a_skill_sd,
        a_bias_mean=a_bias_m, a_bias_sd=a_bias_sd,
        tiers=np.array(tier_set),
        seg_ids=td.seg_ids, rater_ids=td.rater_ids,
        rater_canonical=np.array(td.rater_canonical),
        rater_expertise=np.array(td.rater_expertise),
        src_names=np.array(td.src_names),
        rater_tier_idx=rater_tier_idx)
    print(f"[{args.task} V_{args.variant}] done {el:.1f}s  method={args.method}"
          f"  gauge|Δp|={max_dev:.1e}  conv={conv.get('method')}"
          f"  s̄_loc={summary['s_id_mean_over_loc_anchor']}"
          f"  sd_scale={summary['s_id_std_over_scale_anchor']}"
          f"  → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
