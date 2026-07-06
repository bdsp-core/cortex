"""M18-2 — hierarchical (EB shrinkage) refit of the EXTSET dynamics fits.

F58's mandate on real data: personalized fits on short logs OVERFIT
(held-out −0.008/trial in sim); the fix is hierarchy. This study builds the
two-stage empirical-Bayes estimator and validates it with honest splits:

  WITHIN-user temporal split: reads[0:n/2] (A) vs reads[n/2:n] (B) per
  (user × domain). Fit A and B independently →
    reliability per coordinate  = corr(A, B) across fits (the real-data
                                  identifiability measure, F46-style);
    within-fit noise            V̄ = Var(A − B)/2 per coordinate.
  CROSS-user split (60/40): population (μ, Σ_pop) from TRAIN users' A-fits
  by moment deconvolution Σ_pop = max(Var(A) − V̄, ε); TEST users get
    shrunk_i = μ + B·(A_i − μ),   B = Σ_pop/(Σ_pop + V̄)   (per coordinate).
  HELD-OUT validation (test users): second-half evidence (score_from=n/2,
  3-seed CRN average per F59) under (a) population mean μ, (b) raw
  personalized A, (c) shrunk A. F58 predicts raw < pop ≲ shrunk.

M19-1 refinement: V̄ now comes from an ODD/EVEN interleaved split within
the first half (drift-free noise; the temporal A−B variant is reported for
contrast — its drift inflation was F65's over-shrink cause). The M18
temporal-V̄ record is preserved in data_hier_shrink.npz; this run saves to
data_hier_shrink_oe.npz.

Run:  python3 -m studies.study_hier_shrink [--smoke]   (Pool(42))
      → figures/data_hier_shrink.npz
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np

from training.dynamics_fit import filter_evidence, fit_learner
from training.extset_adapter import load_task2
from training.learner_sim import LearnerParams
from studies.study_extset_dynamics import BASE, PRIOR_MEAN, PRIOR_SD

MIN_READS = 80


def _make_trials(d2, mi, dom):
    y = (d2["label"][mi] == dom).astype(int)
    g = d2["gold"][mi]
    return np.column_stack([d2["s_loo"][mi, dom], y,
                            (g == dom).astype(int),
                            d2["s_sd"][mi, dom], (g >= 0).astype(int)])


def _fit_job(job):
    key, trials = job
    try:
        fr = fit_learner(trials, BASE, seed=42, maxiter=120,
                         prior_sd=PRIOR_SD, prior_mean=PRIOR_MEAN)
        return (key, fr.x)
    except Exception:
        return None


def _ev_job(job):
    key, trials, params_x, half = job
    from dataclasses import replace
    p = replace(BASE, alpha_t=float(np.exp(params_x[0])),
                alpha_sigma=float(np.exp(params_x[1])),
                sigma_inf=float(np.exp(-params_x[2])))
    ev = np.mean([filter_evidence(trials, p, seed=sd, prior_sd=PRIOR_SD,
                                  prior_mean=PRIOR_MEAN, score_from=half)
                  for sd in (42, 43, 44)])
    return (key, float(ev))


def main(smoke=False):
    d2 = load_task2()
    counts = {}
    for u in d2["user"]:
        counts[u] = counts.get(u, 0) + 1
    users = sorted([u for u, c in counts.items() if c >= MIN_READS])
    if smoke:
        users = users[:4]
    order = np.argsort(d2["t"], kind="stable")
    # build A/B fit jobs + full sequences for held-out scoring
    fitjobs, seqs = [], {}
    for u in users:
        m = order[np.isin(order, np.where(d2["user"] == u)[0])]
        for dom in range(6):
            keep = np.isfinite(d2["s_loo"][m, dom])
            mi = m[keep]
            if mi.size < MIN_READS:
                continue
            half = mi.size // 2
            seqs[(u, dom)] = (_make_trials(d2, mi, dom), half)
            fitjobs.append(((u, dom, "A"), _make_trials(d2, mi[:half], dom)))
            fitjobs.append(((u, dom, "B"), _make_trials(d2, mi[half:], dom)))
            # M19-1: DRIFT-AWARE noise — interleaved odd/even split WITHIN
            # the first half (both sub-fits see the same drift trajectory,
            # so Var(A1−A2) is pure estimation noise, unlike A−B which
            # includes real within-user drift — the F65 over-shrink cause)
            fitjobs.append(((u, dom, "A1"),
                            _make_trials(d2, mi[:half][0::2], dom)))
            fitjobs.append(((u, dom, "A2"),
                            _make_trials(d2, mi[:half][1::2], dom)))
    print(f"[{'SMOKE' if smoke else 'FULL'}] {len(users)} users, "
          f"{len(fitjobs)} half-fits")
    with Pool(4 if smoke else 42) as pool:
        fits = dict(r for r in pool.map(_fit_job, fitjobs) if r is not None)
    keys = sorted({(u, d) for (u, d, h) in fits
                   if (u, d, "A") in fits and (u, d, "B") in fits})
    A = np.array([fits[(u, d, "A")] for u, d in keys])
    B = np.array([fits[(u, d, "B")] for u, d in keys])
    names = ("log_alpha_t", "log_alpha_s", "ell_inf")
    rel = [float(np.corrcoef(A[:, j], B[:, j])[0, 1]) for j in range(3)]
    Vbar_temporal = ((A - B).var(axis=0)) / 2.0
    # drift-aware V̄: A1/A2 use half of A's data ⇒ their fit variance is ~2×
    # A's (ML variance ~ 1/n) ⇒ V̄_A = Var(A1−A2)/2 / 2 = Var(A1−A2)/4
    ok_oe = [(u, d) for (u, d) in keys
             if (u, d, "A1") in fits and (u, d, "A2") in fits]
    A1 = np.array([fits[(u, d, "A1")] for u, d in ok_oe])
    A2 = np.array([fits[(u, d, "A2")] for u, d in ok_oe])
    Vbar = ((A1 - A2).var(axis=0)) / 4.0
    print("split-half reliability per coordinate:",
          {n: round(r, 2) for n, r in zip(names, rel)})
    print("V̄ temporal (drift-inflated):", np.round(Vbar_temporal, 3),
          " V̄ odd/even (drift-free):", np.round(Vbar, 3))
    # cross-user split
    uu = sorted({u for u, d in keys})
    rng = np.random.default_rng(7)
    test_users = set(rng.choice(uu, size=max(1, int(0.4 * len(uu))),
                                replace=False).tolist())
    trainA = np.array([fits[(u, d, "A")] for u, d in keys
                       if u not in test_users])
    mu = trainA.mean(axis=0)
    Sig = np.maximum(trainA.var(axis=0) - Vbar, 1e-4)
    Bshrink = Sig / (Sig + Vbar)
    print("population μ:", np.round(mu, 3), " shrink factors B:",
          np.round(Bshrink, 2))
    # held-out second-half evidence on test users
    evjobs = []
    for (u, d) in keys:
        if u not in test_users:
            continue
        trials, half = seqs[(u, d)]
        a = fits[(u, d, "A")]
        sh = mu + Bshrink * (a - mu)
        evjobs += [((u, d, "pop"), trials, mu, half),
                   ((u, d, "raw"), trials, a, half),
                   ((u, d, "shrunk"), trials, sh, half)]
    with Pool(4 if smoke else 42) as pool:
        evs = dict(pool.map(_ev_job, evjobs))
    tk = [(u, d) for (u, d) in keys if u in test_users]
    n2 = np.array([seqs[k][0].shape[0] - seqs[k][1] for k in tk])
    d_raw = np.array([evs[(u, d, "raw")] - evs[(u, d, "pop")]
                      for u, d in tk]) / n2
    d_shr = np.array([evs[(u, d, "shrunk")] - evs[(u, d, "pop")]
                      for u, d in tk]) / n2
    for nm, dd in (("raw personalized", d_raw), ("shrunk", d_shr)):
        print(f"held-out Δ/read vs population, {nm:>17}: {dd.mean():+.4f} "
              f"± {dd.std() / np.sqrt(len(dd)):.4f}  (n={len(dd)} fits)")
    if not smoke:
        np.savez("figures/data_hier_shrink_oe.npz",
                 A=A, B=B, keys=np.array([(u, d) for u, d in keys]),
                 mu=mu, Sig=Sig, Bshrink=Bshrink, Vbar=Vbar,
                 rel=np.array(rel), d_raw=d_raw, d_shr=d_shr)
        print("saved figures/data_hier_shrink_oe.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
