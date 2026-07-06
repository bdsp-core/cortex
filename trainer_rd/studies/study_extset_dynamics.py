"""M17 (F62) — REAL-DATA dynamics anchoring from EXTSET reads (OQ8 = YES).

The user confirmed EXTSET novices received per-read feedback, unblocking the
W3 dynamics workstream (§3C route B, honestly labeled): the within-user gain
is feedback-driven and therefore R–W-interpretable. This study fits the
trainer's learning-dynamics parameters on the REAL task2 reads and produces
the first empirically-grounded population priors for (α_t, α_σ, ℓ_∞).

DATA/PROTOCOL (per the standing EXTSET protocols):
  reads      task2 (timestamps exist), one-vs-rest binary frame per domain
             (d2..d7) — the same reduction as W1/D11.
  signals    exact leave-one-user-out consensus (F40 MANDATORY; shipped-s
             only as sensitivity).
  feedback   y* = expert ≥3/4 consensus (OQ9 primary). Reads WITHOUT
             consensus enter the sequence with fb=0 — the response informs
             the state (reweight) but fires no dynamics update (the MA-3
             semantics; the platform's actual displayed label is unknown for
             those cases — source-gold y* is the prespecified sensitivity).
  fitting    dynamics_fit.fit_learner per (user × domain frame), users with
             ≥60 reads; evidence prior centered on the F43 novice population
             (θ, ℓ) = (−1.0, −0.3), sd 0.5.
  CAVEATS (recorded, not hidden): non-adaptive placement (fine for
  likelihood-based fitting); between-read gaps are ignored, so any real
  forgetting deflates fitted α's — the conservative D14 direction; the
  one-vs-rest frames inherit forced-choice competition (F43/F49).

OUTPUT: per-fit (user, domain, n_reads, log α_t, log α_σ, ℓ_∞); population
medians/IQRs per domain and pooled; comparison against the D7 defaults
(α_t .2, α_σ .06) and the M16 synthetic POP_TRUE; convergent-validity check
corr(fitted ℓ_∞, user gold-accuracy).

Run:  python3 -m studies.study_extset_dynamics [--smoke]   (Pool(42))
      → figures/data_extset_dynamics.npz
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np

from training.dynamics_fit import fit_learner
from training.extset_adapter import load_task2
from training.learner_sim import LearnerParams

MIN_READS = 60
PRIOR_MEAN = (-1.0, -0.3)     # F43 novice population, engine coords (θ, ℓ)
PRIOR_SD = 0.5
BASE = LearnerParams(alpha_t=0.2, alpha_sigma=0.06,
                     sigma_inf=float(np.exp(-(-0.3 + 0.5))),  # ≈ ℓ∞ 0.2 center
                     q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")


def _one_fit(job):
    user, dom, trials = job
    try:
        fr = fit_learner(trials, BASE, seed=42, maxiter=120,
                         prior_sd=PRIOR_SD, prior_mean=PRIOR_MEAN)
        return (user, dom, len(trials), *fr.x)
    except Exception:
        return None


def build_jobs(d2, users, *, smoke=False):
    order = np.argsort(d2["t"], kind="stable")
    jobs = []
    for u in users:
        m = order[np.isin(order, np.where(d2["user"] == u)[0])]
        for dom in range(6):
            s = d2["s_loo"][m, dom]
            keep = np.isfinite(s)
            if keep.sum() < MIN_READS:
                continue
            mi = m[keep]
            y = (d2["label"][mi] == dom).astype(int)
            g = d2["gold"][mi]
            ystar = (g == dom).astype(int)
            fb = (g >= 0).astype(int)          # consensus exists ⇒ feedback
            trials = np.column_stack([d2["s_loo"][mi, dom], y, ystar,
                                      d2["s_sd"][mi, dom], fb])
            jobs.append((int(u), dom, trials))
        if smoke and len(jobs) >= 12:
            break
    return jobs


def main(smoke=False):
    d2 = load_task2()
    counts = {}
    for u in d2["user"]:
        counts[u] = counts.get(u, 0) + 1
    users = sorted([u for u, c in counts.items() if c >= MIN_READS])
    if smoke:
        users = users[:3]
    jobs = build_jobs(d2, users, smoke=smoke)
    print(f"[{'SMOKE' if smoke else 'FULL'}] {len(users)} users ≥{MIN_READS} "
          f"reads → {len(jobs)} (user × domain) fits")
    with Pool(4 if smoke else 42) as pool:
        rows = [r for r in pool.map(_one_fit, jobs) if r is not None]
    arr = np.array([r[2:] for r in rows])       # n, lat, las, linf
    lat, las, linf = arr[:, 1], arr[:, 2], arr[:, 3]
    print(f"fits kept: {len(rows)}")
    print(f"  α_t   median {np.exp(np.median(lat)):.3f} "
          f"IQR [{np.exp(np.percentile(lat, 25)):.3f}, "
          f"{np.exp(np.percentile(lat, 75)):.3f}]   (D7 default 0.200)")
    print(f"  α_σ   median {np.exp(np.median(las)):.3f} "
          f"IQR [{np.exp(np.percentile(las, 25)):.3f}, "
          f"{np.exp(np.percentile(las, 75)):.3f}]   (D7 default 0.060)")
    print(f"  ℓ_∞   median {np.median(linf):+.3f} "
          f"IQR [{np.percentile(linf, 25):+.3f}, "
          f"{np.percentile(linf, 75):+.3f}]   (novice ℓ ≈ −0.3, F43)")
    # convergent validity: fitted ceiling vs realized gold accuracy per user
    acc, ceil = {}, {}
    for (u, dom, n, a, b, c) in rows:
        ceil.setdefault(u, []).append(c)
    for u in ceil:
        m = (d2["user"] == u) & (d2["gold"] >= 0)
        acc[u] = float((d2["label"][m] == d2["gold"][m]).mean())
    us = sorted(ceil)
    r = np.corrcoef([np.mean(ceil[u]) for u in us],
                    [acc[u] for u in us])[0, 1]
    print(f"  convergent validity: corr(mean fitted ℓ_∞, gold accuracy) "
          f"= {r:+.3f} over {len(us)} users")
    if not smoke:
        np.savez("figures/data_extset_dynamics.npz",
                 rows=np.array([r[2:] for r in rows]),
                 users=np.array([r[0] for r in rows]),
                 domains=np.array([r[1] for r in rows]))
        print("saved figures/data_extset_dynamics.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
