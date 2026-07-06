"""M12/V2b — does eval re-certification catch the trainer's false graduations?

F28/F29 established that the SHIPPED trainer gate declares mastery for
essentially any learner after ~50 trials (belief converges to the model's
σ_∞ attractor; tier2 placement is information-poor for the ℓ-vs-ℓ* contrast).
Production, however, never certifies on the trainer's word: graduation routes
the candidate to an EVAL re-certification (fresh hierarchical prior, static
model, A-optimal decision-aligned items, AD6 verdict). This study quantifies
that backstop END-TO-END:

  trainer (shipped tier2) runs to its (possibly false) declaration
  → the learner, FROZEN at its at-declaration true state, sits a real
    K=1 eval session (the actual engine: choose_item/update/AD6)
  → record the verdict. False certification = eval PASS while the latent
    state is below the bar.

Learner responses during eval use the learner's TRUE response function
(zoo link misspecifications included), at items the ENGINE chooses. The
learner is frozen: eval delivers no feedback (σ-learning is f-gated off in
the model; freezing t as well avoids the known F22 artifact).

Run: python3 -m studies.study_recert_backstop [--smoke] → figures/data_recert_backstop.npz
"""
from __future__ import annotations

import sys

import numpy as np

import engine.core_mcmc_general as eng
from training.bank_adapter import BankAdapter, ELL_STAR, TaskCandidates
from engine.policy_general import AD6Policy, PENDING
from studies.study_misspec import TASK, ZOO, build_pool, run_one

MEMBERS = ("wellspec", "powerlaw", "asym_lapse", "anti", "careless",
           "static_below")


def eval_recert(lnr, bank, *, exclude=None, n_part=500, max_q=350,
                pool_size=800, seed=0):
    """K=1 re-certification of a FROZEN learner on TASK, real engine + AD6."""
    rng = np.random.default_rng(seed)
    p = bank.candidates(TASK, exclude_segids=exclude)
    if len(p) > pool_size:
        order = np.argsort(p.s_mean)
        pick = order[np.round(np.linspace(0, len(p) - 1, pool_size)).astype(int)]
        p = TaskCandidates(TASK, p.seg_id[pick], p.s_mean[pick], p.s_sd[pick],
                           p.y_star[pick], p.margin[pick], p.coherent[pick])
    sig, sds, ids = [p.s_mean.copy()], [p.s_sd.copy()], [p.seg_id.copy()]
    state = eng.make_state_hier(n_part, 1, r_assumed=0.378, rng=rng)
    policy = AD6Policy([ELL_STAR[TASK]], [1.0])
    policy.reset(1)
    n_per_task = [0]
    lnr_rng = np.random.default_rng(seed + 1)
    for q in range(max_q):
        if policy._verdicts[0] != PENDING or len(sig[0]) == 0:
            break
        k, s, s_sd, seg_id = eng.choose_item(state, sig, active_domains=[0],
                                             bank_sds=sds, return_sd=True,
                                             bank_segids=ids)
        j = int(np.where(ids[k] == seg_id)[0][0])
        s_real = s + s_sd * lnr_rng.standard_normal()
        y = int(lnr_rng.random() < lnr.p_yes(s_real, 0))   # TRUE response fn
        eng.update(state, k, s, y, s_sd=s_sd)
        sig[k] = np.delete(sig[k], j)
        sds[k] = np.delete(sds[k], j)
        ids[k] = np.delete(ids[k], j)
        if eng.ess(state["w"]) < 0.5 * n_part:
            eng.resample_and_rejuvenate(state, rng, 15, 2.38 / np.sqrt(2.0))
        n_per_task[0] += 1
        if policy(state, {}, n_per_task, 1).stop:
            break
    v = policy.finalize_verdicts()[0]
    ell_hat = float((state["w"] * state["l"][:, 0]).sum())
    return v, ell_hat, n_per_task[0]


def main(smoke=False):
    bank = BankAdapter()
    pool = build_pool()
    members = ("static_below", "careless") if smoke else MEMBERS
    seeds = range(3 if smoke else 15)
    out = {}
    for mem in members:
        rows = []
        for s in seeds:
            r = run_one(mem, "tier2", s, pool, stop_at_decl=True)
            if r["n_decl"] > 400:          # trainer never declared (rare)
                continue
            lnr = r["_lnr"]
            v, ell_hat, nq = eval_recert(lnr, bank,
                                         exclude=set(int(x) for x in pool.seg_id),
                                         seed=4000 + s)
            rows.append((v, ell_hat, nq, r["ell_decl"], r["t_decl"]))
        vs = [r[0] for r in rows]
        out[mem] = rows
        n_pass = sum(v == "PASS" for v in vs)
        print(f"  {mem:>13}: eval verdicts after trainer declaration "
              f"(n={len(rows)}): PASS {n_pass}, "
              f"{ {v: vs.count(v) for v in set(vs) if v != 'PASS'} }  "
              f"med eval ℓ̂ {np.median([r[1] for r in rows]):+.2f} "
              f"vs cut {ELL_STAR[TASK]:.3f}")
    if not smoke:
        np.savez("figures/data_recert_backstop.npz",
                 **{f"{m}_verdicts": np.array([r[0] for r in out[m]])
                    for m in members},
                 **{f"{m}_ellhat": np.array([r[1] for r in out[m]])
                    for m in members},
                 **{f"{m}_nq": np.array([r[2] for r in out[m]])
                    for m in members})
        print("saved figures/data_recert_backstop.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
