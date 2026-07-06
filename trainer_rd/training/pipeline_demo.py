"""Step 8 — end-to-end seamless pipeline demo (the user's stated objective).

Take the test → the results inform the trainer → re-certify. One runnable chain
that exercises every prior step against the real bank/labels:

  1. EVAL      — a simulated mid-skill candidate sits the adaptive certification
                 test (core_mcmc_general engine + AD6 policy); some tasks FAIL/REFER.
  2. HANDOFF   — build a TrainingSeed from the eval posterior (per-task marginals,
                 variance-inflated; eval-seen seg_ids recorded). D1/D2.
  3. TRAIN     — N daily sessions under the Tier-2 policy on the real bank, with
                 the SAME simulated learner responding AND actually learning
                 (learner_sim). Eval-seen segments excluded (D12); feedback only
                 on coherent items (D11). A between-session forgetting gap is
                 applied via the LT2 hook (D9).
  4. RE-CERT   — the trainee re-sits the test; verdicts flip toward PASS.
  5. REPORT    — one structured object capturing the whole telemetry chain:
                 verdict progression, skill/bias trajectories, mode decisions,
                 RT fields, graduation, and the re-test verdict flips.

This is a simulation harness, not production — but every component is the same
module the eventual sibling-app would call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import engine.core_mcmc_general as eng
from engine.policy_general import AD6Policy, PASS, FAIL, PENDING
from training.bank_adapter import (ANCHORED_ALPHA_SIGMA, ANCHORED_ALPHA_T,
                                   BankAdapter, ELL_STAR, SIGMA_STAR,
                                   SIGMA_INF, TASK_CODES)
from engine.instrument_v15 import instrument

from training.learner_sim import Learner, LearnerParams
from training.training_seed import build_seed_from_state, LearnerLedger
from training.training_filter import TaskFilter, filter_from_seed_task
from training.trainer_policy import TrainerPolicy, ModeThresholds

T_STAR = 0.30
EPOCH0 = 1_770_000_000.0
DAY = 86_400.0


@dataclass
class PipelineReport:
    tasks: list
    eval_verdicts: list
    eval_skill: list                 # posterior-mean ℓ per task at eval
    recert_verdicts: list
    recert_skill: list
    true_skill_eval: list            # learner's true ℓ before training
    true_skill_post: list            # learner's true ℓ after training
    flips: list                      # tasks that improved FAIL/REFER → PASS
    n_eval: int
    n_train_trials: int
    n_sessions: int
    train_log_len: int


def _run_eval(true_t, true_l, bank, *, tasks, session_id, n_part=500,
              max_q=350, seed=0, exclude=None, pool_size=800,
              ell_star_vec=None, Sigma_l=None, Sigma_t=None):
    """Adaptive certification session against a simulated rater (true_t,true_l).

    The per-task candidate pool is subsampled to `pool_size` (the prototype's
    stand-in for the engine's `n_subsample` coarse-to-fine selector) so the
    full-bank scan per question stays tractable; the verdicts are unaffected at
    this resolution.

    Instrument opt-in (M11/v15 staging — instrument_v15.Instrument.for_tasks):
    ell_star_vec overrides the v14 module constants; Sigma_l/Sigma_t give the
    hierarchical prior an unstructured covariance (Sigma_t is the v15 corr_t
    swap). Defaults reproduce the shipped M8/M10 behavior exactly."""
    K = len(tasks)
    rng = np.random.default_rng(seed)
    # per-task candidate arrays (engine signal banks) from the real bank,
    # subsampled to a tractable working set (spanning the signal range)
    pools = []
    for t in tasks:
        p = bank.candidates(t, exclude_segids=exclude)
        if len(p) > pool_size:
            order = np.argsort(p.s_mean)
            pick = order[np.round(np.linspace(0, len(p) - 1, pool_size)).astype(int)]
            from training.bank_adapter import TaskCandidates as _TC
            p = _TC(t, p.seg_id[pick], p.s_mean[pick], p.s_sd[pick],
                    p.y_star[pick], p.margin[pick], p.coherent[pick])
        pools.append(p)
    bank_signals = [p.s_mean.copy() for p in pools]
    bank_sds = [p.s_sd.copy() for p in pools]
    bank_segids = [p.seg_id.copy() for p in pools]

    state = eng.make_state_hier(n_part, K, r_assumed=0.378, rng=rng,
                                Sigma_l=Sigma_l, Sigma_t=Sigma_t)
    cuts = (list(ell_star_vec) if ell_star_vec is not None
            else [ELL_STAR[t] for t in tasks])
    policy = AD6Policy(cuts, [1.0] * K)
    policy.reset(K)
    n_per_task = [0] * K
    served = []
    lnr_rng = np.random.default_rng(seed + 1)
    for q in range(max_q):
        active = [k for k in range(K)
                  if policy._verdicts[k] == PENDING and len(bank_signals[k]) > 0]
        if not active:
            break
        # Phase-3.5 production path: the A-optimal selector marginalizes each
        # candidate over its s posterior (prefers high-precision items), and
        # the update marginalizes the observation likelihood too (F20).
        k, s, s_sd, seg_id = eng.choose_item(state, bank_signals,
                                             active_domains=active,
                                             bank_sds=bank_sds, return_sd=True,
                                             bank_segids=bank_segids)
        j = int(np.where(bank_segids[k] == seg_id)[0][0])
        # the simulated rater responds at the segment's (unknown) TRUE signal,
        # drawn from its posterior s_real ~ N(s, s_sd²)
        s_real = s + s_sd * lnr_rng.standard_normal()
        y = eng.simulate_response(s_real, true_t[k], true_l[k], lnr_rng)
        eng.update(state, k, s, y, s_sd=s_sd)
        # serve-once
        bank_signals[k] = np.delete(bank_signals[k], j)
        bank_sds[k] = np.delete(bank_sds[k], j)
        bank_segids[k] = np.delete(bank_segids[k], j)
        if eng.ess(state["w"]) < 0.5 * n_part:
            eng.resample_and_rejuvenate(state, rng, 15, 2.38 / np.sqrt(2 * K))
        n_per_task[k] += 1
        served.append(int(seg_id))
        dec = policy(state, {}, n_per_task, K)
        if dec.stop:
            break
    verdicts = policy.finalize_verdicts()
    return state, verdicts, policy._last_diag, served, len(served)


def run_pipeline(*, tasks=(1, 2, 3), n_sessions=4, trials_per_session=80,
                 seed=0, eval_n_part=500, eval_max_q=350, eval_pool=800):
    """M18 (D31): the pipeline is now v15-COHERENT end-to-end — the eval
    (cuts + corr_t t-block prior) and the trainer mastery targets both use
    instrument('v15'), so the trainer's bar equals the re-cert bar (the D25
    coherence requirement). Particle count stays a demo speed knob
    (eval_n_part; the 1200-particle staging is a production concern)."""
    bank = BankAdapter()
    K = len(tasks)
    ins = instrument("v15")
    sub = ins.for_tasks(tasks)
    cuts = np.array(sub["ell_star"])
    sig_star = np.exp(-cuts)
    sig_inf = ins.sigma_inf
    # a mid-skill candidate: below the cut on the trained tasks, with bias.
    # true ℓ chosen a bit below each cut so eval should NOT pass.
    true_l = np.array(cuts) - 0.45
    true_t = np.array([0.7, -0.6, 0.5][:K])          # engine-coord θ (=−t_plan)

    # 1 ── EVAL (v15 cuts + corr_t prior block)
    state, ev_verdicts, ev_diag, ev_served, n_eval = _run_eval(
        true_t, true_l, bank, tasks=tasks, session_id="eval-demo", seed=seed,
        n_part=eval_n_part, max_q=eval_max_q, pool_size=eval_pool,
        ell_star_vec=list(cuts), Sigma_l=sub["Sigma_l"],
        Sigma_t=sub["Sigma_t"])
    ml = [float((state["w"] * state["l"][:, k]).sum()) for k in range(K)]

    # 2 ── HANDOFF
    seed_obj = build_seed_from_state(
        state, session_id="eval-demo", ell_star=list(cuts),
        verdicts=ev_verdicts, diagnostics=ev_diag, inflate=1.5,
        seg_ids=ev_served, session_type="eval")
    ledger = LearnerLedger("demo-learner")
    ledger.append_entry(seed_path="(in-memory)", session_id="eval-demo",
                        session_type="eval", created_at="2026-06-10T00:00:00+00:00")

    # 3 ── TRAIN: the learner starts at its true eval state; filters seed from
    # the eval posterior marginals (D1/D2). σ_∞ = data-grounded expert ceiling.
    learner_inf = [min(0.82 * sig_star[i], float(sig_inf[t]))
                   for i, t in enumerate(tasks)]
    # the learner needs per-task params; emulate by storing a vector σ_∞
    learner = Learner(np.exp(-true_l), -true_t,
                      LearnerParams(alpha_t=0.2, alpha_sigma=0.06,
                                    sigma_inf=np.array(learner_inf),
                                    q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft"),
                      seed=seed + 5)
    # assumed rates = the F62 anchored priors (D29; fp < tp, F17-safe)
    fparams = [LearnerParams(alpha_t=ANCHORED_ALPHA_T,
                             alpha_sigma=ANCHORED_ALPHA_SIGMA,
                             sigma_inf=learner_inf[i],
                             q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
               for i in range(K)]
    filters = [filter_from_seed_task(seed_obj, i, fparams[i], inflated=True,
                                     rng_seed=seed + 10 + i) for i in range(K)]

    class _Remap:
        def __init__(self, b, ts): self.b, self.ts = b, ts
        def candidates(self, task, **kw): return self.b.candidates(self.ts[task], **kw)

    pol = TrainerPolicy(
        filters, list(cuts), list(sig_star),
        _Remap(bank, tasks), thresholds=ModeThresholds(sd_floor=0.23),
        seed=seed + 20, exclude_segids=ev_served)        # D12: exclude eval-seen

    n_train = 0
    stim_rng = np.random.default_rng(seed + 33)
    for sess in range(n_sessions):
        now0 = EPOCH0 + sess * DAY
        # LT2 forgetting gap between sessions (identity + slight widening, D9)
        if sess > 0:
            for f in filters:
                f.propagate_gap(DAY, widen=0.03)
        for k in range(trials_per_session):
            if pol.all_mastered():
                break
            choice = pol.step(now=now0 + k, session_id=f"train-{sess}")
            if choice is None:
                break
            tk = choice["task"]
            # learner experiences the segment's true signal (F20)
            s_real = choice["s"] + choice["s_sd"] * stim_rng.standard_normal()
            y = learner.step(s_real, tk, choice["y_star"], feedback=True)
            pol.record(choice, y, session_id=f"train-{sess}",
                       rt_ms=float(800 + 600 * stim_rng.random()))
            n_train += 1
        ledger.append_entry(seed_path="(in-memory)", session_id=f"train-{sess}",
                            session_type="training")
        if pol.all_mastered():
            break

    true_l_post = np.array([-np.log(learner.sigma[i]) for i in range(K)])

    # 4 ── RE-CERT (fresh prior, integrity per OQ3 — exclude all seen segs;
    # SAME v15 instrument as the initial eval — the D31 coherence point)
    seen_all = set(ev_served) | pol._served
    rc_state, rc_verdicts, rc_diag, rc_served, n_rc = _run_eval(
        -learner.t, -np.log(learner.sigma), bank, tasks=tasks,
        session_id="recert-demo", seed=seed + 99, exclude=seen_all,
        n_part=eval_n_part, max_q=eval_max_q, pool_size=eval_pool,
        ell_star_vec=list(cuts), Sigma_l=sub["Sigma_l"],
        Sigma_t=sub["Sigma_t"])
    rc_ml = [float((rc_state["w"] * rc_state["l"][:, k]).sum()) for k in range(K)]

    flips = [TASK_CODES[tasks[k]] for k in range(K)
             if ev_verdicts[k] != PASS and rc_verdicts[k] == PASS]

    return PipelineReport(
        tasks=[TASK_CODES[t] for t in tasks],
        eval_verdicts=ev_verdicts, eval_skill=[round(x, 3) for x in ml],
        recert_verdicts=rc_verdicts, recert_skill=[round(x, 3) for x in rc_ml],
        true_skill_eval=[round(float(x), 3) for x in true_l],
        true_skill_post=[round(float(x), 3) for x in true_l_post],
        flips=flips, n_eval=n_eval, n_train_trials=n_train,
        n_sessions=n_sessions, train_log_len=len(pol.log))


if __name__ == "__main__":
    rep = run_pipeline()
    print("tasks           :", rep.tasks)
    print("ℓ* cut-scores   :", [round(ELL_STAR[t], 3) for t in (1, 2, 3)])
    print("eval verdicts   :", rep.eval_verdicts)
    print("eval ℓ̂ (post)   :", rep.eval_skill)
    print("true ℓ @ eval   :", rep.true_skill_eval)
    print(f"→ trained {rep.n_train_trials} trials over {rep.n_sessions} sessions")
    print("true ℓ post     :", rep.true_skill_post)
    print("recert verdicts :", rep.recert_verdicts)
    print("recert ℓ̂        :", rep.recert_skill)
    print("PASS flips      :", rep.flips)
