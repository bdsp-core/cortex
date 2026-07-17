"""Local closed-loop orchestration (G2) — the test → train → retest loop.

Wires the ported trainer to the REAL exam (`scripts/session_controller.CortexSession`,
the SMC + AD6 engine) over one shared bank and one Exposure Ledger, implementing
§1 of docs/TRAINER_INTEGRATION_PLAN.md:

    E0 (all 7, fresh, AD6) → weak set W → build per-task filters from the exam
    posterior → train W (feedback-safe items, no-repeat via the ledger, learner
    learns) → full fresh retest E_{i+1} → iterate until W empty or max_rounds.

Simulation conventions (SDT frame): a segment's per-task signal s encodes
one-vs-rest evidence, so the ground-truth label is y* = 1[s > 0]; the learner
(`trainer.sim.Learner`, plan coords) responds via the shared λ-lapse probit and
learns only on feedback (training) trials. The exam gives no feedback (measurement),
so it uses the learner's CURRENT state without advancing it.

D-INT-4: the retest draws from `inputs.without(seen)` — a hard exclusion of every
seg the participant has been shown — so each retest is an independent cold exam.
D-INT-7: training draws exclude the same seen set (the strict no-repeat used here;
the count-window policy is exercised by the ledger's own tests).
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

from trainer import domains
from trainer.bank_adapter import TaskCandidates
from trainer.conventions import plan_to_engine
from trainer.dynamics import LearnerParams
from trainer.filter import TaskFilter
from trainer.handoff import inflate_cloud

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO, "scripts")


class SimBankAdapter:
    """SDT-frame trainer bank built from a `K7EngineInputs` — the SAME bank the
    exam draws from, so exposure exclusion is meaningful. y* = 1[s>0], margin=1
    (the SDT label is deterministic given s), coherent by construction."""

    def __init__(self, inputs):
        self.inputs = inputs
        self.codes = list(inputs.task_codes)
        self._pools = [self._pool(k) for k in range(len(self.codes))]

    def _pool(self, k):
        man = self.inputs.manifest
        sm = man[f"s_mean_{self.codes[k]}"]
        sd = man[f"s_sd_{self.codes[k]}"]
        valid = sm.notna() & sd.notna()
        seg = man.index[valid].to_numpy(np.int64)
        s = sm[valid].to_numpy(float)
        ss = sd[valid].to_numpy(float)
        y = (s > 0.0).astype(np.int64)
        return TaskCandidates(k, seg, s, ss, y, np.ones(len(seg)),
                              (y == 1) == (s > 0.0))

    def task_pool(self, task):
        return self._pools[int(task)]

    def candidates(self, task, *, exclude_segids=None, feedback_safe=False,
                   min_margin=0.30):
        pool = self._pools[int(task)]
        mask = np.ones(len(pool), dtype=bool)
        if exclude_segids:
            mask &= ~np.isin(pool.seg_id, np.fromiter(
                (int(x) for x in exclude_segids), dtype=np.int64))
        if feedback_safe:
            mask &= pool.coherent & (pool.margin >= float(min_margin))
        return pool.subset(mask)


def lenient_exam_policy(inputs):
    """A fast AD6 policy so a full exam fits inside the 700-seg fixture bank
    (leaving room for training + retest). Production strictness is a bank-size /
    deployment choice, not part of the loop contract."""
    sys.path.insert(0, _SCRIPTS)
    from cortex_policy_k7 import load_ell_star_k7
    from cortex_policy import AD6Policy
    ell = load_ell_star_k7(inputs.task_codes)
    vp = list(np.diag(np.asarray(inputs.Corr_l, dtype=float)))
    return AD6Policy(ell, vp, n_min=3, R_star=0.05, alpha=0.30, Z=1.0)


def _run_exam(inputs, learner, *, seen, policy_factory, n_particles, seed,
              max_questions):
    if _SCRIPTS not in sys.path:
        sys.path.insert(0, _SCRIPTS)
    import session_controller as sc
    exam_inputs = inputs.without(seen) if seen else inputs
    theta, ell = plan_to_engine(learner.sigma, learner.t)     # engine coords
    sess = sc.CortexSession(exam_inputs, session_id="exam", n_particles=n_particles,
                            seed=seed, capture_clouds=True, max_questions=max_questions,
                            policy=policy_factory(exam_inputs))
    res = sess.run(sc.make_simulated_y_source(theta, ell, seed=seed))
    return res


def run_closed_loop(inputs, learner, *, ell_stars, sigma_stars, filter_sigma_inf,
                    ledger, participant_id="sim", policy_factory=lenient_exam_policy,
                    max_rounds=4, train_budget=150, inflate=1.5, n_particles=120,
                    seed=0, filter_params=None, exam_max_questions=120,
                    label_schedule=None, schedule_seed=None):
    """Run the full loop for one simulated learner. Returns a trace dict:
    per-round verdicts + weak set + n_exam/n_train, plus `converged`, the final
    learner engine state, and the ledger's total exposure."""
    K = len(inputs.task_codes)
    codes = list(inputs.task_codes)
    bank = SimBankAdapter(inputs)
    base_params = filter_params or LearnerParams(
        alpha_t=0.097, alpha_sigma=0.047, q_t=0.05, q_sigma=0.02, rho=0.5,
        rule="soft")
    ell_stars = list(ell_stars)
    sigma_stars = list(sigma_stars)
    now = 0.0
    rounds = []
    converged = False

    from trainer.policy import TrainerPolicy

    for r in range(max_rounds):
        seen = ledger.seen_segids(participant_id)
        res = _run_exam(inputs, learner, seen=seen, policy_factory=policy_factory,
                        n_particles=n_particles, seed=seed + r,
                        max_questions=exam_max_questions)
        for sid in res.served_seg_ids:
            dom = codes[inputs.true_task_index(int(sid))]
            ledger.record(participant_id, int(sid), dom, "test", now)
            now += 1.0
        verdicts = list(res.verdicts)
        weak = [k for k in range(K) if verdicts[k] != "PASS"]
        rounds.append({"round": r, "verdicts": verdicts, "weak": list(weak),
                       "n_exam": res.n_questions, "exam_stop": res.stop_reason})
        if not weak:
            converged = True
            break
        if r == max_rounds - 1:
            break                          # measured but no more training

        # build per-task filters from the exam posterior (variance-inflated, D1)
        th, el, w = res.t_traj[-1], res.l_traj[-1], res.w_traj[-1]
        filters = []
        for k in range(K):
            pk = replace(base_params, sigma_inf=float(filter_sigma_inf[k]))
            filters.append(TaskFilter(
                inflate_cloud(th[:, k], w, inflate),
                inflate_cloud(el[:, k], w, inflate), pk, w=w, seed=seed + 17 * k))

        seen = ledger.seen_segids(participant_id)
        # label_schedule passthrough (docs/LABEL_SCHEDULE_PECR.md); the
        # per-round schedule seed varies with r so training rounds don't
        # replay the same label sequence (None default = pinned legacy)
        pol = TrainerPolicy(filters, ell_stars, sigma_stars, bank,
                            seed=seed + 1000 + r, exclude_segids=seen,
                            label_schedule=label_schedule,
                            schedule_seed=(None if schedule_seed is None
                                           else schedule_seed + r))
        n_train = 0
        for _ in range(train_budget):
            choice = pol.step()
            if choice is None:
                break
            task, s, y_star = choice["task"], choice["s"], choice["y_star"]
            y = learner.step(s, task, y_star=y_star, feedback=True)
            pol.record(choice, int(y))
            ledger.record(participant_id, int(choice["seg_id"]), codes[task],
                          "train", now)
            now += 1.0
            n_train += 1
        rounds[-1]["n_train"] = n_train
        rounds[-1]["mastered_after_train"] = [
            k for k in weak if pol.mode_policies[k].is_mastered(filters[k])]

    theta_f, ell_f = plan_to_engine(learner.sigma, learner.t)
    return {"rounds": rounds, "converged": converged,
            "final_theta": theta_f, "final_ell": ell_f,
            "total_exposure": len(ledger.seen_segids(participant_id))}
