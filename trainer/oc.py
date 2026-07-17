"""Operating-characteristic measurement harness (G2). Trainer-only OCs — the
exam-side backstop / independence OCs live in the G2 test (they need the exam
engine). See docs/G2_CLOSEOUT.md for the measured results.
"""
from __future__ import annotations

import numpy as np

from trainer.dynamics import LearnerParams
from trainer.filter import TaskFilter
from trainer.policy import TrainerPolicy
from trainer.sim import Learner
from trainer.trainability import SigmaInfMixtureFilter


def single_task_train_session(bank, ell_stars, sigma_stars, *, weak_task, l_true,
                              learner_params, filter_sigma_inf, use_mixture,
                              budget=200, seed=0, n_particles=300,
                              label_schedule=None, schedule_seed=None):
    """Train a simulated learner that is weak on ONE task (the others pre-mastered
    so the scheduler focuses on it). Returns per-session OC metrics:

      delivered_value : mean learner skill-weight over the weak-task trials
                        (1.0 = ideal placement) — the "per-trial delivered value" OC.
      declarations    : # trials the trainer's gate read the weak task as mastered.
      trainability    : final P(ceiling clears the cut | data) (mixture only) —
                        the D-INT-3 plateau signal.
    """
    K = len(ell_stars)
    learner = Learner(_weak_state(K, weak_task, l_true), np.zeros(K),
                      LearnerParams(**learner_params), seed=seed)
    filters = []
    for k in range(K):
        r = np.random.default_rng(50 + k)
        if k == weak_task:
            th, el = r.normal(0, 0.3, n_particles), r.normal(l_true, 0.3, n_particles)
        else:                                   # pre-mastered: cloud well above cut
            th, el = r.normal(0, 0.2, n_particles), r.normal(2.0, 0.2, n_particles)
        pk = LearnerParams(**{**learner_params,
                              "sigma_inf": float(filter_sigma_inf[k])})
        if use_mixture:
            filters.append(SigmaInfMixtureFilter(
                th, el, pk, ell_inf_mean=-np.log(pk.sigma_inf),
                ell_star=ell_stars[k], seed=seed + 7 * k))
        else:
            filters.append(TaskFilter(th, el, pk, seed=seed + 7 * k))
    pol = TrainerPolicy(filters, list(ell_stars), list(sigma_stars), bank,
                        seed=seed + 99, label_schedule=label_schedule,
                        schedule_seed=schedule_seed)
    vals, declared = [], 0
    for _ in range(budget):
        ch = pol.step()
        if ch is None:
            break
        if ch["task"] == weak_task:
            vals.append(learner.skill_weight(ch["s"], weak_task))
        y = learner.step(ch["s"], ch["task"], y_star=ch["y_star"], feedback=True)
        pol.record(ch, int(y))
        if pol.mode_policies[weak_task].is_mastered(filters[weak_task]):
            declared += 1
    tr = (filters[weak_task].trainability(ell_stars[weak_task])
          if use_mixture else None)
    return {"delivered_value": float(np.mean(vals)) if vals else 0.0,
            "declarations": declared, "trainability": tr}


def _weak_state(K, weak_task, l_true):
    sigma0 = np.exp(-np.full(K, 2.0))
    sigma0[weak_task] = float(np.exp(-l_true))
    return sigma0
