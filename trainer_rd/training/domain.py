"""M27 (F93) — the DOMAIN interface: the formal pluggability seam.

The trainer stack has always been domain-agnostic in practice — every
study swaps the bank behind a thin wrapper, and the policy/filter layers
consume only (candidate arrays, cut-scores, dynamics priors). This module
NAMES that seam so a new assessment domain plugs in without touching any
engine/training code:

    ItemBank        what a domain's item supply must provide
                    (`candidates()` → TaskCandidates; the contract the
                    real BankAdapter and every study wrapper already
                    satisfy — enforced by tests/test_m27).
    ArrayBank       a ready-made ItemBank over in-memory arrays: give it
                    per-task (seg_id, s_mean, s_sd, y_star, margin
                    [, coherent]) and it implements the exclusion /
                    feedback-safety / margin filters with BankAdapter's
                    exact semantics.
    Domain          the full bundle a deployment needs: task names, cut
                    scores ℓ*/σ* (the mastery bars), assumed learner
                    dynamics (LearnerParams factory), and an ItemBank.
                    `fresh_filters()` builds the per-task σ∞-mixture
                    beliefs; `trainer()` wires a TrainerPolicy. One
                    object = one pluggable domain.
    v15_domain()    the shipped instrument as a Domain (BankAdapter +
                    v15 cuts + D29 anchored rates) — the reference
                    construction and the port template.

Design rule (kept deliberately): the stimulus/rendering layer is NOT
part of Domain — it is a delivery-vehicle concern (sandbox/stimulus.py
owns the F73 render contract); headless consumers (studies, pipeline,
port) never need it.

Everything here is additive — no existing call site changed; the suite
pins behavioral identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Sequence

import numpy as np

from training.bank_adapter import TaskCandidates
from training.learner_sim import LearnerParams


class ItemBank:
    """Abstract item supply for K tasks (duck-typed contract; subclassing
    is optional — BankAdapter satisfies it structurally).

    Required method:
        candidates(task, *, exclude_segids=None, feedback_safe=False,
                   min_margin=0.30) -> TaskCandidates
    Returned pools must carry parallel arrays seg_id / s_mean / s_sd /
    y_star / margin (+ coherent) and support __len__ / subset(mask) —
    exactly `training.bank_adapter.TaskCandidates`.
    """

    def candidates(self, task, *, exclude_segids=None, feedback_safe=False,
                   min_margin=0.30) -> TaskCandidates:
        raise NotImplementedError


class ArrayBank(ItemBank):
    """ItemBank over per-task in-memory arrays — the minimal plug-in path
    for a new domain (BankAdapter filter semantics reproduced exactly)."""

    def __init__(self, pools: Dict[int, dict]):
        """pools[task] = dict(seg_id, s_mean, s_sd, y_star, margin
        [, coherent]); coherent defaults to sign-consistency of the label
        with the signal (the D11 definition)."""
        self._pools = {}
        for task, p in pools.items():
            s_mean = np.asarray(p["s_mean"], dtype=np.float64)
            y_star = np.asarray(p["y_star"], dtype=np.int64)
            coherent = np.asarray(
                p.get("coherent", (s_mean > 0) == (y_star == 1)), dtype=bool)
            self._pools[int(task)] = TaskCandidates(
                int(task),
                np.asarray(p["seg_id"], dtype=np.int64),
                s_mean,
                np.asarray(p["s_sd"], dtype=np.float64),
                y_star,
                np.asarray(p["margin"], dtype=np.float64),
                coherent)

    def candidates(self, task, *, exclude_segids=None, feedback_safe=False,
                   min_margin=0.30) -> TaskCandidates:
        pool = self._pools[int(task)]
        mask = np.ones(len(pool), dtype=bool)
        if exclude_segids:
            excl = np.fromiter((int(x) for x in exclude_segids),
                               dtype=np.int64)
            mask &= ~np.isin(pool.seg_id, excl)
        if feedback_safe:
            mask &= pool.coherent & (pool.margin >= float(min_margin))
        return pool.subset(mask)


@dataclass
class Domain:
    """Everything a deployment must supply to train toward ITS bars."""
    name: str
    tasks: Sequence[int]                       # bank task indices served
    task_names: Dict[int, str]
    ell_star: Dict[int, float]                 # mastery cut per task (ℓ)
    sigma_star: Dict[int, float]               # exp(−ℓ*) per task
    assumed_params: Callable[[int], LearnerParams]   # filter-side dynamics
    bank: ItemBank
    # belief-stack defaults (the D22/M21 shipped values; override per port)
    n_particles: int = 300
    mix_j: int = 7
    mix_tau: float = 0.30
    p_static_stratum: float = 0.15
    prior_sd: float = 0.45
    ell_inf_mean: Dict[int, float] = field(default_factory=dict)

    def fresh_filters(self, seed=0):
        """Per-task σ∞-mixture beliefs (the D22 stack) for a fresh learner."""
        from training.mixture_filter import SigmaInfMixtureFilter
        rng = np.random.default_rng(seed)
        out = []
        for i, t in enumerate(self.tasks):
            th0 = self.prior_sd * rng.standard_normal(self.n_particles)
            el0 = self.prior_sd * rng.standard_normal(self.n_particles)
            center = self.ell_inf_mean.get(
                t, -float(np.log(self.assumed_params(t).sigma_inf)))
            out.append(SigmaInfMixtureFilter(
                th0, el0, self.assumed_params(t), ell_inf_mean=center,
                tau=self.mix_tau, J=self.mix_j,
                p_static_stratum=self.p_static_stratum,
                ell_star=self.ell_star[t], seed=seed + 31 * (i + 1)))
        return out

    def trainer(self, filters=None, *, seed=0, **policy_kwargs):
        """A TrainerPolicy wired for this domain (kwargs pass through —
        probes, gates, allocation flags, terminal confirmation, …)."""
        from training.trainer_policy import TrainerPolicy

        class _View:
            def __init__(self, bank, tasks):
                self.bank, self.tasks = bank, tasks

            def candidates(self, task, **kw):
                return self.bank.candidates(self.tasks[task], **kw)

        filters = filters if filters is not None else self.fresh_filters(seed)
        return TrainerPolicy(
            filters, [self.ell_star[t] for t in self.tasks],
            [self.sigma_star[t] for t in self.tasks],
            _View(self.bank, list(self.tasks)), seed=seed + 5,
            **policy_kwargs)


def v15_domain(tasks=(1, 2)) -> Domain:
    """The shipped instrument as a Domain: real 89k bank, v15 credentialed
    cuts (D25), D29 anchored dynamics, D10 expert ceilings."""
    from engine.instrument_v15 import instrument
    from training.bank_adapter import (ANCHORED_ALPHA_SIGMA,
                                       ANCHORED_ALPHA_T, BankAdapter,
                                       ELL_STAR_V15, SIGMA_STAR_V15,
                                       TASK_CODES)
    ins = instrument("v15")

    def assumed(task):
        return LearnerParams(alpha_t=ANCHORED_ALPHA_T,
                             alpha_sigma=ANCHORED_ALPHA_SIGMA,
                             sigma_inf=float(ins.sigma_inf[task]),
                             q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")

    return Domain(
        name="v15",
        tasks=tuple(int(t) for t in tasks),
        task_names={t: TASK_CODES[t] for t in range(7)},
        ell_star={t: float(ELL_STAR_V15[t]) for t in range(7)},
        sigma_star={t: float(SIGMA_STAR_V15[t]) for t in range(7)},
        assumed_params=assumed,
        bank=BankAdapter(),
        ell_inf_mean={t: float(ins.expert_ell[t]) for t in range(7)})
