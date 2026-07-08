"""G-C OC non-regression campaign: legacy vs randomized label schedule on the
real K=7 bank (mirrors tests/test_trainer_g2_oc.py setup), matched seeds.
Richer telemetry than trainer/oc.py: lateness, FG vs true state, terminal
|t_true|, mode mix, adversary metrics on the served weak-task stream."""
import json
import os
import sys

import numpy as np

REPO = "/data/eli-work/repos/ilae-skill-certification-test-multi"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, os.path.join(REPO, "tests"))

from cortex_engine_inputs_k7 import build_k7_engine_inputs
from trainer.dynamics import LearnerParams
from trainer.filter import TaskFilter
from trainer.label_schedule import ScheduleParams, ScheduleRng, SideScheduler, p_side_up
from trainer.orchestrator import SimBankAdapter
from trainer.policy import TrainerPolicy
from trainer.sim import Learner
from trainer.trainability import SigmaInfMixtureFilter

ELL = np.array([0.2382694370243232, 0.1548131161373049, 0.3059231418723579,
                0.2569706571684701, 0.3213979957403151, 0.3539561460242777,
                0.3042300471351564])
SIG = np.exp(-ELL)
K = 7
P = dict(alpha_t=0.10, alpha_sigma=0.12, q_t=0.05, q_sigma=0.02, rho=0.5)


def run_session(bank, *, weak_task, l_true, rule, filter_sigma_inf,
                use_mixture, budget, seed, schedule, n_particles=300):
    sigma0 = np.exp(-np.full(K, 2.0))
    sigma0[weak_task] = float(np.exp(-l_true))
    learner = Learner(sigma0, np.zeros(K), LearnerParams(**{**P, "rule": rule}),
                      seed=seed)
    filters = []
    for k in range(K):
        r = np.random.default_rng(50 + k)
        if k == weak_task:
            th, el = r.normal(0, 0.3, n_particles), r.normal(l_true, 0.3, n_particles)
        else:
            th, el = r.normal(0, 0.2, n_particles), r.normal(2.0, 0.2, n_particles)
        pk = LearnerParams(**{**P, "rule": "soft",
                              "sigma_inf": float(filter_sigma_inf[k])})
        if use_mixture:
            filters.append(SigmaInfMixtureFilter(
                th, el, pk, ell_inf_mean=-np.log(pk.sigma_inf),
                ell_star=ELL[k], seed=seed + 7 * k))
        else:
            filters.append(TaskFilter(th, el, pk, seed=seed + 7 * k))
    pol = TrainerPolicy(filters, list(ELL), list(SIG), bank, seed=seed + 99,
                        label_schedule=schedule)
    vals, first_decl, decl_ltrue, stream, modes = [], None, None, [], []
    for i in range(budget):
        ch = pol.step()
        if ch is None:
            break
        if ch["task"] == weak_task:
            vals.append(learner.skill_weight(ch["s"], weak_task))
            stream.append(int(ch["y_star"]))
            modes.append(ch["mode"])
        y = learner.step(ch["s"], ch["task"], y_star=ch["y_star"], feedback=True)
        pol.record(ch, int(y))
        if (first_decl is None
                and pol.mode_policies[weak_task].is_mastered(filters[weak_task])):
            first_decl = i
            decl_ltrue = float(-np.log(learner.sigma[weak_task]))
    return {
        "delivered_value": float(np.mean(vals)) if vals else 0.0,
        "first_declared": first_decl,
        "decl_ltrue": decl_ltrue,
        "declared_final": bool(pol.mode_policies[weak_task].is_mastered(
            filters[weak_task])),
        "terminal_abs_t_true": float(abs(learner.t[weak_task])),
        "terminal_ltrue": float(-np.log(learner.sigma[weak_task])),
        "weak_trials": len(stream),
        "bias_frac": float(np.mean([m == "bias" for m in modes])) if modes else 0.0,
        "stream": stream,
    }


def oracle_acc(stream, params):
    sch = SideScheduler(params, ScheduleRng(0, 0))
    hits = 0
    for y in stream:
        pu = p_side_up(sch.d, sch.run, params)
        hits += int((1 if pu >= 0.5 else 0) == y)
        sch.commit(1 if y == 1 else -1)
    return hits / max(len(stream), 1)


def anti_repeat(stream):
    if len(stream) < 2:
        return 0.5
    return float(np.mean([stream[i] != stream[i - 1]
                          for i in range(1, len(stream))]))


def main():
    bank = SimBankAdapter(build_k7_engine_inputs())
    fsi = np.exp(-(ELL - 0.10))
    scenarios = [
        ("nearcut_soft", dict(weak_task=2, l_true=0.25, rule="soft",
                           use_mixture=True, budget=240), range(10)),
        ("sharp_static", dict(weak_task=2, l_true=0.25, rule="static",
                          use_mixture=True, budget=240), range(15)),
    ]
    p = ScheduleParams()
    rows = []
    for name, cfg, seeds in scenarios:
        for arm in (None, "randomized"):
            for s in seeds:
                r = run_session(bank, filter_sigma_inf=fsi, seed=s,
                                schedule=arm, **cfg)
                r.update(scenario=name, arm=arm or "legacy", seed=s,
                         oracle=oracle_acc(r.pop("stream"), p) if arm
                         else None)
                rows.append(r)
                print(f"{name:13s} {r['arm']:10s} seed={s:2d} "
                      f"dv={r['delivered_value']:.3f} "
                      f"decl@{r['first_declared']} "
                      f"lt_decl={r['decl_ltrue']} "
                      f"|t|end={r['terminal_abs_t_true']:.3f} "
                      f"bias%={r['bias_frac']:.2f}", flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "oc_campaign2_results.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote", out)

    # aggregate
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["scenario"], r["arm"])].append(r)
    print("\n=== AGGREGATE (legacy vs randomized, matched seeds) ===")
    for (name, arm), rs in sorted(agg.items()):
        dv = np.mean([r["delivered_value"] for r in rs])
        decl = [r["first_declared"] for r in rs if r["first_declared"] is not None]
        fg = np.mean([r["first_declared"] is not None for r in rs])
        tt = np.mean([r["terminal_abs_t_true"] for r in rs])
        lat = np.median(decl) if decl else float("nan")
        print(f"{name:13s} {arm:10s} dv={dv:.3f} declared={fg:.2f} "
              f"med_lateness={lat} |t|end={tt:.3f} n={len(rs)}")


if __name__ == "__main__":
    main()
