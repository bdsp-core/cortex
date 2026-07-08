"""G-C closed-loop matched pair: run_closed_loop legacy vs randomized on the
real exam engine (mirrors tests/test_trainer_g2_closed_loop.py setup)."""
import json
import os
import sys

import numpy as np

REPO = "/data/eli-work/repos/ilae-skill-certification-test-multi"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from cortex_engine_inputs_k7 import build_k7_engine_inputs
from trainer.dynamics import LearnerParams
from trainer.exposure import InMemoryExposureLedger
from trainer.orchestrator import run_closed_loop
from trainer.sim import Learner

ELL = np.array([0.2382694370243232, 0.1548131161373049, 0.3059231418723579,
                0.2569706571684701, 0.3213979957403151, 0.3539561460242777,
                0.3042300471351564])

inp = build_k7_engine_inputs()
K = len(inp.task_codes)
out = []
for seed in (3, 5, 11, 17, 23, 31, 41, 47, 53, 59, 61, 67, 71, 73, 79, 83):
    for arm in (None, "randomized"):
        lp = LearnerParams(alpha_t=0.10, alpha_sigma=0.15, sigma_inf=0.40,
                           q_t=0.05, q_sigma=0.02, rho=0.5, rule="soft")
        learner = Learner(np.full(K, 1.0), np.zeros(K), lp, seed=seed)
        led = InMemoryExposureLedger()
        tr = run_closed_loop(inp, learner, ell_stars=ELL,
                             sigma_stars=np.exp(-ELL),
                             filter_sigma_inf=np.exp(-(ELL + 0.3)),
                             ledger=led, participant_id=f"L{seed}{arm}",
                             max_rounds=3, train_budget=120, n_particles=100,
                             seed=seed + 2, exam_max_questions=110,
                             label_schedule=arm,
                             schedule_seed=None if arm is None else seed)
        final_skill = float((-np.log(learner.sigma)).mean())
        final_abs_t = float(np.abs(learner.t).mean())
        n_pass_final = sum(v == "PASS" for v in tr["rounds"][-1]["verdicts"])
        rec = dict(seed=seed, arm=arm or "legacy", converged=tr["converged"],
                   rounds=len(tr["rounds"]), n_pass_final=n_pass_final,
                   final_verdicts=list(tr["rounds"][-1]["verdicts"]),
                   ell_true=[float(x) for x in -np.log(learner.sigma)],
                   t_true=[float(x) for x in learner.t],
                   final_mean_ell_true=final_skill,
                   final_mean_abs_t_true=final_abs_t,
                   n_train_total=sum(r.get("n_train", 0) for r in tr["rounds"]))
        out.append(rec)
        print(rec, flush=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "closed_loop_pair16_r3.json"), "w") as f:
    json.dump(out, f, indent=1)
print("done")
