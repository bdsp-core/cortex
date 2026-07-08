"""Real-bank served-stream adversary measurement (PECR critique-3 F1/F2).

Runs matched-seed legacy vs randomized single-task sessions on the real
K=7 bank (the oc_campaign2 nearcut_soft harness), collects the weak task's
served stream, and measures every adversary family: label-history oracle,
own-EWMA / cluster (|s| channels), anti-repeat (the owner-observed live
exploit), and sd-cluster (the perceived-ambiguity channel — the
bank-content limitation recorded as the ledger's open owner item).
"""
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
from trainer.label_schedule import ScheduleParams, ScheduleRng, SideScheduler, p_side_up
from trainer.orchestrator import SimBankAdapter
from trainer.policy import TrainerPolicy
from trainer.sim import Learner
from trainer.trainability import SigmaInfMixtureFilter

ELL = np.array([0.2382694370243232, 0.1548131161373049, 0.3059231418723579,
                0.2569706571684701, 0.3213979957403151, 0.3539561460242777,
                0.3042300471351564])
SIG = np.exp(-ELL)
K, WEAK, L_TRUE = 7, 2, 0.25
P = dict(alpha_t=0.10, alpha_sigma=0.12, q_t=0.05, q_sigma=0.02, rho=0.5)
SP = ScheduleParams()


def run_stream(bank, seed, schedule, budget=240):
    sigma0 = np.exp(-np.full(K, 2.0))
    sigma0[WEAK] = float(np.exp(-L_TRUE))
    learner = Learner(sigma0, np.zeros(K),
                      LearnerParams(**{**P, "rule": "soft"}), seed=seed)
    fsi = np.exp(-(ELL - 0.10))
    filters = []
    for k in range(K):
        r = np.random.default_rng(50 + k)
        if k == WEAK:
            th, el = r.normal(0, 0.3, 300), r.normal(L_TRUE, 0.3, 300)
        else:
            th, el = r.normal(0, 0.2, 300), r.normal(2.0, 0.2, 300)
        pk = LearnerParams(**{**P, "rule": "soft", "sigma_inf": float(fsi[k])})
        filters.append(SigmaInfMixtureFilter(
            th, el, pk, ell_inf_mean=-np.log(pk.sigma_inf),
            ell_star=ELL[k], seed=seed + 7 * k))
    pol = TrainerPolicy(filters, list(ELL), list(SIG), bank, seed=seed + 99,
                        label_schedule=schedule)
    stream = []
    for _ in range(budget):
        ch = pol.step()
        if ch is None:
            break
        if ch["task"] == WEAK:
            stream.append((float(ch["s"]), float(ch["s_sd"]),
                           int(ch["y_star"])))
        y = learner.step(ch["s"], ch["task"], y_star=ch["y_star"],
                         feedback=True)
        pol.record(ch, int(y))
    return stream


def oracle(stream):
    sch = SideScheduler(SP, ScheduleRng(0, 0))
    hits = 0
    for _s, _sd, y in stream:
        pu = p_side_up(sch.d, sch.run, SP)
        hits += int((1 if pu >= 0.5 else 0) == y)
        sch.commit(1 if y == 1 else -1)
    return hits / max(len(stream), 1)


def feature_cluster(stream, feat):
    sums = {1: 0.0, 0: 0.0}
    cnts = {1: 0, 0: 0}
    hits = n = 0
    for s, sd, y in stream:
        x = abs(s) if feat == "mag" else sd
        if cnts[1] > 2 and cnts[0] > 2:
            c1, c0 = sums[1] / cnts[1], sums[0] / cnts[0]
            if abs(c1 - c0) > 1e-9:
                hits += int((1 if abs(x - c1) < abs(x - c0) else 0) == y)
                n += 1
        sums[y] += x
        cnts[y] += 1
    return hits / n if n else 0.5


def anti_repeat(stream):
    ys = [y for _s, _sd, y in stream]
    if len(ys) < 2:
        return 0.5
    return float(np.mean([ys[i] != ys[i - 1] for i in range(1, len(ys))]))


def main():
    bank = SimBankAdapter(build_k7_engine_inputs())
    out = {}
    for arm in (None, "randomized"):
        pooled = {"oracle": 0.0, "cluster_mag": 0.0, "sd_cluster": 0.0,
                  "anti_repeat": 0.0}
        n = 0
        for seed in range(8):
            st = run_stream(bank, seed, arm)
            m = len(st)
            pooled["oracle"] += oracle(st) * m
            pooled["cluster_mag"] += feature_cluster(st, "mag") * m
            pooled["sd_cluster"] += feature_cluster(st, "sd") * m
            pooled["anti_repeat"] += anti_repeat(st) * m
            n += m
        out[arm or "legacy"] = {k: round(v / n, 4) for k, v in pooled.items()}
        out[arm or "legacy"]["n"] = n
        print(arm or "legacy", out[arm or "legacy"], flush=True)
    with open(os.path.join(REPO, "results", "label_schedule_oc",
                           "realbank_adversaries.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("written")


if __name__ == "__main__":
    main()
