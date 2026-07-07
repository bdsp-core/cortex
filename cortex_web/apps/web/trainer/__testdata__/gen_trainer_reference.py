"""Dump ground-truth values from the REAL Python trainer (trainer/*.py) for the
TS trainer port to match (G3). Mirrors engine/__testdata__/gen_reference.py.

    python cortex_web/apps/web/trainer/__testdata__/gen_trainer_reference.py

Writes trainer_reference.json next to this file. The vitest specs
(conventions.test.ts, filter.test.ts) load it and assert the TS trainer agrees
within tolerance. Only DETERMINISTIC quantities are dumped — the RNG-driven
propagate/resample are validated statistically (the exam engine's rng.ts contract:
TS uses xoshiro256**, not NumPy PCG64, so streams are not byte-matched)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                     *([os.pardir] * 5)))
sys.path.insert(0, _REPO)

from trainer import conventions as C          # noqa: E402
from trainer.dynamics import LearnerParams     # noqa: E402
from trainer.filter import TaskFilter          # noqa: E402
from trainer.trainability import SigmaInfMixtureFilter  # noqa: E402

_PARAMS = {"alphaT": 0.097, "alphaSigma": 0.047, "sigmaInf": 0.6, "qT": 0.05,
           "qSigma": 0.02, "rho": 0.5, "rule": "soft"}

_OUT = os.path.join(os.path.dirname(__file__), "trainer_reference.json")


def gen_conventions():
    theta = [-0.7, -0.2, 0.0, 0.5, 1.3]
    ell = [-0.4, 0.0, 0.3251, 0.6, 1.1]
    sigma = [0.3, 0.6, 0.7879903487138527, 1.0, 1.6]
    t = [-0.9, -0.1, 0.0, 0.4, 1.2]
    s = [-1.2, -0.3, 0.0, 0.7, 1.5]
    accs = [0.60, 0.7, 0.8413447460685429, 0.90]
    ms = [0.5, 1.0, 1.0770, 1.4]
    ell_star = [0.2382694370243232, 0.3, 0.5337430687087749]
    sig2p, t2p = C.engine_to_plan(theta, ell)
    th2e, el2e = C.plan_to_engine(sigma, t)
    return {
        "LAPSE_RATE": C.LAPSE_RATE,
        "WILSON_OPT_ACC": C.WILSON_OPT_ACC,
        "SKILL_MODE_MULTIPLIER": C.SKILL_MODE_MULTIPLIER,
        "in": {"theta": theta, "ell": ell, "sigma": sigma, "t": t, "s": s,
               "accs": accs, "ms": ms, "ell_star": ell_star},
        "engineToPlan": {"sigma": sig2p.tolist(), "t": t2p.tolist()},
        "planToEngine": {"theta": th2e.tolist(), "ell": el2e.tolist()},
        "pYesPlan": C.p_yes_plan(s, sigma, t).tolist(),
        "pYesEngine": C.p_yes_engine(s, theta, ell).tolist(),
        "difficultyMultiplier": [C.difficulty_multiplier(a) for a in accs],
        "accuracyAtMultiplier": [C.accuracy_at_multiplier(m) for m in ms],
        "aurocFromEll": C.auroc_from_ell(ell).tolist(),
        "aurocFromSigma": C.auroc_from_sigma(sigma).tolist(),
        "sigmaStarFromEllStar": C.sigma_star_from_ell_star(ell_star).tolist(),
    }


def gen_filter():
    rng = np.random.default_rng(11)
    N = 32
    theta0 = rng.normal(0.0, 0.4, N)
    ell0 = rng.normal(0.3, 0.3, N)
    w0 = rng.random(N)
    w0 = w0 / w0.sum()
    p = LearnerParams(alpha_t=0.097, alpha_sigma=0.047, sigma_inf=0.6, rule="soft")
    cases = []
    for s, y, s_sd in [(0.8, 1, 0.0), (-0.5, 0, 0.6), (1.2, 1, 0.85)]:
        f = TaskFilter(theta0, ell0, p, w=w0, seed=0)
        tot = f.reweight(s, y, s_sd)
        cases.append({"s": s, "y": y, "sSd": s_sd, "tot": tot,
                      "wPost": f.w.tolist()})
    f = TaskFilter(theta0, ell0, p, w=w0, seed=0)
    mt, ml = f.mean()
    sdt, sdl = f.sd()
    pi, mcse = f.pass_mass(0.35)

    # DETERMINISTIC propagate drift: simple branch (s_sd=0), noise off (q=0), soft
    # rule → the R–W criterion drift + skill relaxation with zero process noise.
    pd = LearnerParams(alpha_t=0.10, alpha_sigma=0.10, sigma_inf=0.5,
                       q_t=0.0, q_sigma=0.0, rho=0.5, rule="soft")
    fd = TaskFilter(theta0, ell0, pd, w=w0, seed=0)
    fd.propagate(0.7, y_star=1, feedback=True, y=1, s_sd=0.0)
    drift = {"s": 0.7, "yStar": 1, "y": 1,
             "params": {"alphaT": 0.10, "alphaSigma": 0.10, "sigmaInf": 0.5,
                        "qT": 0.0, "qSigma": 0.0, "rho": 0.5, "rule": "soft"},
             "thetaPost": fd.theta.tolist(), "ellPost": fd.ell.tolist()}

    # systematic-resample indices for a fixed stratified offset u0.
    cumw = np.cumsum(w0)
    u0 = 0.37
    u = (u0 + np.arange(N)) / N
    idx = np.minimum(np.searchsorted(cumw, u), N - 1)

    return {
        "theta0": theta0.tolist(), "ell0": ell0.tolist(), "w0": w0.tolist(),
        "reweight": cases,
        "summary": {"meanT": mt, "meanL": ml, "sdT": sdt, "sdL": sdl,
                    "ellStar": 0.35, "pi": pi, "mcse": mcse,
                    "sdFloor": 0.5,
                    "isMastered": f.is_mastered(0.35, sd_floor=0.5)},
        "propagateDrift": drift,
        "resample": {"cumw": cumw.tolist(), "u0": u0, "idx": idx.tolist()},
    }


def gen_trainability():
    rng = np.random.default_rng(13)
    N = 24
    theta0 = rng.normal(0.0, 0.35, N)
    ell0 = rng.normal(0.4, 0.3, N)
    w0 = rng.random(N)
    w0 = w0 / w0.sum()
    p = LearnerParams(alpha_t=0.097, alpha_sigma=0.047, sigma_inf=0.6, rule="soft")
    mf = SigmaInfMixtureFilter(theta0, ell0, p, ell_inf_mean=0.6, tau=0.30, J=7,
                               ell_star=0.35, seed=3, w=w0, exact_kernel=True)
    pi, mcse = mf.pass_mass(0.35)
    mt, ml = mf.mean()
    sdt, sdl = mf.sd()
    return {
        "theta0": theta0.tolist(), "ell0": ell0.tolist(), "w0": w0.tolist(),
        "params": _PARAMS, "ellInfMean": 0.6, "tau": 0.30, "J": 7,
        "ellStar": 0.35, "seed": 3,
        "grid": mf.grid.tolist(), "weights": mf.weights().tolist(),
        "trainability": mf.trainability(0.35),
        "summary": {"meanT": mt, "meanL": ml, "sdT": sdt, "sdL": sdl,
                    "pi": pi, "mcse": mcse},
    }


def gen_policy():
    from trainer.policy import (
        expected_skill_weight, bias_probe_score, cert_probe_score,
        at_bar_accuracy, bias_stationary_sd, derived_t_star,
        bias_correction_score, TaskModePolicy, ModeThresholds)
    from trainer.bank_adapter import TaskCandidates

    rng = np.random.default_rng(21)
    N = 40
    theta = rng.normal(0.1, 0.3, N)
    ell = rng.normal(0.2, 0.3, N)
    w = rng.random(N)
    w = w / w.sum()
    p = LearnerParams(alpha_t=0.10, alpha_sigma=0.10, sigma_inf=0.5, rule="soft")
    filt = TaskFilter(theta, ell, p, w=w, seed=0)
    crng = np.random.default_rng(22)
    M = 30
    seg = np.arange(M)
    s_mean = crng.uniform(-1.5, 1.5, M)
    s_sd = crng.uniform(0.2, 0.9, M)
    y = (s_mean > 0).astype(np.int64)
    margin = np.ones(M)
    coh = (y == 1) == (s_mean > 0)
    cands = TaskCandidates(0, seg, s_mean, s_sd, y, margin, coh)
    sig_hat = float(np.exp(-(w * ell).sum()))
    t_hat = float(-(w * theta).sum())
    ell_star = 0.30
    sigma_star = float(np.exp(-ell_star))

    # deterministic decision sequence: fresh mode-policy, fixed cloud → the mode +
    # chosen item evolve only via the policy's own state (mirror pairing etc.).
    th = ModeThresholds()
    mp = TaskModePolicy(0, ell_star, sigma_star, th)
    from trainer.policy import RetentionScheduler
    ret = RetentionScheduler()
    seq = []
    for i in range(6):
        mode, est = mp.choose_mode(filt, ret, i)
        idx, info = mp.select(mode, est, cands, ret, i, np.random.default_rng(9),
                              filt=filt)
        mp.note_served(mode, int(cands.y_star[idx]))
        seq.append({"mode": mode, "idx": int(idx)})

    return {
        "theta": theta.tolist(), "ell": ell.tolist(), "w": w.tolist(),
        "params": {"alphaT": 0.10, "alphaSigma": 0.10, "sigmaInf": 0.5,
                   "qT": 0.05, "qSigma": 0.02, "rho": 0.5, "rule": "soft"},
        "cand": {"seg": seg.tolist(), "sMean": s_mean.tolist(),
                 "sSd": s_sd.tolist(), "yStar": y.tolist(),
                 "margin": margin.tolist(), "coherent": coh.tolist()},
        "sigHat": sig_hat, "tHat": t_hat, "ellStar": ell_star,
        "sigmaStar": sigma_star,
        "expectedSkillWeight": expected_skill_weight(
            s_mean, s_sd, sig_hat, t_hat, side=1, rho=0.5).tolist(),
        "biasProbeScore": bias_probe_score(s_mean, s_sd, sig_hat, t_hat).tolist(),
        "certProbeScore": cert_probe_score(s_mean, s_sd, sigma_star, t_hat).tolist(),
        "atBarAccuracy": [at_bar_accuracy(float(s_mean[i]), float(s_sd[i]),
                                          sigma_star, t_hat, int(y[i]))
                          for i in range(M)],
        "biasStationarySd": bias_stationary_sd(0.10, 0.05, sig_hat),
        "derivedTStar": derived_t_star(0.10, 0.05, sig_hat),
        "biasCorrectionScore": bias_correction_score(filt, cands).tolist(),
        "decisionSeq": seq,
    }


def main():
    ref = {"conventions": gen_conventions(), "filter": gen_filter(),
           "trainability": gen_trainability(), "policy": gen_policy()}
    with open(_OUT, "w") as fh:
        json.dump(ref, fh, indent=2)
    print("wrote", _OUT)


if __name__ == "__main__":
    main()
