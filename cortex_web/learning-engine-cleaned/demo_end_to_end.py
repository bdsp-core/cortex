"""End-to-end demonstration of the learning engine on SYNTHETIC data.

Runs the full production sequence with no external data or services:

  1. generate a synthetic population (known ground truth);
  2. M1: fit the hierarchical population model (smoke-scale MCMC);
  3. build the population ARTIFACT (the single deployable object);
  4. build an item bank + a curated certification exam
     (curation rule: training must improve the exam score and the bar
     must sit inside the trainable window — see README);
  5. M2: run closed-loop training sessions on fresh learners with the
     production engine (exam-margin placement, double-eta readiness,
     logged-only plateau signals);
  6. exercise the adaptive-testing handoff ports with a mock testing
     result (docs/handoff_contract.md).

Smoke MCMC settings keep the run to a few minutes on a laptop-class
CPU; they validate the pipeline, not estimation power.

Run:  python demo_end_to_end.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpyro
numpyro.enable_x64()

from scipy.stats import binom

from learning_engine import (MSBelief, MSSessionEngine,
                             floor_joint_from_draws, make_multi_cohort,
                             prior_blocks_from_population, run_hier_multi)

ETA = 0.10
BUDGET = 300
N_TRAIN_LEARNERS = 16
M = 4
SEED = 7


class DemoLearner:
    """Ground-truth learner (Wilson-gated soft dynamics, D30 forms)."""

    def __init__(self, th, rng):
        self.t = np.array(th["t0"], float)
        self.u = np.array(th["u0"], float)
        self.u_inf = np.array(th["u_inf"], float)
        self.a_t = np.array(th["alpha_t"], float)
        self.a_s = np.array(th["alpha_s"], float)
        self.lam = float(th["lam"])
        self.rng = rng
        self.M = len(self.t)

    def _p(self, s_row):
        z = (np.asarray(s_row) - self.t) / np.exp(self.u)
        e = np.exp(z - z.max())
        return z, self.lam / self.M + (1 - self.lam) * e / e.sum()

    def exam_accuracy(self, exam):
        return float(np.mean([self._p(s)[1][g] for s, g in
                              zip(exam["s"], exam["gold"])]))

    def step(self, item):
        z, p = self._p(item["s"])
        y = int(self.rng.choice(self.M, p=p))
        ystar = np.zeros(self.M); ystar[item["gold"]] = 1.0
        w = np.abs(z) * np.exp(0.5 * (1.0 - z ** 2))
        self.t = self.t + self.a_t * (p - ystar)
        self.u = self.u - self.a_s * w * (self.u - self.u_inf)
        return y


def build_artifact(post):
    """Population posterior -> the deployable artifact (the production
    schema; see README section 'The artifact')."""
    Msig = post["t0"].shape[-1]
    return dict(
        alpha_t=post["alpha_t"].mean(0).tolist(),
        alpha_s=post["alpha_s"].mean(0).tolist(),
        lam=float(np.median(post["lam"].mean(0))),
        q_t=0.0, q_s=0.006,
        state_prior=dict(
            mu_t0=post["t0"].mean(0).mean(0).tolist(),
            tau_t0=post["t0"].mean(0).std(0).tolist(),
            mu_u0=post["u0"].mean(0).mean(0).tolist(),
            tau_u0=post["u0"].mean(0).std(0).tolist(),
            gamma=post["gamma"].mean(0).tolist(),
        ),
        floor_prior=(
            [float(post["u_inf"][:, :, m].ravel().mean())
             for m in range(Msig)],
            [float(post["u_inf"][:, :, m].ravel().std())
             for m in range(Msig)]),
        floor_joint=floor_joint_from_draws(post["u0"], post["u_inf"]),
    )


def make_bank(rng, n_items=1200):
    """Items in the cohort geometry: gold-channel evidence ambiguous,
    non-gold clearly negative (matches make_multi_cohort)."""
    gold = rng.integers(0, M, size=n_items)
    S = rng.normal(-1.3, 0.5, size=(n_items, M))
    S[np.arange(n_items), gold] = rng.normal(-0.1, 0.6, size=n_items)
    S = np.clip(S, -2.33, 2.33)
    return [dict(item_id=f"item-{j}", s=S[j], gold=int(gold[j]))
            for j in range(n_items)]


def build_exam(art, bank, signals, n_per=10):
    """Curated certification exam (the D45 dry-run requirement): per
    channel, n_per items evenly spaced through the top quartile of
    gold-channel evidence; bar = the score a fully-trained median
    learner clears 90% of the time. A RANDOM item mix can make the
    trained state score WORSE than the start state (see README)."""
    S = np.stack([b["s"] for b in bank])
    gold = np.array([b["gold"] for b in bank])
    gs = S[np.arange(len(bank)), gold]
    idx = []
    for m in range(len(signals)):
        cand = np.flatnonzero(gold == m)
        cand = cand[np.argsort(gs[cand])[::-1]]
        top = cand[:max(n_per, len(cand) // 4)]
        idx.extend(top[np.linspace(0, len(top) - 1, n_per)
                       .astype(int)].tolist())
    exam = dict(n_items=len(idx), s=[bank[j]["s"] for j in idx],
                gold=[int(bank[j]["gold"]) for j in idx])
    med = MSBelief(art, signals, N=1, rng=np.random.default_rng(1))
    med.t = np.asarray(art["state_prior"]["mu_t0"])[None, :]
    med.u = np.asarray(art["floor_prior"][0])[None, :]
    exam["pass_count"] = int(binom.ppf(
        0.10, exam["n_items"], float(med.exam_accuracy(exam)[0])))
    return exam


def main():
    t_all = time.time()
    signals = [f"signal-{m}" for m in range(M)]

    print("[1/6] generating synthetic population (known truth) ...",
          flush=True)
    coh = make_multi_cohort(L=16, K=220, M=M,
                            rng=np.random.default_rng(SEED), form="soft")

    print("[2/6] M1 population fit (smoke MCMC, a few minutes) ...",
          flush=True)
    t0 = time.time()
    mcmc, post = run_hier_multi(coh, warmup=120, samples=120, chains=2,
                                max_tree_depth=7, target_accept=0.9,
                                dense_pop=True, form="soft", use_rt=True,
                                fatigue=True, progress=False)
    div = int(np.sum(mcmc.get_extra_fields()["diverging"]))
    print(f"      fit done in {time.time()-t0:.0f}s, divergences={div}",
          flush=True)

    print("[3/6] building the deployable artifact ...", flush=True)
    art = build_artifact(post)

    print("[4/6] item bank + curated certification exam ...", flush=True)
    rng = np.random.default_rng(SEED + 1)
    bank = make_bank(rng)
    exam = build_exam(art, bank, signals)
    print(f"      exam: {exam['n_items']} items, bar "
          f"{exam['pass_count']}/{exam['n_items']}", flush=True)

    print("[5/6] closed-loop training on fresh learners ...", flush=True)
    fresh = make_multi_cohort(L=N_TRAIN_LEARNERS, K=1, M=M,
                              rng=np.random.default_rng(SEED + 2),
                              form="soft")
    eng = MSSessionEngine(art, signals, bank, exam, ETA, budget=BUDGET,
                          n_particles=400)
    flagged, used, passed = [], [], []
    signal_trace = []
    for j in range(N_TRAIN_LEARNERS):
        lr = DemoLearner(fresh["thetas"][j],
                         np.random.default_rng(1000 + j))

        def on_q(q, bel):
            if j == 0 and q % 25 == 0:   # logged-only plateau signals
                cur = bel.pred_score(exam)
                fd = bel.forecast_score(exam, BUDGET - q, eng.zstar)
                fm = bel.forecast_score_mc(exam, BUDGET - q, eng.zstar,
                                           n_rollouts=4, seed=q)
                signal_trace.append((q, cur, fd - cur, fm - cur))

        r = eng.run(lr.step, seed=2000 + j, on_question=on_q)
        fl = r["ready_at"] >= 0
        flagged.append(fl)
        if fl:
            used.append(r["n_used"])
            a = np.clip(lr.exam_accuracy(exam), 1e-9, 1 - 1e-9)
            score = np.random.default_rng(3000 + j).binomial(
                exam["n_items"], a)
            passed.append(bool(score >= exam["pass_count"]))
    pgr = float(np.mean(passed)) if passed else float("nan")
    print(f"      flagged {100*np.mean(flagged):.0f}%  "
          f"median questions-to-ready "
          f"{np.median(used) if used else float('nan'):.0f}  "
          f"pass-given-ready {pgr:.2f} (n={len(passed)})", flush=True)
    print("      (a partial ready fraction is the engine being honest: "
          "on a heterogeneous population only part of the cohort can "
          "truthfully certify at a fixed bar)", flush=True)
    print("      plateau signals (learner 0, logged-only): q, "
          "pred_score, det gain, mc gain")
    for q, cur, gd, gm in signal_trace[:5]:
        print(f"        q{q:>3}  {cur:.3f}  {gd:+.4f}  {gm:+.4f}")

    print("[6/6] adaptive-testing handoff ports (mock result) ...",
          flush=True)
    th0 = fresh["thetas"][0]
    mock_result = dict(
        dimension_names=signals,
        final_summary=dict(
            offset_mean=(-np.array(th0["t0"])).tolist(),
            offset_variance=(0.05 * np.ones(M)).tolist(),
            log_sensitivity_mean=(-np.array(th0["u0"])).tolist(),
            log_sensitivity_variance=(0.05 * np.ones(M)).tolist()))
    bel = MSBelief(art, signals, N=400, rng=np.random.default_rng(5))
    bel.seed_from_testing(mock_result)
    t_bar, u_bar = bel.mean_state()
    seed_err = float(np.max(np.abs(t_bar - np.array(th0["t0"]))))
    blocks = prior_blocks_from_population(art)
    print(f"      belief seeded from mock testing posterior "
          f"(|t err| {seed_err:.3f}); reverse port emits prior blocks: "
          f"{sorted(blocks)}", flush=True)

    # replay port (contract section 2a): the learner's raw no-feedback
    # pre-test stream seeds a session belief through the engine's own
    # observation model; replayed items are never re-served, and
    # posterior XOR replay seeding is enforced
    lr0 = DemoLearner(th0, np.random.default_rng(6))
    rr = np.random.default_rng(7)
    trials = []
    for q in range(40):
        j = int(rr.integers(len(bank)))
        _, p = lr0._p(bank[j]["s"])
        trials.append(dict(k=q, item_id=bank[j]["item_id"],
                           response=int(rr.choice(M, p=p))))
    r_rp = eng.run(lr0.step, seed=17, testing_trials=trials)
    diag = r_rp["belief"].seed_diag
    overlap = ({l["item_id"] for l in r_rp["log"]}
               & {tr["item_id"] for tr in trials})
    try:
        MSBelief(art, signals, N=100,
                 rng=np.random.default_rng(9)).seed_from_test_replay(
            [(bank[0]["s"], 0)]).seed_from_testing(mock_result)
        xor_guard = False
    except ValueError:
        xor_guard = True
    print(f"      replay port: {diag['n_trials']} pre-test trials "
          f"replayed (unique ancestors {diag['n_unique']}/"
          f"{r_rp['belief'].N}), session used {r_rp['n_used']} questions, "
          f"served-overlap {len(overlap)}", flush=True)

    checks = dict(
        fit_ran=div < 120,
        exam_bar_interior=0 < exam["pass_count"] < exam["n_items"],
        some_flagged=bool(used),
        honesty_within_noise=(not passed) or pgr >= 1 - ETA - 2 * np.sqrt(
            max(pgr * (1 - pgr), 1e-9) / max(len(passed), 1)),
        signals_finite=bool(signal_trace) and np.isfinite(
            np.array([[c, d, m_] for _, c, d, m_ in signal_trace])).all(),
        contract_seed_close=seed_err < 0.25,
        replay_port_ok=(diag["n_trials"] == 40 and diag["n_unique"] > 50
                        and not overlap),
        replay_xor_guard=xor_guard,
    )
    print(f"\nDEMO CHECKS: " + "  ".join(
        f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
    print(f"total {time.time()-t_all:.0f}s")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
