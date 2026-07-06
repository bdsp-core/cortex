"""M16 — Phase-3 dynamics-fitting validation with train/test splits.

The plan's Phase 3 ("fit a hierarchical model of how cases change (σ,t)")
deferred since F4/D7. This study builds the fitter (training/dynamics_fit.py)
and validates it END-TO-END on simulated cohorts with honest splits:

  cohort   heterogeneous learners: log α_t ~ N(log .32, .35²),
           log α_σ ~ N(log .075, .35²), ℓ_∞ ~ N(.80, .20²); BIASED SUB-SKILL
           starts σ0 ~ LogN(log 1.4, .2²), |t0| ~ N(.65, .15²) random sign —
           benchmark-like, so the logs contain a real bias-correction phase
           (α_t identification) and a learning curve spanning ~100 trials
           (α_σ/ℓ_∞ identification); q/ρ shared (fitter holds them fixed —
           well-specified axis; misspec robustness was M12's rung).
           JUSTIFICATION: population means sit 1.25–1.6× ABOVE the trainer's
           D7 defaults (α_t .2, α_σ .06, ℓ_∞ .733) — the D14/F17 world where
           defaults deliberately lean conservative-low; spreads (±42% 1σ)
           are the plausible-heterogeneity band inside F28's ±4× stress
           bound; ℓ_∞ SD .20 sits inside D22's τ=.30 ceiling prior.
           SMOKE-RUN LESSONS BAKED IN (M16): belief-prior starts made the
           task trivially easy (mastery ≈ 8 trials, α_t unidentified — fits
           returned the prior exactly); one-step held-out evidence needs
           multi-seed averaging because resample-branch MC noise is the same
           order as the per-trial signal.
  logging  the trainer that COLLECTS the data runs tier-2 with the DEFAULT
           params (you do not know the dynamics when you run the pilot) and
           the exact kernel; 300 trials/learner, real domain3 pool.
  splits   cross-learner: 32 train (fit) / 16 test (all evaluation);
           within-learner temporal: first-150 fit → last-150 scoring.

  A. RECOVERY (train cohort): fitted vs true (log α_t, log α_σ, ℓ_∞) at
     n=150 vs n=300 — identifiability vs session length (pilot design
     input: how long must a session be before the dynamics are estimable).
  B. HELD-OUT PREDICTION (test cohort, trials 150–300 scored): per-trial
     Δlog-evidence vs the default params for fitted-POPULATION params
     (train-cohort means — what a pilot fit would deploy), PERSONALIZED
     params (fit on the learner's own first 150), and ORACLE (true
     individual params — the ceiling).
  C. RULE-FORM IDENTIFICATION (F12): fresh 12 soft + 12 hard learners; fit
     BOTH rule forms on first 150, classify by held-out last-150 evidence.
  D. CLOSED-LOOP BENEFIT: test learners re-instantiated; trainer runs with
     default vs fitted-population vs oracle params (paired CRN seeds).
     Two layers: (i) trials to TRUE mastery — expected ≈ null (M10 already
     showed tier-2 placement is rate-misspec-robust; this replicates it);
     (ii) DECLARATION quality via the D16 gate — where the assumed dynamics
     (σ_∞ = the F28 attractor) actually bite: false-graduation rate,
     declaration lateness (n_decl − n_true), never-declared count.

Run:  python3 -m studies.study_phase3_fit [--smoke]
      → figures/data_phase3_fit.npz
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

from training.bank_adapter import BankAdapter, ELL_STAR, SIGMA_STAR, TaskCandidates
from training.benchmark_trainer import _select
import functools

from training.dynamics_fit import filter_evidence
from training.dynamics_fit import fit_learner as _fit_learner
from training.learner_sim import Learner, LearnerParams
from training.trainer_greedy import RewardWeights
from training.training_filter import TaskFilter

TASK = 2
T_STAR = 0.30
DEFAULT_FP = LearnerParams(alpha_t=0.2, alpha_sigma=0.06,
                           sigma_inf=0.82 * SIGMA_STAR[TASK],
                           q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
POP_TRUE = {"lat": np.log(0.32), "las": np.log(0.075), "linf": 0.80,
            "sd_lat": 0.35, "sd_las": 0.35, "sd_linf": 0.20}
PRIOR_SD = 0.45          # logging-filter/evidence initial-belief SD: covers
                         # the biased starts (|θ0|≈.65 ≈ 1.4σ) — realistic
                         # "seeded but imperfect" handoff


# same initial-belief prior as the logging filter (threaded, not patched)
fit_learner = functools.partial(_fit_learner, prior_sd=PRIOR_SD)


def build_pool(seed=0, size=600):
    ad = BankAdapter()
    full = ad.candidates(TASK, feedback_safe=True)
    sub = np.random.default_rng(seed).choice(len(full), size=size,
                                             replace=False)
    return TaskCandidates(TASK, full.seg_id[sub], full.s_mean[sub],
                          full.s_sd[sub], full.y_star[sub], full.margin[sub],
                          full.coherent[sub])


def draw_learner(i, *, rule="soft"):
    r = np.random.default_rng(500_000 + i)
    lp = replace(DEFAULT_FP,
                 alpha_t=float(np.exp(POP_TRUE["lat"]
                                      + POP_TRUE["sd_lat"] * r.standard_normal())),
                 alpha_sigma=float(np.exp(POP_TRUE["las"]
                                          + POP_TRUE["sd_las"] * r.standard_normal())),
                 sigma_inf=float(np.exp(-(POP_TRUE["linf"]
                                          + POP_TRUE["sd_linf"] * r.standard_normal()))),
                 rule=rule)
    sigma0 = float(np.exp(np.log(1.4) + 0.2 * r.standard_normal()))
    t0 = float(r.choice([-1, 1]) * (0.65 + 0.15 * r.standard_normal()))
    return lp, sigma0, t0                           # params, σ0, t0


def simulate_log(lp_true, sigma0, t0, pool, *, n_trials=300, seed=0):
    """Run the LOGGING trainer (tier-2, default params, exact kernel) against
    the true learner; return the (s, y, y*, s_sd) log."""
    rng = np.random.default_rng(700_000 + seed)
    lnr = Learner([sigma0], [t0], lp_true, seed=800_000 + seed)
    filt = TaskFilter(PRIOR_SD * rng.standard_normal(400),
                      PRIOR_SD * rng.standard_normal(400), DEFAULT_FP,
                      seed=900_000 + seed, exact_kernel=True)
    state = {"k": 0, "bal": 0}
    W = RewardWeights()
    log = np.zeros((n_trials, 4))
    for k in range(n_trials):
        state["k"] = k
        idx = _select("tier2", filt, pool, state, rng, W)
        s, s_sd = float(pool.s_mean[idx]), float(pool.s_sd[idx])
        y_star = int(pool.y_star[idx])
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        state["bal"] += 1 if y_star == 1 else -1
        log[k] = (s, y, y_star, s_sd)
    return log


def closed_loop_run(lp_true, sigma0, t0, pool, fp, *, budget=400, seed=0):
    """Closed-loop trainer with belief params fp vs the true learner.
    Returns true-mastery crossing, D16-gate declaration, and honesty flags."""
    from training.trainer_policy import ModeThresholds, TaskModePolicy
    rng = np.random.default_rng(700_000 + seed)
    lnr = Learner([sigma0], [t0], lp_true, seed=800_000 + seed)
    filt = TaskFilter(PRIOR_SD * rng.standard_normal(400),
                      PRIOR_SD * rng.standard_normal(400), fp,
                      seed=900_000 + seed, exact_kernel=True)
    mp = TaskModePolicy(0, ELL_STAR[TASK], SIGMA_STAR[TASK], ModeThresholds())
    state = {"k": 0, "bal": 0}
    W = RewardWeights()
    sig_star = SIGMA_STAR[TASK]
    n_true = None
    for k in range(budget):
        state["k"] = k
        idx = _select("tier2", filt, pool, state, rng, W)
        s, s_sd = float(pool.s_mean[idx]), float(pool.s_sd[idx])
        y_star = int(pool.y_star[idx])
        s_real = s + s_sd * rng.standard_normal()
        y = lnr.step(s_real, 0, y_star, feedback=True)
        filt.step(s, y, y_star, s_sd=s_sd, feedback=True)
        mp.note_posterior(filt)
        state["bal"] += 1 if y_star == 1 else -1
        if n_true is None and (lnr.sigma[0] <= sig_star
                               and abs(lnr.t[0]) <= T_STAR):
            n_true = k + 1
        if mp.is_mastered(filt):
            return {"declared": True, "n_decl": k + 1, "n_true": n_true,
                    "fg": n_true is None}
    return {"declared": False, "n_decl": budget, "n_true": n_true,
            "fg": False}


def ev3(trials, params, *, score_from=0):
    """3-seed-averaged prequential evidence: resample-branch MC noise on a
    single CRN evaluation is the same order as the per-trial signal (smoke
    lesson) — averaging over independent filter seeds restores contrast."""
    return float(np.mean([filter_evidence(trials, params, seed=sd,
                                          prior_sd=PRIOR_SD,
                                          score_from=score_from)
                          for sd in (42, 43, 44)]))


def _xtrue(lp):
    return np.array([np.log(lp.alpha_t), np.log(lp.alpha_sigma),
                     -np.log(lp.sigma_inf)])


def main(smoke=False):
    n_train, n_test = (6, 4) if smoke else (24, 16)
    n_rule = 4 if smoke else 10
    n_trials = 200 if smoke else 300
    half = n_trials // 2
    maxiter = 60 if smoke else 120
    pool = build_pool()
    names = ("log_alpha_t", "log_alpha_s", "ell_inf")

    # ── cohort + logs ──
    cohort = [draw_learner(i) for i in range(n_train + n_test)]
    logs = [simulate_log(lp, s0, t0, pool, n_trials=n_trials, seed=i)
            for i, (lp, s0, t0) in enumerate(cohort)]
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n_train + n_test))

    # ── A. recovery on the train cohort, n=half vs n=full ──
    print("A. parameter recovery (train cohort)")
    fits = {}
    for n_fit, tag in ((half, f"n={half}"), (n_trials, f"n={n_trials}")):
        X_hat, X_true = [], []
        for i in train_idx:
            fr = fit_learner(logs[i][:n_fit], DEFAULT_FP, seed=42,
                             maxiter=maxiter)
            X_hat.append(fr.x)
            X_true.append(_xtrue(cohort[i][0]))
        X_hat, X_true = np.array(X_hat), np.array(X_true)
        fits[n_fit] = (X_hat, X_true)
        for j, nm in enumerate(names):
            r = float(np.corrcoef(X_hat[:, j], X_true[:, j])[0, 1])
            rmse = float(np.sqrt(np.mean((X_hat[:, j] - X_true[:, j]) ** 2)))
            bias = float(np.mean(X_hat[:, j] - X_true[:, j]))
            print(f"   {tag:>6} {nm:>12}: corr {r:+.2f}  RMSE {rmse:.3f}  "
                  f"bias {bias:+.3f}  (cohort SD {X_true[:, j].std():.3f})")

    # fitted POPULATION params from the full-length train fits
    Xh_full = fits[n_trials][0]
    pop_x = Xh_full.mean(axis=0)
    fitted_pop = replace(DEFAULT_FP, alpha_t=float(np.exp(pop_x[0])),
                         alpha_sigma=float(np.exp(pop_x[1])),
                         sigma_inf=float(np.exp(-pop_x[2])))
    print(f"   fitted population means: α_t {fitted_pop.alpha_t:.3f} "
          f"(true {np.exp(POP_TRUE['lat']):.3f}, default 0.200), "
          f"α_σ {fitted_pop.alpha_sigma:.3f} "
          f"(true {np.exp(POP_TRUE['las']):.3f}, default 0.060), "
          f"ℓ∞ {pop_x[2]:.3f} (true {POP_TRUE['linf']:.3f}, default 0.733)")

    # ── B. held-out prediction on the test cohort (score trials half..end) ──
    print(f"B. held-out predictive log-evidence per trial "
          f"(test cohort, trials {half}..{n_trials})")
    d_pop, d_per, d_ora = [], [], []
    train_ok = ell_star_ok = 0
    ell_star = ELL_STAR[TASK]
    for i in test_idx:
        lg = logs[i]
        base = ev3(lg, DEFAULT_FP, score_from=half)
        d_pop.append(ev3(lg, fitted_pop, score_from=half) - base)
        pers = fit_learner(lg[:half], DEFAULT_FP, seed=42, maxiter=maxiter)
        d_per.append(ev3(lg, pers.params, score_from=half) - base)
        d_ora.append(ev3(lg, cohort[i][0], score_from=half) - base)
        # decision-relevant: does the personalized ceiling classify
        # trainability (sign of ell_inf - ell_star) correctly?
        truth = -np.log(cohort[i][0].sigma_inf) > ell_star
        train_ok += ((pers.x[2] > ell_star) == truth)
        ell_star_ok += truth
    m = n_trials - half
    for nm, d in (("fitted-pop", d_pop), ("personalized", d_per),
                  ("oracle", d_ora)):
        d = np.array(d) / m
        print(f"   Δ vs default, {nm:>12}: {d.mean():+.4f}/trial "
              f"± {d.std() / np.sqrt(len(d)):.4f}")
    print(f"   personalized trainability classification: {train_ok}/{len(test_idx)}"
          f" correct (cohort truly-trainable: {ell_star_ok}/{len(test_idx)};"
          f" default always predicts trainable)")

    # ── C. rule-form identification (F12) ──
    print(f"C. soft-vs-hard rule identification "
          f"(fit first {half}, score last {m})")
    correct, margins = 0, []
    for j in range(2 * n_rule):
        true_rule = "soft" if j < n_rule else "hard"
        lp, s0, t0 = draw_learner(10_000 + j, rule=true_rule)
        lg = simulate_log(lp, s0, t0, pool, n_trials=n_trials,
                          seed=20_000 + j)
        ev = {}
        for rule in ("soft", "hard"):
            fr = fit_learner(lg[:half], DEFAULT_FP, rule=rule, seed=42,
                             maxiter=maxiter)
            ev[rule] = ev3(lg, fr.params, score_from=half)
        pick = "soft" if ev["soft"] >= ev["hard"] else "hard"
        correct += (pick == true_rule)
        margins.append(abs(ev["soft"] - ev["hard"]))
    print(f"   accuracy {correct}/{2 * n_rule}; "
          f"median held-out |Δevidence| {np.median(margins):.2f} nats")

    # ── D. closed-loop benefit (paired CRN over test learners) ──
    print("D. closed-loop (test cohort, paired): true mastery + declaration")
    arms = {"default": DEFAULT_FP, "fitted-pop": fitted_pop}
    rows = {nm: [closed_loop_run(*cohort[i], pool, fp, seed=50_000 + i)
                 for i in test_idx] for nm, fp in arms.items()}
    rows["oracle"] = [closed_loop_run(*cohort[i], pool, cohort[i][0],
                                      seed=50_000 + i) for i in test_idx]
    for nm, rs in rows.items():
        nt = [r["n_true"] for r in rs if r["n_true"] is not None]
        dec = [r for r in rs if r["declared"]]
        late = [r["n_decl"] - r["n_true"] for r in dec
                if r["n_true"] is not None]
        print(f"   {nm:>10}: true-mastery med "
              f"{np.median(nt) if nt else -1:5.1f} "
              f"({len(nt)}/{len(rs)} reached) | declared {len(dec)}/{len(rs)}"
              f"  FG {sum(r['fg'] for r in rs)}"
              f"  med lateness {np.median(late) if late else -1:5.1f}")

    if not smoke:
        np.savez("figures/data_phase3_fit.npz",
                 X_hat_half=fits[half][0], X_hat_full=fits[n_trials][0],
                 X_true=fits[n_trials][1], pop_x=pop_x,
                 d_pop=np.array(d_pop), d_per=np.array(d_per),
                 d_ora=np.array(d_ora), score_len=m,
                 rule_correct=correct, rule_n=2 * n_rule,
                 rule_margins=np.array(margins),
                 closed_default=np.array(
                     [(r["declared"], r["n_decl"],
                       -1 if r["n_true"] is None else r["n_true"], r["fg"])
                      for r in rows["default"]], dtype=float),
                 closed_fitted=np.array(
                     [(r["declared"], r["n_decl"],
                       -1 if r["n_true"] is None else r["n_true"], r["fg"])
                      for r in rows["fitted-pop"]], dtype=float),
                 closed_oracle=np.array(
                     [(r["declared"], r["n_decl"],
                       -1 if r["n_true"] is None else r["n_true"], r["fg"])
                      for r in rows["oracle"]], dtype=float))
        print("saved figures/data_phase3_fit.npz")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
