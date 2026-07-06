"""M23 — regime-shift study (F80–F83): fit + validate the jump kernel, DMA
forgetting, the RT-floor guard, and the contact e-process on the seven-profile
sandbox logs (USER-A..G, 2026-07-02/03).

Empirical motivation (docs/M23_REGIME_ANALYSIS.md): the 2026-07-03 sessions
show DISCRETE regime shifts — USER-E jumped d′ 0.8 → 3.0 across a 64 s sitting
break (Δlog σ ≈ 1.3, cumulatively impossible under q_σ = 0.02), USER-C jumped
d′ −0.3 → 2.0 overnight — which the smooth kernel tracks with a ~full-session
lag (π = 0.5 crossed at trial 12/26 and 21/23 respectively) while the mixture
trainability sits poisoned by the guessing onboarding (min 0.34/0.36).

Arms:
  A  hyperparameter fit — replay every (user, task) logged sequence through
     SigmaInfMixtureFilter variants (CRN seeds): boundary jump (ε_b, Λ),
     per-trial jump (ε, κ), DMA forgetting γ, stage-2 combos. Criterion:
     pooled prequential log-evidence Σ log p(y_t | H_t) (Dawid 1984) —
     adaptive serving cancels by the same ignorability argument that
     licenses the filter itself. Guard: leave-one-user-out selection.
  A2 breakout lag + trainability recovery on the two real regime shifts,
     base vs the pinned variant.
  B  synthetic tracking — E-like jump learner (σ 3.5 → 0.55 at a session
     boundary, 120 post-jump trials) + WELL-SPECIFIED stationary control
     (learner_sim under the filter's own assumed dynamics): post-jump lag,
     declaration, coverage, stationary width/coverage cost.
  C  FG safety on the M22 production harness (study_participant_gates
     Harness2: TrainerPolicy + cert probes + e-gates over the real bank) —
     static_below (ℓ* − 0.15) and careless (λ = 0.35 at the bar):
     false-graduation counts and trainability excursions under
     {base, jump, forget, both}.
  D  information floor — steady_state_sd under per-trial jumps (if pinned);
     boundary-jump transient width + re-contraction time.
  G  harness-realism re-pin (the PECR rework): stage-1 guardrails used
     single-task serving (double the real per-task evidence rate); under
     the 2-task production harness the stage-1 pin never declares. G
     re-measures lateness + the achievable sd_ℓ cycle-min and records the
     FINAL config bj(0.10, 0.75, 0.33) + fs 0.10 with the sandbox
     declaration floor re-set 0.23 → 0.33 (F5/F77-ii logic). Downstream
     arms (A2/C/D) validate the FINAL config at the shipping floor.
  E  RT-floor guard calibration — two rules (trailing-6 median, 3-consecutive)
     × floor sweep: engaged-envelope false alarms vs USER-D detection latency.
  F  contact e-process — E_s = Π_t KT(c_t | past)/0.55, per session, with the
     Krichevsky–Trofimov universal Bernoulli predictor as numerator: an exact
     nonnegative supermartingale under the chance-band null
     P(c_t = 1 | F_{t−1}) ∈ [0.45, 0.55] (Ville ⇒ anytime-valid α = 0.05 at
     threshold 20). Trials-to-contact on the real logs; null false-contact
     rate, biased-guesser robustness, competent-learner power.

Run:  python3 -m studies.study_m23_regime [--smoke]
Out:  figures/data_m23_regime.npz  (+ printed tables)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.stats import norm

from sandbox import config as C
from training.learner_sim import Learner
from training.mixture_filter import SigmaInfMixtureFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_ROOT = os.path.join(ROOT, "sandbox")
FIG = os.path.join(ROOT, "figures")
USERS = ["A", "B", "C", "D", "E", "F", "G"]
LAPSE = 0.025
MULT = 1.0772256                 # SKILL_MODE_MULTIPLIER (bridge_conventions)
POOL_S_MAX = 1.3                 # sandbox bank |s_mean| ceiling (measured)
BANK_S_SD = 0.057                # sandbox bank median s_sd (measured)
CONTACT_P0 = 0.55                # chance-band envelope (label balance .45-.55)
CONTACT_LOG_THRESH = float(np.log(20.0))   # 1/alpha, alpha = 0.05


# ── data ──
def load_sequences():
    """{(user, task): ordered trial dicts} from the live sandbox logs."""
    seqs = {}
    for u in USERS:
        p = os.path.join(LOG_ROOT, f"logs-USER-{u}", "trials.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p):
            r = json.loads(line)
            seqs.setdefault((u, r["task"]), []).append(r)
    return seqs


def fresh_mixture(task, seed, *, forget=1.0, jump_eps=0.0, jump_kappa=8.0):
    rng = np.random.default_rng(seed)
    th0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
    el0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
    return SigmaInfMixtureFilter(
        th0, el0, C.assumed_params(task), ell_inf_mean=C.EXPERT_ELL[task],
        tau=C.MIX_TAU, J=C.MIX_J, p_static_stratum=C.P_STATIC_STRATUM,
        ell_star=C.ELL_STAR[task], seed=seed + 31 * task,
        forget=forget, jump_eps=jump_eps, jump_kappa=jump_kappa)


# ── arm A: replay + prequential evidence ──
def replay(seq, task, seed, *, forget=1.0, jump_eps=0.0, jump_kappa=8.0,
           bj=None, fs=0.0, want_traj=False):
    """Replay one logged (user, task) sequence. Returns total prequential
    log-evidence (+ per-trial pi/trainability trajectories if asked).
    Boundary jumps (bj = (eps, sd)) fire at every session change; the live
    anchor-block machinery is NOT reproduced (identical omission for all
    variants, so evidence differences attribute to the mechanism)."""
    mix = fresh_mixture(task, seed, forget=forget, jump_eps=jump_eps,
                        jump_kappa=jump_kappa)
    ll, last_session = 0.0, None
    traj = {"pi": [], "train": [], "session": []} if want_traj else None
    for r in seq:
        if (last_session is not None and r["session"] != last_session
                and (bj or fs)):
            mix.boundary_shift(*(bj or (0.0, 0.0, 1.0)), w_share=fs)
        last_session = r["session"]
        p1 = mix.predictive_p(r["s_mean"], r["s_sd"])
        p_obs = p1 if r["y"] == 1 else 1.0 - p1
        ll += np.log(max(p_obs, 1e-12))
        if want_traj:
            pi, _ = mix.pass_mass(C.ELL_STAR[task])
            traj["pi"].append(pi)
            traj["train"].append(mix.trainability(C.ELL_STAR[task]))
            traj["session"].append(r["session"])
        mix.step(r["s_mean"], r["y"], r["y_star"], s_sd=r["s_sd"])
    return (ll, traj) if want_traj else ll


def variant_grid(smoke):
    grid = [("base", {})]
    bj_eps = [0.05, 0.10, 0.20] if not smoke else [0.10]
    bj_sd = [0.50, 0.75, 1.00] if not smoke else [0.75]
    bj_ts = [1.0, 0.33] if not smoke else [0.33]
    for e in bj_eps:
        for s in bj_sd:
            for ts in bj_ts:
                grid.append((f"bj_e{e:g}_s{s:g}_t{ts:g}",
                             {"bj": (e, s, ts)}))
    tj = ([(0.01, 6.0), (0.02, 6.0), (0.01, 10.0)] if not smoke
          else [(0.01, 6.0)])
    for e, k in tj:
        grid.append((f"tj_e{e:g}_k{k:g}", {"jump_eps": e, "jump_kappa": k}))
    fg = [0.95, 0.98, 0.99] if not smoke else [0.98]
    for g in fg:
        grid.append((f"fg_g{g:g}", {"forget": g}))
    return grid


def _score(seqs, keys, seeds, kw):
    per = np.zeros(len(keys))
    for i, (u, t) in enumerate(keys):
        per[i] = np.mean([replay(seqs[(u, t)], t, 1000 + 17 * s, **kw)
                          for s in seeds])
    return per


def arm_A(seqs, seeds, smoke):
    print("\n== ARM A: hyperparameter fit on the real logs ==")
    grid = variant_grid(smoke)
    keys = sorted(seqs.keys())
    ev, kwmap = {}, {}
    for name, kw in grid:
        ev[name] = _score(seqs, keys, seeds, kw)
        kwmap[name] = kw
        print(f"  {name:<16} pooled logE = {ev[name].sum():9.2f} "
              f"(Δ vs base {ev[name].sum() - ev['base'].sum():+7.2f})")

    def best(prefix, ev_):
        cand = {n: v.sum() for n, v in ev_.items() if n.startswith(prefix)}
        return max(cand, key=cand.get) if cand else None

    combos = []
    b_bj, b_fg, b_tj = best("bj", ev), best("fg", ev), best("tj", ev)
    if b_bj:
        for r in ([0.05, 0.10, 0.20] if not smoke else [0.10]):
            combos.append((f"combo_bj+fs{r:g}", {**kwmap[b_bj], "fs": r}))
    if b_bj and b_fg:
        combos.append(("combo_bj+fg", {**kwmap[b_bj], **kwmap[b_fg]}))
    if b_tj and b_fg and not smoke:
        combos.append(("combo_tj+fg", {**kwmap[b_tj], **kwmap[b_fg]}))
    for name, kw in combos:
        ev[name] = _score(seqs, keys, seeds, kw)
        kwmap[name] = kw
        print(f"  {name:<16} pooled logE = {ev[name].sum():9.2f} "
              f"(Δ vs base {ev[name].sum() - ev['base'].sum():+7.2f})")
    print("  -- leave-one-user-out (selection excludes the held-out user) --")
    louo = {}
    names = [n for n in ev if n != "base"]
    for u in USERS:
        idx_in = [i for i, (uu, _) in enumerate(keys) if uu != u]
        idx_out = [i for i, (uu, _) in enumerate(keys) if uu == u]
        if not idx_out:
            continue
        sel = max(names, key=lambda n: ev[n][idx_in].sum())
        d = ev[sel][idx_out].sum() - ev["base"][idx_out].sum()
        louo[u] = (sel, float(d))
        print(f"    held-out {u}: selects {sel:<16} held-out Δ {d:+6.2f}")
    return ev, kwmap, keys, louo


def arm_A_lag(seqs, seeds, pinned):
    """Breakout lag + trainability recovery on the two real regime shifts."""
    print("\n== ARM A2: breakout lag on the real regime shifts ==")
    cases = [("E", 2, 3), ("C", 1, 2)]        # (user, task, breakout session)
    out = {}
    for u, t, s_break in cases:
        seq = seqs[(u, t)]
        for name, kw in [("base", {}), ("m23", pinned)]:
            crosses, t_open, recov = [], [], []
            for s in seeds:
                _, traj = replay(seq, t, 1000 + 17 * s, want_traj=True, **kw)
                pi = np.array(traj["pi"])
                tr = np.array(traj["train"])
                ses = np.array(traj["session"])
                in_b = np.where(ses == s_break)[0]
                cross = next((k for k, i in enumerate(in_b)
                              if pi[i] > 0.5), len(in_b))
                crosses.append(cross)
                t_open.append(tr[in_b[0]])
                rec = next((k for k, i in enumerate(in_b)
                            if tr[i] >= 0.7), len(in_b))
                recov.append(rec)
            out[(u, t, name)] = (float(np.mean(crosses)),
                                 float(np.mean(t_open)),
                                 float(np.mean(recov)))
            print(f"  {u}-t{t} breakout s{s_break} [{name}]: "
                  f"trials to π>0.5 = {np.mean(crosses):.1f}, "
                  f"trainability at open {np.mean(t_open):.2f}, "
                  f"recovery to ≥0.7 in {np.mean(recov):.1f} trials")
    return out


# ── serving loop (headless, sandbox-shaped placement) for arms B/F ──
def serve_loop(respond, mix, task, n_trials, *, boundary_every=40,
               bj=None, fs=0.0, sd_floor=0.23, z=1.0, want=()):
    ell_star = C.ELL_STAR[task]
    sign, rec = 1, {k: [] for k in want}
    declared = None
    for k in range(n_trials):
        if (bj or fs) and k > 0 and k % boundary_every == 0:
            mix.boundary_shift(*(bj or (0.0, 0.0, 1.0)), w_share=fs)
        mt, ml = mix.mean()
        sig_hat, t_hat = float(np.exp(-ml)), float(-mt)
        _, sd_l = mix.sd()
        sig_place = sig_hat * np.exp(z * max(sd_l - sd_floor, 0.0))
        s = float(np.clip(t_hat + sign * MULT * sig_place,
                          -POOL_S_MAX, POOL_S_MAX))
        sign *= -1
        y_star = int(s > 0.0)
        y = respond(k, s, y_star)
        if "p_c" in rec:
            p1 = mix.predictive_p(s, BANK_S_SD)
            rec["p_c"].append(p1 if y_star == 1 else 1.0 - p1)
        if "correct" in rec:
            rec["correct"].append(int(y == y_star))
        if "sig" in rec:                       # PRE-update estimate at trial k
            rec["sig"].append(sig_hat)
        mix.step(s, y, y_star, s_sd=BANK_S_SD)
        if "sd_l" in rec:
            rec["sd_l"].append(mix.sd()[1])
        if "pi" in rec:
            rec["pi"].append(mix.pass_mass(ell_star)[0])
        if "train" in rec:
            rec["train"].append(mix.trainability(ell_star))
        if "ell_q" in rec:                     # post-update 90% interval
            w = mix.w
            rec["ell_q"].append(_wquant(mix.ell, w, [0.05, 0.95]))
        if declared is None and mix.is_mastered(ell_star, sd_floor=sd_floor):
            declared = k
    return rec, declared


def _wquant(x, w, qs):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    cw /= cw[-1]
    return [float(x[i][min(np.searchsorted(cw, q), len(x) - 1)]) for q in qs]


def probit_responder(sigma_fn, t_fn, rng, lapse=LAPSE):
    def respond(k, s, y_star):
        p = lapse + (1 - 2 * lapse) * norm.cdf((s - t_fn(k)) / sigma_fn(k))
        return int(rng.random() < p)
    return respond


def arm_B(task, n_seeds, shortlist, smoke):
    """shortlist: {name: pinned-style kwargs dict}. Returns per-variant
    tracking + stationary metrics for the pinning rule."""
    print("\n== ARM B: synthetic tracking ==")
    n_tr = 200 if not smoke else 120
    jump_at = 80
    arms = {name: ({k: v for k, v in kw.items()
                    if k not in ("bj", "fs")},
                   kw.get("bj"), kw.get("fs", 0.0))
            for name, kw in shortlist.items()}
    res = {}
    for name, (kw, bj, fs) in arms.items():
        lag_sig, lag_pi, decl, cov_post = [], [], [], []
        for sd in range(n_seeds):
            rng = np.random.default_rng(500 + sd)
            mix = fresh_mixture(task, 700 + 13 * sd, **kw)
            sig_true = lambda k: 3.5 if k < jump_at else 0.55
            resp = probit_responder(sig_true, lambda k: 0.05, rng)
            rec, d = serve_loop(resp, mix, task, n_tr, bj=bj, fs=fs,
                                want=("sig", "pi", "ell_q"))
            sig = np.array(rec["sig"])
            pi = np.array(rec["pi"])
            ok = np.abs(np.log(sig[jump_at:]) - np.log(0.55)) < 0.25
            lag_sig.append(next((i for i in range(len(ok))
                                 if ok[i:i + 3].all()), n_tr - jump_at))
            lag_pi.append(next((i for i, p in enumerate(pi[jump_at:])
                                if p > 0.5), n_tr - jump_at))
            decl.append((d - jump_at) if d is not None and d >= jump_at
                        else n_tr - jump_at)
            q = np.array(rec["ell_q"])
            tru = np.array([-np.log(sig_true(k)) for k in range(n_tr)])
            inb = (q[:, 0] <= tru) & (tru <= q[:, 1])
            cov_post.append(inb[jump_at + 20:].mean())
        res[name] = dict(lag_sig=float(np.mean(lag_sig)),
                         lag_pi=float(np.mean(lag_pi)),
                         decl=float(np.mean(decl)),
                         cov_post=float(np.mean(cov_post)))
        print(f"  E-like jump [{name}]: lag(σ̂) {np.mean(lag_sig):5.1f}  "
              f"lag(π>.5) {np.mean(lag_pi):5.1f}  declare {np.mean(decl):5.1f}"
              f" post-jump trials  cov90 post(+20) {np.mean(cov_post):.2f}")
    # WELL-SPECIFIED control: learner follows the assumed dynamics from
    # σ0 = 1.0 toward its ceiling — measures the mechanisms' cost when the
    # M22 model is RIGHT (coverage, width, and declaration reachability)
    for name, (kw, bj, fs) in arms.items():
        cov, sdl, rmse, decl = [], [], [], []
        for sd in range(n_seeds):
            mix = fresh_mixture(task, 1100 + 13 * sd, **kw)
            lnr = Learner([1.0], [0.1], C.assumed_params(task),
                          seed=9000 + sd)
            truth = []

            def respond(k, s, y_star):
                truth.append(-np.log(lnr.sigma[0]))
                return lnr.step(s, 0, y_star=y_star, feedback=True)

            rec, d = serve_loop(respond, mix, task, n_tr, bj=bj, fs=fs,
                                want=("sig", "ell_q", "sd_l"))
            q = np.array(rec["ell_q"])
            tru = np.array(truth)
            cov.append(((q[:, 0] <= tru) & (tru <= q[:, 1]))[40:].mean())
            sdl.append(float(np.mean(rec["sd_l"][-20:])))
            rmse.append(float(np.sqrt(np.mean(
                (np.log(np.array(rec["sig"])[40:]) + tru[40:]) ** 2))))
            decl.append(d if d is not None else n_tr + 1)
        decl = np.array(decl)
        res[name + "_stat"] = dict(cov=float(np.mean(cov)),
                                   sdl=float(np.mean(sdl)),
                                   rmse=float(np.mean(rmse)),
                                   decl_rate=float((decl <= n_tr).mean()),
                                   decl_med=float(np.median(decl)))
        print(f"  well-spec control [{name}]: cov90 {np.mean(cov):.2f}  "
              f"sd_ℓ(end) {np.mean(sdl):.3f}  RMSE(lnσ) {np.mean(rmse):.3f}  "
              f"declares {(decl <= n_tr).mean():.0%} "
              f"@ median {np.median(decl):.0f}")
    return res


def arm_C(n_seeds, pinned, smoke, sd_floor=0.23):
    """FG safety on the M22 production harness (probes + e-gates + bank),
    at the SHIPPING declaration floor (arm G re-measured it to 0.33 for the
    final config; base comparators run at the M22 floor 0.23)."""
    print("\n== ARM C: FG safety (M22 Harness2, production gate stack) ==")
    from studies.study_participant_gates import (Harness2, TASKS as HTASKS)
    from training.bank_adapter import BankAdapter
    bank = BankAdapter()
    n_sessions = 6 if not smoke else 3
    mix_kw = {k: v for k, v in pinned.items() if k not in ("bj", "fs")}
    arms = {"base": ({}, None, 0.0),
            "pinned": (mix_kw, pinned.get("bj"), pinned.get("fs", 0.0))}
    if pinned.get("fs"):
        arms["fs_only"] = ({}, None, pinned["fs"])
    if pinned.get("bj"):
        arms["bj_only"] = ({}, pinned["bj"], 0.0)

    def swapped(seed, kw, floor):
        h = Harness2(bank, seed, floor=C.TRAINABILITY_FLOOR)
        h.pol.th.sd_floor = floor
        if kw:
            rng = np.random.default_rng(seed + 1)
            for i, t in enumerate(HTASKS):
                th0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
                el0 = C.PRIOR_SD * rng.standard_normal(C.N_PARTICLES)
                f = SigmaInfMixtureFilter(
                    th0, el0, C.assumed_params(t),
                    ell_inf_mean=C.EXPERT_ELL[t], tau=C.MIX_TAU, J=C.MIX_J,
                    p_static_stratum=C.P_STATIC_STRATUM,
                    ell_star=C.ELL_STAR[t], seed=(seed + 1) + 31 * t, **kw)
                h.filters[t] = f
                h.pol.filters[i] = f
        return h

    def zoo_responders(seed):
        sig_below = {t: float(np.exp(-(C.ELL_STAR[t] - 0.15)))
                     for t in HTASKS}
        rng = np.random.default_rng(seed)

        def static_below(t, s_real, y_star):
            p = LAPSE + (1 - 2 * LAPSE) * norm.cdf((s_real - 0.1)
                                                   / sig_below[t])
            return int(rng.random() < p)

        def careless(t, s_real, y_star):
            p = 0.35 + 0.30 * norm.cdf((s_real - 0.1)
                                       / float(C.SIGMA_STAR[t]))
            return int(rng.random() < p)
        return {"static_below": static_below, "careless": careless}

    res = {}
    for zname in ("static_below", "careless"):
        for aname, (kw, bj, fs) in arms.items():
            fg, tmax = 0, []
            floor = 0.23 if aname == "base" else sd_floor
            for sd in range(n_seeds):
                h = swapped(2000 + 37 * sd, kw, floor)
                respond = zoo_responders(6000 + sd)[zname]
                mastered_ever = False
                for s_i in range(n_sessions):
                    if s_i > 0 and (bj or fs):
                        for t in HTASKS:
                            h.filters[t].boundary_shift(
                                *(bj or (0.0, 0.0, 1.0)), w_share=fs)
                    h.run_session(respond, n_trials=40)
                mastered_ever = any(any(r["mastered"].values())
                                    for r in h.rows)
                fg += int(mastered_ever)
                tmax.append(max(max(r["train"].values())
                                for r in h.rows[60:]) if len(h.rows) > 60
                            else np.nan)
            res[(zname, aname)] = dict(fg=fg, n=n_seeds,
                                       tmax=float(np.nanmean(tmax)))
            print(f"  {zname:<13} [{aname:<6}]: FG {fg}/{n_seeds}  "
                  f"max trainability(>60) {np.nanmean(tmax):.2f}")
    return res


def arm_G(pinned_stage1, smoke):
    """Harness-realism re-pin (the M23 PECR rework). Arm B's well-specified
    control served ONE task 40 trials per boundary — twice the per-task
    evidence rate of the real 2-task sessions (~20 trials/task/session).
    Under the real rate, per-boundary hazards re-inject width/cross-strata
    variance faster than evidence re-concentrates it, and the stage-1
    pinned variant (bj(0.2,1.0,0.33)+fs0.2) NEVER declares in 8 sessions
    (0/8 seeds) vs M22 median trial 126. This arm re-measures under the
    2-task production harness:
      G1 declaration lateness per candidate at the M22 floor 0.23 + the
         achievable end-of-session sd_ℓ cycle-minimum (F5: the gate floor
         must track the achievable steady-state variance; F77-ii
         precedent — under Gate 4 the cycle-min sits at 0.226–0.263,
         AT/ABOVE the old floor, so declarations starve);
      G3 lateness at the re-measured floor for the FINAL config.
    FG at the final floor and E-like tracking are re-verified in the main
    epilogue (fg/track printout). FINAL (recorded, wired into
    sandbox/config.py): bj(0.10, 0.75, 0.33), fs = 0.10, sandbox
    declaration sd_floor 0.23 → 0.33."""
    print("\n== ARM G: harness-realism re-pin (2-task production rate) ==")
    from studies.study_participant_gates import Harness2, TASKS as HTASKS
    from training.bank_adapter import BankAdapter
    bank = BankAdapter()
    n_seeds = 8 if not smoke else 3
    n_sess = 8 if not smoke else 4
    FINAL = ((0.10, 0.75, 0.33), 0.10)
    FINAL_FLOOR = 0.33

    def lateness(shift, floor, seeds):
        firsts, mins = [], []
        for sd in range(seeds):
            h = Harness2(bank, 5000 + 41 * sd, floor=C.TRAINABILITY_FLOOR)
            h.pol.th.sd_floor = floor
            lnrs = {t: Learner([1.0], [0.1], C.assumed_params(t),
                               seed=8000 + sd + t) for t in HTASKS}

            def respond(t, s_real, y_star):
                return lnrs[t].step(s_real, 0, y_star=y_star, feedback=True)

            first, mn = None, np.inf
            for s_i in range(n_sess):
                if s_i > 0 and shift is not None:
                    for t in HTASKS:
                        h.filters[t].boundary_shift(*shift[0],
                                                    w_share=shift[1])
                h.run_session(respond, n_trials=40)
                mn = min(mn, min(h.filters[t].sd()[1] for t in HTASKS))
                if first is None:
                    idx = next((i for i, r in enumerate(h.rows)
                                if any(r["mastered"].values())), None)
                    if idx is not None:
                        first = idx
            firsts.append(first if first is not None else n_sess * 40 + 1)
            mins.append(mn)
        return np.array(firsts), np.array(mins)

    out = {"final": [list(FINAL[0]), FINAL[1], FINAL_FLOOR]}
    s1 = ((pinned_stage1.get("bj"), pinned_stage1.get("fs", 0.0))
          if pinned_stage1.get("bj") else None)
    for name, shift in (("base", None), ("stage1-pin", s1),
                        ("final", FINAL)):
        f, m = lateness(shift, 0.23, n_seeds)
        out[f"lat23_{name}"] = [float(np.median(f)),
                                float(np.mean(f <= n_sess * 40))]
        print(f"  G1 lateness@floor .23 [{name:<10}]: median "
              f"{np.median(f):5.0f}, rate {np.mean(f <= n_sess * 40):.0%} "
              f"(sd_ℓ cycle-min {np.min(m):.3f}–{np.max(m):.3f})")
    f, m = lateness(FINAL, FINAL_FLOOR, n_seeds)
    out["lat_final"] = [float(np.median(f)),
                        float(np.mean(f <= n_sess * 40))]
    print(f"  G3 lateness@floor {FINAL_FLOOR} [final]: median "
          f"{np.median(f):5.0f}, rate {np.mean(f <= n_sess * 40):.0%}")
    return out


def arm_D(task, pinned, smoke):
    print("\n== ARM D: information floor ==")
    from training.training_filter import steady_state_sd
    p = C.assumed_params(task)
    n_tr = 400 if not smoke else 150
    out = {}
    base = steady_state_sd(p, s_sd=BANK_S_SD, n_trials=n_tr, seed=3)
    out["floor_base"] = list(base)
    print(f"  per-trial floor base: θ-SD {base[0]:.3f}, ℓ-SD {base[1]:.3f}")
    if "jump_eps" in pinned:
        j = steady_state_sd(p, s_sd=BANK_S_SD, n_trials=n_tr, seed=3,
                            filter_kwargs={"jump_eps": pinned["jump_eps"],
                                           "jump_kappa": pinned["jump_kappa"]})
        out["floor_jump"] = list(j)
        print(f"  per-trial floor jump: θ-SD {j[0]:.3f}, ℓ-SD {j[1]:.3f} "
              f"(analytic var factor "
              f"{1 + pinned['jump_eps'] * (pinned['jump_kappa']**2 - 1):.2f})")
    if "bj" in pinned or "fs" in pinned:
        rng = np.random.default_rng(11)
        mix = fresh_mixture(task, 2100)
        resp = probit_responder(lambda k: 0.75, lambda k: 0.0, rng)
        serve_loop(resp, mix, task, 120)
        sd_before = mix.sd()[1]
        mix.boundary_shift(*(pinned.get("bj") or (0.0, 0.0, 1.0)),
                           w_share=pinned.get("fs", 0.0))
        sd_after = mix.sd()[1]
        rec2, _ = serve_loop(resp, mix, task, 40, want=("sd_l",))
        sdl = np.array(rec2["sd_l"])
        recontract = next((i for i, v in enumerate(sdl) if v <= sd_before),
                          len(sdl))
        eps, sd = (pinned.get("bj") or (0.0, 0.0))[:2]
        pred = float(np.sqrt(sd_before ** 2 + eps * sd ** 2))
        out["bj_transient"] = [float(sd_before), float(sd_after), pred,
                               int(recontract)]
        print(f"  boundary jump: sd_ℓ {sd_before:.3f} → {sd_after:.3f} "
              f"(analytic {pred:.3f}), re-contracts ≤ start in "
              f"{recontract} trials")
    return out


def arm_E(seqs):
    print("\n== ARM E: RT-floor guard calibration ==")
    sess = {}
    for (u, t), rs in seqs.items():
        for r in rs:
            sess.setdefault((u, r["session"]), []).append(r)
    for k in sess:
        sess[k] = sorted(sess[k], key=lambda r: r["global_trial"])
    engaged = {k: v for k, v in sess.items() if k != ("D", 1)}
    collapse = sess[("D", 1)]
    rts_d = np.array([r["rt_ms"] for r in collapse])
    out = {}
    for rule, W in (("median", 6), ("consec", 3)):
        print(f"  rule: {'trailing-6 median' if rule == 'median' else '3 consecutive'} < floor")
        for floor in (300, 400, 500, 600, 800):
            fa = 0
            for k, rs in engaged.items():
                rts = np.array([r["rt_ms"] for r in rs])
                if len(rts) < W:
                    continue
                if rule == "median":
                    stat = np.array([np.median(rts[i - W + 1:i + 1])
                                     for i in range(W - 1, len(rts))])
                else:
                    stat = np.array([rts[i - W + 1:i + 1].max()
                                     for i in range(W - 1, len(rts))])
                fa += int((stat < floor).any())
            if rule == "median":
                stat_d = np.array([np.median(rts_d[i - W + 1:i + 1])
                                   for i in range(W - 1, len(rts_d))])
            else:
                stat_d = np.array([rts_d[i - W + 1:i + 1].max()
                                   for i in range(W - 1, len(rts_d))])
            hit = next((i + W - 1 for i, v in enumerate(stat_d)
                        if v < floor), None)
            out[(rule, floor)] = (fa, hit)
            print(f"    floor {floor:>4} ms: engaged-session FAs {fa}/"
                  f"{len(engaged)}, D-s1 fires at trial {hit}")
    print("  (M22 accuracy Gate 3 fired at D-s1 trial 39; session had 40)")
    return out


# ── arm F: contact e-process (KT numerator, per-session restart) ──
def kt_walk(c_seq, p0=CONTACT_P0):
    """log E_t = Σ log(KT(c_t | past)/p0); returns (max, first crossing)."""
    logE, mx, cross = 0.0, -np.inf, None
    n1 = 0
    for i, c in enumerate(c_seq):
        p1 = (n1 + 0.5) / (i + 1.0)          # KT predictive for c=1
        q = p1 if c else 1.0 - p1
        logE += np.log(q / p0)
        mx = max(mx, logE)
        if cross is None and logE >= CONTACT_LOG_THRESH:
            cross = i + 1                    # trials consumed
        n1 += c
    return float(mx), cross


def arm_F(seqs, n_seeds, smoke):
    print("\n== ARM F: contact e-process (shadow; per-session restart) ==")
    print("  real logs (trials to contact, per user-task-session):")
    real = {}
    for (u, t), seq in sorted(seqs.items()):
        by_s = {}
        for r in seq:
            by_s.setdefault(r["session"], []).append(r)
        parts = []
        for sid, rs in sorted(by_s.items()):
            cs = [r["correct"] for r in
                  sorted(rs, key=lambda r: r["global_trial"])]
            mx, cross = kt_walk(cs)
            real[(u, t, sid)] = (mx, cross, len(cs))
            parts.append(f"s{sid}:{cross if cross else '—'}/{len(cs)}")
        print(f"    {u}-t{t}: {'  '.join(parts)}")
    n = n_seeds if not smoke else 30
    out = {"real": real}
    for mu, label in [(0.5, "null μ=0.5"), (0.65, "biased μ=0.65")]:
        hits = 0
        for sd in range(n):
            rng = np.random.default_rng(3000 + sd)
            mix = fresh_mixture(1, 3300 + 13 * sd)
            resp = lambda k, s, y_star: int(rng.random() < mu)
            rec, _ = serve_loop(resp, mix, 1, 40, want=("correct",))
            _, cross = kt_walk(rec["correct"])
            hits += int(cross is not None)
        out[f"fa_{mu}"] = hits / n
        print(f"  {label} guesser, 40-trial session: false contact "
              f"{hits}/{n} (bound α=0.05)")
    lat = []
    for sd in range(n):
        rng = np.random.default_rng(4000 + sd)
        mix = fresh_mixture(1, 4300 + 13 * sd)
        resp = probit_responder(lambda k: 0.7, lambda k: 0.05, rng)
        rec, _ = serve_loop(resp, mix, 1, 40, want=("correct",))
        _, cross = kt_walk(rec["correct"])
        lat.append(cross if cross is not None else 41)
    out["power_lat"] = float(np.median(lat))
    print(f"  power (σ=0.7 learner, 40-trial session): median "
          f"trials-to-contact {np.median(lat):.0f} (41 = censored)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    seqs = load_sequences()
    print(f"loaded {len(seqs)} (user, task) sequences, "
          f"{sum(len(v) for v in seqs.values())} trials")
    seeds = [0, 1, 2] if not args.smoke else [0]

    ev, kwmap, keys, louo = arm_A(seqs, seeds, args.smoke)
    base_sum = ev["base"].sum()
    # ── PINNING RULE (pre-stated, three stages) ──
    # 1. evidence-admissibility: pooled prequential Δ vs base within noise
    #    (tolerance = 2× the seed-to-seed SD of the base pooled logE) or
    #    better — the real data must not disfavor the mechanism;
    # 2. among the per-family best admissible variants, choose by
    #    DECISION-RELEVANT tracking (min post-jump lag(π), arm B) — the
    #    failure mode the mechanism exists to fix;
    # 3. guardrails: well-specified control must keep cov90 ≥ 0.85 and
    #    still DECLARE (rate ≥ 0.9, median lateness ≤ 1.4× base); arm C
    #    must show no FG degradation (checked below; STOP if violated).
    per_seed_base = [sum(replay(seqs[(u, t)], t, 1000 + 17 * s)
                         for (u, t) in keys) for s in seeds]
    tol = 2.0 * (float(np.std(per_seed_base)) if len(seeds) > 1 else 1.0)
    admissible = [n for n in ev
                  if n != "base" and ev[n].sum() - base_sum >= -tol]
    print(f"\n  admissibility tolerance {tol:.2f} nats; "
          f"{len(admissible)}/{len(ev) - 1} variants admissible")
    short = {"base": {}}
    for fam in ("fg", "bj", "tj", "combo_bj+fs", "combo_bj+fg"):
        cand = [n for n in admissible if n.startswith(fam)]
        if cand:
            b = max(cand, key=lambda n: ev[n].sum())
            short[b] = kwmap[b]
    print(f"  arm-B shortlist: {list(short)}")

    resB = arm_B(1, 20 if not args.smoke else 5, short, args.smoke)
    base_st = resB["base_stat"]
    ok = []
    for n in short:
        if n == "base":
            continue
        st = resB[n + "_stat"]
        if (st["cov"] >= 0.85 and st["decl_rate"] >= 0.9
                and st["decl_med"] <= 1.4 * base_st["decl_med"]):
            ok.append(n)
    pin_name = (min(ok, key=lambda n: (resB[n]["lag_pi"], -ev[n].sum()))
                if ok else "base")
    pinned = dict(kwmap.get(pin_name, {}))
    print(f"\n  STAGE-1 PIN: {pin_name}  kwargs={pinned}  "
          f"(evidence Δ {ev[pin_name].sum() - base_sum:+.2f}, "
          f"lag(π) {resB[pin_name]['lag_pi'] if pin_name in resB else '—'})")

    # ── arm G: the stage-1 guardrail was measured at the WRONG serving
    # rate (single-task); re-measure lateness/floor on the 2-task
    # production harness and record the FINAL config (the PECR rework —
    # see arm_G docstring). Downstream arms validate the FINAL config.
    resG = arm_G(pinned, args.smoke)
    final = {"bj": (0.10, 0.75, 0.33), "fs": 0.10}
    final_floor = 0.33
    print(f"  FINAL (arm G): {final}  sandbox sd_floor {final_floor}")

    lag = arm_A_lag(seqs, seeds, final)
    resC = arm_C(20 if not args.smoke else 3, final, args.smoke,
                 sd_floor=final_floor)
    resD = arm_D(1, final, args.smoke)
    resE = arm_E(seqs)
    resF = arm_F(seqs, 200, args.smoke)

    # guardrail 3b: FG non-degradation vs base under the production stack
    fg_bad = [z for z in ("static_below", "careless")
              if resC[(z, "pinned")]["fg"] > resC[(z, "base")]["fg"]]
    if fg_bad:
        print(f"  !! STOP: pinned variant degrades FG on {fg_bad} — "
              f"do NOT enable in the sandbox; record and rework.")
    else:
        print("  FG guardrail: pinned variant does not degrade "
              "false graduation vs base.")

    os.makedirs(FIG, exist_ok=True)
    np.savez(os.path.join(FIG, "data_m23_regime.npz"),
             ev_names=np.array(sorted(ev.keys())),
             ev_matrix=np.array([ev[n] for n in sorted(ev.keys())]),
             seq_keys=np.array([f"{u}-t{t}" for u, t in keys]),
             louo=np.array([f"{u}:{s}:{d:+.2f}"
                            for u, (s, d) in louo.items()]),
             pinned=np.array([json.dumps(pinned)]),
             pin_name=np.array([pin_name]),
             final=np.array([json.dumps({**final,
                                         "sd_floor": final_floor})]),
             armG=np.array([json.dumps(resG)]),
             lag=np.array([f"{u}-t{t}-{n}:{a:.1f}:{b:.2f}:{c:.1f}"
                           for (u, t, n), (a, b, c) in lag.items()]),
             armB=np.array([json.dumps(resB)]),
             armC=np.array([json.dumps({f"{z}|{a}": v for (z, a), v
                                        in resC.items()})]),
             armD=np.array([json.dumps(resD)]),
             armE=np.array([f"{r}|{f}:{fa}:{hit}"
                            for (r, f), (fa, hit) in resE.items()]),
             armF=np.array([json.dumps({k: v for k, v in resF.items()
                                        if k != "real"})]),
             armF_real=np.array([f"{u}-t{t}-s{s}:{mx:.2f}:{cr}"
                                 for (u, t, s), (mx, cr, _) in
                                 resF["real"].items()]))
    print(f"\nsaved figures/data_m23_regime.npz  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
