"""M15 smoke test — math-audit fixes (docs/AUDIT_POMDP_MATH.md, steps 1–7).

Checks: MA-3 feedback gating, MA-1 exact kernel (reduction, equivalence),
MA-9 learn-aware greedy Q, MA-6 honest MCSE, MA-2 mixture trainability,
step-5 e-process gate (type-I + power) and cert-probe placement, MA-4 helper.

Run:  python3 -m tests.test_audit_fixes
"""
import numpy as np
from scipy.stats import norm

from training.bridge_conventions import (LAPSE_RATE, SKILL_MODE_MULTIPLIER,
                                         skill_mode_multiplier)
from training.learner_sim import Learner, LearnerParams
from training.training_filter import TaskFilter
from training.mixture_filter import SigmaInfMixtureFilter
from training.trainer_policy import EProcessGate, cert_probe_score

N_CHECK = 0


def ok(cond, msg):
    global N_CHECK
    N_CHECK += 1
    assert cond, f"CHECK {N_CHECK} FAILED: {msg}"
    print(f"  ok: {msg}")


# ── 1. MA-3: no feedback ⇒ criterion frozen (learner and filter) ──
tp = LearnerParams(alpha_t=0.5, alpha_sigma=0.05, sigma_inf=0.6,
                   q_t=0.0, q_sigma=0.0, rule="soft")
lnr = Learner([1.2], [0.4], tp, seed=0)
for k in range(30):
    lnr.step(0.5, 0, y_star=1, feedback=False)
ok(abs(lnr.t[0] - 0.4) < 1e-12, "MA-3 learner: no feedback ⇒ criterion frozen")

f = TaskFilter(np.full(50, -0.4), np.full(50, 0.2), tp, seed=0)
f.propagate(0.5, 1, feedback=False)
ok(np.allclose(f.theta, -0.4), "MA-3 filter: no feedback ⇒ criterion frozen")

# ── 2. MA-1: exact kernel reduces bit-identically at s_sd = 0 ──
tp2 = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.48,
                    q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
rng = np.random.default_rng(1)
th0, el0 = 0.3 * rng.standard_normal(300), 0.3 * rng.standard_normal(300)
fa = TaskFilter(th0, el0, tp2, seed=7, exact_kernel=False)
fb = TaskFilter(th0, el0, tp2, seed=7, exact_kernel=True)
lnr2 = Learner([1.2], [0.3], tp2, seed=3)
for k in range(40):
    s = (1 if k % 2 == 0 else -1) * 1.0
    y = lnr2.step(s, 0, int(s > 0))
    fa.step(s, y, int(s > 0), s_sd=0.0)
    fb.step(s, y, int(s > 0), s_sd=0.0)
ok(np.array_equal(fa.theta, fb.theta) and np.array_equal(fa.ell, fb.ell),
   "MA-1: exact kernel bit-identical to shipped at s_sd=0")

# ── 3. MA-1: exact kernel == audit-study reference implementation ──
from studies.study_exact_kernel_audit import CondVarTaskFilter
fa = TaskFilter(th0, el0, tp2, seed=7, exact_kernel=True)
fb = CondVarTaskFilter(th0, el0, tp2, seed=7)
lnr3 = Learner([1.2], [0.3], tp2, seed=5)
srng = np.random.default_rng(11)
diverged = False
for k in range(60):
    s = (1 if k % 2 == 0 else -1) * 0.9
    y = lnr3.step(s + 0.85 * srng.standard_normal(), 0, int(s > 0))
    fa.step(s, y, int(s > 0), s_sd=0.85)
    fb.step(s, y, int(s > 0), s_sd=0.85)
    if not (np.allclose(fa.theta, fb.theta) and np.allclose(fa.ell, fb.ell)):
        diverged = True
        break
ok(not diverged, "MA-1: TaskFilter(exact_kernel=True) == study reference")

# ── 4. MA-9: all-static filter ⇒ greedy expected reward is exactly 0 ──
from training.trainer_greedy import RewardWeights, _expected_reward
fs = TaskFilter(th0, el0, tp2, seed=7, p_static=1.0)
fs.learn[:] = 0.0                       # every particle non-learning
Q = _expected_reward(fs, np.array([0.3, -0.8, 1.2]),
                     np.array([1.0, 0.0, 1.0]),
                     RewardWeights(beta_r=0.0), s_sd=0.6)
ok(np.allclose(Q, 0.0), "MA-9: non-learning belief mass earns zero Q")

# ── 5. MA-6: post-resample MCSE never below the ESS-only formula ──
fr = TaskFilter(th0, el0, tp2, seed=7)
fr.w = fr.w * np.exp(-np.linspace(0, 12, fr.N))   # degenerate weights
fr.w /= fr.w.sum()
assert fr.maybe_resample(), "resample should trigger"
pi, mcse = fr.pass_mass(0.0)
mcse_old = np.sqrt(max(pi * (1 - pi), 0.0) / fr.ess())
ok(mcse >= mcse_old - 1e-15 and fr._n_anc < fr.N,
   f"MA-6: honest MCSE ≥ ESS-only ({mcse:.4f} ≥ {mcse_old:.4f}, "
   f"n_anc={fr._n_anc}<{fr.N})")

# ── 6. MA-2: the F28 declaration contrast — shipped certifies the static
# sub-cut learner, the mixture refuses. (Per-seed trainability at 200 trials
# is noisy — the OC-level claims live in studies/study_audit_hardening.py;
# this pins the mechanism and the declaration behavior.) ──
ell_star, sig_star = 0.534, 0.586


def _run_pair(kind, seed, n=200, ssd=0.3):
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.48,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    r = np.random.default_rng(seed)
    th0, el0 = 0.3 * r.standard_normal(200), 0.3 * r.standard_normal(200)
    ship = TaskFilter(th0, el0, tp, seed=seed)
    mix = SigmaInfMixtureFilter(th0, el0, tp, ell_inf_mean=0.765, tau=0.30,
                                J=7, p_static_stratum=0.15, ell_star=ell_star,
                                seed=seed, exact_kernel=True)
    from training.misspec_learners import static_below_cut
    lnr = (static_below_cut(sig_star, seed=seed + 1) if kind == "static"
           else Learner([1.5], [0.2], tp, seed=seed + 1))
    srng = np.random.default_rng(seed + 50)
    fired_ship = fired_mix = False
    for k in range(n):
        sig_hat = float(np.exp(-mix.mean()[1]))
        side = 1 if k % 2 == 0 else -1
        s = side * (sig_star if k % 5 == 4          # bar-anchored probe
                    else SKILL_MODE_MULTIPLIER * sig_hat)
        y = lnr.step(s + ssd * srng.standard_normal(), 0, int(s > 0))
        ship.step(s, y, int(s > 0), s_sd=ssd)
        mix.step(s, y, int(s > 0), s_sd=ssd)
        fired_ship |= ship.is_mastered(ell_star, sd_floor=0.23)
        fired_mix |= mix.is_mastered(ell_star, sd_floor=0.23)
    return fired_ship, fired_mix, mix.trainability(ell_star)


static_rows = [_run_pair("static", sd) for sd in range(3)]
well_rows = [_run_pair("well", sd) for sd in range(3)]
ok(sum(r[0] for r in static_rows) >= 2,
   f"F28 reproduced: POINT filter (exact kernel, M17 default) declares the "
   f"static sub-cut learner ({sum(r[0] for r in static_rows)}/3 seeds)")
ok(sum(r[1] for r in static_rows) == 0,
   "MA-2: mixture NEVER declares the static sub-cut learner (0/3)")
tr_s = float(np.median([r[2] for r in static_rows]))
tr_w = float(np.median([r[2] for r in well_rows]))
ok(tr_s < tr_w,
   f"MA-2: median trainability orders correctly (static {tr_s:.2f} "
   f"< well {tr_w:.2f})")

# ── 7. e-process gate: anytime type-I ≤ α and power against deficits ──
rng = np.random.default_rng(2)
fires_h0, fires_h1 = 0, 0
for rep in range(200):
    g0, g1 = EProcessGate(alpha=0.05), EProcessGate(alpha=0.05)
    for i in range(200):
        a0 = rng.uniform(0.75, 0.9)
        g0.update(rng.random() < a0, a0)          # H0: exactly at bar
        g1.update(rng.random() < a0 - 0.2, a0)    # careless-grade deficit
    fires_h0 += g0.fired
    fires_h1 += g1.fired
ok(fires_h0 <= 0.08 * 200, f"e-gate type-I {fires_h0}/200 ≤ 8% (α=5%)")
ok(fires_h1 >= 0.95 * 200, f"e-gate power vs δ=0.2 deficit: {fires_h1}/200")

# ── 8. cert-probe placement: prefers precise items at |s−t̂| ≈ σ* ──
s_grid = np.linspace(-2, 2, 401)
sc_precise = cert_probe_score(s_grid, 0.1, 0.586, 0.0, lapse=LAPSE_RATE)
best = abs(s_grid[int(np.argmax(sc_precise))])
sc_noisy = cert_probe_score(0.586, 1.0, 0.586, 0.0, lapse=LAPSE_RATE)
ok(0.6 < best < 1.0,
   f"probe optimum |s−t̂|={best:.2f} ≈ 1.35·σ* (scale-parameter optimum)")
ok(float(sc_noisy) < float(cert_probe_score(0.586, 0.1, 0.586, 0.0)),
   "probe score prefers low-s_sd items")

# ── 9. MA-4: λ-parameterized multiplier consistent with frozen constant ──
ok(abs(skill_mode_multiplier(0.025) - SKILL_MODE_MULTIPLIER) < 1e-12,
   "skill_mode_multiplier(0.025) == frozen SKILL_MODE_MULTIPLIER")
ok(1.0 < skill_mode_multiplier(0.006) < SKILL_MODE_MULTIPLIER,
   f"fitted-λ multiplier {skill_mode_multiplier(0.006):.4f} < frozen 1.0772")

# ── 10. M16 D23 wiring: TrainerPolicy probe cadence + e-gate blocking ──
from training.bank_adapter import TaskCandidates
from training.trainer_policy import TrainerPolicy, RetentionScheduler


class _FakeBank:
    def __init__(self, n=400, seed=0):
        r = np.random.default_rng(seed)
        s = np.sort(r.uniform(-2.5, 2.5, n))
        self.tc = TaskCandidates(0, np.arange(n), s,
                                 r.uniform(0.2, 0.9, n),
                                 (s > 0).astype(int), np.full(n, 0.9),
                                 np.ones(n, dtype=bool))

    def candidates(self, task, exclude_segids=None, feedback_safe=True,
                   min_margin=0.30):
        excl = exclude_segids or set()
        keep = np.where([int(sid) not in excl for sid in self.tc.seg_id])[0]
        return self.tc.subset(keep)


def _run_policy(probe_every, n=60, seed=0):
    tp = LearnerParams(alpha_t=0.2, alpha_sigma=0.06, sigma_inf=0.48,
                       q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    r = np.random.default_rng(seed)
    filt = TaskFilter(0.3 * r.standard_normal(300),
                      0.3 * r.standard_normal(300), tp, seed=seed)
    lnr = Learner([1.3], [0.2], tp, seed=seed + 1)
    pol = TrainerPolicy([filt], [0.534], [0.586], _FakeBank(),
                        seed=seed, probe_every=probe_every,
                        retention=RetentionScheduler())
    n_probe = 0
    for k in range(n):
        ch = pol.step()
        if ch is None:
            break
        if ch["info"].get("cert_probe"):
            n_probe += 1
        s_real = ch["s"] + ch["s_sd"] * r.standard_normal()
        y = lnr.step(s_real, 0, ch["y_star"], feedback=True)
        pol.record(ch, y)
    return pol, n_probe


pol_on, n_probe = _run_policy(probe_every=4)
ok(10 <= n_probe <= 15,
   f"D23 wiring: probes served on cadence ({n_probe}/60 at 1-in-4)")
ok(pol_on.mode_policies[0].egate is not None
   and pol_on.mode_policies[0].egate.n == n_probe,
   f"D23 wiring: e-gate saw exactly the probe outcomes (n={n_probe})")
pol_off, n_off = _run_policy(probe_every=None)
ok(n_off == 0 and pol_off.mode_policies[0].egate is None,
   "D23 wiring: probe_every=None ⇒ no probes, no gate (shipped path)")
# elevated gate blocks is_mastered regardless of the posterior
pol_on.mode_policies[0].egate.e_now = 1e9
ok(not pol_on.mode_policies[0].is_mastered(pol_on.filters[0]),
   "D23 wiring: elevated e-gate blocks declaration")

# ── 10b. M17 (OQ6): stationary bias band — formula vs direct recurrence ──
from training.trainer_policy import bias_stationary_sd, derived_t_star
from scipy.special import ndtr


def _sim_stationary_sd(alpha_t, q_t, sigma, item_s_sd, n=40_000, seed=0):
    """Direct recurrence sim under the BOUNDARY-ANCHORED mirror-paired
    stream (D15 skill mode): s̄ = ±m·σ alternating, y* = sign(s̄)."""
    r = np.random.default_rng(seed)
    t, lam, side = 0.0, 0.025, 1
    a = SKILL_MODE_MULTIPLIER * sigma
    ts = np.empty(n)
    for k in range(n):
        s_real = side * a + item_s_sd * r.standard_normal()
        ystar = 1 if side > 0 else 0
        p = lam + (1 - 2 * lam) * ndtr((s_real - t) / sigma)
        t = t + alpha_t * (p - ystar) + q_t * r.standard_normal()
        side = -side                                    # strict alternation
        ts[k] = t
    tail = ts[n // 4:]
    return float(tail.std()), float(tail.mean())


for (a, q, sig, ssd) in [(0.2, 0.04, 0.6, 0.3), (0.35, 0.05, 0.9, 0.3),
                         (0.1, 0.04, 0.5, 0.5)]:
    emp_sd, emp_mean = _sim_stationary_sd(a, q, sig, ssd)
    th_sd = bias_stationary_sd(a, q, sig, item_s_sd=ssd)
    ok(abs(th_sd - emp_sd) / emp_sd < 0.20,
       f"0-bias floor: formula {th_sd:.3f} vs sim {emp_sd:.3f} "
       f"(α={a}, q={q}, σ={sig})")
    ok(abs(emp_mean) < 4 * emp_sd / np.sqrt(30_000 / 50),
       f"0-bias mean: |E[t∞]|={abs(emp_mean):.4f} ≈ 0 under anchored serving")
ok(0.08 <= derived_t_star(0.2, 0.04, 0.6) <= 0.45,
   f"derived t* band in product bounds ({derived_t_star(0.2, 0.04, 0.6):.3f})")

# ── 11. M16: filter-evidence likelihood separates gross dynamics error ──
from training.dynamics_fit import filter_evidence

tp3 = LearnerParams(alpha_t=0.3, alpha_sigma=0.08, sigma_inf=0.45,
                    q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
lnr3 = Learner([1.4], [0.6], tp3, seed=2)
srng = np.random.default_rng(3)
lg = []
for k in range(150):
    sig = float(lnr3.sigma[0])
    s = (1 if k % 2 == 0 else -1) * SKILL_MODE_MULTIPLIER * sig
    ssd = float(srng.uniform(0.3, 0.8))
    y = lnr3.step(s + ssd * srng.standard_normal(), 0, int(s > 0))
    lg.append((s, y, int(s > 0), ssd))
ev_true = np.mean([filter_evidence(lg, tp3, seed=sd, prior_sd=0.45)
                   for sd in (42, 43, 44)])
from dataclasses import replace as _rp
ev_bad = np.mean([filter_evidence(lg, _rp(tp3, sigma_inf=1.4), seed=sd,
                                  prior_sd=0.45) for sd in (42, 43, 44)])
# NB the margin is SMALL by nature (M16 finding: one-step prequential
# evidence is nearly flat in the dynamics params because the filter's
# state-tracking self-corrects — ~0.005 nats/trial even for gross error);
# the check pins the correct DIRECTION deterministically (CRN seeds).
ok(ev_true > ev_bad + 0.3,
   f"M16 fitter: evidence orders true > no-learning dynamics "
   f"({ev_true:.1f} vs {ev_bad:.1f} nats — flat surface is the finding)")

print(f"\nM15 AUDIT-FIX SMOKE TEST PASSED — {N_CHECK} checks.")
