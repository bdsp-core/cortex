"""M23 smoke test — regime-shift mechanisms (F80–F83).

Checks:
  1–3   bit-identity of the OFF defaults: TaskFilter(jump_eps=0) and
        SigmaInfMixtureFilter(forget=1.0) reproduce M22 trajectories exactly;
        boundary_shift(w_share=0) leaves weights untouched.
  4–6   jump-kernel mechanics: innovation-variance inflation matches the
        analytic factor 1 + ε(κ²−1); boundary_jump adds ≈ eps·sd² variance
        on ℓ and scales the θ shock; static strata don't jump.
  7–8   DMA forgetting math (ω^γ) and fixed-share math ((1−a)ω + a·prior).
  9–10  contact e-process: anytime validity under the chance-band null
        (sup crossing rate ≤ α, mean E ≤ 1) and power on a competent
        responder.
  11    RT-floor guard rule fires on a USER-D-shaped RT collapse and stays
        silent on an engaged-session RT profile.
  12    sandbox protocol integration: a robot session with the M23 gates
        enabled runs end-to-end and logs contact_e telemetry.

Run: python3 -m tests.test_m23
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from training.learner_sim import LearnerParams
from training.mixture_filter import SigmaInfMixtureFilter
from training.training_filter import TaskFilter

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    assert cond, f"FAILED: {msg}"
    CHECKS += 1
    print(f"  ok: {msg}")


def _params(**kw):
    d = dict(alpha_t=0.097, alpha_sigma=0.06, sigma_inf=0.6,
             q_t=0.04, q_sigma=0.02, rho=0.6, rule="soft")
    d.update(kw)
    return LearnerParams(**d)


def _run_filter(f, seed=5, n=60):
    rng = np.random.default_rng(seed)
    for k in range(n):
        s = float(rng.uniform(-1.2, 1.2))
        y_star = int(s > 0)
        y = int(rng.random() < 0.8) if y_star else int(rng.random() < 0.25)
        f.step(s, y, y_star, s_sd=0.06)
    return f


def main():
    rng = np.random.default_rng(1)
    th0 = 0.4 * rng.standard_normal(300)
    el0 = 0.4 * rng.standard_normal(300)

    # ── 1: TaskFilter jump_eps=0 is bit-identical to the M22 signature ──
    fa = _run_filter(TaskFilter(th0, el0, _params(), seed=7))
    fb = _run_filter(TaskFilter(th0, el0, _params(), seed=7, jump_eps=0.0,
                                jump_kappa=8.0))
    ok(np.array_equal(fa.theta, fb.theta) and np.array_equal(fa.ell, fb.ell)
       and np.array_equal(fa.w, fb.w),
       "jump_eps=0 TaskFilter is bit-identical (θ/ℓ/w)")

    # ── 2: mixture forget=1.0 bit-identical ──
    ma = _run_filter(SigmaInfMixtureFilter(
        th0, el0, _params(), ell_inf_mean=0.5, ell_star=0.4, seed=3))
    mb = _run_filter(SigmaInfMixtureFilter(
        th0, el0, _params(), ell_inf_mean=0.5, ell_star=0.4, seed=3,
        forget=1.0))
    ok(np.array_equal(ma.log_w, mb.log_w)
       and np.array_equal(ma.strata[0].ell, mb.strata[0].ell),
       "forget=1.0 mixture is bit-identical (log_w, strata)")

    # ── 3: boundary_shift(0,0,w_share=0) is a no-op ──
    w_before = mb.weights().copy()
    ell_before = mb.strata[2].ell.copy()
    mb.boundary_shift(0.0, 0.0, w_share=0.0)
    ok(np.array_equal(mb.weights(), w_before)
       and np.array_equal(mb.strata[2].ell, ell_before),
       "boundary_shift with zero hazard/share is a no-op")

    # ── 4: per-trial jump variance matches 1 + ε(κ²−1) analytically ──
    eps, kap, q = 0.05, 6.0, 0.02
    draws = []
    f = TaskFilter(np.zeros(200_000), np.zeros(200_000),
                   _params(alpha_t=0.0, alpha_sigma=0.0, q_t=0.0, q_sigma=q),
                   seed=11, jump_eps=eps, jump_kappa=kap)
    f.propagate(0.3, 1, True, y=1, s_sd=0.0)
    v = np.var(-f.ell)                       # log σ innovations only
    factor = v / q ** 2
    want = 1 + eps * (kap ** 2 - 1)
    ok(abs(factor - want) / want < 0.05,
       f"per-trial jump variance factor {factor:.2f} ≈ analytic {want:.2f}")

    # ── 5: boundary_jump adds ≈ eps·sd² on ℓ; θ shock scales ──
    f2 = TaskFilter(np.zeros(200_000), np.zeros(200_000), _params(), seed=13)
    f2.boundary_jump(0.1, 0.75, theta_scale=0.33)
    vl, vt = np.var(f2.ell), np.var(f2.theta)
    ok(abs(vl - 0.1 * 0.75 ** 2) / (0.1 * 0.75 ** 2) < 0.05,
       f"boundary_jump ℓ-variance {vl:.4f} ≈ eps·sd² {0.1 * 0.75**2:.4f}")
    ok(abs(vt / vl - 0.33 ** 2) < 0.03,
       f"θ shock variance ratio {vt / vl:.3f} ≈ theta_scale² {0.33**2:.3f}")

    # ── 6: static stratum does not jump ──
    fs_ = TaskFilter(np.zeros(100), np.zeros(100), _params(rule="static"),
                     seed=17)
    fs_.boundary_jump(1.0, 1.0)
    ok(np.all(fs_.ell == 0.0), "static-rule filter ignores boundary_jump")

    # ── 7: DMA forgetting implements ω^γ (normalized) ──
    m = SigmaInfMixtureFilter(th0, el0, _params(), ell_inf_mean=0.5,
                              ell_star=0.4, seed=19, forget=0.9)
    lw = np.full(len(m.log_w), -25.0)
    lw[:3] = np.log([0.5, 0.3, 0.2])
    m.log_w = lw
    w0 = m.weights().copy()
    m.log_w = m.forget * m.log_w             # the exact step() forgetting op
    w1 = m.weights()
    wg = w0 ** 0.9
    wg /= wg.sum()
    ok(np.allclose(w1, wg, atol=1e-12),
       "forget step equals normalized ω^γ (log-gap compression by γ)")

    # ── 8: fixed-share math ──
    m2 = SigmaInfMixtureFilter(th0, el0, _params(), ell_inf_mean=0.5,
                               ell_star=0.4, seed=23)
    _run_filter(m2, seed=29, n=40)
    w_run = m2.weights().copy()
    prior = np.exp(m2._log_prior_w - np.logaddexp.reduce(m2._log_prior_w))
    m2.boundary_shift(0.0, 0.0, w_share=0.25)
    want_w = 0.75 * w_run + 0.25 * prior
    ok(np.allclose(m2.weights(), want_w / want_w.sum(), atol=1e-10),
       "boundary_shift mixes weights as (1−a)·ω + a·prior")

    # ── 9: contact e-process anytime validity under the chance-band null ──
    from sandbox.contact import ContactMonitor
    crossings, means = 0, []
    for sd in range(400):
        r = np.random.default_rng(sd)
        cm = ContactMonitor()
        mu = 0.45 + 0.1 * r.random()          # anywhere in the chance band
        logE_max = -np.inf
        for k in range(60):
            cm.update(int(r.random() < mu))
            logE_max = max(logE_max, cm.log_e)
        crossings += int(logE_max >= np.log(20.0))
        means.append(np.exp(cm.log_e))
    ok(crossings / 400 <= 0.05 + 0.02,
       f"null sup-crossing rate {crossings}/400 ≤ α=0.05 (+MC slack)")
    ok(np.mean(means) < 1.3,
       f"null mean E_T {np.mean(means):.2f} consistent with supermartingale")

    # ── 10: contact power ──
    lat = []
    for sd in range(50):
        r = np.random.default_rng(1000 + sd)
        cm = ContactMonitor()
        t = None
        for k in range(60):
            cm.update(int(r.random() < 0.85))
            if t is None and cm.contact:
                t = k + 1
        lat.append(t if t is not None else 61)
    ok(np.median(lat) <= 25,
       f"contact detected for an 85%-correct responder "
       f"(median {np.median(lat):.0f} ≤ 25 trials)")

    # ── 11: RT-floor guard ──
    from sandbox.contact import rt_floor_breach
    collapse = [3000] * 30 + [260, 240, 300, 220, 250]      # USER-D shape
    engaged = list(np.linspace(4000, 1100, 40))             # fast but engaged
    ok(rt_floor_breach(collapse, floor_ms=500, k=3)
       and not rt_floor_breach(engaged, floor_ms=500, k=3),
       "RT floor fires on a D-shaped collapse, silent on engaged profile")

    # ── 12: sandbox integration (robot session, M23 gates on) ──
    import importlib
    import os
    import shutil
    import tempfile
    from sandbox import config as C
    tmp = tempfile.mkdtemp(prefix="m23_sbx_")
    try:
        C.set_user("M23TEST")
        # redirect the test user's namespace into tmp
        C.STATE_DIR = os.path.join(tmp, "state")
        C.LOG_DIR = os.path.join(tmp, "logs")
        C.TRIALS_JSONL = os.path.join(C.LOG_DIR, "trials.jsonl")
        C.SESSIONS_JSONL = os.path.join(C.LOG_DIR, "sessions.jsonl")
        from sandbox import state_io
        state_io.STATE_NPZ = os.path.join(C.STATE_DIR, "state.npz")
        state_io.STATE_JSON = os.path.join(C.STATE_DIR, "state.json")
        import sandbox.protocol as P
        importlib.reload(P)

        def robot(meta):
            return int(meta["s_real"] > 0), 900.0, False

        sess = P.Session(seed=42)
        n, events = sess.run(robot)
        ok(n > 0, f"robot session served {n} trials with M23 gates enabled")
        import json as _json
        rec = _json.loads(open(C.TRIALS_JSONL).readline())
        ok("contact" in rec and "log_e" in rec["contact"],
           "trial telemetry carries the shadow contact statistic")
        # second session exercises the boundary_shift path
        sess2 = P.Session(seed=43)
        n2, _ = sess2.run(robot)
        ok(n2 > 0, "second session (boundary_shift path) runs clean")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        C.set_user(None)

    print(f"\nM23 SMOKE TEST PASSED — {CHECKS} checks.")


if __name__ == "__main__":
    main()
