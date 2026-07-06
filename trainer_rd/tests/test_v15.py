"""M11 smoke test — v15 staged instrument. Run: python3 -m tests.test_v15

Verifies:
  1. cert_config v15 block complete: 7 tasks, full-precision ell*,
     sigma* == exp(−ell*) to 1e-15, v14/v13 blocks untouched;
  2. instrument('v14'/'v15') constants: cuts, particle counts, expert
     ceilings; every v15 ceiling clears its cut (LT1 feasible);
  3. corr_t swap: v15 t-block == npy Corr_t (≠ Corr_l), v14 reuses Corr_l;
     both PD with unit diagonal (R-gate denominator assumption);
  4. realistic-bias prior: mu_t matches spec, Sigma_t matches npy, draws
     reproduce mu within MC error, +2 clip on domain2..7, zero-bias arm 0;
  5. eval-harness opt-in plumbing: K=3 micro-eval runs under both
     instruments and a skilled rater's posterior clears the v15 cuts;
     defaults (no kwargs) reproduce the shipped path.
"""
import numpy as np
import yaml

import engine.instrument_v15 as iv
from engine.instrument_v15 import instrument, draw_examinee_theta, load_bias_prior

checks = 0


def ok(cond, msg):
    global checks
    assert cond, f"FAIL: {msg}"
    checks += 1
    print(f"  ok: {msg}")


# ── 1. YAML block ──
with open(iv._CONFIG) as fh:
    cfg = yaml.safe_load(fh)
b15 = cfg["ell_star_unified_v15"]
ok(len(b15["tasks"]) == 7 and b15["status"] == "complete",
   "v15 block complete (7 tasks)")
for key, v in b15["tasks"].items():
    assert abs(v["sigma_star"] - np.exp(-v["ell_star"])) < 1e-15, key
    assert len(str(v["ell_star"]).split(".")[1]) > 10, f"{key} not full precision"
ok(True, "sigma* == exp(−ell*) to 1e-15, full-precision, all 7 tasks")
ok(len(cfg["ell_star_unified_v14"]["tasks"]) == 7
   and len(cfg["ell_star_unified_v13"]["tasks"]) == 7,
   "v14 and v13 blocks untouched (7 tasks each)")

# ── 2. instrument constants ──
v14, v15 = instrument("v14"), instrument("v15")
ok(v14.n_particles == 600 and v15.n_particles == 1200,
   "particle counts: frozen 600 / staged 1200")
ok(np.all(v15.ell_star < v14.ell_star), "all 7 v15 cuts below v14")
ok(abs(v15.ell_star[2] - 0.3059231418723579) < 1e-16,
   "domain3 v15 ell* exact (16-digit)")
ok(np.all(v15.expert_ell > v15.ell_star) and np.all(v15.sigma_inf < v15.sigma_star),
   "v15 credentialed ceilings clear every cut (LT1 feasible)")
gap14 = v14.expert_ell - v14.ell_star
gap15 = v15.expert_ell - v15.ell_star
ok(np.all(gap15 > gap14 - 1e-12) and abs(gap15[0] - 0.1678) < 0.001,
   f"ceiling-to-cut gap widens on all tasks (domain1 {gap14[0]:.3f}→{gap15[0]:.3f})")

# ── 3. prior matrices ──
cov = iv.load_fitted_cov()
ok(np.allclose(v15.Sigma_t_prior, cov["Corr_t"])
   and not np.allclose(v15.Sigma_t_prior, cov["Corr_l"]),
   "v15 t-block prior is Corr_t (the swap is real)")
ok(np.allclose(v14.Sigma_t_prior, cov["Corr_l"])
   and np.allclose(v14.Sigma_l_prior, v15.Sigma_l_prior),
   "v14 reuses Corr_l for both blocks; l-block shared (unchanged)")
for nm in ("Corr_l", "Corr_t"):
    M = cov[nm]
    ok(np.allclose(np.diag(M), 1.0) and np.linalg.eigvalsh(M).min() > 0,
       f"{nm} unit-diagonal and positive-definite")

# ── 4. realistic-bias prior ──
mu, St = load_bias_prior()
ok(abs(mu[2] - (-1.727597)) < 1e-9 and abs(mu[0] - 0.484037) < 1e-9,
   "mu_t matches spec (domain1 +0.484, domain3 −1.728)")
ok(np.allclose(St, cov["Sigma_t"]), "bias Sigma_t == npy 'Sigma_t'")
rng = np.random.default_rng(0)
th = draw_examinee_theta(rng, 6000)
ok(np.abs(th.mean(0) - mu).max() < 0.05, "6000 draws reproduce mu_t (<0.05)")
ok(th[:, 1:].max() <= iv.MU_T_CLIP + 1e-12, "+2 clip respected (domain2..7)")
ok(not np.any(draw_examinee_theta(rng, 3, zero_bias=True)),
   "zero-bias arm is exactly 0")
sub = v15.for_tasks([1, 4])
ok(sub["Sigma_t"].shape == (2, 2)
   and abs(sub["Sigma_t"][0, 1] - cov["Corr_t"][1, 4]) < 1e-15,
   "for_tasks subsets the prior matrices correctly")

# ── 5. eval plumbing (micro K=3 sessions; budget-capped) ──
from training.bank_adapter import BankAdapter
from training.pipeline_demo import _run_eval

bank = BankAdapter()
tasks = (1, 2, 3)
for ver in ("v14", "v15"):
    ins = instrument(ver).for_tasks(tasks)
    st, verd, diag, served, n = _run_eval(
        np.zeros(3), np.array(ins["ell_star"]) + 0.8, bank, tasks=tasks,
        session_id=f"t-{ver}", n_part=200, max_q=90, seed=5,
        ell_star_vec=ins["ell_star"], Sigma_l=ins["Sigma_l"],
        Sigma_t=ins["Sigma_t"])
    ml = [float((st["w"] * st["l"][:, k]).sum()) for k in range(3)]
    ok(len(verd) == 3 and all(m > c for m, c in zip(ml, ins["ell_star"])),
       f"{ver} micro-eval runs; skilled rater's ℓ̂ clears all cuts")
st, verd, _, _, _ = _run_eval(np.zeros(3),
                              np.array([0.5, 0.5, 0.5]), bank, tasks=tasks,
                              session_id="default", n_part=150, max_q=40, seed=1)
ok(len(verd) == 3, "no-kwargs default path still runs (shipped behavior)")

print(f"\nV15 SMOKE TEST PASSED — {checks} checks.")
