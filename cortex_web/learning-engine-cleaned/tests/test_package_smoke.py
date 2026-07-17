"""Vendored learning-engine smoke gates (Phase L2 drop, contract v1.1).

Fast, MCMC-free regressions for the properties the host repo relies on:

  g1  the session-time import surface is JAX-free (the host bans JAX at
      runtime module boundaries; calibration exports resolve lazily)
  g2  seeded engine sessions repeat BITWISE under single-thread BLAS
      (the host's reproducibility contract)
  g3  extensibility: an unseen domain cold-starts from the shipped
      artifact's population hyperprior via registry + data only
  g4  replay seeding (contract §2a): raw-stream replay works, floors
      condition on skill when the artifact carries `floor_joint`, and
      posterior-XOR-replay double counting raises
  g5  the vendored contract declares v1.1

Run:  python -m pytest learning-engine-cleaned/tests -q
"""
import json
import os
import subprocess
import sys

# Single-thread BLAS before numpy lands anywhere in this process — the same
# self-pinning discipline as scripts/session_controller.py.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

import numpy as np  # noqa: E402

from learning_engine import MSBelief, MixedBelief, Registry  # noqa: E402

ARTIFACT = os.path.join(PKG_ROOT, "artifacts", "nway_dynamics_v1.json")

M = 3
SIG = [f"d{m}" for m in range(M)]


def _art(joint=True):
    art = dict(alpha_t=[0.1] * M, alpha_s=[0.01] * M, lam=0.02, q_t=0.0,
               q_s=0.006,
               state_prior=dict(mu_t0=[0.0] * M, tau_t0=[0.4] * M,
                                mu_u0=[1.0] * M, tau_u0=[0.4] * M,
                                gamma=[0.5] * M),
               floor_prior=([0.3] * M, [0.46] * M))
    if joint:
        art["floor_joint"] = dict(slope=[0.6] * M, intercept=[-0.3] * M,
                                  resid_sd=[0.25] * M)
    return art


def test_g1_session_import_is_jax_free():
    """Importing the session-time surface (beliefs, adapter) must not pull
    JAX/NumPyro into the process — run in a clean subprocess."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from learning_engine import MSBelief, MixedBelief, Registry\n"
        "from adapter.le_adapter import seed_by_replay, seed_from_exam_cloud\n"
        "assert 'jax' not in sys.modules, 'JAX leaked into session imports'\n"
        "assert 'numpyro' not in sys.modules\n"
        "print('ok')\n" % PKG_ROOT
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=PKG_ROOT)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def _mixed_session(seed):
    art = json.load(open(ARTIFACT))["artifact"]
    reg = Registry([(c, "g") for c in art["codes"]])
    bel = MixedBelief(art, reg, N=200, rng=np.random.default_rng(seed))
    rng = np.random.default_rng(1000 + seed)
    G = len(art["codes"])
    for _ in range(25):
        gold = int(rng.integers(G))
        s = rng.normal(-1.2, 0.4, G)
        s[gold] = rng.normal(0.2, 0.5)
        item = dict(group="g", s=s, gold=gold)
        bel.update(item, int(rng.integers(G)))
    return bel


def test_g2_bitwise_determinism():
    a, b = _mixed_session(7), _mixed_session(7)
    for nm in ("t", "u", "u_inf", "w"):
        assert np.array_equal(getattr(a, nm), getattr(b, nm)), nm


def test_g3_cold_start_from_hyperprior():
    art = json.load(open(ARTIFACT))["artifact"]
    assert "hyper" in art, "shipped artifact must carry the hyperprior block"
    reg = Registry([(c, "g") for c in art["codes"]] + [("newdom", "binary")])
    bel = MixedBelief(art, reg, N=150, rng=np.random.default_rng(3))
    j = reg.index["newdom"]
    assert np.isfinite(bel.t[:, j]).all() and np.isfinite(bel.u[:, j]).all()
    assert (bel.u_inf[:, j] <= bel.u[:, j]).all()


def test_g4_replay_seeding_and_guards():
    rng = np.random.default_rng(11)
    pairs = []
    for _ in range(30):
        gold = int(rng.integers(M))
        s = rng.normal(-1.3, 0.5, M)
        s[gold] = rng.normal(-0.1, 0.6)
        pairs.append((s, int(rng.integers(M))))
    b = MSBelief(_art(), SIG, N=800, rng=np.random.default_rng(12))
    b.seed_from_test_replay(pairs)
    b._shrink(200)
    assert b.seed_diag["n_trials"] == 30 and b.seed_diag["n_unique"] > 20
    # floors condition on skill under floor_joint (clip-tolerant check)
    res = b.u_inf - (-0.3 + 0.6 * b.u)
    inner = res[b.u_inf < b.u - 1e-3]
    assert abs(float(inner.std()) - 0.25) < 0.08
    # posterior XOR replay: double counting must raise
    mock = dict(dimension_names=SIG,
                final_summary=dict(offset_mean=[0.0] * M,
                                   offset_variance=[0.05] * M,
                                   log_sensitivity_mean=[-1.0] * M,
                                   log_sensitivity_variance=[0.05] * M))
    try:
        b.seed_from_testing(mock)
        raise AssertionError("double seeding did not raise")
    except ValueError:
        pass


def test_g5_contract_declares_v11():
    doc = open(os.path.join(PKG_ROOT, "docs", "handoff_contract.md")).read()
    assert "Version 1.1" in doc and "2a" in doc


def test_g6_label_schedule_not_gameable():
    """2026-07-17 live-human finding: strict ± alternation let the first
    real participant learn the label pattern in ~10 items and score 97%
    without reading the media. The served label sequence must carry no
    strict alternation, keep per-domain imbalance bounded (the D12/D50
    long-run balance requirement), and stay seed-deterministic."""
    from adapter.le_adapter import LETrainerPolicy
    art = _art(joint=False)
    art_b = dict(art, codes=["b1"],
                 alpha_t=[0.1], alpha_s=[0.01],
                 state_prior={k: [v[0]]
                              for k, v in art["state_prior"].items()},
                 floor_prior=([art["floor_prior"][0][0]],
                              [art["floor_prior"][1][0]]))
    reg = Registry([("b1", "binary")])
    s_vals = np.concatenate([np.linspace(0.2, 2.0, 50),
                             -np.linspace(0.2, 2.0, 50)])
    cand = dict(seg_id=np.arange(100), s=s_vals,
                s_sd=np.zeros(100),
                y_star=(s_vals > 0).astype(int))

    def run(seed):
        bel = MixedBelief(art_b, reg, N=150,
                          rng=np.random.default_rng(seed))
        pol = LETrainerPolicy(bel, reg, {"b1": 0.0}, lambda c: cand,
                              alpha=0.0, Z=2.0, sd_floor=0.23,
                              rng=np.random.default_rng(seed + 1))
        ys = []
        for _ in range(40):
            ch = pol.step()
            if ch is None:
                break
            ys.append(int(ch["y_star"]))
            pol.record(ch, ch["y_star"])   # always-correct responder
        return ys

    ys = run(5)
    assert len(ys) == 40
    # no strict alternation: at least one same-label adjacent pair
    assert any(a == b for a, b in zip(ys, ys[1:])), \
        "label schedule is a strict alternation — gameable"
    # v3 (D56): pure iid coin — no boundary-forced alternation stretches
    # (the v2 hard bound produced 14-long half-predictable 1010 runs
    # live). For this seed: short alternation runs, moderate rate, and
    # only expectation-level balance (no hard bound).
    runs, cur = [], 1
    for a, b in zip(ys, ys[1:]):
        cur = cur + 1 if a != b else 1
        runs.append(cur)
    assert max(runs) < 10, f"alternation run {max(runs)} — forced pattern?"
    n_alt = sum(1 for a, b in zip(ys, ys[1:]) if a != b)
    assert n_alt / (len(ys) - 1) < 0.75
    bal = np.cumsum(np.where(np.array(ys) == 1, 1, -1))
    assert abs(int(bal[-1])) <= 12          # E-balance sanity, not a bound
    # seed-determinism preserved
    assert ys == run(5)
