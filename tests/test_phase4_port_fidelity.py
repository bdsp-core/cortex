"""Phase 4 — deployment port-fidelity + layered-change regression.

Two-step Phase-4 gate. STEP 1 (4.1–4.2, PI-faithful mode): prove the
PATH-ONLY port + config externalisation introduced ZERO behavioural
change — fast invariants + the Phase-4.2 drift-guard. (The original
slow `test_port_reproduces_pi_baseline_exactly` lived here; it was
DELETED at 4.3 because it fails BY DESIGN once the likelihood is
hardened — its purpose, proving port fidelity, was fulfilled and
committed at 4.1/4.2.)

STEP 2 (4.3+): each intended change is layered and its delta vs the
pristine PI baseline is signed off. 4.3 = λ-lapse hardening: the gate
is lapse-math correctness (λ single-sourced from engine `core`; exact
λ=0 reduction to the PI bare-probit IRLS; p∈[λ,1−λ]) + a CONTROLLED,
population-matched, signed-off delta attribution
(`deployment/phase4_3_hardening_delta.json`, produced by
`pipeline/deployment_delta/phase4_3_lapse_delta.py`).

  * fast/durable: zero `/Users/` paths in deployment/; import/shape;
    Config defaults; 4.2 YAML drift-guard + strictness; 4.3 lapse-math
    + determinism; 4.3 signed-off delta invariants (skip-if-absent).
  * slow: 4.3 live directional-signature regression on a small slice.

Run all:   pytest -m "" tests/test_phase4_port_fidelity.py -q
Skip slow: pytest tests/test_phase4_port_fidelity.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO / "deployment"
BASELINE = REPO / "data" / "deployment_prior" / "sim" / "candidates.csv"
TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda"]


# ── fast / durable ──────────────────────────────────────────────────────────
def test_no_absolute_user_paths_in_deployment():
    """Plan Phase 4 step 2: zero hardcoded /Users/ paths survive the
    port (the PI source hardcoded /Users/mwestover/...)."""
    offenders = []
    for py in sorted(DEPLOY_DIR.glob("*.py")):
        txt = py.read_text()
        for i, line in enumerate(txt.splitlines(), 1):
            if "/Users/" in line:
                offenders.append(f"{py.name}:{i}: {line.strip()}")
    assert not offenders, "absolute /Users/ paths remain:\n" + "\n".join(
        offenders)


def test_deployment_imports_shape_and_config_defaults():
    import engine_paths
    assert Path(engine_paths.DEPLOYMENT_PRIOR).is_dir()
    from deployment import simulate_test as st
    assert st.TASKS == TASKS
    assert st.K == 6 and st.DIM == 12
    c = st.TestConfig()
    # the shipping contract — defaults must match PI EXACTLY
    assert (c.N_min, c.N_max, c.N_min_per_task, c.N_max_per_task,
            c.pass_p, c.fail_p) == (60, 500, 10, 120, 0.95, 0.05)


# ── Phase 4.4-A: K-agnostic runtime + drift-guard ───────────────────────────
def test_deployment_tasks_drift_guard():
    """The module TASKS/K/DIM defaults must equal what the AUTHORITATIVE
    Σ-slot-name parser returns from the frozen artifact (4.2-style
    canonical-default vs artifact drift-guard). Currently K=6; after the
    Phase-4.6 re-freeze this becomes the 7-list and the default updates
    with it — the guard prevents silent drift either way."""
    import engine_paths
    import pandas as pd
    from deployment import simulate_test as st
    parsed = st.deployment_task_names()
    assert parsed == st.TASKS, (
        f"Σ-parsed tasks {parsed} != module default {st.TASKS}")
    assert st.K == len(parsed) and st.DIM == 2 * st.K
    Sig = pd.read_csv(
        Path(engine_paths.DEPLOYMENT_PRIOR) / "Sigma.csv", index_col=0)
    assert Sig.shape == (st.DIM, st.DIM)            # Σ ⇔ K consistent
    # ell_thresholds + case_bank cover exactly the parsed tasks
    S, bank, ell = st.load_deployment(uniform_ell_star=None)
    assert S.shape == (2 * len(parsed), 2 * len(parsed))
    assert len(ell) == len(parsed) and len(bank) == len(parsed)


def test_simulate_candidate_is_K_agnostic():
    """The engine derives K/DIM from Σ.shape — feed a synthetic K=2
    deployment and assert the state scales (proves it will accept the
    K=7 artifact at 4.6 without code change)."""
    import numpy as np
    from deployment import simulate_test as st
    K2 = 2
    Sigma = np.eye(2 * K2) * 0.5 + 0.1            # PD 4×4
    ell = np.zeros(K2)
    bank = {ki: pd.DataFrame({"seg_id": range(50),
                              "s_mean": np.linspace(-2, 2, 50),
                              "s_sd": np.zeros(50)}) for ki in range(K2)}
    cfg = st.TestConfig.from_yaml()
    theta = np.zeros(2 * K2)
    stt = st.simulate_candidate(theta, Sigma, ell, bank, cfg,
                                np.random.default_rng(0))
    assert stt.mu.shape == (2 * K2,)
    assert len(stt.decision) == K2
    assert stt.n_per_task.shape == (K2,)


# ── Phase 4.2: shipping-contract YAML drift-guard + strictness ──────────────
def test_deployment_config_yaml_equals_canonical_defaults():
    """The Phase-4.2 gate, made explicit and cheap: the externalised
    deployment_config.yaml MUST equal the canonical in-code PI defaults
    field-for-field. This is what guarantees the runtime (which now
    reads the YAML via TestConfig.from_yaml) produces sim output
    BIT-IDENTICAL to the Phase-4.1 PI-faithful baseline."""
    import engine_paths
    from deployment import simulate_test as st
    assert Path(engine_paths.DEPLOYMENT_CONFIG).is_file()
    canon = st.TestConfig()                 # hardcoded PI defaults
    contract = st.TestConfig.from_yaml()    # the shipping YAML
    assert contract == canon, (
        f"deployment_config.yaml drifted from canonical defaults: "
        f"{contract} != {canon} — would change clinical decisions")
    # exact values pinned (defence in depth vs both drifting together)
    assert (contract.N_min, contract.N_max, contract.N_min_per_task,
            contract.N_max_per_task, contract.pass_p, contract.fail_p
            ) == (60, 500, 10, 120, 0.95, 0.05)


def test_from_yaml_is_strict(tmp_path):
    """Clinical contract: from_yaml rejects unknown/missing keys and
    wrong types (no silent fallback for a pass/fail rule)."""
    import yaml as _yaml
    from deployment import simulate_test as st
    good = dict(N_min=60, N_max=500, N_min_per_task=10,
                N_max_per_task=120, pass_p=0.95, fail_p=0.05)

    ok = tmp_path / "ok.yaml"
    ok.write_text(_yaml.safe_dump(good))
    assert st.TestConfig.from_yaml(ok) == st.TestConfig()

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(_yaml.safe_dump({**good, "surprise": 1}))
    with pytest.raises(ValueError):
        st.TestConfig.from_yaml(unknown)

    missing = tmp_path / "missing.yaml"
    drop = dict(good); drop.pop("fail_p")
    missing.write_text(_yaml.safe_dump(drop))
    with pytest.raises(ValueError):
        st.TestConfig.from_yaml(missing)

    badtype = tmp_path / "badtype.yaml"
    badtype.write_text(_yaml.safe_dump({**good, "N_max": "lots"}))
    with pytest.raises(TypeError):
        st.TestConfig.from_yaml(badtype)

    boolint = tmp_path / "boolint.yaml"
    boolint.write_text(_yaml.safe_dump({**good, "N_min": True}))
    with pytest.raises(TypeError):
        st.TestConfig.from_yaml(boolint)


def test_baseline_present_and_well_formed():
    assert BASELINE.exists(), "carried PI deployment sim baseline missing"
    b = pd.read_csv(BASELINE)
    assert len(b) == 200
    for t in TASKS:
        assert f"decision_{t}" in b.columns
        assert set(b[f"decision_{t}"]).issubset({"pass", "fail", "refer"})


# ── Phase 4.3: λ-lapse hardening. The 4.1/4.2 exact-reproduction slow
#    test (`test_port_reproduces_pi_baseline_exactly`) was DELETED here —
#    it fails BY DESIGN once the likelihood is hardened (the two-step
#    gate's intended supersession; port fidelity was already proven +
#    committed at 4.1/4.2). The gate now = lapse-math correctness + a
#    signed-off, population-controlled delta attribution. ──────────────
DELTA = REPO / "deployment" / "phase4_3_hardening_delta.json"


def test_lapse_likelihood_one_definition_and_correct():
    """λ literally single-sourced from engine `core`; p floored to
    [λ,1−λ]; the lapse IRLS is the EXACT generalization of the PI
    bare-probit IRLS (the documented λ=0 reduction identity)."""
    import sys as _sys
    import numpy as _np
    from scipy.stats import norm as _N
    from scipy.special import log_ndtr as _lnd
    if str(REPO / "engine") not in _sys.path:
        _sys.path.insert(0, str(REPO / "engine"))
    import core
    from deployment import simulate_test as st
    assert st.LAPSE_RATE is core.LAPSE_RATE          # literal single source
    eta = _np.array([-8., -3., -0.7, 0.0, 0.4, 2.5, 8.])
    ec = _np.clip(eta, -6.0, 6.0)
    p, pp = st._lapse_components(ec)
    assert p.min() >= st.LAPSE_RATE - 1e-12
    assert p.max() <= 1.0 - st.LAPSE_RATE + 1e-12
    # λ=0 reduction identity: lapse formula ≡ PI bare-probit IRLS
    Y = _np.array([0, 1, 0, 1, 1, 0, 1.])
    Phi = _np.exp(_lnd(ec)); phi = _N.pdf(ec)
    w_pi = phi ** 2 / (Phi * (1 - Phi))
    z_pi = ec + (Y - Phi) / _np.clip(phi, 1e-12, None)
    o = 1.0 - 2.0 * 0.0
    p0, pp0 = 0.0 + o * Phi, o * phi               # _lapse_components @ λ=0
    assert _np.allclose(pp0 ** 2 / (p0 * (1 - p0)), w_pi, atol=1e-12)
    assert _np.allclose(ec + (Y - p0) / _np.clip(pp0, 1e-12, None),
                        z_pi, atol=1e-12)


def test_simulate_candidate_seed_deterministic():
    """Hardened engine is seed-deterministic (basic regression net)."""
    import numpy as _np
    from deployment import simulate_test as st
    Sigma, bank, ell = st.load_deployment(uniform_ell_star=None)
    cfg = st.TestConfig.from_yaml()
    theta = _np.zeros(st.DIM)
    for k in range(st.K):
        theta[2 * k + 1] = 0.6
    d1 = st.simulate_candidate(theta, Sigma, ell, bank, cfg,
                               _np.random.default_rng(123)).decision
    d2 = st.simulate_candidate(theta, Sigma, ell, bank, cfg,
                               _np.random.default_rng(123)).decision
    assert d1 == d2


def _delta_artifact():
    if not DELTA.exists():
        pytest.skip("phase4_3_hardening_delta.json not produced — run "
                    "`python -m pipeline.deployment_delta."
                    "phase4_3_lapse_delta`")
    import json
    return json.loads(DELTA.read_text())


def test_phase4_3_signed_off_delta_invariants():
    """Robust invariants of the signed-off CONTROLLED delta (NOT the
    sampling-noisy exact counts): bounded change, the expected
    conservative →REFER-dominated direction + longer tests, and
    near-symmetric PASS↔FAIL (⇒ sampling noise, not systematic bias)."""
    d = _delta_artifact()
    assert d["n_candidates"] == 200 and "CONTROLLED" in d["scope"]
    assert 0.05 <= d["decision_change_frac"] <= 0.30
    assert d["mean_trials_delta"] > 0          # lapse caps info ⇒ longer
    ds = d["directional_summary"]
    assert ds["to_refer_more_conservative"] > ds["from_refer_more_decisive"]
    assert ds["to_refer_more_conservative"] >= 0.4 * d[
        "decision_cells_changed"]
    assert abs(ds["net_stricter_frac"]) <= 0.01   # near-symmetric flips
    for key in ("mechanism", "pass_fail_flip_caveat", "signed_off"):
        assert isinstance(d[key], str) and d[key]
    assert "LIMITATION" in d["pass_fail_flip_caveat"]


@pytest.mark.slow
def test_phase4_3_directional_signature_live():
    """Live regression: the hardened engine + the committed pre-engine
    still produce the conservative signature on a small slice (the full
    signed-off artifact is produced offline)."""
    import importlib
    m = importlib.import_module(
        "pipeline.deployment_delta.phase4_3_lapse_delta")
    s = m.run_study(limit=36)                      # ~45s, no artifact
    assert 0.0 <= s["decision_change_frac"] <= 0.40
    assert s["mean_trials_delta"] > 0
    ds = s["directional_summary"]
    assert ds["to_refer_more_conservative"] >= ds["from_refer_more_decisive"]
    assert abs(ds["net_stricter_frac"]) <= 0.05


# ── Phase 4.3b: multi-seed Monte-Carlo decomposition (strengthening the
#    signed-off delta — separates the SYSTEMATIC lapse effect from
#    intra-candidate sampling noise for the PASS↔FAIL subset). ──────────
MC = REPO / "deployment" / "phase4_3_hardening_delta_mc.json"


def _mc_artifact():
    if not MC.exists():
        pytest.skip("phase4_3_hardening_delta_mc.json not produced — "
                    "run `python -m pipeline.deployment_delta."
                    "phase4_3_multiseed_mc`")
    import json
    return json.loads(MC.read_text())


def test_phase4_3b_mc_decomposition_invariants():
    """ANALYSIS-INTEGRITY + HONESTY invariants (NOT a hoped outcome —
    the MC in fact OVERTURNED the 'just sampling noise' hypothesis and
    found a real, small, systematic pass-share shift; the test asserts
    the decomposition is sound and the finding is truthfully flagged,
    whatever its sign)."""
    d = _mc_artifact()
    assert d["m_seeds"] >= 10
    assert d["n_cells"] == 1200          # 200 cands × 6 tasks
    # probability conservation: the 3 decision-prob deltas sum to 0
    s3 = (d["systematic_d_p_pass"] + d["systematic_d_p_fail"]
          + d["systematic_d_p_refer"])
    assert abs(s3) < 1e-9
    for k in ("systematic_d_p_refer_se", "systematic_d_p_pass_se",
              "systematic_d_p_fail_se", "systematic_pass_share_shift_se"):
        assert d[k] > 0                  # real MC standard errors
    # the conservative REFER increase IS systematic (robust mechanism)
    assert d["systematic_d_p_refer"] > 0
    assert d["refer_increase_is_systematic"] is True
    # the boolean verdict must be SELF-CONSISTENT with the SE test …
    null = abs(d["systematic_pass_share_shift"]) <= 2 * d[
        "systematic_pass_share_shift_se"]
    assert d["passfail_balance_shift_is_null"] is bool(null)
    # … and HONESTLY reflected in the prose (no overclaim either way)
    for key in ("conclusion", "interpretation", "open_item"):
        assert isinstance(d[key], str) and len(d[key]) > 80
    if not d["passfail_balance_shift_is_null"]:
        # a real shift MUST be flagged as a carried-forward open item,
        # not signed off as benign
        assert "re-assess" in d["open_item"].lower()
        assert "not signed off as benign" in d["open_item"].lower()
        assert "OVERTURNED" in d["interpretation"]


@pytest.mark.slow
def test_phase4_3b_mc_runs_small_slice():
    """Live regression that the parallel MC study still executes and
    conserves probability on a small slice (underpowered for the
    significance verdicts — those are gated on the offline artifact)."""
    import importlib
    m = importlib.import_module(
        "pipeline.deployment_delta.phase4_3_multiseed_mc")
    s = m.run_study(limit=6, max_workers=2)        # ~45s, no artifact
    s3 = (s["systematic_d_p_pass"] + s["systematic_d_p_fail"]
          + s["systematic_d_p_refer"])
    assert abs(s3) < 1e-9
    assert s["n_cells"] == 36 and s["m_seeds"] >= 10
