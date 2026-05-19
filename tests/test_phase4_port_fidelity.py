"""Phase 4.1 — deployment port-fidelity regression.

Two-step Phase-4 gate, step 1 (prove the PORT is faithful BEFORE
layering the intended changes — hardening 4.3 / K=7 4.4 / v13 ℓ* 4.5 /
re-freeze 4.6). The PI deployment scripts were ported PATH-ONLY into
`deployment/`; this asserts that introduced ZERO behavioural change by
reproducing the carried PI reference `data/deployment_prior/sim/`
(produced by PI at seed=0) bit-faithfully.

  * fast/durable: zero absolute `/Users/` paths in deployment/ (plan
    Phase 4 step 2); import + shape + Config-default invariants.
  * slow: full seed=0 sim reproduces the committed baseline EXACTLY
    (per-candidate decision identity + posterior numerics within
    LAPACK round-off). NOTE: this exact-reproduction gate is valid
    only in PI-FAITHFUL mode (sub-steps 4.1–4.2). It is INTENTIONALLY
    superseded at 4.3, where likelihood hardening deliberately changes
    the numbers — at which point the two-step gate switches to
    "differences fully attributed to the intended change + signed off".

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


# ── slow: bit-faithful reproduction (PI-FAITHFUL mode gate, 4.1–4.2) ─────────
@pytest.mark.slow
def test_port_reproduces_pi_baseline_exactly(tmp_path):
    """The definitive port-fidelity net: full seed=0 sim must reproduce
    the PI reference candidates.csv with ZERO decision mismatches and
    posterior numerics within LAPACK float64 round-off. Superseded at
    4.3 (hardening intentionally diverges) — see module docstring."""
    if not BASELINE.exists():
        pytest.skip("baseline absent")
    from deployment.run_deployment_sim import main
    main(seed=0, out_dir=str(tmp_path))
    base = pd.read_csv(BASELINE)
    rep = pd.read_csv(tmp_path / "candidates.csv")
    assert list(base.columns) == list(rep.columns)
    assert (base.tier.values == rep.tier.values).all(), "tier/RNG drift"

    dec = [f"decision_{t}" for t in TASKS]
    n_mis = int((base[dec].values != rep[dec].values).sum())
    assert n_mis == 0, f"{n_mis}/1200 decision cells diverged from PI"

    true_cols = [c for c in base.columns if c.startswith("true_")]
    assert np.array_equal(base[true_cols].values, rep[true_cols].values), (
        "true-theta differs ⇒ RNG stream not bit-faithful")
    num = [c for c in base.columns
           if c.startswith(("hat_", "sd_", "n_trials_"))]
    assert np.allclose(base[num].values, rep[num].values,
                       atol=1e-9, rtol=0), (
        "posterior numerics exceed LAPACK round-off ⇒ port not faithful")
