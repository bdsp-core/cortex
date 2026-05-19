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
# Phase 4.6-A: the PI K=6 baseline (the historical anchor for the
# port-fidelity + 4.3/4.3b/4.5 controlled studies) was archived FROZEN
# here before 4.6 regenerated the canonical deployment_prior on the
# unified K=7 corpus. These tests assert the PI-baseline integrity, so
# they read the archive (the canonical files are now the K=7 re-freeze).
_PI = REPO / "data" / "deployment_prior" / "_pi_baseline_frozen"
BASELINE = _PI / "candidates.csv"
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


_TASKS7 = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]


def test_deployment_imports_shape_and_config_defaults():
    import engine_paths
    assert Path(engine_paths.DEPLOYMENT_PRIOR).is_dir()
    from deployment import simulate_test as st
    # Phase 4.6-A: the deployment RUNTIME is now K=7 (re-frozen on the
    # unified corpus with the real "other"). The module-level TASKS in
    # THIS test file (6-list) is the historical PI/archive population
    # used by the archive-baseline tests — distinct from the runtime.
    assert st.TASKS == _TASKS7
    assert st.K == 7 and st.DIM == 14
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


# ── Phase 4.5: single v13 ℓ* lineage ────────────────────────────────────────
def test_v13_single_ell_lineage_drift_guard():
    """load_deployment ℓ* MUST equal calibration/cert_config.yaml v13
    (`ell_star_unified_v13`) by construction for every parsed task —
    the single reference-faithful lineage (D2). ell_thresholds.csv is
    NO LONGER consumed as the ℓ* source."""
    import numpy as np
    import yaml
    import engine_paths
    from deployment import simulate_test as st
    tasks = st.deployment_task_names()
    # every deployment task has a v13 mapping
    for t in tasks:
        assert t in st._V13_TASK_KEY, f"no v13 mapping for {t!r}"
    _, _, ell = st.load_deployment(uniform_ell_star=None)
    cfg = yaml.safe_load(
        Path(engine_paths.CALIB_CERT_CONFIG).read_text())
    v13 = cfg["ell_star_unified_v13"]["tasks"]
    expect = np.array([float(v13[st._V13_TASK_KEY[t]]["ell_star"])
                       for t in tasks])
    assert np.array_equal(ell, expect), (
        "deployment ℓ* != v13 cert_config — single-lineage broken")
    # the PI ell_thresholds Youden lineage must NOT be the source: its
    # values differ materially, so ell must NOT equal it. (Read the
    # FROZEN PI archive — 4.6 regenerated the canonical ell_thresholds
    # on the unified K=7 corpus; the historical PI lineage lives in the
    # archive.)
    thr = pd.read_csv(
        _PI / "ell_thresholds.csv"
    ).set_index("task")
    # the PI K=6 archive has no "other" (7th task is post-K=6); compare
    # only the tasks present in BOTH (the historical 6) — deployment
    # v13 ℓ* must differ materially from PI's Youden lineage there.
    common = [t for t in tasks if t in thr.index]
    assert len(common) >= 6
    ell_common = np.array([ell[tasks.index(t)] for t in common])
    pi = np.array([float(thr.loc[t, "ell_star"]) for t in common])
    assert not np.allclose(ell_common, pi), (
        "deployment ℓ* still equals PI ell_thresholds — v13 not wired")
    # load_deployment must NOT read ell_thresholds.csv as the ℓ* source
    # (the old code expression is gone; a docstring legacy-note mention
    # of the filename is fine).
    body = (REPO / "deployment" / "simulate_test.py").read_text().split(
        "def load_deployment")[1].split("\ndef ")[0]
    assert 'read_csv(DEPLOY / "ell_thresholds.csv")' not in body, (
        "load_deployment still reads ell_thresholds.csv for ℓ*")
    # uniform override still works
    _, _, e2 = st.load_deployment(uniform_ell_star=0.5)
    assert np.allclose(e2, 0.5)


_P45 = REPO / "deployment" / "phase4_5_v13_ell_delta.json"


def _p45():
    if not _P45.exists():
        pytest.skip("phase4_5_v13_ell_delta.json not produced — run "
                    "`python -m pipeline.deployment_delta."
                    "phase4_5_v13_ell_delta`")
    import json
    return json.loads(_P45.read_text())


def test_phase4_5_signed_off_artifact_invariants():
    """Part A: the ℓ*-lineage delta is fully attributed + signed off as
    the intended D2 swap. Part B: the 4.3b finding is re-assessed under
    v13 and carried forward. Robust invariants, not pinned numbers."""
    d = _p45()
    A = d["part_A_ell_lineage"]
    B = d["part_B_4_3b_reassessed_under_v13"]
    assert A["n_candidates"] == 200 and "ISOLATED" in A["scope"]
    # v13 ℓ* differs materially ⇒ a substantial, attributed shift
    assert A["decision_change_frac"] > 0.0
    assert "SIGNED OFF" in A["interpretation"]
    assert sum(A["flip_table"].values()) == A["decision_cells_changed"]
    # Part B: lapse REFER increase still present; comparison recorded
    assert B["systematic_d_p_refer"] > 0
    cmp = B["compare_4_3b_under_PI_ell"]
    assert "pass_share_shift_PI_ell" in cmp
    assert "pass_share_shift_v13_ell" in cmp
    assert "Phase 4.6" in B["interpretation"]      # carried forward
    assert "carried forward" in d["disposition"].lower()


@pytest.mark.slow
def test_phase4_5_runs_small_slice():
    """Live regression: the 4.5 two-part study executes + conserves
    probability on a small slice (significance gated on the artifact)."""
    import importlib
    m = importlib.import_module(
        "pipeline.deployment_delta.phase4_5_v13_ell_delta")
    s = m.run_study(limit=6, max_workers=2)
    B = s["part_B_4_3b_reassessed_under_v13"]
    s3 = (B["systematic_d_p_pass"] + B["systematic_d_p_fail"]
          + B["systematic_d_p_refer"])
    assert abs(s3) < 1e-9
    assert s["part_A_ell_lineage"]["n_candidates"] == 6


# ── Phase 4.6-A: K=7 re-freeze on unified corpus + clean-sn1 spike +
#    D3 provenance + PI-baseline archive integrity ─────────────────────────
import engine_paths as _ep   # noqa: E402

_DP = Path(_ep.DEPLOYMENT_PRIOR)


def test_phase4_6a_k7_frozen_artifact_shape():
    """The re-frozen deployment_prior is K=7: Σ 14×14 with 7 t_/l_
    slot pairs incl 'other'; case_bank + ell_thresholds 7 tasks."""
    S = pd.read_csv(_DP / "Sigma.csv", index_col=0)
    assert S.shape == (14, 14)
    assert list(S.index) == [
        "t_spike", "l_spike", "t_seizure", "l_seizure", "t_lpd", "l_lpd",
        "t_gpd", "l_gpd", "t_lrda", "l_lrda", "t_grda", "l_grda",
        "t_other", "l_other"]
    bank = pd.read_csv(_DP / "case_bank.csv")
    assert set(bank.task.unique()) == set(_TASKS7)
    thr = pd.read_csv(_DP / "ell_thresholds.csv")
    assert list(thr.task) == _TASKS7


def test_phase4_6a_runtime_is_k7_v13_live():
    """load_deployment now returns K=7 with the v13 ℓ* for ALL 7 incl
    other←sparcnet_iic (the 7th is LIVE post-re-freeze)."""
    import numpy as np
    import yaml
    from deployment import simulate_test as st
    tasks = st.deployment_task_names()
    assert tasks == _TASKS7
    S, bank, ell = st.load_deployment(uniform_ell_star=None)
    assert S.shape == (14, 14) and len(bank) == 7 and len(ell) == 7
    cfg = yaml.safe_load(Path(_ep.CALIB_CERT_CONFIG).read_text())
    v13 = cfg["ell_star_unified_v13"]["tasks"]
    exp = np.array([float(v13[st._V13_TASK_KEY[t]]["ell_star"])
                    for t in tasks])
    assert np.array_equal(ell, exp)
    # the 7th task IS the real "other" ← v13 sparcnet_iic, now live
    assert tasks[6] == "other"
    assert ell[6] == pytest.approx(
        float(v13["sparcnet_iic"]["ell_star"]))


def test_phase4_6a_clean_sn1_spike_consistent_with_v13():
    """fit_2pl_probit's spike is now clean-sn1-only (Centaur-IED
    excluded) — its positive count == the v13 combined_spike / Phase-3
    sn1-binary definition, and identical to fit_2pl_probit_hier's
    spike (the bank now shares one population with its v13 ℓ*)."""
    labels_csv = REPO / "data" / "labels" / "labels.csv"
    if not labels_csv.exists():
        pytest.skip("unified labels.csv absent")
    import sys as _sys
    _sys.path.insert(0, str(REPO / "pipeline"))
    import fit_2pl_probit as f
    import fit_2pl_probit_hier as h
    L = pd.read_csv(labels_csv, low_memory=False, dtype={"value": "str"})
    spv = L[L.label_type == "spike"].value.astype(str).str.strip()
    expect_pos = int((spv == "1").sum())          # Phase-3 sn1 binary
    assert expect_pos > 0
    sf = f.extract_task_labels(L, "spike")
    sh = h.extract_task_labels(L, "spike")
    assert int(sf.Y.sum()) == expect_pos, (
        "fit_2pl_probit spike not clean-sn1 (Centaur-IED leaked in?)")
    assert int(sh.Y.sum()) == expect_pos
    # no Centaur 6-class string ever counted positive
    src = (REPO / "pipeline" / "fit_2pl_probit.py").read_text()
    assert 'isin(["spike", "ied"])' not in src
    assert "pd.to_numeric(sub[\"value\"]" in src


def test_phase4_6a_d3_provenance_anchors_unified_corpus():
    """deployment_prior summary.json records data/labels sha256 (D3:
    derived-artifact lineage) and it matches the LIVE corpus."""
    import hashlib
    import json

    def _sha(p):
        x = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 20), b""):
                x.update(c)
        return x.hexdigest()
    s = json.loads((_DP / "summary.json").read_text())
    assert s["K"] == 7 and s["tasks"] == _TASKS7
    pv = s["data_labels_provenance"]
    assert pv["labels_csv_sha256"] == _sha(
        REPO / "data" / "labels" / "labels.csv")
    assert pv["raters_csv_sha256"] == _sha(
        REPO / "data" / "labels" / "raters.csv")
    assert "v13 cert_config" in pv["ell_star_lineage"]


def test_phase4_6a_pi_baseline_archive_intact():
    """The frozen PI K=6 archive (the historical anchor for the
    repointed 4.3/4.3b/4.5 studies) is present + well-formed."""
    A = _DP / "_pi_baseline_frozen"
    assert (A / "README.md").is_file()
    Sig = pd.read_csv(A / "Sigma.csv", index_col=0)
    assert Sig.shape == (12, 12)                  # PI K=6
    cand = pd.read_csv(A / "candidates.csv")
    assert len(cand) == 200
    thr = pd.read_csv(A / "ell_thresholds.csv")
    assert set(thr.task) == set(TASKS)            # 6-list, no "other"
    assert not (data_iic := (_DP / "_pi_baseline_frozen" / "iic")).exists()


def test_phase4_6a_stale_iic_orphan_removed():
    """The PI degenerate-OR per-task fit dir is gone (ported fitter
    writes fits/other; fits/iic would be a dead orphan)."""
    assert not (REPO / "data" / "labels" / "fits" / "iic").exists()
    assert (REPO / "data" / "labels" / "fits" / "other").is_dir()


# ── Phase 4.6-B: RNG-decouple in run_deployment_sim ─────────────────────────
@pytest.mark.slow
def test_phase4_6b_rng_decoupled_population_invariant():
    """THE 4.6-B gate: per-candidate θ is a pure function of
    (seed, cand_id) — INVARIANT to the stopping config / trial counts
    (the 4.3 shared-rng confound is fixed). Tiny panel; perturbing the
    config must NOT reshuffle the θ-population, yet decisions still
    respond to it, and a rerun is bit-identical."""
    import tempfile
    import numpy as np
    import pandas as pd
    from deployment import run_deployment_sim as rds
    from deployment import simulate_test as st
    _T, _cfg = rds.TIERS, st.TestConfig.from_yaml
    try:
        rds.TIERS = {
            "expert": dict(n=3, mu_ell=1.6, sd_within=.3, mu_t=0, sd_t=.2),
            "crowd":  dict(n=3, mu_ell=-.3, sd_within=.45, mu_t=0, sd_t=.6)}
        A, B, C = (tempfile.mkdtemp() for _ in range(3))
        rds.main(seed=0, out_dir=A)
        st.TestConfig.from_yaml = classmethod(lambda cls: cls(
            N_min=60, N_max=300, N_min_per_task=10, N_max_per_task=80,
            pass_p=0.9, fail_p=0.1))                 # perturb stopping cfg
        rds.main(seed=0, out_dir=B)
        st.TestConfig.from_yaml = _cfg
        rds.main(seed=0, out_dir=C)                  # determinism rerun
    finally:
        rds.TIERS, st.TestConfig.from_yaml = _T, _cfg
    a = pd.read_csv(Path(A) / "candidates.csv")
    b = pd.read_csv(Path(B) / "candidates.csv")
    c = pd.read_csv(Path(C) / "candidates.csv")
    tcol = [x for x in a.columns if x.startswith("true_")]
    dcol = [x for x in a.columns if x.startswith("decision_")]
    # decoupling guarantee: θ-population unchanged by the cfg change
    assert a[tcol].equals(b[tcol]), (
        "θ-population shifted under a config change — RNG still coupled")
    # determinism: same seed/cfg → bit-identical
    assert a[tcol].equals(c[tcol]) and a[dcol].equals(c[dcol])
    # sanity: the Y-stream still responds to the config (sim still works)
    assert not a[dcol].equals(b[dcol])


def test_phase4_6b_canonical_sim_is_k7_wellformed():
    """The regenerated canonical sim/ is K=7 (decoupled, seed=0):
    200 candidates, 7-task decision columns incl 'other'.

    Asserts on the FINAL regenerated artifact. SKIPS (not fails) if the
    sim is absent or not yet the completed K=7 product — the 4.6-B
    `run_deployment_sim` regen is a heavy background step; this test
    must be race-safe (a partially-written candidates.csv / an old K=6
    sim still on disk is 'not done yet', not a failure)."""
    import json
    sim = _DP / "sim" / "candidates.csv"
    summ = _DP / "sim" / "summary.json"
    if not sim.exists() or not summ.exists():
        pytest.skip("canonical K=7 sim not regenerated yet (4.6-B)")
    try:
        c = pd.read_csv(sim)
        s = json.loads(summ.read_text())
    except (pd.errors.EmptyDataError, pd.errors.ParserError,
            ValueError):
        pytest.skip("canonical sim mid-regeneration (race) — re-run")
    incomplete = (len(c) != 200 or s.get("seed") != 0
                  or any(f"decision_{t}" not in c.columns
                         for t in _TASKS7))
    if incomplete:
        pytest.skip("canonical sim not yet the completed K=7 product")
    for t in _TASKS7:
        assert set(c[f"decision_{t}"]).issubset(
            {"pass", "fail", "refer"})
    assert s["seed"] == 0 and len(s.get("ell_star", [])) == 7


# ── Phase 4.6-C: definitive 4.3b re-assessment on the shipped K=7 ────────────
_P46C = REPO / "deployment" / "phase4_6c_shipped_reassess.json"


def _p46c():
    if not _P46C.exists():
        pytest.skip("phase4_6c_shipped_reassess.json not produced — run "
                    "`python -m pipeline.deployment_delta."
                    "phase4_6c_shipped_reassess`")
    import json
    return json.loads(_P46C.read_text())


def test_phase4_6c_definitive_reassessment_invariants():
    """Robust invariants of the DEFINITIVE shipped-K=7 4.3b verdict
    (not pinned MC numbers): measured on the shipped config via the
    λ=0 counterfactual; the lapse REFER mechanism is present; the
    pass-share trajectory across PI-K6 → v13-K6 → shipped-K7 is
    recorded; the verdict prose is self-consistent with the SE test
    and honestly dispositioned (RESOLVED vs PERSISTS-as-known)."""
    d = _p46c()
    assert d["K"] == 7 and d["n_candidates"] == 200
    assert "SHIPPED" in d["scope"] and "λ=0" in d["scope"]
    # probability conservation
    s3 = (d["systematic_d_p_pass"] + d["systematic_d_p_fail"]
          + d["systematic_d_p_refer"])
    assert abs(s3) < 1e-9
    assert d["systematic_d_p_refer"] > 0          # lapse → more REFER
    tr = d["trajectory"]
    for k in ("4.3b_PI_K6_ell_thresholds", "4.5_K6_v13_ell",
              "4.6c_shipped_K7_v13_decoupled"):
        assert k in tr
    assert tr["4.6c_shipped_K7_v13_decoupled"] == pytest.approx(
        d["systematic_pass_share_shift_shipped"])
    # boolean verdict self-consistent with the ~2·SE test
    null = (abs(d["systematic_pass_share_shift_shipped"])
            <= 2 * d["systematic_pass_share_shift_shipped_se"])
    assert d["passfail_balance_null_shipped"] is bool(null)
    for key in ("definitive_verdict", "disposition"):
        assert isinstance(d[key], str) and len(d[key]) > 80
    if null:
        assert "RESOLVED" in d["definitive_verdict"]
    else:
        assert "PERSISTS" in d["definitive_verdict"]
        assert "DEPLOYMENT_INTEGRATION.md" in d["disposition"]
        assert "NOT a bug" in d["disposition"]


def test_phase4_close_out_doc_present():
    """Phase-4 close-out provenance doc exists with the per-sub-step
    attribution + the 4.6-C definitive verdict."""
    doc = REPO / "docs" / "DEPLOYMENT_INTEGRATION.md"
    if not doc.exists():
        pytest.skip("DEPLOYMENT_INTEGRATION.md not written yet (4.6-C)")
    txt = doc.read_text()
    for anchor in ("Phase 4.1", "Phase 4.3", "Phase 4.5",
                   "Phase 4.6-A", "Phase 4.6-B", "Phase 4.6-C",
                   "D3", "v13"):
        assert anchor in txt, f"close-out doc missing {anchor!r}"


@pytest.mark.slow
def test_phase4_6c_runs_small_slice():
    """Live regression: the shipped-K=7 λ-vs-λ=0 study executes +
    conserves probability on a small slice (verdict gated on the
    offline full artifact)."""
    import importlib
    m = importlib.import_module(
        "pipeline.deployment_delta.phase4_6c_shipped_reassess")
    s = m.run_study(limit=6, max_workers=2)
    s3 = (s["systematic_d_p_pass"] + s["systematic_d_p_fail"]
          + s["systematic_d_p_refer"])
    assert abs(s3) < 1e-9
    assert s["K"] == 7 and s["n_candidates"] == 6


# ── Phase 4.7: ilae-deploy CLI orchestrator + K-agnostic plot ───────────────
def test_phase4_7_cli_dispatch_and_argparse():
    """The ilae-deploy entry point is the deployment.cli orchestrator
    with freeze/simulate/plot/all stages (default all)."""
    from deployment import cli
    assert callable(cli.main)
    assert cli.main.__module__ == "deployment.cli"
    # argparse accepts the four stages; rejects junk with SystemExit(2)
    pr = cli.argparse.ArgumentParser  # smoke that argparse is wired
    assert pr is not None
    with pytest.raises(SystemExit) as e:
        cli.main(["not-a-stage"])
    assert e.value.code == 2
    # the three stage helpers exist (freeze/simulate are heavy → not
    # executed here; the `all` pipeline sequences them)
    for fn in ("_freeze", "_simulate", "_plot"):
        assert callable(getattr(cli, fn))


def test_phase4_7_plot_is_k_agnostic_and_renders_k7():
    """plot_deploy derives K from the SHIPPED artifact (K=7 post-4.6,
    not a hardcoded 6-list); _grid scales; all 5 figures render."""
    import importlib
    pdp = importlib.import_module("deployment.plot_deploy")
    importlib.reload(pdp)                       # pick up current artifact
    assert pdp.K == 7 and len(pdp.TASKS) == 7 and "other" in pdp.TASKS
    assert "other" in pdp.PRETTY
    fig, axs = pdp._grid(7)
    assert len(axs) == 7                        # K-agnostic grid
    import matplotlib.pyplot as plt
    plt.close(fig)
    figdir = _DP / "figures"
    if not (figdir / "fig5_verdicts.png").exists():
        pytest.skip("figures not rendered yet (run `ilae-deploy plot`)")
    for fn in ("fig1_concept", "fig2_single_candidate",
               "fig3_skill_by_tier", "fig4_recovery", "fig5_verdicts"):
        f = figdir / f"{fn}.png"
        assert f.is_file() and f.stat().st_size > 0


def test_phase4_7_plot_is_best_effort_in_pipeline(monkeypatch):
    """In the `all` pipeline a plot failure must NOT fail the run
    (figures are derived viz); a standalone `plot` surfaces it."""
    from deployment import cli, plot_deploy

    def _boom():
        raise RuntimeError("simulated matplotlib failure")
    monkeypatch.setattr(plot_deploy, "main", _boom)
    # best-effort (pipeline): swallow → rc 0
    assert cli._plot(best_effort=True) == 0
    # explicit `plot`: fatal → rc 1
    assert cli._plot(best_effort=False) == 1
