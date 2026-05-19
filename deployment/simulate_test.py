"""Multi-task adaptive certification test simulator.

Inputs (frozen at deployment):
  - Σ : 12×12 multivariate-Gaussian prior on θ = (t_1, ℓ_1, ..., t_6, ℓ_6)
  - case bank: per-task list of cases with frozen signal s_jk
  - per-task ℓ* threshold (expert vs non-expert Youden cut)

Per candidate:
  - true θ (known in simulation)
  - posterior tracked as Laplace Gaussian (mean μ ∈ R^12, covariance Σ_post)
  - at each trial:
      * select next (task, case) maximizing expected entropy reduction
        of the 12-dim posterior, restricted to tasks not yet decided
      * draw Bernoulli response from true θ
      * recursive-Bayesian probit update of (μ, Σ_post)
      * check per-task stopping:
          P(ℓ_k > ℓ*_k | data) ≥ pass_p  → PASS
          P(ℓ_k > ℓ*_k | data) ≤ fail_p  → FAIL
          else PENDING (continue)
  - stop when all 6 tasks decided OR N_max reached (remaining tasks: REFER)

Mathematical machinery:

  Score vector for trial (k, j) at θ:
    x_kj(θ) = ∂η/∂θ has just two nonzero entries:
       (∂η/∂t_k)  = e^{ℓ_k}
       (∂η/∂ℓ_k) = e^{ℓ_k}(s_jk + t_k) = η_{kj}(θ)

  Probit IRLS pseudo-observation:
    With weight w(η) = φ(η)²/[Φ(η)(1-Φ(η))],
    target z = η + (Y - Φ(η))/φ(η),
    the recursive Gaussian update is:
       Σ_post⁻¹  ←  Σ_post⁻¹ + w · x x^T
       Σ_post⁻¹ μ ←  Σ_post⁻¹ μ + w · x · z

  Info gain criterion (entropy reduction):
    Adding a rank-1 precision w·xx^T reduces log-det of Σ_post by
       log(1 + w · x^T Σ_post x).
    We use this as the case-selection score (plug-in at posterior mean).

─────────────────────────────────────────────────────────────────────────
UNIFIED-MERGE PROVENANCE. Phase 4.1: FAITHFUL PATH-ONLY port of the PI
`scripts/simulate_test.py` (PI hardcoded absolute-ROOT/DEPLOY block →
`engine_paths.DEPLOYMENT_PRIOR` shim; zero absolute paths, CI-enforced
incl. this header). Phase 4.2: PI `TestConfig` → `deployment_config.yaml`
(bit-unchanged; from_yaml strict).

Phase 4.3 (2026-05-18, user-approved FULL scope): the LIKELIHOOD is now
ONE definition shared with Paper-1 — λ-lapse mixture p=λ+(1−2λ)Φ(η)
(λ imported from engine `core`; the SMC vs this Laplace/EKF inference
engines stay SEPARATE per D1, only the likelihood form is shared).
Applied to: response generation (byte-faithful to
`core_mcmc.simulate_response`), the IRLS Newton update
(`irls_w_z`/`_lapse_components` — exact lapse-GLM Fisher/score,
reduces to PI at λ=0; the λ floor removes the Φ(1−Φ)→0 blowup the PI
1e-10 clip patched), the predictive `expected_p`, and the
`select_next_case` info-gain weight. `_p_pass` is UNCHANGED — it is a
latent-ℓ posterior tail prob, NOT a response prob (lapse does not
apply). This DELIBERATELY changes deployment outputs: the Phase-4.1/4.2
exact-reproduction gate is INTENTIONALLY superseded — the two-step gate
switches to "delta vs the pristine PI baseline fully attributed to the
lapse mixture + bounded + signed off" (deployment/phase4_3_hardening_
delta.json). K=7 (4.4), v13 ℓ* (4.5), re-freeze (4.6) layered next.
"""
from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import log_ndtr           # FIX-T0.7: tail-stable log Φ
from scipy.stats import norm

# ── path shim (mirrors bridge/run_mode_b_cert_bridge.py): self-locate the
#    repo root, put engine/ + root on sys.path, resolve via engine_paths
#    (single path authority). Replaces the PI hardcoded ROOT/DEPLOY. ──
_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_DEPLOY_DIR)
for _p in (os.path.join(_REPO, "engine"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import engine_paths  # noqa: E402
# Phase 4.3 (decision 2026-05-18): the LIKELIHOOD is ONE definition —
# import λ from the engine core (single source). The inference engines
# (SMC vs this Laplace/EKF) stay SEPARATE per D1; only the likelihood
# constant/form is shared. core is on sys.path via the shim above; it is
# the numpy engine (no JAX — calibration-stage only).
from core import LAPSE_RATE  # noqa: E402

DEPLOY = Path(engine_paths.DEPLOYMENT_PRIOR)
_ONE_MINUS_2LAMBDA = 1.0 - 2.0 * LAPSE_RATE

# Phase 4.4-A: the engine is now K-AGNOSTIC — it derives K/DIM from the
# loaded artifact's array shapes (Σ.shape ⇒ DIM, DIM//2 ⇒ K) so it scales
# to the K=7 deployment_prior that Phase 4.6 will produce. These
# module-level constants are kept ONLY as the DEFAULT reflecting the
# CURRENTLY-SHIPPED K=6 frozen artifact (back-compat for callers/studies
# that are inherently K=6); the AUTHORITATIVE ordered task list comes
# from `deployment_task_names()` (parsed from the Σ slot-name index). A
# drift-guard test asserts these defaults equal the artifact.
TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda"]
K = len(TASKS)
DIM = 2 * K


def deployment_task_names(deploy_dir=None):
    """AUTHORITATIVE ordered task list for the frozen deployment prior,
    parsed from the Σ slot-name index (`t_<task>`,`l_<task>` pairs — Σ
    defines the θ-vector layout the whole engine uses). This is the
    single source of truth for K and task order; `bank`/`ell` are
    cross-checked against it in `load_deployment`."""
    d = Path(deploy_dir) if deploy_dir is not None else DEPLOY
    idx = list(pd.read_csv(d / "Sigma.csv", index_col=0).index)
    if len(idx) % 2 != 0:
        raise ValueError(f"Σ has odd dim {len(idx)} — not t/ℓ pairs")
    tasks = []
    for i in range(0, len(idx), 2):
        ti, li = str(idx[i]), str(idx[i + 1])
        if not ti.startswith("t_") or not li.startswith("l_"):
            raise ValueError(
                f"Σ slot names not t_/l_ pairs at {i}: {ti!r},{li!r}")
        if ti[2:] != li[2:]:
            raise ValueError(
                f"Σ slot pair task mismatch: {ti!r} vs {li!r}")
        tasks.append(ti[2:])
    return tasks


# ───────────────────────── helpers ─────────────────────────

def _slot(k, parm):
    """Return the global slot index for parm in {'t','l'} on task k."""
    return 2 * k + (0 if parm == "t" else 1)


def _lapse_components(eta_c: np.ndarray):
    """The ONE likelihood definition (Phase 4.3), in η-space.

    Spike-paper Eq. 2 lapse mixture (engine `core`; λ = LAPSE_RATE):
        p(η)  = λ + (1−2λ)·Φ(η)            ∈ [λ, 1−λ]
        p'(η) = (1−2λ)·φ(η)
    Φ via `log_ndtr` (engine FIX-T0.7: cancellation-free in the tails,
    unlike np.clip(norm.cdf,·); the PI 1e-10 Φ-clip is no longer needed —
    the λ floor makes p(1−p) ≥ λ(1−λ) > 0 by construction). `eta_c` is
    expected pre-clipped to the PI [−6,6] stability bound by the caller.
    Returns (p, p').
    """
    Phi = np.exp(log_ndtr(eta_c))
    p = LAPSE_RATE + _ONE_MINUS_2LAMBDA * Phi
    pp = _ONE_MINUS_2LAMBDA * norm.pdf(eta_c)        # p'(η)
    return p, pp


def irls_w_z(eta: np.ndarray, Y: np.ndarray):
    """Lapse-mixture probit IRLS weight + working response (Phase 4.3).

    Fisher-scoring on the ONE likelihood p(η)=λ+(1−2λ)Φ(η):
        w = p'² / [p(1−p)]        z = η + (Y − p)/p'
    Reduces EXACTLY to the PI bare-probit IRLS at λ=0 (p→Φ, p'→φ). The
    λ floor removes the Φ(1−Φ)→0 blowup the PI 1e-10 clip crudely
    patched, so w needs NO Φ-clip; only the working-response denominator
    p' is guarded (it → 0 as φ→0 at the tails — orthogonal to the lapse
    fix, inherent to IRLS; the PI [−6,6] η-bound is kept for that)."""
    eta_c = np.clip(eta, -6.0, 6.0)
    p, pp = _lapse_components(eta_c)
    w = pp ** 2 / (p * (1.0 - p))           # p(1−p) ≥ λ(1−λ) > 0
    pp_c = np.clip(pp, 1e-12, None)         # guard z denom at saturation
    z = eta_c + (Y - p) / pp_c
    return w, z


def expected_p(mu, Sigma_post, k, s):
    """Lapse-mixture predictive P(Y=1) (Phase 4.3) under the standard
    probit-Gaussian conditional. The λ-lapse is an AFFINE wrapper of Φ,
    so it composes EXACTLY with the Φ(η_mean/√(1+v)) approximation:
        E[λ+(1−2λ)Φ(η)] ≈ λ + (1−2λ)·Φ(η_mean/√(1+v))."""
    t = mu[_slot(k, "t")]; l = mu[_slot(k, "l")]
    el = np.exp(l)
    eta = el * (s + t)
    # ∂η/∂θ has two non-zero entries: el (for t_k) and η (for ℓ_k)
    # Variance of η: x^T Σ_post x using 2x2 sub-block on (t_k, ℓ_k)
    j_t, j_l = _slot(k, "t"), _slot(k, "l")
    Stt = Sigma_post[j_t, j_t]
    Stl = Sigma_post[j_t, j_l]
    Sll = Sigma_post[j_l, j_l]
    v = (el ** 2) * (Stt + 2 * (s + t) * Stl + (s + t) ** 2 * Sll)
    z = eta / np.sqrt(1 + max(v, 0.0))
    return LAPSE_RATE + _ONE_MINUS_2LAMBDA * float(np.exp(log_ndtr(z)))


# ───────────────────────── simulator state ─────────────────────────

@dataclass
class TestState:
    mu: np.ndarray                       # (DIM,)
    Sigma_post: np.ndarray               # (DIM, DIM)
    n_per_task: np.ndarray               # (K,)
    decision: list                       # ["pending"|"pass"|"fail"|"refer"] x K
    history: list                        # list of (k, seg_id, s, Y)
    # Tracked over the test for plotting:
    mu_traj: list = field(default_factory=list)
    sd_traj: list = field(default_factory=list)   # marginal SDs per slot
    p_pass_traj: list = field(default_factory=list)  # per-task P(ℓ>ℓ*)


def _post_marg_sd(Sigma_post):
    return np.sqrt(np.diag(Sigma_post))


def _p_pass(mu, Sigma_post, ell_star):
    """Per-task P(ℓ_k > ℓ*_k | data) under Gaussian marginal.
    K-agnostic: K = len(ell_star) (Phase 4.4-A)."""
    k_ = len(ell_star)
    out = np.zeros(k_)
    for k in range(k_):
        m = mu[_slot(k, "l")]
        s = np.sqrt(Sigma_post[_slot(k, "l"), _slot(k, "l")])
        out[k] = 1.0 - norm.cdf(ell_star[k], loc=m, scale=max(s, 1e-6))
    return out


# ───────────────────────── one-trial update ─────────────────────────

def update_state(state: TestState, k: int, s_val: float, Y: float):
    """Recursive Bayesian probit update at trial (k, s_val, Y)."""
    j_t, j_l = _slot(k, "t"), _slot(k, "l")
    mu = state.mu.copy()
    Sigma_post = state.Sigma_post.copy()
    dim = mu.shape[0]; K_ = dim // 2          # K-agnostic (4.4-A)

    # 1 Newton step at current μ
    t = mu[j_t]; l = mu[j_l]
    el = np.exp(l)
    eta = el * (s_val + t)
    w, z = irls_w_z(np.array([eta]), np.array([Y]))
    w = float(w[0]); z = float(z[0])
    # x has two non-zero entries
    x = np.zeros(dim)
    x[j_t] = el
    x[j_l] = el * (s_val + t)

    Sigma_post_inv = np.linalg.inv(Sigma_post)
    # Σ⁻¹ µ_new = Σ⁻¹ µ_old + w·x·z (working response in η-units relative to flat)
    # Equivalently posterior on linearized residual:
    Sigma_post_inv_new = Sigma_post_inv + w * np.outer(x, x)
    Sigma_post_new = np.linalg.inv(Sigma_post_inv_new)
    mu_new = Sigma_post_new @ (Sigma_post_inv @ mu + w * x * z)
    # Clamp ℓ inside [-3, 3] so e^ℓ stays bounded for next-trial info eval
    for kk in range(K_):
        mu_new[_slot(kk, "l")] = float(np.clip(mu_new[_slot(kk, "l")], -3, 3))
        mu_new[_slot(kk, "t")] = float(np.clip(mu_new[_slot(kk, "t")], -3, 3))

    state.mu = mu_new
    state.Sigma_post = Sigma_post_new
    state.n_per_task[k] += 1
    state.history.append((k, s_val, Y))


# ───────────────────────── case selection ─────────────────────────

def select_next_case(state: TestState, bank_by_task, pending_tasks, used_segids,
                      n_min_per_task: int = 0):
    """Pick (k, seg_id, s) maximizing log(1 + w·x^T Σ_post x) over pending tasks.

    If `n_min_per_task` > 0, restrict the candidate set to pending tasks
    that have not yet reached n_min trials (when any such tasks exist).
    This avoids starving slow-to-be-decisive tasks of minimum coverage
    when info-max would prefer to keep probing already-rich tasks.
    """
    if n_min_per_task > 0:
        under = [k for k in pending_tasks if state.n_per_task[k] < n_min_per_task]
        if under:
            pending_tasks = under
    best = None
    for k in pending_tasks:
        df = bank_by_task[k]
        # Filter cases the candidate has already seen
        if used_segids[k]:
            mask = ~np.isin(df["seg_id"].values, list(used_segids[k]))
            sub = df[mask]
        else:
            sub = df
        if len(sub) == 0:
            continue
        s_arr = sub["s_mean"].values
        t_mu = state.mu[_slot(k, "t")]; l_mu = state.mu[_slot(k, "l")]
        el = np.exp(l_mu)
        eta = el * (s_arr + t_mu)
        # Σ_post 2x2 block on (t_k, ℓ_k)
        j_t = _slot(k, "t"); j_l = _slot(k, "l")
        Stt = state.Sigma_post[j_t, j_t]
        Stl = state.Sigma_post[j_t, j_l]
        Sll = state.Sigma_post[j_l, j_l]
        # v = x^T Σ x with x = el * (1, s+t)
        v = (el ** 2) * (Stt + 2 * (s_arr + t_mu) * Stl + (s_arr + t_mu) ** 2 * Sll)
        # Info-gain weight = the SAME lapse Fisher weight as the IRLS
        # update (Phase 4.3 — one likelihood definition, shared helper;
        # no arbitrary Φ-clip — the λ floor bounds p(1−p)).
        eta_c = np.clip(eta, -6.0, 6.0)
        p_sel, pp_sel = _lapse_components(eta_c)
        w = pp_sel ** 2 / (p_sel * (1.0 - p_sel))
        score = np.log1p(np.maximum(w * v, 0.0))
        i_best = int(np.argmax(score))
        if best is None or score[i_best] > best[0]:
            seg = int(sub["seg_id"].iloc[i_best])
            best = (float(score[i_best]), k, seg, float(s_arr[i_best]))
    if best is None:
        return None
    _, k, seg, s_val = best
    return k, seg, s_val


# ───────────────────────── simulator main loop ─────────────────────────

@dataclass
class TestConfig:
    """Test stopping rule configuration.

    N_min_per_task: minimum trials a task must accumulate before its
      verdict can be PASS or FAIL. Below this, the verdict stays PENDING
      regardless of posterior probability — this prevents the cross-task
      prior from forcing an early verdict on a task that has barely been
      probed (e.g. inferring "passes seizure" from a strong spike score
      via the r_ell=0.44 ℓ-ℓ block). Empirically 10 lets each task's own
      data overcome the prior pull on cross-task-correlated candidates.
    """
    N_min: int = 60
    N_max: int = 500
    N_min_per_task: int = 10
    N_max_per_task: int = 120
    pass_p: float = 0.95
    fail_p: float = 0.05

    # Phase 4.2: integer fields of the contract (the rest are floats).
    _INT_FIELDS = ("N_min", "N_max", "N_min_per_task", "N_max_per_task")

    @classmethod
    def from_yaml(cls, path=None) -> "TestConfig":
        """Load the deployment stopping-rule contract from
        `deployment_config.yaml` (plan Phase 4 step 3).

        STRICT (clinical pass/fail rule — no silent fallback): the YAML
        key set must EXACTLY equal the dataclass fields and types must
        match, else ValueError/TypeError. `path=None` resolves the
        canonical `engine_paths.DEPLOYMENT_CONFIG`. The dataclass
        defaults remain canonical; a drift-guard test asserts the YAML
        equals them so deployment output is bit-identical to the
        hardcoded-default path (Phase-4.2 gate).
        """
        p = (Path(path) if path is not None
             else Path(engine_paths.DEPLOYMENT_CONFIG))
        raw = yaml.safe_load(p.read_text())
        if not isinstance(raw, dict):
            raise ValueError(
                f"{p}: deployment config must be a YAML mapping")
        expected = {f.name for f in fields(cls)}
        got = set(raw)
        if got != expected:
            raise ValueError(
                f"{p}: deployment-config keys {sorted(got)} != contract "
                f"{sorted(expected)} (missing={sorted(expected - got)}, "
                f"unknown={sorted(got - expected)})")
        kw = {}
        for name in expected:
            v = raw[name]
            if isinstance(v, bool):  # bool is an int subclass — reject
                raise TypeError(f"{p}: {name} must not be bool ({v!r})")
            if name in cls._INT_FIELDS:
                if not isinstance(v, int):
                    raise TypeError(
                        f"{p}: {name} must be int, got {type(v).__name__}"
                        f" ({v!r})")
            else:  # pass_p, fail_p
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"{p}: {name} must be float, got "
                        f"{type(v).__name__} ({v!r})")
                v = float(v)
            kw[name] = v
        return cls(**kw)


def simulate_candidate(true_theta, Sigma_prior, ell_star, bank_by_task,
                        cfg: TestConfig, rng=None):
    """Simulate one candidate going through the adaptive multi-task test.
    K-agnostic (Phase 4.4-A): DIM = Σ_prior.shape[0], K = DIM//2 — scales
    to whatever K the frozen deployment_prior declares (6 now, 7 at 4.6)."""
    if rng is None:
        rng = np.random.default_rng()
    dim = Sigma_prior.shape[0]
    K_ = dim // 2
    state = TestState(
        mu=np.zeros(dim),
        Sigma_post=Sigma_prior.copy(),
        n_per_task=np.zeros(K_, dtype=int),
        decision=["pending"] * K_,
        history=[],
    )
    used_segids = [set() for _ in range(K_)]
    # snapshot at trial 0
    state.mu_traj.append(state.mu.copy())
    state.sd_traj.append(_post_marg_sd(state.Sigma_post))
    state.p_pass_traj.append(_p_pass(state.mu, state.Sigma_post, ell_star))

    for n in range(cfg.N_max):
        pending = [k for k in range(K_) if state.decision[k] == "pending"]
        if not pending:
            break
        sel = select_next_case(state, bank_by_task, pending, used_segids,
                                 n_min_per_task=cfg.N_min_per_task)
        if sel is None:
            break
        k, seg, s_val = sel
        used_segids[k].add(seg)
        # generate response from TRUE θ — Phase 4.3: BYTE-FAITHFUL to the
        # engine generative form core_mcmc.simulate_response
        # (p = λ + (1−2λ)·norm.cdf(η_true); plain norm.cdf, no η-clip).
        t_true = true_theta[_slot(k, "t")]; l_true = true_theta[_slot(k, "l")]
        eta_true = np.exp(l_true) * (s_val + t_true)
        p_true = LAPSE_RATE + _ONE_MINUS_2LAMBDA * float(norm.cdf(eta_true))
        Y = int(rng.random() < p_true)
        # update posterior
        update_state(state, k, s_val, Y)
        # check stopping
        p_pass_now = _p_pass(state.mu, state.Sigma_post, ell_star)
        for kk in range(K_):
            if state.decision[kk] != "pending":
                continue
            if state.n_per_task[kk] < cfg.N_min_per_task:
                continue
            if p_pass_now[kk] >= cfg.pass_p:
                state.decision[kk] = "pass"
            elif p_pass_now[kk] <= cfg.fail_p:
                state.decision[kk] = "fail"
            elif state.n_per_task[kk] >= cfg.N_max_per_task:
                state.decision[kk] = "refer"
        # snapshot
        state.mu_traj.append(state.mu.copy())
        state.sd_traj.append(_post_marg_sd(state.Sigma_post))
        state.p_pass_traj.append(p_pass_now)

    # anything still pending after N_max → refer
    for kk in range(K_):
        if state.decision[kk] == "pending":
            state.decision[kk] = "refer"

    return state


# ───────────────────────── deployment-state loader ─────────────────────────

# Phase 4.5: deployment-task -> Phase-3/v13 cert_config task key. The
# deployment ell* is now the SINGLE reference-faithful v13 lineage
# (decision 2026-05-18; plan D2 "resolve three-lineage conflict"),
# REPLACING PI's ell_thresholds.csv Youden lineage.
_V13_TASK_KEY = {
    "spike":   "combined_spike",
    "seizure": "sparcnet_sz",
    "lpd":     "sparcnet_lpd",
    "gpd":     "sparcnet_gpd",
    "lrda":    "sparcnet_lrda",
    "grda":    "sparcnet_grda",
    "other":   "sparcnet_iic",
}


def v13_ell_star(tasks):
    """Per-task ℓ* from calibration/cert_config.yaml v13
    (`ell_star_unified_v13`) — the single reference-faithful lineage
    (Phase 4.5). Fail-loud if any deployment task lacks a v13 mapping
    or the v13 key/ell_star is absent."""
    cfg = yaml.safe_load(
        Path(engine_paths.CALIB_CERT_CONFIG).read_text())
    v13 = cfg["ell_star_unified_v13"]["tasks"]
    out = []
    for t in tasks:
        if t not in _V13_TASK_KEY:
            raise KeyError(
                f"deployment task {t!r} has no v13 mapping "
                f"(_V13_TASK_KEY={sorted(_V13_TASK_KEY)})")
        vk = _V13_TASK_KEY[t]
        if vk not in v13 or v13[vk].get("ell_star") is None:
            raise KeyError(
                f"v13 cert_config missing ell_star for {vk!r} "
                f"(deployment task {t!r})")
        out.append(float(v13[vk]["ell_star"]))
    return np.array(out)


def load_deployment(uniform_ell_star: float | None = None):
    """Load the frozen deployment artifact.

    `uniform_ell_star`: if set, OVERRIDE the per-task ℓ* with this
    single scalar (experimental knob). Pass `None` (default) to use the
    per-task v13 reference-faithful ℓ*.

    Phase 4.5: ℓ* is the SINGLE v13 lineage (`v13_ell_star`, from
    calibration/cert_config.yaml `ell_star_unified_v13`), REPLACING PI's
    `ell_thresholds.csv` Youden lineage (decision D2 — one lineage for
    Mode-A, Mode-B AND deployment). `ell_thresholds.csv` is no longer
    consumed for ℓ* (it remains in the frozen artifact as legacy
    provenance only).

    K-agnostic (Phase 4.4-A): the task list+order is parsed
    AUTHORITATIVELY from the Sigma slot-name index; Σ/case_bank are
    cross-checked against it (fail-loud on mismatch). Return signature
    unchanged (3-tuple) so committed callers/studies are not perturbed.
    """
    tasks = deployment_task_names()
    Sigma = pd.read_csv(DEPLOY / "Sigma.csv", index_col=0).values
    assert Sigma.shape == (2 * len(tasks), 2 * len(tasks)), (
        f"Σ {Sigma.shape} inconsistent with {len(tasks)} parsed tasks")
    bank = pd.read_csv(DEPLOY / "case_bank.csv")
    if uniform_ell_star is not None:
        ell_star = np.full(len(tasks), float(uniform_ell_star))
    else:
        ell_star = v13_ell_star(tasks)        # Phase 4.5: single v13 lineage
    bank_by_task = {}
    for ki, t in enumerate(tasks):
        sub = bank[bank.task == t][
            ["seg_id", "s_mean", "s_sd"]].reset_index(drop=True)
        assert len(sub) > 0, f"case_bank.csv has no items for task {t!r}"
        bank_by_task[ki] = sub
    return Sigma, bank_by_task, ell_star


# ───────────────────────── demo / smoke test ─────────────────────────

def _build_demo_candidates(k_tasks):
    """Synthetic candidates at predefined skill profiles. K-agnostic
    (Phase 4.4-A): profiles are built for `k_tasks` (task-0 = spike;
    tasks 1..K-1 = the IIIC group)."""
    profiles = {
        "all_expert":     [(0.0, 1.5)] * k_tasks,
        "all_novice":     [(0.0, -0.5)] * k_tasks,
        "spike_strong_iiic_weak": [(0.0, 1.5)] + [(0.0, 0.0)] * (k_tasks - 1),
        "iiic_strong_spike_weak": [(0.0, -0.5)] + [(0.0, 1.5)] * (k_tasks - 1),
        "biased_expert":  [(0.5, 1.5)] * k_tasks,
        "borderline":     [(0.0, 0.5)] * k_tasks,
    }
    out = {}
    for name, pairs in profiles.items():
        theta = np.zeros(2 * k_tasks)
        for k, (t, l) in enumerate(pairs):
            theta[_slot(k, "t")] = t
            theta[_slot(k, "l")] = l
        out[name] = theta
    return out


def main():
    Sigma, bank_by_task, ell_star = load_deployment()
    tasks = deployment_task_names()
    cfg = TestConfig.from_yaml()   # Phase 4.2: shipping-contract YAML
    rng = np.random.default_rng(0)
    cands = _build_demo_candidates(len(tasks))
    print(f"ℓ* per task: " + ", ".join(f"{t}={es:+.2f}"
                                          for t, es in zip(tasks, ell_star)))
    for name, theta in cands.items():
        state = simulate_candidate(theta, Sigma, ell_star, bank_by_task, cfg, rng)
        print(f"\n{name}:")
        print(f"  decisions: {dict(zip(tasks, state.decision))}")
        print(f"  trials per task: {dict(zip(tasks, state.n_per_task.tolist()))}")
        print(f"  final mu (t, l):")
        for k, t in enumerate(tasks):
            tt = state.mu[_slot(k, 't')]; ll = state.mu[_slot(k, 'l')]
            true_tt = theta[_slot(k, 't')]; true_ll = theta[_slot(k, 'l')]
            print(f"    {t:>8}: t̂={tt:+.2f} (true {true_tt:+.2f})  "
                   f"ℓ̂={ll:+.2f} (true {true_ll:+.2f})")


if __name__ == "__main__":
    main()
