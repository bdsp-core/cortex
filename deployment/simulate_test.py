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
UNIFIED-MERGE PROVENANCE (Phase 4.1, 2026-05-18). This is a FAITHFUL,
PATH-ONLY port of the PI repo's `scripts/simulate_test.py`. The ONLY
substantive change vs the PI source is the PI hardcoded absolute-ROOT /
DEPLOY block (an absolute path under the PI author's home dir into
`data/deployment_prior`), replaced by an `engine_paths.DEPLOYMENT_PRIOR`
shim (UNIFIED_REPO_MERGE_PLAN.md Phase 4 step 2; zero absolute paths —
a CI test enforces none survive, including in this header). All
numerics — K=6, the UNHARDENED probit IRLS, ℓ* from ell_thresholds.csv
— are kept BYTE-FAITHFUL so Phase 4.1 proves PORT fidelity (reproduces
the carried `data/deployment_prior/sim/` reference at seed=0). The
intended changes (λ-lapse hardening 4.3, K=7 4.4, v13 ℓ* 4.5,
re-freeze 4.6) are LAYERED in later sub-steps, each separately gated.
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

DEPLOY = Path(engine_paths.DEPLOYMENT_PRIOR)

TASKS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda"]
K = len(TASKS)
DIM = 2 * K


# ───────────────────────── helpers ─────────────────────────

def _slot(k, parm):
    """Return the global slot index for parm in {'t','l'} on task k."""
    return 2 * k + (0 if parm == "t" else 1)


def irls_w_z(eta: np.ndarray, Y: np.ndarray):
    """Probit IRLS weight and pseudo-observation, η clipped for stability."""
    eta_c = np.clip(eta, -6.0, 6.0)
    phi = norm.pdf(eta_c)
    Phi = np.clip(norm.cdf(eta_c), 1e-10, 1 - 1e-10)
    phi_c = np.clip(phi, 1e-10, None)
    w = phi_c ** 2 / (Phi * (1 - Phi))
    z = eta_c + (Y - Phi) / phi_c
    return w, z


def expected_p(mu, Sigma_post, k, s):
    """Approximate predictive P(Y=1) under current posterior using
    standard probit-Gaussian conditional: ≈ Φ(η_mean / sqrt(1+v))."""
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
    return norm.cdf(eta / np.sqrt(1 + max(v, 0.0)))


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
    """Per-task P(ℓ_k > ℓ*_k | data) under Gaussian marginal."""
    out = np.zeros(K)
    for k in range(K):
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

    # 1 Newton step at current μ
    t = mu[j_t]; l = mu[j_l]
    el = np.exp(l)
    eta = el * (s_val + t)
    w, z = irls_w_z(np.array([eta]), np.array([Y]))
    w = float(w[0]); z = float(z[0])
    # x has two non-zero entries
    x = np.zeros(DIM)
    x[j_t] = el
    x[j_l] = el * (s_val + t)

    Sigma_post_inv = np.linalg.inv(Sigma_post)
    # Σ⁻¹ µ_new = Σ⁻¹ µ_old + w·x·z (working response in η-units relative to flat)
    # Equivalently posterior on linearized residual:
    Sigma_post_inv_new = Sigma_post_inv + w * np.outer(x, x)
    Sigma_post_new = np.linalg.inv(Sigma_post_inv_new)
    mu_new = Sigma_post_new @ (Sigma_post_inv @ mu + w * x * z)
    # Clamp ℓ inside [-3, 3] so e^ℓ stays bounded for next-trial info eval
    for kk in range(K):
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
        # Predictive p at current posterior; weight w at η_mean
        eta_c = np.clip(eta, -6.0, 6.0)
        phi = norm.pdf(eta_c)
        Phi = np.clip(norm.cdf(eta_c), 1e-6, 1 - 1e-6)
        w = phi ** 2 / (Phi * (1 - Phi))
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
    """Simulate one candidate going through the adaptive multi-task test."""
    if rng is None:
        rng = np.random.default_rng()
    state = TestState(
        mu=np.zeros(DIM),
        Sigma_post=Sigma_prior.copy(),
        n_per_task=np.zeros(K, dtype=int),
        decision=["pending"] * K,
        history=[],
    )
    used_segids = [set() for _ in range(K)]
    # snapshot at trial 0
    state.mu_traj.append(state.mu.copy())
    state.sd_traj.append(_post_marg_sd(state.Sigma_post))
    state.p_pass_traj.append(_p_pass(state.mu, state.Sigma_post, ell_star))

    for n in range(cfg.N_max):
        pending = [k for k in range(K) if state.decision[k] == "pending"]
        if not pending:
            break
        sel = select_next_case(state, bank_by_task, pending, used_segids,
                                 n_min_per_task=cfg.N_min_per_task)
        if sel is None:
            break
        k, seg, s_val = sel
        used_segids[k].add(seg)
        # generate response from TRUE θ
        t_true = true_theta[_slot(k, "t")]; l_true = true_theta[_slot(k, "l")]
        eta_true = np.exp(l_true) * (s_val + t_true)
        p_true = norm.cdf(eta_true)
        Y = int(rng.random() < p_true)
        # update posterior
        update_state(state, k, s_val, Y)
        # check stopping
        p_pass_now = _p_pass(state.mu, state.Sigma_post, ell_star)
        for kk in range(K):
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
    for kk in range(K):
        if state.decision[kk] == "pending":
            state.decision[kk] = "refer"

    return state


# ───────────────────────── deployment-state loader ─────────────────────────

def load_deployment(uniform_ell_star: float | None = None):
    """Load the frozen deployment artifact.

    `uniform_ell_star`: if set (default = 0.62, from spike's well-anchored
    Youden calibration), override the per-task ℓ*_k empirical thresholds.
    Reason: the IIIC subtype thresholds derived from our pool have near-
    chance AUROC (~0.5) because the pool is ~95% experts, so a 1.9–2.9
    ℓ* is set by a tiny non-expert tail and is not deployment-realistic.
    For a deployable demo we anchor all tasks to spike's 0.62 cutoff.
    Pass `None` to use the per-task empirical thresholds.
    """
    Sigma = pd.read_csv(DEPLOY / "Sigma.csv", index_col=0).values
    bank  = pd.read_csv(DEPLOY / "case_bank.csv")
    thr   = pd.read_csv(DEPLOY / "ell_thresholds.csv").set_index("task")
    if uniform_ell_star is not None:
        ell_star = np.full(len(TASKS), float(uniform_ell_star))
    else:
        ell_star = np.array([float(thr.loc[t, "ell_star"]) for t in TASKS])
    bank_by_task = {}
    for ki, t in enumerate(TASKS):
        bank_by_task[ki] = bank[bank.task == t][["seg_id", "s_mean", "s_sd"]].reset_index(drop=True)
    return Sigma, bank_by_task, ell_star


# ───────────────────────── demo / smoke test ─────────────────────────

def _build_demo_candidates():
    """Synthetic candidates at predefined skill profiles."""
    profiles = {
        "all_expert":     [(0.0, 1.5)] * K,
        "all_novice":     [(0.0, -0.5)] * K,
        "spike_strong_iiic_weak": [(0.0, 1.5)] + [(0.0, 0.0)] * 5,
        "iiic_strong_spike_weak": [(0.0, -0.5)] + [(0.0, 1.5)] * 5,
        "biased_expert":  [(0.5, 1.5)] * K,
        "borderline":     [(0.0, 0.5)] * K,
    }
    out = {}
    for name, pairs in profiles.items():
        theta = np.zeros(DIM)
        for k, (t, l) in enumerate(pairs):
            theta[_slot(k, "t")] = t
            theta[_slot(k, "l")] = l
        out[name] = theta
    return out


def main():
    Sigma, bank_by_task, ell_star = load_deployment()
    cfg = TestConfig.from_yaml()   # Phase 4.2: shipping-contract YAML
    rng = np.random.default_rng(0)
    cands = _build_demo_candidates()
    print(f"ℓ* per task: " + ", ".join(f"{t}={es:+.2f}"
                                          for t, es in zip(TASKS, ell_star)))
    for name, theta in cands.items():
        state = simulate_candidate(theta, Sigma, ell_star, bank_by_task, cfg, rng)
        print(f"\n{name}:")
        print(f"  decisions: {dict(zip(TASKS, state.decision))}")
        print(f"  trials per task: {dict(zip(TASKS, state.n_per_task.tolist()))}")
        print(f"  final mu (t, l):")
        for k, t in enumerate(TASKS):
            tt = state.mu[_slot(k, 't')]; ll = state.mu[_slot(k, 'l')]
            true_tt = theta[_slot(k, 't')]; true_ll = theta[_slot(k, 'l')]
            print(f"    {t:>8}: t̂={tt:+.2f} (true {true_tt:+.2f})  "
                   f"ℓ̂={ll:+.2f} (true {true_ll:+.2f})")


if __name__ == "__main__":
    main()
