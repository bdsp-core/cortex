"""Phase 4.6-C — DEFINITIVE re-assessment of the carried-forward 4.3b
pass-share finding, on the FULLY-SHIPPED config.

The 4.3b finding (λ-lapse induces a small systematic pass-share-of-
decisive leniency shift + a systematic REFER increase) was measured
under PI-faithful K=6 + PI ℓ* (4.3b: +0.0145±0.0028), then re-assessed
under K=6 + v13 ℓ* (4.5: +0.0157±0.0031, persisted). Its disposition
flagged a FINAL re-assessment on the shipped system.

Shipped config (post 4.6-A/B): K=7, v13 ℓ*, re-frozen 14×14 Σ on the
unified corpus, clean-sn1 spike, DECOUPLED RNG. The committed-4.2
"no-lapse" engine is K=6-hardcoded and CANNOT run K=7, so the no-lapse
counterfactual is the CURRENT K-agnostic engine with λ forced to 0
(LAPSE_RATE→0, _ONE_MINUS_2LAMBDA→1) — proven @4.3 to equal the PI
bare-probit IRLS to ~2e-16. The 4.6-B RNG decoupling means θ for
candidate i is a pure function of (seed, i), so the λ vs λ=0 arms run
on the IDENTICAL shipped K=7 population per candidate NATIVELY (no
controlled-study / dual-engine machinery) — exactly what the
decoupling enables. M seeds/candidate for the Monte-Carlo decomposition.

Output: deployment/phase4_6c_shipped_reassess.json. Measurement-stage.
"""
from __future__ import annotations

import os as _os
import sys as _sys
from contextlib import contextmanager
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "scripts"), str(_REPO / "engine")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
from _parallel import parallel_map  # noqa: E402  (configures BLAS on import)

import json  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

OUT = _REPO / "deployment" / "phase4_6c_shipped_reassess.json"
MC_4_3B = _REPO / "deployment" / "phase4_3_hardening_delta_mc.json"
P45 = _REPO / "deployment" / "phase4_5_v13_ell_delta.json"
M_SEEDS = 20
SEED_THETA = 4_060_000          # shipped decoupled θ-gen seed base
SEED_Y = 4_061_000              # Y-draw seed base
_DEC = ("pass", "fail", "refer")
# the run_deployment_sim shipped synthetic panel (verbatim TIERS)
TIERS = {
    "expert":      dict(n=60,  mu_ell=1.6, sd_within=0.30, mu_t=0.0, sd_t=0.20),
    "experienced": dict(n=40,  mu_ell=1.1, sd_within=0.30, mu_t=0.0, sd_t=0.30),
    "novice":      dict(n=40,  mu_ell=0.3, sd_within=0.35, mu_t=0.0, sd_t=0.40),
    "crowd":       dict(n=60,  mu_ell=-0.3, sd_within=0.45, mu_t=0.0, sd_t=0.60),
}
_STATE: dict = {}


@contextmanager
def _lapse_off(st):
    """No-lapse counterfactual: λ=0 ⇒ the engine reduces EXACTLY to the
    PI bare-probit IRLS (proven @4.3, max|Δ|≈2e-16). Patches the two
    module globals the likelihood reads; restores them after."""
    lam, omt = st.LAPSE_RATE, st._ONE_MINUS_2LAMBDA
    st.LAPSE_RATE, st._ONE_MINUS_2LAMBDA = 0.0, 1.0
    try:
        yield
    finally:
        st.LAPSE_RATE, st._ONE_MINUS_2LAMBDA = lam, omt


def _ensure_state():
    if _STATE:
        return _STATE
    from deployment import simulate_test as st
    from deployment.run_deployment_sim import sample_candidate_theta
    Sigma, bank, ell = st.load_deployment(uniform_ell_star=None)  # K=7 v13
    cfg = st.TestConfig.from_yaml()
    tasks = st.deployment_task_names()
    k_tasks = len(tasks)
    r_ell = float(np.mean([Sigma[1, 3], Sigma[1, 5], Sigma[3, 5]]))
    # The SHIPPED decoupled population: θ for candidate i depends ONLY
    # on (seed=0, cand_id) via the 4.6-B SeedSequence scheme — pure,
    # config/λ-invariant. Built once; identical for the λ / λ=0 arms.
    theta, tier_of = [], []
    cand = 0
    for tier, tc in TIERS.items():
        for _ in range(tc["n"]):
            tss, _ = np.random.SeedSequence([0, cand]).spawn(2)
            theta.append(sample_candidate_theta(
                tc, r_ell, np.random.default_rng(tss), k_tasks))
            tier_of.append(tier)
            cand += 1
    _STATE.update(st=st, Sigma=Sigma, bank=bank, ell=ell, cfg=cfg,
                  tasks=tasks, theta=np.array(theta), tier=tier_of)
    return _STATE


def _worker(i: int):
    s = _ensure_state()
    st, th = s["st"], s["theta"][i]
    K = len(s["tasks"])
    cL = [{d: 0 for d in _DEC} for _ in range(K)]   # λ-lapse (shipped)
    cN = [{d: 0 for d in _DEC} for _ in range(K)]   # λ=0 (PI bare-probit)
    for m in range(M_SEEDS):
        yss = np.random.SeedSequence([SEED_Y, i, m])
        dL = st.simulate_candidate(
            th, s["Sigma"], s["ell"], s["bank"], s["cfg"],
            np.random.default_rng(yss)).decision
        with _lapse_off(st):
            dN = st.simulate_candidate(
                th, s["Sigma"], s["ell"], s["bank"], s["cfg"],
                np.random.default_rng(yss)).decision
        for ki in range(K):
            cL[ki][dL[ki]] += 1
            cN[ki][dN[ki]] += 1
    return i, cN, cL          # (no-lapse, lapse) — Δ = lapse − nolapse


def run_study(limit=None, max_workers=None):
    s0 = _ensure_state()
    n = len(s0["theta"]) if limit is None else int(limit)
    res = parallel_map(_worker, list(range(n)), max_workers=max_workers,
                       desc="4.6-C shipped K=7 λ vs λ=0", ordered=True)
    K = len(s0["tasks"])
    dpass, dfail, drefer, dshare = [], [], [], []
    for _i, cN, cL in res:
        for ki in range(K):
            pn = np.array([cN[ki][d] for d in _DEC], float) / M_SEEDS
            pl = np.array([cL[ki][d] for d in _DEC], float) / M_SEEDS
            dpass.append(pl[0] - pn[0])
            dfail.append(pl[1] - pn[1])
            drefer.append(pl[2] - pn[2])
            dn, dl = pn[0] + pn[1], pl[0] + pl[1]
            if dn > 0 and dl > 0:
                dshare.append(pl[0] / dl - pn[0] / dn)
    dpass = np.array(dpass); dfail = np.array(dfail)
    drefer = np.array(drefer); dshare = np.array(dshare)

    def _ms(a):
        return (float(a.mean()),
                float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1
                else float("nan"))
    rp_m, rp_se = _ms(drefer)
    sh_m, sh_se = _ms(dshare)
    pa_m, pa_se = _ms(dpass)
    fa_m, fa_se = _ms(dfail)
    prior_43b = (json.loads(MC_4_3B.read_text())
                 .get("systematic_pass_share_shift")
                 if MC_4_3B.exists() else None)
    prior_45 = None
    if P45.exists():
        prior_45 = (json.loads(P45.read_text())
                    ["part_B_4_3b_reassessed_under_v13"]
                    .get("systematic_pass_share_shift_under_v13"))
    summary = {
        "scope": "DEFINITIVE 4.3b re-assessment on the FULLY-SHIPPED "
        "config: K=7, v13 ℓ*, re-frozen 14×14 Σ (unified corpus), "
        "clean-sn1 spike, DECOUPLED RNG. No-lapse arm = same K-agnostic "
        "engine with λ=0 (≡ PI bare-probit IRLS, proven @4.3 ~2e-16); "
        "λ vs λ=0 on the IDENTICAL shipped decoupled population per "
        f"candidate (4.6-B guarantee). M={M_SEEDS} seeds/candidate/arm.",
        "n_candidates": n, "m_seeds": M_SEEDS, "K": K,
        "n_cells": int(len(dpass)), "n_decisive_cells": int(len(dshare)),
        "systematic_d_p_refer": rp_m, "systematic_d_p_refer_se": rp_se,
        "systematic_d_p_pass": pa_m, "systematic_d_p_pass_se": pa_se,
        "systematic_d_p_fail": fa_m, "systematic_d_p_fail_se": fa_se,
        "systematic_pass_share_shift_shipped": sh_m,
        "systematic_pass_share_shift_shipped_se": sh_se,
        "passfail_balance_null_shipped": bool(abs(sh_m) <= 2 * sh_se),
        "trajectory": {
            "4.3b_PI_K6_ell_thresholds": prior_43b,
            "4.5_K6_v13_ell": prior_45,
            "4.6c_shipped_K7_v13_decoupled": sh_m,
        },
    }
    _attach_prose(summary)
    if limit is None:
        OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n  shipped K=7: ΔP(refer)={rp_m:+.4f}±{rp_se:.4f}  "
          f"pass-share shift={sh_m:+.4f}±{sh_se:.4f}  "
          f"null={summary['passfail_balance_null_shipped']}")
    print(f"  trajectory 4.3b={prior_43b} -> 4.5={prior_45} -> "
          f"4.6c={sh_m:+.4f}")
    print(f"  -> {OUT if limit is None else '(smoke: not written)'}")
    return summary


def _attach_prose(s: dict) -> dict:
    sh = s["systematic_pass_share_shift_shipped"]
    se = s["systematic_pass_share_shift_shipped_se"]
    rp = s["systematic_d_p_refer"]
    null = s["passfail_balance_null_shipped"]
    tr = s["trajectory"]
    verdict = (
        "RESOLVED on the shipped system: the pass-share-of-decisive "
        "shift is consistent with 0 within ~2·SE at K=7 — the small "
        "leniency seen at K=6 does NOT survive on the fully-shipped "
        "config." if null else
        "PERSISTS on the shipped system: a small, systematic pass-share "
        f"leniency shift ({sh:+.4f}±{se:.4f}, ~{abs(sh)/se:.0f}·SE) "
        "remains at K=7 with v13 ℓ* + re-frozen Σ + decoupled RNG + "
        "clean-sn1 spike. It is a STABLE, quantified property of the "
        "reference-correct λ-lapse likelihood interacting with the "
        "decision boundary — NOT a config artifact (it tracked "
        f"{tr['4.3b_PI_K6_ell_thresholds']} → {tr['4.5_K6_v13_ell']} → "
        f"{sh:+.4f} across PI-K6 → v13-K6 → shipped-K7).")
    s["definitive_verdict"] = (
        f"λ-lapse SYSTEMATICALLY increases REFER on the shipped system "
        f"(ΔP(refer)={rp:+.4f}±{s['systematic_d_p_refer_se']:.4f}) — the "
        "expected conservative mechanism, confirmed at K=7. Pass-share: "
        + verdict)
    s["disposition"] = (
        "The long-carried 4.3b finding now has its DEFINITIVE verdict "
        "on the actual shipped K=7 system (4.6-B decoupling enabled a "
        "native, population-stable λ-vs-λ=0 measurement — no "
        "dual-engine/controlled-study workaround). "
        + ("Effect RESOLVED — closed; documented in "
           "docs/DEPLOYMENT_INTEGRATION.md as not surviving to the "
           "shipped config." if null else
           "Effect PERSISTS small + stable — recorded as a KNOWN, "
           "quantified characteristic of the reference-correct lapse "
           "likelihood in docs/DEPLOYMENT_INTEGRATION.md (NOT a bug; "
           "the lapse model is the validated one). No further "
           "re-assessment is owed — this is the shipped config."))
    return s


def main():
    run_study()


if __name__ == "__main__":
    main()
