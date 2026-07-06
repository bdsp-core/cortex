"""M12/V5 — rung-5 (in-scratch part): real-session replay of the measurement
layer.

Replays ALL THREE recorded real eval sessions (the only real response data in
scratch) through the trainer-side TaskFilter and validates the measurement
layer on real humans:

  1. Convergent validity — recovered ℓ̂ should track empirical accuracy across
     the 3 raters (overall acc 0.235 / 0.52 / 0.57) and across rater×task
     cells (Spearman).
  2. Cross-validation vs PRODUCTION — the summary CSVs record the production
     engine's own per-task ℓ estimates (ell_domain1..7); the scratch filter's
     static-replay ℓ̂ should agree (the filter and the engine share the
     observation model; replay items/responses are identical).
  3. Safety regression — the careless rater (acc 0.235) certifies nothing
     (extends test_step3's single-session check to all sessions).
  4. F28 exposure on real data — replaying with the trainer's ASSUMED
     dynamics (feedback=True, as a trainer session would run) vs the static
     model quantifies how far the dynamics prior drags beliefs on real
     non-learning (no-feedback) data.

Replay protocol: items resolved via BankAdapter.task_pool (full pools,
matching test_step3); static replay = rule="static" filter (pure Bayes
measurement); dynamic replay = assumed trainer params (D7 defaults).

Run: python3 -m studies.study_real_replay  → figures/data_real_replay.npz
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from training.bank_adapter import BankAdapter, ELL_STAR, SIGMA_INF, TASK_CODES
from training.learner_sim import LearnerParams
from training.training_filter import TaskFilter

SESSIONS = (
    "859ca73f-ba36-4670-9bfb-2072e252c595",   # careless (acc 0.235)
    "97d70997-42c7-46bd-96eb-1487c52176ff",
    "f658855c-c9cc-4f8c-8161-d6c6f6a32200",
)
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "sessions")


def replay(trials, ad, *, dynamic=False, n_part=600):
    """Replay one session's trials through per-task filters; returns per-task
    (n, acc, ell_hat, sd_ell, mastered)."""
    code_idx = {c: i for i, c in enumerate(TASK_CODES)}
    filts, stats = {}, {}
    for k in range(7):
        params = LearnerParams(
            alpha_t=(0.2 if dynamic else 0.0),
            alpha_sigma=(0.06 if dynamic else 0.0),
            sigma_inf=SIGMA_INF[k], q_t=(0.04 if dynamic else 0.0),
            q_sigma=(0.02 if dynamic else 0.0),
            rho=0.6, rule=("soft" if dynamic else "static"))
        filts[k] = TaskFilter(np.zeros(n_part) + 1.0 * np.random.default_rng(
            7000 + k).standard_normal(n_part),
            np.zeros(n_part) + 1.0 * np.random.default_rng(
            8000 + k).standard_normal(n_part),
            params, ess_frac=0.5, seed=9000 + k)
        stats[k] = [0, 0]                      # n, n_correct
    for _, row in trials.iterrows():
        k = code_idx[row["task_code"]]
        pool = ad.task_pool(k)
        hit = np.where(pool.seg_id == int(row["seg_id"]))[0]
        if hit.size == 0:
            continue
        y_star = int(pool.y_star[hit[0]])
        s, s_sd = float(pool.s_mean[hit[0]]), float(pool.s_sd[hit[0]])
        y = int(row["response_y"])
        # real eval delivered NO feedback: f-gate off in the dynamic replay
        filts[k].step(s, y, y_star, s_sd=s_sd, feedback=dynamic)
        stats[k][0] += 1
        stats[k][1] += int(y == y_star)
    out = {}
    for k in range(7):
        ml, sd = filts[k].mean()[1], filts[k].sd()[1]
        mast = filts[k].is_mastered(ELL_STAR[k], sd_floor=0.30)
        out[k] = dict(n=stats[k][0],
                      acc=(stats[k][1] / stats[k][0] if stats[k][0] else np.nan),
                      ell_hat=ml, sd_ell=sd, mastered=mast)
    return out


def main():
    ad = BankAdapter()
    rows = []
    for ses in SESSIONS:
        trials = pd.read_csv(os.path.join(HERE, f"{ses}_trials_anonymized.csv"))
        summ = pd.read_csv(os.path.join(HERE, f"{ses}_summary_anonymized.csv"))
        acc_total = float(summ["accuracy"].iloc[0])
        st = replay(trials, ad, dynamic=False)
        dy = replay(trials, ad, dynamic=True)
        for k in range(7):
            prod_ell = float(summ[f"ell_{TASK_CODES[k]}"].iloc[0]) \
                if f"ell_{TASK_CODES[k]}" in summ.columns else np.nan
            rows.append(dict(session=ses[:8], task=k, acc_total=acc_total,
                             prod_ell=prod_ell, **{f"st_{a}": b for a, b in
                                                   st[k].items()},
                             **{f"dy_{a}": b for a, b in dy[k].items()}))
    df = pd.DataFrame(rows)
    ok = df["st_n"] >= 15
    print(df[["session", "task", "st_n", "st_acc", "st_ell_hat", "prod_ell",
              "dy_ell_hat", "st_mastered"]].to_string(index=False,
                                                      float_format="%.3f"))
    # 1. convergent validity
    rho_acc = spearmanr(df.loc[ok, "st_acc"], df.loc[ok, "st_ell_hat"])
    # 2. production cross-validation
    okp = ok & df["prod_ell"].notna()
    rho_prod = spearmanr(df.loc[okp, "st_ell_hat"], df.loc[okp, "prod_ell"])
    rmse_prod = float(np.sqrt(np.mean(
        (df.loc[okp, "st_ell_hat"] - df.loc[okp, "prod_ell"]) ** 2)))
    # per-session means (ordering check)
    per = df[ok].groupby("session").agg(acc=("acc_total", "first"),
                                        ell=("st_ell_hat", "mean"))
    per = per.sort_values("acc")
    print(f"\nconvergent validity: Spearman(acc, ℓ̂) = {rho_acc.statistic:.3f} "
          f"(p={rho_acc.pvalue:.2g}, n={int(ok.sum())})")
    print(f"production cross-val: Spearman = {rho_prod.statistic:.3f}, "
          f"RMSE(ℓ̂ − prod ℓ) = {rmse_prod:.3f} (n={int(okp.sum())})")
    print("session ordering (acc → mean ℓ̂):")
    print(per.to_string(float_format="%.3f"))
    monotone = bool(per["ell"].is_monotonic_increasing)
    careless_mastered = int(df[(df["session"] == SESSIONS[0][:8])
                               ]["st_mastered"].sum())
    drag = float((df.loc[ok, "dy_ell_hat"] - df.loc[ok, "st_ell_hat"]).mean())
    print(f"rater ordering by mean ℓ̂ matches accuracy ordering: {monotone}")
    print(f"careless rater masteries (static replay): {careless_mastered}")
    print(f"F28 exposure on real data: mean(dynamic ℓ̂ − static ℓ̂) = {drag:+.3f}"
          " (no-feedback replay: σ-pull is f-gated OFF — expect ≈ small)")
    np.savez("figures/data_real_replay.npz",
             table=df.drop(columns=["session"]).to_numpy(dtype=float),
             columns=np.array([c for c in df.columns if c != "session"]),
             sessions=np.array([r["session"] for r in rows]),
             rho_acc=rho_acc.statistic, rho_prod=rho_prod.statistic,
             rmse_prod=rmse_prod, monotone=monotone,
             careless_mastered=careless_mastered, dyn_drag=drag)
    print("saved figures/data_real_replay.npz")


if __name__ == "__main__":
    main()
