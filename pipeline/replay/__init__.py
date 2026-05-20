"""Phase 7 sub-step 3 — D6 real-rater replay infrastructure.

The replay harness drives both engines (deployment `simulate_test.py`
and Mode-A `core_mcmc.run_session_mcmc_auroc`) with held-out *real*
rater response sequences instead of Bernoulli draws.

User-locked design (2026-05-19): strict constrained-bank ("Design A")
— the engine selects EV-optimal from the rater's actually-scored
segments; Y comes from `data/labels/labels.csv`; if the rater's
personal bank exhausts before the engine reaches a decision, that
task's verdict is REFER.

This package:
  * `build_rater_replay_bank.py` (7.3-A) — produces the rater_replay
    bank by joining `data/labels/labels.csv` (with the erratum-
    correct {bipd,birds,other}→other mapping for IIIC tasks) with
    `data/deployment_prior/case_bank.csv` (the K=7 frozen item bank).
"""
