"""Phase 7 sub-step 3-B/C — D6 real-rater replay drivers.

The replay harness drives the deployment engine (and Mode-A engine)
with held-out *real* rater response sequences instead of Bernoulli
draws. User-locked design (2026-05-19): strict constrained-bank
("Design A").

  * `run_deployment_replay.py` (7.3-B) — wraps
    `simulate_candidate(y_source=...)` (the optional Phase-7 hook)
    with strict-A bank filtering + Y lookup; produces per-task and
    per-candidate PASS/FAIL/REFER + E[n] tables. Includes the
    fitted-θ Bernoulli paired comparator (7.3-C's headline arm).
"""
