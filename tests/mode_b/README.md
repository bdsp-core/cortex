# Mode-B tests (DEPRECATED for Paper 1)

F3.3 (2026-05-15): these tests exercise the **deprecated** Mode-B
binary-certification engine in `engine_mode_b.py`
(`run_session_mcmc_certification`, the Berger-IUT joint stopping rule,
the boundary prior, the Fisher-info / n_min guards, the Šidák correction,
and the removed-virtual-cert regressions).

Under the Multi-AUROC Precision Protocol reframe (2026-05-15), Mode-B is
NOT the Paper-1 production path.  These tests are preserved (still run in
the default `pytest` invocation via `testpaths = tests`) as groundwork
for Paper 2 (binary credentialing with a prospectively validated expert
panel).  They are kept green so the Mode-B engine remains a trustworthy
starting point when Paper 2 work resumes.

| File | What it pins |
|---|---|
| `test_sidak.py` | Šidák `(1-α)^(1/K)` arithmetic; cert_config `mode_b_legacy.stop_thresh_sidak` |
| `test_mcse_buffer.py` | FIX-T1.3 MCSE-buffered PASS/FAIL stopping; `mcse_final` return key |
| `test_virtual_cert_bound.py` | bivariate-normal Fréchet bound math (sz→grda) |
| `test_virtual_cert_removed.py` | FIX-T1.1 virtual-pairs deprecation; no virtual-pass advantage |
| `test_oc_pilot.py` | small operating-characteristics pilot (`@slow`; `test_oc_borderline_pass_rate` `xfail` — engine-doesn't-terminate, addressed by the Multi-AUROC reframe) |
| `test_smoke.py` | end-to-end Mode-B hier + brute session contract |

Shared fixtures (`synthetic_true_params`, `true_params_array`,
`cert_config`) and the `slow`/`nightly` markers come from the parent
`tests/conftest.py` — pytest applies conftest hierarchically, so no
duplication is needed here.
