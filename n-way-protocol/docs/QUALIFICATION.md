# Qualification contract

The included harness is executable adaptive-selection/frontier infrastructure,
not a completed scientific qualification. Its default stopping point is the
configured direct-domain cap. With `--stopping precision` the harness instead
consults the UNCHANGED production Precision policy after every update through
the `src/precision_cli.ts` sidecar (built by `scripts/build_precision_cli.sh`,
golden-parity-tested against a direct TypeScript invocation of
`precision_bridge.ts`), with the direct-domain cap retained only as a hard
safety cap; `--real-bank` swaps the synthetic generator for a seeded draw from
the staged served-bank axes. A promotion run must use that join, be
preregistered, and use a newly qualified response artifact.

## Prerequisites

- A versioned DR07 refit with at least 32 exact same-state joint draws plus
  state zero, followed by the governed rank/conditioning gate.
- Leakage-controlled distractor fitting: held-out readers and patient/source
  groups; the participant's response must not contribute to its item axis.
- Frozen `beta`, distractor lapse, prior/correlation artifacts, source hashes,
  seeds, populations, and analysis code.
- Owner-approved noninferiority margins set before the locked run.

## Required run families

1. Likelihood and parameter-recovery SBC over low, medium, and high skill and
   bias, correlated/heterogeneous profiles, and signal uncertainty.
2. Misspecification stress: lapse drift, distractor-scale drift, correlated
   criteria, source shift, and noisy/miscalibrated evidence axes.
3. Full adaptive comparison on the served bank using each arm's own selector
   and the unchanged Precision stopping policy.
4. Fixed-sequence historical shadow replay for real-response model checking.
   This does not qualify adaptive selection.
5. Browser/device performance and exact serial-versus-speculative parity.

## Promotion metrics

- Skill and bias posterior mean bias and RMSE.
- 95% interval coverage and width for both parameter blocks.
- The existing categorical coverage guardrail versus binary, plus an absolute
  nominal-coverage gate with Monte Carlo confidence intervals.
- False pass, false fail, refer, cap, and bank-exhaustion rates.
- Median, p90, and p95 total and per-domain burden.
- ESS, resampling frequency, MH acceptance, ancestry, and MCSE reliability.
- Selector regret versus exhaustive evaluation on tractable banks.
- p50/p95/max answer-to-item latency, memory, worker failure, and fallback.

The 1,200-particle, ESS 0.5, 30-MH, and MCSE-inflation values must be
requalified. They are not inherited merely because the stopping rule is
unchanged.

The default planted-population harness uses a shifted/narrower truth
distribution as a prior-mismatch stress test, so its rank histograms are not
formal SBC. Pass `--formal-sbc` to draw planted parameters from the inference
prior and make the rank-uniformity interpretation valid.

## Current research evidence

The post-freeze evidence and limitations are recorded in
`reports/POST_FREEZE_RD_FINDINGS.md`. The most powered development run uses
2,000 formal-SBC seeds at 1,200 particles/30 MH steps and compares the frozen
selector with Fisher augmentation under the same nine-draw artifact mixture.
It supports further qualification of that combined challenger, not production
promotion. In particular, relative coverage was retained while absolute bias
coverage was 0.94542 (95% CI `[0.94118, 0.94965]`).

Development noninferiority checks, point estimates, or intervals containing a
nominal value do not replace the preregistered absolute-coverage gate. Failed
or neutral challengers remain in the reports so they cannot be rediscovered
and selectively presented later.

The subsequent fully integrated comparison is recorded in
`reports/FINAL_INTEGRATION_COMPARISON.md`. At 2,000 matched seeds it passed the
relative coverage and paired-tightening development gates: skill width was
8.04% lower and bias width 2.58% lower. Integrated-minus-frozen coverage was
-0.00342 for skill and -0.00092 for bias. Absolute integrated coverage remained
0.94250 and 0.94550, so the governed absolute gate remains open.

## Resource policy

The parallel harness reserves two logical CPUs and 20% of available memory.
Each process owns one BLAS thread. On the current 48-thread host the CPU ceiling
is therefore 46 worker processes, further limited by replicate count and the
memory estimate.
