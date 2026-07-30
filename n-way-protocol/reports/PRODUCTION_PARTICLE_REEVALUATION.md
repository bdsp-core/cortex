# Production particle-profile re-evaluation

Status: research-only, unqualified response artifact. No production promotion
is authorized by these results.

Run date: 2026-07-20. Host: 48 logical CPUs. Parallel runs used 46 workers,
one BLAS thread per worker, two reserved CPUs, and at least 20% memory
headroom.

## Question

Does replacing the smoke profile (192 particles, 2 MH steps) with the current
production numerical profile (1,200 particles, 30 MH steps) correct the
categorical undercoverage while retaining useful interval tightening?

## A. Matched-frontier population stress

This is the direct comparison to `research_smoke_48.json`. It holds the
4-direct-question/domain frontier, 72-segment bank, selector breadth, response
model, and planted shifted/narrow population fixed. Only particles and MH steps
are promoted to 1,200/30. Replicates increase from 48 to 192.

Command:

```text
bash scripts/qualification.sh smoke --replicates 192 \
  --particles 1200 --mh-steps 30
```

Each arm contributes 1,152 domain-level skill observations and 1,152 bias
observations.

| Metric | Binary | Categorical F1 | N-way change |
|---|---:|---:|---:|
| Skill coverage | 0.9905 | 0.9679 | -0.0226 |
| Bias coverage | 0.9835 | 0.9757 | -0.0078 |
| Skill interval width | 3.5186 | 2.3601 | 32.9% tighter |
| Bias interval width | 2.8187 | 2.3281 | 17.4% tighter |
| Skill RMSE | 0.6163 | 0.5026 | 18.4% lower |
| Bias RMSE | 0.5191 | 0.4563 | 12.1% lower |
| Mean resamples/session | 5.19 | 10.27 | higher categorical demand |
| Mean MH acceptance | 0.235 | 0.220 | both above 0.20 floor |
| Mean distinct ancestry | 0.457 | 0.432 | both above 0.35 floor |

Seed-clustered 95% intervals:

- Categorical skill coverage: `[0.9580, 0.9778]`.
- Categorical bias coverage: `[0.9661, 0.9853]`.
- Skill coverage difference, n-way minus binary: `-0.0226`, 95% CI
  `[-0.0337, -0.0115]`. The point estimate passes the `-0.03` guardrail, but
  the confidence-bound version narrowly does not.
- Bias coverage difference: `-0.0078`, 95% CI `[-0.0197, 0.0041]`; both point
  and confidence-bound guardrails pass.

The binary arm is strongly conservative at this short frontier. Absolute
categorical coverage is also conservative. The remaining relative skill-gate
failure is driven by comparison to binary's 99.0% coverage, not by categorical
undercoverage.

The planted population in this run is deliberately shifted/narrower than the
inference prior. Its posterior-rank plots are therefore a population-mismatch
stress diagnostic, not formal SBC.

## B. Prior-predictive formal SBC

The same 192-replicate, 1,200-particle, 30-MH, four-question frontier is rerun
with planted `t` and `l` drawn from the inference prior. This makes posterior
rank uniformity and SBC coverage interpretations valid.

Command:

```text
bash scripts/qualification.sh smoke --replicates 192 \
  --particles 1200 --mh-steps 30 --formal-sbc
```

| Metric | Binary | Categorical F1 | N-way change |
|---|---:|---:|---:|
| Skill coverage | 0.9592 | 0.9566 | -0.0026 |
| Bias coverage | 0.9531 | 0.9427 | -0.0104 |
| Skill interval width | 3.4978 | 2.4280 | 30.6% tighter |
| Bias interval width | 2.8614 | 2.4386 | 14.8% tighter |
| Skill RMSE | 0.8057 | 0.5691 | 29.4% lower |
| Bias RMSE | 0.6847 | 0.6086 | 11.1% lower |
| Skill rank KS distance | 0.0177 | 0.0231 | both close to uniform |
| Bias rank KS distance | 0.0288 | 0.0237 | both close to uniform |

Seed-clustered 95% intervals:

- Categorical skill coverage: `[0.9449, 0.9683]`.
- Categorical bias coverage: `[0.9273, 0.9581]`.
- Skill coverage difference: `-0.0026`, 95% CI `[-0.0170, 0.0118]`.
- Bias coverage difference: `-0.0104`, 95% CI `[-0.0271, 0.0062]`.
- Both parameter blocks pass the `categorical >= binary - 0.03`
  noninferiority guardrail using the lower confidence bound.

The categorical bias point estimate is 0.943 rather than 0.95, but its
clustered interval contains 0.95 and its rank diagnostic is close to uniform.
This run does not establish an absolute-coverage promotion margin; that margin
must be preregistered for the powered qualification.

## C. Longer 15-question adaptive frontier

An additional 192-replicate run used the production numerical profile with 15
direct questions/domain, a 420-segment bank, and 32 selector representatives
per domain. It is descriptive because its planted population differs from the
inference prior and it was generated before the clustered-CI output was added.

| Metric | Binary | Categorical F1 | N-way change |
|---|---:|---:|---:|
| Skill coverage | 0.9661 | 0.9470 | -0.0191 |
| Bias coverage | 0.9592 | 0.9575 | -0.0017 |
| Skill interval width ratio | — | 0.4685 | 53.1% tighter |
| Bias interval width ratio | — | 0.7308 | 26.9% tighter |
| Skill RMSE ratio | — | 0.5594 | 44.1% lower |
| Bias RMSE ratio | — | 0.7903 | 21.0% lower |

This supports the expected accumulation of cross-domain information as direct
question count grows, while still requiring the governed full stopping-policy
qualification.

## Re-evaluated conclusion

The original 192-particle/2-MH undercoverage was primarily a numerical
particle-cloud problem, not evidence that the conditional F1 likelihood is
intrinsically miscalibrated. At 1,200 particles and 30 MH steps:

1. Absolute skill and bias coverage return to the nominal neighborhood.
2. Prior-predictive rank diagnostics are close to uniform.
3. Skill and bias point estimates remain more accurate in these planted-truth
   runs.
4. Meaningful interval tightening remains, although the smallest smoke profile
   exaggerated its magnitude by underestimating posterior tails.
5. The categorical arm resamples roughly twice as often, confirming that its
   particle/MCSE profile must be qualified independently rather than inherited
   by assertion.

Remaining nonpromotion gates are unchanged: qualified leakage-controlled
artifact, DR07, governed full Precision stopping, approved absolute coverage
margin, model-misspecification/source-shift stress, selector regret on the
served bank, and real-browser/device rollout evidence.

