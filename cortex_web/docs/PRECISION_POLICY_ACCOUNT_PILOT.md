# PrecisionPolicy production rollout

## Scope

`precision_v1` supersedes AD6 for every new authenticated certification
sitting. The API, not the browser, selects the policy and stamps it on the
session row. An absent/legacy stamp is `ad6`; existing AD6 sittings resume and
finish under that persisted stamp. AD6 remains in the engine as the immediate
rollback policy.

This is the frozen, uncalibrated `frontier_p90guard_m3` profile. There is no
`g` parameter in the TypeScript port. The optional Python calibration
arguments remain absent: no interval scale, estimate ramp, or bias scale is
implemented or accepted here.

## Frozen session profile

- Task order: `spike, sz, lpd, gpd, lrda, grda, iic`.
- 1,200 particles; `Corr_l` for skill and `Corr_t` for bias.
- `total_var` selection, stable signal ordering, uncertainty-aware
  coarse-to-fine candidate scan with `n_subsample=128`.
- Randomized first item from the best 10; at most five consecutive questions
  from one domain.
- Equal-tailed weighted 95% intervals for reporting. Stopping uses the
  separate point-centered radius and quantile-MCSE guard.
- Three questions per frozen signal tercile, at least 20 own-domain questions,
  ESS at least half of the particle count, two consecutive precision passes.
- 60 questions per domain, hence no more than 420 in seven domains. The 60th
  answer is evaluated before CAP; BANK is evaluated afterward.
- Certification cuts are used only in final reporting and only for
  `ESTIMATE_COMPLETE` domains: `ABOVE_CUT`, `BELOW_CUT`, or
  `INDETERMINATE_AT_CUT`.

The profile uses the complete exposure-eligible
`v1.6-k7-35k` served bank (manifest SHA-256
`c3ce43b5639bb1a5e15b2c014246c5f4e6916bcb3284d10aaebf1d253e8811fa`).
AD6 continues to receive its existing sampled candidate pool. Precision resume
stores the small temporal exclusion list plus this manifest hash, rather than
persisting all 35k candidate IDs.

## Browser performance implementation

The browser keeps the frozen policy and selector unchanged while avoiding
redundant computation:

- The A-optimal expected-loss calculation uses the law of total variance and
  reuses each coarse candidate's loss during refinement. A direct
  conditional-variance regression pins the algebraic equivalence.
- Each particle cloud is stable-sorted once per domain and reused for every
  interval and MCSE quantile. Reported intervals and guarded radii remain
  numerically identical to the fixed-cloud reference.
- The manifest-expanded, signal-sorted full bank is cached once per session;
  the post-answer remaining-bank view is built once and shared by speculative
  branches.
- Eligible native n-way sessions use a conservatively calibrated persistent
  worker pool for deterministic screening, exact refinement, and MH-history
  shards. Probability-ranked speculation is bounded, observed answers preempt
  unused work, and only the actual response is adopted.
- Worker-side categorical history is cached behind a versioned typed protocol;
  cache changes preserve particle order and the exact likelihood result.
- Runtime load evidence can reduce a pool only at a safe between-question
  boundary. Repeated-maximum evidence and a cooldown prevent isolated browser
  stalls from ratcheting a high-core session to serial execution.
- During required long calculations the UI presents non-statistical trajectory
  progress. Presentation timing never enters inference or policy state.

These are scheduling, allocation, caching, and algebraic optimizations only.
They do not change the
particle count, candidate set, `n_subsample`, selection objective, stopping
rule, intervals, cuts, or status semantics.

Before release, stream a sanitized completed-session fixture into
`apps/web/scripts/replay_precision_session.ts`. The audit reconstructs the
served full-bank session from its exclusion list, bank manifest, production RNG
identity, and recorded answers. It requires exact question order, stop reason,
statuses, determinations, terminal reasons, and verdicts; numeric telemetry must
agree within combined absolute/relative `1e-12` tolerance. Fixtures are not
written into the repository.

## Configuration and rollback

The defaults are:

```text
CORTEX_PRECISION_POLICY_ROLLOUT=all
CORTEX_PRECISION_POLICY_EMAILS=elikeldsen@icloud.com
CORTEX_PRECISION_BUNDLE_URL=/bundle/v1.6-k7-35k
```

The supported rollout values are `all` (public Precision), `email_allowlist`
(server-side canary), and `off` (AD6 for every new sitting). Unknown values also
fail closed to AD6. Set the rollout to `off` and restart the API for an immediate
new-session rollback. Existing sittings retain their stored policy and bank
provenance so resume stays deterministic. A client-supplied policy is ignored
at start, and result ingest rejects a policy that differs from the session
stamp.

## Verification gates

- Scripted fixed-cloud Python/TypeScript parity for intervals, radii, MCSE,
  guard results, reversible statuses, and stopping.
- Deterministic TypeScript Precision golden session and inline/speculative
  clone equality.
- Transactional fail-closed posterior update test.
- Production-sized full-bank selector latency gate at 1,200 particles.
- Public, exact-email canary, and fail-closed API gates; immutable resume stamp;
  compact full-bank resume provenance; result-spoof rejection; and rollback
  switch test.
- Existing AD6 drift, browser, API, type-check, and build regressions.

The accepted extreme-skill interval-coverage limitation remains disclosed and
unchanged; this port does not recalibrate or alter the frozen stopping rule.
