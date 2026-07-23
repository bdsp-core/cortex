# Preliminary percentile preview: release and operations runbook

## Scope

This feature reports seven independent percentiles relative to
quality-screened historical calibration fit records. It does not change an
assessment verdict, stop rule, question choice, training allocation, mastery,
or readiness. It is not a clinician-population norm, percent correct, or an IQ
score.

The public label is:

> Preliminary historical calibration-cohort percentile

The nearby disclosure is:

> This preview compares you with quality-screened historical calibration
> records—not representative clinician norms. It is not percent correct or
> readiness and does not affect the assessment determination.

## Immutable candidate

| Identity | SHA-256 |
|---|---|
| Governed source norm | `e5a532dda595e76e3559efa20a0f8ca45fdaa039bada6e8f99890ac8a3c01262` |
| Runtime metadata content | `6bb77ba9efb8ef53eb7a0b5cc4d0cb6af48a20cb7482d76017f0e4d9e4f08d1b` |
| Runtime float32 binary | `c0ca140310b2410d12d61733478c0b8d4999e25e9f77c204a2caa318a7ee4776` |

The 2.7 MB runtime is generated from the governed 8.2 MB JSON artifact. Both
the browser and API verify the exact binary; each session stores the norm ID,
source hash, score schema, and complete display profile.

## Exposure controls

`CORTEX_PERCENTILE_MODE` is read when the API starts:

- `off`: no new session is stamped or scored. This is the default.
- `shadow`: new sessions are scored and persisted with `display=false`.
- `cohort`: only identities in `CORTEX_PERCENTILE_ALLOWLIST` receive a public
  profile.
- `all`: every new session receives a public profile.

`cohort` and `all` additionally require
`CORTEX_PERCENTILE_RELEASE_SHA256` to exactly equal the runtime binary SHA
above. A missing/incorrect acknowledgement leaves sessions unstamped. Invalid
modes also fail closed.

In-progress stamped sessions retain their immutable profile. To suppress a
candidate immediately, set mode `off`, restart the API, and remove/quarantine
the candidate SPA version through the normal release switch. Do not edit an
artifact in place.

## Local qualification

From `cortex_web` with the repository virtual environment:

```bash
PYTHONPATH=../percentile-policy/src \
  ../.venv/bin/python -m unittest discover \
  -s ../percentile-policy/tests -v

../.venv/bin/python -m pytest -q
npm run lint
npm run typecheck
npm test
npm run build
node apps/web/scripts/csp_verify.mjs
```

Public-mode local exercise:

```bash
export CORTEX_PERCENTILE_MODE=cohort
export CORTEX_PERCENTILE_ALLOWLIST='<test account email>'
export CORTEX_PERCENTILE_RELEASE_SHA256='c0ca140310b2410d12d61733478c0b8d4999e25e9f77c204a2caa318a7ee4776'
```

Acceptance checks:

1. `/api/health?deep=1` reports `percentile.ready=true` and the expected norm
   ID.
2. A nonallowlisted session has `percentileProfile=null`.
3. An allowlisted session returns the exact profile and renders all seven
   scores only after completion.
4. Every row says historical calibration-cohort, shows an approximate 95%
   range, suppresses tails as `<5th`/`>95th`, and keeps AUROC available only
   as technical detail.
5. Refresh/resume preserves the exact session stamp.
6. Altering a metadata byte, binary byte, reported profile, domain set, range,
   or schema causes a fail-closed load/409.
7. Training shows server-authoritative start-to-end percentile movement only
   for touched domains; client-authored summary numbers are ignored.
8. A result submitted by an already-open legacy browser is preserved with
   `unavailable_legacy_client`; no score is inferred or backfilled.
9. Assessment verdicts and baseline tests are byte/behavior compatible with
   mode `off`.

## Staged release requiring explicit approval

No step below is authorized merely by completing local tests.

1. Review the validation report, disclosure, and immutable hashes.
2. Obtain explicit permission to commit and push.
3. Merge/deploy with `CORTEX_PERCENTILE_MODE=off`.
4. Verify production deep health and static artifact checksums.
5. After separate explicit exposure approval, use `shadow`; inspect ingest,
   unavailable rates, latency, score distributions, and disclosure rendering.
6. After clinician/product review, use `cohort` with a named allowlist and the
   exact release SHA acknowledgement.
7. Expand to `all` only through a separately recorded governance decision.

Rollback is configuration-first (`off` + service restart), followed by the
existing application release switch if the candidate bundle itself must be
removed. Stored historical records keep their original norm provenance.

## Monitoring

Monitor:

- deep-health readiness and norm ID;
- count of stamped sessions by norm ID/display status;
- available vs `unavailable_runtime_error` vs `unavailable_legacy_client`;
- result-ingest 409s for provenance/shape mismatch;
- scoring latency by assessment/training surface;
- domain score/range distributions and tail frequency;
- clinician comprehension and mistaken “percent correct/readiness” readings;
- subgroup/source sensitivity before any stronger peer-norm claim.

`deploy/scripts/observe_percentiles.sql` provides identity-free aggregate
counts, availability, interval width, and tail-frequency checks.

## Future CORTEX-user norm

Only one first completed, pre-training assessment per eligible authenticated
person may enter the future norming cohort. Retakes and post-training scores
are excluded. Cutover requires the gates in
`percentile-policy/policy/transition-policy.json`, two stable windows, privacy,
DIF/subgroup, scientific, and governance review. Passing numerical gates never
causes an automatic switch or rewrites old scores.
