# Preliminary percentile preview: local validation report

Date: 2026-07-22
Status: release-qualified candidate; one-user cohort rollout was explicitly
authorized for `elikeldsen@icloud.com` on 2026-07-23

## Candidate and statistical scope

The preview evaluates each final CORTEX domain posterior against 500 bootstrap
reference CDFs derived from quality-screened historical calibration fit rows.
It reports the posterior/reference joint mean rank and direct 2.5th/97.5th
weighted quantiles. Candidate uncertainty, fit uncertainty, and historical
fit-row resampling are included. Unknown population mismatch, source
composition, cross-source identity, and self-selection bias are not included.

Eligible historical fit rows:

| Domain | Rows |
|---|---:|
| Spike | 1,934 |
| Seizure | 1,810 |
| LPD | 1,664 |
| GPD | 1,677 |
| LRDA | 1,593 |
| GRDA | 1,561 |
| Other/IIC | 1,612 |

Immutable identities:

| Object | SHA-256 |
|---|---|
| Source norm content | `e5a532dda595e76e3559efa20a0f8ca45fdaa039bada6e8f99890ac8a3c01262` |
| Runtime metadata content | `6bb77ba9efb8ef53eb7a0b5cc4d0cb6af48a20cb7482d76017f0e4d9e4f08d1b` |
| Runtime binary | `c0ca140310b2410d12d61733478c0b8d4999e25e9f77c204a2caa318a7ee4776` |

The default source build reproduced byte-for-byte in a separate temporary
directory. The compact runtime contains 707,721 little-endian float32 values
(2.7 MB before transfer compression), down from the 8.2 MB governed JSON.

## Correctness and noninterference results

- Percentile policy/artifact/transition suite: **13 passed**.
- Browser suite: **57 files passed, 3 skipped; 251 tests passed, 8 skipped**.
- API/persistence/trainer/research suite: **267 passed**.
- New provenance/rollout/scoring/training contract suite: **8 passed**.
- ESLint: passed with zero warnings.
- TypeScript typecheck: passed.
- Production Vite build: passed.
- CSP/browser mount verification: passed.
- npm high-severity audit: zero vulnerabilities.
- Default-off real-browser smoke: passed from signup through spike and IIIC
  assessment phases.
- Public-mode real-browser smoke: passed full artifact fetch, WebCrypto hash
  verification, seven-domain scoring, result ingest, and final display.
- Research gold/export tests: passed with percentile provenance and interval
  fields covered by codebook version r1.1.

The pre-existing assessment/trainer algorithms are not called by percentile
code to make decisions. With mode `off`, new sessions are unstamped and the
legacy result contract remains valid. Percentile load/scoring errors return an
explicit unavailable record and never change or suppress the assessment
determination. A browser tab loaded before the percentile-capable SPA records
`unavailable_legacy_client` instead of losing its otherwise valid result.

## Cross-runtime parity and performance

For the shipped artifact and candidate cloud
`ell=[-0.25, 0.1, 0.7]`, `weights=[0.2, 0.5, 0.3]`, Python and TypeScript
produced the same Spike result:

- estimate `62.044597563171386`;
- lower `29.48`;
- upper `93.45`;
- sensitivity endpoints `59.76856975317356`, `61.716726964972786`.

Observed local scoring times:

| Surface | Work | Time |
|---|---|---:|
| Browser | 7 domains × 1,200 particles × 500 reference replicas | ~287 ms |
| API | same | ~320 ms |
| Trainer API | 7 domains × 400 particles × 500 replicas | ~170 ms |

The browser unit gate permits up to 5 seconds to avoid machine-specific
flakiness while protecting against accidental algorithmic complexity drift.

## Provenance, persistence, and failure behavior

- Assessment and training session rows store norm ID, source SHA, score schema,
  and the exact profile returned to that sitting.
- Resume requires the stored profile to match the locally verified runtime.
- Assessment result ingest requires the exact session profile, canonical seven
  domains, finite 0–100 fields, valid interval direction, expected schema,
  candidate method, and policy status. The sole compatibility exception is a
  missing report from a legacy browser, which is stored explicitly as
  `unavailable_legacy_client`; a supplied malformed report still returns 409.
- Training start/end ranks are computed from the server-owned particle cloud.
  The finalizer discards client-authored percentile numbers, and a scoring
  exception degrades to unavailable without interrupting training.
- Dashboard/history expose only `display=true` available reports. Legacy and
  shadow records remain blank.
- Metadata/binary alteration, invalid mode, missing public release
  acknowledgement, or profile mismatch fails closed.
- Deep health reports runtime readiness, mode validity, norm ID, and public
  release acknowledgement.

## Remaining limitations and release boundary

This is not a representative physician or clinician norm. The historical fit
rows are not established as unique governed people across studies, and the
95% ranges do not quantify population mismatch. Those limitations are stated
next to participant-facing scores.

At the time this pre-deployment validation was completed, no production action
had been taken and `CORTEX_PERCENTILE_MODE=off` remained the safe default.
Commit, push, deployment, shadow exposure, cohort exposure, and all-user
exposure remain separately auditable governed operations. The authorization
above applies only to the named one-user cohort; it does not authorize
all-user exposure.
