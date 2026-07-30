# CORTEX n-way certification protocol

Status: **standard isolated n-way implementation; not production-integrated or
statistically qualified**.

This directory implements the conditional F1 six-way IIIC response protocol
without modifying `cortex_web`. The production exam continues to use its
binary reduction until a separate green light authorizes promotion.

## Implemented here

- Registry-driven mixed response families: binary spike plus categorical IIIC.
- Exact preservation of the existing focal-class binary marginal.
- Stable six-way F1 likelihood with full per-class `sMean`/`sSd` vectors.
- Transactional particle updates retaining the raw pick.
- Complete categorical history replay during MH rejuvenation.
- Exact six-outcome total-posterior-variance loss.
- Deterministic, bounded full-bank shortlist with content-forcing support.
- Direct-evidence bookkeeping: cross-domain information never increments
  direct-domain counts or content bands.
- Ranked coordinator/helper branch protocol with exact serial fallback.
- Immutable engine profile, artifact, result, and resume contracts.
- Fail-closed server rollout specification and additive database migration.
- Independent Python oracle, cross-language fixtures, and an adaptive planted-
  truth comparison harness for both skill and bias.
- Read-only frozen recovery archive for the pre-iteration 1,200-particle/30-MH
  baseline.
- Research-only artifact cross-fitting/uncertainty propagation, Fisher
  shortlist augmentation, MC-convergence, resampling, empirical-bank
  information, and selector-regret harnesses.
- Machine-readable standard pointer fixing the ensemble/Fisher profile at
  1,200 particles and 30 MH steps for all new isolated n-way work.

Read-only compatibility imports from `cortex_web` are limited to the current
RNG, prior, and linear-algebra primitives. No file outside this directory is
written by the implementation or test commands.

## Run

```bash
cd /data/eli-work/repos/ilae-skill-certification-test-multi/n-way-protocol
bash scripts/test.sh
bash scripts/qualification.sh smoke
bash scripts/integration_experiment.sh smoke
npm run bench
npm run audit:selector
```

The test runner executes TypeScript type-check, TypeScript tests, and Python
tests concurrently. The qualification runner uses process-level parallelism,
leaves two logical CPUs and at least 20% of available memory as headroom, and
pins numerical libraries to one thread per worker to prevent BLAS
oversubscription.

The longer qualification-scale adaptive-frontier command is:

```bash
bash scripts/qualification.sh qualification --replicates 1000
```

It exercises adaptive selection at 1,200 particles and 30 MH steps, but it is
not the locked production stopping-policy qualification. That output remains
research-only while the included artifact says `exploratory_unqualified` and
DR07 says `not_qualified`.

## Directory map

- `src/`: production-shaped TypeScript response, SMC, selector, session, and
  speculation implementation.
- `tests/`: invariants, parity, fail-closed, selector, replay, and performance
  tests.
- `python/nway_protocol/`: independent oracle and adaptive qualification.
- `server/`: API rollout/persistence contract and unapplied migration.
- `schemas/`: artifact and engine-profile JSON schemas.
- `artifacts/`: explicitly unqualified exploratory inputs.
- `frozen/`: immutable, checksummed pre-iteration recovery archive.
- `docs/`: protocol, qualification, and promotion contracts.
- `reports/POST_FREEZE_RD_FINDINGS.md`: complete challenger results and
  nonpromotion decision.
- `reports/POST_FREEZE_RD_FILES.sha256`: digest manifest for the post-freeze
  implementation and retained evidence.
- `reports/FINAL_INTEGRATION_COMPARISON.md`: final 2,000-seed frozen-versus-
  integrated 1,200-particle/30-MH result.

## Isolation boundary

Promotion is deliberately impossible by configuration alone:

1. The code is outside `cortex_web` and is not imported by it.
2. The isolated rollout contract refuses an unqualified or non-DR07 artifact.
3. The migration is an unapplied SQL specification.
4. Existing production session rows and response behavior are untouched.
5. The exploratory artifact contains `promotionForbidden: true`.

## Standard n-way profile

The ensemble/Fisher implementation is the standard approach for all new work
in this isolated directory. `artifacts/nway_standard_rd.json` fixes its engine
profile, response artifact, 1,200-particle count, 30 MH steps, ESS threshold,
evidence report, and frozen rollback archive. The frozen scalar profile remains
available only for comparison, exact historical replay, and rollback.

This standard designation is not a production authorization. The integrated
artifact remains exploratory, non-DR07, and `promotionForbidden`; the isolated
rollout contract continues to fail closed until a qualified ensemble artifact
is supplied and a separate production green light is given.
