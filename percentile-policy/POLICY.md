# Provisional percentile and transition policy

## 1. Status and permitted claim

`historical-calibration-k7-provisional-v1` is a preliminary reporting
approximation. It may be exposed only as a clearly labeled historical preview under the
fail-closed product rollout described in the production runbook.
Its permitted claim is:

> Estimated percentile relative to the quality-screened historical CORTEX
> calibration fit cohort.

The following claims are prohibited:

- clinician percentile;
- physician percentile;
- healthcare-worker percentile;
- peer percentile;
- population percentile;
- IQ or IQ-equivalent score;
- percentage correct;
- probability of passing;
- certified competence.

The percentile is reporting-only. The criterion-referenced verdict and
readiness/pass probability remain separate.

## 2. Historical reference cohort

The source is the manifest-pinned K=7 historical per-domain SDT fit table.
Every eligible fit row receives equal weight within its domain. The code does
not join `rater_name` to a credential or expertise table and does not assert
that fit rows are unique governed people across source studies.

The primary quality policy removes:

- fits marked nonconverged;
- unparseable or nonfinite fit/uncertainty values;
- fewer than 20 trials;
- `SE_ell > 1.0`;
- estimates at the optimizer's sigma bounds.

The threshold is not asserted to remove population mismatch. It only prevents
unidentified fits from destabilizing the numerical reference distribution.
Every score includes point-estimate sensitivity to:

- `SE_ell <= 0.5`;
- at least 50 trials.

## 3. Statistical output

For each domain:

- store the unrounded estimate and lower/upper 95% limits;
- display whole percentile points;
- use direct lower and upper limits, never a symmetric halfwidth;
- display "below the 5th" or "above the 95th" if product policy suppresses
  provisional tails;
- never display 0th or 100th;
- do not aggregate the seven domain percentiles.

Required display:

> 73rd historical calibration-cohort percentile
> Approximate 95% range: 65th–80th

Governance disclosure retained in the immutable profile and documentation
(participant-facing rendering was removed by explicit product-owner direction
on 2026-07-23):

> This provisional comparison uses heterogeneous historical calibration data,
> not representative clinician norms. Its uncertainty range does not measure
> that population mismatch.

## 4. Eligible future CORTEX-user norm

The future estimand is narrower than a clinician-population norm:

> Position among eligible CORTEX users on their first completed assessment,
> before any CORTEX training, during the stated norming window.

Eligible records must be:

- one first completed assessment per authenticated person;
- before any CORTEX testing retake or learning session;
- from a compatible or formally equated instrument/model version;
- nonadministrative, nonqualification, nonsynthetic accounts;
- complete and valid for the domain;
- covered by the approved consent/privacy basis;
- screened under a prespecified data-quality policy.

Repeated assessments and post-training estimates cannot enter the norm. They
would progressively make the comparison population more experienced and cause
the percentile scale to drift as a direct consequence of using CORTEX.

Because CORTEX users are self-selected, the future label remains "CORTEX
first-attempt user percentile" unless recruitment and weighting establish a
representative clinician population.

## 5. Cutover gates

All seven domains must pass:

- effective first-attempt sample size at least 500;
- target effective sample size 1,000 before fine tail claims;
- maximum central 95% halfwidth no more than 5 percentile points;
- maximum anchor percentile drift no more than 3 points over at least two
  consecutive prespecified windows.

The whole cohort must also pass:

- at least 80% coverage of prespecified role/experience/setting/region strata;
- no single site contributing more than 25% after weighting;
- compatible scale/instrument validation;
- posterior calibration and coverage validation;
- differential-item-functioning and subgroup bias review;
- data-quality review;
- privacy/consent review;
- independent scientific review;
- explicit governance cutover approval.

These are minimum gates, not evidence that a sample is representative. The
transition evaluator always returns `automaticCutoverAllowed=false`.

## 6. Versioning and cutover

1. Build an immutable candidate user-norm artifact.
2. Freeze its target definition, fieldwork dates, inclusion rules, weights,
   scale anchors, model, input hashes, and effective sample sizes.
3. Run dual scoring against historical and candidate norms in shadow mode.
4. Review distribution shifts, rank reversals, subgroup effects, tail
   behavior, and clinician comprehension.
5. Obtain explicit scientific and governance approval.
6. Assign a new norm ID and deploy it only for new sessions.
7. Keep historical session percentiles pinned to their original norm.
8. Never silently rewrite old results. Any re-expression under a newer norm
   must be separately labeled.

Review drift at least annually. A rolling three-year first-attempt window is a
reasonable starting proposal after enough data accrue, but must be validated
before adoption.
