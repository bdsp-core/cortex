# Owner ratification of the 2026-08-01 locked-campaign gate outcome

Date: 2026-08-01. Decision maker: the owner, in response to the parked
campaign report (`draw_latent_rd/reports/campaign_draw_latent/
PHASE1F_CAMPAIGN_REPORT.md`), which presented the pre-registered verdict
(`campaign_pass: false` on the absolute-bias clause alone; every skill
clause and every −0.03 noninferiority clause passed) together with three
unblocking options.

## Decision (owner's words)

> "Waive the clause for this run, recording 0.94575/0.94475 in the
> qualified artifact's provenance. These are excellent results."

This is option 2 of the parked report: an explicit owner waiver of the
absolute-bias-coverage clause **for this run only**, with the measured
values recorded permanently in the qualified artifact's provenance:

- cell 1 (continuum SBC): bias coverage 0.94575, Wilson [0.94155, 0.94966]
- cell 2 (served-bank adaptive): bias coverage 0.94475, Wilson [0.94052, 0.94870]

## Scope and what this does NOT change

- The pre-registered verdict stands in the record exactly as evaluated
  (`GATE_VERDICT.json` keeps `campaign_pass: false` for the pre-registered
  rule); the waiver is layered on top as
  `campaign_pass_with_owner_waiver: true`, never rewriting the original
  outcome.
- The waiver covers ONLY the absolute-bias clause of THIS campaign's two
  gating cells. Skill absolute coverage, both noninferiority gates, and
  tightening passed on their own and needed no waiver.
- The bias block remains governed by the paired −0.03 noninferiority
  guardrail versus binary (passed: +0.00008 and −0.00425), and the
  absolute-bias platform property (shared with binary controls; see
  QUALIFICATION.md's standing note) remains a candidate for
  particle-profile requalification as future work.
- All other implementer defaults (atom grid, nesting atom, particle floor,
  −0.03 margin, DR07 criterion v2) were ratified implicitly by this
  promotion decision on the recorded evidence.

## Effect

The qualified re-emission (`scripts/qualify_nesting34_artifact.py`) may
proceed; it records this document's SHA-256, the waived values, and the
full campaign provenance chain in the qualified artifact. The promotion
resumes at Phase 2 of `docs/DRAW_LATENT_PROMOTION_PLAN.md`.
