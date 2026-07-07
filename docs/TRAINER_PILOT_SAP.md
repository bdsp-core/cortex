# Adaptive Trainer — Pilot Statistical Analysis Plan (SAP)

**Version:** 1.0 · **Authored:** 2026-07-07 · **Status:** registered (see §9 governance).
**Companion:** `docs/TRAINER_INTEGRATION_PLAN.md` (the gated build, G0–G5) and
`docs/G4_CLOSEOUT.md` §7 (the prod deploy). This SAP defines how the deployed
adaptive trainer is evaluated and kept safe on the live instrument.

> **Transparency note (read first).** The trainer was deployed to prod at full
> exposure on 2026-07-07 (commit `656927c`) on the project owner's explicit
> decision, **before** this SAP was registered. That ordering is not best
> practice — a pilot SAP normally precedes exposure. This document therefore
> functions as a *post-hoc registered* analysis + safety plan for an
> already-live feature, and the guardrails in §7–§8 are retrofitted controls,
> not pre-registered gates. This is stated plainly so the record is honest.

## 1. Background & rationale

The certification exam measures per-domain skill (ℓ) against a credentialed-panel
cut (ℓ\*) and returns PASS / FAIL / REFER per task. The adaptive trainer plugs in
a **test → train → retest** loop: after a cert, a participant enters feedback-driven
training targeted at non-passed domains, then re-certifies. The trainer **never
certifies** (D18) — the cold retest is the sole arbiter — so training cannot inflate
a certificate; it can only change how quickly a genuinely-skilled reader clears the
bar. The risk we must bound is **false graduation**: the trainer telling a learner a
domain is mastered when a fresh cold retest would not pass it.

## 2. Objectives

- **Primary (safety):** estimate the **confirmed false-graduation rate** — of the
  (participant, domain) pairs the trainer graduates, the fraction whose next fresh
  cold retest does **not** return PASS. Target: bounded and low; a persistently
  elevated rate triggers the stopping rule (§8).
- **Secondary (efficacy):** per-domain **retest skill delta** (Δℓ from the seeding
  cert to the next retest) for trained vs untrained domains; **per-trial delivered
  value** (the trainer's expected-reward signal); **graduation rate**.
- **Operational:** **exposure-budget consumption** (trained segments removed from a
  participant's future cert draws) and per-domain trial volume.

## 3. Design

Observational, within-subject, on the live instrument. Each participant is their own
control: trained domains are compared to their own untrained domains and to their own
pre-training cert. No randomization of individuals in v1; the natural contrast is
trained-vs-untrained domain within a participant. Exposure is controlled by the
`training_mode` flag (§6), which also defines the analyzable cohort.

## 4. Population & unit of analysis

- **Population:** authenticated participants who complete ≥1 cert and start ≥1
  training session while `training_mode` admits them.
- **Unit:** the (participant, domain, training-session) triple for graduation/safety;
  the (participant, domain) pair for retest deltas.
- **Confirmed subset:** triples for which a fresh cert finished *after* the training
  session (so a cold retest verdict exists for that domain).

## 5. Endpoints (operational definitions — computed by the monitor, §7)

| Endpoint | Definition |
|---|---|
| Graduated domain | final trained ℓ for (participant, domain, session) ≥ that domain's ℓ\* |
| Confirmed retest | a graduated triple with a cert finished after the training session |
| **False graduation** | a confirmed triple whose retest verdict ≠ PASS |
| False-graduation rate | falseGraduations / confirmedRetests (null until ≥1 confirmed) |
| Retest Δℓ | (retest ℓ) − (seeding-cert ℓ) per (participant, domain) |
| Delivered value | trainer expected-reward per trial (client telemetry) |
| Exposure consumption | count of `training_trials` per participant |

## 6. Exposure control (the feature flag)

`CORTEX_TRAINING_MODE` (env, re-read per-app; default preserves the live deploy):

- **`all`** — every authenticated participant (current prod posture).
- **`cohort`** — only `CORTEX_TRAINING_ALLOWLIST` (code / 9-digit public_id / email).
- **`off`** — kill-switch: the entry is hidden and the start endpoints 403; **in-flight
  sessions still finalize** (no data loss).

Server-enforced on `POST /api/regimen` and `POST /api/training-sessions`; surfaced to
the client as `trainingEnabled` on `GET /api/dashboard` so the "Resume training" entry
is hidden when disabled. Flipping the mode is a config change + `systemctl restart` —
**no redeploy** — which is the fast reversibility this SAP relies on.

## 7. Monitoring

`GET /api/admin/training-monitor` (admin-gated) returns, live: learners, sessions
started/completed, training trials, exposure rows, per-domain trial counts, graduated
domains, confirmed retests, and the confirmed false-graduation rate. Backed by
`db.training_monitor()` + `db._graduation_safety()`; unit-tested
(`test_training_monitor_false_graduation`, `test_training_monitor_empty`). Review
cadence: at least weekly during the pilot, and on any spike in graduations.

## 8. Sample size, analysis & stopping rules

- **Sample size:** the trainer_rd power note motivates ≈33 evaluable participants per
  contrast for the efficacy delta; the safety endpoint is monitored continuously
  rather than powered to a fixed N (it is a bound, not a hypothesis test).
- **Analysis:** false-graduation rate with a Wilson 95% CI once ≥1 confirmed retest
  exists; per-domain Δℓ via paired within-participant comparison (trained vs
  untrained), reported with bootstrap CIs; delivered value + exposure as descriptive
  distributions. Report per-domain (no single roll-up), matching the exam's
  per-task-certificate policy.
- **Stopping / safety rule:** if the confirmed false-graduation rate CI lower bound
  exceeds a pre-agreed threshold (PI-set; default working bound: point estimate
  > 0.10 with ≥10 confirmed retests), flip `training_mode=off` (kill-switch) and
  review before resuming. Any correctness defect in the trainer's belief update is an
  immediate `off`.

## 9. Governance & sign-off

- **PI sign-off:** **obtained — owner-reported 2026-07-07.** The project owner
  (elikeldsen) reports having presented the covered information (pilot design, the
  safety/false-graduation endpoint, the exposure-ledger side-effect on future cert
  draws, the monitoring, and the flag-based rollback) to the Principal Investigator,
  who reviewed and approved. Recorded here as **owner-attested**; a signed artifact
  (PI name + date, or an approval email reference) should be filed alongside this SAP
  to make the governance record durable and independently verifiable.
- **Waiver acknowledged:** the pre-deploy G5 gate (sign-off + guardrails *before*
  exposure) was waived by owner decision; this SAP + §6/§7 controls are the retrofit.
- **Rollback of record:** `training_mode=off` (soft, immediate) or
  `git revert 656927c` + rerun `cortex_web/deploy/scripts/deploy_app.sh` (hard).

## 10. Open items (human-gated, not autonomously completable)

1. File the durable PI sign-off artifact (name/date/approval reference) beside §9.
2. Run the pilot to N and complete the §8 analysis on accrued real data.
3. Decision to expand training scope beyond the current per-domain set (PI).
