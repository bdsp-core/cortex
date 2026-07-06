# FINAL REPORT — pre-submission state of the learning algorithm (M27, 2026-07-05)

For: Eli, before sending the package to the academic advisors.
Package to send: `docs/ADVISOR_BRIEF.md` (the sign-off vehicle, current
through M27) + `docs/SUBMISSION_CRITERIA.md` (the gate + stopping rule).
Supporting: `docs/ARCHITECTURE.md` (a committee member's first read),
`docs/PROJECT_MEMORY.md` (the full evidence trail), the M21–M27
checkpoint analyses, and the reproducible tree itself.

## 1. Bottom line

The learning algorithm is DONE iterating and submission-ready. The
submission gate (S1–S8) is fully met; the suite is green at 21 files /
299 checks; every study re-runs from the scratch root and reads from
cached artifacts. Four consecutive checkpoint loops (M24–M27) adjudicated
every queued training-side successor end-to-end and none displaced the
shipped defaults — the shipped stack is a measured local optimum under
all currently available data, and the stopping rule (D43) now requires
new data, an advisor request, or a regression to reopen work.

## 2. What this final loop (M27) changed

Directed pre-submission asks: complete open improvements, clean stale
files, make the codebase committee-presentable and pluggable into new
domains. All four open items were completed or adjudicated — two of the
four with honest data-driven NOs, which is the pattern the advisors
should see as a feature of the methodology:

1. **Terminal-confirmation semantics (F90 — built, validated, default
   OFF, D44).** A confirmed task hands off to the retention layer
   instead of being re-polished forever. It wins the pair-completion
   endpoint decisively (+20pp) and it costs within-protocol staleness
   detection (~50% of post-confirmation regressions undetected in-run).
   Because this trade is exactly the D18 "trainer certifies nothing"
   semantics — staleness belongs to scheduled re-certification — the
   flip is filed as advisor decision ask #7 rather than taken
   unilaterally. A port-relevant mechanism bug (retention-bin
   proliferation under gate-bypassing semantics) was found by the
   campaign and fixed.
2. **Contact-triggered onboarding (F91 — SHIPPED ON in the sandbox,
   D45).** The one default flip of this loop, protocol-layer only:
   demonstration items until the contact e-process certifies, then a
   trainability-guarded belief re-open. The no-contact onboarding shape
   the real testers actually exhibited goes from censored@30% to
   declared 148@70%; healthy learners are unharmed; careless responders
   never certify and never falsely declare.
3. **λ(time-on-task) (F92 — CLOSED, no model term).** On 125,860 real
   reads, within-sitting easy-item accuracy RISES +8.7pp/hour —
   learning dominates any fatigue-lapse at these horizons, and
   self-paced data cannot even sign a lapse trend (quitting truncates
   the fatigued tail). Deferred to pilot fixed-length sessions; the
   shipped differenced fatigue guard covers the operational risk.
4. **Domain pluggability + presentability (F93).** The de-facto seam is
   now a named, tested interface: `training/domain.py` (ItemBank /
   ArrayBank / Domain / `v15_domain()`), proven by a synthetic toy
   domain graduating through the UNCHANGED stack. `docs/ARCHITECTURE.md`
   gives the committee the module map, the math↔code correspondence
   table, and the reading order. The EXTSET release moved to
   `data/extset/` behind a single path junction; the root directory is
   clean (11 packages + README); `__pycache__` gone; the archive
   convention holds (`archive/README.md`).

Everything shipped opt-in and default-bit-identical except D45 (a
protocol-layer flip justified by its study); behavioral identity of
every default path is pinned by the suite.

## 3. What the advisors are being asked (brief §6)

Seven asks: D18/D19 ratification; D16 acceptance semantics; OQ3
warm-start vs fresh re-cert; F26/F27 comfort; confirmed-FG endpoint
semantics (D41); the pilot SAP power/endpoints; and NEW — #7,
terminal-confirmation semantics (F90/D44): should confirmed mastery
belong to the re-certification layer? Our recommendation is yes for the
pilot (it has scheduled re-certs), shipped only on their ratification.

## 4. Honest limitations the package states plainly

- Sharp-margin false declarations are information-bounded, not gate
  inefficiency; protection is architectural (D33 confirmation kills
  5/6, D18 re-cert catches 75/75) and FG is priced at the CONFIRMED
  level everywhere.
- The trainer certifies nothing; the definitive methodology test is the
  feedback-driven pilot (no existing dataset contains closed-loop
  training).
- Delivered per-trial training value on the real testers averaged 0.48
  of ideal; the M23–M27 protocol work targets the named sinks
  (regime-shift lag, onboarding no-contact — the latter now addressed
  by D45, pending its field check).
- The F91 demonstration ramp's pedagogical value is deliberately
  unsimulated (it would assume the conclusion) — a field-check
  question, flagged as such.

## 5. State of the tree (for your pre-send review)

- Suite: 21 files / 299 checks green (final run 2026-07-05; command
  list in ADVISOR_BRIEF §7).
- Root: `README.md` + 11 directories (engine / training / sandbox /
  studies / tests / viz / docs / data / config / reference / figures /
  archive).
- New since the freeze: `training/domain.py`, `docs/ARCHITECTURE.md`,
  `docs/M27_TERMINAL.md`, `docs/FINAL_REPORT.md` (this file), three
  study modules + caches, `tests/test_m27.py`; sandbox F90/F91 wiring;
  `data/extset/` relocation.
- Open telemetry watch items for the next field sessions: `demo` trial
  counts + `contact_handoff` events (D45), `hazard_scale` events (only
  if a D42 field ablation is ever requested).

## 6. If the advisors come back with…

- **Ask #7 = yes:** flip `TERMINAL_CONFIRMATION=True` (term3 pin) in
  `sandbox/config.py`; re-run `tests.test_sandbox`, `tests.test_m27`,
  and one robot campaign before the next human session.
- **Analysis requests:** reopen under D43 category 1; each request gets
  a PECR loop and an M# record.
- **Silence on the optimization side:** nothing further is planned —
  the stopping rule stands until new data arrives.
