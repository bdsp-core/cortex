"""Production trainer-integration package (test → train → retest loop).

This is the PRODUCTION home for wiring the adaptive learning algorithm into
the certification exam, per docs/TRAINER_INTEGRATION_PLAN.md. It is distinct
from the frozen R&D subproject `trainer_rd/` (which is a verbatim, self-vendored
prototype that must never be swept into repo-wide scans — see CLAUDE.md).

Gate G0 lands the contracts this package builds on:
  * `trainer.domains`  — the canonical K=7 domain registry (one source of
                         truth for the three naming conventions in the repo).
  * `trainer.exposure` — the Exposure Ledger interface + NoRepeatPolicy
                         (repeat-prevention data layer, D-INT-7/8).
  * `trainer.bank`     — the credentialed-panel high-confidence training bank
                         builder (D-INT-6).

Architecture boundary (D1): this package may depend on `engine/` (for the
shared likelihood, in later gates) but NEVER on `deployment/`, and vice versa.
G0 modules here have no engine/deployment import at all.
"""
