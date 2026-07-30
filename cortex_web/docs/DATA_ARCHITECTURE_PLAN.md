<!-- Design plan from the 2026-06-19 multi-agent architect review. Originally
     DESIGN ONLY; substantially IMPLEMENTED since — status header updated
     2026-07-01 when the doc was committed. -->

# CORTEX Research Data Architecture — Unified Plan

**Status (refreshed 2026-07-21):** phases H0 (fab-data hotfix), R0 (gold ETL), R1
(`ilae-export`), O1 (expertise + consent), O2 (learning linkage) shipped
2026-06-19; O3 (provenance stamping: `sessions.bundle_version`) shipped with
the v1.6 deploy 2026-06-25, extended by `sessions.drawn_seg_ids` 2026-07-01.
L1 shipped when the server-side learning engine became the sole trainer on
2026-07-17. O4 (engine SD/AUROC persistence — optional and re-derivable)
remains open. This document is now a design/history record; current runtime
ownership is defined by `PRODUCTION_BASELINE.md`, `REPOSITORY_STRUCTURE.md`,
and `../README.md`.

**Live-reading rule (2026-07-22):** present-tense statements and source line
numbers below describe the original review, not the current API. The synthetic
trajectory write was removed and quarantined; `POST /api/trajectories` no
longer exists, `GET /api/trajectories` is read-only, and real learning rows are
server-authored through owned `POST /api/training-progress` sessions.
`GET /api/manifest` was not adopted because `POST /api/session` already
returns the stamped per-sitting bank/profile. The retired
`GET /api/training-sessions` route also remains absent. Do not recreate any of
those endpoints merely because the historical plan proposed or audited them.

**Original status:** proposal, revision 2 (post-adversarial-review). No files modified. Grounded in current `main`; where the five audits disagreed, code was the tiebreaker (see §0.3 "adjudicated facts"). Every Critical (C1–C5) and Important (I1–I6) item from the adversarial review is resolved inline; the four Minors (M1–M5) are folded into the relevant sections. A change-log is in §9.

---

## 0. Executive Summary + Design Principles

### 0.1 The core finding
Almost everything the Nature Medicine paper needs **is already produced client-side and already lands in the database** — but it is **trapped in opaque JSON** at the wrong grain, with **no provenance stamping** and **no learning-to-session linkage**. The data is not missing upstream; it is unqueryable and unattributed at the persistence boundary.

Concretely, today:
- **Rich demographics** (expertise, institution, practice_setting, years_reading_eeg, eeg_volume_per_month, self_rated_confidence, sex, country, race_ethnicity, color_vision, prior_test_taken, consent_version, irb_protocol_id) are captured at test-start and persisted — but **only inside `sessions.participant` TEXT JSON** and duplicated inside `results.result` JSON. No columns, no enums, no `GROUP BY`.
- **Full per-question SMC posteriors** (`TrialDiag`: ess, rejuv, pi, mcse, R, tMean, lMean, aurocHw, verdicts, nPerTask, s, sSd) are persisted — but **only inside `trials.diag` and `results.result` TEXT JSON**.
- **Provenance** (engine version, bundle version, cert_config/`certBlock`, engine RNG seed, consent timestamp) is **genuinely never captured**.
- **Learning trajectories** are (a) currently **fabricated constants** that are **already written to the real `param_trajectories` table** and (b) have **no FK** to a training session or source eval session — so no learning curve is attributable to a sitting or protocol. This is an *active contamination*, not a future risk (see C1 disposition, §0.4).
- The **signup-time `expertise` dropdown** is the one field truly **dropped at the network boundary** (`app.py:392`).

### 0.2 Design principles (locked to the session goals)
1. **The live operational DB stays stable.** All operational changes are **additive, nullable, idempotent `ADD COLUMN IF NOT EXISTS` / new tables / new optional API kwargs** — the exact migration class already running safely at every boot (`db.py:131–200`). No rename, drop, retype, `NOT NULL`, or hot-path widening. *(One exception: a pre-announce **hotfix** that stops new synthetic-trajectory writes — see §0.4 C1.)*
2. **Research value comes from a separate additive layer.** A read-only **gold analytics layer** and a **versioned de-identified export product** sit *on top*, reading transactionally-consistent snapshots/replicas, never the primary write path. They cannot destabilize live prod.
3. **Stop the bleeding, don't re-architect.** The forward-fidelity operational changes are the short list in §1.2: persist expertise; stamp provenance + consent on sessions; add the learning-linkage FKs; **stop and quarantine** fabricated trajectory data. Everything else is recovered retroactively from existing JSON by the gold ETL.
4. **Hot path untouched.** `/api/progress` → `upsert_trial` (db.py:393), fired on every answer, is **not widened**. Per-question fidelity is reconstructed in gold from the durable `results.result` blob.
5. **De-identify by allowlist, never denylist.** Gold projects ONLY enumerated quasi-identifier columns; free-text fields are never selected into gold at all. No "scrub-after" step exists anywhere in the pipeline.
6. **Code is the source of truth.** Memory/docs were stale on two points (expertise is "dropped" only at signup, not per-session; demographics are trapped-in-JSON not lost). This plan follows the code.

### 0.3 Adjudicated facts (where audits disagreed → code wins)
- **"Expertise is dropped"** — *Partially true.* The **signup** `RegisterIn.expertise` is dropped at `app.py:392`. The **per-session** `participant.expertise` (richer, same enum) **is** persisted inside `sessions.participant` JSON. Fix = persist the signup field **and** flatten the JSON; do not "recover" what isn't lost.
- **Per-question posteriors** — present in `results.result["trials"]`; recoverable by ETL with **zero** operational change.
- **Posterior SD per task per question (`lSd`/`tSd`)** — *genuinely dropped at the engine source*: computed in `particles.ts:188–193`, discarded at `session.ts:246` (only `tMean`/`lMean` destructured). Recovering this requires a client/engine change, not just an ETL.
- **Trajectory data today** — *fabricated constants* (`Shell.tsx:1015–1027`: `theta:0.1, sd:0.12, rt:2400`). The critique's C1 verified these are **already POSTed and persisted as real rows**, and `app.py:631–637` returns `sample:false` the instant any row exists. They are contaminating prod **now**, not just "served back."
- **Free-text `name`/`institution` in the durable blob** — `Registration.tsx:13–14` free-text fields flow verbatim into both `sessions.participant` and the durable `results.result` (App.tsx:170,180). The result blob is the ETL's canonical source, so any denylist scrub is a re-identification hazard.
- **No deletion path** — there is **zero** DELETE / withdrawal capability anywhere in `db.py` (verified). Any "withdrawal support" claim must be built, and is bounded by DOI irrevocability.

### 0.4 Critical-item dispositions (summary; details in cited sections)
| # | Issue | Disposition | Where |
|---|---|---|---|
| **C1** | Fabricated trajectory rows already persisted + mislabeled `sample:false` | **Pre-announce HOTFIX** (stop the write) + backfill `is_real=0` on ALL existing rows + ETL trusts only `is_real=1`, never API `sample` flag | §0.4-detail, §1.2-C, §5.2, §6-H0 |
| **C2** | Free-text `name`/`institution` in durable blob; denylist scrub insufficient | **Allowlist projection** into gold (free-text never selected) + hard anti-leak assertion/test + stop *storing* free-text institution go-forward | §1.2-F, §4.1, §4.3 |
| **C3** | No withdrawal path; deterministic pseudonym + DOI = irrevocable | Build real withdrawal op (future-export exclusion only); document DOI irrevocability in consent + datasheet; **IRB sign-off before first DOI** | §1.2-D, §4.2, §4.4, §7 |
| **C4** | k=5 k-anonymity meaningless at n≈7 | Pilot publishes **performance data + coarse aggregate marginals only**; NO individual-level quasi-identifier crosses; k is a function of cohort size | §4.4, §7 |
| **C5** | Reproducibility claim is inferred, not recorded, for 100% of pilot | **Run the round-trip** (re-derive verdicts, assert bit-equality) before labeling reproducible; downgrade the data-availability statement if it fails | §3.2, §3.4, §5.2 |

**C1 detail (the active contamination).** `Shell.tsx:1024` calls `api.appendTrajectories(pts)` → `POST /api/trajectories` → `append_trajectory_points` (db.py:535), writing `theta:0.1, sd:0.12, rt:2400` constants into the production `param_trajectories` table. `app.py:631–637` then returns `sample:false` the moment any row exists, so the live dashboard is told fabricated data is real. Three actions, the first **pre-announce**:
1. **HOTFIX (H0, pre-announce):** gate `Shell.tsx:1024` behind a disabled flag (or remove the synthetic write) so no further practice session contaminates the real table.
2. **Backfill (O2):** set `is_real=0` on **all existing** `param_trajectories` rows, not merely `DEFAULT 0` for new inserts.
3. **ETL rule:** gold treats the API `sample` flag as **untrusted** and derives realness solely from `is_real=1`.

### 0.5 Design principles unchanged from rev 1
Three-layer strict isolation; additive-only OLTP; allowlist de-id; manifest sha256-chain discipline mirroring `data/DATA_PROVENANCE.md`. The adversarial reviewer affirmed these as sound — the revision hardens the *edges*, not the architecture.

---

## 1. Target Architecture

### 1.1 Three layers (strict isolation)

```
LAYER 0 — LIVE OLTP (prod Postgres)                  [+additive nullable cols only; +1 hotfix]
  participants  sessions  trials  results  regimens
  training_sessions  param_trajectories  auth_codes
  NEW append-only: consent_events
        │  read-only: TRANSACTIONALLY-CONSISTENT nightly pg_dump  OR  logical read-replica
        │  ETL excludes sessions with status != 'complete' (no half-sessions)
        ▼
LAYER 1 — GOLD ANALYTICS  (separate schema `gold`, derived, rebuildable)
  dim_participant · dim_provenance · dim_consent
  fact_test_session · fact_trial · fact_test_task_outcome
  fact_training_session · fact_trajectory_point
  bridge_participant_journey   (+ canonical views)
  ALLOWLIST projection only — free-text name/institution NEVER selected here
        │  ilae-export CLI (de-identify by allowlist + cohort-aware k-anon + manifest)
        ▼
LAYER 2 — PUBLIC DE-IDENTIFIED EXPORT  (versioned, DOI'd, Nature data-availability)
  /tables (CSV+Parquet per grain) · /per_participant (3-bucket) ·
  /codebook (version-pinned) · export_manifest.json · DEIDENTIFICATION.md
```

**Hard isolation rule** (mirrors the repo's D1 engine/deployment boundary): the live API **never reads** `gold`; `gold` **only reads** Layer 0 snapshots; the export CLI **only reads** `gold`. A runaway analytics/export query cannot lock or slow the live writer because it runs against a replica or a restored nightly dump.

**Snapshot consistency (resolves I4).** The snapshot MUST be transactionally consistent — `pg_dump` is, by default, a single-transaction point-in-time dump; state this explicitly and use it (or a logical replica read inside one repeatable-read transaction). The ETL **excludes any session with `status != 'complete'`** (mirroring the dashboard's existing completed-only filter), so a session whose row exists but whose result blob has not yet landed never enters gold as a half-session. The manifest records `source_snapshot{kind, captured_utc, dump_sha256}` as an **enforced gate**, not a free field.

### 1.2 The SHORT list of additive operational changes (stop losing go-forward data)

All follow `db.py:131–200`'s idempotent pattern. Each new API kwarg defaults to `None` and is pinned by a **drift-guard test** (repo convention) so old SPA builds and the 282-test suite stay byte-identical. **Per-table migration helpers (resolves M1):** `_migrate_participants` (db.py:155) is participants-specific; this plan adds parallel idempotent helpers `_migrate_sessions`, `_migrate_trajectories`, `_migrate_training_sessions`, each mirroring the db.py:155–167 pattern for **both** SQLite and Postgres backends — one helper does NOT cover all tables.

**(A) Persist signup expertise** — the one truly-dropped field.
```sql
ALTER TABLE participants ADD COLUMN IF NOT EXISTS signup_expertise TEXT;
```
Pass `body.expertise` (already arriving, `app.py:98`) into `register_participant` (`db.py:262`). ~3 lines. Independently shippable, reversible.

**(B) Provenance + consent stamping on `sessions`** — the only genuinely-missing capture.
```sql
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS engine_version    TEXT;    -- "web-engine@<git-sha>"
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS bundle_url        TEXT;    -- DEFAULT_BUNDLE_URL (app.py:72)
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS bundle_version    TEXT;    -- real manifest.version (replace hardcoded "v1.1-local", app.py:574)
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS cert_config       TEXT;    -- inputs.certBlock (types.ts:30)
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS engine_seed_int   INTEGER; -- RAW integer seed (resolves M2)
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS engine_session_id TEXT;    -- "web-<info.seed>" (App.tsx:189); stored separately, asserted to agree
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS consent_version   TEXT;    -- operational DISPLAY only, never an export gate
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS consent_utc       TEXT;    -- accept-click time (App.tsx:230-235), currently never recorded
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS irb_protocol_id   TEXT;
```
**Seed binding (resolves M2 + §3.2 caveat):** store the **raw integer** seed (`engine_seed_int`) and the engine session-id string (`engine_session_id`) **separately**, and **assert at write time** that `engine_session_id == "web-" + str(engine_seed_int)`. No string-parsing of a provenance key downstream.

Wiring: `GET /api/manifest` returns a **real** `version`+`certBlock`+`engineVersion` (git SHA injected at `vite build`; cert_config from the bundle's `manifest.json`). `SessionIn` (`app.py:117`) gains the optional kwargs; `create_session` (`db.py:367`) stores them. `Bundle.inputs` must re-expose `version` (dropped at `bundle.ts:47`).

**(C) Learning-linkage FKs + quarantine** — close the attribution gap and stop C1.
```sql
ALTER TABLE param_trajectories ADD COLUMN IF NOT EXISTS training_id       TEXT;    -- FK→training_sessions
ALTER TABLE param_trajectories ADD COLUMN IF NOT EXISTS source_session_id TEXT;    -- FK→sessions (seeding eval)
ALTER TABLE param_trajectories ADD COLUMN IF NOT EXISTS seq_in_session    INTEGER; -- monotonic order (replaces fragile ts-sort)
ALTER TABLE param_trajectories ADD COLUMN IF NOT EXISTS is_real           INTEGER NOT NULL DEFAULT 0; -- quarantine flag
ALTER TABLE training_sessions  ADD COLUMN IF NOT EXISTS regimen_id        TEXT;    -- FK→regimens
ALTER TABLE training_sessions  ADD COLUMN IF NOT EXISTS source_session_id TEXT;    -- FK→sessions
-- C1 BACKFILL (one-time, in the O2 migration, NOT just a default):
UPDATE param_trajectories SET is_real = 0 WHERE is_real IS NULL OR is_real <> 1;
```
`append_trajectory_points` (`db.py:535`) and `create_training_session` (`db.py:507`) accept the optional ids; only the future real-trainer port (Phase L1) sets `is_real=1`. `regimens.source_session_id` already exists (`db.py:81`), completing the chain `trajectory_point → training_session → regimen → source cert session`. The explicit `UPDATE ... is_real=0` quarantines **all rows already in prod** (the fabricated `Shell.tsx` constants), not merely future inserts — this is the C1 backfill the `DEFAULT 0` alone does not provide.

**(D) Consent as an event (append-only table) + a real withdrawal path.**
```sql
CREATE TABLE IF NOT EXISTS consent_events (
  code            TEXT NOT NULL,         -- FK participants.code
  consent_type    TEXT NOT NULL,         -- 'research_irb' | 'privacy' | 'terms'
  consent_version TEXT NOT NULL,
  irb_protocol_id TEXT,
  accepted_utc    TEXT NOT NULL,
  consent_ip      TEXT,                  -- proof-of-act; identifier, NEVER published; retention policy per M5
  withdrawn_utc   TEXT,                  -- set by the withdrawal op (C3); acted upon at export time
  PRIMARY KEY (code, consent_type, consent_version)
);
```
New `POST /api/consent {consentType, consentVersion}` writes a row (ts=`utc_now()`, ip=`_client_ip`). **Withdrawal (resolves C3):** a new operation sets `withdrawn_utc` AND flips the participant's `dim_consent.publishable=false`; the export filter (§4.2) joins **current** `consent_events` so a withdrawn participant is excluded from **all future** exports. Hard limitation, stated in `DEIDENTIFICATION.md` and the IRB consent copy: **already-released DOI'd exports cannot be retracted** — withdrawal affects subsequent releases only. The denormalized `sessions.consent_*` (from B) are for operational DISPLAY only and are **explicitly forbidden as an export gate** (assertion + test; resolves I5). **`consent_ip` retention (M5):** confirm IRB permits indefinite IP retention or add a truncation/retention policy; IP is never exported regardless.

**(E) Per-question fidelity — NO operational change.** The full `TrialDiag` is already in `results.result["trials"]`. The gold ETL parses it. **Do not** widen `trials` (that touches the hot `/api/progress` path). *Exception, deferred to O4:* the per-task posterior SD (`lSd`/`tSd`) is dropped at the engine source — recovering it needs a small bit-safe client change (add `lSd`/`tSd`, per-trial `aurocMean`, `chosenLoss` to `TrialDiag`, stop discarding at `session.ts:246`, carry RT into `result.trials`), drift-guarded at 1e-12. **Post-announce**, not required for v1.0.

**(F) De-id by allowlist + stop storing free-text institution (resolves C2).** Two parts, neither widening the hot path:
1. **Go-forward storage:** the `Registration.tsx:14` free-text `institution` field is replaced by an enum `institution_type` selector; the free-text `name` is retained operationally (needed for the UI) but is **classified as a direct identifier** and never enters gold. This realizes §4.1's "no free-text demographics," which the current form violates.
2. **ETL boundary:** gold's projection is an **explicit allowlist** of enumerated columns. `name`, free-text `institution`, email, IPs are *not in the SELECT list* — there is no "scrub" because the data is never read into gold. A hard ETL **anti-leak assertion + test** fails the build if any gold/export cell matches any known `participants.display_name` / `name` / email-localpart.

---

## 2. The Per-Participant 3-Bucket Model

The paper's organizing requirement — "pull everything per participant across ALL testing + learning sessions" — maps to three buckets, each a gold grain and an export file group.

### Bucket 1 — DEMOGRAPHIC (one row per participant)
- **Gold:** `gold.dim_participant` (PK `participant_sk`). Resolves demographics from the **most-recent non-null** per field across that participant's sessions (`result.participant.*`, allowlisted fields only), with `signup_expertise` as fallback. Adds `subspecialty`, `training_stage`, `age_band`, `region` (see §4). Free-text `name`/`institution` are **not projected** (C2).
- **As-of preservation:** `fact_test_session` keeps the **per-session** demographic snapshot, so longitudinal change is never silently overwritten by the dim's "latest" rule.
- **Export:** `/tables/participants.{csv,parquet}` + `/per_participant/<sk>/demographic.json`. **Pilot exception (C4):** for n≈7, individual-level quasi-identifiers are NOT published — only coarse aggregate marginals (see §4.4).

### Bucket 2 — TESTING (sessions → per-task outcomes → per-question SMC)
- `gold.fact_test_session` (one row per cert session): status, stop_reason, n_questions, durations, FKs to provenance + consent.
- `gold.fact_test_task_outcome` (one row per session × task k = the **publishable headline grain**): `verdict`, `final_auroc`, `auroc_hw`, operating point, `final_pass_mass` (π_k), `final_ell_mean`, `final_theta_mean`, `n_items_task`, `ell_star`, and **both accuracy definitions** `binarized_correct_rate` + `exact_class_match_rate` (resolves I1).
- `gold.fact_trial` (one row per session × trial): full SMC fidelity from `TrialDiag` — `ess`, `rejuvenated`, per-task arrays `pass_mass_by_task[7]`, `mcse_by_task[7]`, `info_gate_R_by_task[7]`, `ell_mean_by_task[7]`, `theta_mean_by_task[7]`, `auroc_hw_by_task[7]`, `verdict_by_task[7]`, plus `reaction_ms` joined from the `trials` table, **plus** `binarized_correct` and `exact_class_match` as **separate columns** (never a single ambiguous `is_correct`; I1).
- **Verdict transitions** are first-class via view `gold.v_verdict_transition(session_sk, task_k, from_verdict, to_verdict, at_trial_index)` (LAG over trial_index).
- **Trial-vs-blob reconciliation (resolves I3):** a gold validation diffs the `trials` table against `results.result["trials"]` per session, logs divergences with the documented rule **blob wins**, and **fails loudly** above a threshold. This is part of the R0 gate, not a footnote.
- **Export:** `test_sessions`, `test_task_outcomes`, `trials` (per-task arrays **exploded to long**: one row per session×trial×task) + nested in `/per_participant/<sk>/testing.json`.

### Bucket 3 — LEARNING (training sittings → trajectory points)
- `gold.fact_training_session` (one row per sitting): `source_session_sk`, `regimen_sk`, task_focus, parsed summary.
- `gold.fact_trajectory_point` (one row per participant × task × time): `training_sk` FK, `phase`, `ell`, `theta`, `sd`, `rt`, `seq_in_task`, **and `is_real`** (quarantine flag). **Realness is derived ONLY from `is_real=1`**, never from the API `sample` flag (C1).
- **Export:** `training_sessions`, `trajectory_points` (filtered to `is_real=1`) + `/per_participant/<sk>/learning.json`. **Until the real trainer ports, `is_real=1` rows = zero**; the export honestly contains no synthetic learning data and the datasheet states this plainly.

### The join spine — `gold.bridge_participant_journey` (one row per participant)
`participant_sk · test_session_sks[] · training_sks[] · trajpoint_count · first/last_test_utc · per_task_latest_verdict[7] · per_task_first_auroc[7] · per_task_last_auroc[7] · delta_auroc[7]`. One SELECT returns a participant's entire longitudinal record.

---

## 3. Provenance & Reproducibility Stamping

### 3.1 Stamp chain (capture → gold → export)
```
manifest.version + bundle.certBlock + git-SHA engine stamp
  → SessionIn kwargs → sessions.{engine_version,bundle_version,cert_config,engine_seed_int,engine_session_id}
    → gold.dim_provenance (deduped tuple; PK includes provenance_inferred — see §3.3)
      → fact_test_session.provenance_sk → fact_trial / fact_test_task_outcome inherit via session_sk
        → export_manifest.json records the DISTINCT provenance tuples (recorded vs inferred separated)
```

### 3.2 `gold.dim_provenance` is self-describing about the likelihood
Each row stamps the **Phase-6 HARD invariants** so every exported fact is reproducible from the export alone:
- `lapse_rate = 0.025` (`../../docs/INVARIANT_AUDIT.md` invariant #2)
- `logit_to_probit = 1/1.7 = 0.588235…` (invariant #1)
- `cert_config` → cross-linked to its `calibration/CALIBRATION_PROVENANCE.md` sha256 entry (D3 chain), e.g. the K=7 `ell_star` block.

**Seed binding closed (M2):** the engine session id `web-<info.seed>` (`App.tsx:189`) and the server `session_id` uuid (`app.py:580`) are distinct; we store the raw `engine_seed_int` and `engine_session_id` separately and **assert agreement at write time**, so reproducibility keys off a clean integer, not a parsed string.

### 3.3 Inferred vs recorded provenance must not collapse (resolves I6)
The `provenance_inferred` boolean is included **in the dim PK/hash**: `provenance_sk = hash(engine_version‖bundle_version‖cert_config‖provenance_inferred)`. An inferred pilot tuple therefore **never deduplicates** with a future identical *recorded* tuple. `export_manifest.provenance_tuples[]` surfaces the flag per tuple. The pilot's 100%-inferred provenance can never be silently laundered into "recorded."

### 3.4 Reproducibility is verified, not asserted (resolves C5)
Before any row is labeled reproducible, run the **round-trip** on the pilot sessions: take the inferred bundle (`/bundle/v1.5-k7`, `app.py:72`) + `sample_seed` + `cert_config v13`, re-run the engine under the BLAS contract (`OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), and assert **bit-equality** (repo's 1e-12 determinism convention) against the stored verdicts.
- **If it passes:** label those rows `reproducible=true` (still `provenance_inferred=true`, documented).
- **If it fails:** the data-availability statement reads *"verdicts as-recorded; engine re-derivation not guaranteed for the pilot cohort,"* and `reproducible=false` is set per row. No reproducibility claim ships unverified.

### 3.5 `export_manifest.json`
Records: `schema_version`, `export_utc`, `code_commit`, `gold_build_commit`, `source_snapshot{kind, captured_utc, dump_sha256}` (enforced, not optional), `provenance_tuples[]` (with lapse_rate + logit_to_probit + `provenance_inferred` + `reproducible`), `consent_versions[]` (+ `publishable` flag), `row_counts{}`, `checksums{path: sha256}`, `deidentification{policy, cohort_n, k_or_aggregate_only, quasi_identifiers[], suppressed_cells, generalized_classes}`, the country→sub-region map sha256 + age-band cutpoints sha256 (M4), the codebook sha256 (M3), and the `BLAS_contract`. Mirrors `data/DATA_PROVENANCE.md` sha256-chain discipline.

---

## 4. Demographics + Consent + De-identification

### 4.1 Demographic field model (enums/bands only — de-id at ingest)
Existing per-session fields (kept, allowlisted): `role_expertise` (7-value enum, reuse `en.json:34-40`), `institution_type`/`practice_setting`, `years_reading_eeg`, `eeg_volume_per_month`, `self_rated_confidence` (1–5), `sex`, `country`, `race_ethnicity`, `color_vision`, `prior_test_taken`.
New fields to add: `subspecialty`, `training_stage`, `age_band` (decade bands, never DOB). `region` derived in gold from `country`.

**Rules:** store **bands/enums, not raw values**. Validate **server-side against an enum whitelist** (reject unknowns 400, mirroring `app.py:381–387`). **No free-text demographics anywhere in the research surface** — free-text `institution` is replaced by `institution_type` at the *form* (C2-F), and free-text `name` is never projected into gold (C2). `country` validated against ISO-3166. **Derivation tables published + sha256'd (M4):** the country→UN-sub-region map and the `age_band` decade cutpoints are vendored into the export and referenced by sha256 in the manifest, so a reviewer can reproduce `region`/`age_band` exactly.

**Collection point:** keep signup minimal (only `role_expertise` + the new `institution_type` enum that replaces free-text) — do not bloat the signup form 2 weeks before announce. Remaining new fields ride in the existing `result.participant` JSON via a **skippable "research profile" step** shown with the consent gate before first test; gold flattens them. All optional (IRB-friendly, non-coercive, non-blocking).

### 4.2 Consent (version + timestamp + IRB + withdrawal)
- Append-only `consent_events` (§1.2-D) is the auditable record; the export `publishable` decision keys off **current `consent_events`** (joined at export time), **never** the frozen denormalized `sessions.consent_*` stamp (resolves I5; the denormalized copy is operational display only, forbidden as a gate by assertion + test).
- A result whose session lacks a covering, non-withdrawn `research_irb` consent authorizing **de-identified publication** is **excluded by construction**.
- **Withdrawal (C3):** flips `publishable=false`; excludes from all *future* exports; cannot retract a released DOI (documented in consent copy + datasheet).
- Current consent copy (`CONSENT_VERSION="v1.0"`, IRB `2016P000058 / 2013P001024`, `Consent.tsx:8-9`) is **UI-only today** — the accept-click time (`App.tsx:230-235`) is never recorded; this plan records it (and `consent_ip` per M5 retention policy).

### 4.3 Identifier classification (Layer-C disposition)
| Field | Class | Export action |
|---|---|---|
| email, display_name/name, signup_ip, consent_ip, google_sub, password_hash | Direct identifier / secret | **NEVER projected into gold** (allowlist; C2) |
| free-text `institution` | Re-id risk | **Not stored go-forward** (replaced by `institution_type`); historical values **not projected** |
| `code` (also JWT subject, `app.py:465`) | Linkable pseudonym | **Replaced by export-only `participant_sk`**; never projected |
| all timestamps (`created_utc`, session times, consent_utc) | Timing quasi-identifier | **Generalize** → month, or relative `days_since_first_session` |
| `country` | Strongest combinatorial re-id risk | **Generalize** → UN sub-region (→continent if small-cell); pilot: aggregate-only (C4) |
| role_expertise, subspecialty, training_stage, institution_type, age_band, sex, years/volume bands, race_ethnicity | Quasi-identifiers | Pilot: **aggregate marginals only** (C4); larger cohort: cohort-aware k-anon |
| self_rated_confidence | Non-identifying | Keep |
| engine_seed_int, engine_session_id, sample_seed, engine/bundle/cert_config versions | Provenance | Keep (essential) |
| all trial/result/trajectory performance data | Scientific payload | Keep (no embedded free-text reaches gold — guaranteed by allowlist, not scrub) |

**Anti-leak guarantee (C2):** because gold is an allowlist projection, no free-text field is ever present to scrub; a hard assertion/test additionally fails the build if any gold cell matches a known identifier value. **Two highest-risk crosses** (flag for the de-id reviewer, relevant only at larger n): `region × subspecialty × training_stage`; `region × institution_type × age_band × sex`.

### 4.4 Pseudonym + cohort-aware disclosure control (resolves C4)
- `participant_sk = HMAC(secret_salt, code)[:16]` — **deterministic** (stable longitudinal linkage across releases) but **un-invertible** without the salt, which lives only in Layer 1 (sealed, never published). Crosswalk `code → sk` stays in gold for audits/withdrawal.
- **k is a function of cohort size, not a fixed constant.** k-anonymity over individual-level quasi-identifier crosses is **statistically vacuous at n≈7** and reads as a red flag to a Nature Medicine reviewer (7 epileptologists with institution_type + region + subspecialty are trivially re-identifiable regardless of k).
  - **Pilot release (n≈7):** publish **performance data + coarse aggregate marginal counts only** (e.g., "expertise: 4 attending / 3 fellow"); **no individual-level quasi-identifier rows and no crosses**; suppress region/subspecialty/institution at the individual grain entirely. The statistician signs off that the pilot export is "performance data + coarse aggregates only."
  - **Larger cohort (future):** apply k-anonymity with **k chosen from cohort size** (documented rule) over `{region, role_expertise, subspecialty, training_stage, institution_type, age_band, sex, years_experience, prior_eeg_volume, race_ethnicity}`; for any class with n < k, **generalize up** then **suppress** the quasi-identifier value (`withheld_small_cell`) — never drop performance rows. `race_ethnicity` default: coarse 3-level or suppress unless the class clears k.
- Every generalization/suppression/aggregation logged in `DEIDENTIFICATION.md` with rule + affected counts (IRB/reviewer-auditable).

---

## 5. Migration / Backfill — the ~7 pilot participants

### 5.1 Recoverable retroactively from existing JSON (no operational change)
For every **completed** pilot session:
- **Demographics** — allowlisted `result.participant.*` from `sessions.participant` / `results.result` JSON (expertise, institution_type, years, volume, confidence, sex, country, race_ethnicity, color_vision, prior_test_taken, consent_version, irb_protocol_id). Free-text `name`/`institution` are **not** ingested.
- **Per-task outcomes** — verdicts + AUROC + **both accuracy definitions** recomputable from `results.result["trials"]` (`TrialDiag`).
- **Per-question SMC** — ess, pi, mcse, R, lMean, tMean, aurocHw, verdicts, nPerTask, s, sSd — all present in `results.result["trials"]`.
- **Reaction time** — from the `trials` table. **Missingness quantified (I2):** RT lives only in the fire-and-forget `/api/progress` checkpoint (`api.ts:370`) and is non-random (correlates with network/fast answers); gold computes per-session `rt_present_frac` and publishes it (§5.3). **No imputation.**
- **Sample seed** — `sessions.sample_seed`.

### 5.2 Permanently lost for the pilot cohort (record honestly)
- **Per-task posterior SD (`lSd`/`tSd`)** — never left the worker (dropped at `session.ts:246`). Unrecoverable historically; available only after the §1.2-E engine change, going forward.
- **Engine/bundle/cert_config version + explicit engine seed** — never stamped. **Backfill heuristic:** the pilot ran on a single known bundle (`/bundle/v1.5-k7`, `app.py:72`) + `cert_config v13`; backfill as **`provenance_inferred=true`** rows, kept distinct in the dim by hashing the inferred flag into the PK (§3.3), and **only labeled `reproducible=true` after the §3.4 round-trip passes** (C5).
- **Consent timestamp** — only the version string exists; accept-click time was never captured. Backfill `consent_version='v1.0'`; leave `consent_utc = NULL` (do **not** fabricate).
- **Learning trajectories** — the only persisted trajectory points are **fabricated constants** (`Shell.tsx`), now quarantined `is_real=0` (including the C1 backfill of all existing rows) → the pilot contributes **zero real learning data**. Stated plainly in the datasheet.
- **Signup expertise** for accounts created before fix (A) — dropped; the richer per-session expertise covers anyone who took a test.

### 5.3 Honesty mechanism in the dataset
A `data_completeness` block per participant/session in the codebook + manifest: booleans/floats `has_posterior_sd`, `provenance_recorded` vs `provenance_inferred`, `reproducible` (set only after §3.4 round-trip), `consent_ts_recorded`, `has_real_learning_data`, and **`rt_present_frac`** (I2). The pilot rows will show `provenance_inferred=true`, `reproducible=<round-trip result>`, `has_posterior_sd=false`, `has_real_learning_data=false`, and an explicit RT-coverage fraction — transparent to reviewers.

---

## 6. Phased Rollout (does NOT destabilize live prod before announce)

**Pre-announce hotfix (highest urgency — stops active contamination):**
- **Phase H0 (C1 hotfix):** gate/remove the `Shell.tsx:1024` synthetic-trajectory write so no further practice session contaminates the real `param_trajectories` table. Smallest possible client change; ship first. Gate: no new rows appear for a practice session; live dashboard unaffected for existing users.

**Pre-announce (next ~2 weeks) — lowest-risk, highest-value, no hot-path touch:**
- **Phase R0 (zero operational risk):** Build the `gold` ETL over **existing JSON blobs only**, allowlist projection. Gate: gold reproduces live `/api/dashboard` + `/api/history` numbers for the same participant **AND** the trial-vs-blob reconciliation (I3) passes its threshold **AND** the anti-leak assertion (C2) passes.
- **Phase R1:** `ilae-export` + version-pinned codebook (M3) + cohort-aware disclosure control (C4) + manifest. Gate: manifest checksums stable across two runs; pilot export is "performance + coarse aggregates only"; every exported column has a codebook row matching `schema_version` (M3); derivation tables sha256'd (M4). *(R0–R1 never touch Layer 0.)*
- **Phase O1 (additive, independently shippable):** **(A)** persist signup expertise + **(D)** `consent_events` + `POST /api/consent` + withdrawal op (C3). Drop-nothing, reversible. Gate: 282-suite green; live signup/auth/results unchanged; per-table migration helper `_migrate_sessions`/`_migrate_participants` idempotent on both backends (M1).
- **Phase O2:** **(C)** learning-linkage FKs + `is_real` quarantine flag + **the one-time `UPDATE is_real=0` backfill of all existing rows** (C1). Gate: existing trajectory write still 200s; ALL historical rows show `is_real=0`; new columns null/0 for old clients; `_migrate_trajectories`/`_migrate_training_sessions` idempotent both backends (M1).

**At/after announce (still additive, but more surface area):**
- **Phase O3:** **(B)** provenance + consent stamping on `sessions` + real `/api/manifest` version/certBlock + `Bundle.inputs` version re-export + raw-integer seed storage with the agreement assertion (M2). Drift-guarded SessionIn kwargs. Gate: old SPA builds still work (provenance null + back-fillable).
- **Phase O4 (engine):** the §1.2-E bit-safe `TrialDiag` extension (`lSd`/`tSd`, per-trial `aurocMean`, `chosenLoss`) + carry RT into `result.trials` (closes the I2 go-forward gap). Drift-guard the engine numerics (1e-12).
- **Phase L1 (the real trainer):** port `cortex_web_development/web_session.py` (the bit-faithful `CortexSession` wrapper) into the production write path, replacing the `Shell.tsx` fabricated constants; write real `param_trajectories` with `is_real=1`, `training_id`, `seq_in_session`, and a new `training_trials` table (the per-question `tel` dict has no home today). Largest piece; explicitly **post-announce**.

**Ordering rationale:** H0 stops the bleed immediately. R0/R1 deliver the paper's data product from day one using only historical JSON — no prod risk. O1/O2 stop the cheapest data losses (expertise, learning-linkage) and execute the C1 backfill before announce. O3/O4/L1 raise go-forward fidelity after the launch-stability window.

---

## 7. Risks, Open Questions, Sign-offs

**Needs PI / statistician sign-off:**
- **Pilot disclosure policy (C4):** confirm the pilot export is "performance data + coarse aggregate marginals only," no individual-level quasi-identifier crosses; and the cohort-size→k rule for future releases (+ `race_ethnicity` policy).
- **Headline accuracy definition (I1):** both `binarized_correct` (from `diag.y`, db.py:416) and `exact_class_match` are computed and exposed as separate columns; statistician picks the headline. Never a single ambiguous `is_correct`.
- **RT missingness (I2):** acceptability of publishing RT with a disclosed non-random missingness fraction; confirm no imputation.
- **Per-task certificate vs single roll-up** — repo's working policy is per-task, no roll-up (D5 / `OPEN_DECISIONS.md`). Confirm for the paper.

**Needs IRB sign-off (before the first DOI):**
- **DOI irrevocability + withdrawal limits (C3):** consent versions authorizing **de-identified publication** (drives `publishable`); explicit acknowledgement that released DOI'd exports cannot be retracted and withdrawal affects future releases only.
- **Reproducibility-claim wording (C5):** whether the data-availability statement says "reproducible" (round-trip passed) or "as-recorded" (round-trip failed/not run).
- Backfilling pilot consent without a recorded timestamp (`consent_utc=NULL`).
- **`consent_ip` retention (M5):** indefinite IP retention permitted, or define a truncation/retention policy.

**Open engineering questions:**
- **Gold substrate:** separate analytics DB vs `gold` schema on the same cluster (read-replica). Recommend replica or transactionally-consistent nightly restore for isolation (I4).
- **Reconciliation threshold (I3):** the divergence count above which R0 fails loudly — set with the statistician.
- **`training_trials` table shape (L1):** home for the per-question `tel` telemetry dict — design at L1, drift-guarded.

---

## 8. Concrete Next Steps

1. **Ship Phase H0 hotfix** — stop the `Shell.tsx` synthetic-trajectory write **today**; it is contaminating the real table on every practice session (C1).
2. **Review this plan** (PI + IRB + statistician) — specifically the §7 sign-off items, with C3/C4/C5 front-loaded before any export artifact.
3. **Build Phase R0** (`ilae-gold build`/`verify`) over existing JSON, allowlist projection; prove gold == live API for the 7 pilots, trial-vs-blob reconciliation passes (I3), anti-leak assertion passes (C2). *(Zero prod risk.)*
4. **Build Phase R1** (`ilae-export run`/`codebook`/`manifest`) with cohort-aware disclosure control (C4), version-pinned codebook (M3), sha256'd derivation tables (M4); produce a v0 **performance-plus-coarse-aggregates** export of the pilot for IRB review of the actual artifact.
5. **Run the §3.4 reproducibility round-trip** on the pilot; set `reproducible` per row; finalize the data-availability wording (C5).
6. **Ship Phase O1** (signup_expertise + consent_events + POST /api/consent + withdrawal op) behind drift-guard tests + per-table migration helpers (M1); 282-suite + live smoke gate.
7. **Ship Phase O2** (learning-linkage FKs + `is_real` quarantine **+ the one-time backfill of all existing rows to `is_real=0`**) (C1).
8. **Schedule post-announce** O3 (provenance stamping + raw-seed M2), O4 (engine `TrialDiag` + durable RT, closing I2 go-forward), L1 (real trainer port + `training_trials`).

**Key source anchors (all absolute):**
`/data/eli-work/repos/ilae-skill-certification-test-multi/cortex_web/services/api/db.py` (schema `:33-126`, migration `:131-200`, participants helper `:155-167`, `upsert_trial` `:393`+`is_correct` `:416`, `create_session` `:367`, `append_trajectory_points` `:535`, `create_training_session` `:507`; **no DELETE path anywhere**) · `…/services/api/app.py` (expertise drop `:98`+`:392`, `SessionIn` `:117`, manifest stub `:574`, default bundle `:72`, session uuid `:580`, trajectories `sample:false` flip `:629-637`) · `…/apps/web/src/App.tsx` (result blob `:170-182`, consent accept `:230-235`, engine seed `:189`) · `…/apps/web/engine/types.ts` (`TrialDiag` `:73-90`, `certBlock` `:30`) · `…/apps/web/engine/particles.ts` (SD computed `:188-193`) · `…/apps/web/engine/session.ts` (SD discarded `:246`) · `…/apps/web/src/components/{Registration.tsx:12-27 (name :13, free-text institution :14), Consent.tsx:8-9, Shell.tsx:1015-1027 (synthetic write :1024)}` · `…/apps/web/src/api.ts` (fire-and-forget `:370`, participant POST `:358-363`) · `scripts/session_controller.py` (real telemetry `:474-507`) · `cortex_web_development/web_session.py` (port target).

---

## 9. Changes from the critique

**Critical (all resolved):**
- **C1** — Reframed fabricated trajectories from "served back as sample:false" to **active prod contamination** (rows already persisted via `Shell.tsx:1024`→db.py:535; `app.py:629-637` mislabels `sample:false`). Added **Phase H0 pre-announce hotfix**, a one-time **`UPDATE is_real=0` backfill of all existing rows** (not just `DEFAULT 0`), and an ETL rule that derives realness only from `is_real=1`, never the API `sample` flag. (§0.3, §0.4, §1.2-C, §2-Bucket3, §5.2, §6-H0/O2, §8.)
- **C2** — Replaced "name-scrub" denylist with **allowlist projection**: free-text `name`/`institution` are never selected into gold; added a hard anti-leak assertion/test; stop *storing* free-text institution go-forward (form uses `institution_type` enum). New §1.2-F. (§0.2-#5, §0.3, §1.2-F, §4.1, §4.3.)
- **C3** — Added a **real withdrawal operation** (flips `publishable=false`, future-export exclusion) and explicit documentation that **DOI'd releases are irrevocable**; IRB sign-off required before first DOI. (§1.2-D, §4.2, §7.)
- **C4** — Made **k a function of cohort size**; pilot (n≈7) publishes **performance data + coarse aggregate marginals only**, no individual-level quasi-identifier crosses. (§4.3, §4.4, §7.)
- **C5** — Added a mandatory **reproducibility round-trip** (re-derive verdicts, assert 1e-12 bit-equality) before labeling any row reproducible; downgrade wording on failure. New `reproducible` completeness field. (§3.4, §5.2, §5.3, §8-step5.)

**Important (all resolved):**
- **I1** — Compute and expose **both** `binarized_correct` and `exact_class_match` as separate columns; statistician picks the headline. (§2-Bucket2, §7.)
- **I2** — **Quantify RT missingness** as `rt_present_frac`, publish it, state the non-random mechanism, no imputation; carry RT into durable `result.trials` at O4. (§5.1, §5.3, §6-O4, §7.)
- **I3** — Added an explicit **trial-vs-blob reconciliation** (blob wins) that fails loudly above a threshold, as part of the R0 gate. (§2-Bucket2, §6-R0, §7.)
- **I4** — Mandated a **transactionally-consistent** snapshot and an ETL rule excluding `status != 'complete'` sessions; enforced `dump_sha256`/`captured_utc` as a gate. (§1.1.)
- **I5** — Export `publishable` keys off **current `consent_events`**, never the frozen denormalized `sessions.consent_*` (forbidden as a gate by assertion/test). (§1.2-D, §4.2.)
- **I6** — `provenance_inferred` is hashed into the `dim_provenance` PK so inferred and recorded tuples never deduplicate. (§3.1, §3.3, §5.2.)

**Minor (folded in):**
- **M1** — Added per-table idempotent migration helpers (`_migrate_sessions`, `_migrate_trajectories`, `_migrate_training_sessions`) for both backends. (§1.2 preamble, §6-O1/O2.)
- **M2** — Store the **raw integer** seed + engine session-id string separately; assert agreement at write time. (§1.2-B, §3.2.)
- **M3** — Gate export on every column having a codebook row matching `schema_version`; codebook sha256 in manifest. (§3.5, §6-R1.)
- **M4** — Vendor + sha256 the country→sub-region map and age-band cutpoints in the manifest. (§4.1, §3.5.)
- **M5** — Flagged indefinite `consent_ip` retention for IRB; optional truncation/retention policy. (§1.2-D, §4.2, §7.)

**Deliberately deferred (with rationale):**
- **Per-task posterior SD (`lSd`/`tSd`), per-trial `aurocMean`/`chosenLoss`, durable RT** — require an engine-source change; gated to **Phase O4 (post-announce)** to protect the launch-stability window. Pilot is honestly marked `has_posterior_sd=false`.
- **Real learning data (`is_real=1`)** — awaits the **Phase L1** trainer port (largest, post-announce); pilot contributes zero real learning data by construction, stated in the datasheet.
- **Larger-cohort individual-level demographic release with cohort-aware k-anon** — deferred until n supports it; pilot ships aggregates only.
