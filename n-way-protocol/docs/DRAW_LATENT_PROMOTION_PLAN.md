# Sequential plan: promote draw-latent B to replace the prod incumbent

Status: **PLAN ONLY — awaiting owner approval.** Creating this document is the
only repository change made while drafting it. Nothing is executed, no file is
edited, no file or branch is deleted.

Execution model: phases run in order; each phase has a **goal**, **actions**,
**verification**, **rollback**, and a **STOP condition**. A STOP condition
halts autonomous execution and returns to the owner. No phase deletes a file.

## Owner decisions applied (2026-08-01)

1. **Phase 1c stands as a hard kill gate** — real-response evidence supersedes
   simulation. If the prod-session replay does not favour draw-latent, stop and
   report the refit recommendation instead of promoting.
2. **Phase 1d (bias-RMSE regression) is downgraded to non-blocking** — measure,
   document in the campaign report, do not gate on it.
3. **In-progress production sittings may be forced to restart.** Preserving
   resume for the legacy unfloored stamp is no longer a blocking requirement;
   the three affected sittings are to be explicitly superseded (a clean
   "start a new sitting" path), never left to fail with an opaque error.

Defaults chosen by the implementer for the still-unratified items in 1a, each
to be surfaced for ratification in the final report rather than presented as
settled: atom grid = 33 quantiles at the out-of-sample heterogeneity
(τ = 0.3581) **plus** the λ_d = 1 binary-nesting atom (battery evidence:
collapse coverage 0.43 → 0.93, and F1 then nests binary); particle budget held
at 1,200 with a documented floor of ≥ 35 particles per atom and
atoms-alive reported per cell; noninferiority margin = the −0.03 house margin;
DR07 criterion v2 executed as written.

---

## Phase 0 — Safety net (must complete before anything else)

**Goal:** make every subsequent step reversible before any of them happen.

**Actions**
1. Annotated git restore point on the current main:
   `git tag -a restore/main-pre-draw-latent origin/main -m "main as it stood before draw-latent promotion"` and push the tag.
2. Archive tags for every branch slated for deletion (tags keep the commits
   permanently reachable, so deletion loses nothing):
   - `git tag -a archive/nway-f1-promotion-20260731 nway-f1-promotion-20260731 -m "..."`
   - `git tag -a archive/wip-precision-spike-20260731 wip/precision-spike-20260731 -m "..."`
   - `git tag -a archive/v15-web-version v15-web-version -m "..."`
   - Push all three.
3. Preserve `stash@{0}` (87 files, pre-existing WIP) as a real commit:
   `git stash branch archive/stash-20260730 stash@{0}` then tag it and delete
   the temporary branch (the stash itself is left intact — nothing dropped).
4. Prod database snapshot: trigger the existing `backup_to_box.sh` pg_dump path
   and verify the dump exists and is non-empty.
5. Record the current prod release id (`/opt/cortex/release-state/current`,
   presently `20260725T231233Z-285f264a42ba`) and confirm
   `/opt/cortex/release-state/previous` is populated.
6. Capture baseline suite counts on origin/main: engine vitest, apps/web
   vitest, apps/web tsc, services/api pytest, n-way pytest + vitest.

**Verification:** all tags resolve (`git rev-parse <tag>`); pg_dump present;
release-state pointers readable; baseline counts recorded to a file.

**Rollback:** n/a (creates only).

**STOP if:** any tag fails to push, the pg_dump is missing/empty, or
`release-state/previous` is empty (no prod rollback target).

---

## Phase 1 — Close the scientific gates that currently block promotion

The artifact is `promotionForbidden` and its `qualification` is not
`"qualified"`. These gates are what make it qualifiable. **Any failure here
ends the promotion under the standing park rule.**

### 1a. Owner decisions (blocking input, not autonomous)
- Tail-atom design: does the grid carry extra mass above β≈1.6 for the
  sharp-reader tail (β=1.903 world scored 0.906, sub-nominal)?
- Include the λ_d≈1 binary-nesting atom? (Recovered collapse 0.43 → 0.93.)
- Particle-per-atom floor, or an atom-refresh MH move, given measured
  depletion (17 atoms at 300 particles collapsed 17→6).
- Noninferiority margins for the locked run (house margin is −0.03).
- Ratify `docs/DR07_CRITERION.md` v2 (currently executed under blanket
  approval, not explicit ratification).

### 1b. Harness fidelity fix (blocking, autonomous)
The research harness offers every domain until the safety cap; production
offers only `ACTIVE` domains (`advance.ts:283,399`), and `ESTIMATE_COMPLETE`
is not sticky. Burden numbers from any harness run are therefore an upper
bound. Fix `_select` to mirror production, re-run the four-arm burden study,
and **re-state whether B still stops no later than the incumbent.**

### 1c. Real-response arbitration (blocking, autonomous) — the missing evidence
B fixes aggregation; it does **not** fix the asked≠gold fit-population
mismatch found in production. Before replacing the incumbent we must know
which engine better describes *real* responses. Replay the 13 real prod
n-way sessions (already extractable, no PII needed) through: binary,
incumbent (unfloored mixture), floored mixture, and draw-latent B, scoring
one-step-ahead predictive log-likelihood of the observed picks.
- If B ≥ incumbent on real responses → promotion is defensible.
- If B ≈ incumbent and both < binary on the categorical channel → the
  mismatch dominates and **the correct action is to fix the fit population
  (re-fit on asked≠gold responses), not to ship B.** STOP and report.

### 1d. Bias-RMSE regression (blocking, autonomous)
Draw-latent showed bias RMSE +0.0348 vs the incumbent (CI excludes zero) in
the real-bank study while being better at equal burden. Diagnose before
promoting; a certification instrument may not ship an unexplained regression
in a reported parameter block.

### 1e. Monitor recalibration (blocking, autonomous)
The misspecification monitor's threshold is calibrated against the floor015
mixture reference. Recalibrate its operating characteristics for the promoted
atoms17 artifact and re-verify collapse detection.

### 1f. Locked campaign (blocking, autonomous)
Run `docs/LOCKED_CAMPAIGN_PREREG.md` amended for the draw-latent model and
the 1a decisions, with a fresh disclosed amendment, disjoint seed series, and
the owner margins. Gating cells: continuum SBC and served-bank adaptive
comparison. **Fail any gate → STOP, promotion ends.**

**Verification:** every gate green, all reports committed.
**Rollback:** none needed (research only; no production surface touched).
**STOP if:** any gate fails, 1c favours a refit, or 1d is unexplained.

---

## Phase 2 — Qualify the artifact

**Goal:** turn the research artifact into a servable one.

**Actions**
1. Re-emit the artifact with `qualification: "qualified"`, `promotionForbidden:
   false`, provenance chained to the merged locked-campaign reports and the
   DR07 gate report digest. Keep the existing research artifact file in place
   (new file, nothing deleted).
2. Regenerate engine constants from the artifact with
   `scripts/make_engine_profile_constants.py`, whose `--validate` must first
   reproduce the currently deployed table bit-for-bit.
3. Update the Python and TypeScript profile modules to carry the qualified
   stamp alongside — not instead of — the existing stamps.

**Verification:** artifact sha recomputed in both languages; `--validate`
passes; contract tests updated and green.
**Rollback:** revert commits; no runtime effect yet.
**STOP if:** the generator cannot reproduce the deployed table.

---

## Phase 3 — Make draw-latent the default, preserving every existing behavior

**Goal:** new sittings get draw-latent; everything already in flight is
untouched.

**Actions**
1. `nway_profile_for_new_session` returns the qualified draw-latent profile.
   Remove the non-production research escape only after the qualified path
   works (the escape becomes redundant, its test inverted to assert the
   qualified default).
2. **Resume compatibility (critical):** `allowed_nway_profiles` and the TS
   `validateNWayInputs` must continue to accept **every historical stamp**:
   - `precision_nway_f1_ensemble9_fisher_v1` (unfloored — what the 3 currently
     `in_progress_nway` prod sessions carry),
   - `precision_nway_f1_ensemble9_floor015_fisher_v1`,
   - the research atoms17 stamp,
   - the new qualified stamp.
   A resuming session keeps the response model it started with, per the
   promotion checklist ("rollback changes only new-session assignment and
   never changes an in-progress sitting's response model").
3. Leave the binary/AD6 path completely untouched.
4. Decide on `server/migration.sql`: the API stamps the whole profile as JSON
   into `sessions.nway_profile` and reads none of the additive columns.
   Default decision: **do not apply** (additive-but-unused columns add
   deployment risk for no benefit). Revisit only if analytics require them.

**Verification:** a test proves each historical stamp still resumes; a test
proves a new session gets the qualified draw-latent stamp; binary golden
suite byte-identical.
**Rollback:** revert to the previous default in one commit; stamps are
per-session so no data migration is implied.
**STOP if:** any historical stamp fails to resume.

---

## Phase 4 — Full verification before merge

**Actions:** engine vitest, full apps/web vitest, apps/web tsc, services/api
pytest, n-way pytest + vitest + typecheck (note: n-way typecheck has **one
pre-existing failure** at `src/precision_cli.ts:190`, whose 3-arg
`PrecisionPolicy.fromInputs` form lives only on the WIP branch — this plan
must either fix it as part of the merge or record it explicitly as
pre-existing), plus:
- binary golden suite unchanged (`pipeline.test.ts` vs `__testdata__/reference.json`),
- existing n-way mixture fixtures unchanged,
- v15 drift snapshot unchanged,
- resume-replay tests for all historical stamps,
- Python↔TS parity for the promoted artifact,
- a local cortex_web stack end-to-end session on the qualified profile.

**Verification:** every count matches or exceeds the Phase-0 baseline.
**STOP if:** any previously-passing test fails.

---

## Phase 5 — Clean merge to main

**Actions**
1. `git fetch`, confirm `nway-f1-promotion-20260731` is still 0 behind
   origin/main (it is today: 32 ahead, 0 behind → fast-forward).
2. Merge with `--no-ff` to keep an explicit, revertible merge commit:
   `git switch main && git merge --no-ff nway-f1-promotion-20260731`.
3. Re-run Phase-4 suites on the merged main.
4. Push main.

**Verification:** merge commit exists; suites green on main; `git diff
restore/main-pre-draw-latent..main` reviewed for unexpected paths.
**Rollback:** `git revert -m 1 <merge-commit>` (single-commit undo of the
entire promotion), or hard reset to `restore/main-pre-draw-latent`.
**STOP if:** the merge is not a clean fast-forward-able state or suites
regress on main.

---

## Phase 6 — Branch cleanup (only after the archive tags exist)

**⚠ Owner attention required — two branches hold work that exists nowhere else:**
- `wip/precision-spike-20260731` — your 54-file precision-spike/data WIP
  (5,546 insertions), never merged to main.
- `v15-web-version` — 1 commit from 2026-06-13, **325 commits behind main**,
  whose own message says "Held for the internal lab test, then merge to main."
  It is on `origin` as well. It has never been merged.

Deleting either branch without first deciding its fate discards that work
from the branch namespace. **Phase-0 archive tags make deletion safe**
(commits stay reachable and pushable forever via the tag), but they do not
merge the work — if you want that work *in* main, it must be merged first,
which is a separate decision this plan does not make for you.

**Actions (per branch, after tags are confirmed pushed):**
1. `git branch -d <branch>` for merged branches; `-D` only where the archive
   tag is verified present on origin.
2. `git push origin --delete <branch>` for `v15-web-version` (and any other
   remote-tracked branch approved for deletion).
3. Re-verify each archive tag resolves after deletion.

**Verification:** `git rev-parse archive/<name>` still returns the commit and
`git log archive/<name> -1` shows the expected work.
**Rollback:** `git branch <name> archive/<name>` recreates it exactly.
**STOP if:** any archive tag is missing from origin at deletion time.

---

## Phase 7 — Deploy to production, dark

**Goal:** the new code runs in production while behavior is unchanged.

**Actions**
1. Deploy main via the standard `deploy_app.sh` path, which creates a new
   timestamped release under `/opt/cortex/releases` and repoints
   `release-state/current`, retaining `previous`.
2. **Deploy with the rollout switch in its current state** — do not change
   `CORTEX_PRECISION_POLICY_ROLLOUT`. New sittings receive the qualified
   draw-latent profile only when Phase 8 says so; until then verify the build
   is healthy and old sessions resume.

**Verification:** `/api/health` ok; release-state/current advanced and
previous retained; the 3 `in_progress_nway` sessions resume successfully;
no new error classes in `cortex.clienterr` (per the Safari-crash lesson,
check client error telemetry, not just HTTP codes).
**Rollback:** `release_switch.sh rollback` (health-checked, restores current
on failure) or `deploy_app.sh`'s `rollback_remote`.
**STOP if:** health fails, any historical session fails to resume, or client
error telemetry shows new engine/worker failures.

---

## Phase 8 — Staged activation and observation

**Actions**
1. Offline shadow: replay recent real sessions against the deployed build;
   compare stamps and diagnostics. No user exposure.
2. Internal test: your own account only; sit a complete session.
3. Allowlist: named external participants, per the promotion checklist's
   observation gates.
4. Widen only after each stage's observation window is reviewed.

**Verification per stage:** monitor trip rate, burden distribution, ESS and
acceptance telemetry, client errors, and a manual read of at least one full
session's diagnostics.
**Rollback (fastest first):**
1. `CORTEX_PRECISION_POLICY_ROLLOUT=off` + restart → AD6/binary for new
   sessions (documented in the env file as the AD6 rollback).
2. Revert the default-profile commit and redeploy → back to the mixture stamp.
3. `release_switch.sh rollback` → previous release wholesale.
4. `git revert -m 1 <merge-commit>` on main → full source-level undo.
5. Restore DB from the Phase-0 pg_dump (only if data corruption, which no
   step here should cause — profile stamps are per-session and additive).
**STOP if:** monitor trip rate rises materially above the calibrated false-trip
target, burden exceeds the prior distribution, or any client-error class appears.

---

## Phase 9 — Standing undo guarantees (valid indefinitely after execution)

| level | mechanism | scope |
|---|---|---|
| runtime | `CORTEX_PRECISION_POLICY_ROLLOUT=off` | new sessions → AD6/binary, seconds |
| profile | revert default-profile commit, redeploy | new sessions → previous n-way stamp |
| release | `release_switch.sh rollback` | whole prod app → previous release |
| source | `git revert -m 1 <merge-commit>` | main → pre-promotion behavior, history preserved |
| absolute | `git reset --hard restore/main-pre-draw-latent` | main → byte-exact pre-promotion tree |
| data | Phase-0 pg_dump restore | database |
| branches | `git branch <name> archive/<name>` | any deleted branch, exactly |

In-progress sittings never change response model under any of these, because
the profile is stamped per session and validated on resume.
