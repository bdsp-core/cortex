# Local draw-latent session console

A fully local session server plus browser page for sitting a real adaptive
n-way IIIC session against the NEW draw-latent engine
(`responseAggregation: "draw_latent"`, construction B) before any production
integration. No external network: the server binds 127.0.0.1, reads only
repo-local files, and uses node's built-in `http` module; the page is a single
static HTML file with vanilla JS.

## Launch

```bash
scripts/local_server.sh            # build + serve on http://127.0.0.1:8734/
PORT=9000 scripts/local_server.sh  # alternative port
```

The build follows the precision-CLI sidecar pattern: `tsc --noEmit` against
`local_server/tsconfig.json` for type safety, then the toolchain's `rolldown`
bundles `local_server/server.ts` into `.local-server-dist/local_server.mjs`.

Inputs read at boot (both must exist locally):

- `.artifacts/categorical_bank_axes.csv` — the real served bank
  (20,502 IIIC segments; untracked staged research input).
- `artifacts/iiic_conditional_f1_engine_frame_atoms17_rd.json` — the 17-atom
  engine-frame artifact (research-only; every atom carries the 0.15 lapse
  floor; there is deliberately no lambda=1 collapse atom).

## What a session is

Each session drives the isolated TS engine read-only, production-shaped:

- 1200-particle draw-latent cloud (`makeProtocolState`, per-particle atom
  lineage sampled from the artifact weights), identity prior blocks;
- item selection through `chooseProtocolCandidate`
  (`categorical_fisher_totalvar_v1`) over a seeded 420-segment draw from the
  real bank, restricted to domains the policy still reports ACTIVE;
- posterior updates + ESS-gated resample/rejuvenate (0.5 ESS fraction, 30 MH
  steps) through `advanceProtocol`;
- stopping via the UNCHANGED production `PrecisionPolicy` evaluated in-process
  through `src/precision_bridge.ts` after every answer: per-domain cap 60,
  nMin 20, content-band edges = full-served-bank signal terciles (identical to
  the values production derives before any per-session draw; the spike domain
  has no candidates on this bank and terminalizes as UNDETERMINABLE_BANK on
  the first evaluation).

Simulated readers answer with the conditional-F1 truth model exactly as
`draw_latent_rd/harness.py` does: truth `t, l ~ N(0, 1)` per domain (`l`
shifted by `skill_shift`), pick sampled from
`f1_probabilities(truth, segment, asked_k, beta, lapse)`.

## Production-layout data capture

Every session (manual and simulated) persists to a local SQLite database in
the **exact cortex_web production table layout**, so testing here produces
analyzable data in prod shape and analysis written against it later runs
against the real production tables unchanged.

- **DB file**: `local_server/data/local_test.db` (the whole
  `local_server/data/` dir is gitignored). Override the data dir with
  `LOCAL_SERVER_DATA_DIR` (the sanity check does, to keep its scratch DB
  separate).
- **Schema**: `local_server/schema.sql`. Two tables:
  - `sessions` — column-for-column the production table:
    the base columns from
    `cortex_web/services/api/persistence/schema.py`, plus the additive
    columns from `persistence/migrations.py` `SESSIONS`
    (`bundle_version`, `drawn_seg_ids`, `termination_policy`,
    `compute_mode`, `candidate_exclusion`, `candidate_bank_sha256`,
    `nway_profile`, `norm_id`, `norm_sha256`, `score_schema_version`,
    `norm_profile`), plus the approved additive columns from this repo's
    `server/migration.sql` (`engine_profile_id`, `response_model`,
    `response_artifact_id`, `response_artifact_sha256`, `selector_version`,
    `engine_algorithm_version`). No extra columns.
  - `trials` — column-for-column the production table
    (`persistence/schema.py` + `migrations.py` `TRIALS`):
    `session_id, trial_index, seg_id, task_k, pick, is_correct, reaction_ms,
    diag, received_utc, shown_client_utc, answered_client_utc`, primary key
    `(session_id, trial_index)`, `trial_index` 0-based as in prod.
  - Local relaxation (names/types/PKs are the parity claim, constraints are
    not): NOT NULL / FK constraints tied to the production accounts system
    are dropped because local sessions have no participant account
    (`code` is NULL here).
- **Honest fills** (`sessions`): `participant` = `owner-local` (manual) or
  `sim:<preset>` (simulated; `sim:custom` for explicit reader params);
  `sample_seed` = the session seed; `status` uses the production vocabulary
  (`in_progress_nway` open, `complete` at stop); `drawn_seg_ids` = JSON
  array of the session's seeded bank draw (REAL seg ids);
  `candidate_bank_sha256` = sha256 of the served bank CSV;
  `termination_policy` = `precision_frozen`; `compute_mode` =
  `local_isolated_draw_latent_rd`; `nway_profile` = JSON of the engine
  profile including `responseAggregation` (sorted keys, as prod stores it);
  `response_artifact_id`/`sha256` from the atoms17 artifact file;
  `bundle_version` = branch name + short commit of HEAD at server boot;
  `candidate_exclusion`, `norm_*`, `score_schema_version`, `code` are NULL
  (no temporal exclusion list, norms, or accounts locally). A manual session
  handed off to a simulated reader keeps `participant` = `owner-local`
  (mode at creation); its simulated trials are distinguishable by NULL
  `reaction_ms`/client stamps.
- **The `diag` column carries the draw-latent measurement payload** (that is
  what the TEXT column is for in prod), JSON per trial, post-update:
  `atom_posterior` (17 atom masses), `atom_map_beta`, `atom_mean_beta`,
  `skill`/`bias` (per-domain posterior mean + equal-tailed 95% CI, K=7),
  `ess`, `resampled` (ESS-gated resample/rejuvenate flag),
  `precision_statuses` (per-domain Precision selection states, K=7), and
  `policy_stop`.
- **Local-gold convention**: the asked domain IS the presented item's class
  (the harness convention — `task_k` is the class the segment was served
  as), so `is_correct = (pick == task_k)`. This differs from production
  gold scoring, where `is_correct` is scored against an independent gold
  label; treat local `is_correct` accordingly.
- **Timing**: `reaction_ms`, `shown_client_utc`, `answered_client_utc` are
  measured by the BROWSER page (armed when a question renders, captured on
  the answer click, sent with the answer POST). Simulated answers persist
  NULL for all three. `received_utc` (and `started_utc`/`finished_utc`) are
  server-side stamps in the production canonical ISO-Z second format.
- **Write path** (why there are two layers): this node is v20 (no
  `node:sqlite`) and `better-sqlite3` is not vendored under
  `../cortex_web/node_modules`, so the server journals every row as
  newline-JSON in the same two-table shape
  (`data/ndjson/<session>.session.json` + `<session>.trials.jsonl`, written
  at create/every answer) and invokes `local_server/persist.py` (python3
  stdlib `sqlite3`) synchronously at session stop to materialize the real
  `.db`. Rebuild the DB from the journal at any time (including sessions
  abandoned before stopping) with:

  ```bash
  python3 local_server/persist.py --db local_server/data/local_test.db \
      --data-dir local_server/data --all
  ```

### Analyzing captured sessions

```bash
python3 local_server/analyze.py                            # default DB
python3 local_server/analyze.py path/to/local_test.db      # explicit DB
```

Stdlib-only report, per session: per-domain burden and accuracy (with final
Precision statuses), reaction-time p50/p90 over manual trials, stop reason,
and the atom-posterior trajectory (posterior-mean beta after trials
10/25/50/end, final MAP atom), followed by a cross-session table. Because
the layout is the production layout, the same queries run against real prod
tables unchanged (the `diag` payload and the local-gold `is_correct`
semantics above are the local-only parts).

One deliberate latency deviation: the selector's shortlist quotas are reduced
(`coarsePerTask 6, entropyPerTask 3, fisherPerTask 3`) so a browser answer
returns in about a second; the exact total-posterior-variance loss that makes
the final choice is unchanged. Expect rejuvenation-heavy answers late in a
session to take several seconds — history replay grows with question count.

## HTTP API

- `POST /session` — body `{mode: "manual"|"simulated", reader?: {preset?|beta,
  lapse, skillShift}, seed?, bankSegments?, particles?, mhSteps?, truthSd?,
  selector?}`. Presets: `sharp_expert` (beta 1.6), `typical` (beta 1.0),
  `diffuse` (beta 0.7), `guesser` (lapse 1.0).
- `GET /session/:id` — current item (asked class + question number), per-domain
  posterior mean and 95% CI for skill `l` and bias `t`, atom posterior over the
  17 betas, per-domain Precision status, stopped flag + reason.
- `POST /session/:id/answer` — `{pick: 1..6}` (Seizure, LPD, GPD, LRDA, GRDA,
  IIC/Other) plus optional browser timing `{shownClientUtc, answeredClientUtc,
  reactionMs}` (the page sends these; absent/invalid fields persist as NULL).
- `POST /session/:id/autostep` — `{n, reader?}`; simulates `n` answers. On a
  manual session, passing `reader` attaches a simulated reader (hand-off).
- `GET /sessions` — persisted sessions (status, question counts, stop reason,
  whether the SQLite file has the row yet).
- `GET /healthz` — bank/artifact/profile stamp.

## The page

- **Current question** — question counter, asked class, six answer buttons, and
  a collapsible table of the segment's six latent signals (a served session
  would show the EEG instead; the table lets you role-play an informed reader).
  The Simulate buttons run the selected preset from the setup card.
- **Domain status** — chips with the Precision selection state and n/60 per
  domain, plus the point-centered radius vs. its tolerance target and the
  persistence streak.
- **Per-domain skill estimates** — posterior mean with equal-tailed 95%
  credible interval per IIIC domain, prior-SD units.
- **Inferred distractor sharpness** — posterior mass over the 17 artifact
  atoms; the dashed line is the uniform prior weight, the labeled bar is the
  MAP atom.
- **Bias estimates** — report-only `t` coordinate; never consulted by stopping.
- **Session summary** — appears at stop: reason, question count, per-domain
  final statuses and skill intervals, MAP and posterior-mean beta.

## Suggested experiments

1. **Sit a session yourself** (mode Manual). Answer from the signal table (pick
   the class with the strongest signal to play an expert; answer against it to
   play a biased reader) and watch the CIs tighten and domains complete. Note
   the policy requires at least 20 questions in each of six domains, so a full
   manual sitting is 120+ answers — you can hand off to a simulated reader at
   any point with the Simulate buttons.
2. **Run the Guesser preset** and watch the atom posterior. The 17-atom
   artifact has NO lambda=1 atom, so uniform wrong picks cannot be represented
   exactly: expect degraded-but-floored behavior, not a clean "guesser
   detected" signal — atom mass drifts toward the lowest-beta atoms (the
   flattest distractor softmax the floored frame offers), skill estimates go
   low and wide, and domains take most of their 60-question caps.
3. **Run Sharp expert (beta 1.6)** and watch mass shift toward the high-beta
   atoms. On a single session this shift is directional, not certain: atom
   lineage is fixed at cloud creation and rides resampling, so the R&D
   campaign's MAP atom tracks the world beta only about half the time at 17
   atoms. Re-run with a few seeds to see the spread.

## Sanity check

```bash
node local_server/sanity_check.mjs         # production shape (several minutes)
node local_server/sanity_check.mjs --fast  # reduced smoke (about a minute)
```

Boots the server on port 8735 (with a scratch persistence dir under
`.local-server-dist/sanity-data/`), runs a simulated sharp-expert session to
completion through the HTTP API, and asserts it stops via the policy
(`all_estimated_or_undeterminable`) with sane question counts (120..360),
terminal per-domain statuses, and a normalized atom posterior. It then
asserts the production-layout capture: the SQLite DB exists, holds exactly
one `sessions` row (`complete`, stop reason populated, `sim:sharp_expert`,
full `drawn_seg_ids`), `trials` rows equal to `n_questions` with every
`diag` parsing to 17 atom masses, `GET /sessions` lists the sitting, and
`analyze.py` renders its report from the DB.
