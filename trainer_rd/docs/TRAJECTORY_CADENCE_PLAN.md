# Evolution-Curve Cadence Plan — per-session trajectory points

How the per-domain evolution curves (ℓ, θ, RT) on the deployed dashboard get
their data points: when a point is emitted, what it contains, and how the
trainer's running posterior produces it. This is the data/cadence contract
between the trainer stack in this directory and the deployed dashboard.

**No clinical terminology in this document.** The seven assessed targets are
referred to as `domain1 … domain7`, mapped to the engine task index `k`:

| engine `k` | this doc |
|---|---|
| 0 | domain1 |
| 1 | domain2 |
| 2 | domain3 |
| 3 | domain4 |
| 4 | domain5 |
| 5 | domain6 |
| 6 | domain7 |

So `domainN ≡ engine task index k = N − 1`.

---

## 1. Decision (locked)

1. **Cadence = per training session.** One trajectory point per `(domain,
   session)`, emitted when a session is finalized — *not* per calendar day and
   *not* per individual trial. A calendar day with three sessions yields three
   points; an idle day yields none.
2. **Two point kinds, distinguished by `phase`:**
   - **Anchor points** — `phase ∈ {eval, recert}` — produced by a full adaptive
     certification test. Independently meaningful, tight posteriors. Rendered
     **darker / larger** on the dashboard.
   - **Interim points** — `phase = train` — produced by a training session.
     Rendered **lighter / smaller**.
3. **Interim points are a running posterior, not a per-session re-fit.** The
   belief is seeded once from the eval cloud and carried forward across every
   subsequent session; each session reweights/propagates that belief over the
   session's items. A point is the posterior summary at session end. We never
   re-estimate a domain from a single session's items in isolation.
4. **Min-evidence floor.** A session emits a point for `domainN` only if the
   session contributed at least `K_MIN` scored items to `domainN`. Domains the
   session didn't meaningfully touch do not advance (no point). `K_MIN` is an
   open value (see §7), provisionally `K_MIN = 6`.

Rationale for "per session, floored" over daily/weekly: a point should rest on
*new evidence*, and the session is the natural unit of evidence here. Daily/
weekly rollups either invent points on days with no signal or smear sessions of
very different size into one marker. Point density should track real activity.

---

## 2. The point payload

Each emitted point, per domain, matches the deployed `param_trajectories`
schema (one row per point):

| field | source | notes |
|---|---|---|
| `task_k` | domain index `N − 1` | |
| `phase` | `eval` \| `train` \| `recert` | drives dark/light rendering |
| `ell` | posterior mean of ℓ at session end | ENGINE coords |
| `theta` | posterior mean of θ at session end | ENGINE coords |
| `sd` | posterior SD of ℓ at session end | the ±σ band on the ℓ chart |
| `rt` | **median** reaction time over this session's `domainN` items (ms) | matches the eval-point RT convention; `null` if no timing |
| `ts` | session finalize timestamp (UTC) | x-axis ordering |
| `training_id` | the session id | `null` for cert anchors |
| `source_session_id` | seeding cert session id | provenance back to the eval |
| `seq_in_session` | monotonic order | |
| `is_real` | `1` | the dashboard only ever shows `is_real=1` |

Coordinate convention: store ENGINE coords `(θ, ℓ)`. The trainer's internal
`(σ, t)` plan coords convert via `bridge_conventions.engine_to_plan` /
`engine_to_plan`-inverse only — never inline (the σ = exp(−ℓ), t = −θ sign-flip
bridge is the single source of truth). `sd` is the SD of ℓ specifically (it is
the band the dashboard draws around the ℓ curve).

---

## 3. The running posterior (the estimator)

This is already the design of the trainer filter; the cadence layer just reads
it at session boundaries.

- **Seed once.** `training_seed.build_seed_from_state` turns the eval SMC cloud
  into a per-domain `TrainingSeed` (variance-inflated, D1; per-domain marginal,
  D2). This is the prior for the *first* training session.
- **Carry forward.** `training_filter.TaskFilter` (one per domain) holds the
  belief. Per trial: **reweight** (lapse-mixture likelihood with s_sd
  attenuation) → **propagate** (transition kernel `T`, identical to
  `learner_sim.Learner.step`) → **resample** (multinomial when ESS < frac·N).
  The filter state at the end of session *s* is the prior for session *s+1* —
  recorded in the append-only `LearnerLedger` (D9 / LT1).
- **Between-session decay.** The gap between sessions is applied to the
  carried-forward prior via the LT2 hook `TaskFilter.propagate_gap(dt_seconds)`
  (identity + optional variance widening today; the fitted forgetting kernel
  drops in here later without touching the per-trial loop).
- **Emit.** At session finalize, summarize each touched domain's filter:
  `ell = E[ℓ]`, `theta = E[θ]`, `sd = SD[ℓ]`. These are posterior summaries of
  a belief that began as the eval cloud and has been updated by every session
  since — exactly "carry the prior forward, update with the new items."

What an interim point is **not**: a fit to that session's items alone. A single
low-volume session must not whipsaw the curve; the carried-forward prior is what
prevents that, and the `K_MIN` floor is the backstop.

---

## 4. Where decay shows up on the curve (important nuance)

Forgetting (LT2) is applied to the **prior of the next session**, not to the
point we plot for the session just finished. So:

- The plotted `train` point = the **post-session posterior** (what the trainee
  actually demonstrated by the end of that sitting).
- The decay dip lives between points: the next session *starts* from a widened/
  shifted prior, so if they return after a long gap and a domain has decayed,
  the *next* point reflects that lower starting belief being re-raised.

We do **not** emit a separate "session-start, post-decay" point. One point per
session, post-update. This keeps the curve readable (one marker per sitting) and
the semantics clean (a point = "where you were when you finished training").

---

## 5. Cert vs training rendering (deployed dashboard)

The dashboard already groups points per domain and draws the ℓ/θ/RT mini-charts
(`charts/MiniChart`). The only addition needed is **phase-aware marker styling**:

- `phase ∈ {eval, recert}` → filled, darker, larger dot (anchor).
- `phase = train` → lighter, smaller dot.
- The dashed ℓ\* cut, the θ=0 reference, and the eval→training→re-cert phase axis
  are unchanged.

Contract requirement: the trainer **must** tag every emitted point with the
correct `phase` so the dashboard can style without guessing. Anchors come from
the certification path (already `eval`; re-tests `recert`); the trainer emits
`train`.

(The deployed `MiniChart` currently distinguishes only the last point; the
phase-color pass is a small follow-up there, ready to land when the first
`train` points exist.)

---

## 6. Integration path (trainer → deployed app)

The deployed app already has the sink and the read path:

- `POST /api/trajectories` accepts a batch of points
  `{taskK, phase, ell, theta, sd, rt, trainingId, sourceSessionId, seqInSession, isReal}`.
- `GET /api/trajectories` returns only `is_real = 1` rows, grouped per domain.
- A finished **training session** should, at finalize, post one point per
  touched domain with `phase = "train"`, `isReal = true`. (Mirror of how a
  finished **cert test** already writes its `eval` anchor point server-side.)

When `pipeline_demo` (or its ported equivalent) runs inside the deployment
runtime, the per-session emission hook is the single new call: summarize each
`TaskFilter`, apply the `K_MIN` floor, build the point batch, post it. Nothing
about the dashboard read path or schema changes.

Suggested seam in this repo: a small `emit_session_points(filters, session_meta)
-> list[Point]` helper (next to `training_seed`) that the trainer calls at
session finalize; `pipeline_demo` already walks sessions and holds the filters,
so it's the natural first caller and test harness.

---

## 7. Open questions / to calibrate on pilot data

1. **`K_MIN` value.** Provisional 6. Calibrate so a floored session's posterior
   SD is stable enough that the point isn't visually noisy. Likely 5–8.
2. **Forgetting kernel.** Identity today. Once the Ebbinghaus/LT2 kernel is
   fitted, confirm the "decay between points, not on the plotted point"
   semantics (§4) still reads well — especially the first point after a long
   layoff.
3. **Very active users.** If someone runs many short sessions in a day, the
   curve could get dense. Option: cap to N points/domain/day by keeping the
   last finalized session of the day, or merge same-day sessions before
   emitting. Decide only if density becomes a real problem.
4. **RT robustness.** Median per session is the convention. Confirm sessions
   with very few timed items don't produce a jumpy RT curve; the `K_MIN` floor
   should mostly cover this.
5. **Graduation interaction.** A domain that graduates (F14/AD6 mastery on the
   filtered posterior) — keep emitting points after graduation, or stop? Lean
   toward continuing (the curve should keep showing maintenance), but mark the
   graduation point.

---

## 8. Status

- **Deployed today:** the eval anchor point. Finishing a certification test
  writes one real (`is_real=1`, `phase=eval`) point per domain — `ell`/`theta`/
  `sd` from the test posterior + median RT from the session's trials — and the
  dashboard plots that single operating point per domain with room for the
  series to fill in.
- **This plan:** the `phase=train` per-session cadence, gated on `K_MIN`, from
  the carried-forward `TaskFilter`. Lands when the trainer is ported into the
  deployment runtime (the `learner_sim`-validated filter is the same module the
  deployed app would call).
