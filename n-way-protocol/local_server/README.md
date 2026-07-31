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
  IIC/Other).
- `POST /session/:id/autostep` — `{n, reader?}`; simulates `n` answers. On a
  manual session, passing `reader` attaches a simulated reader (hand-off).
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

Boots the server on port 8735, runs a simulated sharp-expert session to
completion through the HTTP API, and asserts it stops via the policy
(`all_estimated_or_undeterminable`) with sane question counts (120..360),
terminal per-domain statuses, and a normalized atom posterior.
