# How the Adaptive Certification Test Works

> A technical walkthrough of the test-taker engine — the model,
> the Bayesian inference, the adaptive item selection, and the pass/fail/refer
> decision rule — with pointers to the actual code.
>
> Audience: a strong technical reader (engineer / statistician) who wants to
> understand the algorithm well enough to reason about or modify it. Every
> claim is cited to `file:line`. **Code wins over prose** — if a citation has
> drifted, trust the code and fix this doc.
>
> Scope: this describes the *production* path that the desktop app actually
> runs. The methodology repo's Mode-A AUROC-halfwidth driver
> (`run_session_mcmc_auroc`) and the offline Laplace/EKF deployment engine are
> **not** the live test and are called out only where it avoids confusion.

---

## 0. The 60-second version

A candidate sits a short adaptive test. For each of tasks/domains the test
estimates two latent traits — the candidate's **skill** and **bias** — from
their yes/no answers, using a **Bayesian particle filter** (Sequential Monte
Carlo). After every answer it:

1. **re-estimates** the joint posterior over all N+M traits (N skills + M biases) -> should be N=M;
2. **chooses the next question** to be maximally informative (the item that most
   reduces posterior uncertainty);
3. **checks a per-task decision rule (AD6)** — does the posterior now place ≥95%
   probability that the candidate's skill is above (PASS) or below (FAIL) a
   pre-calibrated cut-score? If neither, keep going.

The session ends when every task is resolved, the question cap is hit, or the
item bank is exhausted. Unresolved tasks become **REFER**.

---

## 1. The measurement model (Signal Detection Theory + lapse)

Each task is modeled as a one-dimensional **signal-detection** problem. A test
segment carries a scalar **signal strength** `s` (how strongly the target
pattern is present, on a probit scale; negative = absent/weak). The candidate
has, for that task `k`:

- a **bias** `θ_k` (their internal decision criterion / propensity to say "yes"), and
- a **log-skill** `ℓ_k`, where the internal noise SD is `σ_k = exp(−ℓ_k)` and
  **skill `= 1/σ_k = exp(ℓ_k)`**. Higher `ℓ_k` ⇒ less noise ⇒ sharper discrimination.

The probability the candidate answers "yes" to a segment of signal `s` is the
**probit lapse mixture** (`engine/core_mcmc.py:87-113`):

```
z          = exp(ℓ_k) · (s + θ_k)
P(y=1 | z) = λ + (1 − 2λ) · Φ(z)          # λ = 0.025, fixed everywhere
P(y=0 | z) = λ + (1 − 2λ) · Φ(−z)
```

- `Φ` is the standard normal CDF.
- `λ = 0.025` is the **lapse rate** — an irreducible floor/ceiling so that no
  response is ever assigned probability 0 or 1 (a confident, correct candidate
  still slips 2.5% of the time). It is a **🔒 locked invariant** (`LAPSE_RATE`); identical value in the calibration fitter and the
  deployment engine — one likelihood definition across the whole system).
- **Numerical stability:** the log-likelihood uses `scipy.special.log_ndtr`
  (stable `log Φ`) + `logsumexp` to mix in the `log λ` floor
  (`core_mcmc.py:99-104`). The system **never** uses clipped `norm.cdf`, which
  collapses to 1.0 at |z|≳6 and would bias confident candidates' posteriors
  (FIX-T0.7; see `CLAUDE.md` "what NOT to do").

### 1.1 Skill ↔ AUROC

Skill is reported to candidates as an **AUROC** (area under the ROC), a
monotone function of `ℓ` only — bias does **not** affect discrimination:

```
AUROC(ℓ) = Φ( √2 / √(exp(−2ℓ) + 1) )
```

Worth internalizing: `ℓ = 0` (skill = 1, σ = 1) already gives **AUROC ≈ 0.841**,
not chance. AUROC → 0.5 only as `ℓ → −∞`. The shipped cut-scores `ℓ*` (§6)
sit around `ℓ* ≈ 0.25–0.53`, i.e. **AUROC bars of roughly 0.87–0.89**. Note: we may investigate this aspect further. Run thorough mathematical calibration test to determine how we can more thorough fix these numerical alignments.

---

## 2. The Bayesian setup — what is estimated and the prior

The unknown is the full **N+M-vector** `(θ_1..θ_M, ℓ_1..ℓ_N)`. The engine carries
the *joint* posterior over all of it, so evidence on one task informs
correlated tasks ("borrowing strength").

**Prior** : a zero-mean multivariate normal on the `θ`-block and the `ℓ`-block separately, with fitted
**correlation** matrices. Critically, the live engine passes the
**unit-diagonal correlation** `Corr_l` (not the raw fitted covariance) for
*both* blocks:

```python
state = make_state_hier(self.N, self.K, R_ASSUMED, rng,
                        Sigma_l=inp.Corr_l, Sigma_t=inp.Corr_l)
```

Why the correlation, not the covariance? (FIX-T1.7) The
raw fitted `Σ_ℓ` has tiny marginal SDs (~0.10–0.24) reflecting *inter-rater
spread in the calibration cohort*, not the plausible range of an examinee's
skill. Using it as a prior would pin `ℓ*` several prior-SDs from zero and make
FAIL fire after one question. Using the unit-diagonal correlation gives each
trait a **N(0, 1) marginal prior** with the empirically-fitted *cross-task
correlation structure* (e.g. domain1↔domain2 ≈ 0.65; one mildly negative pair).

Consequences worth noting:
- **`Var_prior(ℓ_k) = 1.0`** for every task (unit diagonal). This is the
  denominator of the AD6 information gate (§5).
- The prior pass-mass is `P(ℓ_k > ℓ*_k) = Φ(−ℓ*_k) ≈ 0.30–0.40` — i.e. before
  any data the engine already thinks a random candidate has a ~⅓ chance of
  being above the bar. Resolving a verdict means moving that mass to ≥0.95 or ≤0.05.
- The bias block currently **reuses the skill correlation** (`Sigma_t=inp.Corr_l`).
  That is a modeling shortcut, not a fitted bias-correlation — flagged as a
  candidate improvement.

---

## 3. Inference: the SMC particle cloud + MH rejuvenation

The posterior is represented by **N = 600 weighted particles**, each a full 14-vector with a weight `w_i`.
State is a dict with arrays `t` (N×M), `l` (N×N), `w` (N), plus cached
`log_prior`, `log_lik`, and the question `history`.

**(a) Initialize** — sample N particles from the prior.

**(b) Reweight on each answer**: multiply
each particle's weight by the likelihood of the observed `(k, s, y)` under that
particle, renormalize, and append to history:

```python
w_i  ←  w_i · P(y | z_i),   z_i = exp(ℓ_{i,k})·(s + θ_{i,k})
```

**(c) Effective sample size & degeneracy**:
`ESS = 1 / Σ w_i²`. As answers accumulate, weight concentrates on a few
particles and ESS falls.

**(d) Resample + MCMC rejuvenate** when `ESS < ESS_THRESHOLD_FRAC · N`
(= 0.5 · 600 = 300):
- **Multinomial resample** the cloud by weight (kills low-weight particles,
  duplicates high-weight ones), reset weights to uniform.
- **Rejuvenate** with `N_MH_STEPS = 15` Metropolis-Hastings steps
  (`mh_rejuvenate`). The proposal is a Gaussian random
  walk with covariance `proposal_scale² · cov(cloud)`, where
  `proposal_scale = 2.38/√(2K) ≈ 0.636` — the Roberts-Gelman-Gilks optimal
  scaling for a Gaussian target. Acceptance uses
  the cached `log_prior + log_lik`, with the proposed-position likelihood
  recomputed by sweeping the full question history (`_log_lik_history`).

This combination (resample to fight degeneracy, MCMC moves to restore diversity)
is what keeps the cloud from collapsing onto a few duplicated points while
*exactly* targeting the current posterior — no Gaussian-blob drift.

> **Reproducibility:** bit-exact results require single-thread BLAS
> (`OPENBLAS/MKL/... = 1`) set before numpy import — done in `conftest.py` and
> the entry points. The MCMC is deterministic given the seed.

---

## 4. Signal-uncertainty marginalization (the `s_sd` term)

The segment signal `s` is itself an *estimate* (a posterior mean from the
calibration), with its own SD `s_sd`. The engine marginalizes the likelihood
over `s ~ N(s, s_sd²)` in closed form by **attenuating `z`**:

```
z  ←  z / √( 1 + (exp(ℓ_k) · s_sd)² )
```

A noisier segment (larger `s_sd`) shrinks `z` toward 0, pulling `P(yes)` toward
the lapse-symmetric 0.5 — i.e. an uncertain item carries less evidence. This
also feeds item selection (§5), so the selector *prefers high-precision
(low-`s_sd`) items*. When `s_sd = 0` the math is bit-identical to the
no-uncertainty engine (Phase 3.5; the feature is additive and drift-guarded).

`s_mean`/`s_sd` per (segment, task) come from `data/labels/segment_signals.csv`
and are surfaced to the engine via `as_engine_arrays()`.

---

## 5. Adaptive item selection (Global Expected-Variance / A-optimal)

After each answer the engine picks the next `(task k, segment s)` that
**minimizes the expected total posterior variance** over all N+M traits — the
Bayesian A-optimal design (`choose_item` + `_expected_loss_vec`).

For a candidate item on task `k`, using the *current* cloud:

1. Predict `P(yes)` by averaging each particle's `P(y=1|z_i)` over the weights.
2. Form the two **hypothetical posteriors** — reweight the cloud as if the
   answer were "yes", and as if "no".
3. Compute the total trait variance under each, and combine:
   ```
   ExpectedLoss(item) = P(yes)·Var_total(after "yes") + P(no)·Var_total(after "no")
   Var_total = Σ_k Var(θ_k) + Σ_k Var(ℓ_k)            # all N+M traits
   ```
4. Pick the item (across all *active* tasks and all candidate segments) with the
   smallest `ExpectedLoss` (`choose_item`, `core_mcmc.py:610-634`).

Because the posterior is *joint*, asking task `k` also shrinks variance in
correlated tasks — that is the pooling benefit. A deterministic
**coarse-to-fine argmin** (`_coarse_to_fine_argmin`)
makes the scan over large per-task banks fast while remaining bitwise identical
to the exhaustive argmin (it refines the top-M coarse cells).

> **Design note for modifiers:** the verdict (§5–6) depends *only* on the tail
> probability `π_k = P(ℓ_k > ℓ*_k)`. The A-optimal objective above also spends
> the question budget shrinking the M *bias* variances and the skill variances
> of tasks already far from their cut-score — variance that does not move any
> verdict. Re-aligning this objective with the decision is the single biggest
> efficiency lever. REMEMBER THIS!

**First-question randomization** (`_pick_top_n`):
trial 0 is drawn uniformly from the `FIRST_ITEM_TOPN = 10` most-informative
items so every candidate doesn't start with the identical question.

---

## 6. The decision rule: AD6 (`policy.py`)

This is the heart of certification — the rule that turns a posterior into
PASS / FAIL / REFER. It is the **only** place stopping/verdict logic lives; the
session treats it as an injectable `TerminationPolicy`.
The K=7 wiring is based on a set 7 tasks/domains (which can be generalizeable to X tasks), which builds an
`AD6Policy` with the 7 cut-scores and the prior variances.

After **every** trial, for each task `k` (`AD6Policy.__call__`,
`policy.py:267-314`):

```
π_k    = Σ_i w_i · 1[ ℓ_k^(i) > ℓ*_k ]              # posterior pass-mass
mcse_k = √( π_k (1 − π_k) / ESS )                    # Monte-Carlo error of π_k
R_k    = 1 − Var_post(ℓ_k) / Var_prior(ℓ_k)          # information / sufficiency gate
```

Task `k` becomes **RESOLVED** iff it has enough data **and** enough information
**and** the posterior is decisively on one side of the cut:

```
require:  n_k ≥ N_MIN   AND   R_k ≥ R*
PASS if:  π_k − Z·mcse_k ≥ 1 − α
FAIL if:  π_k + Z·mcse_k ≤ α
```

with the shipped constants (`policy.py:113-128`):

| Symbol | Value | Meaning |
|---|---|---|
| `N_MIN` | **20** | min answers in a task before any verdict |
| `R*` | **0.30** | info gate: posterior must explain ≥30% of prior variance (SD contracts ≥~16%) |
| `α` | **0.05** | error budget — 95% posterior confidence to PASS/FAIL |
| `Z` | **2.0** | MC-error buffer (≈95% envelope around `π_k`) against finite-particle noise |

Key behaviors:

- **The info gate `R_k`** prevents certifying a task whose posterior is still
  prior-dominated (the "one-vs-rest degeneracy" — e.g. a task barely asked).
  Since `Var_prior = 1`, `R_k = 1 − Var_post(ℓ_k)`.
- **The `Z·mcse` buffer** widens the bar to guard against the fact that `π_k` is
  estimated from 600 particles. Near the boundary `mcse ≈ 0.009–0.018`, so the
  *effective* PASS bar is `π_k ≳ 0.968` rather than the nominal 0.95. (This is
  why wide posteriors — e.g. a real test taker's — struggle to PASS;.)
- **Monotonic verdict lock** (`policy.py:290-298`): once a task is PASS or
  FAIL it never changes, even if later answers would move it. This is a
  deliberate adaptive-design choice (CAT literature would reconcile at end of
  test; the trade-off was considered and the lock kept).
- **Session stops** when all 7 tasks are RESOLVED (`stop_reason="all_resolved"`,
  `policy.py:309-312`).

**End-of-session finalization** (`finalize_verdicts`, `policy.py:316-328`):
each still-PENDING task becomes
- `REFER_BORDERLINE` if its gate had opened (`R_k ≥ R*`) but the posterior never
  crossed — "we measured you, you're near the line", or
- `REFER_UNINFORMATIVE` if the gate never opened — "we couldn't measure this
  task" (the genuinely uninformative case, e.g. an exhausted/degenerate bank).

The cut-scores `ℓ*_k` are the **Youden CV-top-14** values from
`ell_star_unified_v14` (loaded by
`load_ell_star_k7`, `policy_k7.py:60-98`): domain 1 0.325, domain 2 0.256,
domain 3 0.534, domain 4 0.330, domain 5 0.479, domain 6 0.486, domain 7 0.442 (AUROC bars ≈ 0.87–0.89).
Note: `ell_thresholds.csv` holds *different, legacy*
values and is **not** the live source.

---

## 7. Session orchestration (`scripts/session_controller.py`)

session_controller is the main loop. Per
trial it: selects an item → calls `y_source(k, seg_id, s)` → `update`s the cloud →
rejuvenates if ESS is low → recomputes telemetry → calls the policy → stops if
the policy says so.

Several wrinkles a modifier must know:

- **Active-domain filtering**: resolved tasks and empty banks are dropped from
  selection; a `MAX_CONSEC_SAME_DOMAIN` variety cap (default off) can force a
  domain switch after a long single-task run (`:281-298`).

- **Stopping conditions**: (1) all tasks resolved; (2)
  `MAX_QUESTIONS_DEFAULT = 500` cap (honored as `min(cap, bank_size)`,
  `:75-79`); (3) bank exhausted / no selectable active domain.

- **Extended data-collection mode** (`EXTENDED_DATA_COLLECTION_DEFAULT = True`): the internal build keeps asking
  questions *past* the point AD6 would have stopped, to gather calibration data
  across all tasks. The **official result is byte-frozen at the
  would-have-stopped snapshot** — the certificate, verdicts, question count, and
  visualizations are identical to a non-extended run; the extra trials are
  recorded with a `post_decision=True` flag and excluded from the official
  stats. (Set the constant False to restore plain early-stop for a public build.)

---

## 8. Where the data comes from (`engine_inputs_k7.py`)

`build_k7_engine_inputs()` (`engine_inputs_k7.py:187-301`) assembles
everything the engine needs and validates it hard (raises on any defect):

- **Item bank** — strict family→task mapping. Each segment is
  served **at most once** per session.
- **Per-(segment, task) signals** — columns
  `s_mean_{code}` / `s_sd_{code}`; NaN where a segment isn't a valid candidate
  for that task.
- **Prior** — Provides the 7×7 (Generalizable N×M `Corr_l` (skill) and
  `Corr_t` (bias) correlation matrices, with the domain order asserted to equal
  the canonical.

The per-task banks are returned by `as_engine_arrays()` as 7 (generablizable to N) arrays of possibly
different lengths, parallel `bank_sds` and `bank_segids` exactly the shape `choose_item` consumes.

---

## 9. One trial, end to end (worked trace)

1. **Select.** `_select` builds the per-task candidate arrays for the remaining
   pool, computes active domains , and calls `choose_item`,
   which returns the A-optimal `(k, s, s_sd, seg_id)`
2. **Ask.** The viewer shows segment `seg_id`; the candidate answers; the
   GUI→engine queue yields `y`.
3. **Update.** `update(state, k, s, y, s_sd)` reweights the 600 particles by the
   lapse-mixture likelihood (with `s_sd` attenuation) and appends to history
4. **Rejuvenate if needed.** If `ESS < 300`, multinomial-resample + 15 MH steps
5. **Summarize.** Recompute per-task posterior means/variances and AUROC
   posterior for telemetry.
6. **Decide.** `policy(state, tel, n_per_task, K)` computes `π_k, mcse_k, R_k`
   for all tasks, updates the monotonic verdict locks, and returns whether to
   stop.
7. **Stop or continue.** If all resolved → stop (or, in extended mode, freeze
   the snapshot and keep collecting). Otherwise loop.

At session end, `finalize_verdicts()` converts remaining PENDINGs to the two
REFER flavors, and a `SessionResult` carries the frozen verdicts, question
count, per-task AUROC means + halfwidths, posterior means, and the full trial
telemetry.

---

## 10. Config cheat-sheet (live values)

| Constant | Value | Where |
|---|---|---|
| `N_PARTICLES` | 600 |
| `ESS_THRESHOLD_FRAC` | 0.5 |
| `N_MH_STEPS` | 15 |
| `proposal_scale` | `2.38/√(2K) ≈ 0.636` |
| `FIRST_ITEM_TOPN` | 10 |
| `MAX_QUESTIONS_DEFAULT` | 500 |
| `LAPSE_RATE λ` | 0.025 🔒 |
| `N_MIN` | 20 |
| `R*` | 0.30 |
| `α` | 0.05 |
| `Z` | 2.0 |
| `ℓ*` (v14) | domain 1 .325 / domain 2 .256 / domain 3 .534 / domain 4 .330 / domain 5 .479 / domain 6 .486 / domain 7 .442 |

---

## 11. Known limitations (and what's being worked on)

- **Item selection is decision-agnostic.** A-optimal minimizes *total* trait
  variance (incl. the 7 nuisance biases), not the certification quantity
  `π_k = P(ℓ_k>ℓ*_k)`. This wastes questions ⇒ slower resolution. (Plan: a
  decision-aligned objective.)
- **`π_k` is a noisy hard-threshold MC estimate.** Its MC noise is exactly what
  the `Z·mcse` buffer absorbs, which tightens the effective PASS bar from 0.95
  to ~0.968. (Plan: a Rao-Blackwellized/smoothed `π_k` + possibly more
  particles ⇒ recover nominal α.)
- **Calibration, not engine:** the uniform ~AUROC-0.88 `ℓ*` bar may exceed
  achievable expert performance on ambiguous tasks, and
  requiring all-7 (or generalizable to N) clean PASS is a near-impossible product of marginals. No
  inference change fixes a mis-set cut-score — these are PI/calibration
  decisions (per-pattern `ℓ*`, tiered certification, human-vs-sim α).
```
