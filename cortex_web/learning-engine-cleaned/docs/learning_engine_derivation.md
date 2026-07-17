# The Learning Engine: a constant-free, population-learned protocol

*Mathematical derivation for the abstractable learning engine (2026-07-11).
Decisions D21+ in [DECISION_LOG.md](../DECISION_LOG.md); implementation gates
G1–G6 at the end. Companion to
[learning_with_feedback_design_note.tex](learning_with_feedback_design_note.tex)
and the D8–D20 methodology round.*

---

## 0. Inputs, interfaces, and the constants audit

The engine is a module between a **testing algorithm** (sequential Bayesian,
not designed here) and a **certification exam**. A field plugs in by
supplying exactly four things, none of which is a tuning constant — each is an
operational fact about that field:

| input | symbol | meaning |
|---|---|---|
| item bank | $\{(s_j, \mu_j)\}$ | expert-labeled items on a signed difficulty axis $s$ (sub-type tags $\mu$ reserved for the multi-dimensional extension) |
| exam spec | $\mathcal{E} = (N_e, c_e, \mathcal{D}_e)$ | number of exam items, passing count, exam case-mix distribution |
| risk tolerance | $\eta$ | max acceptable probability that a flagged-ready learner fails |
| data repository | $\mathcal{R}$ | historical response streams $(s_k, y_k, r_k, \mathrm{fb}_k)$ from the population |

The single blessed constant is the design lapse $\lambda = 0.025$ (still
*estimated* per learner in inference). Everything else below — every shape,
rate, noise scale, calibration factor, and placement target — is learned from
$\mathcal{R}$ or derived from $\mathcal{E}$.

**Handoff from the testing algorithm.** The test hands the engine its full
posterior $b_0(x_0)$ over the initial state $x_0 = (t_0, \log\sigma_0)$ *and*
its raw response stream. The posterior is a sufficient summary under the
shared observation model; the raw stream is retained because the engine's
filter can re-consume it losslessly (answer *patterns* — streaks, error runs,
response times — carry information the point summary discards, and the
learned-dynamics layer can use them). In simulation the testing algorithm is
mimicked by the measurement-only pre-test already built
(`policy.run_pretest`), which produces exactly this object.

## 1. The generative model: a law of learning with learned shapes

### 1.1 State and observation channels

Per learner $i$, latent state $x_k = (t_k, u_k)$ with $u = \log\sigma$,
evolving per trial $k$. Two observation channels per trial:

**Correctness** (as before):
$$y_k \sim \mathrm{Bernoulli}(p_k), \qquad
  p_k = \lambda_i + (1 - 2\lambda_i)\,\Phi(z_k),
  \qquad z_k = \tfrac{s_k - t_k}{\sigma_k}.$$

**Response time** (new — the chronometric channel):
$$\log r_k \sim \mathcal{N}\!\big(\psi_{0,i} - \psi_1\,|z_k| + \psi_2\,\ell_k,\; \tau_R^2\big),$$
with $\ell_k$ the session load (time-on-task, $k/K$), $\psi_{0,i}$ a
per-learner base speed (hierarchical), and $(\psi_1, \psi_2, \tau_R)$
population-level. The chronometric law encodes two standard empirical facts —
easier decisions are faster ($\psi_1 > 0$), fatigue slows responses
($\psi_2 > 0$) — but their *magnitudes* are learned, not assumed. RT is where
fatigue and insight appear before accuracy moves, which is what makes the
realtime layer (§4) fast. Fields without RT drop the channel; every formula
below degrades gracefully to binary-only.

*Abstractability note:* RT is one instance of a per-trial auxiliary channel
with a smooth state-dependent likelihood $h(\text{aux}_k \mid x_k, s_k)$; any
such channel (confidence, edit count) slots into the same filter update.

### 1.2 Dynamics: the learned law

$$u_{k+1} = u_k - \alpha_{\sigma,i}\, W(z_k;\, c)\,\big(u_k - u_{\infty,i}\big)
            \;+\; r_{f,i}\,\tfrac{k}{K} \;+\; \xi_u,
  \qquad \xi_u \sim \mathcal{N}(0, q_u^2),$$
$$t_{k+1} = t_k + \alpha_{t,i}\, G(\delta_k;\, d) + \xi_t,
  \qquad \delta_k = \hat y_k - y_k^\star,\quad \xi_t \sim \mathcal{N}(0, q_t^2).$$

The two **shape functions are population-learned**, replacing the postulated
Wilson gate and Rescorla–Wagner link:

$$W(z; c) = \sum_{j=1}^{J} c_j\, B_j(|z|), \qquad c \in \Delta^{J-1}
  \;(\text{simplex}),$$
with $B_j$ a fixed nonnegative smooth basis (normalised Gaussian bumps on a
$z$-grid spanning the design's support). The simplex constraint
$\sum_j c_j = 1$ **fixes the scale of $W$ by construction**, so the rate
$\alpha_\sigma$ carries all magnitude — this kills the $\alpha$–shape ridge
(the old $\alpha_\sigma\rho$ pathology, D3) structurally rather than by
hoping. The prior on $c$ is a Dirichlet centred on the projection of the
Wilson shape $z\phi(z)/\phi(1)$ onto the basis: **the derivation is demoted
from axiom to prior mean**, and the data may confirm or overrule it.

$$G(\delta; d) = \mathrm{sign}(\delta) \sum_{j=1}^{J_g} d_j M_j(|\delta|),
  \qquad d \in \Delta^{J_g-1},$$
with $M_j$ a monotone basis on $[0,1]$ ($M_j(0)=0$, increasing — e.g.
$|\delta|^{a_j}$ for a grid of exponents $a_j$). $G$ is odd and monotone by
construction (feedback never pushes the criterion away from the truth — the
one structural assumption we keep, and it is falsifiable at cohort level via
the pooled likelihood-ratio machinery of D1). This family nests
Rescorla–Wagner ($G = \delta$), the cross-entropy-gradient behaviour
(concave $|G|$: near-threshold errors dominate), and the hard-form
expectation — so the tier-1/tier-2 bias-placement tension (D11) becomes an
*estimable question* instead of a modelling choice.

**Hierarchy.** Per-learner: $t_0, \sigma_0, \sigma_\infty, \alpha_t,
\alpha_\sigma, \lambda, r_f, \psi_0$ — all non-centred hierarchical with
population means and spreads sampled (empirical Bayes by construction).
Cohort-level: $c, d, q_t, q_u, \psi_1, \psi_2, \tau_R$. This split is not
taste — it is the identifiability boundary measured in D5/D15: individual
flexibility is unrecoverable from ~10³ binary trials; population structure is
what protects every signal we care about (the hier-ADF result: only
hierarchical shrinkage on $\theta$ preserves the $q$ information).

### 1.3 What "extracting latent truth from personal noise" means here

The engine's answer to the modern-ML analogy is precise: the latent truth is
the state path $x_{0:K}$ and the population law $(W, G, \dots)$; the personal
noise is (i) Bernoulli response noise, (ii) the lapse channel, (iii) process
noise $\xi$, (iv) RT residuals. Three mechanisms separate them:
1. **Hierarchical shrinkage** pools thousands of noisy individual streams
   into sharp population posteriors (the only estimator class that survived
   D15's absorption analysis);
2. **Marginalisation** — the assumed-density filter integrates the noise path
   out of every likelihood (deterministic, exact at $q=0$), so noise is
   *accounted for*, never fitted;
3. **Multi-channel evidence** — RT breaks degeneracies binary data cannot
   (a slow-correct trial and a fast-correct trial update the fatigue and
   skill posteriors differently).

## 2. Estimation: the population learner (module M1)

The two-channel ADF update is the old one with the RT factor inside the same
quadrature. With Gauss–Hermite points $x^{(q)}$ and weights $w^{(q)}$
representing the belief $\mathcal{N}(m_k, P_k)$:

$$Z_k = \sum_q w^{(q)}\;
   \underbrace{\mathrm{Bern}\big(y_k \mid p(x^{(q)})\big)}_{\text{correctness}}\;
   \underbrace{\mathcal{N}\big(\log r_k \mid m_R(x^{(q)}), \tau_R^2\big)}_{\text{RT}},
 \qquad \log p(\text{data}) = \sum_k \log Z_k,$$

posterior moments by the reweighted points, prediction through the learned
drift exactly as in `adfilter.py`/`jaxmodel.adf_loglik_one`. M1 is then a
single hierarchical NUTS fit with this marginal likelihood inside — the
architecture validated in D15 (8 latent dims per learner), now with shape
weights $(c, d)$ as additional cohort-level parameters.

**Design-support caveat (inherited from D3):** $W$ is identified only where
the historical design has difficulty variation — an adaptively-collected
repository concentrated at one $z$ cannot reveal the gate's shape (the
likelihood is flat in $c$ there, exactly the old $\rho$ flatness). M1 must
therefore report a *support diagnostic* (the design's $z$-occupancy against
the basis grid), and deployment must preserve future identifiability, which
is what exploration (§5) is for.

## 3. Control and stopping: margin to the exam

### 3.1 Readiness, derived from the exam spec

For a learner in state $x$, exam items drawn i.i.d. $s \sim \mathcal{D}_e$
give per-item accuracy
$$\bar a(x) = \mathbb{E}_{s\sim\mathcal{D}_e}\big[P(\text{correct}\mid s, x)\big]$$
— which is exactly the deployment utility $U(x)$ with $\mathcal{D} =
\mathcal{D}_e$ (the D10 object). The exam score is then
$\mathrm{Bin}(N_e, \bar a(x))$, so the pass probability is exact:
$$P_{\text{pass}}(x) = \Pr\big[\mathrm{Bin}(N_e, \bar a(x)) \ge c_e\big]
  = I_{\bar a(x)}(c_e,\, N_e - c_e + 1)$$
(regularised incomplete beta). Readiness is a statement about the *belief*,
not the point estimate:
$$\boxed{\;\text{READY} \iff \pi(b_k) \equiv \mathbb{E}_{x\sim b_k}\big[P_{\text{pass}}(x)\big] \;\ge\; 1 - \eta.\;}$$
Both teaching and *measurement* raise $\pi$: moving $x$ toward mastery, and
sharpening $b$ around a passing $x$, are the same currency. The old
three-constant graduation rule and the separate "keep measuring vs keep
teaching" dilemma disappear into one exam-anchored quantity.

**Honesty condition (the calibration theorem we must verify, G3):** among
learners flagged READY, the realised exam pass rate must be $\ge 1-\eta$.
This holds iff $b_k$ is calibrated — which is why the D16 machinery (bands
widened by the cohort-estimated $q$) is not cosmetic: an over-confident belief
flags learners early and breaks the guarantee. Readiness inherits its honesty
from the calibrated filter.

### 3.2 The running reward: margin, not probability

$\pi$ saturates (flat at 0 far from the bar, flat at 1 past it), so
$\pi$-greedy item selection has vanishing gradients exactly where training
happens. The de Moivre–Laplace margin
$$m(x) = \frac{\bar a(x) - c_e/N_e}{\sqrt{\bar a(x)\,(1-\bar a(x))/N_e}},
 \qquad \pi \approx \Phi(m),$$
is the monotone reparameterisation with no flat regions. The per-trial reward
is the **expected margin gain**
$$R_k = \mathbb{E}_{b}\big[m(x_{k+1}) - m(x_k)\big],$$
and stopping is $\pi(b) \ge 1-\eta$, i.e. $m$-scale threshold
$\Phi^{-1}(1-\eta)$. Because $\bar a = U(\cdot; \mathcal{D}_e)$, this reward
is a variance-normalised sharpening of the validated utility reward — the
same telescoping, the same zero-constants property, now with the exam's own
case-mix and passing bar supplying the units. Minimising questions-to-ready
is then approximated greedily (maximise expected margin gain per question)
with the tier-3 rollout as the planning upgrade when the greedy gap matters;
the equal-β protocol (D20) remains the comparison arm in every experiment.

### 3.3 Placement under the learned law

The skill-mode placement target is no longer a number at all:
$$z^\star = \arg\max_z \widehat{W}(z)$$
under the current population posterior — the engine trains where *its learned
law* says learning is fastest. If the data confirm the Wilson prior,
$z^\star \to 1$; if a field's learners peak elsewhere, the engine follows the
data. The 85% rule becomes a *prediction to test*, closing the noncircularity
loop (§6).

## 4. The realtime layer: fatigue, insight, and pacing

The deployed belief is a filter over the augmented state
$(t_k, u_k, r_{f})$ — the fatigue rate joins the state with its population
posterior as prior. Then:
* **Fatigue** is identified online chiefly through RT ($\psi_2 \ell_k$ rises)
  and secondarily through late accuracy decline; the posterior on $r_f$
  shifts within a session, the forward projection of $u$ bends upward, and
  both placement (via $\hat\sigma$) and the readiness projection respond. No
  detector, no threshold — the state estimate *is* the response.
* **Insight** (a step drop in $u$) is absorbed by the filter's process noise:
  a run of fast-correct answers at surprising difficulty re-weights the
  belief downward in $u$ within a few trials (this is the D6 mechanism that
  kept regret at oracle level under step learners — now faster with RT).
* **Pacing** stays automatic: items ride at $\hat t_k \pm z^\star
  \hat\sigma_k$, so every state revision immediately moves placement.

The D5 information limit is respected, not fought: the engine never claims to
*label* an individual's deviation from binary data alone; it tracks the state
those deviations produce, and the cohort monitor (M3) does the labelling at
the resolution where it is possible.

## 5. Exploration without an exploration constant

A deployed engine that always places at $\arg\max \widehat W$ collects data
that can never improve $\widehat W$ (the D3 flatness, now self-inflicted).
The constant-free resolution is **Thompson sampling at the session level**:
for each session, draw one population-law sample
$(c, d, \text{rates}) \sim$ M1-posterior and run the session optimally under
that draw. Exploration is then exactly proportional to the population
posterior's remaining uncertainty — wide posterior, diverse placements;
converged posterior, near-deterministic policy — with no $\epsilon$, no decay
schedule, and a regret profile that shrinks as the law is learned. M3 feeds
the accumulated sessions back into M1 refits.

## 6. The noncircularity protocol

The claim "we learned how people learn" is defended by four layers, none of
which scores a model on data generated from itself:

1. **Cross-family recovery.** Generate cohorts from laws *outside* the fitted
   family — alternative gates (flat, narrow-peaked, shifted-peak, asymmetric),
   hard-form criterion learners, plateau/step/fatigue/random-walk dynamics —
   and require $\widehat W, \widehat G$ to recover each truth's shape from
   non-adaptive data.
2. **Held-out forward prediction as the only currency.** Model comparison by
   log predictive density on future blocks and unseen learners; in-sample fit
   is never reported.
3. **The emergence test.** The Wilson shape is only a prior. On synthetic
   non-Wilson truths the posterior must move off it (sensitivity); on real
   data, if $\widehat W$ concentrates near the $z\phi(z)$ shape, the 85% rule
   has been *discovered from data* — a new, testable statement of learning
   theory. Symmetrically for $G$: the data decide between R–W and
   gradient-like criterion learning (D11 resolved empirically).
4. **The payoff test.** The learned-law policy must beat the fixed-law policy
   in questions-to-readiness precisely when the truth departs from the prior
   shape, and match it when it doesn't — otherwise learning the law bought
   nothing and the parametric engine stands.

## 7. Modular architecture

```
M1  Population Learner (batch)
    in:  repository R (response streams), design-support diagnostic
    out: population posterior artifact  Θ = {c, d, rates, spreads, q, ψ, τ_R}
M2  Session Engine (online, per learner)
    in:  Θ, testing-algorithm handoff (posterior b0 + raw stream), exam spec E, η
    loop: two-channel ADF/particle belief → margin-greedy item request
          → observe (y, r) → update → READY flag when π(b) ≥ 1−η
    out: item requests; readiness flag; session log (streams back to R)
M3  Cohort Monitor (scheduled)
    in:  accumulated session logs
    out: pooled-LR model checks (D1), fatigue statistics (D14), calibration
         re-estimation (D16), M1 refit triggers, support diagnostics
```
All three consume and produce declared schemas; the testing algorithm and the
exam are external ports. In this round's simulations the testing port is the
simulated pre-test and the exam port is a simulated $(N_e, c_e,
\mathcal{D}_e)$ draw.

## 8. Gated implementation plan

Each gate: reduced-scale iterate → acceptance criteria → full-scale
confirmation where marked (§compute follows the standing reduced/full
policy). Criteria numbers refer to the four standing goals (speed, accuracy,
no-constants, abstractability).

* **G1 — Response-time channel.** Chronometric model in the generator;
  two-channel ADF (numpy + JAX) and particle belief. *Accept:* channel
  parameters recovered hierarchically; online fatigue tracking lag vs
  binary-only measurably reduced; fatigue detection AUC > the 0.85 binary
  ceiling. (criteria 1, 2)
* **G2 — Learned shapes.** Simplex-basis $W$ and $G$ in the hierarchical
  model, Wilson/R–W priors. *Accept:* recovers Wilson truth AND three
  non-Wilson gates from non-adaptive pilots (shape RMSE + held-out lpd);
  recovers $G$ form (soft vs hard-expectation at cohort level); no new
  ridge — $\alpha_\sigma$ recovery no worse than D9's. (2, 3)
* **G3 — Margin reward + readiness.** $\bar a, m, \pi$ from an exam spec;
  margin-greedy tier-1; stopping rule; adaptive-length sessions.
  *Accept:* honesty — flagged-ready learners pass the simulated exam at
  $\ge 1-\eta$; questions-to-ready ≤ fixed-K baselines at equal pass rates;
  equal-β arm (D20) reported alongside. (1, 3, 4)
* **G4 — Learned-law policy + Thompson exploration.** Placement at
  $\arg\max \widehat W$; session-level posterior draws. *Accept:* beats the
  fixed-$z^\star$ policy on questions-to-ready under non-Wilson truths,
  matches it under Wilson truth; exploration preserves identifiability
  (support diagnostic non-degenerate over simulated deployment). (1, 2)
* **G5 — End-to-end modular pipeline v2.** M1→M2→M3 with schemas; full-scale
  run: historical repository → learn law → deploy → readiness flags →
  simulated exams across deviant learner populations. *Accept:* headline
  table (questions-to-ready, pass-rate calibration, robustness) at study
  scale; zero constants outside $\{\lambda, \mathcal{E}, \eta\}$. (all)
* **G6 — Docs, decision log, figures.** D21+ entries with measured results;
  figs 14+; design-note addendum.

## 9. Gate outcomes (all closed 2026-07-11; details in DECISION_LOG D22–D27)

| gate | headline result |
|---|---|
| G1 | RT channel: fatigue detection 45% → **100%** @5% FA (AUC 1.00) in joint mixed-cohort fits; ψ0 corr 0.99; per-cohort fits leave ψ2 unidentified (fit mixed cohorts) |
| G2 | **The design theorem**: gate unidentifiable from static designs (−0.1 nats), identified by banded between-subjects designs (+48.7 nats); link learnable from static data; prior fallback safe |
| G3 | Readiness honest at 90–91% vs the 90% guarantee; questions 250 → 162 (−35%) |
| G4 | Prior right: all engines tie; prior wrong: fixed-law flag dishonest (73%), learned law restores honesty; Thompson exploration free, spread ±0.29 |
| G5 | Self-consistent engine: 100% ready, **median 141 questions, pass-given-ready exactly 90%** across correct/fatigue/random-walk populations |

The theory-shaped contributions this round: (i) the exam-anchored margin as
the single training currency with a provable honesty condition; (ii) the
identification boundary for the law of learning — response-time channels and
hierarchical marginalised filters extend what cohorts can know, but the
difficulty gate is knowable only through designed placement variation, which
the engine's own Thompson exploration supplies.
