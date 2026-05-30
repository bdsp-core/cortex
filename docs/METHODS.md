# Methods: the math behind the adaptive certification engine

This document derives the statistical machinery of the ILAE skill
certification system from first principles and ties every equation to the
line of code that implements it. It is written to be read on its own, but
each section ends with a "where it lives" pointer so a reviewer can check
the claim against the source.

Two ideas carry the whole project. The first is that a clinician's skill
at reading an EEG pattern is a latent quantity we can only observe through
noisy yes or no answers, so we should treat scoring as Bayesian inference,
not arithmetic. The second is that questions are expensive (each one costs
a busy clinician real time), so the test should choose each question to be
maximally informative and stop the moment it has learned enough. Everything
below is in service of those two ideas.

A note on honesty up front. Where the code deliberately departs from a
textbook method, this document says so rather than borrowing the textbook's
authority. The clearest example is the stopping rule: it is a posterior
precision criterion, not a Wald sequential probability ratio test, and the
engine source says as much at `engine/core_mcmc.py:40`. See the closing
section "What this is not."

---

## 1. The response model

### 1.1 One latent skill, one noisy answer

Fix a single task (say, "is there an epileptiform spike?"). A clinician
facing segment $j$ produces a binary label $y \in \{0, 1\}$. We model that
label with a signal detection account: the segment carries a real signal
$s_j$ (how strongly the pattern is actually present), the clinician applies
a personal decision criterion, and their internal sensitivity sets how
reliably signal becomes a correct call.

Write the clinician's latent state as a bias $t$ and a log discrimination
$\ell$. The probit signal that drives their answer is

$$
z \;=\; e^{\ell}\,(s_j + t).
$$

Here $e^{\ell}$ is the discrimination (a sharper reader has larger $\ell$,
so the same signal produces a more decisive $z$), and $t$ shifts the
operating point (a liberal reader who tends to say "yes" has positive $t$).

### 1.2 The lapse mixture

A perfect probit model says a clinician with huge $\ell$ never errs on an
easy segment. Real readers slip: they misclick, they blink, they are
interrupted. We absorb that with a symmetric lapse rate $\lambda$, fixed at
$0.025$ across the entire system. The probability of a "yes" is

$$
P(y = 1 \mid z) \;=\; \lambda + (1 - 2\lambda)\,\Phi(z),
\qquad \lambda = 0.025,
$$

where $\Phi$ is the standard normal CDF. The response is floored at
$\lambda$ and ceilinged at $1 - \lambda$, so no answer is ever treated as
literally impossible. This single definition is shared by the Paper-1
engine, the deployment engine, and the calibration fitters; that the same
likelihood appears in all three is a deliberate invariant, audited in
`docs/INVARIANT_AUDIT.md`.

The lapse rate lives at `engine/core.py:21` (`LAPSE_RATE = 0.025`) and is
mirrored as `LAMBDA = 0.025` in
`pipeline/reference_calibration/fit_sdt_per_domain.py:37`.

### 1.3 Marginalizing the segment signal

The segment signal $s_j$ is not known exactly; calibration delivers it as a
Gaussian posterior $s_j \sim \mathcal{N}(s_j, \sigma_s^2)$. Rather than
sample $s_j$, the engine integrates it out in closed form. Because a probit
of a Gaussian is again a probit with an inflated scale, the marginal "yes"
probability is

$$
P(y = 1) \;=\; \lambda + (1 - 2\lambda)\,
\Phi\!\left( \frac{e^{\ell}\,(s_j + t)}{\sqrt{1 + (e^{\ell}\sigma_s)^2}} \right).
$$

When $\sigma_s \to 0$ the denominator collapses to $1$ and we recover the
point-signal model exactly, which is why turning segment uncertainty on is
backward compatible to the bit. The derivation is written out in the
docstring at `engine/core.py:59`; the runtime form is at
`engine/core_mcmc.py:356`.

### 1.4 Why log space

Confident readers produce $|z| \gtrsim 6$, where $\Phi(z)$ rounds to $1.0$
in double precision and the naive likelihood loses all information in the
tail. The engine never forms $\Phi$ directly when a logarithm will follow.
Instead it computes $\log \Phi$ with `scipy.special.log_ndtr` and mixes in
the lapse floor with `logsumexp`:

$$
\log P(y = 1 \mid z) \;=\;
\operatorname{logsumexp}\!\big( \log(1 - 2\lambda) + \log \Phi(z),\;\; \log \lambda \big).
$$

This is the fix recorded as FIX-T0.7; the rationale (a clipped
`norm.cdf` "collapses to 1.0 at |z| >= 6 and biases the posteriors of
confident raters") is in the comment at `engine/core.py:85`, and the
implementation is `_log_p_response` at `engine/core_mcmc.py:87`.

---

## 2. The Paper-1 engine: a particle cloud that learns a clinician

The production research engine is
`engine/core_mcmc.run_session_mcmc_auroc(method="hier", ...)`. It carries a
joint posterior over all $K = 7$ tasks at once, so evidence about one
pattern sharpens belief about correlated patterns. This is sequential
importance resampling (SIR) with a Metropolis-Hastings rejuvenation move.

### 2.1 State and prior

The latent state is a $2K$ vector stacking every task's bias and log
discrimination,

$$
\theta \;=\; (t_1, \dots, t_K,\; \ell_1, \dots, \ell_K) \in \mathbb{R}^{2K}.
$$

The posterior is represented by $N$ weighted particles
$\{(\theta^{(i)}, w_i)\}_{i=1}^{N}$, each particle also caching its log
prior, its cumulative log likelihood, and the running question history
(`make_state_hier`, `engine/core_mcmc.py:290`).

The prior is a zero mean multivariate normal in two blocks, $t \sim
\mathcal{N}(0, \Sigma_t)$ and $\ell \sim \mathcal{N}(0, \Sigma_\ell)$, with
density

$$
\log p(\theta) \;=\; -\tfrac{1}{2}\,\theta^\top \Sigma^{-1} \theta
\;-\; \tfrac{1}{2}\log\det \Sigma \;+\; \text{const}.
$$

Crucially $\Sigma_\ell$ is not diagonal: it is fitted from real raters
(loaded from `Sigma_l_fitted_k7.npy`), so the off-diagonal correlations let
a strong showing on, say, GPD raise the prior expectation for GRDA before a
single GRDA question is asked. A compound symmetry fallback
$\Sigma = (1 - r)I + rJ$ exists for backward compatibility. See
`log_prior_hier` at `engine/core_mcmc.py:167` and the fitted correlations
quoted in the comment block at `engine/core_mcmc.py:117`.

### 2.2 The Bayesian filter update

After the clinician answers item $(k, s)$ with label $y$, each particle
multiplies its weight by that particle's likelihood and the cloud
renormalizes,

$$
\tilde{w}_i \;=\; w_i \cdot P\!\big(y \mid \theta^{(i)}, k, s\big),
\qquad
w_i \;=\; \frac{\tilde{w}_i}{\sum_{j} \tilde{w}_j}.
$$

The cumulative log likelihood is incremented in place so a later
rejuvenation can replay it exactly (`update`, `engine/core_mcmc.py:346`).

### 2.3 Degeneracy and effective sample size

Repeated reweighting concentrates the mass on a few particles. We measure
that with the effective sample size

$$
\mathrm{ESS} \;=\; \frac{1}{\sum_{i} w_i^2},
$$

which ranges from $1$ (one particle holds all the weight) to $N$ (uniform).
This is the Kong 1994 diagnostic; see `ess` at `engine/core_mcmc.py:375`.

### 2.4 Resampling

When $\mathrm{ESS}$ falls below half the cloud ($\mathrm{ESS} < N/2$, the
default `ess_threshold_frac = 0.5` at `engine/core_mcmc.py:701`), the engine
resamples. It draws $N$ particle indices with replacement in proportion to
the weights (multinomial resampling),

$$
i_1, \dots, i_N \;\stackrel{\text{iid}}{\sim}\; \mathrm{Categorical}(w_1, \dots, w_N),
\qquad w \leftarrow \tfrac{1}{N}\mathbf{1},
$$

duplicating high weight particles and discarding low weight ones
(`resample_and_rejuvenate`, `engine/core_mcmc.py:432`).

### 2.5 Metropolis-Hastings rejuvenation

Resampling fixes the weights but leaves duplicates stacked on identical
coordinates, which throws away diversity. The engine restores it with a
genuine MCMC move that leaves the current posterior invariant. For each
particle it runs $n_{\mathrm{mh}} = 15$ steps of an adaptive random walk
Metropolis sampler.

The proposal is Gaussian with covariance shaped by the cloud itself: each
sweep recomputes the empirical $2K \times 2K$ covariance $\widehat{C}$ of
the particles, Choleskys it as $\widehat{C} = L L^\top$, and proposes

$$
\theta' \;=\; \theta \;+\; \rho\, L\,\varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, I),
\qquad \rho = 0.5,
$$

so the proposal covariance is $\rho^2 \widehat{C}$. The scale $\rho$ follows
the Roberts, Gelman and Gilks guidance that random walk efficiency peaks
near $2.38/\sqrt{d}$. Because the proposal is symmetric, the Hastings ratio
drops out and the acceptance probability is the plain Metropolis ratio of
posteriors,

$$
\alpha \;=\; \min\!\Big( 1,\;
\exp\big[ \big(\log p(\theta') + \log L(\theta')\big)
- \big(\log p(\theta) + \log L(\theta)\big) \big] \Big),
$$

where $\log L$ is the full question history replayed at the proposed state.
A move is accepted when $\log u < \log \alpha$ for $u \sim \mathrm{Unif}(0,1)$.
See `mh_rejuvenate` at `engine/core_mcmc.py:381`, with the history replay in
`_log_lik_history` at `engine/core_mcmc.py:220`.

This is the step that the docstring at `engine/core_mcmc.py:22` was built to
provide: it replaces an older West style kernel smoothing jitter that
quietly shrank the support toward its mean. MCMC rejuvenation moves
particles without distorting the distribution they came from.

### 2.6 Choosing the next question: A-optimal design

A good question is one whose answer we cannot predict and whose resolution
shrinks our uncertainty the most. The engine makes that precise. For a
candidate item $(k, s)$ it forms the two look ahead posteriors, the cloud
reweighted as if the answer were "yes" and as if it were "no", and scores
the candidate by the expected total posterior variance it would leave
behind:

$$
(k^\star, s^\star) \;=\;
\arg\min_{k, s} \;\Big[
p_{\text{yes}} \sum_{k'} \big( \mathrm{Var}_{y=1}[t_{k'}] + \mathrm{Var}_{y=1}[\ell_{k'}] \big)
+ (1 - p_{\text{yes}}) \sum_{k'} \big( \mathrm{Var}_{y=0}[t_{k'}] + \mathrm{Var}_{y=0}[\ell_{k'}] \big)
\Big],
$$

with $p_{\text{yes}} = \sum_i w_i\,P(y=1 \mid \theta^{(i)})$ the posterior
predictive, and the look ahead weights $w_i^{(1)} \propto P_i\, w_i$,
$w_i^{(0)} \propto (1 - P_i)\, w_i$. Minimizing the trace of the expected
posterior covariance is the classical A-optimal design criterion. Notice
the sum runs over all $K$ tasks, so the engine will sometimes ask a GRDA
question because, through the fitted prior covariance, it is the fastest way
to pin down LRDA. The implementation is `_expected_loss_vec` at
`engine/core_mcmc.py:472`, wrapped by `choose_item` at
`engine/core_mcmc.py:566`. An earlier KL information gain criterion was
removed (FIX-T0.5, docstring at `engine/core_mcmc.py:9`); A-optimality is
what ships.

### 2.7 Reporting skill as AUROC, and knowing when to stop

Skill is reported on a scale clinicians already trust: the area under the
ROC curve. Under the signal detection model, a reader with log
discrimination $\ell$ (hence internal noise $\sigma = e^{-\ell}$) facing
unit reference signal has

$$
\mathrm{AUROC}(\ell) \;=\;
\Phi\!\left( \frac{\sqrt{2}}{\sqrt{e^{-2\ell} + 1}} \right).
$$

Bias $t$ does not enter; AUROC is pure discrimination. Pushing every
particle's $\ell_k$ through this map turns the posterior over skill into a
posterior over AUROC, from which weighted quantiles give a point estimate
and a credible interval per task (`auroc_from_l` and
`auroc_quantiles_from_particles_hier`, `engine/auroc.py:32` and
`engine/auroc.py:39`).

The test stops when every task is known precisely enough. Writing the 95%
credible interval of task $k$'s AUROC as $[q_{0.025}, q_{0.975}]$, the rule
is

$$
\max_{k}\; \tfrac{1}{2}\big( q_{0.975}(\mathrm{AUROC}_k) - q_{0.025}(\mathrm{AUROC}_k) \big)
\;<\; \delta, \qquad \delta = 0.05.
$$

In words: keep asking until the widest task specific AUROC interval is
narrower than $0.05$ in half width. This is a posterior precision target,
evaluated at `engine/core_mcmc.py:791`. It is not a sequential hypothesis
test; see the closing section.

---

## 3. The deployment engine: a Kalman filter for skill

The clinical runtime in `deployment/simulate_test.py` answers a different
need. It must be fast, online, and explainable to a regulator, so it trades
the particle cloud for a Gaussian (Laplace) posterior updated by an
extended Kalman filter. It never imports the Paper-1 engine; the module
boundary is the architecture (decision D1).

### 3.1 Laplace posterior and the EKF update

The belief over $\theta \in \mathbb{R}^{2K}$ is a Gaussian
$\mathcal{N}(\mu, \Sigma_{\text{post}})$ initialized at the frozen prior.
Each answer is a probit GLM observation linearized at the current mean, the
defining move of an extended Kalman filter. With the score vector
$x = \partial \eta / \partial \theta$ (nonzero only in the answered task's
two slots, $x_{t_k} = e^{\ell_k}$ and $x_{\ell_k} = \eta$), the lapse GLM
Fisher weight and working response are

$$
w(\eta) \;=\; \frac{p'(\eta)^2}{p(\eta)\,(1 - p(\eta))},
\qquad
z \;=\; \eta + \frac{y - p(\eta)}{p'(\eta)},
$$

and the information form update is a rank one correction,

$$
\Sigma_{\text{post}}^{-1} \;\leftarrow\; \Sigma_{\text{post}}^{-1} + w\, x x^\top,
\qquad
\mu \;\leftarrow\; \Sigma_{\text{post}}\big( \Sigma_{\text{post}}^{-1}\mu + w\, x\, z \big).
$$

The equations are quoted in the module docstring at
`deployment/simulate_test.py:30` and implemented in `irls_w_z` and
`update_state` at `deployment/simulate_test.py:160` and
`deployment/simulate_test.py:230`. Item selection maximizes the log
determinant gain of that rank one update, $\log(1 + w\, x^\top
\Sigma_{\text{post}} x)$, which is the information theoretic twin of the
Paper-1 A-optimal rule (`select_next_case`, `deployment/simulate_test.py:267`).

### 3.2 PASS, FAIL, REFER

A task is decided by how much posterior mass sits above its cut score
$\ell^\star_k$. Under the Gaussian marginal,

$$
P(\ell_k > \ell^\star_k \mid \text{data}) \;=\;
1 - \Phi\!\left( \frac{\ell^\star_k - \mu_{\ell_k}}{\sigma_{\ell_k}} \right).
$$

The contract (in `deployment/deployment_config.yaml`) certifies the task
PASS when this probability reaches $0.95$, FAIL when it drops to $0.05$, and
otherwise keeps asking, subject to a global floor of $60$ and ceiling of
$500$ questions and per task bounds of $10$ and $120$. Tasks still
unresolved at the ceiling are referred rather than forced
(`simulate_candidate`, `deployment/simulate_test.py:467`).

---

## 4. Calibration: turning raters into a cut score

The engines above are only as honest as the numbers they are fed: each
segment's signal $s_j$, each task's cut score $\ell^\star_k$, and the prior
covariance $\Sigma$. Calibration produces all three, and it does so without
letting the answer leak into its own definition.

### 4.1 Two stage signal detection fits

Stage one is a Rasch style main effects fit that gives every segment a
difficulty on the logit scale. Stage two fits, for each rater $j$ and task,
a probit lapse signal detection model by maximum likelihood,

$$
P(y = 1 \mid c; \sigma_j, \theta_j) \;=\;
\lambda + (1 - 2\lambda)\,\Phi\!\left( \frac{c - \theta_j}{\sigma_j} \right),
$$

optimized over the criterion $\theta_j$ and the log noise $\log \sigma_j$
with L-BFGS-B, standard errors from a numerical Hessian. The rater's skill
is $\ell_j = -\log \sigma_j$ (sharper rater, smaller noise, larger skill)
and the engine bias is $t = -\theta_j$. This is the modular cut of Plummer
2015: the heavy joint model supplies segment signals, while per rater
parameters are fit in a separate, cheap, decoupled stage. See
`fit_sdt_per_domain.py:51` for the negative log likelihood and `:71` for the
optimizer.

### 4.2 The 1.7 bridge

Stage one speaks logits; stage two speaks probits. The logistic and normal
CDFs differ by a scale factor close to $1.7$, so the pipeline divides every
logit difficulty by $1.7$ before the probit fit:

$$
c_{\text{probit}} \;=\; \frac{c_{\text{logit}}}{1.7}.
$$

The constant is `LOGIT_TO_PROBIT = 1.0 / 1.7` at
`fit_sdt_per_domain.py:38`, and the orchestrator asserts it at runtime
(`pipeline/run_unified_calibration.py:201`) so the bridge can never silently
drift.

### 4.3 The cut score, defined without circularity

The cut score $\ell^\star_k$ is the skill at which an expert and a non
expert are best separated, by the Youden J statistic,

$$
\ell^\star_k \;=\; \arg\max_{\ell}\; \big[ F_{\text{nonexpert}}(\ell) - F_{\text{expert}}(\ell) \big],
\qquad J = \max_{\ell} \big[ F_{\text{nonexpert}}(\ell) - F_{\text{expert}}(\ell) \big],
$$

with $F$ the empirical CDFs of fitted skill. The subtle danger is defining
"expert" by skill on the very task you are calibrating, which inflates $J$
by construction. The pipeline avoids it with leave one task out cross
validation: a rater's expert status for task $k$ is decided by their mean
skill on the other six tasks, and only the top 14 of that ranking form the
panel, with 95% intervals from 2000 bootstrap resamples. Expert and non
expert pools are split 70/30 and 50/50 into train and validation, and the
cut score is computed on train only. The CV rationale is quoted in the
docstring at `pipeline/reference_calibration/run_youden_calibration.py:1`;
the splits are enforced at `pipeline/run_unified_calibration.py:458`.

### 4.4 The joint hierarchical model

Behind the segment signals is a hierarchical item response model fit in
NumPyro. Each observation has linear predictor

$$
\eta_{i, d, j} \;=\; \exp\!\big( \ell_i + \log \gamma_d \big)\,
\big( s_j + \delta_d - t_i \big),
$$

with per rater skill and bias drawn from expertise tier means, a per source
discrimination $\gamma_d$ and offset $\delta_d$ (sum to zero, anchored), and
the segment signal $s_j = \tau_s \tilde{s}_j$ as the quantity carried
downstream. Production inference is stochastic variational inference with a
low rank multivariate normal guide; NUTS is retained as a gold standard
validation chain gated at $\hat{R} < 1.05$. The model is defined at
`pipeline/joint_calibration/model_k7.py:178` (the shipped variant B), the
inference driver at `pipeline/joint_calibration/fit_joint_k7.py:139`.

### 4.5 Cross task covariance

Finally the prior covariance is estimated by pairing raters who scored in
two tasks and taking the empirical covariance of their skills and biases,
yielding the $14 \times 14$ matrix $\Sigma$ the engines load. The shipped
deployment form collapses the skill block to a single average correlation
$r_\ell = 0.367$ (the K=7 value, retiring the older K=6 figure of $0.326$).
See `scripts/build_sigma_l_k7.py:108` for the pairwise estimator and
`deployment/freeze_deployment_prior.py:124` for the average.

---

## 5. How we know it works

Calibration and inference are validated, not asserted.

- Simulation based calibration (Talts et al. 2018): simulate from the
  prior, fit, and check that the rank of the truth in the posterior is
  uniform. Skeleton and runners at `tests/test_phase35_sbc_engine.py` and
  `scripts/run_sbc_validation.py`.
- Posterior coverage: nominal 90% intervals should cover the truth 90% of
  the time; tracked in `tests/test_posterior_coverage.py`.
- Bit exact determinism: the engine is reproducible across machines under
  single thread BLAS, and serial equals parallel to the bit
  (`tests/test_parallel_determinism.py`).
- Real rater replay: rather than only simulate, the system replays the
  recorded answers of 21 real clinicians through the engine
  (`docs/PHASE7_REPLAY_HEADLINE.md`).

---

## 6. What this is not

To keep the documentation trustworthy, three explicit non claims.

1. The stopping rule is not a Wald sequential probability ratio test. It is
   a posterior precision criterion (AUROC half width below $\delta$). The
   numerical boundaries can be made to look SPRT like, but the operating
   characteristics differ under adaptive question selection, and the source
   says so at `engine/core_mcmc.py:40`.
2. The joint calibration model is a two parameter probit item response
   model, not a pure Rasch model; Rasch fixes discrimination, this estimates
   it per rater and per source.
3. The production calibration posterior is variational (SVI), with NUTS used
   only to validate it. We do not claim full MCMC posteriors in production.

---

## 7. References

The methods above draw on, and where noted depart from, the following.

- Metropolis, Rosenbluth, Rosenbluth, Teller and Teller (1953); Hastings
  (1970). The Metropolis-Hastings sampler used in rejuvenation.
- Gordon, Salmond and Smith (1993); Doucet, de Freitas and Gordon (2001).
  The bootstrap / SIR particle filter.
- Roberts, Gelman and Gilks (1997). Optimal scaling of random walk
  Metropolis, the source of the $2.38/\sqrt{d}$ proposal guidance.
- Kong, Liu and Wong (1994). Effective sample size for importance sampling.
- Gelman et al. (2013). Split $\hat{R}$ convergence diagnostic.
- Talts, Betancourt, Simpson, Vehtari and Gelman (2018). Simulation based
  calibration.
- Plummer (2015); Jacob, Murray, Holmes and Robert (2017). Cuts in Bayesian
  graphical models, the modular inference framework.
- Patz and Junker (1999); Bafumi and Gelman (2007). MCMC and hierarchical
  treatment of rater item response models.
- Youden (1950). The J statistic for threshold selection.
- Chaloner and Verdinelli (1995). Bayesian optimal (A-optimal) experimental
  design.

Each claim in sections 1 through 5 carries a `file:line` pointer; the code
is the final authority where it and any prose disagree.
