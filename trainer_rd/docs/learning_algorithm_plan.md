\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[round,authoryear]{natbib}
\usepackage{parskip}
\usepackage{microtype}
\usepackage[svgnames]{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=DarkBlue, urlcolor=DarkBlue, citecolor=DarkBlue}

\title{From measurement to training:\\
  \large modelling learning with feedback for an signal pattern-recognition
  certification test}
\author{Anonymous}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
The current ENGINE certification test is built on a signal-detection-theory
(SDT) model in which each examinee is summarised by two parameters per
task: a perceptual noise $\sigma$ (sensitivity, $1/\sigma$ is the skill)
and a decision criterion $t$ (bias). Bayesian adaptive design
(QUEST$+$-style) lets us \emph{estimate} $(\sigma,t)$ efficiently. In
this note we summarise two internal design memos
(\texttt{learning-theory1.md} and \texttt{learning-theory2.md}) that
address the complementary problem: given $(\hat\sigma,\hat t)$ for a
learner, what is the optimal sequence of cases (with feedback) to
actively \emph{move} those parameters toward better values?
The shift in viewpoint is that, in a training setting, $\sigma$ and $t$
are no longer stationary unknowns but \emph{state variables}, and our
job is to drive them. The design that emerges is a small,
interpretable, posterior-driven policy that explicitly separates three
training objectives, reuses the existing inference machinery, and
exploits more structure than either the perceptual-learning or
POMDP-teaching literatures use in isolation.
\end{abstract}

\section{The conceptual shift: assessment vs.\ training}
\label{sec:shift}

The existing ENGINE framework treats $(\sigma,t)$ as fixed unknowns
and selects each next case to maximise expected information gain about
the joint posterior. That is provably efficient for \emph{measurement},
but it is the wrong objective for \emph{training}, where the learner's
$(\sigma,t)$ are precisely the things we want to change.

\begin{center}
\begin{tabular}{@{}lll@{}}
\toprule
 & \textbf{Assessment} & \textbf{Training} \\
\midrule
$(\sigma,t)$ are \dots & stationary unknowns & state variables we want to move \\
Item-selection objective & expected information gain & expected improvement in future performance \\
Reward signal & posterior-variance reduction & $\Delta(-|t|),\;\Delta(-\sigma)$, retention \\
Feedback role & none required & central; drives the dynamics \\
\bottomrule
\end{tabular}
\end{center}

A measurement-optimal stimulus is typically placed \emph{at} the
criterion (uncertain by design); a training-optimal stimulus is
typically placed at a different difficulty depending on which of three
distinct learning sub-problems the learner currently has
(Sec.~\ref{sec:threeproblems}).

\section{Four (overlapping) literatures}
\label{sec:lit}

\paragraph{Adaptive psychophysics and computerised adaptive testing.}
QUEST and QUEST$+$ \citep{watson1983quest,watson2017questplus} and
item-response-theory CAT both maximise information about a fixed
ability parameter; neither models the parameter as changing. They are
the right tools for our \emph{measurement} sub-system and the wrong
ones for a trainer.

\paragraph{Psychophysics of perceptual learning.}
Dosher and Lu's Augmented Hebbian Reweighting Model and its successors
give a mechanistic, trial-by-trial account of how repeated practice
reduces $\sigma$ (via channel reweighting, noise suppression, signal
enhancement) and how feedback shifts $t$. The single most
operationally useful empirical result, from
\citet{aberg2012feedback} and earlier work by Herzog and colleagues,
is that \textbf{block feedback primarily moves $\sigma$ while
trial-by-trial feedback primarily moves $t$}, and that biased
trial-wise feedback can be used to drive $t$ in a controllable
direction.\footnote{We do \emph{not} propose to do this — see the
caveats in Sec.~\ref{sec:caveats}.} The two parameters respond to
different knobs and probably need different schedules.

\paragraph{The 85\,\% rule for optimal training difficulty.}
\citet{wilson202085percent} derive that for a broad class of
stochastic-gradient-descent binary classifiers — and argue this
generalises to humans on perceptual binary-decision tasks — the
\emph{rate} of learning is maximised when training accuracy is held
near $85\%$ (error rate $\approx 15.87\%$). This is a formal derivation
of Vygotsky's zone of proximal development and Bjork's
``desirable difficulty,'' and converges with established staircasing
practice. Critically the $85\%$ refers to \emph{training} accuracy and
\emph{not} test accuracy — exactly the distinction this note is
about.

\paragraph{Perceptual-learning systems in imaging.}
The closest existing analog to what we want to build is the Kellman
group's an adaptive-learning module family, scheduled by
the PLM algorithm (Adaptive Response-Time-based Sequencing,
\citealp{mettler2011arts}), which prioritises categories by per-category
accuracy, response time as a fluency proxy, and trials-since-last-
presentation. \citet{kellman2024palm} is the first to integrate SDT
into the pipeline: a running window $d'$ serves as retirement criterion
and beats accuracy-based retirement on both immediate and delayed
post-tests. As far as we can tell this is the only existing
perceptual-learning system that unifies measurement, sequencing, and
mastery in a single framework, and we borrow liberally from its
framing.

\paragraph{POMDP / Bayesian knowledge tracing.}
The decision-theoretic ``right answer'' to optimal teaching is
\citet{rafferty2016pomdp}: cast teaching as planning in a partially
observable Markov decision process whose hidden state is the learner's
knowledge state. This is the cleanest formal statement of our problem,
but exact POMDP planning is heavy, and the standard formulations
discretise mastery into binary states rather than maintaining a
continuous posterior over $(\sigma,t)$. Spaced-repetition schedulers
(SM-2, the Anki family) are simple heuristic approximations of this
class and have strong empirical backing in education.

\section{Why our problem has \emph{more} structure than any of the above}
\label{sec:structure}

The domain1 training problem is unusually well-conditioned because
we already have:

\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item a principled \emph{generative model} of expert behaviour
        (a two-parameter probit with a lapse term);
  \item a joint \emph{posterior} over $(\sigma,t)$ per examinee per
        task, maintained by the ENGINE engine;
  \item stimuli that live on a known, signed \emph{difficulty axis $s$}
        (the latent SDT signal), not in a discrete skill-tag space.
\end{enumerate}

This means learning can be expressed as \emph{dynamics on the
posterior-mean trajectory} $(\sigma_k, t_k)$ rather than as discrete
mastery transitions. Discrete-state BKT / POMDP work does not exploit
this; the Kellman/PLM priority score does not either. Our trainer can
do better because the generative model is richer.

\section{The POMDP formulation}
\label{sec:framework}

The trainer is a partially observable Markov decision process
(POMDP): we don't know the learner's true $(\sigma, t)$, we get only
the binary response $y_k$ each trial, and we have to choose the next
stimulus $s_k$ to maximise long-run improvement. This section is
organised in three parts. First the abstract POMDP setup and its
formal optimal solution
(Secs.~\ref{sec:pomdp-tuple}--\ref{sec:policy-formal}): what the
problem is and what an optimal policy would look like if we could
compute it. Then the specific functional forms we adopt for the
observation, transition, and reward components
(Secs.~\ref{sec:obs}--\ref{sec:reward}). Finally the tractable
approximations to the Bellman equation that yield the practical
algorithms we will actually run
(Sec.~\ref{sec:approximations}).

\subsection{The POMDP tuple}
\label{sec:pomdp-tuple}

A POMDP is a 7-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O},
T, Z, R, \gamma)$. For the trainer:

\begin{description}[leftmargin=2.2em,style=nextline,itemsep=2pt]
  \item[State $\mathcal{S}$.] $\theta_k = (\sigma_k, t_k, \phi_k)$.
    The perceptual noise $\sigma_k \in \mathbb{R}_{>0}$, the
    criterion $t_k \in \mathbb{R}$, and an optional vector $\phi_k$
    holding nuisance state (lapse rate, sub-type-specific skill
    offsets, fatigue). \emph{In testing $\theta$ is treated as
    stationary; in training it evolves under $T$.}
  \item[Action $\mathcal{A}$.] Choice of stimulus $s_k$ for the next
    trial. In ENGINE the action space is discrete (pick one of the
    $\sim$10$^4$ signal segments in the bank, each indexed by its latent
    difficulty $s$ on the signed SDT axis plus a categorical
    morphology / sub-type tag $\mu$). For the math we treat the
    action as the pair $(s, \mu) \in \mathbb{R} \times
    \{1,\dots,M\}$.
  \item[Observation $\mathcal{O}$.] The learner's binary response
    $y_k \in \{0, 1\}$. Optionally an interaction trace
    (reaction time, number of answer changes, etc.) feeding $\phi_k$;
    we ignore that auxiliary channel below for clarity.
  \item[Transition kernel $T$.] $T(\theta_{k+1} \mid \theta_k, s_k,
    y_k, \mathrm{fb}_k)$: the learning dynamics. We assume $T$
    factorises across components,
    \[
      T(\theta_{k+1} \mid \theta_k, s_k, y_k, \mathrm{fb}_k)
      \;=\;
      T_t(t_{k+1} \mid \cdot)\,
      T_\sigma(\log\sigma_{k+1} \mid \cdot)\,
      \delta(\phi_{k+1} - \phi_k),
    \]
    with $T_t$ and $T_\sigma$ Gaussian and $\phi$ held constant
    within a session. The deterministic means of $T_t$ and
    $T_\sigma$ are derived in Secs.~\ref{sec:gt}--\ref{sec:gsig}.
  \item[Observation kernel $Z$.] $Z(y_k \mid \theta_k, s_k)$: the
    SDT-with-lapse model. Detailed in Sec.~\ref{sec:obs}.
  \item[Reward $R$.] $R(\theta_k, \theta_{k+1})$: improvement in
    bias, skill, and retention. Detailed in
    Sec.~\ref{sec:reward}.
  \item[Discount $\gamma$.] $\gamma \in (0, 1]$. For a fixed-length
    training session of $K$ trials, take $\gamma = 1$ and the planning
    horizon as the residual session length; for an open-ended trainer,
    $\gamma \lesssim 1$ to weight near-term improvement more.
\end{description}

Because $\theta_k$ is not directly observable, the trainer must act
on its \emph{belief state} $b_k$, the posterior over $\theta_k$ given
all history through trial $k$. The POMDP reduces to an MDP on the
infinite-dimensional belief simplex; in practice we represent $b_k$
with the same SMC particle cloud the ENGINE engine already maintains.

\subsection{Belief state and the Bayesian filter}
\label{sec:belief}

Let the belief at the start of trial $k$ be $b_k(\theta) = p(\theta_k
\mid \mathcal{H}_k)$, where $\mathcal{H}_k = \{s_j, y_j, \mathrm{fb}_j
\}_{j < k}$ is the history. After observing $y_k$ and revealing
$\mathrm{fb}_k$, two operations update the belief:

\smallskip
\noindent\emph{Measurement step (Bayes on $Z$):}
\begin{equation}
  b_k^{+}(\theta) \;\propto\; Z(y_k \mid \theta, s_k)\,b_k(\theta).
  \label{eq:bayes}
\end{equation}

\smallskip
\noindent\emph{Prediction step (push through $T$):}
\begin{equation}
  b_{k+1}(\theta') \;=\;
    \int T(\theta' \mid \theta, s_k, y_k, \mathrm{fb}_k)\,
         b_k^{+}(\theta)\,\mathrm{d}\theta.
  \label{eq:predict}
\end{equation}

Together, Eqs.~\ref{eq:bayes}--\ref{eq:predict} are a sequential
Bayesian (state-space) filter — exactly the operation a Kalman filter
performs in linear-Gaussian settings, generalised here to a nonlinear
non-Gaussian state. The Gaussian process noises baked into $T_t$ and
$T_\sigma$ (Secs.~\ref{sec:gt}--\ref{sec:gsig}) are operationally
important: they prevent the posterior from concentrating prematurely,
which would make the trainer unable to track real drift in
$(\sigma_k, t_k)$.

In ENGINE the SMC particle representation makes this concrete: the
existing $N$-particle cloud over $(\sigma, t)$ is reweighted by
Eq.~\ref{eq:bayes}, then each particle is propagated forward by
sampling from $T$ in Eq.~\ref{eq:predict}; resampling whenever
ESS drops below a threshold. The only engine change required is the
addition of the propagation step — the reweighting step is already
there.

\subsection{Optimal policy: value function and Bellman equation}
\label{sec:policy-formal}

A \textbf{policy} $\pi$ is a function from belief states to actions:
$\pi(b)$ tells the trainer which stimulus to present given everything
it currently knows about the learner. To compare policies, we
summarise each by the long-run reward it produces. The
\textbf{state-value function} of belief $b$ at trial index $k$ under
policy $\pi$ is the expected sum of future rewards earned from trial
$k$ until the end of the session:
\begin{equation}
  V_k^{\pi}(b) \;=\;
    \mathbb{E}_{\pi}\!\left[
      \sum_{j = 0}^{K - k - 1} \gamma^{j}\,R_{k+j}
      \;\Big|\; b_k = b
    \right],
  \label{eq:value}
\end{equation}
where $K$ is the total session length (Sec.~\ref{sec:pomdp-tuple}),
$\gamma$ is the discount factor, and the expectation averages over
the random observations produced by $Z$, the random learning
transitions produced by $T$, and any randomness in $\pi$. The
remaining horizon $K - k$ shrinks as the session progresses, which
is why $V_k^\pi$ carries a $k$ subscript even when the policy itself
is stationary --- there is genuinely less reward left to earn near
the end of the session. (For an open-ended trainer the limit $K
\to \infty$ with $\gamma < 1$ recovers the usual time-homogeneous
$V^\pi(b)$.)

The \textbf{optimal} value function $V_k^{\star}(b) = \max_\pi
V_k^\pi(b)$ is the best return achievable from $b$ at trial $k$.
By the principle of optimality, $V_k^{\star}$ satisfies a recursive
consistency condition --- the \textbf{Bellman equation} --- which
says the value of being at $b$ at trial $k$ equals the best one-step
expected reward plus the discounted value of wherever that step
lands you at trial $k+1$:
\begin{equation}
  V_k^{\star}(b) \;=\;
    \max_{s \in \mathcal{A}}\;
    \mathbb{E}_{y \sim Z(\cdot | b, s)}\!
    \biggl[
      \underbrace{\mathbb{E}_{\theta, \theta' \sim T, b}\!
        \bigl[R(\theta, \theta')\bigr]}_{\text{expected immediate reward}}
      \;+\;
      \underbrace{\gamma\,V_{k+1}^{\star}\!\bigl(b'(b, s, y)\bigr)}
        _{\text{discounted future value}}
    \biggr],
  \label{eq:bellman}
\end{equation}
with terminal condition $V_K^{\star}(b) \equiv 0$. Here $b'(b, s, y)$
is the post-update belief produced by
Eqs.~\ref{eq:bayes}--\ref{eq:predict} after taking action $s$ and
observing $y$. The outer $\max_s$ picks the stimulus whose
expected-reward-plus-future-value is largest; the inner expectations
average over what the learner might respond and how their state
might transition. The optimal policy is correspondingly
time-indexed and reads off the argmax,
\begin{equation}
  \pi_k^{\star}(b) \;=\; \arg\max_{s \in \mathcal{A}}\;\{\cdots\},
  \label{eq:pistar}
\end{equation}
i.e.\ at trial $k$ and belief $b$, present the stimulus that
maximises the bracketed quantity in Eq.~\ref{eq:bellman}.

Eqs.~\ref{eq:value}--\ref{eq:pistar} imply a backward-induction
recipe: starting from $V_K^\star \equiv 0$, sweep $k$ from $K - 1$
down to $0$, computing $V_k^\star$ from $V_{k+1}^\star$ via
Eq.~\ref{eq:bellman}, and reading off $\pi_k^\star$ at each step.
This is intractable in closed form on our continuous belief space.
Filling in the specific $Z$, $T$, $R$
(Secs.~\ref{sec:obs}--\ref{sec:reward}) and finding tractable
approximations to the recursion (Sec.~\ref{sec:approximations}) is
what the rest of this section does.

\subsection{Observation model: probit-lapse SDT}
\label{sec:obs}

For stimulus $s_k$ and learner state $\theta_k = (\sigma_k, t_k,
\lambda_k)$ (we make the lapse rate $\lambda$ explicit here),
\begin{equation}
  Z(y_k = 1 \mid \theta_k, s_k) \;=\;
    (1 - 2\lambda_k)\,\Phi\!\left(\frac{s_k - t_k}{\sigma_k}\right)
    \;+\; \lambda_k,
  \label{eq:obs}
\end{equation}
identical to the ENGINE live-test engine. $\Phi$ is the standard
normal CDF; the lapse mixture absorbs attentional / motor errors.
This is the same model used during measurement — we are not
re-deriving the observation channel, just naming it explicitly so the
state dynamics below have something to interact with.

\subsection{Transition kernel: criterion dynamics $g_t$}
\label{sec:gt}

We parameterise the criterion update as
\begin{equation}
  t_{k+1} \;=\; t_k + \alpha_t\,g_t(s_k, y_k^{\star}, \theta_k)
              + \xi_{t,k},
  \quad \xi_{t,k} \sim \mathcal{N}(0, q_t^2),
  \label{eq:tdyn-generic}
\end{equation}
and need to choose the update function $g_t$. We adopt a
Rescorla--Wagner-style \textbf{prediction-error} rule, where the
prediction is the learner's model-implied probability of saying
``yes'' (continuous) and the outcome is the ground-truth label
(binary). Let
\begin{equation}
  \hat y_k \;\equiv\; Z(y_k = 1 \mid \theta_k, s_k)
  \;=\; (1 - 2\lambda_k)\,\Phi\!\bigl((s_k - t_k)/\sigma_k\bigr) + \lambda_k
  \label{eq:yhat}
\end{equation}
be the predicted probability (Eq.~\ref{eq:obs} re-read as a function
of state), and define the \textbf{(soft) outcome prediction error}
\begin{equation}
  \delta_k \;\equiv\; \hat y_k - y_k^{\star}
  \;\in\; [-1, +1],
  \label{eq:delta}
\end{equation}
where $y_k^{\star} \in \{0, 1\}$ is the ground-truth label revealed
by feedback $\mathrm{fb}_k$. Sign interpretation:
\begin{itemize}[leftmargin=*,itemsep=1pt]
  \item $\delta_k > 0$: predicted yes more strongly than warranted.
        The criterion is too liberal — raise it.
  \item $\delta_k < 0$: predicted yes less strongly than warranted.
        The criterion is too conservative — lower it.
  \item $\delta_k \approx 0$: prediction matched the truth. Small
        update.
\end{itemize}
Taking $g_t(s_k, y_k^{\star}, \theta_k) = \delta_k$ specialises
Eq.~\ref{eq:tdyn-generic} to
\begin{equation}
  \boxed{\;
    t_{k+1} \;=\; t_k + \alpha_t\,\delta_k + \xi_{t,k},
    \quad \xi_{t,k} \sim \mathcal{N}(0, q_t^2).
  \;}
  \label{eq:tdyn}
\end{equation}
Equivalently, this defines the criterion factor of the transition
kernel as $T_t(t_{k+1} \mid \theta_k, s_k, y_k^{\star}) =
\mathcal{N}\!\bigl(t_{k+1};\, t_k + \alpha_t\,\delta_k,\, q_t^2\bigr)$.

\paragraph{Relationship to cross-entropy gradient descent.}
Equation~\ref{eq:delta} is the textbook Rescorla--Wagner update —
continuous prediction minus binary outcome — and we adopt it as a
postulated learning rule, not as a derivation. It is worth noting
that it shares its sign and zeros with the gradient of cross-entropy
loss with respect to $t$. If the learner is implicitly performing
stochastic gradient descent on
\[
  L_k(t) \;=\; -y_k^{\star}\log\hat y_k(t)
              \;-\; (1 - y_k^{\star})\log\bigl(1 - \hat y_k(t)\bigr),
\]
then a short calculation gives
\begin{equation}
  \frac{\partial L_k}{\partial t}
  \;=\; -\,\frac{(1 - 2\lambda_k)\,\phi(z_k)/\sigma_k}
                {\hat y_k\,(1 - \hat y_k)}
        \;\bigl(\hat y_k - y_k^{\star}\bigr),
  \qquad z_k = \frac{s_k - t_k}{\sigma_k},
  \label{eq:cegradient}
\end{equation}
so R--W and gradient descent both produce $\Delta t \propto +\delta_k$,
but differ by a state-dependent positive multiplier $m_k = (1{-}2
\lambda_k)\phi(z_k)/[\sigma_k\,\hat y_k(1{-}\hat y_k)]$, which R--W
replaces with the constant $\alpha_t$. This is an approximation, not
an identity: $m_k$ varies trial-to-trial (largest near threshold,
smallest in the tails), so treating it as constant is a behavioural
simplification justified empirically by the long success of R--W in
fitting choice data \citep{stuttgen2011trial}, not by algebra. What
the soft form buys, relative to the hard form below, is that
\emph{every} trial contributes: large updates on confident errors,
small on confident-correct trials, intermediate on near-threshold
correct trials. The implausible prediction of a strictly binary rule
— that a 51\,\%-confident correct ``yes'' produces zero criterion
shift — is avoided.

\paragraph{Hard-error form as a special case.}
An older SDT-criterion-learning tradition
\citep{kac1962criterion,friedman1968criterion,stuttgen2011trial} uses
instead the \emph{hard} prediction error
\begin{equation}
  \delta_k^{\mathrm{hard}}
  \;\equiv\; y_k - y_k^{\star}
  \;\in\; \{-1, 0, +1\},
  \label{eq:deltahard}
\end{equation}
which fires only on errors. This is what Eq.~\ref{eq:delta} reduces
to if we replace the model-implied probability $\hat y_k$ with the
learner's binary response $y_k$ — equivalently, if the learner has
introspective access only to their own categorical choice and not to
its underlying probability. Because $y_k \sim \mathrm{Bernoulli}(\hat
y_k)$, the hard form is an \emph{unbiased} but noisier estimator of
the soft form:
\begin{equation}
  \mathbb{E}\!\bigl[\,y_k - y_k^{\star} \mid \theta_k, s_k, y_k^{\star}\bigr]
  \;=\; \hat y_k - y_k^{\star}.
  \label{eq:hard-unbiased}
\end{equation}
The two formulations therefore have the same expected dynamics but
differ in trial-level variance, and — more interestingly — they make
different qualitative predictions on confident-correct vs.\
near-threshold-correct trials. Which one better describes real
learners is an empirical question (and is identifiable from
training-trial data once we have any). We take the soft form as the
default for everything below; the hard form is the natural ablation
baseline for the validation in Sec.~\ref{sec:next}.

\paragraph{Parameter glossary for Eq.~\ref{eq:tdyn}.}
\begin{description}[leftmargin=2.2em,style=nextline,itemsep=1pt]
  \item[$\alpha_t \ge 0$.] Criterion learning rate, in units of the
    perceptual axis (the same units as $s$ and $t$). Estimated
    per-learner; empirical-Bayes prior from the pooled pilot cohort.
    Under the soft form, $\alpha_t$ absorbs the $(1{-}2\lambda)\phi/
    [\sigma\,\hat y(1{-}\hat y)]$ scaling of Eq.~\ref{eq:cegradient}.
    Plausible magnitudes lie in $[0.05, 0.5]$; the corresponding
    hard-form rate \citep{stuttgen2011trial} is comparable but
    applies only on the subset of trials with $y_k \neq y_k^{\star}$,
    so produces a noisier trajectory with the same expectation
    (Eq.~\ref{eq:hard-unbiased}).
  \item[$\xi_{t,k}$.] Trial-level process noise with variance $q_t^2$;
    accounts for criterion fluctuation not captured by the
    deterministic R--W update (attention, mood, drift). Acts as the
    state-space-filtering noise that prevents the posterior from
    collapsing prematurely (see Sec.~\ref{sec:belief}).
  \item[$\delta_k$.] Soft prediction error in $[-1, +1]$
    (Eq.~\ref{eq:delta}); reduces to the hard form
    (Eq.~\ref{eq:deltahard}) under the introspection assumption above.
\end{description}

\subsection{Transition kernel: skill dynamics $g_\sigma$}
\label{sec:gsig}

Skill ($\sigma^{-1}$) evolves much more slowly than criterion. The
empirical learning curve in perceptual-learning experiments is
well-approximated by an exponential approach to an asymptote — i.e.\
$\sigma_k$ relaxes from $\sigma_0$ toward $\sigma_\infty$:
\begin{equation}
  \sigma_k \;\approx\;
    \sigma_\infty + (\sigma_0 - \sigma_\infty)\,
    \exp\!\bigl(-k/\tau_\sigma\bigr).
  \label{eq:sigmaschedule}
\end{equation}
The per-trial discretisation of this trajectory is a first-order
relaxation:
\begin{equation}
  \log\sigma_{k+1} - \log\sigma_k
  \;\approx\; -\frac{1}{\tau_\sigma}\bigl(\log\sigma_k - \log\sigma_\infty\bigr).
  \label{eq:sigma-relax}
\end{equation}
Two refinements turn this into a model we can drive with a policy.

\smallskip\noindent\textbf{(a) Difficulty-appropriateness weight.}
Not every trial contributes equally to skill. The 85\,\% rule
\citep{wilson202085percent} says that trials at the \emph{learner's
current} 85\,\% accuracy difficulty contribute most. Let
\begin{equation}
  w(s_k, \theta_k) \;=\;
    \exp\!\left(-\,
      \frac{\bigl(\,|s_k - t_k|/\sigma_k - 1.04\bigr)^2}{2\rho^2}
    \right)
  \;\in (0, 1]
  \label{eq:weight}
\end{equation}
be a soft Gaussian indicator that peaks at $|s_k - t_k|/\sigma_k
\approx 1.04$ (the 85\,\% point under unit-variance probit with
lapse $\approx 0.025$, Eq.~\ref{eq:eightyfive}), with width $\rho$
controlling how forgiving the criterion is. $w \approx 1$ for an
ideally placed trial, $w \to 0$ for trials either far too easy (no
challenge) or far too hard (no signal).

\smallskip\noindent\textbf{(b) Feedback-dependence.}
A trial without veridical feedback contributes negligibly to
$\sigma$-learning \citep{aberg2012feedback}. Let
$f_k \in \{0, 1\}$ indicate whether immediate feedback was delivered
on trial $k$.

\smallskip\noindent Combining (a) and (b),
\begin{equation}
  g_\sigma(s_k, \theta_k, \mathrm{fb}_k)
  \;=\; -\,f_k \cdot w(s_k, \theta_k) \cdot
        \bigl(\log\sigma_k - \log\sigma_\infty\bigr),
  \label{eq:gsigma}
\end{equation}
yielding
\begin{equation}
  \boxed{\;
    \log\sigma_{k+1} \;=\; \log\sigma_k + \alpha_\sigma\,
      g_\sigma(s_k, \theta_k, \mathrm{fb}_k) + \xi_{\sigma,k},
    \quad \xi_{\sigma,k} \sim \mathcal{N}(0, q_\sigma^2).
  \;}
  \label{eq:sdyn}
\end{equation}
Equivalently, this defines the skill factor of the transition kernel
as $T_\sigma(\log\sigma_{k+1} \mid \theta_k, s_k, \mathrm{fb}_k) =
\mathcal{N}\!\bigl(\log\sigma_{k+1};\, \log\sigma_k + \alpha_\sigma\,
g_\sigma,\, q_\sigma^2\bigr)$.

\paragraph{Parameter glossary for Eq.~\ref{eq:sdyn}.}
\begin{description}[leftmargin=2.2em,style=nextline,itemsep=1pt]
  \item[$\alpha_\sigma \ge 0$.] Skill learning rate. With
    Eq.~\ref{eq:gsigma}'s form, $\alpha_\sigma = 1/\tau_\sigma$ when
    $w = f = 1$. Per-learner; typical perceptual-learning $\tau_\sigma$
    values in the literature are tens to a few hundred trials.
  \item[$\sigma_\infty$.] Asymptotic noise floor — the irreducible
    perceptual noise even after long training, set by the learner's
    intrinsic sensory limits. Per-learner; weakly identified from a
    short training session, hence the empirical-Bayes prior.
  \item[$\rho$.] Width of the 85\,\%-rule weight; how off-target a
    trial can be and still meaningfully contribute. Cohort-level
    nuisance.
  \item[$\xi_{\sigma,k}$.] Process noise on $\log \sigma$ with
    variance $q_\sigma^2$. Same role as $\xi_{t,k}$ in
    Eq.~\ref{eq:tdyn}.
\end{description}

\paragraph{What is \emph{not} in $g_\sigma$.}
We omit (i) explicit dependence on the correctness $c_k = \mathbf{1}
[y_k = y_k^{\star}]$ — feedback delivers the correct answer either
way, and the perceptual-learning literature suggests being wrong is
not strictly necessary for $\sigma$-improvement — and (ii)
sub-type-specific transfer terms, which can be folded in via $\phi_k$
once we have enough data to fit them. Both are natural extensions for
Phase 3.

\subsection{Reward}
\label{sec:reward}

The trainer cares about three things: driving $|t|$ toward zero
(unbiased decisions), driving $\sigma$ toward its irreducible floor
$\sigma_\infty$ (sharper discrimination), and retention of
previously mastered material. We define the per-trial reward as the
improvement in each of these:
\begin{equation}
  R(\theta_k, \theta_{k+1})
  \;=\;
  \beta_t \bigl(\,|t_k| - |t_{k+1}|\,\bigr)
  \;+\;
  \beta_\sigma \bigl(\sigma_k - \sigma_{k+1}\bigr)
  \;+\;
  \beta_r \cdot \mathrm{ret}_k.
  \label{eq:reward}
\end{equation}
Each term is constructed so that progress in the desired direction
yields positive reward:
\begin{itemize}[leftmargin=*,itemsep=1pt]
  \item $\beta_t(|t_k| - |t_{k+1}|) > 0$ when $|t|$ shrinks — the
        criterion has moved closer to zero, so the learner is less
        biased after the trial than before.
  \item $\beta_\sigma(\sigma_k - \sigma_{k+1}) > 0$ when $\sigma$
        shrinks — the learner's discrimination has sharpened.
  \item $\mathrm{ret}_k \in \{0, +1\}$ flags a successful spaced
        retrieval of a previously mastered sub-category; $+1$ if
        the trial drew from a sub-category the learner had retired
        and the response was correct, $0$ otherwise.
\end{itemize}
Because the first two terms are \emph{differences} between
consecutive states, $R$ is literally the state-improvement on trial
$k$, and the cumulative return $\sum_k R_k$ telescopes (modulo the
retention bonus) into the total improvement over the session.

The weights $\beta_t, \beta_\sigma, \beta_r$ are chosen by the
protocol designer based on what matters: a small residual
bias matters more than the marginal $\sigma$-reduction once $\sigma$
is already low; spaced retrieval matters more once $|t|, \sigma$ are
below tolerance.

\subsection{Tractable approximations to the Bellman equation}
\label{sec:approximations}

Exact solution of Eq.~\ref{eq:bellman} is intractable: the belief
simplex is infinite-dimensional, the action space contains
$\sim$10$^4$ items, and no closed-form value function exists on
continuous $(\sigma, t)$. We outline three approximation tiers, in
increasing order of effort and fidelity. The proposal in
Secs.~\ref{sec:threeproblems}--\ref{sec:proposal} is tier 2; tier 1
is the natural ablation baseline; tier 3 is the upgrade path once
we have a fitted dynamics model from pilot data.

\subsubsection{Tier 1 --- One-step greedy (myopic) policy}
\label{sec:tier1}

The simplest approximation drops the discounted-future-value term in
Eq.~\ref{eq:bellman} entirely and optimises only the expected
immediate reward at the current belief:
\begin{equation}
  \pi_{\mathrm{greedy}}(b) \;=\;
    \arg\max_{s \in \mathcal{A}}\;
    \mathbb{E}\!\bigl[\,R(\theta_k, \theta_{k+1}) \;\big|\; b, s\,\bigr].
  \label{eq:greedy}
\end{equation}
The output $\pi_{\mathrm{greedy}}(b)$ is the single stimulus
$s^\star$ that wins this argmax: the policy is a function from belief
states to actions, and applying it at $b$ returns one action. This
reduces the POMDP to a contextual bandit on $b$.

\paragraph{Where past responses enter.}
All previous responses $y_1, \dots, y_k$ enter only through the
current belief $b_k$, which is the Bayes-filter posterior over the
learner's hidden state given the entire history
$\mathcal{H}_k = \{s_j, y_j, \mathrm{fb}_j\}_{j \le k}$
(Sec.~\ref{sec:belief}). In the SMC representation the particles
$\{(\theta^{(i)}, w^{(i)})\}_{i=1}^N$ \emph{are} that compressed
history: each response $y_j$ reweights particles via the likelihood
$Z(y_j \mid \theta^{(i)}, s_j)$, and $T$ propagates them forward
(Eqs.~\ref{eq:bayes}--\ref{eq:predict}). The greedy policy itself is
memoryless --- the only input it reads is $b_k$.

\paragraph{Evaluating the expectation.}
Expanding the conditioning in Eq.~\ref{eq:greedy} over the random
hidden state $\theta \sim b$, the random response
$y \sim Z(\cdot \mid \theta, s)$, and the random next state
$\theta' \sim T(\cdot \mid \theta, s, y, y^\star)$:
\begin{equation}
  \mathbb{E}[R \mid b, s]
  \;=\;
  \int b(\theta) \,\sum_{y \in \{0, 1\}}
    Z(y \mid \theta, s)\;
    \mathbb{E}_{\theta' \sim T(\cdot \mid \theta, s, y, y^\star)}
      \!\bigl[R(\theta, \theta')\bigr]
    \,\mathrm{d}\theta,
  \label{eq:greedy-expectation}
\end{equation}
where $y^\star = y^\star(s)$ is the ground-truth label of the
candidate stimulus (the trainer picks from a labelled bank, so
$y^\star$ is known given $s$). We approximate the inner expectation
by the deterministic mean of $T$ (i.e.\ drop the Gaussian process
noise, which integrates out exactly for the linear $\sigma$-term of
$R$ and up to a small Jensen correction for the $|t|$-term):
\[
  \mathbb{E}_{\theta' \sim T}[R(\theta, \theta')]
  \;\approx\;
  R\bigl(\theta,\, \bar\theta'(\theta, s, y, y^\star)\bigr),
  \qquad
  \bar\theta'(\theta, s, y, y^\star) \;\equiv\;
  \mathbb{E}_T[\theta' \mid \theta, s, y, y^\star].
\]
With the SMC cloud representing $b_k$, the outer integral becomes a
weighted sum and the quantity we maximise is
\begin{equation}
  Q(s) \;\equiv\; \widehat{\mathbb{E}}[R \mid b_k, s]
  \;=\;
  \sum_{i=1}^N w^{(i)}\!
  \sum_{y \in \{0, 1\}}
    \underbrace{Z(y \mid \theta^{(i)}, s)}_{p_y^{(i)}}
    \;
    R\bigl(\theta^{(i)},\,\bar\theta'^{(i, y)}\bigr),
  \label{eq:Qgreedy}
\end{equation}
where $\bar\theta'^{(i, y)} = \bar\theta'(\theta^{(i)}, s, y,
y^\star)$.

Under the soft-form choices we adopted (Sec.~\ref{sec:gt}: $\delta_k
= \hat y_k - y^\star$, independent of the response $y_k$; and
Sec.~\ref{sec:gsig}: $g_\sigma$ has no correctness term),
$\bar\theta'^{(i, y)}$ does \emph{not} actually depend on $y$, so
the inner sum over $y$ collapses to a single term per particle. The
general form in Eq.~\ref{eq:Qgreedy} is the one to use for the
hard-error ablation (Eq.~\ref{eq:deltahard}), where $y$ does enter.

\smallskip
\noindent\textbf{Algorithm (one-step greedy).}\\
\noindent\emph{Inputs:} current belief $b_k =
\{(\theta^{(i)}, w^{(i)})\}_{i=1}^N$ (which already encodes all past
responses $y_1, \dots, y_k$ via the Bayes filter); candidate set
$\mathcal{A}_k$ with known labels.
\begin{enumerate}[leftmargin=*,itemsep=1pt]
  \item For each candidate $(s, \mu) \in \mathcal{A}_k$ with label
    $y^\star = y^\star(s)$:
    \begin{enumerate}[leftmargin=1.5em,itemsep=0pt]
      \item For each particle $\theta^{(i)} =
        (\sigma^{(i)}, t^{(i)}, \phi^{(i)})$, compute the predicted
        ``yes'' probability
        \[
          p_1^{(i)} \;=\; Z(1 \mid \theta^{(i)}, s)
          \;=\; (1 - 2\lambda)\,\Phi\!\bigl((s - t^{(i)})/\sigma^{(i)}\bigr)
          + \lambda
          \qquad (\text{Eq.~\ref{eq:obs}})
        \]
        and the soft prediction error $\delta^{(i)} = p_1^{(i)} -
        y^\star$ (Eq.~\ref{eq:delta}).
      \item Compute the deterministic next state via the means of
        $T_t$ and $T_\sigma$:
        \[
          \bar t'^{(i)} = t^{(i)} + \alpha_t \delta^{(i)},
          \qquad
          \log\bar\sigma'^{(i)} = \log\sigma^{(i)} + \alpha_\sigma
            g_\sigma(s, \theta^{(i)}, \mathrm{fb}=1)
          \qquad (\text{Eqs.~\ref{eq:tdyn},~\ref{eq:sdyn}}).
        \]
      \item Compute the per-particle reward $R^{(i)} =
        \beta_t(|t^{(i)}| - |\bar t'^{(i)}|) +
        \beta_\sigma(\sigma^{(i)} - \bar\sigma'^{(i)}) +
        \beta_r\,\mathrm{ret}^{(i)}$ (Eq.~\ref{eq:reward}).
    \end{enumerate}
  \item Aggregate: $Q(s) = \sum_{i=1}^N w^{(i)} R^{(i)}$ (the inner
    sum over $y$ in Eq.~\ref{eq:Qgreedy} has collapsed under the
    soft form; see remark above).
  \item Select $s^\star = \arg\max_{s \in \mathcal{A}_k} Q(s)$ and
    present that stimulus to the learner.
  \item After the learner responds with $y_k$ and feedback reveals
    $y_k^\star$, run the Bayes filter
    (Eqs.~\ref{eq:bayes}--\ref{eq:predict}) to update $b_k \to
    b_{k+1}$ and loop.
\end{enumerate}
\noindent\emph{Cost:} $\mathcal{O}(|\mathcal{A}_k| \cdot N)$ per
trial; trivially parallel across candidates.

\subsubsection{Tier 2 --- Mode-conditional greedy}
\label{sec:tier2}

Replace the single $\arg\max$ of Eq.~\ref{eq:greedy} with three
mode-specific selection rules, gated by cut-offs on posterior
summaries. This loses some expressivity vs.\ Eq.~\ref{eq:greedy}
but gains interpretability and avoids reward-weight ($\beta_t,
\beta_\sigma, \beta_r$) sensitivity, since the mode itself decides
which objective dominates.

\smallskip
\noindent\textbf{Algorithm (mode-conditional greedy).}\\
\noindent\emph{Inputs:} belief $b_k$; cut-offs $t^\star,
\sigma^\star$; posterior-SD multiplier $c$; candidate set
$\mathcal{A}_k$.
\begin{enumerate}[leftmargin=*,itemsep=1pt]
  \item Compute posterior means $\hat t_k = \mathbb{E}_b[t]$,
    $\hat\sigma_k = \mathbb{E}_b[\sigma]$, and $\mathrm{SD}_b(t)$.
  \item Choose mode:
    \begin{itemize}[leftmargin=1.5em,itemsep=0pt]
      \item if $|\hat t_k| > \max(t^\star,\, c\cdot\mathrm{SD}_b(t))$:
        mode $\leftarrow$ \emph{bias-correction};
      \item else if $\hat\sigma_k > \sigma^\star$ and the learning
        curve still has non-negligible slope: mode $\leftarrow$
        \emph{skill-building};
      \item else: mode $\leftarrow$ \emph{retention}.
    \end{itemize}
  \item Sample $s_k$ according to mode:
    \begin{itemize}[leftmargin=1.5em,itemsep=0pt]
      \item \emph{bias-correction}: $s_k$ near $\hat t_k$, sign
        balanced across recent bias-mode trials.
      \item \emph{skill-building}: $|s_k - \hat t_k| \approx
        1.04\,\hat\sigma_k$ (Eq.~\ref{eq:eightyfive}); rotate $\mu$
        through morphology sub-types for diversity.
      \item \emph{retention}: draw from previously-mastered bins on
        a spaced (SM-2-style) schedule.
    \end{itemize}
  \item Deliver veridical trial-wise feedback with feature-based
    rationale.
\end{enumerate}
This is the policy elaborated in
Secs.~\ref{sec:threeproblems}--\ref{sec:proposal}.

\subsubsection{Tier 3 --- Short-horizon Monte Carlo rollout}
\label{sec:tier3}

Approximate the entire bracketed quantity inside the $\arg\max$ of
Eq.~\ref{eq:bellman} --- expected immediate reward plus
$\gamma\,V_{k+1}^\star(b'(b_k, s, y))$ --- by averaging the
discounted returns of $L$ simulated rollouts of length
$H \le K - k$ trials under a base policy $\pi_0$ (taken to be the
tier-2 rule). Each rollout takes the candidate $s$ at trial $k$,
draws a particle from $b_k$, and simulates the next $H$ trials
forward using the fitted dynamics, recording the cumulative
discounted reward. Truncating at depth $H$ approximates
$V_{k+1}^\star$ by ignoring rewards beyond $k + H$, which is the
standard receding-horizon approximation; bias decays as
$\gamma^H$ when $\gamma < 1$, and as the tail-reward magnitude
beyond $k + H$ when $\gamma = 1$.

\smallskip
\noindent\textbf{Algorithm (MC rollout).}\\
\noindent\emph{Inputs:} current trial index $k$ and belief $b_k$;
rollout depth $H = \min(H_{\max}, K - k)$; sample count $L$; base
policy $\pi_0$; candidate set $\mathcal{A}_k$.
\begin{enumerate}[leftmargin=*,itemsep=1pt]
  \item For each candidate $s \in \mathcal{A}_k$ with label
    $y^\star = y^\star(s)$:
    \begin{enumerate}[leftmargin=1.5em,itemsep=0pt]
      \item For rollout $l = 1, \dots, L$:
        \begin{itemize}[leftmargin=1.5em,itemsep=0pt]
          \item \emph{Trial $k$ (the candidate):} sample
            $\theta_k^{(l)} \sim b_k$;
            $y_k^{(l)} \sim Z(\cdot \mid \theta_k^{(l)}, s)$;
            $\theta_{k+1}^{(l)} \sim T(\cdot \mid \theta_k^{(l)},
            s, y_k^{(l)}, y^\star)$; record $R_k^{(l)} =
            R(\theta_k^{(l)}, \theta_{k+1}^{(l)})$.
          \item Compute belief $b_{k+1}^{(l)}$ from $b_k$, $s$,
            $y_k^{(l)}$ via Eqs.~\ref{eq:bayes}--\ref{eq:predict}.
          \item \emph{Trials $k+1, \dots, k+H-1$ (base policy):}
            for $j = k+1, \dots, k+H-1$ choose $s_j^{(l)} =
            \pi_0(b_j^{(l)})$, look up $y_j^\star$, sample
            $y_j^{(l)}, \theta_{j+1}^{(l)}$ as above, record
            $R_j^{(l)}$, update $b_{j+1}^{(l)}$.
          \item Return $G^{(l)} = \sum_{j = 0}^{H-1} \gamma^j
            R_{k+j}^{(l)}$.
        \end{itemize}
      \item Estimate $\hat Q_k(s) = \tfrac{1}{L} \sum_{l=1}^L
        G^{(l)}$, the rollout estimate of the bracketed quantity
        in Eq.~\ref{eq:bellman} at trial $k$.
    \end{enumerate}
  \item Select $s^\star = \arg\max_{s \in \mathcal{A}_k} \hat
    Q_k(s)$ and present it.
  \item After observing the actual $y_k$, run the Bayes filter
    to update $b_k \to b_{k+1}$ and loop.
\end{enumerate}
\noindent\emph{Cost:} $\mathcal{O}(|\mathcal{A}_k| \cdot L \cdot H
\cdot N)$ per trial; embarrassingly parallel across candidates and
rollouts. Closes the gap to a full POMDP solve when needed for
Phase 3.

\section{Three training sub-problems, three policies}
\label{sec:threeproblems}

The single most important conceptual point — one the existing
literature has somewhat muddled — is that \emph{what the trainer should
do depends on what is wrong with the learner}. There are three
distinct objectives, each with a different ``optimal-stimulus'' rule.

\paragraph{Bias correction (reduce $|t|$).}
\textit{Fast, feedback-driven.} The prior work and recognition-memory
literature [ref] shows that $t$ is adjusted
implicitly and rapidly in response to outcome feedback, even without
the learner's awareness. The right intervention is to train at stimuli
near the learner's current $\hat t_k$, with balanced sign, so that
feedback is maximally informative about criterion placement. An
unbalanced base rate would induce an opposite bias by accident.

\paragraph{Skill acquisition (reduce $\sigma$).}
\textit{Slow, many trials, classically stimulus-specific.} This is
where the 85\,\% rule applies directly. The right intervention is to
train at intermediate difficulty calibrated to the learner's
\emph{current} psychometric function. For the unit-variance probit
with lapse $\lambda \approx 0.025$ used in ENGINE, the $|s-\hat t_k|$
that produces $\approx 85\%$ correct is
\begin{equation}
  |s - \hat t_k| \;\approx\; 1.04\,\hat\sigma_k.
  \label{eq:eightyfive}
\end{equation}
Diversity across morphologies matters here — Sec.~\ref{sec:policy}
returns to this.

\paragraph{Retention and fluency.}
\textit{Spaced, interleaved.} Spaced retrieval practice with feedback
has strong support in education [ref]
and especially in pattern-recognition contexts (radiology and
histopathology). The right intervention is interleaved return-visits
to previously mastered sub-categories on a roughly exponential spacing
schedule (SM-2 / Anki-family).

These three objectives demand different policies. A single-objective
adaptive algorithm (QUEST$+$ for estimation, or a bare 85\,\%-rule
staircase) will be wrong roughly two-thirds of the time.

\section{A concrete proposal}
\label{sec:proposal}

The proposal is deliberately a small, interpretable policy that
exploits the existing posterior rather than a full POMDP solve.

\subsection{Learner-state model}
Augment the existing $(\sigma,t)$ generative model with the
learning-dynamics layer of Eqs.~\eqref{eq:tdyn}--\eqref{eq:sigmaschedule}.
Estimate $\tau_\sigma$, $\sigma_\infty$, $\alpha_t$ per learner from
their accumulated trial history; use empirical-Bayes priors derived
from the pooled pilot cohort.

\subsection{Item-selection policy}
\label{sec:policy}
On trial $k$, with posterior means $\hat\sigma_k, \hat t_k$ in hand:

\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item \textbf{Choose the mode.}
    \begin{itemize}[leftmargin=*,itemsep=1pt]
      \item \emph{Bias-correction mode} if $|\hat t_k|$ is large
            relative to its posterior SD \emph{and} above a
            tolerance $t^\star$.
      \item \emph{Skill-building mode} otherwise, as long as
            $\hat\sigma_k$ is still decreasing (non-negligible
            learning-curve slope from Eq.~\ref{eq:sigmaschedule}) and
            above the target $\sigma^\star$.
      \item \emph{Retention mode} otherwise (skill plateaued, bias
            within tolerance).
    \end{itemize}
  \item \textbf{Sample $s_k$ accordingly.}
    \begin{itemize}[leftmargin=*,itemsep=1pt]
      \item Bias mode: $s_k$ near $\hat t_k$, with balanced sign.
      \item Skill mode: $|s_k - \hat t_k| \approx 1.04\,\hat\sigma_k$
            (Eq.~\ref{eq:eightyfive}); rotate through morphology
            sub-types for diversity.
      \item Retention mode: sample from previously seen difficulty
            bins on a spaced schedule.
    \end{itemize}
  \item \textbf{Always give trial-wise veridical feedback.} The
        correct answer plus a brief feature-based rationale
        (annotations on the trace itself, in the style of the Kellman
        perceptual-learning methods). The feature explanation turns the trial into a
        \emph{learning-to-see} episode rather than a pure
        reinforcement episode.
\end{enumerate}

This is not a full POMDP solve but it exploits all the structure the
generative model provides — more than the standard adaptive-learning
literature does.

\subsection{Stopping / mastery criterion}
Retire the learner from a task — or graduate them to
maintenance-spacing — when, jointly,
\begin{equation}
  \mathrm{SD}(\sigma) \le \delta_\sigma,\qquad
  \hat\sigma \le \sigma^\star,\qquad
  |\hat t| \le t^\star,
\end{equation}
for pre-specified targets. This is a Bayesian upgrade of the
Kellman/Massey ``running-$d'$ window'' rule.

\subsection{Suggested phased pipeline}
\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item \textbf{Phase 1 — Calibration.} 10--20 trials of the existing
        QUEST$+$ design to seed $(\hat\sigma_0, \hat t_0)$.
  \item \textbf{Phase 2 — Adaptive training.} At each trial, update
        the posterior over $(\sigma,t)$, score candidate cases against
        Eq.~\ref{eq:reward}, select per Sec.~\ref{sec:policy}.
  \item \textbf{Phase 3 — Learn the learning model.} From accumulated
        trial-by-trial data, fit a hierarchical model of how different
        cases change $(\sigma,t)$ (identifying which cases reduce bias
        fastest, which improve discrimination, and any sub-type-
        specific learning effects). Use the fitted dynamics to
        upgrade the policy from heuristic to contextual-bandit, and
        eventually to a planned POMDP rollout.
\end{enumerate}

\section{Caveats}
\label{sec:caveats}

\paragraph{Stationarity assumption breaks during training.} The
existing ENGINE posterior-update implicitly assumes $(\sigma,t)$ are
fixed; during training they move. Three options: (a) reset the
posterior per session and re-estimate from scratch; (b) explicitly
model the dynamics and \emph{filter} (sequential Bayesian state-space
filtering) rather than just accumulate; (c) assume within-session
stationarity and reset between sessions. Option (b) is the most
principled.

\paragraph{Biased feedback is ethically off-limits.} Herzog showed
that biased trial-wise feedback shifts the perceived identity of a
stimulus — explicitly \emph{not} what we want in a skill
trainer, where the learner must internalise the ground-truth label
distribution. \textbf{Use veridical feedback only.} Influence $t$ via
the distribution of stimuli sampled and the presented base rate, never
by lying about correctness.

\paragraph{Identifiability under joint motion.} If the learner
improves quickly, a drop in error rate could reflect either $\sigma$
shrinking or $t$ moving toward zero. The existing framework
disentangles these via the stimulus-level response pattern, but it is
worth verifying disentanglement still holds when both parameters move
simultaneously. A small simulation study paralleling the existing
measurement-side identifiability work would settle this.

\paragraph{Near-threshold is optimal for measurement, not learning.}
A trainer that simply re-uses the measurement-optimal stimulus
placement will sit the learner at near-50\,\% performance — useful for
estimating $t$, demoralising for learning. The 85\,\% rule and the
Sec.~\ref{sec:policy} mode-switch exist precisely to avoid this trap.

\section{Next steps}
\label{sec:next}

\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item Extend the existing ENGINE simulation harness with the
        learning-dynamics model
        (Eqs.~\ref{eq:tdyn}--\ref{eq:sigmaschedule}); verify joint
        identifiability of $(\sigma_k, t_k)$ trajectories during
        simulated training.
  \item Benchmark the three-mode policy against simpler baselines —
        method-of-constant-stimuli with feedback, a pure 85\,\%-rule
        staircase, a pure threshold-tracking adaptive scheme — on
        learning-curve metrics:
        \begin{itemize}[leftmargin=*,itemsep=1pt]
          \item trials to reach target $\sigma^\star$,
          \item trials to reach $|t|\le t^\star$,
          \item post-test performance at held-out difficulty bins,
          \item retention at $\Delta t = 7$ and $30$ days.
        \end{itemize}
  \item Build a sibling app to ENGINE — same posterior, same
        engine — that runs the trainer policy and delivers
        feature-annotated feedback. Target form-factor: $\sim
        15$~minutes per day, web or desktop bundle, same distribution
        story as the current pilot.
  \item Validate against real expert data: do measured $(\sigma,t)$
        trajectories under the trainer actually follow the shapes
        Eqs.~\ref{eq:tdyn}--\ref{eq:sigmaschedule} predict? Re-fit and
        iterate.
\end{enumerate}

\section*{Source documents}
This note synthesises \texttt{learning-theory1.md} and
\texttt{learning-theory2.md} from \texttt{the main repo}.
Treat that pair as the primary source; this LaTeX version exists to be
a single shareable design memo with consistent notation and
mathematics.

\begin{thebibliography}{99}\small

\bibitem[Aberg \& Herzog(2012)]{aberg2012feedback}
Aberg, K.~C. and Herzog, M.~H. (2012).
``Different types of feedback change decision criterion and sensitivity differently in perceptual learning,'' \emph{Journal of Vision}.

\bibitem[Carpenter et al.(2022)]{[ref]}
Carpenter, S.~K. et~al. (2022).
``The science of effective learning with spacing and retrieval practice,'' \emph{Nature Reviews Psychology}.

\bibitem[Friedman et al.(1968)]{friedman1968criterion}
Friedman, M.~P., Carterette, E.~C., and Anderson, N.~H. (1968).
``Long-term probability learning with a random schedule of reinforcement,'' \emph{Journal of Experimental Psychology}.

\bibitem[Han \& Dobbins(2009)]{[ref]}
Han, S. and Dobbins, I.~G. (2009).
``Examining recognition criterion rigidity during testing using a biased-feedback technique,'' \emph{Memory \& Cognition}.

\bibitem[Kac(1962)]{kac1962criterion}
Kac, M. (1962).
``A note on learning signal detection,'' \emph{IRE Transactions on Information Theory}.

\bibitem[Kellman et al.(2024)]{kellman2024palm}
Kellman, P.~J., Jacoby, S., Massey, C.~M., and Krasne, S. (2024).
``Connecting Adaptive Perceptual Learning and Signal Detection Theory in Skin Cancer Screening.''

\bibitem[Mettler et al.(2011)]{mettler2011arts}
Mettler, E., Massey, C.~M., and Kellman, P.~J. (2011, 2016).
``A comparison of adaptive and fixed schedules of practice,'' \emph{Journal of Experimental Psychology: General}.

\bibitem[Rafferty et al.(2016)]{rafferty2016pomdp}
Rafferty, A.~N., Brunskill, E., Griffiths, T.~L., and Shafto, P. (2016).
``Faster teaching via POMDP planning,'' \emph{Cognitive Science}.

\bibitem[St\"uttgen et al.(2011)]{stuttgen2011trial}
St\"uttgen, M.~C., Yildiz, A., and G\"unt\"urk\"un, O. (2011, 2013).
Trial-by-trial criterion learning in signal detection.

\bibitem[Watson \& Pelli(1983)]{watson1983quest}
Watson, A.~B. and Pelli, D.~G. (1983).
``QUEST: a Bayesian adaptive psychometric method,'' \emph{Perception \& Psychophysics}.

\bibitem[Watson(2017)]{watson2017questplus}
Watson, A.~B. (2017).
``QUEST$+$: A general multidimensional Bayesian adaptive psychometric method,'' \emph{Journal of Vision}.

\bibitem[Wilson et al.(2019)]{wilson202085percent}
Wilson, R.~C., Shenhav, A., Straccia, M., and Cohen, J.~D. (2019).
``The Eighty Five Percent Rule for optimal learning,'' \emph{Nature Communications}.

\end{thebibliography}

\end{document}

