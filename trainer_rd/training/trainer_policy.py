"""Step 5 — Tier-2 mode-conditional policy + cross-task scheduler.

The trainer's item-selection policy (plan §8.2/§10), made K-task-real (F8/D3).
Three within-task modes (plan §7), a deficiency-weighted interleave scheduler
across tasks, AD6-style graduation on the filtered posterior, and the review
guards from the design review:

  * bias-correction : items near t̂_k, RUNNING LABEL BALANCE enforced (F6b);
  * skill-building  : |s − t̂_k| ≈ 1.077·σ̂_k (D6), sides alternated, drawn from
    feedback-safe (coherent) items so the label matches the signal side (F7);
  * retention       : spaced review of mastered difficulty bins behind a
    decay-model interface (SM-2 default), reward eligibility-gated to DUE bins (F6a).

Mastery (F14/F5): task k graduates when the filtered posterior clears the AD6
gate against ℓ*_k AND |t̂_k| ≤ t* AND the skill SD has contracted past the
information floor.

Deficiency score (OQ1, resolved here): per unmastered task,
    deficiency_k = max( 1 − π_k ,  clip(|t̂_k|/t* − 1, 0, 1) )
the worse of the skill pass-mass shortfall and the bias over-tolerance. The
scheduler trains the argmax, with a consecutive-same-task cap (R7, mirrors the
production controller's variety rule) and retention insertions for due tasks.

Telemetry: every trial emits a TrialRecord (Step 1 schema) carrying mode, RT,
absolute epoch (D9), and a fatigue/lapse-probe flag on very-easy items (F15).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from training.bridge_conventions import SKILL_MODE_MULTIPLIER, engine_to_plan
from training.training_seed import TrialRecord

BIAS, SKILL, RETENTION = "bias", "skill", "retention"

# Bias tolerance t* (plan coords). M17 (OQ6 RESOLVED, user-ratified): the
# product goal is ZERO bias; the achievable zero is the stationary
# fluctuation band derived in `bias_stationary_sd` below, and the DEFAULT
# gate is now that derived band (ModeThresholds.t_star=None). The 0.30
# constant remains only as the scheduler's ranking fallback and for
# historical comparisons.
DEFAULT_T_STAR = 0.30
# Very-easy lapse-probe placement: |s−t̂|/σ̂ this large ⇒ accuracy ≈ 1−λ (F15).
LAPSE_PROBE_MULT = 3.0


def expected_skill_weight(s, s_sd, sig_hat, t_hat, *, side=1,
                          m=SKILL_MODE_MULTIPLIER, rho=0.5):
    """E[w(d_real)] — the learner's expected skill-learning weight when item s
    carries stimulus uncertainty s_sd (F20-aware placement).

    d_real = (s_real − t̂)/σ̂ with s_real ~ N(s, s_sd²); the Gaussian weight
    smeared by Gaussian stimulus noise has the closed form
        E[w] = ρ/√(ρ²+τ²) · exp(−(μ − side·m)² / (2(ρ²+τ²))),  τ = s_sd/σ̂
    (single placement branch; side ∈ {−1, +1} picks which 85%-point).
    Maximizing this instead of minimizing |s − target| makes the policy prefer
    LOW-s_sd items near the target — under real-bank noise (s_sd ~ 0.6–1.0)
    naive nearest-target placement is blurred into ineffectiveness."""
    mu = (np.asarray(s, dtype=np.float64) - t_hat) / sig_hat
    v = rho * rho + (np.asarray(s_sd, dtype=np.float64) / sig_hat) ** 2
    return rho / np.sqrt(v) * np.exp(-(mu - side * m) ** 2 / (2.0 * v))


def bias_correction_score(filt, cands):
    """Bias-mode CORRECTIVE item score (M10): the expected one-step
    |t|-REDUCTION over the particle cloud — the β_t term of the Tier-1 greedy
    Q, already F20-attenuated. The drift-optimal corrective items sit between
    the TRUE boundary and the criterion (the zone the learner misclassifies),
    where |δ| → 1. The caller's running label balance (F6b) bounds the
    base-rate envelope. NOTE: these items are information-POOR about t (the
    response is near-deterministic), so pure-corrective serving lets t̂ go
    stale while true t moves — overshoot past 0 and a flapping mastery gate
    (dual-control failure, observed M10). Bias mode therefore ALTERNATES this
    score with `bias_probe_score`."""
    from training.trainer_greedy import RewardWeights, _expected_reward
    return _expected_reward(filt, cands.s_mean,
                            cands.y_star.astype(np.float64),
                            RewardWeights(beta_t=1.0, beta_sigma=0.0,
                                          beta_r=0.0),
                            s_sd=cands.s_sd)


def expected_progress_score(filt, cands, *, weights=None, bar_ell=None):
    """M24 (F84): stratum-aware posterior expected one-step progress Q(s)
    for every candidate — the plan's Eq. (greedy-expectation) computed under
    the HONEST posterior instead of a plug-in point estimate.

    The M22 skill mode ranks by E[w] (the difficulty-appropriateness weight)
    at the plug-in (σ̂, boundary), stacking three approximations the
    seven-profile data expose (delivered training value w_true averaged 0.48
    of ideal; the F80-lag sessions fell to 0.15–0.17):
      (i)   point estimate instead of the posterior — a wide/bimodal belief
            (fresh onboarding; post-boundary-shift) is served at its mean;
      (ii)  E[w] instead of E[w·gap] — items are not credited by how much
            the mass they train can actually still improve;
      (iii) mixture structure ignored — ceiling strata, weights, and the
            static stratum never enter placement.
    This scorer removes all three for the σ channel — the same treatment
    the bias channel received at D15 (`bias_correction_score` IS the β_t
    term). For a SigmaInfMixtureFilter, Q(s) = Σ_j ω_j·Q_j(s) with each
    stratum scored under ITS OWN dynamics (σ_∞^(j), per-particle learn
    flags); static-rule strata contribute exactly 0 (their transition is
    the identity ⇒ R ≡ 0 — the pooled-cloud shortcut wrongly credits that
    mass, F84-ii). For a plain TaskFilter it reduces to the tier-1 Q.
    bar_ell: see `_expected_reward` (F85 bar-referenced credit)."""
    from training.trainer_greedy import RewardWeights, _expected_reward
    w = weights or RewardWeights(beta_t=0.0, beta_sigma=1.0, beta_r=0.0)
    y = cands.y_star.astype(np.float64)
    strata = getattr(filt, "strata", None)
    if strata is None:
        if filt.p.rule == "static":
            return np.zeros(len(cands))
        return _expected_reward(filt, cands.s_mean, y, w, s_sd=cands.s_sd,
                                bar_ell=bar_ell)
    om = filt.weights()
    q = np.zeros(len(cands))
    for om_j, f_j in zip(om, strata):
        if f_j.p.rule == "static" or om_j < 1e-9:
            continue
        q += om_j * _expected_reward(f_j, cands.s_mean, y, w,
                                     s_sd=cands.s_sd, bar_ell=bar_ell)
    return q


def expected_progress_rate(filt, ell_star, t_star, *, s_sd_nominal=0.0,
                           rho=0.5, beta_t=1.0, beta_sigma=1.0):
    """M24 (F85): task-level EXPECTED PROGRESS PER TRIAL toward this task's
    own bar, at an IDEAL placement — the principled successor to worst-first
    deficiency (the F77 open item). Per stratum j and particle i:

        EP_σ = α_σ · w̄_i · (ℓ_∞^(j) − ℓ_i)⁺ · 1[ℓ_i < ℓ*]
        EP_t = min(α_t, (|t_i| − t*)⁺)

    with w̄_i = ρ/√(ρ² + (s̄_sd/σ_i)²) the noise-smeared weight at the 85%
    optimum (placement feasibility caveat: a pool whose easiest item is far
    inside a huge-σ learner's chance zone cannot realize w̄ — EP is an upper
    bound there, which errs toward SERVING struggling learners; the
    within-task scorer uses real items). EP_k = Σ_j ω_j 1[rule_j ≠ static]
    Σ_i w_ij (β_σ·EP_σ + β_t·EP_t).

    Reading: how much bar-ward progress (in latent units) one trial of this
    task is expected to buy, under the current belief. It SUBSUMES the
    D36 trainability discount from first principles — static-stratum and
    at-ceiling mass contribute exactly 0 (B's plateau self-deprioritizes),
    a near-mastery task's above-bar mass contributes 0 (finishing is the
    finish-first phase's job — measurement, not learning), and a wide
    onboarding posterior keeps EP high through its trainable tail. β's are
    the plan-§reward designer weights (both latent-unit scales)."""
    strata = getattr(filt, "strata", None)
    om_strata = (zip(filt.weights(), strata) if strata is not None
                 else [(1.0, filt)])
    ep = 0.0
    for om_j, f_j in om_strata:
        p = f_j.p
        if p.rule == "static" or om_j < 1e-9:
            continue
        sig = np.exp(-np.clip(f_j.ell, -40.0, 40.0))
        wbar = rho / np.sqrt(rho ** 2 + (float(s_sd_nominal) / sig) ** 2)
        gap = np.maximum(-np.log(p.sigma_inf) - f_j.ell, 0.0)
        below = f_j.ell < float(ell_star)
        ep_sig = p.alpha_sigma * wbar * gap * below
        ep_t = np.minimum(p.alpha_t,
                          np.maximum(np.abs(f_j.theta) - float(t_star), 0.0))
        ep += om_j * float((f_j.w * f_j.learn
                            * (beta_sigma * ep_sig + beta_t * ep_t)).sum())
    return float(ep)


def bias_probe_score(s, s_sd, sig_hat, t_hat):
    """Bias-mode PROBE item score: per-trial information about the criterion
    under stimulus noise, φ(z̃)/√(1+(s_sd/σ̂)²) with z̃ the attenuated
    normalized distance from t̂. Maximal for precise items near the criterion —
    the measurement half of the bias mode's dual-control alternation."""
    att = np.sqrt(1.0 + (np.asarray(s_sd, dtype=np.float64) / sig_hat) ** 2)
    z = (np.asarray(s, dtype=np.float64) - t_hat) / sig_hat / att
    return np.exp(-0.5 * z * z) / att


def cert_probe_score(s, s_sd, sigma_star, t_hat, lapse=0.025):
    """M15 certification-probe placement: per-trial Fisher information about
    ℓ evaluated AT THE CUT STATE (σ = σ*, t = t̂), s_sd-attenuated.

    Training placement targets difficulty against the BELIEVED σ̂, which
    tracks the belief toward its σ_∞ attractor — so it is least informative
    about the ℓ-vs-ℓ* contrast exactly when the belief has run ahead of
    truth (F28/F31: the served-item accuracy gap between σ = 1.25σ* and σ*
    is ~0.02 under bank s_sd). Anchoring the information target at the BAR
    breaks that self-confirmation loop.

    With z̃ = (s − t̂)/σ* / att, att² = 1 + (s_sd/σ*)² and ∂z̃/∂ℓ = z̃/att²:
        I_ℓ(s) = [(1−2λ)·φ(z̃)·z̃/att²]² / (p̃(1−p̃)),
    maximized by LOW-s_sd items at z̃ ≈ ±1.35 under λ=0.025 (ℓ is a
    SCALE-type parameter — the extra z̃ factor pushes the optimum past the
    location-parameter z̃=0 and the no-lapse scale optimum 1.57 is pulled in
    by the lapse floor), i.e. |s − t̂| ≈ 1.35·σ*·att."""
    from scipy.stats import norm as _norm
    att2 = 1.0 + (np.asarray(s_sd, dtype=np.float64) / sigma_star) ** 2
    z = (np.asarray(s, dtype=np.float64) - t_hat) / sigma_star / np.sqrt(att2)
    p = lapse + (1.0 - 2.0 * lapse) * _norm.cdf(z)
    dp = (1.0 - 2.0 * lapse) * _norm.pdf(z) * np.abs(z) / att2
    return dp * dp / (p * (1.0 - p))


class EProcessGate:
    """M15 anytime-valid below-bar refutation monitor (the D19 evidence-gate
    upgrade; MA-7's replacement for ad-hoc repeated-look corrections).

    Watches certification-probe outcomes c_i ∈ {0,1} against a0_i = the
    AT-BAR accuracy H0 predicts for the served probe (probit-lapse at σ = σ*,
    t = t̂). Maintains a mixture of Hoeffding e-processes
        e_λ(k) = exp( λ·Σ_i (a0_i − c_i) − k·λ²/8 ),   E(k) = mean_λ e_λ(k).
    Increments (a0−c) ∈ (−1,1) are 1/2-sub-Gaussian, so each e_λ is a
    nonnegative supermartingale under every H0 with E[c_i|F_{i−1}] ≥ a0_i,
    and Ville's inequality gives P(sup_k E(k) ≥ 1/α | H0) ≤ α at EVERY
    stopping time — honest type-I control under continuous monitoring, which
    the D16 z=1.645 trailing-window gate only approximates. `fired` ⇒ the
    learner performs BELOW the bar on probes ⇒ block graduation (protective
    direction: a false fire, prob ≤ α, only delays declaration).

    λ grid: the optimal bet against an accuracy deficit δ is λ* = 4δ with
    growth rate 2δ² per probe; {0.05, 0.1, 0.2, 0.4, 0.8} covers
    δ ∈ [0.0125, 0.2]. Careless-grade deficits (δ ≈ 0.19) fire in ~30–60
    probes; near-bar deficits (δ ≈ 0.05) need ~hundreds — that is the
    Bernoulli information bound (n ≳ δ⁻²), not test inefficiency; the
    σ_∞-mixture handles the near-bar case through the full likelihood.
    The uniform λ-mixture costs ≤ log 5 in log-e versus the oracle λ."""

    LAMBDAS = (0.05, 0.1, 0.2, 0.4, 0.8)

    def __init__(self, alpha=0.05):
        self.alpha = float(alpha)
        self.log_e = np.zeros(len(self.LAMBDAS))
        self.n = 0
        self.fired = False
        self.e_now = 1.0

    def update(self, correct, a0):
        """Record one probe outcome; returns the current mixture e-value."""
        lam = np.asarray(self.LAMBDAS)
        self.log_e += lam * (float(a0) - float(bool(correct))) - lam * lam / 8.0
        self.n += 1
        e = float(np.exp(self.log_e - np.log(len(lam))).sum())
        self.e_now = e
        if e >= 1.0 / self.alpha:
            self.fired = True
        return e

    @property
    def elevated(self):
        """Current mixture e ≥ 1/α — the declaration-BLOCKING predicate (M16).
        Unlike the sticky `fired` (the Ville-valid sup-crossing test, kept for
        diagnostics), `elevated` self-releases when a LEARNING learner
        outgrows an early deficit (increments turn negative, e decays); a
        PERSISTENT deficit keeps positive drift so the block is effectively
        permanent for exactly the learners it should block. Blocking on the
        current value is more lenient than sticky rejection, so the sticky
        α-guarantee upper-bounds the block's false-positive persistence."""
        return self.e_now >= 1.0 / self.alpha


_GH9_X, _GH9_W = np.polynomial.hermite.hermgauss(9)
_GH9_WN = _GH9_W / np.sqrt(np.pi)


def bias_stationary_sd(alpha_t, q_t, sigma_hat, *, item_s_sd=0.3,
                       offset_mult=SKILL_MODE_MULTIPLIER, lapse=0.025):
    """M17 (OQ6) — the stationary SD of the criterion under BOUNDARY-ANCHORED
    serving: the rigorous answer to "train to zero bias".

    DERIVATION. Under the soft R–W rule with mirror-paired, label-alternating
    items anchored on the TRUE boundary (s̄ = ±offset_mult·σ̂, y* = sign(s̄) —
    the D15 skill-mode stream), the criterion follows
        t' = t + α_t·(p̂(s_real, t) − y*) + ξ,   ξ ~ N(0, q_t²).
    (1) Mean: the pair drift α_t(1−2λ)[Φ((a−t)/σ_e) + Φ((−a−t)/σ_e) − 1]
        (a = offset, σ_e = √(σ² + s_sd²)) vanishes exactly at t = 0 and is
        restoring — F23 formalized: the stream ANCHOR sets E[t_∞] = 0.
        Anchoring is load-bearing: placement that tracks the criterion
        (s̄ = t̂) has NO restoring force — the attractor follows the learner
        and t random-walks (verified: SD ≈ 3 vs ≈ 0.13; this is F23/M10's
        herding pathology reproduced analytically).
    (2) Restoring gain per trial: κ = α_t(1−2λ)·φ(a/σ_e)/σ_e.
    (3) Fluctuation floor (linearized AR(1) around 0):
        Var_∞ = (q_t² + α_t²·Var[p̂]) / (κ(2−κ))  +  (α_t·δ̄)²/(2−κ)²,
        where δ̄ = 1 − p̄(a) is the mean |prediction error| on the correct
        side and the last term is the deterministic intra-pair oscillation
        from STRICT label alternation. Alternation is itself load-bearing:
        iid-random labels would inject α_t²·¼ into the diffusion instead —
        an order-of-magnitude worse floor (formalizes F6b beyond fairness).
    The offset trades restoring gain (max at a=0) against oscillation and
    per-item noise (min at large a); at the skill-mode offset the band is
    within ~15% of the a-optimum, so ZERO-BIAS SERVING ≈ SKILL SERVING —
    no dedicated 0-bias mode is needed once |t̂| is inside the band.
    The achievable "zero bias" is the band ±c·SD_∞; demanding less buys
    pure lateness (M17 priority: FG ≥ lateness). Validated against direct
    simulation of the recurrence (test_audit_fixes, ≤20% across the grid)."""
    from scipy.stats import norm as _norm
    sig_e = np.sqrt(sigma_hat ** 2 + item_s_sd ** 2)
    a = offset_mult * sigma_hat                     # mirrored item position
    kappa = min(alpha_t * (1.0 - 2.0 * lapse)
                * float(_norm.pdf(a / sig_e)) / sig_e, 0.9)
    # GH over s_real ~ N(a, s_sd²): p̂ moments at the correct-side item, t=0
    z = (a + np.sqrt(2.0) * item_s_sd * _GH9_X) / sigma_hat
    p_nodes = lapse + (1.0 - 2.0 * lapse) * _norm.cdf(z)
    p_bar = float((_GH9_WN * p_nodes).sum())
    var_p = float((_GH9_WN * p_nodes ** 2).sum() - p_bar ** 2)
    delta_bar = 1.0 - p_bar
    var_stat = ((q_t ** 2 + alpha_t ** 2 * var_p) / (kappa * (2.0 - kappa))
                + (alpha_t * delta_bar) ** 2 / (2.0 - kappa) ** 2)
    return float(np.sqrt(var_stat))


def derived_t_star(alpha_t, q_t, sigma_hat, *, c=2.0, item_s_sd=0.3,
                   lapse=0.025, lo=0.08, hi=0.45):
    """The 0-bias gate band: c·SD_∞ (c=2 ⇒ ~95% of the stationary mass),
    clipped to sane product bounds. σ̂-adaptive: the demanded band tracks
    what the physics of the learner's own noise allows — 'zero bias' means
    the criterion is statistically indistinguishable from the fluctuation
    floor around the anchored zero."""
    return float(np.clip(c * bias_stationary_sd(alpha_t, q_t, sigma_hat,
                                                item_s_sd=item_s_sd,
                                                lapse=lapse), lo, hi))


def at_bar_accuracy(s, s_sd, sigma_star, t_hat, y_star, lapse=0.025):
    """H0 ('learner exactly at the mastery bar', σ=σ*, t=t̂) P(correct) for a
    served probe, stimulus noise included via the closed-form attenuation —
    the e-gate's per-probe reference a0_i."""
    from scipy.stats import norm as _norm
    att = np.sqrt(1.0 + (float(s_sd) / sigma_star) ** 2)
    p1 = lapse + (1.0 - 2.0 * lapse) * _norm.cdf(
        (float(s) - t_hat) / sigma_star / att)
    return float(p1 if y_star == 1 else 1.0 - p1)


# ───────────── retention scheduler (decay-model interface, D9/LT2) ─────────────

class RetentionScheduler:
    """Spaced-review scheduler keyed by (task, difficulty-bin). SM-2-style is
    the Phase-2 default instance; the interface (`due`, `update_on_retrieval`)
    is what the fitted Ebbinghaus retrievability/stability model replaces in
    Phase 3 (memory §3A) without touching the policy."""

    def __init__(self, ease=2.0, first_interval=5.0):
        self.ease = float(ease)
        self.first = float(first_interval)
        self._next = {}        # bin_key -> next-due time
        self._ivl = {}         # bin_key -> current interval

    def register(self, bin_key, now):
        """A bin becomes review-eligible (task just mastered)."""
        if bin_key not in self._next:
            self._ivl[bin_key] = self.first
            self._next[bin_key] = now + self.first

    def due(self, bin_key, now):
        return bin_key in self._next and now >= self._next[bin_key]

    def due_bins(self, now):
        return [b for b in self._next if now >= self._next[b]]

    def update_on_retrieval(self, bin_key, correct, now):
        if bin_key not in self._ivl:
            self.register(bin_key, now)
        if correct:
            self._ivl[bin_key] *= self.ease
        else:
            self._ivl[bin_key] = self.first          # reset on lapse
        self._next[bin_key] = now + self._ivl[bin_key]


# ───────────── per-task mode policy ─────────────

@dataclass
class ModeThresholds:
    # None (M17 default) ⇒ the σ̂-adaptive derived 0-bias band
    # `derived_t_star` (2× the stationary fluctuation SD); a float pins the
    # legacy fixed tolerance (0.30 was the pre-OQ6 assumption).
    t_star: float | None = None
    sd_floor: float = 0.23              # ℓ-SD mastery floor (M3: ~1.5×0.154)
    c_sd: float = 1.0                   # posterior-SD multiplier for bias gate
    alpha: float = 0.05
    Z: float = 2.0
    rho: float = 0.5                    # assumed 85%-weight width (F20 placement)
    # M18: the D16 mean-skill branch was designed for v14's sub-floor
    # domain1 margin (F19); under the v15 margins (all > 1.5× floor) it is
    # suspected of leaking premature declarations through the mixture's
    # strata-dragged mean ℓ̂ (F63 static_below residual). Toggle for the
    # M18 hardening ablation; True = shipped behavior.
    meanskill_gate: bool = True
    # M21 (F75): novice-onboarding placement. Skill-mode difficulty is set
    # against a CONSERVATIVE σ quantile while the belief is still wide:
    # σ_place = σ̂ · exp(z · max(sd_ℓ − sd_floor, 0)) — anneals to exactly
    # σ̂ once the posterior reaches the variance floor. 0.0 (default) is
    # bit-identical to M20 behavior. Motivation: a fresh optimistic prior
    # placed both real testers at ~50% served accuracy for whole sessions
    # (target 84%) while the filter descended toward their true σ.
    skill_sigma_z: float = 0.0
    # M24 (F84, opt-in): skill-mode placement by STRATUM-AWARE POSTERIOR
    # EXPECTED PROGRESS (`expected_progress_score`) instead of the plug-in
    # E[w] argmax — the σ-channel analog of the D15 bias scorer. Supersedes
    # the F75 z-heuristic when on (the posterior integration IS the
    # principled version of "be conservative while wide"); the mirror
    # magnitude-pairing device is bypassed (label-side alternation and
    # boundary-centered candidate signing remain — criterion-herding safety
    # is re-verified by study_m24_selection arm P). False ⇒ bit-identical.
    progress_placement: bool = False
    # M25 (F87, opt-in): skill-mode placement by ITEM-LEVEL TIER-3 ROLLOUT
    # (`trainer_rollout.rollout_item_q`) — the non-myopic F84 successor.
    # A stratum-aware progress shortlist is re-ranked by H-step CRN
    # Monte-Carlo rollout value under the mixture-honest dynamics; runs
    # inside the F84-v6 safety devices (mirror window + finishing switch).
    # Mutually exclusive with progress_placement (rollout wins if both).
    # False ⇒ bit-identical.
    rollout_placement: bool = False
    rollout_H: int = 6                  # rollout depth
    rollout_L: int = 8                  # rollouts per candidate
    rollout_nro: int = 96               # belief particles inside rollouts
    rollout_short: int = 12             # shortlist re-ranked by rollout
    # M25 (F88, opt-in): sharp-margin declaration conjuncts — BOTH
    # MEASURED INEFFECTIVE and rejected for the sandbox (PECR record in
    # docs/M25_SELECTION_V2.md; kept as opt-in ablation knobs). The
    # shipped stack declares ~30–37% of static learners parked 0.15 BELOW
    # the cut within 240 trials. Instrumentation (study_sharp_margin arm
    # X1, 20 seeds/margin): every false declaration lands in a
    # post-boundary SESSION but at median ~22 session-trials in (only
    # 14–22% inside a 12-trial window), with trainability 0.92, a QUIET
    # e-gate (median 0.75), and π barely past the line (0.959–0.961) —
    # the mixture is genuinely fooled by a margin below its per-trial
    # information rate (KL ≈ 0.005 nats/trial ⇒ ~600 trials for BF 20),
    # not caught mid-transient. Hence: boundary_refractory (block
    # declaration for N served trials after note_boundary()) trims 6→5/20
    # at a real lateness cost (well-spec BOTH-declared 156→180, −15pp);
    # min_trainability is a no-op at this margin (trainability at FG is
    # 0.92); denser probes cannot beat the Bernoulli bound. The effective
    # sharp-margin filters are ARCHITECTURAL: the D33 confirmation
    # lifecycle (arm X4) and the D18 re-cert backstop (F32).
    boundary_refractory: int = 0
    min_trainability: float = 0.0


class TaskModePolicy:
    """Chooses mode + item for ONE task from its filter + candidate pool."""

    # mean-skill gate (M10 sandbox, F19): trailing window + one-sided z
    MEANSKILL_W = 20
    MEANSKILL_Z = 1.645

    def __init__(self, task, ell_star, sigma_star, thresholds: ModeThresholds):
        self.task = int(task)
        self.ell_star = float(ell_star)
        self.sigma_star = float(sigma_star)
        self.th = thresholds
        self._bias_label_balance = 0      # running (#y*=1) − (#y*=0) in bias mode
        self._skill_side = 1
        self._last_skill_s = None         # partner signal for mirror pairing
        self._bias_probe = False          # alternates correct/probe (dual control)
        self._mastered = False
        self._ml_hist: list[float] = []   # per-served-trial posterior-mean ℓ
        self.egate = None                 # M16 D23: optional EProcessGate
        self._refract = 0                 # M25 F88: boundary refractory

    def effective_t_star(self, filt, sig_hat):
        """M17 (OQ6): the 0-bias gate band — derived stationary band by
        default, fixed value when ModeThresholds.t_star is set."""
        if self.th.t_star is not None:
            return self.th.t_star
        return derived_t_star(filt.p.alpha_t, filt.p.q_t, sig_hat)

    # mode gate (plan §8.2 step 1) ----------------------------------------
    def choose_mode(self, filt, retention: RetentionScheduler, now):
        mt, ml = filt.mean()
        sig_hat, t_hat = engine_to_plan(mt, ml)
        sd_t = filt.sd()[0]                            # θ-SD == t-SD
        if self.is_mastered(filt):
            return RETENTION, (sig_hat, t_hat)
        # bias-correction if |t̂| large vs both tolerance and its posterior SD
        t_star = self.effective_t_star(filt, sig_hat)
        if abs(t_hat) > max(t_star, self.th.c_sd * sd_t):
            return BIAS, (sig_hat, t_hat)
        return SKILL, (sig_hat, t_hat)

    def note_boundary(self):
        """M25 (F88): the caller applied a boundary_shift to this task's
        filter — start the declaration refractory (no-op at the default 0)."""
        self._refract = max(self._refract, int(self.th.boundary_refractory))

    def note_posterior(self, filt):
        """Record this trial's posterior-mean ℓ (call ONCE per served trial —
        feeds the mean-skill graduation gate)."""
        if self._refract > 0:
            self._refract -= 1
        self._ml_hist.append(float(filt.mean()[1]))
        if len(self._ml_hist) > 3 * self.MEANSKILL_W:
            del self._ml_hist[:-2 * self.MEANSKILL_W]

    def _meanskill_ok(self, filt):
        """F19 gate (M10 sandbox 'G3'): trailing-W mean of ℓ̂ minus a one-sided
        z·SE (autocorrelation-adjusted effective n) clears ℓ*. Detects mastery
        for domains whose ceiling-to-cut gap is below the variance floor, where
        the pass-mass gate NEVER fires (domain1); false-graduation stayed 0 in
        the sandbox campaign."""
        W = self.MEANSKILL_W
        if len(self._ml_hist) < W:
            return False
        w = np.array(self._ml_hist[-W:])
        r1 = float(np.corrcoef(w[:-1], w[1:])[0, 1]) if w.std() > 1e-9 else 0.9
        r1 = min(max(r1, 0.0), 0.98)
        n_eff = max(W * (1.0 - r1) / (1.0 + r1), 1.0)
        _, sd_l = filt.sd()
        return bool(sd_l <= self.th.sd_floor and
                    w.mean() - self.MEANSKILL_Z * sd_l / np.sqrt(n_eff)
                    >= self.ell_star)

    def is_mastered(self, filt):
        """Hybrid graduation (M10): AD6 pass-mass gate OR the mean-skill gate,
        plus a CONFIDENT bias condition |t̂| + 0.5·sd_t ≤ t* — the margin stops
        the gate flapping when t̂ wanders at the tolerance boundary (observed
        M10: boundary flapping caused mastery↔unmastered churn).
        M16 (D23): when a cert-probe e-gate is attached, an ELEVATED e-value
        (evidence of below-bar probe performance) blocks declaration."""
        if self.egate is not None and self.egate.elevated:
            return False
        if self._refract > 0:                       # M25 F88 refractory
            return False
        if (self.th.min_trainability > 0.0          # M25 F88 conjunct
                and hasattr(filt, "trainability")
                and filt.trainability(self.ell_star)
                < self.th.min_trainability):
            return False
        mt, ml = filt.mean()
        sig_hat, t_hat = engine_to_plan(mt, ml)
        sd_t = filt.sd()[0]
        skill_ok = (filt.is_mastered(self.ell_star, sd_floor=self.th.sd_floor,
                                     alpha=self.th.alpha, Z=self.th.Z)
                    or (self.th.meanskill_gate and self._meanskill_ok(filt)))
        t_star = self.effective_t_star(filt, sig_hat)
        return bool(skill_ok and abs(t_hat) + 0.5 * sd_t <= t_star)

    # item selection (plan §8.2 step 2) -----------------------------------
    def select(self, mode, est, cands, retention, now, rng, filt=None):
        """Return (idx into cands, info dict). cands is a TaskCandidates view
        (feedback-safe for bias/skill). Returns None if the pool is empty.
        filt: the task's TaskFilter — enables the cloud-scored bias-correction
        pick (M10); without it bias mode falls back to nearest-criterion."""
        if len(cands) == 0:
            return None
        sig_hat, t_hat = est
        if mode == BIAS:
            # enforce running label balance: prefer the under-served label (F6b)
            want = 0 if self._bias_label_balance > 0 else 1
            target = t_hat                              # near the criterion
            if filt is not None and not self._bias_probe:
                # corrective half (dual control): max expected |t|-reduction
                # over the cloud. Cap the scored set — a random 4096 of a 60k
                # pool loses nothing and bounds the (n_cands × N) broadcast.
                pick = None
                view = cands
                if len(cands) > 4096:
                    pick = rng.choice(len(cands), size=4096, replace=False)
                    view = cands.subset(pick)
                j = self._argmax_score(bias_correction_score(filt, view),
                                       view, want_label=want)
                idx = int(pick[j]) if pick is not None else j
            else:
                # probe half: max information about t (precise near-criterion
                # item) so t̂ tracks the criterion being moved — without these
                # the corrective items (information-poor) let t̂ go stale and
                # the trainer overshoots past t = 0 (M10 dual-control finding)
                sc = bias_probe_score(cands.s_mean, cands.s_sd, sig_hat, t_hat)
                idx = self._argmax_score(sc, cands, want_label=want)
            was_probe = self._bias_probe or filt is None
            self._bias_probe = not self._bias_probe
            return idx, {"target": target, "want_label": want,
                         "probe": was_probe}
        if mode == SKILL:
            side = self._skill_side
            # Placement centers on the TRUE BOUNDARY (s=0), NOT the criterion:
            # under the R–W dynamics a balanced-label stream centered at c is a
            # stable attractor pulling the criterion to c — centering on t̂
            # herds the learner's bias TO t̂ (observed M10: t̂ and true t pinned
            # at the mode threshold). Boundary-centered placement makes the
            # skill-mode attractor the bias target t=0, so skill training
            # maintains calibration; the ≤t* difficulty mis-centering this
            # costs is bounded by the mode gate. Also the F7 guard collapses:
            # labels match the placement side by construction.
            # MIRROR PAIRING: independent per-side argmax serves whatever each
            # side's precise items happen to be (bank asymmetry: precise
            # positives at ≈+0.96, precise negatives at ≈−0.55) — stream
            # midpoint ≈ +0.2 ⇒ criterion herded there (M10). Each pick
            # therefore targets the NEGATED magnitude of its partner, pinning
            # the pairwise midpoint at 0 by construction.
            # M24 (F84): posterior expected-progress placement — argmax of
            # the stratum-aware tier-1 σ-progress over the side's labeled
            # candidates. Cap 1024 bounds the n_cands × N_total broadcast
            # (the mixture pools ~2400 particles; the bank's per-side pools
            # are ~35k and highly redundant — the argmax over a 1024 random
            # subsample concedes only the gap between the top-0.003 and
            # top-0.1 score quantiles of a smooth score surface). No
            # plug-in target exists on this path; the posterior decides
            # which difficulty trains the most mass.
            # M24 (F84): posterior-progress placement WITH the M10 mirror
            # invariant. The progress argmax hedges difficulty over the
            # honest posterior (jump scenario: delivered w_true 0.31 →
            # 0.40; the only rule that declares the contact scenario) —
            # but a free per-trial argmax abandons the M10 magnitude
            # pairing, the served stream's midpoint un-pins, and the R–W
            # attractor herds the learner's criterion to it (measured:
            # |t_true| end 0.115 → 0.19–0.26, which stalls the
            # declaration gate's bias condition — wellspec declaration 38
            # → 94+ with UNCHANGED served accuracy/w_true; the failure was
            # never difficulty, it was calibration). Rule: the FIRST pick
            # of each ± pair is the free progress argmax; its mirror
            # PARTNER is progress-argmaxed within |s| ∈ |partner|·(1±0.15)
            # — the pairwise midpoint stays pinned at 0 by construction,
            # exactly the legacy device, with the posterior choosing among
            # the mirror-compatible items.
            if ((self.th.progress_placement or self.th.rollout_placement)
                    and filt is not None):
                want = 1 if side > 0 else 0
                pick = None
                view = cands
                if len(cands) > 1024:
                    pick = rng.choice(len(cands), size=1024, replace=False)
                    view = cands.subset(pick)
                view0 = view          # pre-mirror view: rollout π0 pool
                if (self._last_skill_s is not None
                        and (self._last_skill_s > 0) != (side > 0)):
                    mag = abs(self._last_skill_s)
                    m_mask = np.abs(np.abs(view.s_mean) - mag) <= 0.15 * max(
                        mag, 0.3)
                    if (m_mask & (view.y_star == want)).any():
                        sub = np.where(m_mask)[0]
                        view = view.subset(sub)
                        pick = (pick[sub] if pick is not None
                                else sub)
                # F84-iii (dual control at the item level): pure-progress
                # placement raises delivered training value but STARVES
                # declaration evidence in the finishing window (measured:
                # wellspec w_true 0.50 → 0.60 but declaration 38 → 59) —
                # once the task is declaration-imminent (π − 2·mcse past
                # the F70 finishing threshold) items switch to ℓ-Fisher-
                # information placement AT THE CURRENT BELIEF (the
                # cert_probe_score formula evaluated at σ̂, optimum
                # |s−t̂| ≈ 1.35·σ̂·att): finishing is measurement of where
                # the learner IS — π and sd_ℓ contract on posterior-
                # informative items, and 1.35σ̂ sits close enough to the
                # 1.077σ̂ training optimum that w stays ≈ 0.86. Two
                # REJECTED variants, measured: bar-anchored finishing info
                # (σ* in place of σ̂ — near-deterministic responses for an
                # above-bar bulk, declaration 59 → 96+) and bar-referenced
                # progress credit (zeroes the bulk's vote; the argmax
                # chases the below-bar high-σ tail with over-easy items,
                # w_true 0.60 → 0.37). Bar-referencing is correct only at
                # the ALLOCATION level (F85), where tasks, not
                # difficulties, are ranked.
                pi, mcse = filt.pass_mass(self.ell_star)
                rolled = False
                if pi - 2.0 * mcse >= 0.70:      # F70 finishing threshold
                    sc = cert_probe_score(view.s_mean, view.s_sd,
                                          max(sig_hat, 1e-6), t_hat)
                    finishing = True
                    j = self._argmax_score(sc, view, want_label=want)
                elif self.th.rollout_placement:
                    # M25 (F87): stratum-aware progress shortlist, re-ranked
                    # by the item-level H-step CRN rollout value. The
                    # shortlist honors the label side and the mirror window
                    # (view is already mirror-filtered above); the rollout's
                    # π0 draws follow-up items from the same view.
                    from training.trainer_rollout import rollout_item_q
                    sc0 = expected_progress_score(filt, view)
                    m = np.where(view.y_star == want)[0]
                    if m.size == 0:
                        m = np.arange(len(view))
                    short = m[np.argsort(sc0[m])[::-1]
                              [:self.th.rollout_short]]
                    q = rollout_item_q(
                        filt, view.s_mean[short], view.s_sd[short],
                        view.y_star[short],
                        pool_s=view0.s_mean, pool_sd=view0.s_sd,
                        pool_y=view0.y_star,
                        t_star=self.effective_t_star(filt,
                                                     max(sig_hat, 1e-6)),
                        H=self.th.rollout_H, L=self.th.rollout_L,
                        n_ro=self.th.rollout_nro, rng=rng,
                        step_parity=0 if side > 0 else 1)
                    j = int(short[int(np.argmax(q))])
                    finishing = False
                    rolled = True
                else:
                    sc = expected_progress_score(filt, view)
                    finishing = False
                    j = self._argmax_score(sc, view, want_label=want)
                idx = int(pick[j]) if pick is not None else j
                self._last_skill_s = float(cands.s_mean[idx])
                self._skill_side *= -1
                return idx, {"target": float(cands.s_mean[idx]),
                             "want_label": want, "progress": True,
                             "rollout": rolled, "finishing": finishing}
            # M21 (F75): conservative-quantile placement during onboarding
            sig_place = sig_hat
            if self.th.skill_sigma_z > 0.0 and filt is not None:
                sd_l = filt.sd()[1]
                sig_place = sig_hat * float(np.exp(
                    self.th.skill_sigma_z
                    * max(sd_l - self.th.sd_floor, 0.0)))
            m_eff = SKILL_MODE_MULTIPLIER
            if (self._last_skill_s is not None
                    and (self._last_skill_s > 0) != (side > 0)):
                m_eff = abs(self._last_skill_s) / max(sig_place, 1e-6)
            target = side * m_eff * sig_place
            want = 1 if side > 0 else 0
            # F20: maximize the EXPECTED learning weight under stimulus noise
            ew = expected_skill_weight(cands.s_mean, cands.s_sd, sig_place,
                                       0.0, side=side, m=m_eff,
                                       rho=self.th.rho)
            idx = self._argmax_score(ew, cands, want_label=want)
            self._last_skill_s = float(cands.s_mean[idx])
            self._skill_side *= -1
            return idx, {"target": target, "want_label": want}
        # RETENTION: draw a due bin's item (eligibility-gated, F6a)
        due = retention.due_bins(now)
        target = t_hat + (SKILL_MODE_MULTIPLIER + 0.5) * sig_hat   # easier review
        idx = self._nearest(cands, target)
        bin_key = (self.task, self._bin_of(cands.s_mean[idx], est))
        return idx, {"target": target, "bin": bin_key,
                     "eligible": retention.due(bin_key, now) or not due}

    def _nearest(self, cands, target, want_label=None):
        s = cands.s_mean
        if want_label is not None:
            mask = cands.y_star == want_label
            if mask.any():
                sub = np.where(mask)[0]
                return int(sub[np.argmin(np.abs(s[sub] - target))])
        return int(np.argmin(np.abs(s - target)))

    def _argmax_score(self, score, cands, want_label=None):
        if want_label is not None:
            mask = cands.y_star == want_label
            if mask.any():
                sub = np.where(mask)[0]
                return int(sub[np.argmax(score[sub])])
        return int(np.argmax(score))

    @staticmethod
    def _bin_of(s, est):
        sig_hat, t_hat = est
        d = (s - t_hat) / max(sig_hat, 1e-6)
        return int(np.round(d))                         # coarse difficulty bin

    # bookkeeping after a trial --------------------------------------------
    def note_served(self, mode, y_star):
        if mode == BIAS:
            self._bias_label_balance += 1 if y_star == 1 else -1


# ───────────── cross-task scheduler (D3) ─────────────

class DeficiencyScheduler:
    """Deficiency-weighted interleave across tasks (D3/OQ1), with a
    consecutive-same-task cap (R7)."""

    def __init__(self, K, t_star=DEFAULT_T_STAR, max_consec=5, *,
                 finish_first=False, finish_threshold=0.70,
                 finish_budget=60, cooldown=60, trainability_floor=None,
                 finish_sd_tol=1.0, progress_alloc=False,
                 progress_s_sd=0.0, progress_lifecycle=False,
                 refinish_threshold=0.50, explore_every=0):
        self.K = K
        # M24 (F85, opt-in): rank unmastered tasks by EXPECTED PROGRESS PER
        # TRIAL toward their own bars (`expected_progress_rate`) instead of
        # worst-first deficiency. Takes PRECEDENCE over the D36
        # trainability_floor discount (which it subsumes from first
        # principles — static/at-ceiling posterior mass contributes zero
        # progress). finish-first, suspensions, and the consec cap are
        # unchanged (they are the measurement/variety half). progress_s_sd:
        # nominal bank stimulus noise for the ideal-placement weight.
        self.progress_alloc = bool(progress_alloc)
        self.progress_s_sd = float(progress_s_sd)
        # M25 (F86, opt-in — the F85-ii successor): EP-v2 couples the EP
        # ranking to the CONFIRMATION LIFECYCLE. A task that has EVER
        # satisfied the declaration gate (`ever_declared`) no longer
        # competes for TRAINING trials by EP — its need is MEASUREMENT
        # (re-confirming after the Gate-4 boundary hazard re-widens it),
        # which is the finish-first phase's job. MEASURED VERDICT
        # (study_m25_selection arm Q2, docs/M25_SELECTION_V2.md §1.4):
        # behaviorally INERT at K=2 — the finishing lane spends the same
        # ~28 trials/session re-confirming the declared task that the EP
        # ranking did, because the re-polish cost is a property of the
        # Gate-4 hazard (which re-widens every declared task at every
        # boundary), not of the allocator; Gate-1 dominates the
        # BOTH-declared endpoint and the sandbox default stays off (D41).
        # Kept opt-in as machinery for K≥3 rosters and for the queued
        # hazard-level successor. Lifecycle routing:
        #   * ever-declared AND finishing-eligible (bias in band, sd within
        #     finish_sd_tol·floor) → measurement lane: served through the
        #     finish-first machinery at the LOWER `refinish_threshold`
        #     (the task has already demonstrated the bar once; the budget +
        #     cooldown bound its absorption — the F70-validated device);
        #   * ever-declared but NOT eligible (bias walked off / deep
        #     re-widening = genuine regression) → back in the EP lane:
        #     that is honest re-TRAINING need, not measurement;
        #   * never-declared → EP lane (F85 unchanged).
        # explore_every (the F85-ii exploration floor): every k-th pick, an
        # EP-lane task not served within the last `explore_every` picks is
        # served instead of the EP argmax (oldest first). The M23
        # fixed-share re-opens the PRIOR toward a written-off task, but
        # without served trials the exonerating evidence never arrives —
        # the floor guarantees the evidence stream that lets an E-shaped
        # recovery re-open its own trainability. 0 = off; both default-off
        # ⇒ bit-identical to M24.
        self.progress_lifecycle = bool(progress_lifecycle)
        self.refinish_threshold = float(refinish_threshold)
        self.explore_every = int(explore_every)
        self.ever_declared = set()
        self._pick_no = 0
        self._last_pick = {}
        # ranking scale only (relative weighting): None (derived-gate mode,
        # M17) falls back to the fixed constant
        self.t_star = float(t_star if t_star is not None else DEFAULT_T_STAR)
        self.max_consec = int(max_consec)
        self.finish_first = bool(finish_first)
        self.finish_threshold = float(finish_threshold)
        self.finish_budget = int(finish_budget)
        self.cooldown = int(cooldown)
        # M22 (F77, opt-in): discount a task's deficiency by the mixture's
        # P(trainable) — raw worst-first conflates "far from cut" with
        # "worth training": in the four-participant pilot the weakest task
        # absorbed 60–85% of trials WHILE its trainability collapsed
        # (USER-B 0.15, USER-D 0.05), starving near-mastery tasks (USER-B's
        # domain2 at π 0.63, six trials/session). effective = deficiency ×
        # (floor + (1−floor)·trainability); the floor keeps a residual
        # service level and the wide fresh prior (trainability ≈ 0.75)
        # makes the discount mild during onboarding. None ⇒ bit-identical.
        self.trainability_floor = (None if trainability_floor is None
                                   else float(trainability_floor))
        # M22 (F77-ii): finishing ELIGIBILITY tolerance on the sd floor.
        # The mixture's cross-strata variance keeps sd_ℓ a hair above
        # sd_floor until the ceiling strata concentrate — which needs
        # on-task trials the worst-first pick won't grant a low-deficiency
        # task: sd_l 0.232 vs floor 0.23 deadlocked finishing entirely
        # (the residual crack in the F70 deadlock fix). Eligibility uses
        # sd_ℓ ≤ finish_sd_tol·sd_floor; DECLARATION keeps the strict
        # floor (is_mastered unchanged ⇒ no FG channel). 1.0 = legacy.
        self.finish_sd_tol = float(finish_sd_tol)
        self._fin_spent, self._cool = {}, {}
        self._last = None
        self._consec = 0

    def deficiency(self, filt, ell_star):
        pi, _ = filt.pass_mass(ell_star)
        _, t_hat = engine_to_plan(*filt.mean())
        skill_short = 1.0 - pi
        bias_over = float(np.clip(abs(t_hat) / self.t_star - 1.0, 0.0, 1.0))
        return max(skill_short, bias_over)

    # M19 (F70): worst-first interleave STARVES near-mastery tasks in
    # multi-task service — as π rises, a task's deficiency falls below its
    # rivals' and the scheduler abandons it exactly when it needs the last
    # consecutive evidence to clear π−Z·mcse ≥ 0.95 (K=7 deadlock: NO task
    # ever graduates, π plateaus ≈ 0.75). The budgeted FINISH-FIRST phase
    # fixes that for the mixture stack, but showed second-order policy
    # interactions in the point-filter/v14 integration (bias-correction
    # interrupted mid-flight, F24-like) — therefore OPT-IN (finish_first),
    # default OFF ⇒ bit-identical legacy behavior; the K=7 protocol study
    # runs it ON. Port item: tune per deployment before making it default.

    def mark_declared(self, k):
        """M25 (F86): record that task k has (at some point) satisfied the
        declaration gate — the confirmation-lifecycle key. Called by
        `TrainerPolicy.record` at every mastery transition and seedable by
        callers with a persistent lifecycle (the sandbox's provisional/
        confirmed/revoked meta survives across sessions)."""
        self.ever_declared.add(int(k))

    def pick(self, filters, ell_stars, mode_policies, exclude=()):
        """Choose the next task: highest deficiency among unmastered tasks
        (legacy default); with finish_first=True, a budgeted finishing phase
        (declaration-imminent tasks completed before worst-first resumes).
        Tasks in `exclude` (M22: in-session suspensions, e.g. a consistency
        pause) are not served. None if all mastered/excluded."""
        lifecycle = self.progress_alloc and self.progress_lifecycle
        need_close = self.finish_first or lifecycle
        if need_close:
            for k in list(self._cool):
                self._cool[k] -= 1
                if self._cool[k] <= 0:
                    del self._cool[k]
        self._pick_no += 1
        defs, close, meas = {}, {}, {}
        for k in range(self.K):
            if k in exclude:
                continue
            if mode_policies[k].is_mastered(filters[k]):
                self.ever_declared.add(k)
                continue
            eligible = False
            if need_close:
                # FINISHING only when the pass-mass margin is the SOLE
                # missing declaration condition (bias in band, sd at floor)
                pi, mcse = filters[k].pass_mass(ell_stars[k])
                sd_t, sd_l = filters[k].sd()
                sig_hat, t_hat = engine_to_plan(*filters[k].mean())
                bias_ok = (abs(t_hat) + 0.5 * sd_t
                           <= mode_policies[k].effective_t_star(filters[k],
                                                                sig_hat))
                floor_ok = (sd_l <= self.finish_sd_tol
                            * mode_policies[k].th.sd_floor)
                eligible = bias_ok and floor_ok
                close[k] = (pi - 2.0 * mcse) if eligible else -1.0
            if lifecycle and k in self.ever_declared and eligible:
                # F86 measurement lane: an ever-declared, finishing-eligible
                # task never competes by EP — it is served through the
                # finishing phase (refinish_threshold below) or, when no
                # task has any training claim, by legacy deficiency.
                meas[k] = self.deficiency(filters[k], ell_stars[k])
                continue
            if self.progress_alloc:            # M24 (F85)
                sig_hat, _ = engine_to_plan(*filters[k].mean())
                defs[k] = expected_progress_rate(
                    filters[k], ell_stars[k],
                    mode_policies[k].effective_t_star(filters[k], sig_hat),
                    s_sd_nominal=self.progress_s_sd,
                    rho=mode_policies[k].th.rho)
            else:
                defs[k] = self.deficiency(filters[k], ell_stars[k])
                if (self.trainability_floor is not None
                        and hasattr(filters[k], "trainability")):
                    tr = float(filters[k].trainability(ell_stars[k]))
                    defs[k] *= (self.trainability_floor
                                + (1.0 - self.trainability_floor) * tr)
        if not defs and not meas:
            return None
        ep_lane = bool(defs)      # exploration floor applies to this lane
        if not defs:
            # every unmastered task is in the measurement lane and none is
            # past its finishing threshold / off cooldown ⇒ rank by
            # deficiency rather than starving them (the F70 lesson)
            defs = meas
            ep_lane = False
        if need_close:
            def _thr(k):
                return (self.refinish_threshold
                        if (lifecycle and k in self.ever_declared)
                        else self.finish_threshold)
            finishing = {k: c for k, c in close.items()
                         if c >= _thr(k) and k not in self._cool
                         and (self.finish_first or
                              (lifecycle and k in self.ever_declared))}
        else:
            finishing = {}
        for k in list(self._fin_spent):
            if k not in finishing:
                self._fin_spent[k] = 0
        from_finish = bool(finishing)
        if finishing:
            order = sorted(finishing, key=lambda k: -finishing[k])
        else:
            order = sorted(defs, key=lambda k: -defs[k])
        choice = order[0]
        if (self._last == choice and self._consec >= self.max_consec
                and (len(order) > 1 or from_finish)):
            # R7 variety cap (legacy semantics when finish_first is off)
            alts = (order[1:]
                    or [k for k in sorted(defs, key=lambda k: -defs[k])
                        if k != choice])
            if alts:
                choice = alts[0]
        if (self.explore_every > 0 and self.progress_alloc and ep_lane
                and not from_finish and len(defs) > 1):
            # F86 exploration floor: serve the longest-unserved EP-lane
            # task once per explore_every picks (oldest first)
            for k in defs:
                self._last_pick.setdefault(k, self._pick_no)
            starved = [k for k in defs
                       if self._pick_no - self._last_pick[k]
                       >= self.explore_every]
            if starved and choice not in starved:
                choice = min(starved, key=lambda k: self._last_pick[k])
        if from_finish and choice in finishing:
            self._fin_spent[choice] = self._fin_spent.get(choice, 0) + 1
            if self._fin_spent[choice] > self.finish_budget:
                self._cool[choice] = self.cooldown
                self._fin_spent[choice] = 0
        self._last_pick[choice] = self._pick_no
        self._consec = self._consec + 1 if choice == self._last else 1
        self._last = choice
        return choice


# ───────────── orchestrator ─────────────

class TrainerPolicy:
    """Drives a training session over K per-task filters + a real bank."""

    def __init__(self, filters, ell_stars, sigma_stars, bank, *,
                 thresholds: ModeThresholds = None, seed=0,
                 max_consec=5, exclude_segids=None, min_margin=0.30,
                 retention: RetentionScheduler = None,
                 probe_every=None, egate_alpha=0.05, finish_first=False,
                 trainability_floor=None, finish_sd_tol=1.0,
                 progress_alloc=False, progress_s_sd=0.0,
                 progress_lifecycle=False, refinish_threshold=0.50,
                 explore_every=0, terminal_confirmation=False,
                 maint_probes=0, stale_alpha=0.05):
        """probe_every (M16 D23, opt-in): every probe_every-th served trial of
        an unmastered task is a CERTIFICATION PROBE — the item maximizing
        Fisher information about ℓ at the CUT state (`cert_probe_score`),
        label-alternating — and its outcome feeds a per-task `EProcessGate`
        whose elevated state blocks graduation (attached to the task's
        TaskModePolicy). None (default) ⇒ bit-identical to shipped."""
        self.filters = filters
        self.K = len(filters)
        self.ell_stars = list(ell_stars)
        self.sigma_stars = list(sigma_stars)
        self.bank = bank
        self.th = thresholds or ModeThresholds()
        self.mode_policies = [TaskModePolicy(k, ell_stars[k], sigma_stars[k], self.th)
                              for k in range(self.K)]
        # scheduler ranking keeps a fixed scale (relative weighting only)
        self._sched_t_star = self.th.t_star or DEFAULT_T_STAR
        self.probe_every = probe_every
        self._probe_ctr = [0] * self.K
        self._probe_flip = [1] * self.K
        if probe_every is not None:
            for mp in self.mode_policies:
                mp.egate = EProcessGate(egate_alpha)
        self.scheduler = DeficiencyScheduler(self.K, self._sched_t_star,
                                             max_consec,
                                             finish_first=finish_first,
                                             trainability_floor=
                                             trainability_floor,
                                             finish_sd_tol=finish_sd_tol,
                                             progress_alloc=progress_alloc,
                                             progress_s_sd=progress_s_sd,
                                             progress_lifecycle=
                                             progress_lifecycle,
                                             refinish_threshold=
                                             refinish_threshold,
                                             explore_every=explore_every)
        # M22: tasks suspended for the rest of the session (e.g. by a
        # consistency pause); cleared by the caller at session boundaries
        self.suspended = set()
        # caller-supplied scheduler so interval units match the caller's `now`
        # clock (trial index by default; epoch seconds in the pipeline) — F18
        self.retention = retention if retention is not None else RetentionScheduler()
        self.rng = np.random.default_rng(seed)
        self.exclude = set(int(x) for x in (exclude_segids or []))
        self.min_margin = float(min_margin)
        self.log: list[TrialRecord] = []
        self._served = set()
        self._trial = 0
        self._was_mastered = [False] * self.K
        # M27 (F90, opt-in): TERMINAL-CONFIRMATION semantics. A task the
        # caller marks terminal (on its D33 CONFIRMATION) leaves the
        # training rotation for good: the deficiency scheduler never picks
        # it, so the Gate-4 boundary hazard — which stays FULLY applied to
        # its belief (nothing in the safety machinery is weakened, the D42
        # lesson) — no longer converts into a ~28-trials/session re-polish
        # tax; the RETENTION layer owns the task (due-driven review trials
        # + `maint_probes` at-bar certification probes per session). Every
        # terminal-task outcome feeds an anytime-valid stale-mastery
        # e-gate (Ville, α = stale_alpha): P(an at/above-bar learner is
        # ever wrongly revoked) ≤ α per task, while a genuinely regressed
        # learner accumulates evidence until the gate fires and the task
        # returns to the training rotation (`terminal_events`). Default
        # False = bit-identical (empty terminal set touches no path).
        self.terminal_confirmation = bool(terminal_confirmation)
        self.maint_probes = int(maint_probes)
        self.stale_alpha = float(stale_alpha)
        self.terminal = set()
        self._stale_gates = {}
        self._maint_due = [0] * self.K
        self.terminal_events = []

    def _cands(self, task, feedback_safe=True):
        excl = self.exclude | self._served
        return self.bank.candidates(task, exclude_segids=excl,
                                    feedback_safe=feedback_safe,
                                    min_margin=self.min_margin)

    def all_mastered(self):
        return all(k in self.terminal
                   or self.mode_policies[k].is_mastered(self.filters[k])
                   for k in range(self.K))

    # ── M27 (F90): terminal-confirmation lifecycle ──
    def mark_terminal(self, k, now=None):
        """Caller hook at a task's D33 CONFIRMATION (only meaningful with
        terminal_confirmation=True). Fresh stale gate per terminal spell.

        The task's review schedule is CONSOLIDATED to the single bin
        (k, 0): transition-registered difficulty bins proliferate under
        the M10 gate flapping, and once is_mastered no longer gates the
        retention pre-pick they would ALL fire every session (measured:
        ~20 retention trials/session — re-creating the very tax terminal
        semantics removes). One bin ⇒ one SM-2 schedule owns the task."""
        if not self.terminal_confirmation:
            return
        k = int(k)
        self.terminal.add(k)
        self._stale_gates[k] = EProcessGate(self.stale_alpha)
        for b in [b for b in self.retention._next if b[0] == k
                  and b != (k, 0)]:
            del self.retention._next[b]
            self.retention._ivl.pop(b, None)
        if now is not None:
            self.retention.register((k, 0), now)

    def note_session_open(self):
        """Caller hook at each session open: arm the per-session
        maintenance-probe budget for terminal tasks."""
        for k in self.terminal:
            self._maint_due[k] = self.maint_probes

    def _revoke_terminal(self, k):
        self.terminal.discard(k)
        self._stale_gates.pop(k, None)
        self._maint_due[k] = 0
        self.terminal_events.append({"event": "stale_revoked", "task": k,
                                     "trial": self._trial})

    def step(self, *, now=None, session_id="train", rt_ms=np.nan):
        """One training trial: schedule → choose mode → select → present to the
        learner (caller supplies the response via the returned closure pattern).
        Here we return the chosen (task, mode, seg_id, s, s_sd, y_star); the
        caller obtains y and calls `record`. now defaults to the trial index."""
        now = self._trial if now is None else now
        # F18 fix (M10 sandbox design B, due-driven): a mastered task whose
        # retention bin is DUE is served before the deficiency pick — review
        # actually happens, and the review trial refreshes the stale belief.
        task = None
        for k in range(self.K):
            if (k not in self.suspended
                    and (k in self.terminal            # M27 F90: terminal
                         or self.mode_policies[k].is_mastered(
                             self.filters[k]))
                    and any(b[0] == k for b in self.retention.due_bins(now))):
                task = k
                break
        # M27 (F90): maintenance probes — a terminal task's per-session
        # at-bar evidence budget (feeds the stale-mastery gate); served
        # after due retention, before any training pick
        maintenance = False
        if task is None and self.maint_probes > 0:
            for k in sorted(self.terminal):
                if k not in self.suspended and self._maint_due[k] > 0:
                    task, maintenance = k, True
                    break
        if task is None:
            task = self.scheduler.pick(self.filters, self.ell_stars,
                                       self.mode_policies,
                                       exclude=self.suspended
                                       | self.terminal)   # M27 F90
        if task is None:
            return None
        if task in self.terminal:
            # a terminal task is ONLY ever served retention or maintenance
            # (its gate may read unmastered after a boundary — that no
            # longer schedules training)
            mt, ml = self.filters[task].mean()
            mode, est = (SKILL if maintenance else RETENTION,
                         engine_to_plan(mt, ml))
        else:
            mode, est = self.mode_policies[task].choose_mode(
                self.filters[task], self.retention, now)
        cands = self._cands(task, feedback_safe=(mode != RETENTION))
        if len(cands) == 0:
            cands = self._cands(task, feedback_safe=False)
        # M16 D23: certification-probe override on the task's probe cadence
        # (unmastered tasks only — retention trials are never hijacked)
        if (len(cands) > 0
                and (maintenance                    # M27 F90 at-bar probe
                     or (self.probe_every is not None and mode != RETENTION
                         and self._probe_ctr[task] % self.probe_every
                         == self.probe_every - 1))):
            sig_hat, t_hat = est
            want = 1 if self._probe_flip[task] > 0 else 0
            self._probe_flip[task] *= -1
            m = np.where(cands.y_star == want)[0]
            if m.size == 0:
                m = np.arange(len(cands))
            sc = cert_probe_score(cands.s_mean[m], cands.s_sd[m],
                                  self.sigma_stars[task], t_hat)
            idx = int(m[np.argmax(sc)])
            a0 = at_bar_accuracy(cands.s_mean[idx], cands.s_sd[idx],
                                 self.sigma_stars[task], t_hat,
                                 int(cands.y_star[idx]))
            info = {"target": None, "want_label": want,
                    "cert_probe": True, "a0": a0,
                    "maintenance": maintenance}
        else:
            sel = self.mode_policies[task].select(mode, est, cands,
                                                  self.retention, now,
                                                  self.rng,
                                                  filt=self.filters[task])
            if sel is None:
                return None
            idx, info = sel
        return {
            "task": task, "mode": mode,
            "seg_id": int(cands.seg_id[idx]), "s": float(cands.s_mean[idx]),
            "s_sd": float(cands.s_sd[idx]), "y_star": int(cands.y_star[idx]),
            "margin": float(cands.margin[idx]), "info": info, "now": now,
        }

    def record(self, choice, y, *, session_id="train", rt_ms=np.nan,
               answer_changes=-1, feedback=True):
        """Apply the response: step the filter, update retention, log."""
        task, mode = choice["task"], choice["mode"]
        s, s_sd, y_star = choice["s"], choice["s_sd"], choice["y_star"]
        self.filters[task].step(s, int(y), y_star, s_sd=s_sd, feedback=feedback)
        self.mode_policies[task].note_served(mode, y_star)
        self.mode_policies[task].note_posterior(self.filters[task])
        # M16 D23: probe bookkeeping — cadence counter + e-gate update
        # (maintenance probes belong to the M27 stale gate, not the
        # declaration e-gate)
        if self.probe_every is not None:
            self._probe_ctr[task] += 1
            if (choice["info"].get("cert_probe")
                    and not choice["info"].get("maintenance")):
                self.mode_policies[task].egate.update(
                    int(y) == y_star, choice["info"]["a0"])
        # M27 (F90): every terminal-task outcome feeds the stale-mastery
        # e-gate; a Ville crossing revokes terminal status (the task
        # returns to the training rotation with a fresh gate on re-entry)
        if task in self.terminal:
            if choice["info"].get("maintenance"):
                self._maint_due[task] = max(self._maint_due[task] - 1, 0)
                a0_stale = choice["info"]["a0"]
            else:
                _, t_hat0 = engine_to_plan(*self.filters[task].mean())
                a0_stale = at_bar_accuracy(s, s_sd, self.sigma_stars[task],
                                           t_hat0, y_star)
            g = self._stale_gates.get(task)
            if g is not None:
                g.update(int(y) == y_star, a0_stale)
                if g.fired:
                    self._revoke_terminal(task)
        # retention bookkeeping (a terminal task's reviews all belong to
        # its consolidated (task, 0) schedule — M27 F90)
        if mode == RETENTION and "bin" in choice["info"]:
            bin_key = ((task, 0) if task in self.terminal
                       else choice["info"]["bin"])
            self.retention.update_on_retrieval(
                bin_key, correct=(int(y) == y_star), now=choice["now"])
        # register a review bin at the mastery TRANSITION only (registering on
        # every mastered trial proliferates bins → permanently-due queue that
        # starves unmastered tasks once retention serving exists — M10/F18)
        now_mastered = self.mode_policies[task].is_mastered(self.filters[task])
        if now_mastered:
            self.scheduler.mark_declared(task)   # M25 F86 lifecycle key
        if (now_mastered and not self._was_mastered[task]
                and task not in self.terminal):
            # (terminal tasks own ONE consolidated schedule — M27 F90;
            # gate flapping must not re-proliferate difficulty bins)
            est = engine_to_plan(*self.filters[task].mean())
            bin_key = (task, TaskModePolicy._bin_of(s, est))
            self.retention.register(bin_key, choice["now"])
        self._was_mastered[task] = now_mastered
        # fatigue/lapse probe flag (F15): very-easy item
        sig_hat, t_hat = engine_to_plan(*self.filters[task].mean())
        is_probe = abs(s - t_hat) / max(sig_hat, 1e-6) >= LAPSE_PROBE_MULT
        self.log.append(TrialRecord(
            session_id=session_id, trial_index=self._trial, task_k=task,
            seg_id=choice["seg_id"], s=s, s_sd=s_sd, y=int(y), y_star=y_star,
            feedback_given=feedback, mode=mode, rt_ms=float(rt_ms),
            answer_changes=int(answer_changes), t_epoch=float(choice["now"]),
            post_decision=bool(is_probe)))     # reuse flag as probe marker (proto)
        self._served.add(choice["seg_id"])
        self._trial += 1
