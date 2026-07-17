"""Restricted-randomization label schedule (PECR: docs/LABEL_SCHEDULE_PECR.md).

Replaces the trainer's four deterministic label-sequence devices (strict
skill-side alternation, bias-mode balance forcing, the mirror "same
difficulty -> flipped label" echo, the cert-probe flip) with a randomized
walk primitive plus STATIC paired-bin serving (the mirror device), whose
guarantees are proven rather than assumed:

  * anchoring        E[side] = 0 (symmetric schedule),
  * bounded imbalance |D_n| <= cap (hard outer backstop),
  * bounded predictability: the OPTIMAL history-informed guesser's
    next-label accuracy G <= 0.55 (closed form on the exact chain below),
  * bounded criterion-band cost: stationary fluctuation-band inflation vs
    strict alternation <= 15% across the operating envelope (the
    schedule-general extension of `bias_stationary_sd`'s derivation).

DESIGN. Side process sigma_n in {-1,+1} with imbalance D_n = sum sigma and
signed run r_n. P(sigma=+1 | D=d, r) is a fair coin softly tilted against
imbalance (gamma_d per step beyond |d| >= d0) and against long runs (taper
gamma_r per step beyond |r| >= r0, floored at q_min so no interior state is
ever deterministic), hard-forced only at |d| = cap. Soft tilts are the whole
point: every HARD rule is a free win for the optimal guesser (a strict
alternation is 100% guessable; a hard run-cap-3 alone costs G ~ 0.59), while
soft pressure buys the same variance control at a fraction of the
predictability price. Shipped default (cap=8, d0=3, gamma_d=0.04, r0=3,
gamma_r=0.04, q_min=0.25): G = 0.5485, worst-regime band inflation 9.8%,
P(run >= 5) = 3.2%, P(|D| >= 6) = 19%.

WHY THE BAND TOLERATES RANDOMIZATION (the load-bearing math). Linearizing
the soft R-W criterion recursion at t = 0 under boundary-anchored serving,
    t' = (1-kappa) t - alpha_t*delta_bar*sigma_n + alpha_t*eps_n + xi_n,
the stationary variance splits into a schedule-independent diffusion term
and  alpha_t^2 delta_bar^2 * V[sigma]  with
    V[sigma] = (1/2pi) integral |H(w)|^2 f_sigma(w) dw,
H the AR(1) transfer function (low-pass). Strict alternation concentrates
all side-process spectral mass at the Nyquist frequency (maximally
attenuated, V = 1/(2-kappa)^2 — exactly `bias_stationary_sd`'s intra-pair
term); iid sides are flat (V = 1/(kappa(2-kappa))). At the shipped dynamics
(alpha_t 0.097-0.15, q_t 0.05) even the iid extreme inflates the TOTAL band
only ~11%, so a mildly anti-persistent walk sits well inside the 15% gate;
the binding constraint is G vs imbalance control, not the band.

PARITY. Randomness is a counter-based splitmix64 stream keyed by
(seed, stream_id, counter) — bit-identical to the splitmix64 already in
cortex_web/apps/web/engine/rng.ts, so the TS port reproduces every schedule
decision exactly (golden cross-language vectors in the tests). The numpy
Generator threaded through TrainerPolicy is NOT consumed by any schedule
decision; legacy paths are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_M64 = (1 << 64) - 1
_PHI = 0x9E3779B97F4A7C15          # splitmix64 increment (matches rng.ts)
_STREAM_SALT = 0xD1B54A32D192ED03  # decorrelates stream_id from seed


def sm64(x):
    """One splitmix64 output step (state x + PHI, then the finalizer) —
    bit-identical to `splitmix64` in cortex_web/apps/web/engine/rng.ts."""
    x = (x + _PHI) & _M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return z ^ (z >> 31)


class ScheduleRng:
    """Counter-based uniform stream: draw n is a pure function of
    (seed, stream_id, n) — resumable, order-robust, trivially portable."""

    def __init__(self, seed, stream_id=0):
        self._key = sm64((int(seed) & _M64) ^ ((int(stream_id) * _STREAM_SALT) & _M64))
        self._n = 0

    def uniform(self):
        """U[0,1) with 53-bit mantissa (same >> 11 / 2^53 as rng.ts)."""
        u = sm64((self._key + self._n * _PHI) & _M64)
        self._n += 1
        return (u >> 11) / 9007199254740992.0  # 2^53


@dataclass(frozen=True)
class ScheduleParams:
    """Shipped default = the gate-feasible knee of the exact frontier
    (docs/LABEL_SCHEDULE_PECR.md §2-3)."""
    cap: int = 8          # hard imbalance backstop (only deterministic states)
    d0: int = 3           # soft anti-imbalance tilt starts at |D| >= d0
    gamma_d: float = 0.04 # tilt per step of |D| beyond d0... (applied flat)
    r0: int = 3           # soft anti-run taper starts at run length r0
    gamma_r: float = 0.04 # taper strength per run step beyond r0-1
    q_min: float = 0.25   # continue-probability floor (no interior certainty)


# The probe/corrective TYPE flip is not a label stream (no G-A gate); it
# needs tight dual-control bounds instead: cap 2 => probe fraction -> 1/2
# with same-type runs <= 4, soft-tilted toward alternation inside.
FLIP_PARAMS = ScheduleParams(cap=2, d0=1, gamma_d=0.15, r0=2, gamma_r=0.15,
                             q_min=0.30)


def p_side_up(d, r, p: ScheduleParams):
    """P(next side = +1 | imbalance d, signed run r). Pure — shared by the
    runtime scheduler, the exact chain analysis, and the TS port."""
    if d <= -p.cap:
        return 1.0
    if d >= p.cap:
        return 0.0
    pr = 0.5
    if abs(d) >= p.d0:
        pr -= (1.0 if d > 0 else -1.0) * p.gamma_d
    j = abs(r)
    if j >= p.r0:
        t = p.gamma_r * (j - p.r0 + 1)
        pr += (-t if r > 0 else t)
    if r > 0:
        pr = max(pr, p.q_min)
    elif r < 0:
        pr = min(pr, 1.0 - p.q_min)
    return float(min(max(pr, 0.0), 1.0))


class SideScheduler:
    """The runtime walk. `propose()` draws the next side (one counter tick);
    `commit(side)` advances the state with the side actually SERVED — which
    may deviate from the proposal (pool-exhaustion fallback), so the walk
    always tracks the true served-label imbalance. One instance per task:
    the learner sees one label stream per task across all modes, so skill
    sides, bias wants, and cert-probe wants all consult the same walk."""

    def __init__(self, params: ScheduleParams = None, rng: ScheduleRng = None):
        self.p = params or ScheduleParams()
        self.rng = rng or ScheduleRng(0)
        self.d = 0
        self.run = 0                      # signed: +j = last j sides were +1
        self.clamped = 0                  # cap-discarded commits (telemetry)

    def propose(self):
        pu = p_side_up(self.d, self.run, self.p)
        return 1 if self.rng.uniform() < pu else -1

    def commit(self, side):
        # The walk state is a BOUNDED controller state, not an unbounded
        # counter: commits are proposal-gated by the policy (retention and
        # other non-scheduled serving never feed the walk), so the clamp
        # binds only under sustained pool-exhaustion deviation; `clamped`
        # counts those events (gated ≈ 0 in the test battery's runs).
        side = 1 if side > 0 else -1
        d_new = self.d + side
        if abs(d_new) > self.p.cap:
            self.clamped += 1
            d_new = max(min(d_new, self.p.cap), -self.p.cap)
        self.d = d_new
        self.run = (self.run + side) if (self.run * side) > 0 else side


BIN_WIDTH = 0.2      # |s| bin width for paired-bin serving (PECR §5c)


def paired_bin_choice(s_mean, y_star, want, target_mag, *, width=BIN_WIDTH,
                      rho_mag=0.5, s_sd=None):
    """Round-3 mirror device (PECR §5c) — STATIC paired-bin serving.

    The legacy mirror pairing existed to kill the bank-asymmetry DC offset
    (precise positives ≈ +0.96 vs precise negatives ≈ −0.55 herd the
    criterion to the served-stream midpoint — M10). Round 1 (debt queue)
    and round 2 (EWMA equalizer) both chased that symmetry through
    HISTORY-DEPENDENT magnitude coupling, and both leaked the label
    through perceived difficulty (0.568 / 0.61–0.66 measured) — the bank
    asymmetry is STATIC, so any dynamic compensator either under-corrects
    or side-splits its targets. This device is history-free:

      * bin the candidate pool by |s| (fixed width);
      * eligible bins = those where BOTH labels have items ("paired");
      * choose the bin SIDE-BLIND — argmax over paired bins of
        magnitude-profile × precision-availability,
            w(b) = exp(−(mid_b − target)² / (2·(rho_mag·target)²))
                   / (1 + max_side(min in-bin-side s_sd)²),
        a symmetric function of the two sides (no label information):
        the precision factor is the F20 lesson re-applied at the bin
        level — a bin is only as discriminating as its LESS-precise
        side, and serving s_sd-blurred items flatters a just-below-cut
        learner (measured: dropping precision-awareness inflated
        sharp-margin false graduation 0.60 → 0.80);
      * the caller serves the WANTED side inside that bin by its usual
        precision-aware score (E[w]) — side-legitimate inside a
        side-blind bin.

    Served MAGNITUDES then match across sides to within one bin width BY
    CONSTRUCTION: the |s|-difficulty channel has no dynamic component to
    attack and its static residual is ≤ width (measured: within-bin side
    gap ≤ 0.02, all adversaries at chance incl. the real bank). SCOPE
    (critique-3 F1/F2): the construction bounds the |s| family only —
    other perceivable served features are bounded by MEASUREMENT, and one
    is materially open: on the real bank, item precision s_sd is
    label-correlated at the POOL level (feedback-safe positives precise,
    negatives noisy on 6/7 tasks, task 0 reversed; label-correlated on
    all 7), so a perceived-ambiguity guesser reads 0.935 on the LEGACY
    trainer and 0.998 here (both saturated) — a bank-content limitation
    no serving policy can close (precise negatives are near-absent,
    0-3 per measured task), recorded as an owner item in the PECR
    ledger (§6.1).

    `want` is accepted but deliberately UNUSED — that unusedness IS the
    side-blindness property (locked by the want-flip equality assertion
    in test_paired_bin_choice); do not "optimize" it into the choice.

    Returns a boolean mask over the candidates (the chosen bin — label
    masking is the caller's job), or None when no paired bin exists
    (degenerate pool ⇒ caller falls back to the legacy unconstrained
    argmax)."""
    mags = np.abs(np.asarray(s_mean, dtype=np.float64))
    labels = np.asarray(y_star)
    bins = np.floor(mags / float(width)).astype(np.int64)
    pos = set(np.unique(bins[labels == 1]).tolist())
    neg = set(np.unique(bins[labels == 0]).tolist())
    paired = pos & neg
    if not paired:
        return None
    t = float(target_mag)
    sd = (None if s_sd is None
          else np.asarray(s_sd, dtype=np.float64))
    best, best_w = None, -1.0
    for b in sorted(paired):   # sorted: deterministic across languages
        mid = (b + 0.5) * float(width)
        w = np.exp(-((mid - t) ** 2) / (2.0 * (rho_mag * max(t, 0.3)) ** 2))
        if sd is not None:
            in_b = bins == b
            worst_side_sd = max(
                float(sd[in_b & (labels == 1)].min()),
                float(sd[in_b & (labels == 0)].min()))
            w /= 1.0 + worst_side_sd ** 2
        if w > best_w:
            best, best_w = b, w
    return bins == best


# ─────────────────── exact analysis (design + gates) ───────────────────

def _chain(params: ScheduleParams, r_sat=None):
    """Transition structure of the augmented (D, signed run) chain."""
    p = params
    r_sat = r_sat or max(p.r0 + int(np.ceil((0.5 - p.q_min) / max(p.gamma_r, 1e-9))) + 1, 4)
    states = [(d, r) for d in range(-p.cap, p.cap + 1)
              for r in list(range(-r_sat, 0)) + list(range(1, r_sat + 1))]
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))
    pu = np.zeros(n)
    nxt_up = np.full(n, -1, dtype=np.int64)
    nxt_dn = np.full(n, -1, dtype=np.int64)
    for i, (d, r) in enumerate(states):
        pu[i] = p_side_up(d, min(max(r, -r_sat), r_sat), p)
        up = (min(d + 1, p.cap), min((r + 1) if r > 0 else 1, r_sat))
        dn = (max(d - 1, -p.cap), max((r - 1) if r < 0 else -1, -r_sat))
        if pu[i] > 0:
            P[i, index[up]] += pu[i]
            nxt_up[i] = index[up]
        if pu[i] < 1:
            P[i, index[dn]] += 1.0 - pu[i]
            nxt_dn[i] = index[dn]
    return states, pu, P, nxt_up, nxt_dn


def _stationary(P):
    w, v = np.linalg.eig(P.T)
    pi = np.abs(np.real(v[:, np.argmin(np.abs(w - 1.0))]))
    return pi / pi.sum()


def guess_rate(params: ScheduleParams = None):
    """Optimal history-informed guesser's stationary next-label accuracy.
    Label history determines (D, r) exactly, so this IS the adversary
    bound: G = sum_state pi(state) * max(p_up, 1-p_up). Gate G-A: <= 0.55."""
    p = params or ScheduleParams()
    _, pu, P, _, _ = _chain(p)
    pi = _stationary(P)
    return float((pi * np.maximum(pu, 1.0 - pu)).sum())


_RHO_CACHE: dict = {}


def side_autocov(params: ScheduleParams = None, H=6000):
    """Autocovariance rho(h), h = 0..H, of the stationary side process."""
    p = params or ScheduleParams()
    key = (p, H)
    if key in _RHO_CACHE:
        return _RHO_CACHE[key]
    _, pu, P, nup, ndn = _chain(p)
    pi = _stationary(P)
    mu = 2.0 * pu - 1.0
    up_w = pi * pu                       # weight of an up-step from state i
    dn_w = pi * (1.0 - pu)
    up_ok = nup >= 0
    dn_ok = ndn >= 0
    rho = np.zeros(H + 1)
    rho[0] = 1.0
    ph_mu = mu.copy()
    for h in range(1, H + 1):
        rho[h] = ((up_w[up_ok] * ph_mu[nup[up_ok]]).sum()
                  - (dn_w[dn_ok] * ph_mu[ndn[dn_ok]]).sum())
        ph_mu = P @ ph_mu
    _RHO_CACHE[key] = rho
    return rho


def v_factor(kappa, params: ScheduleParams = None, H=6000, _rho=None):
    """Schedule variance factor V = [1 + 2 sum beta^h rho(h)]/(1-beta^2),
    beta = 1-kappa: the low-pass-weighted spectral mass of the side
    process. Alternation limit: 1/(2-kappa)^2; iid: 1/(kappa(2-kappa))."""
    rho = side_autocov(params, H=H) if _rho is None else _rho
    beta = 1.0 - float(kappa)
    h = np.arange(1, len(rho))
    return float((1.0 + 2.0 * (beta ** h * rho[1:]).sum()) / (1.0 - beta ** 2))


def stationary_sd_scheduled(alpha_t, q_t, sigma_hat, *, params=None,
                            item_s_sd=0.3, offset_mult=None, lapse=0.025,
                            v_override=None):
    """Schedule-general stationary criterion SD under boundary-anchored
    serving — `bias_stationary_sd`'s derivation with the strict-alternation
    V replaced by the schedule's exact V. With
    v_override = 1/(2-kappa)^2 it reproduces `bias_stationary_sd`
    EXACTLY (locked by test); `bias_stationary_sd` itself is untouched and
    remains the derived-t* authority."""
    from scipy.stats import norm as _norm

    from trainer.conventions import SKILL_MODE_MULTIPLIER
    m = SKILL_MODE_MULTIPLIER if offset_mult is None else offset_mult
    sig_e = np.sqrt(sigma_hat ** 2 + item_s_sd ** 2)
    a = m * sigma_hat
    kappa = min(alpha_t * (1.0 - 2.0 * lapse)
                * float(_norm.pdf(a / sig_e)) / sig_e, 0.9)
    x, w = np.polynomial.hermite.hermgauss(9)
    wn = w / np.sqrt(np.pi)
    z = (a + np.sqrt(2.0) * item_s_sd * x) / sigma_hat
    p_nodes = lapse + (1.0 - 2.0 * lapse) * _norm.cdf(z)
    p_bar = float((wn * p_nodes).sum())
    var_p = float((wn * p_nodes ** 2).sum() - p_bar ** 2)
    delta_bar = 1.0 - p_bar
    V = v_factor(kappa, params) if v_override is None else float(v_override)
    var_stat = ((q_t ** 2 + alpha_t ** 2 * var_p) / (kappa * (2.0 - kappa))
                + (alpha_t * delta_bar) ** 2 * V)
    return float(np.sqrt(var_stat))
