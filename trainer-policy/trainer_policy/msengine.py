"""The multi-signal learning engine (Q4/D36): the production form of M2.

Sits between the adaptive TESTING engine (testing-algo-cleaned: sequential
Bayesian particle cloud over per-dimension offsets and log-sensitivities)
and a certification exam. Consumes the handoff contract defined in
docs/handoff_contract.md:

  testing posterior  ->  belief seed        (t = -offset, sigma = exp(-l))
  item bank          ->  placement space    (their ItemCandidate schema)
  M1 population      ->  dynamics + floors  (rates, gate, floor dispersion)
  exam spec + eta    ->  readiness bar

State per belief particle: criteria t (M,), log-skills u (M,), and OWN
floors u_inf (M,) — the D33 floor-dispersion fix is built in, not bolted
on. Observation model and dynamics are the validated multiclass forms
(D30): lapse-mixed softmax over per-channel z, soft prediction errors,
Wilson (or learned) gate allocating skill updates across channels.

Placement (D24 generalized): each trial evaluates one candidate item per
channel (the bank item whose gold-channel z is nearest that channel's
gate peak under the current mean state) and asks which candidate buys the
largest expected one-step gain in the exam margin
    m(state) = (U - bar) / sqrt(U (1-U) / N_e),
U = expected exam accuracy under the exam case-mix. Readiness flags when
P[pass | belief] >= 1 - eta via the exact binomial survival per particle.
"""
# Copied from ideal-test-learning/sim/msengine.py on 2026-07-16 (D45 close-out state).
# Only import paths were rewritten for packaging; the code is otherwise verbatim.

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom

from .learnmodel import learn_weight_coef


def _softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# handoff ports (docs/handoff_contract.md)
# ---------------------------------------------------------------------------

def belief_seed_from_testing_result(result: dict, signals: list[str],
                                    n_particles: int, rng,
                                    cloud: dict | None = None):
    """Seed (t, u) particles from an adaptive-testing SessionResult dict.

    Preferred: `cloud` = the final captured particle cloud
    {response_offset (N,K), log_sensitivity (N,K), weights (N,)} — the full
    joint posterior including cross-dimension correlations survives the
    handoff. Fallback: the Gaussian summary in result["final_summary"].
    Mapping (contract section 3): t = -offset, u = -log_sensitivity."""
    names = list(result["dimension_names"])
    idx = [names.index(s) for s in signals]
    if cloud is not None:
        w = np.asarray(cloud["weights"], float)
        w = w / w.sum()
        pick = rng.choice(len(w), size=n_particles, p=w)
        t = -np.asarray(cloud["response_offset"], float)[pick][:, idx]
        u = -np.asarray(cloud["log_sensitivity"], float)[pick][:, idx]
        return t, u
    fs = result["final_summary"]
    om = np.asarray(fs["offset_mean"], float)[idx]
    ov = np.asarray(fs["offset_variance"], float)[idx]
    lm = np.asarray(fs["log_sensitivity_mean"], float)[idx]
    lv = np.asarray(fs["log_sensitivity_variance"], float)[idx]
    t = -(om + np.sqrt(ov) * rng.standard_normal((n_particles, len(idx))))
    u = -(lm + np.sqrt(lv) * rng.standard_normal((n_particles, len(idx))))
    return t, u


def prior_blocks_from_population(artifact: dict):
    """The reverse port: the M1 population posterior supplies the TESTING
    engine's declared prior blocks (contract section 5). The one-factor
    transfer structure becomes the cross-dimension covariance
    gamma gamma^T + diag(tau^2) on log-sensitivities."""
    mu_t0 = np.asarray(artifact["state_prior"]["mu_t0"])
    tau_t0 = np.asarray(artifact["state_prior"]["tau_t0"])
    mu_u0 = np.asarray(artifact["state_prior"]["mu_u0"])
    tau_u0 = np.asarray(artifact["state_prior"]["tau_u0"])
    gamma = np.asarray(artifact["state_prior"]["gamma"])
    return dict(
        offset_mean=(-mu_t0).tolist(),
        offset_covariance=np.diag(tau_t0 ** 2).tolist(),
        log_sensitivity_mean=(-mu_u0).tolist(),
        log_sensitivity_covariance=(np.outer(gamma, gamma)
                                    + np.diag(tau_u0 ** 2)).tolist(),
    )


# ---------------------------------------------------------------------------
# the belief
# ---------------------------------------------------------------------------

@dataclass
class MSBelief:
    """RBPF over per-particle (t, u, u_inf), all (N, M)."""

    artifact: dict
    signals: list[str]
    N: int = 600
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        art = self.artifact
        M = len(self.signals)
        self.M = M
        self.a_t = np.asarray(art["alpha_t"], float)
        self.a_s = np.asarray(art["alpha_s"], float)
        # rate dispersion (D42): particles carry their own learning rates
        # drawn from the population posterior, so a slower-than-average
        # learner is representable and evidence can select for it — the
        # D36 floor medicine extended to the dynamics. Without this the
        # skill update is clock-driven regardless of the response stream.
        for nm, key in (("a_t", "alpha_t_sd"), ("a_s", "alpha_s_sd")):
            sd = art.get(key)
            if sd is not None:
                mu = getattr(self, nm)
                draws = mu[None, :] + np.asarray(sd, float) \
                    * self.rng.standard_normal((self.N, M))
                setattr(self, nm, np.maximum(draws, 0.1 * mu[None, :]))
        self.lam = float(art["lam"])
        self.q_t = float(art.get("q_t", 0.0))
        self.q_s = float(art.get("q_s", 0.006))
        self.w_coef = art.get("w_coef")
        self.w_centers = art.get("w_centers")   # None = default basis
        sp = art["state_prior"]
        self.t = (np.asarray(sp["mu_t0"]) + np.asarray(sp["tau_t0"])
                  * self.rng.standard_normal((self.N, M)))
        # one-factor skill prior: shared ability + per-signal deviation
        eta = self.rng.standard_normal((self.N, 1))
        self.u = (np.asarray(sp["mu_u0"]) + np.asarray(sp["gamma"]) * eta
                  + np.asarray(sp["tau_u0"])
                  * self.rng.standard_normal((self.N, M)))
        self._draw_floors()
        self.w = np.full(self.N, 1.0 / self.N)
        self._seed_src = None    # posterior XOR replay guard (contract §2a)

    def _draw_floors(self):
        """Per-particle floors. With the artifact's `floor_joint` block
        (contract §4 v1.1) floors are drawn CONDITIONALLY on each
        particle's current skill u — the state–floor joint the population
        fit measured, whose severing at seeding was the D49
        attainability-optimism mechanism. Without the block: the marginal
        floor_prior draw, byte-identical to the legacy behavior."""
        fj = self.artifact.get("floor_joint")
        if fj is not None:
            self.u_inf = (np.asarray(fj["intercept"], float)
                          + np.asarray(fj["slope"], float) * self.u
                          + np.asarray(fj["resid_sd"], float)
                          * self.rng.standard_normal((self.N, self.M)))
        else:
            fp = self.artifact["floor_prior"]  # (mu (M,), sd (M,)) on log sigma_inf
            self.u_inf = (np.asarray(fp[0]) + np.asarray(fp[1])
                          * self.rng.standard_normal((self.N, self.M)))
        self.u_inf = np.minimum(self.u_inf, self.u - 1e-3)

    def seed_from_testing(self, result: dict, cloud: dict | None = None):
        """Replace the state prior with the testing engine's posterior.
        Mutually exclusive with seed_from_test_replay (contract §2a:
        the posterior and the raw stream encode the same responses)."""
        if getattr(self, "_seed_src", None) is not None:
            raise ValueError(
                f"belief already seeded ({self._seed_src}); posterior and "
                "replay seeding never combine (contract §2a)")
        self.t, self.u = belief_seed_from_testing_result(
            result, self.signals, self.N, self.rng, cloud)
        if self.artifact.get("floor_joint") is not None:
            self._draw_floors()  # re-condition floors on the seeded skill
        else:
            self.u_inf = np.minimum(self.u_inf, self.u - 1e-3)
        self.w = np.full(self.N, 1.0 / self.N)
        self._seed_src = "cloud" if cloud is not None else "summary"

    def seed_from_test_replay(self, trials):
        """Seed by replaying the EXACT pre-test sequence through the
        belief's own observation model (contract §2a): `trials` is the
        test's raw stream in served order, [(s_row, response_idx), ...]
        with FULL n-way picks (binarized replay measurably corrupts
        criterion estimates, exp_v0). Reweight-only — the test shows no
        feedback, so under the feedback-gated law the dynamics do not
        advance; and because the test chose each item from past
        responses only, the replayed likelihood is exact (adaptive
        selection is ignorable). Mutually exclusive with
        seed_from_testing (double counting). Records seed_diag."""
        if getattr(self, "_seed_src", None) is not None:
            raise ValueError(
                f"belief already seeded ({self._seed_src}); posterior and "
                "replay seeding never combine (contract §2a)")
        n = 0
        for s_row, y in trials:
            self.observe(np.asarray(s_row, float), int(y))
            n += 1
        self._seed_src = "replay"
        self.seed_diag = dict(src="replay", n_trials=n,
                              ess=float(1.0 / np.sum(self.w ** 2)))
        return self

    def _shrink(self, n):
        """Weighted subsample back to n particles — the contraction that
        follows an EXPANDED replay (the depletion guard: ~10^2
        reweight-only observations visibly deplete a production-size
        cloud, D49 mechanism 1). Surviving unique-ancestor count lands
        in seed_diag."""
        idx = self.rng.choice(self.N, size=n, p=self.w)
        for nm in ("t", "u", "u_inf"):
            setattr(self, nm, getattr(self, nm)[idx].copy())
        for nm in ("a_t", "a_s"):
            v = getattr(self, nm)
            if v.ndim == 2:
                setattr(self, nm, v[idx].copy())
        self.N = n
        self.w = np.full(n, 1.0 / n)
        if getattr(self, "seed_diag", None) is not None:
            self.seed_diag["n_unique"] = int(len(np.unique(idx)))
        return self

    # --- observation model (D30) ---
    def _probs(self, s_row):
        """(N, M) response probabilities for one item's evidence row."""
        z = (s_row[None, :] - self.t) / np.exp(self.u)
        p = self.lam / self.M + (1 - self.lam) * _softmax_rows(z)
        return z, p

    def _gate(self, z_abs):
        if self.w_coef is not None:
            # RAW basis expansion — the fit's convention (multimodel._gate):
            # the simplex fixes the scale and alpha_s carries the magnitude,
            # so no renormalization here (a batch-max rescale would erase
            # the cross-item relative weighting the gate exists to encode)
            return learn_weight_coef(z_abs, self.w_coef,
                                     centers=self.w_centers)
        return z_abs * np.exp(0.5 * (1.0 - z_abs ** 2))

    def observe(self, s_row, y_idx):
        """Reweight-only update for NO-FEEDBACK items (D43 pre-test
        seeding): the response is evidence about the state, but the
        learner receives no feedback, so the dynamics do not advance."""
        _, p = self._probs(np.asarray(s_row, float))
        lik = np.clip(p[:, y_idx], 1e-12, None)
        self.w = self.w * lik
        tot = self.w.sum()
        self.w = (np.full(self.N, 1.0 / self.N) if tot <= 0
                  else self.w / tot)
        if 1.0 / np.sum(self.w ** 2) < self.N / 2:
            idx = self.rng.choice(self.N, self.N, p=self.w)
            self.t, self.u = self.t[idx].copy(), self.u[idx].copy()
            self.u_inf = self.u_inf[idx].copy()
            if self.a_t.ndim == 2:
                self.a_t = self.a_t[idx].copy()
                self.a_s = self.a_s[idx].copy()
            self.w = np.full(self.N, 1.0 / self.N)

    def update(self, s_row, y_idx, gold_idx):
        """One answered item: reweight, then push the assumed dynamics."""
        z, p = self._probs(np.asarray(s_row, float))
        lik = np.clip(p[:, y_idx], 1e-12, None)
        self.w = self.w * lik
        tot = self.w.sum()
        self.w = (np.full(self.N, 1.0 / self.N) if tot <= 0
                  else self.w / tot)
        if 1.0 / np.sum(self.w ** 2) < self.N / 2:
            idx = self.rng.choice(self.N, self.N, p=self.w)
            self.t, self.u = self.t[idx].copy(), self.u[idx].copy()
            self.u_inf = self.u_inf[idx].copy()
            if self.a_t.ndim == 2:            # per-particle rates ride along
                self.a_t = self.a_t[idx].copy()
                self.a_s = self.a_s[idx].copy()
            self.w = np.full(self.N, 1.0 / self.N)
        ystar = np.zeros(self.M); ystar[gold_idx] = 1.0
        delta = p - ystar[None, :]
        g = self._gate(np.abs(z))
        self.t = self.t + self.a_t * delta
        self.u = self.u - self.a_s * g * (self.u - self.u_inf)
        if self.q_t > 0:
            self.t += self.rng.normal(0, self.q_t, self.t.shape)
        if self.q_s > 0:
            self.u += self.rng.normal(0, self.q_s, self.u.shape)

    # --- exam readiness (D24 generalized) ---
    def exam_accuracy(self, exam):
        """(N,) expected exam accuracy per particle."""
        acc = np.zeros(self.N)
        for s_row, gold in zip(exam["s"], exam["gold"]):
            _, p = self._probs(np.asarray(s_row, float))
            acc += p[:, gold]
        return acc / len(exam["gold"])

    def pass_prob(self, exam):
        a = np.clip(self.exam_accuracy(exam), 1e-9, 1 - 1e-9)
        return float(np.sum(self.w * binom.sf(exam["pass_count"] - 1,
                                              exam["n_items"], a)))

    # --- score prediction + accuracy calibration (D43) ---
    def _cal(self, a):
        """Apply the believed->realized accuracy correction (isotonic
        knots fitted on prior cohorts; None = identity)."""
        c = getattr(self, "calib", None)
        if c is None:
            return a
        return np.interp(a, np.asarray(c[0], float),
                         np.asarray(c[1], float))

    def pred_score(self, exam, cal=False):
        """Posterior-mean predicted exam accuracy (E[score]/n_items)."""
        a = self.exam_accuracy(exam)
        return float(np.sum(self.w * (self._cal(a) if cal else a)))

    def pass_prob_quantile_cal(self, exam, q):
        """Double-eta quantile on CALIBRATED per-particle accuracies
        (Signal B, D43)."""
        a = np.clip(self._cal(self.exam_accuracy(exam)), 1e-9, 1 - 1e-9)
        pp = binom.sf(exam["pass_count"] - 1, exam["n_items"], a)
        order = np.argsort(pp)
        cw = np.cumsum(self.w[order])
        return float(pp[order][np.searchsorted(cw, q)])

    def ready_cal(self, exam, eta):
        return self.pass_prob_quantile_cal(exam, eta) >= 1.0 - eta

    def forecast_score(self, exam, n_steps, z_place, cal=False):
        """Predicted exam accuracy after n_steps MORE questions, pushing
        the belief's own deterministic mean skill dynamics forward with
        items at |z| = z_place (the learned-optimal placement -> an
        upper bound on remaining gain), channels served in rotation.
        The criterion t is held fixed (its update is response-dependent
        and mean-zero at calibrated marginals). Signal A (D43) flags
        when forecast - current < one exam item. MEASURED DEFECT (D43):
        omits evidence-driven reweighting, so it under-promises the
        belief's own future gain and fires early — forecast_score_mc is
        the repair."""
        g_star = float(np.atleast_1d(self._gate(np.array([z_place])))[0])
        u = self.u.copy()
        for step in range(int(n_steps)):
            m = step % self.M
            a_m = self.a_s[:, m] if self.a_s.ndim == 2 else self.a_s[m]
            u[:, m] = u[:, m] - a_m * g_star * (u[:, m] - self.u_inf[:, m])
        saved = self.u
        self.u = u
        s = self.pred_score(exam, cal=cal)
        self.u = saved
        return s

    def _clone(self, seed=0):
        """Independent copy for forecast rollouts: shares the artifact,
        owns its state arrays and rng."""
        import copy
        b = copy.copy(self)
        b.rng = np.random.default_rng(seed)
        for nm in ("t", "u", "u_inf", "w", "a_t", "a_s"):
            setattr(b, nm, np.array(getattr(self, nm), copy=True))
        return b

    def forecast_score_mc(self, exam, n_steps, z_place, n_rollouts=6,
                          cal=False, seed=0):
        """Expected-posterior-path forecast (D45 item 2, the Signal A
        repair): simulate n_rollouts futures of n_steps questions at
        |z| = z_place (channels in rotation, off-gold evidence at the
        exam-typical level), SAMPLING each response from the belief's
        own predictive and reassimilating it through the full update
        (reweight + resample + dynamics) — so the forecast includes the
        evidence-driven particle selection that the deterministic push
        omits. Returns the mean pred_score at the horizon."""
        S = np.asarray(exam["s"], float)
        gold = np.asarray(exam["gold"], int)
        offm = np.ones_like(S, bool)
        offm[np.arange(len(gold)), gold] = False
        off = float(S[offm].mean())
        tot = 0.0
        for r in range(int(n_rollouts)):
            b = self._clone(seed=1000 * seed + r + 1)
            for step in range(int(n_steps)):
                m = step % self.M
                t_bar, u_bar = b.mean_state()
                s_row = np.full(self.M, off)
                s_row[m] = t_bar[m] + z_place * np.exp(u_bar[m])
                _, p = b._probs(s_row)
                pbar = b.w @ p
                y = int(b.rng.choice(self.M, p=pbar / pbar.sum()))
                b.update(s_row, y, m)
            tot += b.pred_score(exam, cal=cal)
        return tot / n_rollouts

    def margin(self, exam, acc=None):
        a = np.clip(self.exam_accuracy(exam) if acc is None else acc,
                    1e-9, 1 - 1e-9)
        bar = exam["pass_count"] / exam["n_items"]
        m = (a - bar) / np.sqrt(a * (1 - a) / exam["n_items"])
        return float(np.sum(self.w * m))

    def pass_prob_quantile(self, exam, q):
        a = np.clip(self.exam_accuracy(exam), 1e-9, 1 - 1e-9)
        pp = binom.sf(exam["pass_count"] - 1, exam["n_items"], a)
        order = np.argsort(pp)
        cw = np.cumsum(self.w[order])
        return float(pp[order][np.searchsorted(cw, q)])

    def ready(self, exam, eta):
        """Double-eta rule (D36): P(pass | theta) >= 1-eta must hold with
        posterior probability >= 1-eta. Floors are always dispersed here."""
        return self.pass_prob_quantile(exam, eta) >= 1.0 - eta

    # --- placement ---
    def mean_state(self):
        return (self.w @ self.t, self.w @ self.u)

    def margin_gain(self, s_row, gold_idx, exam):
        """Expected one-step exam-margin gain of presenting this item
        (deterministic soft push per particle — response-independent)."""
        z, p = self._probs(np.asarray(s_row, float))
        ystar = np.zeros(self.M); ystar[gold_idx] = 1.0
        delta = p - ystar[None, :]
        g = self._gate(np.abs(z))
        t2 = self.t + self.a_t * delta
        u2 = self.u - self.a_s * g * (self.u - self.u_inf)
        saved = self.t, self.u
        base = self.margin(exam)
        self.t, self.u = t2, u2
        gain = self.margin(exam) - base
        self.t, self.u = saved
        return gain


# ---------------------------------------------------------------------------
# the session engine
# ---------------------------------------------------------------------------

class MSSessionEngine:
    """Joint training over all signals against one exam.

    bank: list of dict(item_id, s (M,), gold (int)) — built from the same
    crowd-anchored axes the testing engine's ItemBank consumes."""

    def __init__(self, artifact, signals, bank, exam, eta, budget=400,
                 n_particles=600, zstar=1.0):
        self.art, self.signals = artifact, signals
        self.bank, self.exam, self.eta = bank, exam, eta
        self.budget, self.N, self.zstar = budget, n_particles, zstar
        # vectorized bank views (a per-trial python loop over a 5k bank
        # would dominate the session cost)
        self._ids = np.array([b["item_id"] for b in bank])
        self._S = np.stack([np.asarray(b["s"], float) for b in bank])
        self._gold = np.array([b["gold"] for b in bank])
        self._gold_s = self._S[np.arange(len(bank)), self._gold]
        self._id2j = {str(i): j for j, i in enumerate(self._ids)}

    def _candidates(self, belief, served_mask):
        """One candidate per channel: the unserved gold-m item whose
        gold-channel z under the mean state is nearest the gate peak."""
        t_bar, u_bar = belief.mean_state()
        z = (self._gold_s - t_bar[self._gold]) / np.exp(u_bar[self._gold])
        d = np.abs(np.abs(z) - self.zstar)
        d[served_mask] = np.inf
        out = []
        for m in range(belief.M):
            dm = np.where(self._gold == m, d, np.inf)
            j = int(np.argmin(dm))
            if np.isfinite(dm[j]):
                out.append(dict(item_id=self._ids[j], s=self._S[j],
                                gold=int(self._gold[j]), _j=j))
        return out

    def run(self, learner_step, seed=0, testing_result=None,
            testing_cloud=None, testing_trials=None, replay_expand=4,
            mastery=False, on_question=None):
        """learner_step(item) -> response index. Returns the session dict.

        Seeding (contract §2): testing_result (+ optional testing_cloud)
        = the posterior payload, OR testing_trials = the §2a raw trial
        stream ([{"k", "item_id", "response"}, ...]; a record may carry
        an explicit "s" row instead of item_id) replayed through the
        belief's own observation model on a replay_expand-times cloud,
        then contracted to N (the depletion guard). Never both (double
        counting). Replayed item_ids join the served set (no-repeat).
        mastery=False: stop at the first readiness flag (certification
        mode). mastery=True (D42): record the first flag but keep
        training to the full budget — the mode that maximizes the final
        exam score at a fixed question budget.
        on_question(k, belief): passive per-question observer (D45:
        logged-only signals — it must never stop the session)."""
        if testing_result is not None and testing_trials is not None:
            raise ValueError("seed with testing_result OR testing_trials, "
                             "never both (contract §2a double counting)")
        rng = np.random.default_rng(seed)
        served_mask = np.zeros(len(self.bank), bool)
        if testing_trials is not None:
            belief = MSBelief(self.art, self.signals,
                              N=self.N * int(replay_expand), rng=rng)
            pairs = []
            for tr in testing_trials:
                y = int(tr["response"] if "response" in tr else tr["y"])
                if "s" in tr:
                    pairs.append((np.asarray(tr["s"], float), y))
                    continue
                j = self._id2j.get(str(tr["item_id"]))
                if j is None:
                    raise KeyError(
                        f"replay item {tr['item_id']!r} not in the bank "
                        "and no explicit 's' row (contract §2a)")
                served_mask[j] = True
                pairs.append((self._S[j], y))
            belief.seed_from_test_replay(pairs)
            belief._shrink(self.N)
        else:
            belief = MSBelief(self.art, self.signals, N=self.N, rng=rng)
            if testing_result is not None:
                belief.seed_from_testing(testing_result, testing_cloud)
        log = []
        ready_at = -1
        n_used = 0
        for k in range(self.budget):
            if on_question is not None:
                on_question(k, belief)
            if (ready_at < 0) and belief.ready(self.exam, self.eta):
                ready_at = k
                if not mastery:
                    break
            cands = self._candidates(belief, served_mask)
            if not cands:
                break
            gains = [belief.margin_gain(c["s"], c["gold"], self.exam)
                     for c in cands]
            item = cands[int(np.argmax(gains))]
            y = int(learner_step(item))
            belief.update(item["s"], y, item["gold"])
            served_mask[item["_j"]] = True
            n_used = k + 1
            log.append(dict(k=k, item_id=str(item["item_id"]),
                            gold=int(item["gold"]), response=y))
        return dict(ready_at=ready_at,
                    n_used=(ready_at if (ready_at >= 0 and not mastery)
                            else n_used),
                    pass_prob=belief.pass_prob(self.exam),
                    log=log, belief=belief)
