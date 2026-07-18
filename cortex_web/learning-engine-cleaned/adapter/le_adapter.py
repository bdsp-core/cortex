"""Adapter: the cleaned learning engine behind the incumbent trainer's
surfaces (integration plan §4, P3).

Self-contained: imports learning_engine + numpy/scipy only (no JAX, no
imports from the host repo — the harness does the gluing), so it runs
under the host's single-thread-BLAS determinism regime. Coordinates:
the host exam emits engine coords (theta = offset, ell = log-skill);
the criterion sign-flip (t = -theta, u = -ell) happens HERE, in one
place.

Mastery semantics mirror the host's gates and take the host's design
inputs (cuts, alpha, Z, sd_floor) as ARGUMENTS — no constant is defined
in this module.
"""
import numpy as np

from learning_engine import MixedBelief, Registry


def seed_from_exam_cloud(belief, theta, ell, w, rng):
    """Replace the belief's state particles with draws from the host
    exam's (already inflated) particle cloud. theta/ell: (Np, M) in
    ENGINE coords; w: (Np,). Sign-flip applied here."""
    w = np.asarray(w, float)
    w = w / w.sum()
    pick = rng.choice(len(w), size=belief.N, p=w)
    belief.t = -np.asarray(theta, float)[pick]
    belief.u = -np.asarray(ell, float)[pick]
    belief.condition_floors_on_state()   # joint floors when the artifact
    # carries floor_joint (contract §4 v1.1); the plain clip otherwise
    belief.w = np.full(belief.N, 1.0 / belief.N)
    return belief


def seed_by_replay(belief, trials):
    """Seed by replaying the EXACT test sequence through the engine's
    own mixed-link observation model (roadmap §2.6). `belief` must be
    freshly constructed from the population prior; `trials` is the
    test log as [(item_dict, response_index), ...] with FULL n-way
    picks where the test recorded them. Reweight-only (`observe`) —
    correct for no-feedback tests under the feedback-gated law. Never
    combine with `seed_from_exam_cloud` (double counting)."""
    for item, y in trials:
        belief.observe(item, int(y))
    return belief


def run_training_session(policy, learner_step, budget, *,
                         on_question=None):
    """The production session structure (roadmap §2.1–2.3): report
    attainability first, then train until every domain is mastered or
    unattainable or candidates run out (the terminating triad) or the
    budget ends — the caller re-exams ONCE when this returns (readiness-
    triggered, not fixed-cadence). Returns the session report.

    A domain is DECLARED unattainable only when the ESS-adjusted upper
    bound on P(ceiling clears cut) is below alpha (the host's MCSE
    discipline; Z and alpha are the same design inputs the mastery gate
    uses) — a point estimate below alpha from a depleted particle cloud
    (e.g. after replay seeding) does not declare. Declared domains are
    frozen out for the whole session: reported, never served."""
    attain = {c: dict(p=policy.trainability(c),
                      unattainable=policy.unattainable(c))
              for c in policy.reg.codes}
    policy.frozen_unattainable = {
        c for c, a in attain.items() if a["unattainable"]}
    n = 0
    for k in range(int(budget)):
        if on_question is not None:
            on_question(k, policy)
        ch = policy.step()
        if ch is None:
            break
        y = learner_step(ch)
        policy.record(ch, y)
        n += 1
    return dict(attainability=attain, n_used=n,
                mastered={c: policy.is_mastered(c)
                          for c in policy.reg.codes},
                pass_mass={c: policy.pass_mass(c)
                           for c in policy.reg.codes})


class LETrainerPolicy:
    """The engine as a drop-in training policy: step() -> choice dict,
    record(choice, y) -> belief update. Serves per-domain binary items
    (the host's current training contract; native n-way serving is the
    UI-gated upgrade). Task selection: largest readiness deficiency
    1 - pass_mass among unmastered domains; item selection: asked-domain
    |z| nearest the gate peak under the mean state."""

    def __init__(self, belief, registry, ell_stars, candidates, *,
                 alpha, Z, sd_floor, exclude=None, zstar=1.0, rng=None,
                 case_mix=(), restrict=None, alloc="greedy", draw_k=1,
                 share_cap=None):
        self.bel, self.reg = belief, registry
        self.ell = dict(ell_stars)
        # Exposure-share cap (D62): no domain may exceed this fraction of the
        # session's served items — once it does, it drops out of eligibility
        # until other domains catch up. A NON-value guardrail: Thompson only
        # de-concentrates in proportion to posterior uncertainty, so a
        # cert-seeded (confident) belief still pins to one domain (the live
        # pilot: 85% share). This bounds share regardless of confidence.
        # None = off. Value-driven allocation still chooses AMONG the
        # under-cap domains, so learning efficiency degrades only at the
        # margin. Warmup lets the first few items go value-first.
        self.share_cap = share_cap
        # allocation mode across domains (D61): "greedy" = argmax expected
        # value-per-item (locks a session onto the worst domain — live-
        # exploitable, D54/D58); "thompson" = argmax value under ONE shared
        # posterior draw, so a domain is served with probability
        # P_posterior(it is the highest-value domain). Constant-free (the
        # belief's own uncertainty is the temperature — the §5/D26 doctrine
        # at item level); belief/placement math bit-identical either way.
        self.alloc = alloc
        self.draw_k = int(draw_k)   # 1 = pure Thompson; >1 = variance-reduced
        # roadmap §2.2: train the WEAK domains (the exam's non-PASS
        # set); None = all registry domains
        self.restrict = set(restrict) if restrict is not None else None
        self.case_mix = list(case_mix)   # declared evaluation mix for
        # the n-way reduction map (roadmap §3.3); binary domains need none
        self.cand = candidates          # callable(code) -> dict of arrays
        self.alpha, self.Z, self.sd_floor = alpha, Z, sd_floor
        self.served = set(exclude or ())
        self.zstar = zstar
        self.rng = rng or np.random.default_rng()
        self.log = []
        self.frozen_unattainable = set()
        self._side_ct = {}
        self._side_bal = {}   # per-domain served-label balance (+1/-1 sum)

    # --- mastery (host AD6 semantics on the reduced cloud) ---
    def _ell_cloud(self, code):
        return self.bel.ell_cloud(code, self.case_mix)

    def pass_mass(self, code):
        e = self._ell_cloud(code)
        return float(np.sum(self.bel.w * (e > self.ell[code])))

    def pass_mass_lcb(self, code):
        """ESS-adjusted lower confidence bound on pass-mass — the
        evidence-gated readiness statistic. Retiring a domain on the
        LCB (not the point mass) keeps prior-driven dynamics from
        declaring readiness the data cannot support (the same
        confident-declaration discipline as `unattainable`)."""
        pi = self.pass_mass(code)
        j = self.reg.index[code]
        ess = min(1.0 / np.sum(self.bel.w ** 2),
                  float(len(np.unique(self.bel.u[:, j]))))
        mcse = np.sqrt(max(pi * (1 - pi), 1e-12) / max(ess, 1.0))
        return pi - self.Z * mcse

    def is_mastered(self, code):
        e = self._ell_cloud(code)
        w = self.bel.w
        pi = float(np.sum(w * (e > self.ell[code])))
        ess = 1.0 / np.sum(w ** 2)
        mcse = np.sqrt(max(pi * (1 - pi), 1e-12) / ess)
        mu = float(np.sum(w * e))
        sd = float(np.sqrt(max(np.sum(w * e * e) - mu * mu, 0.0)))
        return (sd <= self.sd_floor
                and pi - self.Z * mcse >= 1.0 - self.alpha)

    # --- serving ---
    def trainability(self, code):
        """P(the skill ceiling clears the cut) from the floor posterior."""
        return self.bel.trainability(code, self.ell[code],
                                     items=self.case_mix)

    def unattainable(self, code):
        """Confident futility: the Z-sd upper bound on trainability is
        below alpha. ESS accounts for particle depletion (weight
        concentration AND duplicate collapse after resampling), so a
        noisy near-alpha point estimate withholds declaration."""
        pi = self.trainability(code)
        j = self.reg.index[code]
        ess = min(1.0 / np.sum(self.bel.w ** 2),
                  float(len(np.unique(self.bel.u_inf[:, j]))))
        mcse = np.sqrt(max(pi * (1 - pi), 1e-12) / max(ess, 1.0))
        return pi + self.Z * mcse < self.alpha

    def _expected_push(self, code, s):
        """Belief-expected skill gain of serving signal s on `code`:
        E_w[a_s * g(|z|) * (u - u_inf)] per particle — the §2.3
        value-per-item objective. Goes to zero as the tracked state
        approaches the floor posterior, so a saturated domain retires
        itself from the allocation without any threshold."""
        j = self.reg.index[code]
        b = self.bel
        z = (float(s) - b.t[:, j]) / np.exp(b.u[:, j])
        g = b._gate(np.abs(z))
        a = b.a_s[:, j] if b.a_s.ndim == 2 else b.a_s[j]
        return float(np.sum(b.w * a * g * (b.u[:, j] - b.u_inf[:, j])))

    def _draw_idx(self):
        """draw_k posterior-sample particle indices (~ w) — the shared
        draw that turns the value-per-item argmax into Thompson sampling."""
        return self.rng.choice(self.bel.N, size=self.draw_k, p=self.bel.w)

    def _push_at(self, code, s, idx):
        """a_s * g(|z|) * (u - u_inf) evaluated at the drawn particle
        set `idx` (D61): a single posterior sample of this domain's
        value-per-item (draw_k=1) — the k=1 noise IS the exploration,
        proportional to the belief's remaining uncertainty."""
        j = self.reg.index[code]
        b = self.bel
        z = (float(s) - b.t[idx, j]) / np.exp(b.u[idx, j])
        g = b._gate(np.abs(z))
        a = b.a_s[idx, j] if b.a_s.ndim == 2 else b.a_s[j]
        return float(np.mean(a * g * (b.u[idx, j] - b.u_inf[idx, j])))

    def _bias_at(self, code, idx):
        """a_t * |t| at the drawn particle set (D62): a posterior sample of
        this domain's criterion-correction magnitude — the bias-mode
        value-per-item, the analogue of `_push_at` for skill mode. The
        CONFIDENCE gate (|mean t| > Z*sd) stays on the mean elsewhere; only
        the choice AMONG confident-bias domains is drawn."""
        j = self.reg.index[code]
        b = self.bel
        a = b.a_t[idx, j] if b.a_t.ndim == 2 else b.a_t[j]
        return float(np.mean(a * np.abs(b.t[idx, j])))

    def _eligible(self):
        """Codes that may be served now, with their unserved candidate
        masks. alpha = 0 runs FULL-TRAINING mode: no readiness
        retirement, no futility — train the whole budget."""
        elig = {}
        for code in self.reg.codes:
            if self.restrict is not None and code not in self.restrict:
                continue
            if self.alpha > 0 and \
                    self.pass_mass_lcb(code) >= 1.0 - self.alpha:
                continue    # readiness: the belief CONFIDENTLY says
                # this domain would pass the exam bar at the declared
                # risk (LCB — evidence-gated, never prior-dynamics-led)
            if self.alpha > 0 and (code in self.frozen_unattainable
                                   or self.unattainable(code)):
                continue    # futility stop: ceiling confidently cannot
                # clear the cut at the declared risk — do not spend
                # exposure (session-frozen once declared at the report)
            c = self.cand(code)
            sid = np.asarray(c["seg_id"])
            if self.served:
                keep = ~np.isin(sid, np.fromiter(
                    self.served, dtype=np.int64, count=len(self.served)))
            else:
                keep = np.ones(len(sid), dtype=bool)
            if keep.any():
                elig[code] = (c, keep)
        # Exposure-share cap: drop domains already over their share, unless
        # that would empty the eligible set (at most 1/share_cap domains can
        # be over it, so with >= 2 eligible there is always an under-cap one).
        if self.share_cap is not None and len(elig) > 1:
            total = len(self.log)
            warmup = max(4, int(round(1.0 / self.share_cap)))
            if total >= warmup:
                counts = {}
                for row in self.log:
                    counts[row["task"]] = counts.get(row["task"], 0) + 1
                over = [code for code in elig
                        if counts.get(code, 0) / total > self.share_cap]
                if over and len(over) < len(elig):
                    for code in over:
                        del elig[code]
        return elig

    # Label-schedule randomization, second iteration (2026-07-17 live
    # findings). v1: strict ± alternation — the first real participant
    # learned the pattern in ~10 items (97% @ ~280 ms without reading the
    # media). v2 added a fair coin with a HARD imbalance bound — but the
    # balance walk hovers at the reflecting boundary, where every other
    # side is FORCED: measured live as 14-long half-predictable 1010
    # stretches (alternation rate 0.62 vs the coin's 0.50). v3 (current):
    # a pure iid fair coin — drawing the side first and serving the best
    # item within it is distribution-identical to "best positive vs best
    # negative, coin decides". Maximum-entropy sequence, zero forced
    # moves; the D12/D50 balance requirement is LONG-RUN and holds in
    # expectation (transient wander is the documented corrective — and
    # coherent pools keep the served labels tracking the drawn sides).
    # _side_bal stays as a logged diagnostic only. Same rng as the rest
    # of the policy → sessions stay seed-deterministic.
    def _side(self, code):
        return 1 if self.rng.random() < 0.5 else -1

    def _choice(self, code, c, keep, i_kept, mode):
        sid = np.asarray(c["seg_id"])
        ii = np.where(keep)[0][i_kept]
        self._side_ct[code] = self._side_ct.get(code, 0) + 1
        y_star = int(np.asarray(c["y_star"])[ii])
        self._side_bal[code] = (self._side_bal.get(code, 0)
                                + (1 if y_star == 1 else -1))
        return dict(task=code, seg_id=int(sid[ii]),
                    s=float(np.asarray(c["s"])[ii]),
                    s_sd=float(np.asarray(c.get(
                        "s_sd", np.zeros(len(sid))))[ii]),
                    y_star=y_star, mode=mode)

    def step(self):
        elig = self._eligible()
        if not elig:
            return None
        t_bar, u_bar = self.bel.mean_state()
        w = self.bel.w
        # BIAS MODE first, evidence-gated: fires for a domain only
        # when the belief is CONFIDENT the criterion is off
        # (|t_hat| > Z * posterior sd — reuses the existing Z, no new
        # constant; guards against chasing phantom offsets under
        # tracking noise, a failure mode measured when the gate was
        # removed). Placement is anchored at the POOL'S OWN label
        # boundary s ~ 0, alternating sign — for coherent pools the
        # base-rate drift restores t toward 0 regardless of belief
        # error (the D12 corrective anchored at the field-defined
        # boundary): it can waste an item, never anti-correct.
        # One shared posterior draw for BOTH modes (D61/D62): domain
        # selection in bias AND skill mode ranks by value under this draw,
        # so a domain is served with probability P(it is the highest-value
        # domain). Bias mode dominates sessions with a cert-seeded (sharp)
        # belief — leaving it greedy silently pinned the pilot to one
        # domain, D61's skill-only fix notwithstanding. idx=None => greedy
        # argmax over the mean (unchanged).
        idx = self._draw_idx() if self.alloc == "thompson" else None
        bias_code, bias_val = None, -np.inf
        for code in elig:
            j = self.reg.index[code]
            mu = float(w @ self.bel.t[:, j])
            sd = float(np.sqrt(max(
                w @ (self.bel.t[:, j] ** 2) - mu * mu, 1e-12)))
            if abs(mu) > self.Z * sd:              # evidence gate (on mean)
                a = self.bel.a_t
                val = (self._bias_at(code, idx) if idx is not None
                       else (float(w @ a[:, j]) if a.ndim == 2
                             else float(a[j])) * abs(mu))
                if bias_code is None or val > bias_val:
                    bias_code, bias_val = code, val
        if bias_code is not None:
            c, keep = elig[bias_code]
            s_all = np.asarray(c["s"])[keep]
            side = self._side(bias_code)
            ss = np.where(np.sign(s_all) == side, np.abs(s_all),
                          np.inf)
            if not np.isfinite(ss).any():
                ss = np.abs(s_all)
            return self._choice(bias_code, c, keep, int(np.argmin(ss)),
                                "bias")
        # SKILL MODE: balanced-sign placement at the gate peak
        # (alternate +z*, -z* per domain — sign balance keeps t = 0
        # attracting under the soft criterion update, the D12
        # fixed-point doctrine; feedback-safe pools are coherent so
        # the served side also balances labels). Allocation across
        # domains by expected value per item (greedy, the §2.3
        # objective) OR by value under one shared posterior draw
        # (thompson, D61) — saturated domains self-retire on the MEAN
        # push either way, so placement + retirement stay identical. Reuses
        # the single `idx` draw shared with bias mode above.
        best, best_gain, best_i = None, 0.0, None
        for code, (c, keep) in elig.items():
            j = self.reg.index[code]
            z = ((np.asarray(c["s"])[keep] - t_bar[j])
                 / np.exp(u_bar[j]))
            side = self._side(code)
            i = int(np.argmin(np.abs(z - side * self.zstar)))
            s_i = float(np.asarray(c["s"])[keep][i])
            mean_push = self._expected_push(code, s_i)
            if mean_push <= 0.0:
                continue                      # saturated: retire (mean-gated)
            rank = mean_push if idx is None else self._push_at(code, s_i, idx)
            if best is None or rank > best_gain:
                best, best_gain, best_i = code, rank, i
        if best is None:
            return None
        c, keep = elig[best]
        return self._choice(best, c, keep, best_i, "skill")

    def record(self, choice, y, feedback=True):
        item = dict(domain=choice["task"], s=choice["s"],
                    s_sd=choice.get("s_sd", 0.0) or None,
                    gold=int(choice["y_star"]))
        self.bel.update(item, int(y), feedback=feedback)
        self.served.add(choice["seg_id"])
        self.log.append(dict(choice, y=int(y)))
