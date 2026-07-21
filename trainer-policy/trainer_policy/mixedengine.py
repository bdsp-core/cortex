"""Mixed-link session engine (integration plan §3, P1.2).

Registry-driven generalization of the production engine: one learner
state (t, u, u_inf) over ALL registry domains, two observation links —
binary probit for detection domains, lapse-mixed softmax for n-way
groups — and the SAME learning law in both (soft prediction error on
the criterion; gated relaxation of log-skill toward the floor, gate
evaluated at each domain's own |z|). A binary trial updates its domain;
an n-way trial updates its group's domains. Nothing here assumes a
domain count, a group count, or a group size.

Artifact schema (extends the package README's schema):
  codes         ordered domain codes the per-domain arrays align to
  alpha_t/alpha_s (+_sd), lam, q_t, q_s, w_coef, w_centers   as before
  state_prior   mu_t0, tau_t0, mu_u0, tau_u0, gamma   (per domain)
  floor_prior   (mu, sd) per domain
  hyper         POPULATION HYPERPRIOR block (plan §3.5): scalars
                mu_alpha_t, sd_alpha_t, mu_alpha_s, sd_alpha_s,
                mu_t0, tau_t0, mu_u0, tau_u0, gamma, floor_mu, floor_sd
COLD-START: a registry domain absent from `codes` is instantiated from
`hyper` by partial pooling — it enters with the population distribution
and earns domain-specific values at the next refit. No new constants.
"""
import numpy as np
from scipy.special import ndtr
from scipy.stats import binom

from .learnmodel import learn_weight_coef
from .registry import ell_reduced


def floor_joint_from_draws(u0_draws, ui_draws):
    """Per-domain regression of the floor u_inf on the starting skill u0
    across POOLED posterior draws (draws x participants — the D36
    tail-coverage convention). Pooling mixes posterior noise into the
    regressor, which attenuates the slope: the conservative direction
    (under-conditioning widens floors, never narrows them beyond what
    the measurement supports). Returns the contract §4 `floor_joint`
    block {slope, intercept, resid_sd}, each length M."""
    u0_draws = np.asarray(u0_draws)
    ui_draws = np.asarray(ui_draws)
    M = u0_draws.shape[-1]
    slope, intercept, resid = [], [], []
    for m in range(M):
        x = u0_draws[..., m].ravel()
        y = ui_draws[..., m].ravel()
        vx = float(x.var())
        a = (float(((x - x.mean()) * (y - y.mean())).mean() / vx)
             if vx > 0 else 0.0)
        b = float(y.mean() - a * x.mean())
        r = float(np.sqrt(max(((y - a * x - b) ** 2).mean(), 1e-12)))
        slope.append(a); intercept.append(b); resid.append(r)
    return dict(slope=slope, intercept=intercept, resid_sd=resid)


def artifact_from_posterior(post, codes):
    """Population posterior -> the deployable artifact, INCLUDING the
    population hyperprior block that cold-starts ADDED domains (plan
    §3.5). Per-domain values are posterior means; the hyper block is
    the population distribution of those values across fitted domains
    (partial pooling: a new domain enters with the population spread
    and earns its own values at the next refit)."""
    a_t = np.asarray(post["alpha_t"]).mean(0)
    a_s = np.asarray(post["alpha_s"]).mean(0)
    t0 = np.asarray(post["t0"]).mean(0)      # (L, M)
    u0 = np.asarray(post["u0"]).mean(0)
    ui = np.asarray(post["u_inf"])            # (draws, L, M)
    M = t0.shape[-1]
    sp = dict(mu_t0=t0.mean(0).tolist(), tau_t0=t0.std(0).tolist(),
              mu_u0=u0.mean(0).tolist(), tau_u0=u0.std(0).tolist(),
              gamma=np.asarray(post["gamma"]).mean(0).tolist())
    fmu = [float(ui[:, :, m].ravel().mean()) for m in range(M)]
    fsd = [float(ui[:, :, m].ravel().std()) for m in range(M)]
    hyper = dict(
        mu_alpha_t=float(a_t.mean()), sd_alpha_t=float(a_t.std()),
        mu_alpha_s=float(a_s.mean()), sd_alpha_s=float(a_s.std()),
        mu_t0=float(np.mean(sp["mu_t0"])),
        tau_t0=float(np.mean(sp["tau_t0"])),
        mu_u0=float(np.mean(sp["mu_u0"])),
        tau_u0=float(np.mean(sp["tau_u0"])),
        gamma=float(np.mean(sp["gamma"])),
        floor_mu=float(np.mean(fmu)), floor_sd=float(np.mean(fsd)))
    return dict(codes=list(codes), alpha_t=a_t.tolist(),
                alpha_s=a_s.tolist(),
                lam=float(np.median(np.asarray(post["lam"]).mean(0))),
                q_t=0.0, q_s=0.006, state_prior=sp,
                floor_prior=(fmu, fsd),
                floor_joint=floor_joint_from_draws(post["u0"], ui),
                hyper=hyper)


def _col(art, key, code, hyper_key):
    """Per-domain artifact value with hyperprior cold-start fallback."""
    codes = list(art.get("codes", []))
    if code in codes:
        return float(np.asarray(art[key], float)[codes.index(code)])
    return float(art["hyper"][hyper_key])


class MixedBelief:
    """Particle belief over (t, u, u_inf) across the registry domains."""

    def __init__(self, artifact, registry, N=400, rng=None):
        self.art, self.reg = artifact, registry
        self.N, self.rng = N, rng or np.random.default_rng()
        self.lam = float(artifact["lam"])
        self.q_t = float(artifact.get("q_t", 0.0))
        self.q_s = float(artifact.get("q_s", 0.006))
        self.w_coef = artifact.get("w_coef")
        self.w_centers = artifact.get("w_centers")
        M = registry.M
        h = artifact.get("hyper", {})
        codes = list(artifact.get("codes", []))

        def per_dom(key, sp_key, hyper_key, block="state_prior"):
            out = np.empty(M)
            for i, c in enumerate(registry.codes):
                if c in codes:
                    out[i] = float(np.asarray(
                        artifact[block][sp_key], float)[codes.index(c)])
                else:
                    out[i] = float(h[hyper_key])
            return out
        mu_t0 = per_dom("t", "mu_t0", "mu_t0")
        tau_t0 = per_dom("t", "tau_t0", "tau_t0")
        mu_u0 = per_dom("u", "mu_u0", "mu_u0")
        tau_u0 = per_dom("u", "tau_u0", "tau_u0")
        gamma = per_dom("g", "gamma", "gamma")
        fmu = np.array([_col(artifact, "_f0", c, "floor_mu")
                        if c not in codes else
                        float(np.asarray(artifact["floor_prior"][0],
                                         float)[codes.index(c)])
                        for c in registry.codes])
        fsd = np.array([float(h.get("floor_sd", 0.3))
                        if c not in codes else
                        float(np.asarray(artifact["floor_prior"][1],
                                         float)[codes.index(c)])
                        for c in registry.codes])
        self.a_t = np.array([_col(artifact, "alpha_t", c, "mu_alpha_t")
                             for c in registry.codes])
        self.a_s = np.array([_col(artifact, "alpha_s", c, "mu_alpha_s")
                             for c in registry.codes])
        for nm, key, hk in (("a_t", "alpha_t_sd", "sd_alpha_t"),
                            ("a_s", "alpha_s_sd", "sd_alpha_s")):
            sd = artifact.get(key)
            hsd = h.get(hk)
            if sd is not None or hsd is not None:
                mu = getattr(self, nm)
                sdv = np.array([
                    float(np.asarray(sd, float)[codes.index(c)])
                    if (sd is not None and c in codes)
                    else float(hsd if hsd is not None else 0.0)
                    for c in registry.codes])
                draws = mu[None, :] + sdv[None, :] \
                    * self.rng.standard_normal((N, M))
                setattr(self, nm, np.maximum(draws, 0.1 * mu[None, :]))
        eta = self.rng.standard_normal((N, 1))
        self.t = mu_t0 + tau_t0 * self.rng.standard_normal((N, M))
        self.u = (mu_u0 + gamma * eta
                  + tau_u0 * self.rng.standard_normal((N, M)))
        self.u_inf = fmu + fsd * self.rng.standard_normal((N, M))
        self.condition_floors_on_state()   # joint floors when the
        # artifact carries floor_joint (contract §4 v1.1); plain clip
        # otherwise — the legacy rng stream is unchanged either way up
        # to this point, so artifacts without the block reproduce the
        # historical draws exactly
        self.w = np.full(N, 1.0 / N)

    def condition_floors_on_state(self):
        """Redraw per-particle floors from the artifact's `floor_joint`
        block conditional on the CURRENT skill particles (contract §4
        v1.1). Call after any seeding that replaces u wholesale — cloud
        replacement severs the state–floor joint, the D49
        attainability-optimism mechanism. Cold-start domains (absent
        from the artifact's codes) keep their hyper marginal draw.
        Without the block this is only the floor-below-skill clip."""
        fj = self.art.get("floor_joint")
        if fj is not None:
            codes = list(self.art.get("codes", []))
            for i, c in enumerate(self.reg.codes):
                if c in codes:
                    k = codes.index(c)
                    self.u_inf[:, i] = (
                        float(fj["intercept"][k])
                        + float(fj["slope"][k]) * self.u[:, i]
                        + float(fj["resid_sd"][k])
                        * self.rng.standard_normal(self.N))
        self.u_inf = np.minimum(self.u_inf, self.u - 1e-3)
        return self

    # --- observation ---
    def _p_item(self, item):
        """(dims, z, p_response) for one item; p_response is (N, R)
        over the item's response alphabet (2 for binary, G for n-way)."""
        dims = self.reg.item_dims(item)
        if "domain" in item:
            j = dims[0]
            z = (float(item["s"]) - self.t[:, j]) / np.exp(self.u[:, j])
            sd = item.get("s_sd")
            if sd:
                z = z / np.sqrt(1 + (float(sd) / np.exp(self.u[:, j])) ** 2)
            p1 = self.lam + (1 - 2 * self.lam) * ndtr(z)
            return dims, z[:, None], np.stack([1 - p1, p1], axis=1)
        s = np.asarray(item["s"], float)
        z = (s[None, :] - self.t[:, dims]) / np.exp(self.u[:, dims])
        sd = item.get("s_sd")
        if sd is not None:
            z = z / np.sqrt(1 + (np.asarray(sd, float)[None, :]
                                 / np.exp(self.u[:, dims])) ** 2)
        zz = z - z.max(axis=1, keepdims=True)
        e = np.exp(zz)
        p = self.lam / len(dims) + (1 - self.lam) * e / e.sum(
            axis=1, keepdims=True)
        return dims, z, p

    def _reweight(self, lik):
        self.w = self.w * np.clip(lik, 1e-12, None)
        tot = self.w.sum()
        self.w = (np.full(self.N, 1.0 / self.N) if tot <= 0
                  else self.w / tot)
        if 1.0 / np.sum(self.w ** 2) < self.N / 2:
            idx = self.rng.choice(self.N, self.N, p=self.w)
            for nm in ("t", "u", "u_inf"):
                setattr(self, nm, getattr(self, nm)[idx].copy())
            for nm in ("a_t", "a_s"):
                v = getattr(self, nm)
                if v.ndim == 2:
                    setattr(self, nm, v[idx].copy())
            self.w = np.full(self.N, 1.0 / self.N)

    def _gate(self, z_abs):
        if self.w_coef is not None:
            return learn_weight_coef(z_abs, self.w_coef,
                                     centers=self.w_centers)
        return z_abs * np.exp(0.5 * (1.0 - z_abs ** 2))

    def observe(self, item, y):
        """Reweight-only (no-feedback trials)."""
        dims, z, p = self._p_item(item)
        self._reweight(p[:, int(y)])

    def update(self, item, y, feedback=True):
        """One answered item: reweight, then (if feedback was shown)
        push the assumed learning dynamics on the item's domains."""
        dims, z, p = self._p_item(item)
        self._reweight(p[:, int(y)])
        if not feedback:
            return
        dims, z, p = self._p_item(item)     # post-resample state
        a_t = self.a_t[:, dims] if self.a_t.ndim == 2 else self.a_t[dims]
        a_s = self.a_s[:, dims] if self.a_s.ndim == 2 else self.a_s[dims]
        if "domain" in item:
            delta = (p[:, 1] - float(item["gold"]))[:, None]
        else:
            ystar = np.zeros(len(dims)); ystar[int(item["gold"])] = 1.0
            delta = p - ystar[None, :]
        g = self._gate(np.abs(z))
        self.t[:, dims] = self.t[:, dims] + a_t * delta
        self.u[:, dims] = self.u[:, dims] - a_s * g * (
            self.u[:, dims] - self.u_inf[:, dims])
        if self.q_t > 0:
            self.t[:, dims] += self.rng.normal(0, self.q_t,
                                               self.t[:, dims].shape)
        if self.q_s > 0:
            self.u[:, dims] += self.rng.normal(0, self.q_s,
                                               self.u[:, dims].shape)

    # --- scoring / readiness ---
    def p_correct(self, item):
        dims, z, p = self._p_item(item)
        return p[:, int(item["gold"])]

    def exam_accuracy(self, exam):
        acc = np.zeros(self.N)
        for it in exam["items"]:
            acc += self.p_correct(it)
        return acc / len(exam["items"])

    def pred_score(self, exam):
        return float(np.sum(self.w * self.exam_accuracy(exam)))

    def pass_prob_quantile(self, exam, q):
        a = np.clip(self.exam_accuracy(exam), 1e-9, 1 - 1e-9)
        pp = binom.sf(exam["pass_count"] - 1, len(exam["items"]), a)
        order = np.argsort(pp)
        cw = np.cumsum(self.w[order])
        return float(pp[order][np.searchsorted(cw, q)])

    def ready(self, exam, eta):
        return self.pass_prob_quantile(exam, eta) >= 1.0 - eta

    def ell_cloud(self, domain, items):
        """Per-particle reduced log-skill on the instrument's scale."""
        return ell_reduced(self.t, self.u, self.lam, self.reg,
                           domain, items)

    def pass_mass(self, domain, ell_star, items):
        """pi = P(ell_domain > ell*) — the instrument's per-domain
        readiness statistic, computed from the reduced cloud."""
        e = self.ell_cloud(domain, items)
        return float(np.sum(self.w * (e > float(ell_star))))

    def trainability(self, domain, ell_star, items=None):
        """P(the skill CEILING clears the cut): reduced ell at u = u_inf."""
        saved = self.u
        self.u = self.u_inf
        try:
            e = self.ell_cloud(domain, items or [])
        finally:
            self.u = saved
        return float(np.sum(self.w * (e > float(ell_star))))

    def mean_state(self):
        return self.w @ self.t, self.w @ self.u

    def margin(self, exam):
        a = np.clip(self.exam_accuracy(exam), 1e-9, 1 - 1e-9)
        bar = exam["pass_count"] / len(exam["items"])
        m = (a - bar) / np.sqrt(a * (1 - a) / len(exam["items"]))
        return float(np.sum(self.w * m))

    def margin_gain(self, item, exam):
        """Expected one-step exam-margin gain of serving this item
        (deterministic per-particle push, response-independent)."""
        dims, z, p = self._p_item(item)
        a_t = self.a_t[:, dims] if self.a_t.ndim == 2 else self.a_t[dims]
        a_s = self.a_s[:, dims] if self.a_s.ndim == 2 else self.a_s[dims]
        if "domain" in item:
            delta = (p[:, 1] - float(item["gold"]))[:, None]
        else:
            ystar = np.zeros(len(dims)); ystar[int(item["gold"])] = 1.0
            delta = p - ystar[None, :]
        g = self._gate(np.abs(z))
        t2 = self.t.copy(); u2 = self.u.copy()
        t2[:, dims] += a_t * delta
        u2[:, dims] -= a_s * g * (self.u[:, dims] - self.u_inf[:, dims])
        base = self.margin(exam)
        st, su = self.t, self.u
        self.t, self.u = t2, u2
        gain = self.margin(exam) - base
        self.t, self.u = st, su
        return gain


class MixedSessionEngine:
    """Joint training across all registry domains against one exam.

    bank: list of items in the registry item schema, each carrying a
    unique 'item_id'. Placement: per domain, the unserved candidate
    whose asked-domain |z| under the mean state is nearest the gate
    peak; serve the candidate with the largest expected margin gain."""

    def __init__(self, artifact, registry, bank, exam, eta, budget=300,
                 n_particles=400, zstar=1.0):
        self.art, self.reg = artifact, registry
        self.bank, self.exam, self.eta = bank, exam, eta
        self.budget, self.N, self.zstar = budget, n_particles, zstar

    def _cand(self, bel, served):
        t_bar, u_bar = bel.mean_state()
        out = []
        for j, code in enumerate(self.reg.codes):
            best, bd = None, np.inf
            for i, it in enumerate(self.bank):
                if i in served:
                    continue
                dims = self.reg.item_dims(it)
                if j not in dims:
                    continue
                if "domain" not in it and int(it["gold"]) != \
                        dims.index(j):
                    continue
                s_asked = (float(it["s"]) if "domain" in it
                           else float(np.asarray(it["s"])[dims.index(j)]))
                z = abs((s_asked - t_bar[j]) / np.exp(u_bar[j]))
                d = abs(z - self.zstar)
                if d < bd:
                    best, bd = i, d
            if best is not None:
                out.append(best)
        return out

    def run(self, learner_step, seed=0, mastery=False, on_question=None):
        rng = np.random.default_rng(seed)
        bel = MixedBelief(self.art, self.reg, N=self.N, rng=rng)
        served, log = set(), []
        ready_at, n_used = -1, 0
        for k in range(self.budget):
            if on_question is not None:
                on_question(k, bel)
            if ready_at < 0 and bel.ready(self.exam, self.eta):
                ready_at = k
                if not mastery:
                    break
            cands = self._cand(bel, served)
            if not cands:
                break
            gains = [bel.margin_gain(self.bank[i], self.exam)
                     for i in cands]
            i = cands[int(np.argmax(gains))]
            item = self.bank[i]
            y = int(learner_step(item))
            bel.update(item, y)
            served.add(i)
            n_used = k + 1
            log.append(dict(k=k, item_id=item.get("item_id", i), y=y))
        return dict(ready_at=ready_at,
                    n_used=(ready_at if ready_at >= 0 and not mastery
                            else n_used),
                    log=log, belief=bel)
