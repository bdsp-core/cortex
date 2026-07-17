"""Domain registry: the declarative task structure (integration plan §3.5).

The registry is DATA, not code: an ordered list of (code, link) pairs,
where link is "binary" (a present/absent detection domain) or the name
of an n-way GROUP (the set of classes competing in one identification
pick). Any number of binary domains and disjoint n-way groups may
coexist; adding a domain to the pipeline is a registry edit plus data,
never an engine change. State dimensions (t, u, u_inf) are indexed by
registry order.

Item schema (dicts, all optional keys omitted where unused):
  binary item:  dict(domain=<code>, s=float, s_sd=float, gold=0|1)
  n-way item:   dict(group=<name>, s=(G,) evidence per class in group
                     order, s_sd=(G,), gold=<position in group>)
An exam is dict(items=[...], pass_count=int) or any iterable of items
for scoring purposes.

Also here: the certification instrument's AUROC(ell) bijection and the
DERIVED reduction map (plan §3.4) — the one-vs-rest log-skill a
multiclass belief implies for each n-way domain on a given case-mix,
on the same scale the per-domain cuts are defined on. For binary
domains the reduction is the identity (their ell = -u exactly).
"""
import numpy as np
from scipy.special import ndtr, ndtri

_AUROC_MAX = float(ndtr(np.sqrt(2.0)))     # supremum as ell -> inf


class Registry:
    """Ordered domain registry. domains: [(code, "binary"|group_name)]."""

    def __init__(self, domains):
        self.domains = list(domains)
        self.codes = [c for c, _ in self.domains]
        if len(set(self.codes)) != len(self.codes):
            raise ValueError("duplicate domain codes")
        self.index = {c: i for i, c in enumerate(self.codes)}
        self.link = {c: l for c, l in self.domains}
        self.groups = {}
        for c, l in self.domains:
            if l != "binary":
                self.groups.setdefault(l, []).append(self.index[c])
        self.binary = [self.index[c] for c, l in self.domains
                       if l == "binary"]
        for g, idx in self.groups.items():
            if len(idx) < 2:
                raise ValueError(f"n-way group '{g}' needs >= 2 domains")

    @property
    def M(self):
        return len(self.codes)

    def item_dims(self, item):
        """State indices an item's evidence addresses (group order for
        n-way; the single domain index for binary)."""
        if "domain" in item:
            return [self.index[item["domain"]]]
        return list(self.groups[item["group"]])

    def to_dict(self):
        return {"domains": [list(d) for d in self.domains]}

    @classmethod
    def from_dict(cls, d):
        return cls([tuple(x) for x in d["domains"]])


def auroc_of_ell(ell):
    """The certification instrument's skill->discrimination bijection:
    AUROC(ell) = Phi(sqrt(2) / sqrt(exp(-2 ell) + 1))."""
    e = np.asarray(ell, float)
    return ndtr(np.sqrt(2.0) / np.sqrt(np.exp(-2.0 * e) + 1.0))


def ell_of_auroc(a):
    """Inverse of auroc_of_ell (valid for chance < a < Phi(sqrt(2)));
    inputs are clipped into the open interval."""
    a = np.clip(np.asarray(a, float), 0.5 + 1e-9, _AUROC_MAX - 1e-9)
    z = ndtri(a)
    return -0.5 * np.log(2.0 / z ** 2 - 1.0)


def p_assert(t, u, lam, item, registry):
    """Per-particle probability of ASSERTING the item's asked class —
    the instrument's binarized observable applied to the full model.
    t, u: (N, M) state arrays. Returns (N,) for the asked class
    (binary: the 'present' response; n-way: pick == gold class)."""
    t = np.asarray(t, float); u = np.asarray(u, float)
    if "domain" in item:
        j = registry.index[item["domain"]]
        z = (float(item["s"]) - t[:, j]) / np.exp(u[:, j])
        z = _attenuate(z, u[:, j], item.get("s_sd"))
        return lam + (1.0 - 2.0 * lam) * ndtr(z)
    dims = registry.item_dims(item)
    s = np.asarray(item["s"], float)
    z = (s[None, :] - t[:, dims]) / np.exp(u[:, dims])
    sd = item.get("s_sd")
    if sd is not None:
        z = z / np.sqrt(1.0 + (np.asarray(sd, float)[None, :]
                               / np.exp(u[:, dims])) ** 2)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    g = len(dims)
    return lam / g + (1.0 - lam) * p[:, int(item["gold"])]


def _attenuate(z, u, s_sd):
    """Closed-form evidence-uncertainty attenuation (the instrument's
    own convention): z /= sqrt(1 + (s_sd / sigma)^2). No-op if s_sd
    is absent."""
    if s_sd is None:
        return z
    return z / np.sqrt(1.0 + (float(s_sd) / np.exp(u)) ** 2)


def ell_reduced(t, u, lam, registry, domain, items):
    """The DERIVED reduction map (plan §3.4): per-particle one-vs-rest
    log-skill for `domain` on the case-mix `items`, on the instrument's
    ell scale.

    Binary domains: the identity ell = -u (exact — no case-mix needed).
    N-way domains: PROJECT the full model onto the instrument's binary
    family over the case-mix — their estimator assumes
    P(assert k) = lam + (1-2 lam) Phi(e^ell (s_k + theta)), so the ell
    their instrument would asymptotically measure is the probit-scale
    slope of our implied assert-probability curve in the asked-class
    evidence. Per particle: y_i = probit((p_assert_i - lam)/(1-2 lam)),
    ell = log(d y / d s) by least squares over the case-mix. Derived
    quantity, no fitted constants, and it cannot saturate (an EMPIRICAL
    pairwise AUROC over a finite mix censors all high skill at 1.0 —
    measured and rejected during P1). On a binary domain this
    projection is exactly -u, which doubles as its self-check."""
    t = np.asarray(t, float); u = np.asarray(u, float)
    j = registry.index[domain]
    if registry.link[domain] == "binary":
        return -u[:, j]
    xs, ys = [], []
    for it in items:
        if it.get("group") != registry.link[domain]:
            continue
        dims = registry.item_dims(it)
        if j not in dims:
            continue
        pos = dims.index(j)
        p = p_assert(t, u, lam, dict(it, gold=pos), registry)
        xs.append(float(np.asarray(it["s"], float)[pos]))
        ys.append(ndtri(np.clip((p - lam) / (1.0 - 2.0 * lam),
                                1e-6, 1.0 - 1e-6)))
    if len(xs) < 3:
        raise ValueError(f"case-mix has too few items for domain "
                         f"'{domain}'")
    x = np.array(xs)
    if x.var() <= 0:
        raise ValueError(f"case-mix has no evidence spread for domain "
                         f"'{domain}'")
    Y = np.stack(ys, axis=1)                        # (N, n_items)
    slope = ((Y - Y.mean(axis=1, keepdims=True))
             @ (x - x.mean())) / (len(x) * x.var())
    return np.log(np.maximum(slope, 1e-9))
