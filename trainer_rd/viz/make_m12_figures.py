"""M12/V6 — fig8: the verification-ladder mis-specification envelope.

Four panels from the cached M12 study npz files (no recomputation):
  A  gate operating characteristic P(declare) vs ceiling-to-cut margin Δ
     (shipped / oracle / hardened3) — the F29 flat-OC vs step-function story
  B  false-graduation (latent, cert-bar semantics) per zoo member,
     shipped vs hardened3 (F28 + mitigation)
  C  closed-loop SBC 90% coverage of ℓ vs trial checkpoint,
     shipped vs F30-corrected kernel (smear_w)
  D  declared vs true trials-to-mastery per member (shipped tier2):
     declaration time is learner-independent (F29's other face)

Run: python3 -m viz.make_m12_figures  → figures/fig8_misspec_envelope.{png,pdf,svg}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz.viz_style as vs

vs.use_style(font_size=15)

oc = np.load("figures/data_gate_oc.npz", allow_pickle=True)
oc2 = np.load("figures/data_gate_oc2.npz", allow_pickle=True)
camp = np.load("figures/data_misspec_campaign.npz", allow_pickle=True)
mit = np.load("figures/data_misspec_mitig.npz", allow_pickle=True)
mit2 = np.load("figures/data_misspec_mitig2.npz", allow_pickle=True)
sbc = np.load("figures/data_train_sbc.npz", allow_pickle=True)
sbc2 = np.load("figures/data_train_sbc_smearw.npz", allow_pickle=True)

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
(axA, axB), (axC, axD) = axes

# ── A: OC curves ──────────────────────────────────────────────────────────
deltas = oc["deltas"]
arm_style = {"shipped": (vs.BAD, "-", "o", "shipped gate"),
             "oracle": (vs.NEUTRAL, "--", "s", "oracle (knows ceiling)"),
             "hardened3": (vs.GOOD, "-", "D", "hardened (M12)")}
arms = list(oc["arms"])
for arm, (c, ls, mk, lab) in arm_style.items():
    if arm in arms:
        a = arms.index(arm)
        p, ci = oc["p_decl"][a], oc["ci"][a]
    elif arm in list(oc2["arms"]):
        a = list(oc2["arms"]).index(arm)
        p, ci = oc2["p_decl"][a], oc2["ci"][a]
    else:
        continue
    axA.fill_between(deltas, ci[:, 0], ci[:, 1], color=c, alpha=0.15, lw=0)
    axA.plot(deltas, p, ls, marker=mk, color=c, lw=2.5, ms=7, label=lab)
axA.axvline(0, color=vs.NEUTRAL, lw=1.2, zorder=1)
axA.set_xlabel("true ceiling − cut  (Δ in ℓ)")
axA.set_ylabel("P(gate declares mastery)")
axA.set_title("A — graduation operating characteristic", loc="left")
axA.set_ylim(-0.03, 1.06)
vs.style_ax(axA)
axA.legend(loc="center right", frameon=False, fontsize=12)

# ── B: FG-latent per member, shipped vs hardened3 ─────────────────────────
members = list(mit["members"])
budget = float(mit["budget"])
def fg_lat(d, cfg_list, cfg):
    j = cfg_list.index(cfg)
    nd, el, td = d["n_decl"][:, j], d["ell_decl"][:, j], d["t_decl"][:, j]
    decl = nd <= budget
    fg = decl & ((el < 0.534) | (np.abs(td) > 0.30))
    return fg.mean(axis=-1)
fg_ship = fg_lat(mit, list(mit["configs"]), "shipped")
fg_hard = fg_lat(mit2, list(mit2["configs"]), "hardened3")
x = np.arange(len(members))
axB.bar(x - 0.2, fg_ship, 0.38, color=vs.BAD, edgecolor=vs.ACCENT_DARK,
        label="shipped")
axB.bar(x + 0.2, fg_hard, 0.38, color=vs.GOOD, edgecolor=vs.ACCENT_DARK,
        label="hardened (M12)")
axB.set_xticks(x)
axB.set_xticklabels(members, rotation=40, ha="right", fontsize=11)
axB.set_ylabel("P(false graduation, cert-bar)")
axB.set_title("B — false graduation across the zoo (tier2)", loc="left")
vs.style_ax(axB)
axB.legend(frameon=False, fontsize=12)

# ── C: SBC coverage ───────────────────────────────────────────────────────
checks = [10, 50, 150, 300]
for d, c, lab in ((sbc, vs.BAD, "shipped kernel"),
                  (sbc2, vs.GOOD, "smeared-w kernel (F30 fix)")):
    cov = [float(np.mean(np.abs(d[f"pit_{n}"][:, 0] - 0.5) < 0.45))
           for n in checks]
    axC.plot(checks, cov, "-o", color=c, lw=2.5, ms=7, label=lab)
axC.axhline(0.90, color=vs.NEUTRAL, lw=1.2, ls="--")
axC.text(295, 0.905, "nominal", color=vs.NEUTRAL, fontsize=11, ha="right")
axC.set_xlabel("training trial")
axC.set_ylabel("90% central coverage of ℓ")
axC.set_title("C — closed-loop SBC (well-specified)", loc="left")
axC.set_ylim(0.75, 1.0)
vs.style_ax(axC)
axC.legend(frameon=False, fontsize=12, loc="lower right")

# ── D: declared vs true mastery ───────────────────────────────────────────
cmem = list(camp["members"])
jp = list(camp["policies"]).index("tier2")
B = float(camp["budget"])
for i, m in enumerate(cmem):
    nt, nd = camp["zoo_n_true"][i, jp], camp["zoo_n_decl"][i, jp]
    mt = np.median(nt[nt <= B]) if (nt <= B).any() else B * 1.12
    md = np.median(nd[nd <= B]) if (nd <= B).any() else B * 1.12
    col = vs.BAD if m in ("anti", "careless", "static_below") else \
        vs.TASK_FILL[i % 3]
    axD.scatter(mt, md, s=110, color=col, edgecolor=vs.ACCENT_DARK, zorder=3)
    axD.annotate(m, (mt, md), textcoords="offset points", xytext=(8, 6),
                 fontsize=10)
lim = B * 1.2
axD.plot([0, lim], [0, lim], color=vs.NEUTRAL, lw=1.2, ls="--")
axD.axvspan(B, lim, color=vs.NEUTRAL, alpha=0.10, lw=0)
axD.text(B * 1.02, 120, "never truly\nmasters", fontsize=10, color=vs.NEUTRAL)
axD.set_xlabel("median trials to TRUE mastery")
axD.set_ylabel("median trials to DECLARED mastery")
axD.set_title("D — declaration is learner-independent (shipped)", loc="left")
axD.set_xlim(0, lim)
axD.set_ylim(0, 150)
vs.style_ax(axD)

fig.tight_layout(h_pad=2.5, w_pad=2.0)
vs.save_fig(fig, "fig8_misspec_envelope")
print("saved figures/fig8_misspec_envelope.{png,pdf,svg}")
