"""M10.1 — animated MP4 demonstration of the learning algorithm.

One continuous story, simulated with the ACTUAL pipeline modules (the same
code validated in test_step8) and rendered in the beautiful_figure aesthetic:

  CERTIFICATION EVAL  →  TRAINING (Tier-2, real bank, LT2 gaps)  →  RE-CERT

Panels:
  A  belief state space (plan coords): the active task's particle cloud,
     posterior-mean trails and true states for all tasks, mastery target zone;
  B  the active task's psychometric curve — truth vs filter estimate — with
     the served item (label + stimulus uncertainty);
  C  skill trajectories ℓ̂ ±1 SD vs time with the cert cut-scores ℓ*_k;
  D  bias trajectories t̂ vs time with the ±t* tolerance band;
  E  served-trial strip (mode per trial).

Run:  python3 -m viz.make_demo_video        → figures/learning_algorithm_demo.mp4
"""
from __future__ import annotations

import os

import numpy as np
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle
from scipy.stats import norm

import engine.core_mcmc_general as eng
from engine.policy_general import AD6Policy, PASS, PENDING
from training.bank_adapter import BankAdapter, ELL_STAR, SIGMA_STAR, SIGMA_INF, TASK_CODES
from training.bridge_conventions import LAPSE_RATE
from training.learner_sim import Learner, LearnerParams
from training.training_seed import build_seed_from_state
from training.training_filter import filter_from_seed_task
from training.trainer_policy import TrainerPolicy, ModeThresholds
from viz.viz_style import (TASK_FILL, TASK_EDGE, NEUTRAL, MODE_FILL, GOOD, BAD,
                       ACCENT_DARK, FIGDIR, use_style, style_ax)

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

TASKS = (1, 2, 3)                       # domain2, domain3, domain4
K = 3
T_STAR = 0.30
EPOCH0, DAY = 1_770_000_000.0, 86_400.0
EVAL_EVERY = 4                          # record every 4th eval question
N_CLOUD = 200                           # particles drawn in panel A
FPS = 7

rng_global = np.random.default_rng(11)


# ───────────────────────── simulation + trace ─────────────────────────

def _engine_task_stats(state):
    w = state["w"]
    out = []
    for k in range(K):
        th, el = state["t"][:, k], state["l"][:, k]
        mt, ml = float((w * th).sum()), float((w * el).sum())
        st = float(np.sqrt((w * (th - mt) ** 2).sum()))
        sl = float(np.sqrt((w * (el - ml) ** 2).sum()))
        out.append((mt, ml, st, sl))
    return out


def _cloud_subsample(th, el, w, n=N_CLOUD, rng=None):
    rng = rng or rng_global
    idx = rng.choice(len(th), size=min(n, len(th)), p=w / w.sum())
    return th[idx].copy(), el[idx].copy()


def run_eval_traced(true_t, true_l, bank, *, phase, seed, exclude=None,
                    n_part=500, max_q=500, pool_size=800):
    """Adaptive cert session (mirrors pipeline_demo._run_eval) + frame trace."""
    rng = np.random.default_rng(seed)
    pools = []
    for t in TASKS:
        p = bank.candidates(t, exclude_segids=exclude)
        if len(p) > pool_size:
            order = np.argsort(p.s_mean)
            pick = order[np.round(np.linspace(0, len(p) - 1, pool_size)).astype(int)]
            p = p.subset(np.isin(np.arange(len(p)), pick))
        pools.append(p)
    bs = [p.s_mean.copy() for p in pools]
    bsd = [p.s_sd.copy() for p in pools]
    bid = [p.seg_id.copy() for p in pools]

    state = eng.make_state_hier(n_part, K, r_assumed=0.378, rng=rng)
    policy = AD6Policy([ELL_STAR[t] for t in TASKS], [1.0] * K)
    policy.reset(K)
    n_per_task = [0] * K
    served, frames = [], []
    lrng = np.random.default_rng(seed + 1)
    for q in range(max_q):
        active = [k for k in range(K)
                  if policy._verdicts[k] == PENDING and len(bs[k]) > 0]
        if not active:
            break
        k, s, s_sd, seg_id = eng.choose_item(state, bs, active_domains=active,
                                             bank_sds=bsd, return_sd=True,
                                             bank_segids=bid)
        s_real = s + s_sd * lrng.standard_normal()
        y = eng.simulate_response(s_real, true_t[k], true_l[k], lrng)
        eng.update(state, k, s, y, s_sd=s_sd)
        j = int(np.where(bid[k] == seg_id)[0][0])
        bs[k] = np.delete(bs[k], j); bsd[k] = np.delete(bsd[k], j)
        bid[k] = np.delete(bid[k], j)
        if eng.ess(state["w"]) < 0.5 * n_part:
            eng.resample_and_rejuvenate(state, rng, 15, 2.38 / np.sqrt(2 * K))
        n_per_task[k] += 1
        served.append(int(seg_id))
        dec = policy(state, {}, n_per_task, K)
        if q % EVAL_EVERY == 0 or dec.stop:
            th, el = _cloud_subsample(state["t"][:, k], state["l"][:, k],
                                      state["w"])
            frames.append(dict(
                phase=phase, mode=phase, task=k, s=float(s), s_sd=float(s_sd),
                y=int(y), y_star=None, q=q,
                stats=_engine_task_stats(state), cloud=(th, el),
                true=[( float(np.exp(-true_l[i])), float(-true_t[i]))
                      for i in range(K)],
                verdicts=list(policy._verdicts), mastered=[False] * K))
        if dec.stop:
            break
    verdicts = policy.finalize_verdicts()
    for fr in frames:
        fr["final_verdicts_pending"] = verdicts
    return state, verdicts, policy._last_diag, served, frames


def run_demo(seed0=11):
    bank = BankAdapter()
    true_l = np.array([ELL_STAR[t] - 0.45 for t in TASKS])
    true_t = np.array([0.7, -0.6, 0.5])
    learner_inf = [min(0.82 * SIGMA_STAR[t], SIGMA_INF[t]) for t in TASKS]

    # 1 ── EVAL
    state, ev_verd, ev_diag, ev_served, ev_frames = run_eval_traced(
        true_t, true_l, bank, phase="eval", seed=seed0)

    # 2 ── HANDOFF + TRAIN
    seed_obj = build_seed_from_state(
        state, session_id="demo", ell_star=[ELL_STAR[t] for t in TASKS],
        verdicts=ev_verd, diagnostics=ev_diag, inflate=1.5,
        seg_ids=ev_served, session_type="eval")
    learner = Learner(np.exp(-true_l), -true_t,
                      LearnerParams(alpha_t=0.2, alpha_sigma=0.06,
                                    sigma_inf=np.array(learner_inf),
                                    q_t=0.04, q_sigma=0.02, rho=0.6,
                                    rule="soft"), seed=seed0 + 5)
    fparams = [LearnerParams(alpha_t=0.18, alpha_sigma=0.05,
                             sigma_inf=learner_inf[i], q_t=0.04, q_sigma=0.02,
                             rho=0.6, rule="soft") for i in range(K)]
    filters = [filter_from_seed_task(seed_obj, i, fparams[i], inflated=True,
                                     rng_seed=seed0 + 10 + i) for i in range(K)]

    class _Remap:
        def __init__(self, b, ts): self.b, self.ts = b, ts
        def candidates(self, task, **kw):
            return self.b.candidates(self.ts[task], **kw)

    pol = TrainerPolicy(filters, [ELL_STAR[t] for t in TASKS],
                        [SIGMA_STAR[t] for t in TASKS], _Remap(bank, TASKS),
                        thresholds=ModeThresholds(t_star=T_STAR, sd_floor=0.23),
                        seed=seed0 + 20, exclude_segids=ev_served)
    srng = np.random.default_rng(seed0 + 30)
    tr_frames = []
    sess_starts = []
    for sess in range(4):
        if sess > 0:
            for f in filters:
                f.propagate_gap(DAY, widen=0.03)
        sess_starts.append(len(tr_frames))
        for j in range(80):
            if pol.all_mastered():
                break
            ch = pol.step(now=EPOCH0 + sess * DAY + j)
            if ch is None:
                break
            tk = ch["task"]
            s_real = ch["s"] + ch["s_sd"] * srng.standard_normal()
            y = learner.step(s_real, tk, ch["y_star"], feedback=True)
            pol.record(ch, y)
            f = filters[tk]
            th, el = _cloud_subsample(f.theta, f.ell, f.w)
            tr_frames.append(dict(
                phase="train", session=sess, mode=ch["mode"], task=tk,
                s=ch["s"], s_sd=ch["s_sd"], y=int(y), y_star=ch["y_star"],
                stats=[(float((g.w * g.theta).sum()), float((g.w * g.ell).sum()),
                        g.sd()[0], g.sd()[1]) for g in filters],
                cloud=(th, el),
                true=[(float(learner.sigma[i]), float(learner.t[i]))
                      for i in range(K)],
                mastered=[pol.mode_policies[i].is_mastered(filters[i])
                          for i in range(K)],
                verdicts=None))
        if pol.all_mastered():
            break

    # 3 ── RE-CERT
    seen = set(ev_served) | pol._served
    _, rc_verd, _, _, rc_frames = run_eval_traced(
        -learner.t, -np.log(learner.sigma), bank, phase="recert",
        seed=seed0 + 88, exclude=seen)

    return dict(ev_frames=ev_frames, tr_frames=tr_frames, rc_frames=rc_frames,
                ev_verdicts=ev_verd, rc_verdicts=rc_verd,
                sess_starts=sess_starts, true_l0=true_l)


# ───────────────────────── rendering ─────────────────────────

PHASE_LABEL = {"eval": "CERTIFICATION EVAL", "train": "TRAINING (Tier-2)",
               "recert": "RE-CERTIFICATION"}
VERD_COLOR = {"PASS": GOOD, "FAIL": BAD, "REFER_BORDERLINE": NEUTRAL,
              "REFER_HARD": NEUTRAL, "PENDING": '#cccccc'}


def _p_yes_plan(s, sigma, t):
    return LAPSE_RATE + (1 - 2 * LAPSE_RATE) * norm.cdf((s - t) / sigma)


def render(trace, out_path, fps=FPS):
    frames = trace["ev_frames"] + trace["tr_frames"] + trace["rc_frames"]
    n_ev, n_tr = len(trace["ev_frames"]), len(trace["tr_frames"])
    N = len(frames)
    hold = fps * 2                                  # 2 s verdict holds
    timeline = (list(range(N))
                + [n_ev - 1] * 0)                   # plain sequence; holds below
    # insert holds after eval and at the very end
    seq = list(range(n_ev)) + [n_ev - 1] * hold \
        + list(range(n_ev, n_ev + n_tr)) \
        + list(range(n_ev + n_tr, N)) + [N - 1] * hold

    use_style(16)
    fig = plt.figure(figsize=(19.2, 10.8))
    gs = fig.add_gridspec(2, 3, left=0.055, right=0.985, top=0.86, bottom=0.16,
                          wspace=0.30, hspace=0.42,
                          width_ratios=[1.15, 1, 1])
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 2])
    axE = fig.add_subplot(gs[1, 1])
    ax_strip = fig.add_axes([0.055, 0.045, 0.93, 0.045])

    # precompute trajectory arrays over the FRAME index
    ml = np.array([[fr["stats"][k][1] for k in range(K)] for fr in frames])
    sl = np.array([[fr["stats"][k][3] for k in range(K)] for fr in frames])
    mt = np.array([[-fr["stats"][k][0] for k in range(K)] for fr in frames])
    st = np.array([[fr["stats"][k][2] for k in range(K)] for fr in frames])
    true_ell = np.array([[-np.log(fr["true"][k][0]) for k in range(K)]
                         for fr in frames])
    true_t = np.array([[fr["true"][k][1] for k in range(K)] for fr in frames])
    cuts = np.array([ELL_STAR[t] for t in TASKS])

    def draw(fi):
        fr = frames[fi]
        for ax in (axA, axB, axC, axD, axE, ax_strip):
            ax.clear()
        ph = fr["phase"]
        k = fr["task"]
        col, edg = TASK_FILL[k], TASK_EDGE[k]

        # ── header ──
        fig.suptitle("", y=0.99)
        fig.texts = [t for t in fig.texts if False]
        sess_txt = (f"  ·  session {fr.get('session', 0) + 1}"
                    if ph == "train" else "")
        fig.text(0.055, 0.945, PHASE_LABEL[ph] + sess_txt,
                 fontsize=24, fontweight='bold', color=ACCENT_DARK)
        fig.text(0.055, 0.905,
                 f"task: {TASK_CODES[TASKS[k]]}   ·   "
                 + (f"mode: {fr['mode']}" if ph == "train"
                    else "A-optimal adaptive item selection")
                 + f"   ·   frame {fi + 1}/{len(frames)}",
                 fontsize=16, color='#666666')
        if ph != "train":
            verds = (trace["ev_verdicts"] if ph == "eval"
                     else trace["rc_verdicts"])
            show = verds if fi in (len(trace["ev_frames"]) - 1, len(frames) - 1) \
                or fr is frames[-1] else fr["verdicts"]
            for i, v in enumerate(show):
                short = {"REFER_BORDERLINE": "REFER", "REFER_HARD": "REFER",
                         "PENDING": "testing…"}.get(v, v)
                fig.text(0.55 + 0.15 * i, 0.92,
                         f"{TASK_CODES[TASKS[i]]}: {short}",
                         fontsize=15, fontweight='bold',
                         color=VERD_COLOR.get(v, NEUTRAL))
        else:
            for i in range(K):
                m = fr["mastered"][i]
                fig.text(0.55 + 0.15 * i, 0.92,
                         f"{TASK_CODES[TASKS[i]]}: "
                         + ("MASTERED" if m else "training"),
                         fontsize=15, fontweight='bold',
                         color=GOOD if m else '#999999')

        # ── A: belief state space (plan coords) ──
        th, el = fr["cloud"]
        sig_c, t_c = np.exp(-el), -th
        axA.scatter(t_c, sig_c, s=22, color=col, edgecolors=edg,
                    linewidths=0.4, alpha=0.45, zorder=3,
                    label=f"belief cloud ({TASK_CODES[TASKS[k]]})")
        sstar = SIGMA_STAR[TASKS[k]]
        axA.add_patch(Rectangle((-T_STAR, 0.25), 2 * T_STAR, sstar - 0.25,
                                facecolor=GOOD, alpha=0.12, zorder=1))
        axA.axhline(sstar, color=GOOD, lw=1.6, ls='--', zorder=2)
        axA.axvline(-T_STAR, color=NEUTRAL, lw=1.2, ls=':', zorder=2)
        axA.axvline(T_STAR, color=NEUTRAL, lw=1.2, ls=':', zorder=2)
        for i in range(K):
            sig_i, t_i = fr["true"][i]
            axA.scatter([t_i], [sig_i], marker='*', s=460,
                        color=TASK_FILL[i], edgecolors=ACCENT_DARK,
                        linewidths=1.4, zorder=5)
            axA.scatter([mt[fi, i]], [np.exp(-ml[fi, i])], marker='o', s=90,
                        color=TASK_FILL[i], edgecolors=TASK_EDGE[i],
                        linewidths=1.5, zorder=4)
        axA.set_xlim(-1.1, 1.1); axA.set_ylim(0.3, 1.9)
        axA.set_xlabel("bias / criterion  t"); axA.set_ylabel("noise  σ  (skill = 1/σ)")
        axA.set_title("belief state  (★ truth, ● posterior mean)", fontsize=16)
        axA.text(0, 0.34, "mastery zone", color=GOOD, fontsize=14,
                 ha='center', fontstyle='italic')
        style_ax(axA)

        # ── B: psychometric curve of the active task ──
        sg = np.linspace(-3, 3.6, 200)
        sig_true, t_true = fr["true"][k]
        axB.plot(sg, _p_yes_plan(sg, sig_true, t_true), color=edg, lw=2.6,
                 label="true")
        axB.plot(sg, _p_yes_plan(sg, np.exp(-ml[fi, k]), mt[fi, k]),
                 color=NEUTRAL, lw=2.6, ls='--', label="estimated")
        y_at = _p_yes_plan(fr["s"], sig_true, t_true)
        lab = fr.get("y_star")
        mcol = GOOD if (lab == 1 or (lab is None and fr["y"] == 1)) else BAD
        axB.errorbar([fr["s"]], [y_at], xerr=[fr["s_sd"]], fmt='o', ms=11,
                     color=mcol, mec=ACCENT_DARK, mew=1.4, elinewidth=2.2,
                     capsize=5, zorder=5,
                     label="served item ± s_sd")
        axB.axvline(0, color='#bbbbbb', lw=1.0, ls=':')
        axB.set_ylim(-0.04, 1.04); axB.set_xlim(-3, 3.6)
        axB.set_xlabel("stimulus signal  s"); axB.set_ylabel("P(respond 'present')")
        axB.set_title("psychometric curve", fontsize=16)
        axB.legend(loc='lower right', frameon=False, fontsize=13)
        style_ax(axB)

        # ── C: skill trajectories ──
        x = np.arange(fi + 1)
        for i in range(K):
            axC.fill_between(x, ml[:fi + 1, i] - sl[:fi + 1, i],
                             ml[:fi + 1, i] + sl[:fi + 1, i],
                             color=TASK_FILL[i], alpha=0.18, lw=0)
            axC.plot(x, ml[:fi + 1, i], color=TASK_FILL[i], lw=2.4)
            axC.plot(x, true_ell[:fi + 1, i], color=TASK_EDGE[i], lw=1.4,
                     ls=':')
            axC.axhline(cuts[i], color=TASK_FILL[i], lw=1.2, ls='--',
                        alpha=0.8)
        axC.set_xlim(0, N); axC.set_ylim(-1.3, 1.3)
        axC.set_xlabel("frame"); axC.set_ylabel("skill  ℓ")
        axC.set_title("skill ℓ̂ ±1 SD  (· · truth, - - cut ℓ*)", fontsize=16)
        style_ax(axC)

        # ── D: bias trajectories ──
        axD.axhspan(-T_STAR, T_STAR, color=GOOD, alpha=0.10, lw=0)
        for i in range(K):
            axD.plot(x, mt[:fi + 1, i], color=TASK_FILL[i], lw=2.4)
            axD.plot(x, true_t[:fi + 1, i], color=TASK_EDGE[i], lw=1.4, ls=':')
        axD.axhline(0, color='#bbbbbb', lw=1.0)
        axD.set_xlim(0, N); axD.set_ylim(-1.0, 1.0)
        axD.set_xlabel("frame"); axD.set_ylabel("bias  t")
        axD.set_title("bias t̂  (· · truth, band = |t| ≤ t*)", fontsize=16)
        style_ax(axD)

        # ── E: posterior SD (information) ──
        for i in range(K):
            axE.plot(x, sl[:fi + 1, i], color=TASK_FILL[i], lw=2.2)
        axE.axhline(0.23, color=NEUTRAL, lw=1.6, ls='--')
        axE.text(N * 0.99, 0.235, "mastery sd floor", color=NEUTRAL,
                 ha='right', fontsize=13)
        axE.set_xlim(0, N); axE.set_ylim(0, 0.65)
        axE.set_xlabel("frame"); axE.set_ylabel("posterior SD(ℓ)")
        axE.set_title("belief uncertainty", fontsize=16)
        style_ax(axE)

        # ── strip: phase/mode per frame ──
        for j in range(fi + 1):
            ph_j = frames[j]["phase"]
            key = frames[j]["mode"] if ph_j == "train" else ph_j
            ax_strip.add_patch(Rectangle((j, 0), 1, 1,
                                         color=MODE_FILL.get(key, '#dddddd'),
                                         lw=0))
        ax_strip.set_xlim(0, N); ax_strip.set_ylim(0, 1)
        ax_strip.set_yticks([]); ax_strip.set_xticks([])
        ax_strip.set_title("eval  /  bias  /  skill  /  retention  /  re-cert",
                           fontsize=12, color='#888888', loc='left', pad=2)
        for spine in ax_strip.spines.values():
            spine.set_visible(False)
        return []

    writer = animation.FFMpegWriter(fps=fps, bitrate=6000,
                                    extra_args=['-pix_fmt', 'yuv420p'])
    anim = animation.FuncAnimation(fig, lambda i: draw(seq[i]),
                                   frames=len(seq), blit=False)
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    return out_path, len(seq)


if __name__ == "__main__":
    print("simulating the eval → train → re-cert chain ...")
    trace = run_demo()
    print(f"  frames: eval {len(trace['ev_frames'])}, "
          f"train {len(trace['tr_frames'])}, recert {len(trace['rc_frames'])}")
    print(f"  eval verdicts  : {trace['ev_verdicts']}")
    print(f"  recert verdicts: {trace['rc_verdicts']}")
    out = os.path.join(FIGDIR, "learning_algorithm_demo.mp4")
    print("rendering ...")
    path, nseq = render(trace, out)
    print(f"wrote {path} ({nseq} frames @ {FPS} fps ≈ {nseq / FPS:.0f}s)")
