"""M20 — sandbox analytics report (the delivery-vehicle analytics prototype).

Reads trials.jsonl + sessions.jsonl and produces:
  * a text report: per-session summaries, per-task current state vs the v15
    bars, mastery lifecycle (provisional → confirmed / revoked, D33),
    accuracy/RT trends, GapAnchor stability trajectory (the consolidation
    pilot endpoint, F70-ii), and the D18 re-cert handoff recommendation;
  * figures: skill ℓ̂±SD vs cut and bias t̂ vs the derived band over
    calendar time, session boundaries marked (sandbox/logs/figures/).

Run:  python3 -m sandbox.report
"""
from __future__ import annotations

import json
import os

import numpy as np

from sandbox import config as C


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=str, default=None,
                    help="tester profile to report on (M21/F72)")
    args = ap.parse_args(argv)
    if args.user:
        C.set_user(args.user)
    trials = _read_jsonl(C.TRIALS_JSONL)
    sessions = _read_jsonl(C.SESSIONS_JSONL)
    if not trials:
        print("no trials logged yet — run `python3 -m sandbox.session`")
        return
    print(f"═══ sandbox analytics — profile {C.USER or 'default'}: "
          f"{len(trials)} trials, {len(sessions)} sessions ═══")
    # per-session table
    print(f"\n{'sess':>4} {'gap':>8} {'trials':>6} {'acc':>5} "
          f"{'med RT ms':>9}  events")
    for s in sessions:
        tr = [t for t in trials if t["session"] == s["session"]]
        acc = np.mean([t["correct"] for t in tr]) if tr else float("nan")
        rt = np.median([t["rt_ms"] for t in tr]) if tr else float("nan")
        ev = "; ".join(e["event"] + (f"(t{e['task']})" if "task" in e else "")
                       for e in s.get("events", [])) or "-"
        gap = s.get("gap_s", 0.0)
        gap_str = f"{gap/3600:.1f}h" if gap < 86_400 else f"{gap/86400:.1f}d"
        print(f"{s['session']:>4} {gap_str:>8} {s['n_trials']:>6} "
              f"{acc:5.2f} {rt:9.0f}  {ev}")
    # per-task state (from the last trial's belief snapshots per task)
    print("\nper-task state (latest belief):")
    for t in C.TASKS:
        tr = [x for x in trials if x["task"] == t]
        if not tr:
            continue
        b = tr[-1]["belief"]
        n = len(tr)
        acc = np.mean([x["correct"] for x in tr[-40:]])
        status = ("CONFIRMED" if str(t) in tr[-1]["confirmed"] else
                  "provisional" if str(t) in tr[-1]["provisional"] else
                  "training")
        cons = tr[-1].get("consistency")
        cons_str = (f"  glr {cons['glr']:.1f}{'⚑' if cons['flag'] else ''}"
                    if cons else "")
        print(f"  {C.TASK_NAMES[t]:>9}: n={n:<4} recent-acc {acc:.2f}  "
              f"ℓ̂ {b['ell_hat']:+.2f}±{b['sd_l']:.2f} vs cut "
              f"{C.ELL_STAR[t]:.2f} (π {b['pi']:.2f})  "
              f"t̂ {b['t_hat']:+.2f} (band ±{b['band']:.2f})  "
              f"trainability {b['trainability']:.2f}  "
              f"egate e={b['egate_e']:.1f}  [{status}]{cons_str}")
    # anchor / consolidation trajectory (the F70-ii pilot endpoint)
    anchor_events = [(s["session"], e) for s in sessions
                     for e in s.get("events", []) if e["event"] == "anchor"]
    if anchor_events:
        print("\nGapAnchor trajectory (consolidation endpoint):")
        for sess_no, e in anchor_events:
            print(f"  session {sess_no}: task {e['task']} r̂={e['r_hat']} "
                  f"→ S={e['S_days']} days")
    # D18 handoff recommendation
    conf = [t for t in C.TASKS
            if trials and str(t) in trials[-1]["confirmed"]]
    if conf and len(conf) == len(C.TASKS):
        print("\n★ RECOMMENDATION: all sandbox tasks CONFIRMED — hand off "
              "to re-certification (D18).")
    elif conf:
        print(f"\nrecommendation: {len(conf)}/{len(C.TASKS)} tasks confirmed;"
              " keep training the rest.")
    _figures(trials, sessions)


def _figures(trials, sessions):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                          # matplotlib optional
        print(f"(figures skipped: {e})")
        return
    os.makedirs(C.FIG_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {t: c for t, c in zip(C.TASKS,
                                   ("#9671bd", "#77b5b6", "#7e7e7e"))}
    for t in C.TASKS:
        tr = [x for x in trials if x["task"] == t]
        if not tr:
            continue
        g = [x["global_trial"] for x in tr]
        ell = np.array([x["belief"]["ell_hat"] for x in tr])
        sd = np.array([x["belief"]["sd_l"] for x in tr])
        th = [x["belief"]["t_hat"] for x in tr]
        band = [x["belief"]["band"] for x in tr]
        c = colors.get(t, "#333333")
        axes[0].plot(g, ell, color=c, lw=1.6, label=C.TASK_NAMES[t])
        axes[0].fill_between(g, ell - sd, ell + sd, color=c, alpha=0.18)
        axes[0].axhline(C.ELL_STAR[t], color=c, ls="--", lw=0.9, alpha=0.7)
        axes[1].plot(g, th, color=c, lw=1.6)
        axes[1].plot(g, band, color=c, ls=":", lw=0.8, alpha=0.7)
        axes[1].plot(g, [-b for b in band], color=c, ls=":", lw=0.8,
                     alpha=0.7)
    for s in sessions[1:]:
        first = next((x["global_trial"] for x in trials
                      if x["session"] == s["session"]), None)
        if first is not None:
            for ax in axes:
                ax.axvline(first - 0.5, color="#bbbbbb", lw=0.7, alpha=0.6)
    axes[0].set_ylabel("skill ℓ̂ ± SD (dashed = v15 cut)")
    axes[0].legend(loc="lower right", frameon=False)
    axes[1].set_ylabel("bias t̂ (dotted = derived 0-bias band)")
    axes[1].axhline(0.0, color="#999999", lw=0.8)
    axes[1].set_xlabel("global trial (grey lines = session boundaries)")
    fig.suptitle("sandbox learner trajectory — skill & bias vs the v15 bars")
    fig.tight_layout()
    out = os.path.join(C.FIG_DIR, "trajectory.png")
    fig.savefig(out, dpi=160)
    print(f"\nfigure written: {out}")


if __name__ == "__main__":
    main()
