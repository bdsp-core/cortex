"""M20 — the sandbox protocol engine (one manual/robot session, end to end).

Session flow (the delivery-vehicle prototype):
  1. LOAD persisted state (or initialize a fresh learner).
  2. GAP handling: if wall-clock gap since the last session exceeds
     MIN_GAP_S, run MixtureGapAnchor probe blocks — MEASURE-OR-LEAVE-ALONE
     (F70-iii): only the top-deficiency unconfirmed task and any tasks
     awaiting confirmation get evidence blocks; nothing else is touched.
  3. POST-GAP CONFIRMATION (D33, the C2 rule): a task declared mastered
     last session is only CONFIRMED if the gate still holds after the gap
     and its re-anchor — else the declaration is revoked (kills the F70
     post-gap-transient false graduations).
  4. SERVE up to TRIALS_PER_SESSION trials through the full validated stack:
     TrainerPolicy (deficiency interleave + finish-first D32, cert probes +
     e-gates D23, derived 0-bias band F60, mean-skill off D30) over
     per-task σ∞-mixtures (D22) with anchored rates (D29), v15 bars (D25).
  5. LOG every trial to trials.jsonl (production-shaped telemetry), append
     a session summary, SAVE state. Mid-session quit is safe at any trial.

The responder is injectable: the CLI provides the human (rendered stimulus,
RT-timed keypress) or the robot (simulated learner) — same code path.
"""
from __future__ import annotations

import ast
import json
import os
import time

import numpy as np

from sandbox import config as C
from sandbox import state_io
from sandbox.contact import ContactMonitor, rt_floor_breach
from sandbox.stimulus import draw_s_real, render
from training.bank_adapter import BankAdapter
from training.bridge_conventions import engine_to_plan
from training.consistency import ConsistencyMonitor
from training.mixture_filter import AdaptiveBoundaryHazard
from training.trainer_policy import (ModeThresholds, RetentionScheduler,
                                     TrainerPolicy)


class _Remap:
    def __init__(self, bank, tasks):
        self.bank, self.tasks = bank, tasks

    def candidates(self, task, **kw):
        return self.bank.candidates(self.tasks[task], **kw)


def _jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj) + "\n")


def _belief_snapshot(filt, gate, mp, ell_star):
    sig_hat, t_hat = engine_to_plan(*filt.mean())
    sd_t, sd_l = filt.sd()
    pi, mcse = filt.pass_mass(ell_star)
    return {"sig_hat": round(sig_hat, 4), "t_hat": round(t_hat, 4),
            "ell_hat": round(float(filt.mean()[1]), 4),
            "sd_t": round(sd_t, 4), "sd_l": round(sd_l, 4),
            "pi": round(pi, 4), "mcse": round(mcse, 4),
            "trainability": round(filt.trainability(ell_star), 4),
            "band": round(mp.effective_t_star(filt, sig_hat), 4),
            "egate_e": round(gate.e_now, 3),
            "egate_elevated": bool(gate.elevated)}


class Session:
    def __init__(self, *, now_fn=time.time, seed=None):
        self.now_fn = now_fn
        st = state_io.load_state()
        if st is None:
            st = state_io.fresh_state()
        self.filters, self.gates, self.anchors, self.meta = st
        self.session_no = int(self.meta["session_no"]) + 1
        self.seed = (seed if seed is not None
                     else 100_000 + 7 * self.session_no)
        self.rng = np.random.default_rng(self.seed)
        self.bank = BankAdapter()
        K = len(C.TASKS)
        retention = RetentionScheduler(first_interval=86_400.0)
        for k_str, nxt in self.meta["retention_next"].items():
            retention._next[ast.literal_eval(k_str)] = nxt
        for k_str, ivl in self.meta["retention_ivl"].items():
            retention._ivl[ast.literal_eval(k_str)] = ivl
        self.pol = TrainerPolicy(
            [self.filters[t] for t in C.TASKS],
            [C.ELL_STAR[t] for t in C.TASKS],
            [C.SIGMA_STAR[t] for t in C.TASKS],
            _Remap(self.bank, list(C.TASKS)),
            thresholds=ModeThresholds(meanskill_gate=False,
                                      skill_sigma_z=1.0,
                                      sd_floor=C.SD_FLOOR,
                                      boundary_refractory=
                                      C.BOUNDARY_REFRACTORY,
                                      min_trainability=C.MIN_TRAINABILITY),
            seed=self.seed + 5, probe_every=C.PROBE_EVERY,
            finish_first=True, retention=retention,
            exclude_segids=self.meta["served"],
            # M24 (F85): EP allocation supersedes the D36 discount (the
            # scheduler ignores trainability_floor when progress_alloc is
            # on; kept for provenance/rollback)
            trainability_floor=C.TRAINABILITY_FLOOR,
            finish_sd_tol=C.FINISH_SD_TOL,
            progress_alloc=C.PROGRESS_ALLOC,
            progress_s_sd=C.PROGRESS_S_SD,
            # M25 (F86): EP-v2 lifecycle coupling + exploration floor —
            # inert while PROGRESS_ALLOC is False
            progress_lifecycle=C.PROGRESS_LIFECYCLE,
            refinish_threshold=C.REFINISH_THRESHOLD,
            explore_every=C.EXPLORE_EVERY,
            # M27 (F90): terminal-confirmation semantics — no-op while
            # TERMINAL_CONFIRMATION is False
            terminal_confirmation=C.TERMINAL_CONFIRMATION,
            maint_probes=C.MAINT_PROBES,
            stale_alpha=C.STALE_ALPHA)
        # M27 (F90): already-CONFIRMED tasks resume terminal; restore
        # their stale-gate state from meta
        if C.TERMINAL_CONFIRMATION:
            sg = self.meta.get("stale_gates", {})
            for t_str in self.meta["confirmed"]:
                i = list(C.TASKS).index(int(t_str))
                self.pol.mark_terminal(i)
                if t_str in sg:
                    g = self.pol._stale_gates[i]
                    rec = sg[t_str]
                    g.log_e = np.array(rec[0], dtype=np.float64)
                    g.n = int(rec[1])
                    g.fired = bool(rec[2])
                    g.e_now = float(rec[3])
        # M25 (F86): seed the scheduler's confirmation-lifecycle memory —
        # "ever declared" persists across sessions (a revoked provisional
        # still counts; meta['ever_declared'] accumulates in run())
        ever = (set(self.meta["provisional"]) | set(self.meta["confirmed"])
                | set(self.meta.get("ever_declared", [])))
        for t_str in ever:
            self.pol.scheduler.mark_declared(list(C.TASKS).index(int(t_str)))
        # M26 (F89): declaration-lifecycle hazard state (per-task ever/
        # n-confirms/pending-boundary). Tracked ALWAYS (telemetry + json
        # persistence); the scale is only APPLIED to Gate 4 when
        # HAZARD_ADAPT is on — False is bit-identical to M25.
        self.hazard = AdaptiveBoundaryHazard.from_json(
            self.meta.get("hazard_lifecycle"), K,
            strength=C.HAZARD_STRENGTH, floor=C.HAZARD_FLOOR)
        for t_str in ever:
            self.hazard.ever[list(C.TASKS).index(int(t_str))] = True
        # restore continuity (gates + probe cadence + policy bits)
        for i, t in enumerate(C.TASKS):
            mp = self.pol.mode_policies[i]
            mp.egate = self.gates[t]
            mp._bias_label_balance = self.meta["bias_balance"][str(t)]
            mp._skill_side = self.meta["skill_side"][str(t)]
            mp._last_skill_s = self.meta["last_skill_s"][str(t)]
            self.pol._probe_ctr[i] = self.meta["probe_ctr"][str(t)]
            self.pol._probe_flip[i] = self.meta["probe_flip"][str(t)]
        # M21 (F72/D34): shadow learner-consistency monitors — log-only,
        # never mutate beliefs. A fired monitor means "recent behavior is
        # not the person the belief describes" (switch/strategy/identity).
        cm = self.meta.get("consistency", {})
        self.monitors = {t: (ConsistencyMonitor.from_json(cm[str(t)])
                             if str(t) in cm else ConsistencyMonitor())
                         for t in C.TASKS}
        # M23 (F83): shadow contact e-process — per task, RESTARTED each
        # session (per-session anytime-valid α; the statistic answers "is
        # there perceptual contact NOW"). Log-only (D34 principle).
        self.contact = {t: ContactMonitor(alpha=C.CONTACT_ALPHA)
                        for t in C.TASKS}
        self.events = []

    # ── gap + anchors + confirmation ──
    def open_gap_phase(self):
        """Returns list of (task_bank_idx, n_probes) anchor plans."""
        last = self.meta["last_session_end"]
        now = self.now_fn()
        self.gap_s = 0.0 if last is None else max(now - float(last), 0.0)
        if last is None or self.gap_s < C.MIN_GAP_S:
            return []
        plans, budget = [], C.MAX_ANCHOR_TRIALS
        provisional = {int(t) for t in self.meta["provisional"]}
        unconfirmed = [t for t in C.TASKS
                       if str(t) not in self.meta["confirmed"]]
        if not unconfirmed:
            return []
        # top-deficiency unconfirmed task
        defs = []
        for i, t in enumerate(C.TASKS):
            if str(t) in self.meta["confirmed"]:
                continue
            defs.append((self.pol.scheduler.deficiency(self.filters[t],
                                                       C.ELL_STAR[t]), t))
        top = max(defs)[1]
        order = [top] + [t for t in provisional if t != top]
        for t in order:
            if budget <= 0:
                break
            n = min(self.anchors[t].open_session(
                self.filters[t], self.gap_s, theta_base=0.0, ell_base=0.0,
                seed=self.seed + 900 + t), budget)
            plans.append((t, n))
            budget -= n
        return plans

    def confirm_phase(self, now):
        """D33: confirm or revoke provisional masteries (post-gap)."""
        out = []
        for t_str in list(self.meta["provisional"]):
            t = int(t_str)
            i = list(C.TASKS).index(t)
            ok = self.pol.mode_policies[i].is_mastered(self.filters[t])
            if ok:
                self.meta["confirmed"][t_str] = now
                out.append((t, "CONFIRMED"))
                # M26 (F89): a confirmation demonstrates the gate across
                # the boundary — pays the pending hazard debt
                self.hazard.on_gate_pass(i)
                # M27 (F90): a confirmed task goes terminal (no-op while
                # TERMINAL_CONFIRMATION is False)
                self.pol.mark_terminal(i, now=now)
            else:
                out.append((t, "REVOKED"))
                # M26 (F89): revocation is direct re-draw evidence — the
                # full pinned hazard returns (scale(0) = 1)
                self.hazard.on_revoked(i)
            del self.meta["provisional"][t_str]
        return out

    # ── one full session ──
    def run(self, responder, *, max_trials=None):
        max_trials = max_trials or C.TRIALS_PER_SESSION
        t0 = self.now_fn()
        plans = self.open_gap_phase()
        n_served = 0
        quit_early = False
        self._anchor_served = set()
        self._paused_tasks = set()       # M22 Gate 2 (session-local)
        self._fatigue = []               # M22 Gate 3 (anchor trials count)
        self._rts = []                   # M23 Gate 3b (session-local)
        # anchor probe blocks (count toward the session budget; serve-once
        # holds here too — the pool is re-fetched with cumulative exclusions
        # each probe, else pick_probe's deterministic argmax repeats items)
        for t, n_probes in plans:
            for _ in range(n_probes):
                if n_served >= max_trials or quit_early:
                    break
                excl = set(self.meta["served"]) | self._anchor_served
                pool = self.bank.candidates(t, feedback_safe=True,
                                            exclude_segids=excl)
                if len(pool) == 0:
                    break
                idx = self.anchors[t].pick_probe(pool, C.SIGMA_STAR[t])
                rec, y, quit_early = self._one_trial(
                    t, "anchor", int(pool.seg_id[idx]),
                    float(pool.s_mean[idx]), float(pool.s_sd[idx]),
                    int(pool.y_star[idx]), responder)
                if quit_early:
                    break
                self._anchor_served.add(int(pool.seg_id[idx]))
                self.pol.exclude.add(int(pool.seg_id[idx]))
                self.anchors[t].anchor_step(rec["s_mean"], y, rec["y_star"],
                                            s_sd=rec["s_sd"])
                n_served += 1
            _, r_hat = self.anchors[t].close_anchor(seed=self.seed + 990 + t)
            self.events.append({"event": "anchor", "task": t,
                                "r_hat": round(r_hat, 3),
                                "S_days": round(self.anchors[t].S / 86400, 2)})
        if plans:                        # person-level stability pooling
            S = float(np.mean([self.anchors[t].S for t in C.TASKS]))
            for t in C.TASKS:
                self.anchors[t].S = S
        for t, what in self.confirm_phase(t0):
            self.events.append({"event": what.lower(), "task": t})
        # M23 Gate 4 (F80/F81): session-boundary regime-shift update on
        # EVERY boundary (USER-E's jump crossed a 64 s restart the ≥4 h
        # GapAnchor never sees). Ordering is load-bearing: AFTER the D33
        # confirm phase — confirmation must reflect evidence at close, not
        # prior widening — and before serving, so the widened tail lets one
        # session of likelihood traverse a regime shift (and the F75
        # conservative placement reads the honest post-boundary width).
        if self.meta["last_session_end"] is not None:
            # M26 (F89): π-based hazard credit at the open — AFTER anchors
            # + the D33 confirm phase (the credit reads anchored evidence),
            # BEFORE the boundary re-widens (see credit_from_pi)
            for i, t in enumerate(C.TASKS):
                if self.hazard.ever[i]:
                    pi_o, _ = self.filters[t].pass_mass(C.ELL_STAR[t])
                    self.hazard.credit_from_pi(i, pi_o)
            eps_b, lam_b, ths_b = C.BOUNDARY_JUMP
            for i, t in enumerate(C.TASKS):
                # M26 (F89): evidence-adaptive hazard for declared tasks —
                # ×1.0 (bit-identical) while HAZARD_ADAPT is False
                sc = self.hazard.scale(i) if C.HAZARD_ADAPT else 1.0
                self.filters[t].boundary_shift(eps_b * sc, lam_b, ths_b,
                                               w_share=C.BOUNDARY_WSHARE
                                               * sc)
                self.hazard.on_boundary(i)
                if sc < 1.0:
                    self.events.append({"event": "hazard_scale", "task": t,
                                        "scale": round(sc, 3),
                                        "n_confirms": self.hazard.n[i]})
                # M25 (F88): boundary declaration refractory — no-op at
                # the default BOUNDARY_REFRACTORY = 0
                self.pol.mode_policies[i].note_boundary()
        # M27 (F90): arm the terminal maintenance-probe budget (no-op
        # with an empty terminal set)
        self.pol.note_session_open()
        self._demo_rr = 0
        self._demo_served = set()
        self._demo_flip = {t: 1 for t in C.TASKS}
        # main serving loop
        while n_served < max_trials and not quit_early:
            # M27 (F91): demonstration ramp for tasks without certified
            # perceptual contact (no-op while CONTACT_ONBOARD is False)
            demo_t = None
            if C.CONTACT_ONBOARD:
                cc = self.meta.setdefault("contact_certified", {})
                uncert = [t for t in C.TASKS if not cc.get(str(t))
                          and t not in self._paused_tasks]
                if uncert:
                    demo_t = uncert[self._demo_rr % len(uncert)]
                    self._demo_rr += 1
            if demo_t is not None:
                served_ok = self._demo_trial(demo_t, responder)
                if served_ok is None:        # empty pool — fall through
                    demo_t = None
                elif served_ok is False:
                    break                    # quit_early
                else:
                    n_served += 1
                    continue
            ch = self.pol.step(now=self.now_fn())
            if ch is None:
                self.events.append({"event": "all_mastered_or_empty"})
                break
            t = C.TASKS[ch["task"]]
            rec, y, quit_early = self._one_trial(
                t, ch["mode"] + ("+cert" if ch["info"].get("cert_probe")
                                 else ""),
                ch["seg_id"], ch["s"], ch["s_sd"], ch["y_star"], responder)
            if quit_early:
                break
            self.pol.record(ch, y)
            n_served += 1
            # M27 (F90): a fired stale-mastery gate revokes CONFIRMED —
            # the task returns to the training rotation next pick
            while self.pol.terminal_events:
                tev = self.pol.terminal_events.pop(0)
                t_rev = C.TASKS[tev["task"]]
                self.meta["confirmed"].pop(str(t_rev), None)
                self.hazard.on_revoked(tev["task"])
                self.events.append({"event": "stale_revoked",
                                    "task": t_rev,
                                    "global_trial":
                                        self.meta["global_trial"]})
                print(f"  [protocol] {C.TASK_NAMES[t_rev]} mastery looks "
                      f"stale — the task returns to training.")
            # M22 Gate 2 (F78): consistency pause — a fired flag means the
            # recent responses are not the person/behavior the belief
            # describes; keep updating NOTHING further on that task this
            # session (USER-D: 10 post-flag trials drove trainability
            # 0.22 → 0.04). Suspension is session-local.
            if (C.CONSISTENCY_PAUSE and rec["consistency"]["flag"]
                    and ch["task"] not in self.pol.suspended):
                self.pol.suspended.add(ch["task"])
                self._paused_tasks.add(t)
                self.events.append({"event": "consistency_pause", "task": t,
                                    "glr": rec["consistency"]["glr"],
                                    "global_trial":
                                        self.meta["global_trial"]})
                print(f"  [protocol] {C.TASK_NAMES[t]} paused for this "
                      f"session — recent answers don't match your profile "
                      f"on this task (re-read the instructions; it resumes "
                      f"next session).")
            # M23 Gate 3b (F82): RT-floor careless channel — k consecutive
            # RTs below the floor mean the responses are no longer
            # perceptual decisions (USER-D's collapse ran at ~206 ms vs
            # ≥1059 ms in every engaged window; scanning the trace takes
            # ≳1 s). Union with the F79 accuracy guard below; fires ~3
            # trials earlier on the D-shaped collapse, 0 engaged FAs.
            if rt_floor_breach(self._rts, floor_ms=C.RT_FLOOR_MS,
                               k=C.RT_FLOOR_K):
                self.events.append(
                    {"event": "rt_floor_break",
                     "rt_ms": [round(r, 0) for r in
                               self._rts[-C.RT_FLOOR_K:]],
                     "global_trial": self.meta["global_trial"]})
                print(f"  [protocol] responses are coming back faster than "
                      f"the display can be read — ending the session here; "
                      f"progress is saved.")
                break
            # M22 Gate 3 (F79): fatigue guard — end the session when the
            # trailing realized-vs-predicted accuracy deficit rises well
            # above the SESSION'S OWN EARLY BASELINE (differenced: fatigue
            # is a decline, not a level — a raw deficit false-fires on
            # onboarding belief-optimism; the differenced form fires on the
            # two real late-session collapses (USER-B s2 +0.41, USER-D s1
            # +0.37) and stays silent on USER-A/C).
            if len(self._fatigue) >= 2 * C.FATIGUE_W:
                base = self._fatigue[:C.FATIGUE_W]
                w = self._fatigue[-C.FATIGUE_W:]
                deficit = ((np.mean([p for _, p in w])
                            - np.mean([c for c, _ in w]))
                           - (np.mean([p for _, p in base])
                              - np.mean([c for c, _ in base])))
                if deficit > C.FATIGUE_DELTA:
                    self.events.append(
                        {"event": "fatigue_break",
                         "deficit": round(float(deficit), 3),
                         "global_trial": self.meta["global_trial"]})
                    print(f"  [protocol] accuracy has dropped well below "
                          f"your level earlier this session (drop "
                          f"{deficit:.2f}) — ending the session here; "
                          f"progress is saved.")
                    break
            # provisional-declaration watch (D33 lifecycle) — sweep ALL
            # tasks: a newly-mastered task stops being served (the
            # scheduler excludes it), so watching only the served task
            # would never observe its own declaration
            for i2, t2 in enumerate(C.TASKS):
                known = (str(t2) in self.meta["provisional"]
                         or str(t2) in self.meta["confirmed"])
                if (not known
                        and self.pol.mode_policies[i2].is_mastered(
                            self.filters[t2])):
                    self.meta["provisional"][str(t2)] = self.now_fn()
                    # M25 (F86): the lifecycle memory survives revocation
                    ev = set(self.meta.get("ever_declared", []))
                    ev.add(str(t2))
                    self.meta["ever_declared"] = sorted(ev)
                    self.hazard.on_gate_pass(i2)      # M26 (F89)
                    self.events.append({"event": "provisional", "task": t2,
                                        "global_trial":
                                            self.meta["global_trial"]})
                elif known and self.hazard.pending[i2]:
                    # M26 (F89): an ever-declared task re-demonstrating
                    # the gate after this session's boundary pays the
                    # hazard debt (the re-polish completing is itself the
                    # "no re-draw" observation)
                    if self.pol.mode_policies[i2].is_mastered(
                            self.filters[t2]):
                        self.hazard.on_gate_pass(i2)
        self._finalize(t0, n_served, quit_early)
        return n_served, self.events

    def _demo_trial(self, t, responder):
        """M27 (F91): one demonstration trial — the easiest available
        feedback-safe item on the alternating label side, filter stepped
        directly (protocol-layer serving, like anchor blocks). Returns
        True when served, None on an empty pool, False on quit."""
        excl = (set(self.meta["served"])
                | getattr(self, "_anchor_served", set())
                | getattr(self, "_demo_served", set())
                | self.pol.exclude | self.pol._served)
        pool = self.bank.candidates(t, feedback_safe=True,
                                    exclude_segids=excl)
        if len(pool) == 0:
            return None
        want = 1 if self._demo_flip[t] > 0 else 0
        self._demo_flip[t] *= -1
        m = np.where(pool.y_star == want)[0]
        if m.size == 0:
            m = np.arange(len(pool))
        sign = 1.0 if want == 1 else -1.0
        sc = sign * pool.s_mean[m] - pool.s_sd[m]
        j = int(m[np.argmax(sc)])
        rec, y, quit_early = self._one_trial(
            t, "demo", int(pool.seg_id[j]), float(pool.s_mean[j]),
            float(pool.s_sd[j]), int(pool.y_star[j]), responder)
        if quit_early:
            return False
        self.filters[t].step(rec["s_mean"], y, rec["y_star"],
                             s_sd=rec["s_sd"])
        self._demo_served.add(int(pool.seg_id[j]))
        self.pol.exclude.add(int(pool.seg_id[j]))
        return True

    def _one_trial(self, task, mode, seg_id, s_mean, s_sd, y_star, responder):
        s_real = draw_s_real(s_mean, s_sd, seg_id, self.session_no,
                             y_star=y_star)
        trace, marker, samples = render(s_real, seg_id, self.session_no)
        i = list(C.TASKS).index(task)
        mp, filt, gate = (self.pol.mode_policies[i], self.filters[task],
                          self.gates[task])
        meta = {"task": task, "task_name": C.TASK_NAMES[task],
                "trial_in_session":
                    self.meta["global_trial"] % 10_000,  # display only
                "s_real": s_real, "trace": trace, "marker": marker}
        y, rt_ms, quit_early = responder(meta)
        if quit_early:
            return None, None, True
        correct = int(y == y_star)
        # shadow consistency update (filter is still PRE-update here; its
        # predictive is the monitor's null)
        p_pred = filt.predictive_p(s_mean, s_sd)
        cons = self.monitors[task].update(s_mean, s_sd, y, p_pred,
                                          rt_ms=rt_ms)
        # M23 (F83): shadow contact e-process + Gate-3b RT tracking
        cont = self.contact[task].update(correct)
        self._rts.append(float(rt_ms))
        if cont["contact"] and not getattr(self.contact[task],
                                           "_announced", False):
            self.contact[task]._announced = True
            self.events.append({"event": "contact", "task": task,
                                "n": self.contact[task].n,
                                "global_trial": self.meta["global_trial"]})
            # M27 (F91): first-EVER certification persists; under the
            # onboarding protocol the task's belief takes one Gate-4
            # shift at the handoff (contact IS the regime shift the M23
            # hazard models — evidence-triggered instead of boundary-
            # scheduled), re-opening guessing-poisoned trainability
            cc = self.meta.setdefault("contact_certified", {})
            if not cc.get(str(task)):
                cc[str(task)] = self.meta["global_trial"]
                if (C.CONTACT_ONBOARD and C.DEMO_HANDOFF_SHIFT
                        and self.filters[task].trainability(
                            C.ELL_STAR[task])
                        < C.DEMO_HANDOFF_MIN_TRAIN):
                    self.filters[task].boundary_shift(
                        *C.BOUNDARY_JUMP, w_share=C.BOUNDARY_WSHARE)
                    i_t = list(C.TASKS).index(task)
                    self.pol.mode_policies[i_t].note_boundary()
                    self.hazard.on_boundary(i_t)
                    self.events.append({"event": "contact_handoff",
                                        "task": task})
        # fatigue-guard pair: realized vs belief-predicted correctness
        self._fatigue.append((correct,
                              p_pred if y_star == 1 else 1.0 - p_pred))
        if cons["flag"]:
            self.events.append({"event": "consistency_flag", "task": task,
                                "glr": cons["glr"], "rt_z": cons["rt_z"],
                                "global_trial": self.meta["global_trial"]})
        rec = {"epoch": self.now_fn(), "session": self.session_no,
               "global_trial": self.meta["global_trial"],
               "task": task, "task_name": C.TASK_NAMES[task],
               "seg_id": seg_id, "mode": mode,
               "s_mean": s_mean, "s_sd": s_sd, "s_real": round(s_real, 4),
               "y": int(y), "y_star": int(y_star), "correct": correct,
               "rt_ms": round(rt_ms, 1), "feedback": True,
               "consistency": cons, "contact": cont,
               "belief": _belief_snapshot(filt, gate, mp, C.ELL_STAR[task]),
               "provisional": list(self.meta["provisional"]),
               "confirmed": list(self.meta["confirmed"])}
        _jsonl(C.TRIALS_JSONL, rec)
        self.meta["global_trial"] += 1
        return rec, int(y), False

    def _finalize(self, t0, n_served, quit_early):
        now = self.now_fn()
        self.meta["session_no"] = self.session_no
        self.meta["last_session_end"] = now
        self.meta["served"] = sorted(set(self.meta["served"])
                                     | self.pol._served
                                     | getattr(self, "_anchor_served", set())
                                     | getattr(self, "_demo_served", set()))
        # persist policy continuity + retention bins
        for i, t in enumerate(C.TASKS):
            mp = self.pol.mode_policies[i]
            self.meta["bias_balance"][str(t)] = mp._bias_label_balance
            self.meta["skill_side"][str(t)] = mp._skill_side
            self.meta["last_skill_s"][str(t)] = mp._last_skill_s
            self.meta["probe_ctr"][str(t)] = self.pol._probe_ctr[i]
            self.meta["probe_flip"][str(t)] = self.pol._probe_flip[i]
        self.meta["retention_next"] = {repr(k): v for k, v
                                       in self.pol.retention._next.items()}
        self.meta["retention_ivl"] = {repr(k): v for k, v
                                      in self.pol.retention._ivl.items()}
        # M22 Gate 2: a paused task starts the next session with a FRESH
        # monitor window — a re-instructed participant must not be
        # insta-flagged by the stale anomalous window (if the behavior
        # persists, a fresh window re-flags within ~a window's length)
        from training.consistency import ConsistencyMonitor as _CM
        for t in getattr(self, "_paused_tasks", set()):
            self.monitors[t] = _CM()
        self.meta["consistency"] = {str(t): self.monitors[t].to_json()
                                    for t in C.TASKS}
        self.meta["hazard_lifecycle"] = self.hazard.to_json()   # M26 (F89)
        # M27 (F90): persist terminal stale-gate state
        self.meta["stale_gates"] = {
            str(C.TASKS[i]): [list(map(float, g.log_e)), g.n,
                              bool(g.fired), float(g.e_now)]
            for i, g in self.pol._stale_gates.items()}
        summary = {"epoch_start": t0, "epoch_end": now,
                   "session": self.session_no, "gap_s": round(self.gap_s, 1),
                   "n_trials": n_served, "quit_early": bool(quit_early),
                   "events": self.events,
                   "tasks": {str(t): _belief_snapshot(
                       self.filters[t], self.gates[t],
                       self.pol.mode_policies[list(C.TASKS).index(t)],
                       C.ELL_STAR[t]) for t in C.TASKS}}
        _jsonl(C.SESSIONS_JSONL, summary)
        state_io.save_state(self.filters, self.gates, self.anchors, self.meta)
