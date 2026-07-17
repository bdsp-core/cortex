"""Learning-engine training service (Phase L3): the vendored engine as the
server-side training decision-maker behind the web API.

The engine is a PURE decision-maker here: it holds a per-sitting particle
belief, seeds it by replaying the participant's latest certification
sitting (handoff contract v1.1 §2a — the raw stream, full n-way picks),
and answers "which item next?" / "update on this response". It writes
NOTHING: the per-trial ledger keeps its single validated writer (the
client checkpoint outbox → `record_training_progress`), and the belief is
REBUILDABLE from that same ledger, so a process restart loses no state
(belief = f(artifact, test stream, recorded responses)).

Design constraints honored:
  - heavy imports (numpy + the vendored package) are LAZY: this module
    imports under the minimal-CI environment; endpoints 503 when the
    numeric stack is absent;
  - session-time code is numpy-only (JAX stays calibration-side — the
    vendored package's __init__ resolves calibration exports lazily);
  - the single-Uvicorn-worker invariant makes the in-memory session cache
    sound; a threading lock covers FastAPI's sync-endpoint threadpool;
  - cut targets come from the bundle manifest's `ellStar` (the v15 block
    the exam certifies at — trainer and exam share one source);
  - serving is one-vs-rest binary (the current training UI contract);
    native n-way serving remains the UI-gated upgrade.

Rebuild caveat (documented, not hidden): restoring a sitting replays the
ledger's recorded responses, which reproduces the BELIEF exactly; the
placement side-alternation counters restart, so the post-restart item
sequence may differ from the un-restarted one. Any response not yet
flushed by the client outbox at crash time is absent from the rebuild —
the belief is then one answer behind until the outbox re-flushes.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


def _now_epoch() -> float:
    return time.time()


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _utc_now() -> str:
    return _iso(_now_epoch())

# The vendored package lives INSIDE cortex_web/ so the standard deploy
# (which rsyncs cortex_web/ only) ships it to the box; the repo-root
# location is the pre-integration fallback for older checkouts.
_CW = Path(__file__).resolve().parents[2]          # cortex_web/
_PKG = _CW / "learning-engine-cleaned"
if not _PKG.exists():
    _PKG = _CW.parent / "learning-engine-cleaned"
ARTIFACT_PATH = _PKG / "artifacts" / "nway_dynamics_v1_1.json"
N_PARTICLES = 400
# Replay seeding runs on an EXPANDED cloud contracted to N_PARTICLES after
# (the D51 depletion guard): ~150 reweight-only observations collapse a
# 400-particle cloud to a handful of ancestors, which turned the futility
# and bias gates into per-sitting coin flips (measured live, 2026-07-17:
# trainability 0.9999 vs 0.0000 across seeds for the same learner).
REPLAY_EXPAND = 4
N_CASE = 24
# Mastery/futility design inputs (host ModeThresholds Z/sd_floor). ALPHA is
# the serving-gate risk: 0 = PRACTICE MODE (the pilot default — futility and
# mastery retirement are REPORTED, never serving-blocking; roadmap §2.1
# report-first doctrine. The first live learner measured far below every
# v15 bar — the V2 cut-reachability finding — so certification-mode
# futility honestly refuses to serve anything, which is the wrong posture
# for a practice trainer). Set CORTEX_TRAINER_ENGINE_ALPHA=0.05 for the
# certification-economy behavior.
ALPHA = float(os.environ.get("CORTEX_TRAINER_ENGINE_ALPHA", "0"))
Z, SD_FLOOR = 2.0, 0.23
# ── Phase L4 serving-layer design inputs (env-tunable; DESIGN-category
# constants in the roadmap ledger — they shape serving, never the belief
# or selection mathematics). ─────────────────────────────────────────
# Interleave cap: after this many consecutive items on one domain the
# allocator must offer another eligible domain (greedy value-per-item
# locked 109/120 items onto one domain live; sim-checked, see D57).
MAX_CONSEC = int(os.environ.get("CORTEX_TRAINER_MAX_CONSEC", "8"))
# Retention (the incumbent's expanding-interval pattern, domain-level):
# every REVIEW_EVERY-th question serves a due review item, at most
# REVIEW_MAX per sitting; a correct review multiplies the domain's
# interval by RETENTION_EASE, a miss resets it to RETENTION_FIRST_S.
REVIEW_EVERY = int(os.environ.get("CORTEX_TRAINER_REVIEW_EVERY", "10"))
REVIEW_MAX = int(os.environ.get("CORTEX_TRAINER_REVIEW_MAX", "4"))
RETENTION_FIRST_S = float(os.environ.get(
    "CORTEX_TRAINER_RETENTION_FIRST_S", "600"))
RETENTION_EASE = 2.0
# Native n-way training serving (full identification pick on the n-way
# family — the exam's own format and coding; ~1.6 bits/trial more than
# the binarized question). "off" reverts to one-vs-rest serving.
NWAY = os.environ.get("CORTEX_TRAINER_NWAY", "on").strip().lower() != "off"

_ENG = None


def _engine():
    """Lazy, cached import of the numeric stack + vendored package."""
    global _ENG
    if _ENG is None:
        import sys
        if str(_PKG) not in sys.path:
            sys.path.insert(0, str(_PKG))
        import numpy as np
        from learning_engine import MixedBelief, Registry
        from adapter.le_adapter import LETrainerPolicy
        _ENG = (np, MixedBelief, Registry, LETrainerPolicy)
    return _ENG


def enabled(cfg: dict, code: str, participant) -> bool:
    """The engine-trainer exposure gate: CORTEX_TRAINER_ENGINE ∈
    off|cohort|all (mirrors dashboard_logic.training_enabled; default
    OFF — the incumbent client trainer stays the production posture
    until the team promotes the engine)."""
    mode = (cfg or {}).get("trainer_engine", "all")
    if mode == "off":
        return False
    if mode == "all":
        return True
    allow = (cfg or {}).get("trainer_engine_allowlist") or frozenset()
    if not allow:
        return False
    cand = {str(code).strip().lower()}
    for key in ("public_id", "email"):
        v = (participant or {}).get(key)
        if v:
            cand.add(str(v).strip().lower())
    return bool(cand & allow)


def _seed_from(training_id: str) -> int:
    # Stable, python-hash-independent seed from the sitting id.
    import hashlib
    return int.from_bytes(
        hashlib.sha256(training_id.encode()).digest()[:6], "big")


class EngineSession:
    """One participant sitting: belief + policy + pending-choice ledger."""

    def __init__(self, db, bank, training_id: str, code: str,
                 seg_ids, restrict_ks=None):
        np, MixedBelief, Registry, LETrainerPolicy = _engine()
        self.np = np
        self.training_id, self.code = training_id, code
        codes = list(bank.engine["taskCodes"])
        self.codes = codes
        art = json.loads(ARTIFACT_PATH.read_text())["artifact"]
        if len(art["codes"]) != len(codes) - 1:
            raise ValueError("artifact n-way domain count != manifest tasks")
        art = dict(art, codes=codes[1:])       # position remap onto task order
        self.reg = Registry([(codes[0], "binary")]
                            + [(c, "g") for c in codes[1:]])
        ell = bank.engine["ellStar"]           # the manifest's certifying block
        self.ell_star = {c: float(ell[k]) for k, c in enumerate(codes)}

        pool = [bank._by_id[int(s)] for s in (seg_ids or [])
                if int(s) in bank._by_id]
        self._by_id = {int(s["segId"]): s for s in bank.segments}
        self._cand = {}
        for k, c in enumerate(codes):
            rows = [s for s in pool
                    if k in (s.get("applicableTaskIdx") or [])]
            self._cand[c] = dict(
                seg_id=np.array([s["segId"] for s in rows], dtype=np.int64),
                s=np.array([float(s["sMean"][k]) for s in rows]),
                s_sd=np.array([float((s.get("sSd") or [0.0] * len(codes))[k])
                               for s in rows]),
                y_star=np.array([1 if float(s["sMean"][k]) > 0 else 0
                                 for s in rows], dtype=np.int64))
        # Case mix for the n-way reduction map: drawn from the FULL bank
        # (a declared evaluation mix, stable across sittings).
        words = list(bank.engine["taskPatternWords"])
        case_mix = []
        per_dom = {c: 0 for c in codes[1:]}
        for s in bank.segments:
            cls = s.get("patternClass")
            if cls in words[1:]:
                k = words.index(cls)
                dom = codes[k]
                if per_dom.get(dom, N_CASE) < N_CASE:
                    per_dom[dom] += 1
                    case_mix.append(dict(
                        group="g", s=[float(x) for x in s["sMean"][1:]],
                        gold=k - 1))
        restrict = None
        if restrict_ks:
            restrict = {codes[int(k)] for k in restrict_ks
                        if 0 <= int(k) < len(codes)}
        belief = MixedBelief(art, self.reg, N=N_PARTICLES * REPLAY_EXPAND,
                             rng=np.random.default_rng(
                                 _seed_from(training_id)))
        self.policy = LETrainerPolicy(
            belief, self.reg, self.ell_star,
            lambda c: self._cand.get(c),
            alpha=ALPHA, Z=Z, sd_floor=SD_FLOOR, case_mix=case_mix,
            restrict=restrict,
            rng=np.random.default_rng(_seed_from(training_id) + 1))
        self.restrict = restrict
        self._pending: dict[int, dict] = {}
        self.seq = 0
        # L4 serving-layer state: n-way exemplar pools (true-class segs per
        # n-way domain, the identification question's candidates), the
        # cross-sitting retention table, and the interleave/review counters.
        # Mixed-gold n-way pool (D58): ALL group-class segments, any true
        # class — difficulty is targeted on the TRAINED domain's channel,
        # but the gold (the correct button) is the segment's own class, so
        # it varies unpredictably item to item. True-class-only pools made
        # the gold constant within a domain run (exploited live: 36/36 @
        # 368 ms by pressing one button). Same decoupling the belief math
        # always had: the softmax update credits all channels regardless
        # of gold; only the candidate-set definition changes.
        words = list(bank.engine["taskPatternWords"])
        grows = [s for s in pool if s.get("patternClass") in words[1:]]
        self._nway_pool = dict(
            seg_id=np.array([s["segId"] for s in grows], dtype=np.int64),
            srow=np.array([[float(x) for x in s["sMean"][1:]]
                           for s in grows]).reshape(len(grows), -1),
            gold=np.array([words.index(s["patternClass"])
                           for s in grows], dtype=np.int64))
        self._db = db
        self._retention = {r["domain"]: dict(r)
                           for r in db.get_retention(code)}
        self._n_served = 0
        self._reviews_served = 0
        self._consec = [None, 0]      # [domain, run length]
        self._seed_belief(db)
        self._rebuild_from_ledger(db)
        self._contract_belief(N_PARTICLES)
        self.attain0 = self.attainability()   # seed-time report (persisted)

    # ── seeding + rebuild ─────────────────────────────────────────
    def _item_for(self, seg: dict, task_k: int):
        if task_k == 0:
            return dict(domain=self.codes[0], s=float(seg["sMean"][0]))
        return dict(group="g", s=[float(x) for x in seg["sMean"][1:]])

    def _seed_belief(self, db) -> None:
        """Replay the latest finalized certification sitting (contract
        §2a): reweight-only — the test shows no feedback. No sitting →
        the population prior stands (still valid, just wider)."""
        latest = db.latest_result_for_code(self.code)
        src = latest.get("session_id") if latest else None
        if not src:
            self.n_seeded = 0
            return
        n = 0
        for t in db.session_trials(src):
            if t.get("pick") is None or t["trial_index"] != n:
                break
            seg = self._by_id.get(int(t["seg_id"]))
            if seg is not None:
                k = int(t["task_k"] or 0)
                p = int(t["pick"])
                # Pick coding (measured on real web data): picks are 0-based
                # TASK-axis indices; the binary task answers 0 = "yes" with
                # sentinel K = "no"; n-way group index = pick - 1.
                if k == 0:
                    self.policy.bel.observe(self._item_for(seg, 0),
                                            1 if p == 0 else 0)
                else:
                    g = p - 1
                    if 0 <= g < len(self.codes) - 1:
                        self.policy.bel.observe(self._item_for(seg, k), g)
            n += 1
        self.n_seeded = n

    def _rebuild_from_ledger(self, db) -> None:
        """Re-apply THIS sitting's already-recorded responses (feedback
        shown ⇒ dynamics advance) in seq order — crash/restart recovery."""
        rows = db._fetchall(
            "SELECT * FROM training_trials WHERE training_id=? AND "
            "pick IS NOT NULL ORDER BY COALESCE(seq_in_session, 0)",
            (self.training_id,))
        for r in rows:
            seg = self._by_id.get(int(r["seg_id"]))
            k = int(r["task_k"] or 0)
            if seg is None:
                continue
            if (r["link"] if "link" in r.keys() else None) == "nway":
                g = int(r["pick"]) - 1        # task-axis -> group index
                gy = int(r["y_star"] or k)    # gold = true class (D58);
                # legacy rows carried y_star == trained domain — identical
                if 0 <= g < len(self.codes) - 1:
                    self.policy.bel.update(
                        dict(group="g",
                             s=[float(x) for x in seg["sMean"][1:]],
                             gold=gy - 1), g)
            else:
                item = dict(domain=self.codes[k],
                            s=float(seg["sMean"][k]),
                            gold=int(r["y_star"] or 0))
                self.policy.bel.update(item, int(r["pick"]))
            self.policy.served.add(int(r["seg_id"]))
            self._n_served += 1
            self.seq = max(self.seq, int(r["seq_in_session"] or 0) + 1)

    def _contract_belief(self, n: int) -> None:
        """Weighted subsample of the expanded seeding cloud back to the
        session size — the D51 depletion guard's contraction step (the
        vendored MSBelief has this as `_shrink`; MixedBelief keeps its
        arrays public, so the service applies the same operation here).
        Records the surviving unique-ancestor count for diagnostics."""
        np, bel = self.np, self.policy.bel
        idx = bel.rng.choice(bel.N, size=n, p=bel.w)
        for nm in ("t", "u", "u_inf"):
            setattr(bel, nm, getattr(bel, nm)[idx].copy())
        for nm in ("a_t", "a_s"):
            v = getattr(bel, nm)
            if getattr(v, "ndim", 1) == 2:
                setattr(bel, nm, v[idx].copy())
        bel.N = n
        bel.w = np.full(n, 1.0 / n)
        self.seed_unique = int(len(np.unique(idx)))

    def attainability(self) -> dict:
        """Per-domain trainability report (roadmap §2.1: attainability is
        REPORTED first; in practice mode it never blocks serving)."""
        return {c: round(float(self.policy.trainability(c)), 4)
                for c in self.codes}

    # ── serving surface (L4 wrappers around the untouched selector) ──
    def _nway_choice_for(self, code: str, mode: str):
        """Full-identification item for an n-way domain (D58 mixed-gold):
        difficulty targeted on the TRAINED domain's channel with the D56
        iid side coin — positive side lands true exemplars of the domain,
        negative side lands confusable other-class segments near the same
        difficulty — so the correct button varies unpredictably while the
        placement formula stays byte-identical."""
        np = self.np
        c = self._nway_pool
        if not len(c["seg_id"]):
            return None
        keep = ~np.isin(c["seg_id"], np.fromiter(
            self.policy.served, dtype=np.int64,
            count=len(self.policy.served))) if self.policy.served else \
            np.ones(len(c["seg_id"]), bool)
        if not keep.any():
            return None
        j = self.reg.index[code]
        k = self.codes.index(code)
        t_bar, u_bar = self.policy.bel.mean_state()
        z = (c["srow"][keep, k - 1] - t_bar[j]) / np.exp(u_bar[j])
        side = self.policy._side(code)
        i = int(np.argmin(np.abs(z - side * self.policy.zstar)))
        ii = np.where(keep)[0][i]
        return dict(task=code, seg_id=int(c["seg_id"][ii]),
                    s=float(c["srow"][ii][k - 1]), s_sd=0.0,
                    y_star=int(c["gold"][ii]),   # task-axis TRUE class
                    mode=mode, link="nway",
                    srow=[float(x) for x in c["srow"][ii]])

    def _binary_choice_for(self, code: str, mode: str):
        """Best one-vs-rest item for a GIVEN domain — the policy's own
        skill-mode selection applied to one domain (used for reviews)."""
        np = self.np
        c = self._cand.get(code)
        if c is None or not len(c["seg_id"]):
            return None
        keep = ~np.isin(c["seg_id"], np.fromiter(
            self.policy.served, dtype=np.int64,
            count=len(self.policy.served))) if self.policy.served else \
            np.ones(len(c["seg_id"]), bool)
        if not keep.any():
            return None
        j = self.reg.index[code]
        t_bar, u_bar = self.policy.bel.mean_state()
        z = (c["s"][keep] - t_bar[j]) / np.exp(u_bar[j])
        side = self.policy._side(code)
        i = int(np.argmin(np.abs(z - side * self.policy.zstar)))
        ii = np.where(keep)[0][i]
        return dict(task=code, seg_id=int(c["seg_id"][ii]),
                    s=float(c["s"][ii]), s_sd=float(c["s_sd"][ii]),
                    y_star=int(c["y_star"][ii]), mode=mode, link="binary")

    def _choice_for(self, code: str, mode: str):
        if NWAY and self.reg.link.get(code) != "binary":
            return self._nway_choice_for(code, mode)
        return self._binary_choice_for(code, mode)

    def _review_choice(self):
        """R1 retention: the longest-overdue domain with servable items
        (the incumbent's expanding-interval pattern, domain granularity)."""
        now = _utc_now()
        due = sorted((r["next_due_utc"], d)
                     for d, r in self._retention.items()
                     if r["next_due_utc"] <= now)
        for _, d in due:
            ch = self._choice_for(d, "review")
            if ch is not None:
                return ch
        return None

    def _step_with_cap(self):
        """R2 interleave: the validated allocator picks; after MAX_CONSEC
        consecutive items on one domain it must offer another eligible
        domain (fallback to the hot domain when nothing else is left).
        Serving constraint only — per-domain selection math untouched."""
        hot, run = self._consec
        if hot is not None and run >= MAX_CONSEC:
            saved = self.policy.restrict
            base = set(saved) if saved is not None else set(self.codes)
            self.policy.restrict = base - {hot}
            try:
                ch = self.policy.step()
            finally:
                self.policy.restrict = saved
            if ch is not None:
                return ch
        return self.policy.step()

    def next_item(self):
        ch = None
        if (self._reviews_served < REVIEW_MAX and self._n_served > 0
                and self._n_served % REVIEW_EVERY == 0):
            ch = self._review_choice()
        if ch is None:
            ch = self._step_with_cap()
            if ch is not None and NWAY \
                    and self.reg.link.get(ch["task"]) != "binary":
                # native n-way serving: same allocation winner, served in
                # the identification format (the exam's own coding)
                ch = self._nway_choice_for(ch["task"], ch["mode"]) or ch
        if ch is None:
            return None
        self._pending[int(ch["seg_id"])] = ch
        return dict(task=self.codes.index(ch["task"]),
                    segId=int(ch["seg_id"]), s=float(ch["s"]),
                    sSd=float(ch.get("s_sd") or 0.0),
                    yStar=int(ch["y_star"]), mode=str(ch["mode"]),
                    link=str(ch.get("link", "binary")))

    def _apply_retention(self, code: str, correct: bool) -> None:
        now_s = _now_epoch()
        cur = self._retention.get(code)
        if cur is None:
            iv = RETENTION_FIRST_S
        else:
            iv = (float(cur["interval_s"]) * RETENTION_EASE if correct
                  else RETENTION_FIRST_S)
        nxt = _iso(now_s + iv)
        self._retention[code] = dict(domain=code, next_due_utc=nxt,
                                     interval_s=iv)
        self._db.upsert_retention(self.code, code, nxt, iv)

    def record(self, seg_id: int, pick: int):
        ch = self._pending.pop(int(seg_id), None)
        if ch is None:
            raise KeyError(f"no pending engine item for segment {seg_id}")
        pick = int(pick)
        if ch.get("link") == "nway":
            g = pick - 1                     # task-axis -> group index
            if not (0 <= g < len(self.codes) - 1):
                raise KeyError(f"n-way pick {pick} out of range")
            self.policy.bel.update(          # gold = the seg's TRUE class
                dict(group="g", s=ch["srow"],
                     gold=int(ch["y_star"]) - 1), g)
            self.policy.served.add(int(ch["seg_id"]))
            self.policy.log.append(dict(
                {kk: v for kk, v in ch.items() if kk != "srow"}, y=pick))
            correct = (pick == int(ch["y_star"]))
        else:
            self.policy.record(ch, pick)
            correct = (pick == int(ch["y_star"]))
        # retention bookkeeping: reviews reschedule by outcome; first
        # contact with a domain registers it (the incumbent's register())
        if ch["mode"] == "review":
            self._apply_retention(ch["task"], correct)
        elif ch["task"] not in self._retention:
            self._apply_retention(ch["task"], False)   # register at first
        if ch["mode"] == "review":
            self._reviews_served += 1
        hot, run = self._consec
        self._consec = [ch["task"],
                        run + 1 if hot == ch["task"] else 1]
        self._n_served += 1
        self.seq += 1
        return ch

    def snapshot(self):
        np, w = self.np, self.policy.bel.w
        out = []
        for k, c in enumerate(self.codes):
            j = self.reg.index[c]
            e = self.policy._ell_cloud(c)
            mu = float(w @ e)
            sd = float(np.sqrt(max(float(w @ (e * e)) - mu * mu, 0.0)))
            out.append(dict(
                task=k, mastered=bool(self.policy.is_mastered(c)),
                skill=mu,
                theta=float(-(w @ self.policy.bel.t[:, j])),  # engine coords
                sd=sd, passMass=float(self.policy.pass_mass(c)),
                trainability=float(self.policy.trainability(c))))
        return out

    def all_mastered(self) -> bool:
        codes = (self.restrict if self.restrict is not None
                 else [c for c in self.codes if len(self._cand[c]["seg_id"])])
        return all(self.policy.is_mastered(c) for c in codes) if codes \
            else False


class EngineManager:
    """In-memory sitting cache (single-worker invariant); every sitting is
    rebuildable from the DB, so eviction/restart is safe."""

    def __init__(self):
        self._sessions: dict[str, EngineSession] = {}
        self._lock = threading.Lock()

    def start(self, db, bank, training_id, code, seg_ids, restrict_ks):
        with self._lock:
            es = EngineSession(db, bank, training_id, code, seg_ids,
                               restrict_ks)
            self._sessions[training_id] = es
            return es, es.next_item(), es.snapshot()

    def record(self, training_id, seg_id, pick):
        with self._lock:
            es = self._sessions.get(training_id)
            if es is None:
                raise LookupError(training_id)
            es.record(seg_id, pick)
            return es, es.next_item(), es.snapshot()
