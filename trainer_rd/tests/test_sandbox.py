"""M20 smoke test — the manual-tester sandbox (isolated state; never touches
sandbox/state or sandbox/logs).

Run:  python3 -m tests.test_sandbox
"""
import json
import os
import shutil
import tempfile

import numpy as np

N_CHECK = 0


def ok(cond, msg):
    global N_CHECK
    N_CHECK += 1
    assert cond, f"CHECK {N_CHECK} FAILED: {msg}"
    print(f"  ok: {msg}")


# ── isolate all sandbox paths BEFORE anything touches them ──
import sandbox.config as C

TMP = tempfile.mkdtemp(prefix="sandbox_test_")
C.STATE_DIR = os.path.join(TMP, "state")
C.LOG_DIR = os.path.join(TMP, "logs")
C.FIG_DIR = os.path.join(C.LOG_DIR, "figures")
C.TRIALS_JSONL = os.path.join(C.LOG_DIR, "trials.jsonl")
C.SESSIONS_JSONL = os.path.join(C.LOG_DIR, "sessions.jsonl")
import sandbox.state_io as sio

sio.STATE_NPZ = os.path.join(C.STATE_DIR, "state.npz")
sio.STATE_JSON = os.path.join(C.STATE_DIR, "state.json")

from sandbox.protocol import Session
from sandbox.stimulus import draw_s_real, render, window_elevation
from training.learner_sim import Learner, LearnerParams

# ── 1. stimulus: rendered evidence is monotone in s_real ──
els = []
for s_real in (-2.0, -0.5, 0.5, 2.0):
    vals = [window_elevation(render(s_real, seg, 1)[2])
            for seg in range(200)]
    els.append(np.mean(vals))
ok(all(els[i] < els[i + 1] for i in range(3)),
   f"stimulus monotone in s_real (window elevations {np.round(els, 2)})")
ok(render(1.0, 42, 3)[0] == render(1.0, 42, 3)[0],
   "rendering deterministic given (seg, session)")

# ── 1b. M21 (F74): label-conditioned draw never contradicts the label ──
draws1 = [draw_s_real(0.3, 0.9, seg, 1, y_star=1) for seg in range(300)]
draws0 = [draw_s_real(0.3, 0.9, seg, 1, y_star=0) for seg in range(300)]
ok(all(d > 0 for d in draws1) and all(d <= 0 for d in draws0),
   "s_real draw is truncated to the label-consistent sign (F74)")
ok(draw_s_real(0.3, 0.9, 7, 1, y_star=1)
   == draw_s_real(0.3, 0.9, 7, 1, y_star=1),
   "truncated draw still deterministic per (seg, session)")

# ── 1c. M21 (F73): render physics leave headroom to the v15 bars ──
import math

sig_ideal = C.RENDER_NOISE * math.sqrt(
    1.0 / (C.TEMPLATE_HI - C.TEMPLATE_LO)
    + 1.0 / (C.RENDER_N - (C.TEMPLATE_HI - C.TEMPLATE_LO))) / C.RENDER_GAIN
worst_bar = min(C.SIGMA_STAR[t] for t in C.TASKS)
ok(sig_ideal < 0.60 * worst_bar,
   f"ideal-observer σ_eff {sig_ideal:.3f} ≤ 60% of the tightest bar "
   f"{worst_bar:.3f} (F73 headroom)")

# ── 2. robot end-to-end session + persistence roundtrip ──
CLOCK = {"now": 1_800_000_000.0}


def now_fn():
    CLOCK["now"] += 30.0
    return CLOCK["now"]


class MiniRobot:
    def __init__(self, seed=5):
        p = LearnerParams(alpha_t=0.25, alpha_sigma=0.10,
                          sigma_inf=0.55, q_t=0.03, q_sigma=0.015, rho=0.6)
        self.subs = {t: Learner([1.1], [0.4], p, seed=seed + t)
                     for t in C.TASKS}

    def responder(self, meta):
        y = self.subs[meta["task"]].respond(meta["s_real"], 0)
        # 900 ms: an engaged-plausible RT — the M23 F82 RT-floor gate
        # (500 ms / 3 consecutive) treats sustained faster runs as
        # non-perceptual responding and ends the session
        return int(y), 900.0, False


robot = MiniRobot()
s1 = Session(now_fn=now_fn, seed=11)
n1, ev1 = s1.run(robot.responder, max_trials=20)
ok(n1 == 20 and os.path.exists(sio.STATE_NPZ),
   f"session 1 served {n1} trials and persisted state")
with open(C.TRIALS_JSONL) as fh:
    logs = [json.loads(x) for x in fh]
ok(len(logs) == 20 and all("belief" in r for r in logs),
   "trial telemetry logged with belief snapshots")

st = sio.load_state()
ok(st is not None and np.array_equal(
       st[0][C.TASKS[0]].strata[0].theta,
       s1.filters[C.TASKS[0]].strata[0].theta),
   "state roundtrip: stratum clouds bit-identical after reload")

# ── 3. resume + gap anchoring on the wall clock ──
CLOCK["now"] += 86_400.0                      # a day passes
s2 = Session(now_fn=now_fn, seed=12)
n2, ev2 = s2.run(robot.responder, max_trials=20)
ok(any(e["event"] == "anchor" for e in ev2),
   f"gap > MIN_GAP triggers GapAnchor (events {ev2})")
with open(C.TRIALS_JSONL) as fh:
    logs = [json.loads(x) for x in fh]
ok(logs[-1]["global_trial"] == 39 and logs[-1]["session"] == 2,
   "global trial counter continues across sessions")
segs = [r["seg_id"] for r in logs]
ok(len(segs) == len(set(segs)), "serve-once: no segment repeated, ever")

# ── 4. D33 lifecycle: provisional → (gap) → confirm/revoke executes ──
# manufacture near-mastery beliefs so a declaration fires this session
st = sio.load_state()
filters, gates, anchors, meta = st
t0 = C.TASKS[0]
rng = np.random.default_rng(0)
for s in filters[t0].strata:
    s.ell = C.ELL_STAR[t0] + 0.45 + 0.05 * rng.standard_normal(s.N)
    s.theta = 0.0 + 0.05 * rng.standard_normal(s.N)
    s.w = np.full(s.N, 1.0 / s.N)
filters[t0].log_w[:] = 0.0
from training.trainer_policy import EProcessGate
gates[t0] = EProcessGate(alpha=0.05)     # fresh gate (the persisted one is
                                         # honestly elevated from the
                                         # robot's early below-bar probes)
sio.save_state(filters, gates, anchors, meta)
CLOCK["now"] += 600.0                         # same sitting (no gap)
s3 = Session(now_fn=now_fn, seed=13)
n3, ev3 = s3.run(robot.responder, max_trials=15)
ok(any(e["event"] == "provisional" for e in ev3),
   f"near-mastery belief triggers PROVISIONAL declaration (events {ev3})")
CLOCK["now"] += 86_400.0                      # gap → confirmation pass
s4 = Session(now_fn=now_fn, seed=14)
n4, ev4 = s4.run(robot.responder, max_trials=12)
ok(any(e["event"] in ("confirmed", "revoked") for e in ev4),
   f"post-gap CONFIRMATION rule executed (events {ev4})")

# ── 4b. M21 (F72): shadow consistency telemetry + persistence ──
with open(C.TRIALS_JSONL) as fh:
    logs = [json.loads(x) for x in fh]
ok(all("consistency" in r and "glr" in r["consistency"] for r in logs),
   "every trial carries the shadow consistency statistic")
st = sio.load_state()
ok("consistency" in st[3] and all(str(t) in st[3]["consistency"]
                                  for t in C.TASKS),
   "consistency monitor state persists in meta")
from training.consistency import ConsistencyMonitor

m0 = ConsistencyMonitor.from_json(st[3]["consistency"][str(C.TASKS[0])])
ok(len(m0.buf) > 0, "monitor window buffer survives the json roundtrip")

# ── 4c. M21 (F72): per-tester profiles namespace state and logs ──
_orig = (C.STATE_DIR, C.LOG_DIR, C.FIG_DIR, C.TRIALS_JSONL,
         C.SESSIONS_JSONL, sio.STATE_NPZ, sio.STATE_JSON, C.ROOT)
C.ROOT = TMP
C.set_user("alice")
ok(C.STATE_DIR.endswith("state-alice") and "logs-alice" in C.TRIALS_JSONL
   and sio.STATE_NPZ == os.path.join(C.STATE_DIR, "state.npz"),
   "set_user namespaces state, logs and state_io paths")
ok(sio.load_state() is None,
   "a fresh profile starts with no state (isolation from other testers)")
(C.STATE_DIR, C.LOG_DIR, C.FIG_DIR, C.TRIALS_JSONL,
 C.SESSIONS_JSONL, sio.STATE_NPZ, sio.STATE_JSON, C.ROOT) = _orig
C.USER = None

# ── 4d. M21 (F73): resume across a render recalibration is blocked ──
filters, gates, anchors, meta = sio.load_state()
meta_bad = dict(meta)
meta_bad["render"] = [0.55, 1.0, 48]        # the M20 constants
sio.save_state(filters, gates, anchors, meta_bad)
try:
    sio.load_state()
    ok(False, "render-param mismatch must refuse to resume")
except RuntimeError as e:
    ok("render" in str(e), f"render-param mismatch blocks resume: {e}")
sio.save_state(filters, gates, anchors, meta)     # restore

# ── 4e. M22 Gate 2: a fired consistency flag pauses the task in-session ──
CLOCK["now"] += 600.0
s5 = Session(now_fn=now_fn, seed=15)
for t in C.TASKS:
    s5.monitors[t].threshold = 0.05        # hair trigger — mechanism test
n5, ev5 = s5.run(robot.responder, max_trials=14)
pauses = [e for e in ev5 if e["event"] == "consistency_pause"]
ok(len(pauses) >= 1, f"flag → consistency_pause event (events {ev5})")
with open(C.TRIALS_JSONL) as fh:
    logs = [json.loads(x) for x in fh]
s5_logs = [r for r in logs if r["session"] == s5.session_no]
for p in pauses:
    served_after = [r for r in s5_logs if r["task"] == p["task"]
                    and r["global_trial"] > p["global_trial"]]
    ok(not served_after,
       f"task {p['task']} not served again after its pause")
st = sio.load_state()
m_reset = st[3]["consistency"][str(pauses[0]["task"])]
ok(len(m_reset.get("buf", [])) == 0 and not m_reset.get("flagged"),
   "paused task's monitor starts the next session with a fresh window")

# ── 4f. M22 Gate 3: fatigue guard ends the session (wiring test) ──
CLOCK["now"] += 600.0
_orig_delta = C.FATIGUE_DELTA
C.FATIGUE_DELTA = -0.5                     # any differenced deficit fires
s6 = Session(now_fn=now_fn, seed=16)
n6, ev6 = s6.run(robot.responder, max_trials=32)
C.FATIGUE_DELTA = _orig_delta
ok(any(e["event"] == "fatigue_break" for e in ev6),
   f"fatigue guard fires and ends the session (served {n6}, events {ev6})")
ok(n6 <= 2 * C.FATIGUE_W + 1,
   f"session ended at the first eligible fatigue check ({n6} trials)")

# ── 5. report builds from the logs ──
import sandbox.report as rep

rep.main()
ok(os.path.exists(os.path.join(C.FIG_DIR, "trajectory.png")),
   "analytics report + trajectory figure built")

shutil.rmtree(TMP)
print(f"\nSANDBOX SMOKE TEST PASSED — {N_CHECK} checks.")
