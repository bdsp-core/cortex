"""V4 batch shadow: the vendored learning engine run READ-ONLY over the web
DB's real per-trial streams (adoption roadmap V4; learning handoff contract
v1.1). No serving behavior changes; the output is analysis material:

  - seeding: each participant's latest finalized test sitting is replayed
    through the engine's own observation model (contract §2a — the raw
    stream, full n-way picks), on the vendored population artifact
    (`nway_dynamics_v1_1.json`, floor_joint included);
  - tracking: the participant's recorded TRAINING responses (Phase L2
    fields in `training_trials`) are stepped through the engine belief
    with feedback, in shown order;
  - readouts per participant: per-domain readiness mass P(ell > ell*)
    at seed time and after training, and the realized difficulty-gate
    weight of every served training item under the shadow belief (the
    placement-precision statistic).

Mapping assumptions (assert-guarded; confirm on first real run):
  - manifest taskCodes[0] is the binary detection task; taskCodes[1:] are
    the n-way identification family, in the SAME order as the artifact's
    domain codes (mapped by position — the artifact uses generic codes).
  - exam trials on n-way tasks store `pick` as the 0-based index into the
    n-way family (task index - 1); binary-task trials store pick in {0,1}.
  - training trials are binary one-vs-rest: `pick` = yes/no in {0,1},
    `y_star` = the item's one-vs-rest label.

Usage:
  python -m research.shadow.le_shadow --db services/api/cortex.db \
      --manifest apps/web/public/bundle/v1.6-k7-35k/manifest.json \
      --out shadow_report.jsonl
Run offline/batch only — never in the serving path.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CW = os.path.abspath(os.path.join(_HERE, "..", ".."))     # cortex_web/
_PKG = os.path.join(_CW, "learning-engine-cleaned")
if not os.path.isdir(_PKG):                                # pre-integration
    _PKG = os.path.join(os.path.dirname(_CW), "learning-engine-cleaned")
sys.path.insert(0, _PKG)

from learning_engine import MixedBelief, Registry  # noqa: E402

DEFAULT_ARTIFACT = os.path.join(_PKG, "artifacts",
                                "nway_dynamics_v1_1.json")
N_CASE = 24        # case-mix items per n-way domain for the reduction map
N_PARTICLES = 400


def _connect_ro(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def build_engine_inputs(manifest: dict, artifact: dict):
    """Registry + position-remapped artifact + per-domain case mixes."""
    codes = list(manifest["taskCodes"])
    nway = codes[1:]
    assert len(artifact["codes"]) == len(nway), (
        "artifact n-way domain count != manifest n-way task count")
    art = dict(artifact, codes=nway)   # position mapping (generic -> task codes)
    reg = Registry([(codes[0], "binary")] + [(c, "g") for c in nway])
    segmap = {int(s["segId"]): s for s in manifest["segments"]}
    words = list(manifest["taskPatternWords"])
    case_mix: dict[str, list] = {c: [] for c in nway}
    for s in manifest["segments"]:
        cls = s.get("patternClass")
        if cls in words[1:]:
            k = words.index(cls)          # true task index 1..K-1
            dom = nway[k - 1]
            if len(case_mix[dom]) < N_CASE:
                case_mix[dom].append(dict(
                    group="g", s=[float(x) for x in s["sMean"][1:]],
                    gold=k - 1))
    return reg, art, segmap, case_mix


def stream_from_trials(rows) -> list[dict]:
    """Contiguous fully-picked prefix, served order (contract §2a)."""
    out: list[dict] = []
    for r in rows:
        if r["pick"] is None or r["trial_index"] != len(out):
            break
        out.append(dict(trialIndex=r["trial_index"], segId=r["seg_id"],
                        taskK=r["task_k"], pick=r["pick"]))
    return out


def _item_for(reg: Registry, seg: dict, task_k: int):
    codes = reg.codes
    if task_k == 0:
        return dict(domain=codes[0], s=float(seg["sMean"][0]))
    return dict(group="g", s=[float(x) for x in seg["sMean"][1:]])


def replay_seed(bel: MixedBelief, stream, segmap, reg: Registry) -> int:
    """Pick coding (MEASURED on real web data, 2026-07-17): `pick` is a
    0-based index on the TASK axis — n-way answers land in 1..K-1, and the
    binary task uses 0 = "yes, it is" with the sentinel K = "no". So the
    binary response is (pick == 0) and the n-way group index is pick-1."""
    n = 0
    for tr in stream:
        seg = segmap.get(int(tr["segId"]))
        if seg is None:
            continue
        k, p = int(tr["taskK"]), int(tr["pick"])
        if k == 0:
            bel.observe(_item_for(reg, seg, 0), 1 if p == 0 else 0)
        else:
            g = p - 1
            if not (0 <= g < len(reg.codes) - 1):
                continue        # foreign coding — skip rather than corrupt
            bel.observe(_item_for(reg, seg, k), g)
        n += 1
    return n


def readiness(bel: MixedBelief, art, case_mix, ell_star) -> dict:
    out = {}
    for i, c in enumerate(art["codes"]):
        out[c] = float(bel.pass_mass(c, float(ell_star[i + 1]),
                                     case_mix[c]))
    return out


def shadow_participant(conn, code: str, reg, art, segmap, case_mix,
                       ell_star) -> dict | None:
    sess = conn.execute(
        "SELECT session_id FROM sessions WHERE code=? AND status!='in_progress' "
        "ORDER BY started_utc DESC LIMIT 1", (code,)).fetchone()
    if sess is None:
        return None
    trials = conn.execute(
        "SELECT * FROM trials WHERE session_id=? ORDER BY trial_index",
        (sess["session_id"],)).fetchall()
    stream = stream_from_trials(trials)
    if not stream:
        return None
    bel = MixedBelief(art, reg, N=N_PARTICLES,
                      rng=np.random.default_rng(0))   # deterministic shadow
    n_seed = replay_seed(bel, stream, segmap, reg)
    pm0 = readiness(bel, art, case_mix, ell_star)

    train = conn.execute(
        "SELECT tt.* FROM training_trials tt JOIN training_sessions ts "
        "ON tt.training_id = ts.training_id WHERE tt.code=? AND "
        "tt.pick IS NOT NULL ORDER BY ts.started_utc, "
        "COALESCE(tt.answered_client_utc, tt.shown_utc)", (code,)).fetchall()
    gate_w = []
    n_train = 0
    for r in train:
        seg = segmap.get(int(r["seg_id"]))
        k = int(r["task_k"] or 0)
        if seg is None or k <= 0:
            continue
        dom = art["codes"][k - 1]
        t_bar, u_bar = bel.mean_state()
        j = reg.index[dom]
        s_asked = float(seg["sMean"][k])
        z = abs((s_asked - t_bar[j]) / np.exp(u_bar[j]))
        gate_w.append(float(bel._gate(np.array([z]))[0]))
        item = dict(domain=dom, s=s_asked, gold=int(r["y_star"] or 0))
        bel.update(item, int(r["pick"]))
        n_train += 1
    pm1 = readiness(bel, art, case_mix, ell_star)
    return dict(code=code, session_id=sess["session_id"],
                n_replayed=n_seed, n_train=n_train,
                readiness_at_seed=pm0, readiness_after_training=pm1,
                mean_gate_weight=(float(np.mean(gate_w)) if gate_w else None),
                n_gate_weights=len(gate_w))


def run(db_path: str, manifest_path: str, artifact_path: str,
        out_path: str | None) -> list[dict]:
    manifest = json.load(open(manifest_path))
    artifact = json.load(open(artifact_path))["artifact"]
    reg, art, segmap, case_mix = build_engine_inputs(manifest, artifact)
    ell_star = list(manifest["ellStar"])
    conn = _connect_ro(db_path)
    rows = []
    for p in conn.execute("SELECT code FROM participants ORDER BY code"):
        r = shadow_participant(conn, p["code"], reg, art, segmap,
                               case_mix, ell_star)
        if r is not None:
            rows.append(r)
    conn.close()
    if out_path:
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = run(a.db, a.manifest, a.artifact, a.out)
    print(f"[shadow] {len(rows)} participants shadowed")
    for r in rows[:10]:
        pm1 = {k: round(v, 2) for k, v in
               r["readiness_after_training"].items()}
        print(f"  {r['code']}: replayed {r['n_replayed']}, trained "
              f"{r['n_train']}, gate-w "
              f"{r['mean_gate_weight'] and round(r['mean_gate_weight'], 3)}, "
              f"readiness {pm1}")


if __name__ == "__main__":
    main()
