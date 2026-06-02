"""Freeze the v1.3.6 paper-grade CORTEX instrument (NEJM AI Paper 2).

Hash-locks the data artifacts that define a session (question bank, cut-score
calibration, engine prior) and records the param config + git provenance, so the
entire cohort is collected on a BIT-IDENTICAL instrument. `--verify` (or
check_drift()) re-checks the live files + params against the frozen manifest —
the drift-guard for the enrollment window.

    .venv/bin/python scripts/freeze_instrument_v1_3_6.py            # write freeze
    .venv/bin/python scripts/freeze_instrument_v1_3_6.py --verify   # drift check
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "scripts", REPO / "engine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# The data artifacts that define a session's questions + scoring + prior.
INSTRUMENT_FILES = {
    "bank": "data/eeg_bank.h5",
    "cert_config": "calibration/cert_config.yaml",
    "prior": "Sigma_l_fitted_k7.npy",
}
FREEZE_PATH = REPO / "data" / "INSTRUMENT_FREEZE_v1_3_6.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _params() -> dict:
    """The live engine config — read from the actual code constants so the
    manifest reflects what the app will run."""
    import cortex_policy as cp
    import session_controller as sc
    return {
        "alpha": cp.DEFAULT_ALPHA, "n_min": cp.DEFAULT_N_MIN,
        "r_star": cp.DEFAULT_R_STAR, "z": cp.DEFAULT_Z,
        "max_questions": sc.MAX_QUESTIONS_DEFAULT,
        "n_particles": sc.N_PARTICLES,
        "max_consecutive_same_domain": sc.MAX_CONSEC_SAME_DOMAIN_DEFAULT,
        "ell_star_block": "ell_star_unified_v14",
    }


def _bank_composition() -> dict:
    with h5py.File(REPO / INSTRUMENT_FILES["bank"], "r") as f:
        n_spike = len(f["spike"]) if "spike" in f else 0
        by_class = collections.Counter(
            str(f["iiic"][k].attrs.get("pattern_class", "")) for k in f["iiic"]
        ) if "iiic" in f else collections.Counter()
    return {"n_spike": int(n_spike), "n_iiic": int(sum(by_class.values())),
            "iiic_by_class": {k: int(v) for k, v in sorted(by_class.items())}}


def _git() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()
        dirty = len(subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain"]).decode().splitlines())
        return {"sha": sha, "dirty_files": dirty}
    except Exception:
        return {"sha": None, "dirty_files": None}


def build(frozen_utc: str) -> dict:
    files = {}
    for name, rel in INSTRUMENT_FILES.items():
        p = REPO / rel
        files[name] = {"path": rel, "sha256": _sha256(p), "bytes": p.stat().st_size}
    return {
        "instrument": "cortex-v1.3.6 paper-grade (NEJM AI Paper 2)",
        "frozen_utc": frozen_utc,
        "git": _git(),
        "params": _params(),
        "bank": _bank_composition(),
        "files": files,
    }


def check_drift() -> list[str]:
    """Return a list of drift problems (empty == instrument matches the freeze)."""
    if not FREEZE_PATH.exists():
        return [f"no freeze manifest at {FREEZE_PATH.name}"]
    m = json.loads(FREEZE_PATH.read_text())
    problems = []
    for name, rec in m["files"].items():
        live = _sha256(REPO / rec["path"])
        if live != rec["sha256"]:
            problems.append(f"{name} ({rec['path']}): live {live[:12]} != "
                            f"frozen {rec['sha256'][:12]}")
    live_p = _params()
    for k, v in m["params"].items():
        if live_p.get(k) != v:
            problems.append(f"param {k}: live {live_p.get(k)!r} != frozen {v!r}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--utc", default=None, help="freeze timestamp (ISO)")
    args = ap.parse_args()
    if args.verify:
        problems = check_drift()
        if problems:
            print("INSTRUMENT DRIFT DETECTED:")
            for p in problems:
                print("  " + p)
            sys.exit(1)
        print("instrument matches the freeze (all files + params OK)")
        return
    now = args.utc or datetime.datetime.now(datetime.timezone.utc).isoformat()
    m = build(now)
    FREEZE_PATH.write_text(json.dumps(m, indent=2) + "\n")
    print(f"wrote {FREEZE_PATH}")
    print(json.dumps({"params": m["params"], "bank": m["bank"], "git": m["git"],
                      "files": {k: v["sha256"][:12] for k, v in m["files"].items()}},
                     indent=2))


if __name__ == "__main__":
    main()
