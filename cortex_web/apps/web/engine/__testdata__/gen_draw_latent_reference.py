"""Dump draw-latent ground truth from the REAL Python oracle for the TS port.

Same discipline as gen_reference.py: imports the normative engine —
n-way-protocol/draw_latent_rd/engine.py (construction B) — so the reference
is the actual oracle numerics, not a re-derivation. The atom table is the
PRODUCTION research artifact (iiic_conditional_f1_engine_frame_atoms17_rd),
so the fixture also binds the deployed nway_profile.ts constants to the
oracle. Runs on a fixed injected cloud, a fixed atom assignment, and a fixed
observation sequence (wrong picks, a right pick, a binary spike hit and
miss); no RNG is consumed anywhere.

    python3 cortex_web/apps/web/engine/__testdata__/gen_draw_latent_reference.py

Writes draw_latent_reference.json next to this file;
draw_latent_parity.test.ts loads it and asserts the production engine
agrees within the established n-way parity tolerance (5e-7).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]  # ilae-skill-certification-test-multi/
sys.path.insert(0, str(REPO / "n-way-protocol"))
sys.path.insert(0, str(REPO / "n-way-protocol" / "python"))

from draw_latent_rd.engine import (  # noqa: E402
    DrawCloud, atom_posterior, update_draw_latent,
)
from nway_protocol.reference import Observation, ParticleCloud  # noqa: E402

OUT = Path(__file__).resolve().parent / "draw_latent_reference.json"
ARTIFACT = (
    REPO / "n-way-protocol/artifacts/iiic_conditional_f1_engine_frame_atoms17_rd.json"
)

K = 7
N = 6
GROUP = (1, 2, 3, 4, 5, 6)
# Spread over the 17 atoms so several distinct betas are exercised.
ATOM_ASSIGNMENT = (0, 4, 8, 12, 16, 9)
INITIAL_WEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.18, 0.12)

SEGMENTS = (
    {
        "segId": 9101,
        "sMean": [-0.7, -0.3, 0.15, 0.55, -0.1, 0.8, -0.45],
        "sSd": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    },
    {
        "segId": 9102,
        "sMean": [0.4, 0.9, -0.55, -0.2, 0.65, -0.85, 0.3],
        "sSd": [0.08, 0.06, 0.04, 0.09, 0.05, 0.03, 0.07],
    },
)

# Wrong picks, a correct pick, and binary spike hit + miss, both segments.
OBSERVATIONS = (
    {"kind": "categorical_f1", "askedK": 3, "segmentIndex": 0, "rawPick": 5},
    {"kind": "categorical_f1", "askedK": 1, "segmentIndex": 1, "rawPick": 1},
    {"kind": "binary", "askedK": 0, "segmentIndex": 0, "rawPick": 0},
    {"kind": "binary", "askedK": 0, "segmentIndex": 1, "rawPick": K},
    {"kind": "categorical_f1", "askedK": 2, "segmentIndex": 0, "rawPick": 6},
    {"kind": "categorical_f1", "askedK": 5, "segmentIndex": 1, "rawPick": 2},
)


def production_atoms() -> tuple[tuple[float, float, float], ...]:
    payload = json.loads(ARTIFACT.read_text())
    assert payload["variant"] == "engine_frame_hierarchical_atoms17"
    return tuple(
        (float(d["beta"]), float(d["distractor_lapse"]), float(d["weight"]))
        for d in payload["bootstrap"]["draws"]
    )


def fixed_cloud_parameters() -> tuple[list[list[float]], list[list[float]]]:
    t = [[(n - 2.5) * 0.18 + (k - 3) * 0.03 for k in range(K)] for n in range(N)]
    l = [[(n - 2.5) * 0.22 - (k - 3) * 0.025 for k in range(K)] for n in range(N)]
    return t, l


def main() -> None:
    atoms = production_atoms()
    t, l = fixed_cloud_parameters()
    cloud = ParticleCloud(
        t=np.asarray(t, dtype=np.float64),
        l=np.asarray(l, dtype=np.float64),
        w=np.asarray(INITIAL_WEIGHTS, dtype=np.float64),
        log_prior=np.zeros(N),
        log_lik=np.zeros(N),
        corr_t=np.eye(K),
        corr_l=np.eye(K),
    )
    dcloud = DrawCloud(
        cloud=cloud, atoms=atoms,
        draw=np.asarray(ATOM_ASSIGNMENT, dtype=np.int64),
    )

    steps = []
    for spec in OBSERVATIONS:
        segment = SEGMENTS[spec["segmentIndex"]]
        s_mean = np.asarray(segment["sMean"], dtype=np.float64)
        s_sd = np.asarray(segment["sSd"], dtype=np.float64)
        if spec["kind"] == "binary":
            observation = Observation(
                kind="binary", asked_k=spec["askedK"],
                segment_index=spec["segmentIndex"], raw_pick=spec["rawPick"],
                y=int(spec["rawPick"] == spec["askedK"]),
            )
        else:
            observation = Observation(
                kind="categorical_f1", asked_k=spec["askedK"],
                segment_index=spec["segmentIndex"], raw_pick=spec["rawPick"],
                group=GROUP,
            )
        update_draw_latent(dcloud, observation, s_mean, s_sd)
        steps.append({
            "observation": dict(spec),
            "weights": dcloud.cloud.w.tolist(),
            "logLik": dcloud.cloud.log_lik.tolist(),
        })

    payload = {
        "schemaVersion": 1,
        "K": K,
        "group": list(GROUP),
        "atoms": [list(atom) for atom in atoms],
        "atomAssignment": list(ATOM_ASSIGNMENT),
        "t": t,
        "l": l,
        "initialWeights": list(INITIAL_WEIGHTS),
        "segments": [dict(segment) for segment in SEGMENTS],
        "steps": steps,
        "atomPosterior": atom_posterior(dcloud),
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
