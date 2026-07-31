#!/usr/bin/env python3
"""Emit the draw-latent Python/TypeScript parity fixture.

Runs the normative draw-latent oracle (draw_latent_rd/engine.py,
construction B) on a fixed injected particle cloud, a fixed atom
assignment, and a fixed observation sequence, and records the
per-step per-particle log likelihoods, normalized weights, cumulative
log likelihoods, and the final atom posterior. The TypeScript engine
must reproduce every value within the parity tolerance used by
tests/python_parity.test.ts.

Deterministic by construction: no RNG is consumed anywhere.

Usage (from n-way-protocol/):
    PYTHONPATH="python:." python3 scripts/make_draw_latent_fixture.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from draw_latent_rd.engine import (
    DrawCloud,
    _observation_log_probability,
    atom_posterior,
    update_draw_latent,
)
from nway_protocol.reference import Observation, ParticleCloud

K = 7
N = 6
GROUP = (1, 2, 3, 4, 5, 6)
ATOMS = ((0.85, 0.15, 0.5), (1.05, 0.0, 0.3), (1.35, 0.05, 0.2))
ATOM_ASSIGNMENT = (0, 1, 2, 0, 1, 2)
INITIAL_WEIGHTS = (0.10, 0.15, 0.20, 0.25, 0.18, 0.12)

SEGMENTS = (
    {
        "segmentIndex": 0,
        "segId": 9101,
        "sMean": [-0.7, -0.3, 0.15, 0.55, -0.1, 0.8, -0.45],
        "sSd": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
        "applicableTaskIdx": [0, 1, 2, 3, 4, 5, 6],
    },
    {
        "segmentIndex": 1,
        "segId": 9102,
        "sMean": [0.4, 0.9, -0.55, -0.2, 0.65, -0.85, 0.3],
        "sSd": [0.08, 0.06, 0.04, 0.09, 0.05, 0.03, 0.07],
        "applicableTaskIdx": [0, 1, 2, 3, 4, 5, 6],
    },
)

# Wrong picks, a correct pick, and a binary spike trial, across both segments.
OBSERVATIONS = (
    {"kind": "categorical_f1", "askedK": 3, "segmentIndex": 0, "rawPick": 5},
    {"kind": "categorical_f1", "askedK": 1, "segmentIndex": 1, "rawPick": 1},
    {"kind": "binary", "askedK": 0, "segmentIndex": 0, "rawPick": 0},
    {"kind": "categorical_f1", "askedK": 5, "segmentIndex": 1, "rawPick": 2},
    {"kind": "categorical_f1", "askedK": 2, "segmentIndex": 0, "rawPick": 6},
)


def fixed_cloud_parameters() -> tuple[list[list[float]], list[list[float]]]:
    """Deterministic (t, l) literals — same shape family as tests/fixtures.ts."""
    t = [
        [(n - 2.5) * 0.18 + (k - 3) * 0.03 for k in range(K)] for n in range(N)
    ]
    l = [
        [(n - 2.5) * 0.22 - (k - 3) * 0.025 for k in range(K)] for n in range(N)
    ]
    return t, l


def main() -> None:
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
        cloud=cloud,
        atoms=ATOMS,
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
        step_log_lik = _observation_log_probability(dcloud, observation, s_mean, s_sd)
        update_draw_latent(dcloud, observation, s_mean, s_sd)
        steps.append({
            "observation": dict(spec),
            "stepLogLik": step_log_lik.tolist(),
            "weights": dcloud.cloud.w.tolist(),
            "logLik": dcloud.cloud.log_lik.tolist(),
        })

    payload = {
        "schemaVersion": 1,
        "K": K,
        "group": list(GROUP),
        "atoms": [list(atom) for atom in ATOMS],
        "atomAssignment": list(ATOM_ASSIGNMENT),
        "t": t,
        "l": l,
        "initialWeights": list(INITIAL_WEIGHTS),
        "segments": list(SEGMENTS),
        "steps": steps,
        "atomPosterior": atom_posterior(dcloud),
    }
    output = Path(__file__).resolve().parents[1] / (
        "tests/fixtures/python_draw_latent_reference.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
