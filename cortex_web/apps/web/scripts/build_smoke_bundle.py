"""Build a deterministic, non-clinical bundle for browser-flow tests.

The fixture exercises EEG-only spike rendering, EEG + spectrogram IIIC
rendering, and domain transitions without requiring the private production
bank in CI. It is never an engine-accuracy fixture; numerical qualification
continues to use the frozen trajectory and real-bank harnesses.
"""
from __future__ import annotations

import json
from pathlib import Path


TASK_CODES = ["spike", "sz", "lpd", "gpd", "lrda", "grda", "iic"]
TASK_LABELS = ["Spike", "Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
TASK_WORDS = ["spike", "seizure", "lpd", "gpd", "lrda", "grda", "other"]
TASK_CLASSES = ["spike", "iiic", "iiic", "iiic", "iiic", "iiic", "iiic"]
CHANNELS = [
    "Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz",
    "Pz", "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2", "EKG",
]


def _signals(task_k: int, level: int) -> tuple[list[float], list[float]]:
    means = [0.0] * len(TASK_CODES)
    sds = [0.0] * len(TASK_CODES)
    value = (-1.25, 0.0, 1.25)[level % 3]
    if task_k == 0:
        means[0] = value
        sds[0] = 0.25
    else:
        for k in range(1, len(TASK_CODES)):
            means[k] = value if k == task_k else -0.2 * value
            sds[k] = 0.25
    return means, sds


def build_bundle(output: Path) -> Path:
    segment_dir = output / "seg"
    segment_dir.mkdir(parents=True, exist_ok=True)
    segments: list[dict] = []

    for index in range(6):
        seg_id = 900_000 + index
        means, sds = _signals(0, index)
        eeg_name = f"seg/{seg_id}.eeg"
        (output / eeg_name).write_bytes(bytes(20 * 1281 * 2))
        segments.append({
            "segId": seg_id,
            "testClass": "spike",
            "patternClass": "spike",
            "sMean": means,
            "sSd": sds,
            "applicableTaskIdx": [0],
            "fsHz": 128.0,
            "nCh": 20,
            "nSamp": 1281,
            "channelNames": CHANNELS,
            "specShape": None,
            "specTime": None,
            "specFreq": None,
            "eeg": eeg_name,
            "spec": "",
        })

    for index in range(14):
        task_k = 1 + index % 6
        seg_id = 910_000 + index
        means, sds = _signals(task_k, index)
        eeg_name = f"seg/{seg_id}.eeg"
        spec_name = f"seg/{seg_id}.spec"
        (output / eeg_name).write_bytes(bytes(20 * 6000 * 2))
        # A deterministic gradient gives the canvas non-uniform pixels while
        # remaining explicitly synthetic and tiny.
        (output / spec_name).write_bytes(bytes(i % 256 for i in range(64 * 96)))
        segments.append({
            "segId": seg_id,
            "testClass": "iiic",
            "patternClass": TASK_WORDS[task_k],
            "sMean": means,
            "sSd": sds,
            "applicableTaskIdx": list(range(1, 7)),
            "fsHz": 200.0,
            "nCh": 20,
            "nSamp": 6000,
            "channelNames": CHANNELS,
            "specShape": [64, 96],
            "specTime": [0.0, 30.0],
            "specFreq": [0.5, 25.0],
            "eeg": eeg_name,
            "spec": spec_name,
        })

    identity = [
        [float(i == j) for j in range(len(TASK_CODES))]
        for i in range(len(TASK_CODES))
    ]
    manifest = {
        "version": "_smoke",
        "eegScale": 4.0,
        "specDbRange": [-10.0, 25.0],
        "taskCodes": TASK_CODES,
        "taskLabels": TASK_LABELS,
        "taskPatternWords": TASK_WORDS,
        "taskClasses": TASK_CLASSES,
        "certBlock": "synthetic-browser-smoke",
        "ellStar": [0.0] * len(TASK_CODES),
        "corrL": identity,
        "perDomainCap": 60,
        "nSegments": len(segments),
        "segments": segments,
        "engineProfile": "synthetic-browser-smoke",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, separators=(",", ":")))
    return manifest_path


if __name__ == "__main__":
    build_bundle(Path(__file__).parents[1] / "public" / "bundle" / "_smoke")
