"""Shared, mode-agnostic bridge helpers.

F3.2 (2026-05-15): extracted from `run_mode_b_cert_bridge.py` so that
neither the Mode-A bridge nor the Phase-1/2 scripts have to import from a
Mode-B-named module (the coupling smell flagged by the architecture
audit).  These helpers are pure I/O / config / data-shaping utilities
with no Mode-A vs Mode-B semantics.

Canonical home for: load_config, _require, _autodetect_banks_dir,
_default_rater_matrix_path, load_bank_signals, load_raters,
build_true_params, plus the ENGINE_REPO / DEFAULT_CONFIG_PATH constants.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────
# This file lives at <ENGINE_REPO>/bridge/_common.py.  The core engine
# modules live in the parent directory; add it to sys.path so the engine
# imports resolve whether a script is run as `-m bridge.X` or directly.
_BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE_REPO = os.path.dirname(_BRIDGE_DIR)
if ENGINE_REPO not in sys.path:
    sys.path.insert(0, ENGINE_REPO)

DEFAULT_CONFIG_PATH = os.path.join(ENGINE_REPO, "cert_config.yaml")


# ── config loader ─────────────────────────────────────────────────────
def load_config(path: str) -> Dict[str, Any]:
    """Read cert_config.yaml.  Requires PyYAML.

    The config schema is documented in cert_config.yaml itself.  Any value
    not present here is treated as a hard error — the bridge does NOT
    silently fall back to hardcoded defaults (see W1-C / R5).
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to parse cert_config.yaml.  "
            "`pip install pyyaml` in the run environment."
        ) from e
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"cert_config.yaml must parse to a dict; got {type(cfg)}")
    return cfg


def _require(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(
            f"cert_config.yaml is missing required key {key!r}.  "
            f"This bridge does not provide a hardcoded fallback."
        )
    return cfg[key]


# ── data loaders ──────────────────────────────────────────────────────
def _autodetect_banks_dir() -> str:
    """Pick the first existing curated-banks directory among known locations.

    Order (in-repo first → self-contained per D9; sibling fallback retained
    for back-compat with the methodology-repo layout):
      1. this repo     …/ilae-skill-certification-unified/data/curated_banks/
      2. sibling repo  …/ilae-skill-certification-test-main/data/curated_banks/
    """
    project_root = os.path.dirname(ENGINE_REPO)
    candidates = [
        os.path.join(ENGINE_REPO, "data", "curated_banks"),
        os.path.join(project_root, "ilae-skill-certification-test-main",
                     "data", "curated_banks"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "No curated bank directory found.  Searched:\n  "
        + "\n  ".join(candidates)
        + "\nPass --banks-dir explicitly."
    )


def _default_rater_matrix_path() -> str:
    """Default to the in-repo engine inputs (data/engine_inputs/).

    Was the sibling repo's `data/prepared/cross_domain_rater_matrix.csv`;
    now a byte-identical vendored copy lives in this repo (see
    data/engine_inputs/MANIFEST.json — sha256-identical, so no
    behavioural change). Pass `--rater-matrix` to point elsewhere.
    """
    return os.path.join(ENGINE_REPO, "data", "engine_inputs",
                        "cross_domain_rater_matrix.csv")


def load_bank_signals(banks_dir: str, domains: List[str]) -> List[np.ndarray]:
    """Load the per-domain bank signals.  Mirrors the legacy bridge."""
    signals = []
    print(f"Bank signals loaded from {banks_dir}:")
    for d in domains:
        path = os.path.join(banks_dir, f"sparcnet_{d}.json")
        with open(path) as f:
            bank = json.load(f)
        s = np.array(bank["s_probit"], dtype=float)
        signals.append(s)
        print(f"  {d}: {len(s)} items  c_probit "
              f"in [{s.min():.3f}, {s.max():.3f}]")
    return signals


def load_raters(rater_matrix_path: str) -> pd.DataFrame:
    return pd.read_csv(rater_matrix_path)


def build_true_params(row: "pd.Series", domains: List[str]) -> List[float]:
    """Convert a rater-matrix row to the flat (theta_0, l_0, theta_1, l_1, ...) form."""
    params: List[float] = []
    for d in domains:
        theta = float(row[f"theta_{d}"])
        sigma = float(row[f"sigma_{d}"])
        l = float(-np.log(sigma))
        params.extend([theta, l])
    return params
