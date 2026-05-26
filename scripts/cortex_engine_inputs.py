"""CORTEX internal test — IIIC-only engine inputs (Phase A).

Assembles the K=6 IIIC inputs the SMC particle-cloud engine
(engine/core_mcmc.py) needs to drive a live adaptive certification
session, from artifacts already in the repo — no calibration campaign:

  * the 100 IIIC segments in   data/eeg_bank.h5          (the test bank)
  * per-task signals from      data/labels/iiic_segment_signals.csv
  * the K=6 fitted prior from  Sigma_l_fitted.npy

The spike task is deliberately excluded: 0 of the 100 spike bank
segments have a calibrated signal, so no spike segment is
engine-servable (see docs/LIVE_TEST_INTEGRATION_PLAN.md §4). This module
is IIIC-only — exactly the engine's native K=6 methodology configuration.

`build_iiic_engine_inputs()` -> `IIICEngineInputs` is consumed by
`scripts/session_controller.py` (Phase B). Run this file directly for a
standalone validation report:

    .venv/bin/python scripts/cortex_engine_inputs.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd

# engine/core_mcmc.py uses flat sibling imports (`from auroc import ...`),
# so engine/ must be on sys.path before load_fitted_Sigma resolves — the
# same path setup scripts/viz_smc_collapse.py uses.
# Frozen PyInstaller bundles: __file__ for a PYZ-loaded module does not
# resolve to a real filesystem path whose parent.parent is the data unpack
# root. sys._MEIPASS is set by the bootloader to that root; bundled datas
# (engine/, data/, Sigma_l_fitted.npy, cortex_config.yaml) live under it.
if getattr(sys, "frozen", False):
    _REPO = sys._MEIPASS
else:
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE = os.path.join(_REPO, "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)
from core_mcmc import load_fitted_Sigma  # noqa: E402

BANK_PATH = os.path.join(_REPO, "data", "eeg_bank.h5")
SIGNALS_CSV = os.path.join(_REPO, "data", "labels", "iiic_segment_signals.csv")
SIGMA_PATH = os.path.join(_REPO, "Sigma_l_fitted.npy")

# Canonical K=6 IIIC task table — list index == engine task index k.
#   code         — engine domain code; the column suffix in the signals CSV
#                  AND the domain order in Sigma_l_fitted.npy
#   label        — answer-button text (matches BankViewer._IIIC_OPTIONS)
#   pattern_word — the data/eeg_bank.h5 `pattern_class` attr spelling
TASKS = [
    ("sz",   "Seizure", "seizure"),
    ("lpd",  "LPD",     "lpd"),
    ("gpd",  "GPD",     "gpd"),
    ("lrda", "LRDA",    "lrda"),
    ("grda", "GRDA",    "grda"),
    ("iic",  "Other",   "other"),
]
TASK_CODES = [t[0] for t in TASKS]
TASK_LABELS = [t[1] for t in TASKS]
TASK_PATTERN_WORDS = [t[2] for t in TASKS]
K = len(TASKS)


@dataclass
class IIICEngineInputs:
    """Everything the engine + viewer need for an IIIC-only live session."""

    manifest: pd.DataFrame   # indexed by seg_id; cols: pattern_class,
                             # s_mean_<code> + s_sd_<code> for each task
    Corr_l: np.ndarray       # (6, 6) fitted prior correlation, used for
                             # BOTH the l- and t-blocks (see build note)
    task_codes: list
    task_labels: list
    task_pattern_words: list

    @property
    def all_seg_ids(self) -> list:
        return [int(s) for s in self.manifest.index]

    def h5_group(self, seg_id: int) -> str:
        """Path of the segment inside data/eeg_bank.h5."""
        return f"iiic/{int(seg_id)}"

    def pattern_class(self, seg_id: int) -> str:
        """The segment's true IIIC label (the `pattern_class` h5 attr)."""
        return str(self.manifest.at[int(seg_id), "pattern_class"])

    def true_task_index(self, seg_id: int) -> int:
        """Engine task index k whose class is this segment's true label."""
        return self.task_pattern_words.index(self.pattern_class(seg_id))

    def as_engine_arrays(self, seg_ids=None):
        """Build ``(bank_signals, bank_sds, bank_segids)`` for ``choose_item``.

        Each is a list of K arrays (one per task). The same candidate
        segments appear in every task's array — a segment carries a
        distinct ``(s_mean, s_sd)`` per task. Pass ``seg_ids`` to restrict
        the candidate pool: the Phase-B controller passes the
        not-yet-served segments so the engine never serves the same EEG
        twice — ``choose_item`` has no de-duplication of its own.
        """
        if seg_ids is None:
            seg_ids = self.all_seg_ids
        seg_ids = [int(s) for s in seg_ids]
        missing = [s for s in seg_ids if s not in self.manifest.index]
        if missing:
            raise KeyError(f"seg_ids not in manifest: {missing[:5]}")
        rows = self.manifest.loc[seg_ids]
        ids = np.asarray(seg_ids, dtype=int)
        bank_signals = [rows[f"s_mean_{c}"].to_numpy(float)
                        for c in self.task_codes]
        bank_sds = [rows[f"s_sd_{c}"].to_numpy(float)
                    for c in self.task_codes]
        bank_segids = [ids.copy() for _ in self.task_codes]
        return bank_signals, bank_sds, bank_segids

    def without(self, seg_ids):
        """A copy with the given segments removed from the pool — used to
        reserve a tutorial segment so the engine never serves it."""
        drop = {int(s) for s in seg_ids}
        keep = [s for s in self.manifest.index if int(s) not in drop]
        return IIICEngineInputs(
            manifest=self.manifest.loc[keep].copy(), Corr_l=self.Corr_l,
            task_codes=list(self.task_codes),
            task_labels=list(self.task_labels),
            task_pattern_words=list(self.task_pattern_words))


def build_iiic_engine_inputs(verbose: bool = False) -> IIICEngineInputs:
    """Assemble the IIIC engine inputs; raise on any data-integrity defect."""
    # 1. the IIIC bank segments + their true pattern_class label
    with h5py.File(BANK_PATH, "r") as f:
        if "iiic" not in f:
            raise RuntimeError(f"{BANK_PATH} has no 'iiic' group")
        pc = {}
        for sid_str in f["iiic"]:
            try:
                sid = int(sid_str)
            except ValueError:
                continue
            pc[sid] = str(f["iiic"][sid_str].attrs.get("pattern_class", ""))
    bank_ids = sorted(pc)
    if not bank_ids:
        raise RuntimeError("no IIIC segments found in the bank")

    # 2. per-task signals, restricted to the bank segments
    sig_cols = ([f"s_mean_{c}" for c in TASK_CODES]
                + [f"s_sd_{c}" for c in TASK_CODES])
    iss = pd.read_csv(SIGNALS_CSV).set_index("seg_id")
    missing_cols = [c for c in sig_cols if c not in iss.columns]
    if missing_cols:
        raise RuntimeError(f"{SIGNALS_CSV} missing columns: {missing_cols}")
    not_in_signals = [s for s in bank_ids if s not in iss.index]
    if not_in_signals:
        raise RuntimeError(
            f"{len(not_in_signals)} bank IIIC segs have no signal row: "
            f"{not_in_signals[:5]}")
    man = iss.loc[bank_ids, sig_cols].copy()
    man.insert(0, "pattern_class", [pc[s] for s in bank_ids])
    man.index.name = "seg_id"

    # A NaN signal would poison choose_item's expected-loss argmin — drop
    # any such segment. (Verified 0 of 100 today; kept as a defensive gate.)
    bad = man.index[man[sig_cols].isna().any(axis=1)].tolist()
    if bad:
        print(f"  WARN: dropping {len(bad)} segs with NaN signals: {bad[:5]}")
        man = man.drop(index=bad)
    if man.empty:
        raise RuntimeError("no usable IIIC segments after NaN filtering")

    # every true label must resolve to one of the 6 engine tasks
    unknown = sorted(set(man["pattern_class"]) - set(TASK_PATTERN_WORDS))
    if unknown:
        raise RuntimeError(f"pattern_class with no task mapping: {unknown}")

    # 3. the K=6 fitted prior. Sigma_l_fitted.npy's own build note: Sigma_t
    # is contaminated by theta-clipping at 2.0 — "use Sigma_l for both
    # blocks". Corr_l (its correlation form) is what viz_smc_collapse.py
    # passes for BOTH the l- and t-blocks; Phase B follows that precedent.
    S = load_fitted_Sigma(SIGMA_PATH)
    domains = [str(d) for d in np.asarray(S["domains"])]
    if domains != TASK_CODES:
        raise RuntimeError(
            f"Sigma_l_fitted.npy domain order {domains} != {TASK_CODES}")
    Corr_l = np.asarray(S["Corr_l"], dtype=float)
    if Corr_l.shape != (K, K):
        raise RuntimeError(f"Corr_l shape {Corr_l.shape} != ({K}, {K})")

    inp = IIICEngineInputs(
        manifest=man, Corr_l=Corr_l,
        task_codes=list(TASK_CODES), task_labels=list(TASK_LABELS),
        task_pattern_words=list(TASK_PATTERN_WORDS))
    if verbose:
        _report(inp)
    return inp


def _report(inp: IIICEngineInputs) -> None:
    man = inp.manifest
    print("CORTEX IIIC engine inputs — validation report")
    print(f"  bank            : {BANK_PATH}")
    print(f"  signals         : {SIGNALS_CSV}")
    print(f"  segments        : {len(man)} IIIC (engine-servable)")
    print(f"  tasks (K={K})     : {inp.task_codes}")
    dist = Counter(man["pattern_class"])
    print("  true-label mix  : "
          + "  ".join(f"{w}:{dist.get(w, 0)}" for w in inp.task_pattern_words))
    print("  per-task signal ranges:")
    for c in inp.task_codes:
        sm, ss = man[f"s_mean_{c}"], man[f"s_sd_{c}"]
        print(f"    {c:5s}: s_mean [{sm.min():+.2f}, {sm.max():+.2f}]   "
              f"s_sd [{ss.min():.3f}, {ss.max():.3f}]")
    bs, bsd, bseg = inp.as_engine_arrays()
    print(f"  engine arrays   : bank_signals {len(bs)}x{len(bs[0])}   "
          f"bank_sds {len(bsd)}x{len(bsd[0])}   "
          f"bank_segids {len(bseg)}x{len(bseg[0])}")
    print(f"  prior Corr_l    : {inp.Corr_l.shape}  "
          f"diag={np.round(np.diag(inp.Corr_l), 2)}")
    print(f"  max test length : {len(man)} questions "
          f"(each segment is served at most once)")


if __name__ == "__main__":
    build_iiic_engine_inputs(verbose=True)
