"""CORTEX K=7 engine inputs — Phase 9 Layer 6a sibling of `cortex_engine_inputs.py`.

K=7 extension of the K=6 IIIC predecessor. Assembles the K=7 inputs the SMC
particle-cloud engine (`engine/core_mcmc.py`) needs to drive a live adaptive
certification session over spike + 6 IIIC tasks, from artifacts that are now
in the repo at K=7:

  * IIIC bank segments + spike bank segments in `data/eeg_bank.h5`
    (Layer 6a build produces the K=7 bank: 300 IIIC + 50 calibrated spike)
  * Per-(task, seg) signals from `data/labels/segment_signals.csv`
    (produced by Layer 2 — assemble_outputs_k7.py)
  * The K=7 fitted prior from `Sigma_l_fitted_k7.npy`
    (produced by Layer 5 — build_sigma_l_k7.py)

Key K=7-specific design (vs K=6 cortex_engine_inputs.py):

  * **Per-segment task mask**: spike segments serve task=spike only; IIIC
    segments serve the 6 IIIC tasks only (strict segment-type → task mapping
    per Eli's earlier choice Q4 — cross-bank task probing not supported).
  * **Per-task banks**: `as_engine_arrays()` returns a list of K=7 arrays,
    each potentially DIFFERENT length (spike: ~50 segs; IIIC: ~300 segs).
    Engine's `choose_item` iterates per-task and picks optimal (k, seg) pair.
  * **Spike task entry first**: TASKS[0] = spike for consistency with
    cert_config v14 / deployment Sigma slot order.

The K=6 `cortex_engine_inputs.py` stays untouched; CORTEX v1.1.5 (K=6 IIIC)
continues to consume it. K=7 promotion happens at Layer 6a ship time.
"""
from __future__ import annotations

import logging
import os
import sys
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mirror cortex_engine_inputs.py's path resolution (PyInstaller-aware)
if getattr(sys, "frozen", False):
    _REPO = sys._MEIPASS
else:
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENGINE = os.path.join(_REPO, "engine")
if _ENGINE not in sys.path:
    sys.path.insert(0, _ENGINE)
from core_mcmc import load_fitted_Sigma  # noqa: E402

BANK_PATH = os.path.join(_REPO, "data", "eeg_bank.h5")
SIGNALS_CSV = os.path.join(_REPO, "data", "labels", "segment_signals.csv")
SIGMA_PATH = os.path.join(_REPO, "Sigma_l_fitted_k7.npy")
SERVED_MANIFEST_PATH = os.path.join(
    _REPO, "cortex_web", "apps", "web", "public", "bundle",
    "v1.6-k7-35k", "manifest.json")
SERVED_MANIFEST_SHA256 = (
    "c3ce43b5639bb1a5e15b2c014246c5f4e6916bcb3284d10aaebf1d253e8811fa"
)

# Canonical K=7 task table (engine task index k == list index).
#   code           — engine domain code; the column suffix in segment_signals.csv
#                    AND the domain order in Sigma_l_fitted_k7.npy
#   label          — answer-button text (IIIC) or Yes/No-question label (spike)
#   pattern_word   — the `pattern_class` h5 attr spelling for IIIC; the
#                    `test_class` H5 group name for spike
#   test_family    — "iiic" or "spike" (determines UI dispatch + bank H5 group)
TASKS = [
    ("spike", "Spike",   "spike",   "spike"),
    ("sz",    "Seizure", "seizure", "iiic"),
    ("lpd",   "LPD",     "lpd",     "iiic"),
    ("gpd",   "GPD",     "gpd",     "iiic"),
    ("lrda",  "LRDA",    "lrda",    "iiic"),
    ("grda",  "GRDA",    "grda",    "iiic"),
    ("iic",   "Other",   "other",   "iiic"),
]
TASK_CODES = [t[0] for t in TASKS]
TASK_LABELS = [t[1] for t in TASKS]
TASK_PATTERN_WORDS = [t[2] for t in TASKS]
TASK_FAMILIES = [t[3] for t in TASKS]
K = len(TASKS)
assert K == 7, "K=7 invariant"

# Task family → indices into TASKS
SPIKE_TASK_INDEX = 0
IIIC_TASK_INDICES = list(range(1, 7))  # sz, lpd, gpd, lrda, grda, iic


@dataclass
class K7EngineInputs:
    """Everything the engine + viewer need for a K=7 live session.

    The manifest holds BOTH IIIC and spike segments, distinguished by
    `family` column ('iiic' or 'spike'). Per-task signals are in the
    s_mean_{code} / s_sd_{code} columns; NaN where the segment isn't a
    valid candidate for that task (cross-family).
    """
    manifest: pd.DataFrame   # indexed by seg_id; cols: family, pattern_class,
                             # s_mean_<code> + s_sd_<code> for each of 7 tasks
    Corr_l: np.ndarray       # (7, 7) fitted prior correlation for ell-block
    Corr_t: np.ndarray       # (7, 7) fitted prior correlation for t-block
    task_codes: list
    task_labels: list
    task_pattern_words: list
    task_families: list
    # Deterministic coarse-to-fine selection assumes candidate order is smooth
    # in signal. The served-manifest calibration profile enables per-domain
    # stable signal ordering; the reference fixture preserves historical order.
    sort_candidates_by_signal: bool = False
    source_manifest_path: str | None = None
    source_manifest_sha256: str | None = None
    _engine_cache: tuple | None = field(default=None, init=False, repr=False)

    @property
    def all_seg_ids(self) -> list:
        return [int(s) for s in self.manifest.index]

    def family(self, seg_id: int) -> str:
        """'iiic' or 'spike' — which test-family the segment belongs to."""
        return str(self.manifest.at[int(seg_id), "family"])

    def h5_group(self, seg_id: int) -> str:
        """Path of the segment inside data/eeg_bank.h5."""
        fam = self.family(seg_id)
        return f"{fam}/{int(seg_id)}"

    def pattern_class(self, seg_id: int) -> str:
        """The segment's true label (the `pattern_class` H5 attr)."""
        return str(self.manifest.at[int(seg_id), "pattern_class"])

    def true_task_index(self, seg_id: int) -> int:
        """Engine task index k whose class is this segment's true label.
        IIIC seg: returns 1..6 by pattern_class.
        Spike seg: returns 0 (spike task).
        """
        fam = self.family(seg_id)
        if fam == "spike":
            return SPIKE_TASK_INDEX
        return self.task_pattern_words.index(self.pattern_class(seg_id))

    def valid_task_indices(self, seg_id: int) -> list:
        """Engine task indices for which this segment is a valid candidate.
        Spike seg: [0] (spike only).
        IIIC seg:  [1, 2, 3, 4, 5, 6] (all 6 IIIC tasks).
        """
        fam = self.family(seg_id)
        if fam == "spike":
            return [SPIKE_TASK_INDEX]
        return list(IIIC_TASK_INDICES)

    def as_engine_arrays(self, seg_ids=None):
        """Build per-task `(bank_signals, bank_sds, bank_segids)` lists.

        Each is a list of K=7 arrays; arrays for different tasks may have
        DIFFERENT lengths (spike task: only spike segs; IIIC tasks: only IIIC
        segs). Engine's `choose_item` picks optimal (k, j) where j indexes
        into the per-task array.

        `seg_ids` (optional): restrict the candidate pool (e.g. exclude
        already-served segments).
        """
        if self.sort_candidates_by_signal:
            if self._engine_cache is None:
                signals, sds, ids = [], [], []
                for code in self.task_codes:
                    sm_col = f"s_mean_{code}"
                    sd_col = f"s_sd_{code}"
                    valid = self.manifest[sm_col].notna() & self.manifest[sd_col].notna()
                    values = self.manifest.loc[valid].sort_values(
                        sm_col, kind="mergesort")
                    signals.append(values[sm_col].to_numpy(float))
                    sds.append(values[sd_col].to_numpy(float))
                    ids.append(values.index.to_numpy(int))
                self._engine_cache = (signals, sds, ids)
            cached_signals, cached_sds, cached_ids = self._engine_cache
            if seg_ids is None:
                return (
                    [row.copy() for row in cached_signals],
                    [row.copy() for row in cached_sds],
                    [row.copy() for row in cached_ids],
                )
            allowed = np.asarray([int(s) for s in seg_ids], dtype=int)
            output_signals, output_sds, output_ids = [], [], []
            for values, uncertainty, ids in zip(
                    cached_signals, cached_sds, cached_ids):
                keep = np.isin(ids, allowed, assume_unique=True)
                output_signals.append(values[keep])
                output_sds.append(uncertainty[keep])
                output_ids.append(ids[keep])
            return output_signals, output_sds, output_ids

        if seg_ids is None:
            seg_ids = self.all_seg_ids
        seg_ids = [int(s) for s in seg_ids]
        missing = [s for s in seg_ids if s not in self.manifest.index]
        if missing:
            raise KeyError(f"seg_ids not in manifest: {missing[:5]}")
        rows = self.manifest.loc[seg_ids]

        bank_signals = []
        bank_sds = []
        bank_segids = []
        for code in self.task_codes:
            sm_col = f"s_mean_{code}"
            sd_col = f"s_sd_{code}"
            # Per-task: keep only segs with a valid (non-NaN) signal for this task
            valid_mask = rows[sm_col].notna() & rows[sd_col].notna()
            v = rows[valid_mask]
            if self.sort_candidates_by_signal:
                v = v.sort_values(sm_col, kind="mergesort")
            bank_signals.append(v[sm_col].to_numpy(float))
            bank_sds.append(v[sd_col].to_numpy(float))
            bank_segids.append(v.index.to_numpy(int))
        return bank_signals, bank_sds, bank_segids

    def without(self, seg_ids):
        """A copy with the given segments removed from the pool (e.g.
        reserve a tutorial segment so the engine never serves it)."""
        drop = {int(s) for s in seg_ids}
        keep = [s for s in self.manifest.index if int(s) not in drop]
        return K7EngineInputs(
            manifest=self.manifest.loc[keep].copy(),
            Corr_l=self.Corr_l, Corr_t=self.Corr_t,
            task_codes=list(self.task_codes),
            task_labels=list(self.task_labels),
            task_pattern_words=list(self.task_pattern_words),
            task_families=list(self.task_families),
            sort_candidates_by_signal=self.sort_candidates_by_signal,
            source_manifest_path=self.source_manifest_path,
            source_manifest_sha256=self.source_manifest_sha256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_served_manifest_engine_inputs(
        manifest_path: str | os.PathLike = SERVED_MANIFEST_PATH,
        *, verify_frozen_hash: bool = True) -> K7EngineInputs:
    """Load the exact served v1.6-k7-35k web manifest for calibration.

    This is a read-only calibration-bank loader. The compact 700-segment H5
    bank remains ``build_k7_engine_inputs()`` and the parity fixture. Candidate
    arrays are stably ordered by each domain's signal so the validated F4.1
    deterministic coarse-to-fine selector operates on its required smooth
    one-dimensional ordering.
    """
    path = Path(manifest_path).resolve()
    observed_hash = _sha256(path)
    if verify_frozen_hash and observed_hash != SERVED_MANIFEST_SHA256:
        raise RuntimeError(
            f"served manifest hash {observed_hash} != frozen "
            f"{SERVED_MANIFEST_SHA256}")
    obj = json.loads(path.read_text())
    if obj.get("taskCodes") != TASK_CODES:
        raise RuntimeError(
            f"served task order {obj.get('taskCodes')} != {TASK_CODES}")
    if int(obj.get("nSegments", -1)) != 35193:
        raise RuntimeError(
            f"served nSegments {obj.get('nSegments')} != 35193")
    if int(obj.get("nParticles", -1)) != 1200:
        raise RuntimeError(
            f"served nParticles {obj.get('nParticles')} != 1200")
    if int(obj.get("perDomainCap", -1)) != 60:
        raise RuntimeError(
            f"served perDomainCap {obj.get('perDomainCap')} != 60")
    if obj.get("engineProfile") != "v15":
        raise RuntimeError(
            f"served engineProfile {obj.get('engineProfile')} != v15")

    records = []
    seen = set()
    family_counts = Counter()
    for segment in obj["segments"]:
        seg_id = int(segment["segId"])
        if seg_id in seen:
            raise RuntimeError(f"duplicate served segId {seg_id}")
        seen.add(seg_id)
        family = str(segment["testClass"])
        applicable = [int(k) for k in segment["applicableTaskIdx"]]
        expected = [0] if family == "spike" else list(IIIC_TASK_INDICES)
        if applicable != expected:
            raise RuntimeError(
                f"segId {seg_id} applicable {applicable} != {expected}")
        s_mean = segment["sMean"]
        s_sd = segment["sSd"]
        if len(s_mean) != K or len(s_sd) != K:
            raise RuntimeError(f"segId {seg_id} signal vector is not K=7")
        row = {
            "seg_id": seg_id,
            "family": family,
            "pattern_class": str(segment["patternClass"]),
        }
        for k, code in enumerate(TASK_CODES):
            if k in applicable:
                mean = float(s_mean[k])
                sd = float(s_sd[k])
                if not np.isfinite(mean) or not np.isfinite(sd) or sd < 0.0:
                    raise RuntimeError(
                        f"segId {seg_id} has invalid {code} signal")
                row[f"s_mean_{code}"] = mean
                row[f"s_sd_{code}"] = sd
            else:
                row[f"s_mean_{code}"] = np.nan
                row[f"s_sd_{code}"] = np.nan
        records.append(row)
        family_counts[family] += 1
    if len(records) != int(obj["nSegments"]):
        raise RuntimeError(
            f"manifest records {len(records)} != nSegments {obj['nSegments']}")
    if family_counts != Counter({"spike": 16527, "iiic": 18666}):
        raise RuntimeError(f"unexpected served families {family_counts}")

    Corr_l = np.asarray(obj["corrL"], dtype=float)
    Corr_t = np.asarray(obj["corrT"], dtype=float)
    if Corr_l.shape != (K, K) or Corr_t.shape != (K, K):
        raise RuntimeError("served prior blocks are not 7x7")
    manifest = pd.DataFrame.from_records(records).set_index("seg_id")
    return K7EngineInputs(
        manifest=manifest,
        Corr_l=Corr_l,
        Corr_t=Corr_t,
        task_codes=list(TASK_CODES),
        task_labels=list(obj["taskLabels"]),
        task_pattern_words=list(obj["taskPatternWords"]),
        task_families=list(obj["taskClasses"]),
        sort_candidates_by_signal=True,
        source_manifest_path=str(path),
        source_manifest_sha256=observed_hash,
    )


def build_k7_engine_inputs(verbose: bool = False) -> K7EngineInputs:
    """Assemble the K=7 engine inputs from the K=7 bank + K=7 signals + K=7 prior.

    Raises on any data-integrity defect:
      - bank H5 missing iiic/ or spike/ group
      - any bank seg has no signal row in segment_signals.csv
      - Sigma_l_fitted_k7 domain order != TASKS
      - any IIIC seg has unknown pattern_class
    """
    # 1. Bank segments + true labels for BOTH groups
    bank_iiic: dict[int, str] = {}     # seg_id → pattern_class
    bank_spike: dict[int, str] = {}    # seg_id → 'spike' (placeholder; true binary y comes from UI)
    with h5py.File(BANK_PATH, "r") as f:
        if "iiic" not in f and "spike" not in f:
            raise RuntimeError(f"{BANK_PATH} has neither 'iiic' nor 'spike' group")
        if "iiic" in f:
            for sid_str in f["iiic"]:
                try:
                    sid = int(sid_str)
                except ValueError:
                    continue
                bank_iiic[sid] = str(f["iiic"][sid_str].attrs.get("pattern_class", ""))
        if "spike" in f:
            for sid_str in f["spike"]:
                try:
                    sid = int(sid_str)
                except ValueError:
                    continue
                bank_spike[sid] = "spike"
    if not bank_iiic and not bank_spike:
        raise RuntimeError(f"no segments in {BANK_PATH}")

    # 2. Per-task signals — read segment_signals.csv (K=7; from Layer 2)
    sig_cols = ([f"s_mean_{c}" for c in TASK_CODES]
                + [f"s_sd_{c}" for c in TASK_CODES])
    iss = pd.read_csv(SIGNALS_CSV).set_index("seg_id")
    missing_cols = [c for c in sig_cols if c not in iss.columns]
    if missing_cols:
        raise RuntimeError(
            f"{SIGNALS_CSV} missing columns: {missing_cols}. "
            "Run pipeline/joint_calibration/assemble_outputs_k7.py")

    # Verify every bank seg has a signals-csv row (per-task NaN is fine —
    # cross-family signal columns are NaN by design)
    all_bank_ids = sorted(set(bank_iiic) | set(bank_spike))
    not_in_signals = [s for s in all_bank_ids if s not in iss.index]
    if not_in_signals:
        raise RuntimeError(
            f"{len(not_in_signals)} bank segs have no signal row in "
            f"{SIGNALS_CSV}: {not_in_signals[:5]}")

    # Build the manifest: family + pattern_class + 7-task signals
    man_rows = []
    for sid in all_bank_ids:
        row = {"seg_id": sid}
        if sid in bank_iiic:
            row["family"] = "iiic"
            row["pattern_class"] = bank_iiic[sid]
        else:
            row["family"] = "spike"
            row["pattern_class"] = "spike"
        # Pull per-task signals from segment_signals row
        sig_row = iss.loc[sid]
        for col in sig_cols:
            row[col] = sig_row[col]
        man_rows.append(row)
    man = pd.DataFrame(man_rows).set_index("seg_id")

    # Validation: spike segs should have NaN for IIIC signal columns (and vice versa)
    spike_segs = man[man["family"] == "spike"]
    iiic_segs = man[man["family"] == "iiic"]
    # Defensive: drop segs where the OWN-family signal column is NaN (un-calibrated)
    for fam, fam_segs in (("spike", spike_segs), ("iiic", iiic_segs)):
        if fam == "spike":
            sm_cols = ["s_mean_spike"]
        else:
            sm_cols = [f"s_mean_{c}" for c in ("sz", "lpd", "gpd", "lrda", "grda", "iic")]
        bad = fam_segs.index[fam_segs[sm_cols].isna().any(axis=1)].tolist()
        if bad:
            logger.warning(
                "K=7: dropping %d %s segs with NaN own-family signal: %s",
                len(bad), fam, bad[:5])
            man = man.drop(index=bad)
    if man.empty:
        raise RuntimeError("no usable K=7 segs after own-family NaN filtering")

    # Verify IIIC pattern_class is one of the 6 known classes
    iiic_segs_clean = man[man["family"] == "iiic"]
    iiic_words = TASK_PATTERN_WORDS[1:]  # exclude 'spike' at index 0
    unknown = sorted(set(iiic_segs_clean["pattern_class"]) - set(iiic_words))
    if unknown:
        raise RuntimeError(
            f"IIIC seg pattern_class not in K=6 IIIC vocabulary: {unknown}")

    # 3. K=7 prior — Sigma_l_fitted_k7.npy
    S = load_fitted_Sigma(SIGMA_PATH)
    domains = [str(d) for d in np.asarray(S["domains"])]
    if domains != TASK_CODES:
        raise RuntimeError(
            f"Sigma_l_fitted_k7.npy domain order {domains} != {TASK_CODES}")
    Corr_l = np.asarray(S["Corr_l"], dtype=float)
    Corr_t = np.asarray(S["Corr_t"], dtype=float)
    if Corr_l.shape != (K, K) or Corr_t.shape != (K, K):
        raise RuntimeError(
            f"K=7 Corr_l shape {Corr_l.shape} or Corr_t {Corr_t.shape} "
            f"!= ({K}, {K})")

    inp = K7EngineInputs(
        manifest=man, Corr_l=Corr_l, Corr_t=Corr_t,
        task_codes=list(TASK_CODES), task_labels=list(TASK_LABELS),
        task_pattern_words=list(TASK_PATTERN_WORDS),
        task_families=list(TASK_FAMILIES))
    if verbose:
        _report(inp)
    return inp


def _report(inp: K7EngineInputs) -> None:
    man = inp.manifest
    print("CORTEX K=7 engine inputs — validation report")
    print(f"  bank            : {BANK_PATH}")
    print(f"  signals         : {SIGNALS_CSV}")
    print(f"  prior           : {SIGMA_PATH}")
    fams = Counter(man["family"])
    print(f"  segments        : {len(man)} total "
          f"(iiic: {fams.get('iiic', 0)}, spike: {fams.get('spike', 0)})")
    print(f"  tasks (K={K})     : {inp.task_codes}")
    print(f"  task families   : {inp.task_families}")
    bs, bsd, bseg = inp.as_engine_arrays()
    print(f"  per-task bank sizes:")
    for k, code in enumerate(inp.task_codes):
        print(f"    {code:>5}: {len(bs[k]):>4} segs  "
              f"(s_mean range [{bs[k].min():+.2f}, {bs[k].max():+.2f}])")
    print(f"  prior Corr_l    : {inp.Corr_l.shape}  "
          f"diag={np.round(np.diag(inp.Corr_l), 2)}")
    print(f"  prior Corr_t    : {inp.Corr_t.shape}  "
          f"diag={np.round(np.diag(inp.Corr_t), 2)}")
    print(f"  max test length : {len(man)} questions "
          f"(each segment served at most once)")


if __name__ == "__main__":
    build_k7_engine_inputs(verbose=True)
