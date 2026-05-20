"""Standalone EEG bank viewer — adapted from morgoth-viewer.

Loads segments out of /Volumes/Extreme SSD/eeg_bank.h5 and displays them
with PyQt6 + pyqtgraph. A trimmed-down viewer with everything the user
asked for and nothing they didn't:

    * Load a segment from the bank (10 examples preselected, but the
      dropdown is populated from the full bank).
    * EEG display in bipolar / average / Laplacian montage.
    * Gain (microvolts per division) selector.
    * Bandpass + notch filter selectors.
    * Pan left / right buttons.
    * Optional spectrogram panel (4 regional means: LL, RL, LP, RP),
      computed on the displayed clip — placeholder for the "10-min
      spectrogram" feature, will be tuned with the user later.

Skipped on purpose: automated label/prediction overlays, IED/PDR
detection, event tables, report generation, cluster review.

Run:
    /Users/mwestover/GithubRepos/morgoth-viewer/morgoth_viewer_app/venv/bin/python \
        scripts/eeg_bank_viewer.py
"""
from __future__ import annotations
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Allow concurrent read while bank builds are still writing.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QLabel, QCheckBox, QSplitter
)
from scipy import signal as sig


BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "eeg_bank.h5"
SPEC_PATH = Path(__file__).resolve().parent.parent / "data" / "eeg_bank_spec.h5"   # precomputed 10-min spectrograms
LABELS_DIR = Path(__file__).resolve().parent.parent / "data" / "labels"


# ──────────────────────── data helpers ────────────────────────

CHANNELS_19 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
               'Fz', 'Cz', 'Pz',
               'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

BIPOLAR_MONTAGE = [
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fz', 'Cz'), ('Cz', 'Pz'),
]

LAPLACIAN_NEIGHBORS = {
    'Fp1': ['F3', 'F7', 'Fz'],  'F3':  ['Fp1', 'C3', 'Fz', 'F7'],
    'C3':  ['F3', 'P3', 'Cz', 'T3'], 'P3': ['C3', 'O1', 'Pz', 'T5'],
    'F7':  ['Fp1', 'F3', 'T3'], 'T3':  ['F7', 'C3', 'T5'],
    'T5':  ['T3', 'P3', 'O1'],  'O1':  ['T5', 'P3', 'Pz'],
    'Fz':  ['Fp1', 'F3', 'Cz', 'Fp2', 'F4'],
    'Cz':  ['Fz', 'C3', 'Pz', 'C4'],
    'Pz':  ['Cz', 'P3', 'O1', 'P4', 'O2'],
    'Fp2': ['F4', 'F8', 'Fz'],  'F4': ['Fp2', 'C4', 'Fz', 'F8'],
    'C4':  ['F4', 'P4', 'Cz', 'T4'], 'P4': ['C4', 'O2', 'Pz', 'T6'],
    'F8':  ['Fp2', 'F4', 'T4'], 'T4':  ['F8', 'C4', 'T6'],
    'T6':  ['T4', 'P4', 'O2'],  'O2':  ['T6', 'P4', 'Pz'],
}


# ──────────────────────── filter bank ────────────────────────

class FilterBank:
    """Compact filter bank — Butterworth bandpass + IIR notch, cached SOS."""
    BANDPASS = {
        "0.5-70 Hz": (0.5, 70.0), "0.5-40 Hz": (0.5, 40.0),
        "0.5-30 Hz": (0.5, 30.0), "0.5-20 Hz": (0.5, 20.0),
        "1-70 Hz":   (1.0, 70.0),
        "off":       None,
    }
    NOTCH = {"60 Hz": 60.0, "50 Hz": 50.0, "off": None}

    def __init__(self, fs):
        self.fs = float(fs)
        self._bp = {}
        self._nt = {}

    def _bp_sos(self, key):
        if key not in self._bp:
            v = self.BANDPASS.get(key)
            if v is None:
                self._bp[key] = None
            else:
                lo, hi = v
                hi = min(hi, self.fs / 2 - 1)
                self._bp[key] = sig.butter(N=2, Wn=[lo, hi], btype='band',
                                            fs=self.fs, output='sos')
        return self._bp[key]

    def _nt_sos(self, key):
        if key not in self._nt:
            f0 = self.NOTCH.get(key)
            if f0 is None:
                self._nt[key] = None
            else:
                b, a = sig.iirnotch(w0=f0, Q=30.0, fs=self.fs)
                self._nt[key] = sig.tf2sos(b, a)
        return self._nt[key]

    def apply(self, data, bandpass_key, notch_key):
        out = data.astype(np.float64).copy()
        bp = self._bp_sos(bandpass_key)
        if bp is not None:
            for i in range(out.shape[0]):
                out[i] = sig.sosfiltfilt(bp, out[i])
        nt = self._nt_sos(notch_key)
        if nt is not None:
            for i in range(out.shape[0]):
                out[i] = sig.sosfiltfilt(nt, out[i])
        return out


# ──────────────────────── montage transforms ────────────────────────

def apply_bipolar(data, channel_names):
    """Display-version of bipolar: 18 channel pairs + NaN separators between
    groups + EKG appended at the end. Use `apply_bipolar_clean` for
    spectrogram computation (no separators, no EKG)."""
    ch_to_idx = {nm: i for i, nm in enumerate(channel_names) if i < data.shape[0]}
    rows, names = [], []
    n = data.shape[1]
    blank = np.full(n, np.nan)
    for idx, (a, b) in enumerate(BIPOLAR_MONTAGE):
        if idx in (4, 8, 12, 16):  # separators
            rows.append(blank.copy()); names.append('')
        if a in ch_to_idx and b in ch_to_idx:
            rows.append(data[ch_to_idx[a]] - data[ch_to_idx[b]])
            names.append(f"{a}-{b}")
    # EKG row if available
    rows.append(blank.copy()); names.append('')
    for cand in ('EKG', 'ECG'):
        if cand in ch_to_idx:
            rows.append(data[ch_to_idx[cand]]); names.append(cand)
            break
    return np.asarray(rows), names


def apply_bipolar_clean(data, channel_names):
    """Clean (18, n_samples) bipolar — matches morgoth's `_fcn_bipolar`
    in compute_features.py, used for region averaging in the spectrogram.
    Returns array indexed [0..17] = LL, RL, LP, RP, central (Fz-Cz, Cz-Pz).
    """
    ch_to_idx = {nm: i for i, nm in enumerate(channel_names) if i < data.shape[0]}
    n = data.shape[1]
    bp = np.zeros((18, n), dtype=data.dtype)
    # Index assignments verbatim from compute_features._fcn_bipolar
    # Channel name → index in the 19-ch order is implicit via ch_to_idx
    def diff(a, b):
        if a in ch_to_idx and b in ch_to_idx:
            return data[ch_to_idx[a]] - data[ch_to_idx[b]]
        return np.zeros(n, dtype=data.dtype)
    # Left temporal
    bp[0] = diff('Fp1', 'F7');  bp[1] = diff('F7', 'T3')
    bp[2] = diff('T3', 'T5');   bp[3] = diff('T5', 'O1')
    # Right temporal
    bp[4] = diff('Fp2', 'F8');  bp[5] = diff('F8', 'T4')
    bp[6] = diff('T4', 'T6');   bp[7] = diff('T6', 'O2')
    # Left parasagittal
    bp[8]  = diff('Fp1', 'F3'); bp[9]  = diff('F3', 'C3')
    bp[10] = diff('C3', 'P3');  bp[11] = diff('P3', 'O1')
    # Right parasagittal
    bp[12] = diff('Fp2', 'F4'); bp[13] = diff('F4', 'C4')
    bp[14] = diff('C4', 'P4');  bp[15] = diff('P4', 'O2')
    # Central
    bp[16] = diff('Fz', 'Cz');  bp[17] = diff('Cz', 'Pz')
    return bp


def apply_average(data, channel_names):
    ch_to_idx = {nm: i for i, nm in enumerate(channel_names) if i < data.shape[0]}
    n_ch_eeg = min(19, data.shape[0])
    avg = np.nanmean(data[:n_ch_eeg], axis=0)
    rows, names = [], []
    n = data.shape[1]
    blank = np.full(n, np.nan)
    for i in range(n_ch_eeg):
        if i == 8:  # after O1
            rows.append(blank.copy()); names.append('')
        if i == 11:  # after Pz
            rows.append(blank.copy()); names.append('')
        rows.append(data[i] - avg)
        names.append(f"{channel_names[i] if i < len(channel_names) else f'ch{i}'}-Av")
    _append_ekg(rows, names, data, ch_to_idx, blank)
    return np.asarray(rows), names


def apply_laplacian(data, channel_names):
    ch_to_idx = {nm: i for i, nm in enumerate(channel_names) if i < data.shape[0]}
    rows, names = [], []
    n = data.shape[1]
    blank = np.full(n, np.nan)
    for i, name in enumerate(channel_names[:min(19, data.shape[0])]):
        if i == 8:
            rows.append(blank.copy()); names.append('')
        if i == 11:
            rows.append(blank.copy()); names.append('')
        nbrs = LAPLACIAN_NEIGHBORS.get(name, [])
        nbr_idxs = [ch_to_idx[nb] for nb in nbrs if nb in ch_to_idx]
        if nbr_idxs:
            ref = np.mean(data[nbr_idxs], axis=0)
        else:
            ref = np.zeros(n)
        rows.append(data[i] - ref)
        names.append(f"{name}-L")
    _append_ekg(rows, names, data, ch_to_idx, blank)
    return np.asarray(rows), names


def _append_ekg(rows, names, data, ch_to_idx, blank):
    """Append separator + EKG row to a montage's display rows."""
    rows.append(blank.copy()); names.append('')
    for cand in ('EKG', 'ECG'):
        if cand in ch_to_idx:
            rows.append(data[ch_to_idx[cand]]); names.append(cand)
            return
    # No EKG channel — drop the separator we just added so we don't end with a blank
    rows.pop(); names.pop()


# ──────────────────────── spectrogram ────────────────────────

def _butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=3):
    """Match morgoth's compute_features._butter_bandpass:
    Butterworth N=3, applied with scipy.signal.filtfilt (zero-phase)."""
    nyq = 0.5 * fs
    b, a = sig.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return sig.filtfilt(b, a, data, axis=-1)


def _butter_notch_filtfilt(data, freq, fs, Q=30):
    """Match morgoth's compute_features._butter_notch (iirnotch + filtfilt)."""
    b, a = sig.iirnotch(freq, Q, fs)
    return sig.filtfilt(b, a, data, axis=-1)


def compute_regional_spectrograms(data_bipolar_clean, fs, window_size=4.0,
                                    step_size=1.0, fmin=0.5, fmax=25.0):
    """Compute 4-region (LL, RL, LP, RP) mean spectrograms from a clean
    18-channel bipolar montage (no NaN separators). Matches morgoth-viewer's
    compute_features.py:
        * scipy.signal.spectrogram, scaling='density', mode='psd'
        * 4-sec window, 1-sec step
        * 0.5–25 Hz range
        * mean of 4 bipolar channels per region
    Returns dict of {region_name: (n_freqs, n_times)}, freqs, stimes.
    """
    nperseg = int(window_size * fs)
    noverlap = int((window_size - step_size) * fs)
    if nperseg > data_bipolar_clean.shape[1]:
        return None, None, None
    region_chans = {'LL': [0, 1, 2, 3], 'RL': [4, 5, 6, 7],
                     'LP': [8, 9, 10, 11], 'RP': [12, 13, 14, 15]}
    out = {}
    freqs = stimes = None
    for region, idxs in region_chans.items():
        Sxx_sum = None
        for ch in idxs:
            f, t, Sxx = sig.spectrogram(data_bipolar_clean[ch], fs=fs,
                                          nperseg=nperseg, noverlap=noverlap,
                                          scaling='density', mode='psd')
            if freqs is None:
                freqs, stimes = f, t
            Sxx_sum = Sxx if Sxx_sum is None else Sxx_sum + Sxx
        out[region] = Sxx_sum / max(len(idxs), 1)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return {k: v[mask] for k, v in out.items()}, freqs[mask], stimes


# ──────────────────────── example picker ────────────────────────

def pick_examples(n_total=10):
    """Pick `n_total` diverse seg_ids from the bank covering IIIC subtypes +
    spike. Use segments.csv as the source-of-truth list and try-open each
    candidate (avoids relying on the bank's /segments group iteration,
    which can get corrupted by interrupted writes).

    Prefer seg_ids that already have precomputed `sdata` so the user sees
    the 10-min spectrogram immediately.
    """
    if not BANK_PATH.exists():
        raise SystemExit(f"Bank not found at {BANK_PATH}")
    print(f"Opening bank: {BANK_PATH}")

    seg = pd.read_csv(LABELS_DIR / "segments.csv", low_memory=False,
                       usecols=["seg_id", "source_dataset", "subtype"])
    seg["seg_id"] = seg["seg_id"].astype(int)
    print(f"  segments.csv rows: {len(seg):,}")

    # Compute pattern_class plurality per seg from labels.csv
    lab = pd.read_csv(LABELS_DIR / "labels.csv", low_memory=False,
                       dtype={"value": "str"})
    pc = lab[lab["label_type"] == "pattern_class"]
    plur = (pc.groupby("seg_id")["value"]
              .agg(lambda s: s.mode().iloc[0] if len(s) else None)
              .rename("plurality"))
    seg = seg.merge(plur, on="seg_id", how="left")

    # Identify seg_ids that already have precomputed sdata in the SIDE file.
    have_sdata: set[int] = set()
    bank_seg_ids: set[int] = set()
    in_bank_domains: dict[int, str] = {}
    with h5py.File(BANK_PATH, "r") as f:
        for domain in ["iiic", "spike"]:
            if domain not in f:
                continue
            for sid_str in f[domain]:
                try:
                    sid = int(sid_str)
                    bank_seg_ids.add(sid)
                    in_bank_domains[sid] = domain
                except ValueError:
                    continue
    if SPEC_PATH.exists():
        try:
            with h5py.File(SPEC_PATH, "r") as fs:
                for sid in seg["seg_id"]:
                    try:
                        g = fs[f"segments/{sid}"]
                        if "sdata" in g:
                            have_sdata.add(int(sid))
                    except Exception:
                        pass
        except Exception:
            pass
    print(f"  in bank (try-open): {len(bank_seg_ids):,}  with sdata: {len(have_sdata):,}")
    seg = seg[seg["seg_id"].isin(bank_seg_ids)]

    def _safe_pick(df, n=1):
        for sid in df["seg_id"]:
            if int(sid) in bank_seg_ids:
                return int(sid)
        return None

    # Pick: prefer segments with precomputed sdata (one per IIIC class), then
    # fall back to non-precomputed for the spike examples
    picks: list[int] = []
    classes = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    for cls in classes:
        cands_sdata = seg[(seg["plurality"] == cls) & seg["seg_id"].isin(have_sdata)]
        if len(cands_sdata):
            picks.append(int(cands_sdata["seg_id"].iloc[0]))
            continue
        cands = seg[seg["plurality"] == cls]
        if len(cands):
            picks.append(int(cands["seg_id"].iloc[len(cands) // 3]))
    # Fill remainder with spikes_hm (or spike) — these don't have 10-min ctx
    spikes_with_sdata = seg[
        seg["source_dataset"].astype(str).str.contains("spikes", na=False)
        & seg["seg_id"].isin(have_sdata)
    ]
    for sid in spikes_with_sdata["seg_id"]:
        if len(picks) >= n_total:
            break
        if int(sid) not in picks:
            picks.append(int(sid))
    spikes = seg[seg["source_dataset"].astype(str).str.contains("spikes", na=False)]
    if len(spikes):
        step = max(1, len(spikes) // (n_total - len(picks) + 1))
        for i in range(0, len(spikes), step):
            if len(picks) >= n_total:
                break
            sid = int(spikes["seg_id"].iloc[i])
            if sid not in picks:
                picks.append(sid)
    picks = picks[:n_total]
    print(f"  picked {len(picks)} examples: {picks}")
    print(f"  of which precomputed-sdata: "
          f"{sum(1 for s in picks if s in have_sdata)}/{len(picks)}")
    return picks, seg.set_index("seg_id"), in_bank_domains


# ──────────────────────── viewer ────────────────────────

class BankViewer(QMainWindow):
    def __init__(self, seg_ids, seg_meta, seg_domains):
        super().__init__()
        self.setWindowTitle("EEG Bank Viewer")
        self.resize(1500, 950)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.bank = h5py.File(BANK_PATH, "r")
        self.spec_file = None
        if SPEC_PATH.exists():
            try:
                self.spec_file = h5py.File(SPEC_PATH, "r")
            except Exception as e:
                print(f"  WARN: could not open spec file {SPEC_PATH}: {e}",
                      flush=True)
                self.spec_file = None
        self.seg_ids = seg_ids
        self.seg_meta = seg_meta
        self.seg_domains = seg_domains
        self.current_idx = 0
        self.t_start = 0.0
        self.window_s = 10.0

        self._build_ui()
        self._load_segment(0)

    def closeEvent(self, ev):
        self.bank.close()
        if self.spec_file is not None:
            try:
                self.spec_file.close()
            except Exception:
                pass
        super().closeEvent(ev)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Top bar: example selector + segment metadata
        top = QHBoxLayout()
        self.example_box = QComboBox()
        for i, sid in enumerate(self.seg_ids):
            meta = self.seg_meta.loc[sid] if sid in self.seg_meta.index else None
            sd = meta["source_dataset"] if meta is not None else "?"
            sub = meta["subtype"] if meta is not None and not pd.isna(meta["subtype"]) else ""
            plur = meta["plurality"] if meta is not None and "plurality" in meta and not pd.isna(meta["plurality"]) else ""
            label = f"[{i+1}] seg {sid}  {sd}  {sub}  ({plur})"
            self.example_box.addItem(label.strip())
        self.example_box.currentIndexChanged.connect(self._load_segment)
        top.addWidget(QLabel("Example:"))
        top.addWidget(self.example_box, 1)
        self.meta_label = QLabel("")
        top.addWidget(self.meta_label)
        outer.addLayout(top)

        # Plots: spectrogram (LEFT, fixed 250-300 px) + EEG (RIGHT, all
        # remaining space) — matches morgoth-viewer's main_window layout
        # exactly (setMinimumWidth(250), setMaximumWidth(300)).
        plots_row = QHBoxLayout()
        plots_row.setSpacing(5)
        outer.addLayout(plots_row, 1)

        spec_container = QWidget()
        spec_container.setMinimumWidth(250)
        spec_container.setMaximumWidth(300)
        spec_layout = QVBoxLayout(spec_container)
        spec_layout.setSpacing(2); spec_layout.setContentsMargins(0, 0, 0, 0)

        # Morgoth's jet-style colormap (verbatim from gui/widgets/spectrogram.py)
        jet_colors = [
            (0, 0, 127), (0, 0, 255), (0, 127, 255), (0, 255, 255),
            (127, 255, 127), (255, 255, 0), (255, 127, 0),
            (255, 0, 0), (127, 0, 0),
        ]
        jet_cmap = pg.ColorMap(pos=np.linspace(0, 1, len(jet_colors)),
                                color=jet_colors)
        jet_lut = jet_cmap.getLookupTable()

        self.spec_plots = []
        self.spec_images = []
        for i, region in enumerate(('LL', 'RL', 'LP', 'RP')):
            p = pg.PlotWidget()
            p.setBackground('w')
            p.setMouseEnabled(x=False, y=False)
            p.getViewBox().setMenuEnabled(False)
            p.getAxis('left').setLabel(f'Freq (Hz) - {region}')
            p.getAxis('left').setWidth(60)
            # Only show time axis on the bottom panel
            if i < 3:
                p.getAxis('bottom').setHeight(0)
                p.getAxis('bottom').setStyle(showValues=False)
            else:
                p.getAxis('bottom').setLabel('Time (s)')
            img = pg.ImageItem()
            img.setLookupTable(jet_lut)
            p.addItem(img)
            spec_layout.addWidget(p)
            self.spec_plots.append(p); self.spec_images.append(img)
        plots_row.addWidget(spec_container)
        self.spec_container = spec_container

        self.eeg_plot = pg.PlotWidget()
        self.eeg_plot.showGrid(x=True, y=False, alpha=0.25)
        self.eeg_plot.setLabel("bottom", "Time (s)")
        # Disable mouse interaction entirely: wheel-zoom kept stretching the
        # time axis annoyingly. Pan via buttons / ← → arrows instead.
        self.eeg_plot.setMouseEnabled(x=False, y=False)
        self.eeg_plot.getViewBox().setMenuEnabled(False)
        # Also disable wheel zoom on spectrogram panels for the same reason
        plots_row.addWidget(self.eeg_plot, 1)

        # Bottom controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Montage:"))
        self.montage_box = QComboBox()
        self.montage_box.addItems(["bipolar", "average", "laplacian"])
        self.montage_box.currentTextChanged.connect(self._redraw)
        ctrl.addWidget(self.montage_box)

        ctrl.addWidget(QLabel("Gain (µV/div):"))
        self.gain_box = QComboBox()
        # Match morgoth's scale ladder verbatim (viewer_widget.py line 3515)
        self.scale_ladder = [1, 2, 7, 10, 15, 20, 30, 50, 70, 100, 200, 300]
        for g in self.scale_ladder:
            self.gain_box.addItem(str(g))
        self.gain_box.setCurrentText("100")
        self.gain_box.currentTextChanged.connect(self._redraw)
        ctrl.addWidget(self.gain_box)

        ctrl.addWidget(QLabel("Bandpass:"))
        self.bp_box = QComboBox()
        self.bp_box.addItems(FilterBank.BANDPASS.keys())
        self.bp_box.setCurrentText("0.5-70 Hz")
        self.bp_box.currentTextChanged.connect(self._redraw)
        ctrl.addWidget(self.bp_box)

        ctrl.addWidget(QLabel("Notch:"))
        self.notch_box = QComboBox()
        self.notch_box.addItems(FilterBank.NOTCH.keys())
        self.notch_box.setCurrentText("60 Hz")
        self.notch_box.currentTextChanged.connect(self._redraw)
        ctrl.addWidget(self.notch_box)

        ctrl.addWidget(QLabel("Window (s):"))
        self.win_box = QComboBox()
        for w in (5, 10, 15, 20, 30):
            self.win_box.addItem(str(w))
        self.win_box.setCurrentText("10")
        self.win_box.currentTextChanged.connect(self._on_window_change)
        ctrl.addWidget(self.win_box)

        self.spec_cb = QCheckBox("Show spectrogram")
        self.spec_cb.setChecked(True)
        self.spec_cb.toggled.connect(self._on_spec_toggle)
        ctrl.addWidget(self.spec_cb)

        ctrl.addStretch(1)
        self.prev_btn = QPushButton("◀ Prev example")
        self.prev_btn.clicked.connect(lambda: self._step_example(-1))
        ctrl.addWidget(self.prev_btn)
        self.pan_l_btn = QPushButton("◀ Pan")
        self.pan_l_btn.clicked.connect(lambda: self._pan(-1))
        ctrl.addWidget(self.pan_l_btn)
        self.pan_r_btn = QPushButton("Pan ▶")
        self.pan_r_btn.clicked.connect(lambda: self._pan(+1))
        ctrl.addWidget(self.pan_r_btn)
        self.next_btn = QPushButton("Next example ▶")
        self.next_btn.clicked.connect(lambda: self._step_example(+1))
        ctrl.addWidget(self.next_btn)
        outer.addLayout(ctrl)

    def _on_window_change(self, _):
        self.window_s = float(self.win_box.currentText())
        self._redraw()

    def _on_spec_toggle(self, on):
        self.spec_container.setVisible(on)
        if on:
            self._redraw()

    def _step_example(self, delta):
        new_idx = (self.current_idx + delta) % len(self.seg_ids)
        self.example_box.setCurrentIndex(new_idx)

    def _pan(self, direction):
        step = self.window_s * 0.5
        new = self.t_start + direction * step
        max_t = max(0.0, self.duration - self.window_s)
        self.t_start = float(np.clip(new, 0.0, max_t))
        self._redraw()

    def _load_segment(self, idx):
        self.current_idx = int(idx)
        sid = self.seg_ids[self.current_idx]
        domain = self.seg_domains.get(sid, "iiic")
        try:
            grp = self.bank[f"{domain}/{sid}"]
            dset = grp["eeg30s"]
            self.data = np.asarray(dset, dtype=np.float32)
        except Exception as e:
            print(f"  WARN: seg {sid} unreadable ({e}); skipping", flush=True)
            self.meta_label.setText(f"seg {sid}: UNREADABLE")
            self.data = None
            return
        self.fs = float(dset.attrs.get("fs_hz", 200.0))
        ch_attr = dset.attrs.get("channel_names")
        if ch_attr is not None:
            self.channel_names = [b.decode("utf-8", errors="ignore") if isinstance(b, (bytes, np.bytes_)) else str(b)
                                   for b in np.asarray(ch_attr)]
        else:
            self.channel_names = CHANNELS_19[:self.data.shape[0]]
        self.duration = self.data.shape[1] / self.fs
        self.t_start = 0.0
        if self.window_s > self.duration:
            self.window_s = self.duration
            self.win_box.setCurrentText(str(int(round(self.window_s))))
        self.filter_bank = FilterBank(self.fs)
        self.meta_label.setText(
            f"fs={self.fs:.0f} Hz · {self.data.shape[0]} ch × "
            f"{self.data.shape[1]} samp ({self.duration:.1f} s)"
        )
        self._redraw()

    def _redraw(self):
        if not hasattr(self, "data"):
            return
        self._draw_eeg()
        if self.spec_cb.isChecked():
            self._draw_spectrogram()

    def _draw_eeg(self):
        self.eeg_plot.clear()
        n_samples = self.data.shape[1]
        start = max(0, int(self.t_start * self.fs))
        end = min(n_samples, start + int(self.window_s * self.fs))
        if end <= start:
            return
        window = self.data[:, start:end]
        t = np.arange(start, end) / self.fs

        # Filter
        bp = self.bp_box.currentText()
        notch = self.notch_box.currentText()
        try:
            filtered = self.filter_bank.apply(window, bp, notch).astype(np.float32)
        except Exception as e:
            print(f"filter failed: {e}; falling back to unfiltered")
            filtered = window

        # Montage
        montage = self.montage_box.currentText()
        if montage == "bipolar":
            disp, names = apply_bipolar(filtered, self.channel_names)
        elif montage == "average":
            disp, names = apply_average(filtered, self.channel_names)
        else:
            disp, names = apply_laplacian(filtered, self.channel_names)

        n_ch = len(names)
        gain_uv = float(self.gain_box.currentText())
        z = 1.0 / gain_uv  # so that gain_uv µV → 1 division unit
        clip = 3 * gain_uv

        y_ticks = []
        for i in range(n_ch):
            name = names[i]
            offset = float(n_ch - i)
            y_ticks.append((offset, name))
            if name == '':
                continue
            ch = np.nan_to_num(disp[i], nan=0.0)
            is_ecg = name.upper().startswith(('EKG', 'ECG'))
            if is_ecg:
                mu = np.nanmean(ch); sd = np.nanstd(ch) or 1.0
                y = 0.2 * (ch - mu) / sd
                color = (200, 0, 0)
            else:
                y = z * np.clip(ch, -clip, clip)
                color = (0, 0, 0)
            self.eeg_plot.plot(t, y + offset,
                                pen=pg.mkPen(color=color, width=1))
        self.eeg_plot.setYRange(0, n_ch + 1)
        self.eeg_plot.setXRange(t[0], t[-1], padding=0.005)
        self.eeg_plot.getAxis("left").setTicks([y_ticks])
        self._add_scale_bar(gain_uv, n_ch, t[-1])
        self.eeg_plot.setLabel("top",
            f"seg {self.seg_ids[self.current_idx]}  "
            f"[{self.t_start:.1f} - {self.t_start + self.window_s:.1f} s of "
            f"{self.duration:.1f} s]  montage={montage}  "
            f"gain={gain_uv:.0f}µV/div")

    def _draw_spectrogram(self):
        """Display regional spectrograms matching morgoth-viewer exactly.

        Order of preference:
          1. Pre-computed `sdata`/`sfreqs`/`stimes` stored on the segment
             group in the H5 (morgoth's offline approach). When that
             exists, the viewer just slices and displays — no recompute.
          2. Otherwise compute on-the-fly with N=3 Butterworth + iirnotch
             (filtfilt, matching compute_features._butter_bandpass) +
             clean bipolar (no NaN separators) + region averaging. This
             is a stopgap until the 10-min precompute batch lands.

        Display:
          * 10*log10(S + eps) → dB
          * Fixed color levels [-10, 25] dB (no autoscale)
          * Jet colormap (verbatim from morgoth)
          * Y-axis ticks labelled by actual frequency
        """
        # Try precomputed from the side file first
        sid = self.seg_ids[self.current_idx]
        precomp = None
        if self.spec_file is not None:
            try:
                grp = self.spec_file[f"segments/{sid}"]
                if "sdata" in grp and "sfreqs" in grp and "stimes" in grp:
                    sdata = np.asarray(grp["sdata"], dtype=np.float32)
                    sfreqs = np.asarray(grp["sfreqs"], dtype=np.float32).flatten()
                    stimes = np.asarray(grp["stimes"], dtype=np.float32).flatten()
                    n_freqs = len(sfreqs)
                    # morgoth on-disk format is (n_times, n_freqs*4); transpose if needed
                    if sdata.shape[0] != n_freqs * 4 and sdata.shape[1] == n_freqs * 4:
                        sdata = sdata.T
                    if sdata.shape[0] == n_freqs * 4:
                        regs = {'LL': sdata[0*n_freqs:1*n_freqs],
                                'RL': sdata[1*n_freqs:2*n_freqs],
                                'LP': sdata[2*n_freqs:3*n_freqs],
                                'RP': sdata[3*n_freqs:4*n_freqs]}
                        precomp = (regs, sfreqs, stimes)
            except Exception:
                precomp = None
        if precomp is None:
            # On-the-fly compute, matching morgoth's compute_features.py
            data = np.nan_to_num(self.data, nan=0.0).astype(np.float64)
            try:
                f0 = _butter_bandpass_filtfilt(data, 0.5, 70.0, self.fs, order=3)
                f0 = _butter_notch_filtfilt(f0, 60.0, self.fs, Q=30)
            except Exception:
                f0 = data
            bipolar_clean = apply_bipolar_clean(f0, self.channel_names)
            regs, freqs, stimes = compute_regional_spectrograms(bipolar_clean, self.fs)
            if regs is None:
                for img in self.spec_images:
                    img.clear()
                return
            precomp = (regs, freqs, stimes)
        regs, freqs, stimes = precomp
        n_freqs = len(freqs)
        t0, t1 = float(stimes[0]), float(stimes[-1])
        eps = np.finfo(np.float32).eps
        for i, (img, plot, region) in enumerate(zip(self.spec_images,
                                                     self.spec_plots,
                                                     ('LL', 'RL', 'LP', 'RP'))):
            S = np.asarray(regs[region], dtype=np.float32)
            spec_db = 10.0 * np.log10(S + eps)   # (n_freqs, n_times)
            # pyqtgraph ImageItem wants (width=n_times, height=n_freqs)
            img.setImage(spec_db.T)
            img.setRect(t0, 0.0, t1 - t0, float(n_freqs))
            img.setLevels([-10.0, 25.0])         # MATLAB jet range
            plot.setXRange(t0, t1, padding=0)
            plot.setYRange(0.0, float(n_freqs), padding=0)
            # Frequency ticks
            step = max(1, n_freqs // 5)
            y_ticks = [(j, f'{freqs[j]:.0f}')
                        for j in range(0, n_freqs, step)]
            plot.getAxis('left').setTicks([y_ticks])

    # ─────────────── scale bar ───────────────

    def _add_scale_bar(self, gain_uv, n_ch, x_end):
        """Draw morgoth-style scale bar: 1-s horizontal + gain_uv vertical."""
        time_bar = 1.0
        x_off = self.window_s * 0.06 + 0.6   # ~6% from right edge
        y_pos = 0.5
        x_start = x_end - x_off
        pen = pg.mkPen(color=(0, 0, 0), width=2)
        # Horizontal (1 sec)
        self.eeg_plot.plot([x_start, x_start + time_bar],
                            [y_pos, y_pos], pen=pen)
        # Vertical (gain_uv → 1 plot-unit, since z_scale = 1/gain_uv)
        self.eeg_plot.plot([x_start, x_start],
                            [y_pos, y_pos + 1.0], pen=pen)
        t_lbl = pg.TextItem("1 s", anchor=(0.5, 1), color=(0, 0, 0))
        t_lbl.setPos(x_start + time_bar / 2, y_pos - 0.05)
        self.eeg_plot.addItem(t_lbl)
        amp_lbl = pg.TextItem(f"{int(gain_uv)} µV",
                                anchor=(1, 0.5), color=(0, 0, 0))
        amp_lbl.setPos(x_start - 0.1, y_pos + 0.5)
        self.eeg_plot.addItem(amp_lbl)

    # ─────────────── keyboard ───────────────

    # Keys we intercept application-wide so combo boxes don't eat them
    _HOTKEYS = {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up,
                 Qt.Key.Key_Down, Qt.Key.Key_Control, Qt.Key.Key_N,
                 Qt.Key.Key_P}

    def eventFilter(self, obj, event):
        """App-wide key intercept (installed on QApplication so we beat
        QComboBox / QPushButton focus). Matches morgoth's eventFilter
        pattern but only for the hotkeys we handle.
        """
        if event.type() == QEvent.Type.KeyPress and event.key() in self._HOTKEYS:
            self.keyPressEvent(event)
            return True   # consume — don't let combo boxes see it
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        """Morgoth-style keyboard shortcuts.

          ← / →  : pan ±10 s (or ±half window if window shorter than 10s)
          ↑ / ↓  : step through gain ladder (↑ = bigger traces)
          Ctrl   : cycle montage bipolar → average → laplacian → bipolar
          n / p  : next / previous example
        """
        key = event.key()
        if key == Qt.Key.Key_Left:
            self._pan(-1 if self.window_s < 10 else -10 / self.window_s)
        elif key == Qt.Key.Key_Right:
            self._pan(+1 if self.window_s < 10 else +10 / self.window_s)
        elif key == Qt.Key.Key_Up:
            cur = float(self.gain_box.currentText())
            idx = min(range(len(self.scale_ladder)),
                       key=lambda i: abs(self.scale_ladder[i] - cur))
            if idx > 0:
                self.gain_box.setCurrentText(str(self.scale_ladder[idx - 1]))
        elif key == Qt.Key.Key_Down:
            cur = float(self.gain_box.currentText())
            idx = min(range(len(self.scale_ladder)),
                       key=lambda i: abs(self.scale_ladder[i] - cur))
            if idx < len(self.scale_ladder) - 1:
                self.gain_box.setCurrentText(str(self.scale_ladder[idx + 1]))
        elif key == Qt.Key.Key_Control:
            order = ["bipolar", "average", "laplacian"]
            i = order.index(self.montage_box.currentText())
            self.montage_box.setCurrentText(order[(i + 1) % len(order)])
        elif key == Qt.Key.Key_N:
            self._step_example(+1)
        elif key == Qt.Key.Key_P:
            self._step_example(-1)
        else:
            super().keyPressEvent(event)


def main():
    # Stop Qt's macOS Cmd/Ctrl swap so the physical Control key actually
    # produces Qt.Key.Key_Control (matching morgoth's behavior on
    # Linux/Windows). Must be set before QApplication is constructed.
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_MacDontSwapCtrlAndMeta, True)
    app = QApplication(sys.argv)
    seg_ids, seg_meta, seg_domains = pick_examples(n_total=10)
    if not seg_ids:
        raise SystemExit("No segments to display")
    win = BankViewer(seg_ids, seg_meta, seg_domains)
    # Install app-wide event filter so combo boxes don't swallow arrow
    # keys / Ctrl before we see them.
    app.installEventFilter(win)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
