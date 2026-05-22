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
import csv
import datetime
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
from PyQt6.QtCore import Qt, QEvent, QTimer, QRect, QPoint
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QPainter, QPainterPath,
                         QPen, QPixmap)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QFrame, QPushButton, QLabel, QCheckBox, QSplitter,
    QLineEdit, QFormLayout, QStackedWidget
)
from scipy import signal as sig


BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "eeg_bank.h5"
SPEC_PATH = Path(__file__).resolve().parent.parent / "data" / "eeg_bank_spec.h5"   # precomputed 10-min spectrograms
LABELS_DIR = Path(__file__).resolve().parent.parent / "data" / "labels"
LOGO_PATH = (Path(__file__).resolve().parent.parent / "data"
             / "Brain_Data_Science_Platform.png")


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

    # Bank membership + IIIC pattern class come straight from the bank's
    # per-segment group attrs — far faster than the old path, which did a
    # full read of the 2.1M-row labels.csv just to recover the plurality.
    have_sdata: set[int] = set()
    bank_seg_ids: set[int] = set()
    in_bank_domains: dict[int, str] = {}
    iiic_class: dict[int, str] = {}
    with h5py.File(BANK_PATH, "r") as f:
        for domain in ["iiic", "spike"]:
            if domain not in f:
                continue
            for sid_str in f[domain]:
                try:
                    sid = int(sid_str)
                except ValueError:
                    continue
                bank_seg_ids.add(sid)
                in_bank_domains[sid] = domain
                if domain == "iiic":
                    pc = f[domain][sid_str].attrs.get("pattern_class")
                    if pc is not None:
                        iiic_class[sid] = str(pc)
    seg["plurality"] = seg["seg_id"].map(iiic_class)
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
    # Fill the remainder with spike segments, identified directly from the
    # bank's `spike` group — the segments.csv `source_dataset` string is
    # not a reliable spike marker. Even-stride sample across the group.
    spike_ids = sorted(s for s, d in in_bank_domains.items() if d == "spike")
    if spike_ids and len(picks) < n_total:
        step = max(1, len(spike_ids) // (n_total - len(picks)))
        for i in range(0, len(spike_ids), step):
            if len(picks) >= n_total:
                break
            if spike_ids[i] not in picks:
                picks.append(spike_ids[i])
    picks = picks[:n_total]
    print(f"  picked {len(picks)} examples: {picks}")
    print(f"  of which precomputed-sdata: "
          f"{sum(1 for s in picks if s in have_sdata)}/{len(picks)}")
    return picks, seg.set_index("seg_id"), in_bank_domains


# ──────────────────────── viewer ────────────────────────

class AnswerButton(QPushButton):
    """A QPushButton kept in its platform-native style. When flashed it
    paints a coloured rounded outline on top of the native rendering —
    the brief 'answer selected' highlight. No stylesheet is set, so the
    button stays native on macOS / Windows / Linux alike."""
    _OUTLINE = QColor("#f5a623")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash = False

    def set_flash(self, on):
        on = bool(on)
        if on != self._flash:
            self._flash = on
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)            # native button rendering
        if not self._flash:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(self._OUTLINE)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Inset more on top/bottom than left/right: the native button does
        # not fill the widget's full height, so a larger vertical inset
        # keeps the outline hugging the button rather than its dead margin.
        painter.drawRoundedRect(self.rect().adjusted(2, 5, -2, -5), 6, 6)


class TutorialOverlay(QWidget):
    """Coach-marks overlay: dims the viewer and spotlights regions in
    sequence with explanatory cards, ending in a 'begin' confirmation.

    `steps` is a list of (target_widget_or_None, title, body); `on_finish`
    is called when the final card's button is pressed."""

    _ACCENT = QColor("#f5a623")
    _VEIL = QColor(8, 9, 12, 190)

    def __init__(self, parent, steps, on_finish):
        super().__init__(parent)
        self._steps = steps
        self._on_finish = on_finish
        self._i = 0
        self._card = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # The card matches the viewer's native system font, not the serif.
        sysfam = QFontDatabase.systemFont(
            QFontDatabase.SystemFont.GeneralFont).family()

        card = QWidget(self)
        card.setObjectName("tutCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setFixedWidth(420)
        card.setStyleSheet(
            "#tutCard { background-color: #14161c;"
            " border: 1px solid #3a3d45; }")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(24, 20, 24, 20)
        cl.setSpacing(0)

        self._step_lbl = QLabel()
        sfont = QFont(sysfam)
        sfont.setPointSize(9)
        sfont.setWeight(QFont.Weight.Medium)
        sfont.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        self._step_lbl.setFont(sfont)
        self._step_lbl.setStyleSheet("color: #6b7280;")
        cl.addWidget(self._step_lbl)

        self._title_lbl = QLabel()
        tfont = QFont(sysfam)
        tfont.setPointSize(17)
        tfont.setWeight(QFont.Weight.DemiBold)
        self._title_lbl.setFont(tfont)
        self._title_lbl.setStyleSheet("color: #eef1f5;")
        cl.addSpacing(7)
        cl.addWidget(self._title_lbl)

        self._body_lbl = QLabel()
        self._body_lbl.setWordWrap(True)
        bfont = QFont(sysfam)
        bfont.setPointSize(12)
        self._body_lbl.setFont(bfont)
        self._body_lbl.setStyleSheet("color: #aab0ba;")
        cl.addSpacing(10)
        cl.addWidget(self._body_lbl)

        self._next_btn = QPushButton()
        self._next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next_btn.setFixedHeight(38)
        nb_font = _btn_font()
        nb_font.setFamily(sysfam)
        self._next_btn.setFont(nb_font)
        self._next_btn.setStyleSheet(_BTN_CSS)
        self._next_btn.clicked.connect(self._advance)
        cl.addSpacing(16)
        brow = QHBoxLayout()
        brow.addStretch(1)
        brow.addWidget(self._next_btn)
        cl.addLayout(brow)

        self._card = card
        self._show_step()

    def _advance(self):
        if self._i >= len(self._steps) - 1:
            self._on_finish()
            return
        self._i += 1
        self._show_step()

    def _target_rect(self):
        target = self._steps[self._i][0]
        if target is None:
            return None
        tl = target.mapTo(self.parentWidget(), QPoint(0, 0))
        return QRect(tl, target.size()).adjusted(-7, -7, 7, 7)

    def _show_step(self):
        if self._card is None:
            return
        _, title, body = self._steps[self._i]
        n = len(self._steps)
        self._step_lbl.setText(f"STEP {self._i + 1} OF {n}")
        self._title_lbl.setText(title)
        self._body_lbl.setText(body)
        # A word-wrapped QLabel under-reports its height through adjustSize(),
        # so the card came out too short and clipped the text. Size the body
        # label explicitly so the card grows to fit each step. Clear the
        # previous step's height clamp first, or heightForWidth() stays
        # pinned to that stale value.
        m = self._card.layout().contentsMargins()
        inner = self._card.width() - m.left() - m.right()
        self._body_lbl.setFixedWidth(inner)
        self._body_lbl.setMinimumHeight(0)
        self._body_lbl.setMaximumHeight(16777215)        # QWIDGETSIZE_MAX
        self._body_lbl.setFixedHeight(self._body_lbl.heightForWidth(inner))
        last = self._i == n - 1
        self._next_btn.setText("BEGIN" if last else "NEXT")
        self._next_btn.setFixedWidth(132 if last else 100)
        self.relayout()

    def relayout(self):
        """Place the explanatory card beside the spotlighted region — off
        the region itself wherever the geometry allows — then repaint."""
        if self._card is None:
            return
        self._card.adjustSize()
        cw, ch = self._card.width(), self._card.height()
        W, H = self.width(), self.height()
        tr = self._target_rect()
        if tr is None or W <= 0 or H <= 0:
            self._card.move(max(0, (W - cw) // 2), max(0, (H - ch) // 2))
            self.update()
            return

        gap = 22

        def clamp(v, lo, hi):
            return max(lo, min(v, max(lo, hi)))

        # Candidates: below / above / right / left of the region. The first
        # with zero overlap wins; otherwise the least-overlapping one (a
        # region as large as the EEG plot may leave no fully-clear spot).
        candidates = [
            (clamp(tr.center().x() - cw // 2, 0, W - cw),
             tr.y() + tr.height() + gap),
            (clamp(tr.center().x() - cw // 2, 0, W - cw),
             tr.y() - gap - ch),
            (tr.x() + tr.width() + gap,
             clamp(tr.center().y() - ch // 2, 0, H - ch)),
            (tr.x() - gap - cw,
             clamp(tr.center().y() - ch // 2, 0, H - ch)),
        ]
        best, best_overlap = None, None
        for px, py in candidates:
            px, py = clamp(px, 0, W - cw), clamp(py, 0, H - ch)
            inter = QRect(px, py, cw, ch).intersected(tr)
            iw, ih = inter.width(), inter.height()
            overlap = iw * ih if iw > 0 and ih > 0 else 0
            if best is None or overlap < best_overlap:
                best, best_overlap = (px, py), overlap
            if overlap == 0:
                break
        self._card.move(*best)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._card is not None:
            self.relayout()

    def paintEvent(self, event):
        if self._card is None:
            return
        p = QPainter(self)
        r = self.rect()
        tr = self._target_rect()
        if tr is None:
            p.fillRect(r, self._VEIL)
            return
        x, y, w, h = tr.x(), tr.y(), tr.width(), tr.height()
        W, H = r.width(), r.height()
        p.fillRect(QRect(0, 0, W, y), self._VEIL)
        p.fillRect(QRect(0, y + h, W, H - y - h), self._VEIL)
        p.fillRect(QRect(0, y, x, h), self._VEIL)
        p.fillRect(QRect(x + w, y, W - x - w, h), self._VEIL)
        pen = QPen(self._ACCENT)
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRect(x, y, w, h))


class BankViewer(QMainWindow):
    # Tutorial state — class defaults, overridden once a tutorial starts.
    _tutorial_active = False
    _overlay = None

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

        # The viewer is the functional tool — native system font throughout;
        # the serif app font is kept for the intro / branding screens only.
        self.setFont(QFontDatabase.systemFont(
            QFontDatabase.SystemFont.GeneralFont))
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

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._overlay is not None:
            self._overlay.setGeometry(self.centralWidget().rect())

    def start_tutorial(self, tutorial_sid, tutorial_domain="iiic"):
        """Render a dedicated example segment and run the coach-marks
        walkthrough over it; the real test begins only when it finishes."""
        self._tutorial_active = True
        self._render(tutorial_sid, tutorial_domain)
        steps = [
            (self._region_top, "Choosing an answer",
             "For each recording, choose the pattern that best matches "
             "what you see. Selecting an answer is what advances you "
             "to the next question.\n\n"
             "• Seizure:  an electrographic seizure\n"
             "• LPD / GPD:  lateralized or generalized periodic "
             "discharges\n"
             "• LRDA / GRDA:  lateralized or generalized rhythmic delta "
             "activity\n"
             "• Other:  a pattern fitting none of the above\n\n"
             "Some recordings instead ask only whether an epileptiform "
             "spike is present."),
            (self.spec_container, "The spectrogram",
             "A compressed time-frequency summary of the recording. A "
             "quick way to spot rhythmic or evolving activity before "
             "reading the waveforms.\n\n"
             "The four panels are brain regions: LL and RL are the left "
             "and right temporal chains; LP and RP are the left and "
             "right parasagittal chains."),
            (self.eeg_plot, "The EEG",
             "The raw tracings. Each row is a derivation between two "
             "electrodes. Pan through the recording with the ◀ ▶ buttons "
             "or the left / right arrow keys. "
             "Adjust the gain with the ▲ ▼ buttons "
             "or the up / down arrow keys."),
            (self._region_ctrl, "Display controls",
             "These change how the EEG is displayed — never your "
             "answer:\n\n"
             "• Montage: how electrode pairs are combined (bipolar, "
             "average, Laplacian)\n"
             "• Gain: vertical scale, in µV per division\n"
             "• Bandpass: keeps a frequency band, removing slow drift "
             "and high-frequency noise\n"
             "• Notch: removes 50 / 60 Hz mains interference\n"
             "• Window: how many seconds of EEG are shown at once"),
            (None, "Ready to begin",
             "That is the full interface. When you select Begin, the "
             "assessment starts and this example is replaced by the "
             "first recording."),
        ]
        self._overlay = TutorialOverlay(self.centralWidget(), steps,
                                        self._finish_tutorial)
        self._overlay.setGeometry(self.centralWidget().rect())
        self._overlay.show()
        self._overlay.raise_()

    def _finish_tutorial(self):
        if self._overlay is not None:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None
        self._tutorial_active = False
        self._load_segment(0)

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Top bar: answer-selection panel
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
        # example_box is kept but never shown — it is only the segment index
        # that answer-advance / prev / next drive.

        # Current-question readout — sits where the example selector was.
        self.seg_info_lbl = QLabel("")
        top.addWidget(self.seg_info_lbl)

        # Answer-selection panel: the test taker answers with the number
        # keys — IIIC segments 1-6 (pattern class), spike segments 1-2
        # (yes/no). Rendered display-only for now; selection capture and
        # question advancement are deliberately not wired yet.
        top.addStretch(1)
        self.answer_buttons = []
        for i in range(6):
            btn = AnswerButton("")
            btn.setMinimumWidth(135)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda _=False, idx=i: self._answer_chosen(idx))
            top.addWidget(btn)
            self.answer_buttons.append(btn)
        self._region_top = QWidget()
        self._region_top.setLayout(top)
        outer.addWidget(self._region_top)

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
            p.getAxis('left').setWidth(35)
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
        self._region_ctrl = QWidget()
        self._region_ctrl.setLayout(ctrl)
        outer.addWidget(self._region_ctrl)

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

    # Answer options: IIIC pattern classes on keys 1-6, spike yes/no on
    # keys 1-2. IIIC order matches the engine's IIIC task list.
    _IIIC_OPTIONS = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
    _SPIKE_OPTIONS = ["Spike", "No spike"]

    def _refresh_answer_panel(self, domain):
        opts = self._IIIC_OPTIONS if domain == "iiic" else self._SPIKE_OPTIONS
        for i, btn in enumerate(self.answer_buttons):
            btn.set_flash(False)
            if i < len(opts):
                btn.setText(f"{i + 1}  ·  {opts[i]}")
                btn.setVisible(True)
            else:
                btn.setVisible(False)

    _FLASH_MS = 85

    def _answer_chosen(self, choice):
        """Register an answer — from a number key or a button click. If the
        choice is valid for the current segment's domain, flash that button
        and advance to the next segment."""
        sid = self.seg_ids[self.current_idx]
        domain = self.seg_domains.get(sid, "iiic")
        n_opts = 6 if domain == "iiic" else 2
        if choice < n_opts:
            self._flash_and_advance(choice)

    def _flash_and_advance(self, choice):
        """Briefly outline the chosen answer button, then advance to the
        next segment (the flash is cleared by the next _refresh_answer_panel)."""
        self.answer_buttons[choice].set_flash(True)
        QTimer.singleShot(self._FLASH_MS, lambda: self._step_example(+1))

    def _load_segment(self, idx):
        self.current_idx = int(idx)
        sid = self.seg_ids[self.current_idx]
        self._render(sid, self.seg_domains.get(sid, "iiic"))

    def _render(self, sid, domain):
        """Load + draw an arbitrary bank segment. Also used for the
        tutorial example, which is not part of the test set."""
        self._cur_sid = sid
        self._cur_domain = domain
        try:
            grp = self.bank[f"{domain}/{sid}"]
            dset = grp["eeg30s"]
            self.data = np.asarray(dset, dtype=np.float32)
        except Exception as e:
            print(f"  WARN: seg {sid} unreadable ({e}); skipping", flush=True)
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
        self._refresh_answer_panel(domain)
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
        # UI shows the question number; the segment id (self._cur_sid) is
        # retained for internal use only, not displayed.
        qlabel = ("Tutorial example" if self._tutorial_active else f"Question {self.current_idx + 1}")
        self.seg_info_lbl.setText(
            f"{qlabel}:  "
            f"EEG Segment = [{self.t_start:.1f} - {self.t_start + self.window_s:.1f} s of "
            f"{self.duration:.1f} s],  Montage = {montage},  "
            f"Gain = {gain_uv:.0f} µV/div")

    @staticmethod
    def _extract_precomp(grp):
        """Pull (regs, sfreqs, stimes) out of an H5 group holding a
        precomputed regional spectrogram (`sdata`/`sfreqs`/`stimes`).
        Returns None if the group has no such datasets."""
        if not all(k in grp for k in ("sdata", "sfreqs", "stimes")):
            return None
        sdata = np.asarray(grp["sdata"], dtype=np.float32)
        sfreqs = np.asarray(grp["sfreqs"], dtype=np.float32).flatten()
        stimes = np.asarray(grp["stimes"], dtype=np.float32).flatten()
        n_freqs = len(sfreqs)
        # morgoth on-disk format is (n_times, n_freqs*4); transpose if needed
        if sdata.shape[0] != n_freqs * 4 and sdata.shape[1] == n_freqs * 4:
            sdata = sdata.T
        if sdata.shape[0] != n_freqs * 4:
            return None
        return ({'LL': sdata[0 * n_freqs:1 * n_freqs],
                 'RL': sdata[1 * n_freqs:2 * n_freqs],
                 'LP': sdata[2 * n_freqs:3 * n_freqs],
                 'RP': sdata[3 * n_freqs:4 * n_freqs]},
                sfreqs, stimes)

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
        # Source preference for the precomputed regional spectrogram:
        #   1. the full ~10-min `sdata` stored inside the main bank group
        #   2. a side spec file (data/eeg_bank_spec.h5), if present
        #   3. on-the-fly compute from the 30-s clip (fallback)
        sid = self._cur_sid
        domain = self._cur_domain
        precomp = None
        try:
            precomp = self._extract_precomp(self.bank[f"{domain}/{sid}"])
        except Exception:
            precomp = None
        if precomp is None and self.spec_file is not None:
            try:
                precomp = self._extract_precomp(self.spec_file[f"segments/{sid}"])
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
            # Small padding so each spectrogram sits inside its panel with a
            # margin instead of flush to the edges (which looks cut off).
            plot.setXRange(t0, t1, padding=0.05)
            plot.setYRange(0.0, float(n_freqs), padding=0.05)
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

    # Number keys 1-6 select an answer (IIIC: 6-way; spike: keys 1-2).
    _ANSWER_KEYS = (Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3,
                    Qt.Key.Key_4, Qt.Key.Key_5, Qt.Key.Key_6)

    # Keys we intercept application-wide so combo boxes don't eat them
    _HOTKEYS = {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up,
                Qt.Key.Key_Down, Qt.Key.Key_Control, Qt.Key.Key_N,
                Qt.Key.Key_P, *_ANSWER_KEYS}

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
          1-6    : select answer (IIIC 6-way; spike 1-2) → next segment
        """
        if self._tutorial_active:
            return                       # viewer is inert during the tutorial
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
        elif key in self._ANSWER_KEYS:
            # Answer keys 1-6 (IIIC) / 1-2 (spike) — same path as a click
            # on the corresponding answer button.
            self._answer_chosen(self._ANSWER_KEYS.index(key))
        else:
            super().keyPressEvent(event)


# ──────────────────────── landing page ────────────────────────

class LandingPage(QWidget):
    """CORTEX landing screen — text over a static EEG backdrop."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CORTEX")
        self.resize(980, 660)

        # Static backdrop: one real bank segment, drawn once (no animation
        # timer, so idle CPU stays at zero).
        self._eeg = self._load_backdrop_eeg()

        root = QVBoxLayout(self)
        root.setContentsMargins(60, 0, 60, 0)
        root.addStretch(2)

        # Wordmark — large + bold, dominating the page
        title = QLabel("CORTEX")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tf = QFont()
        tf.setPointSize(84)
        tf.setWeight(QFont.Weight.Bold)
        tf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 3)
        title.setFont(tf)
        title.setStyleSheet("color: #eef1f5; background: transparent;")
        root.addWidget(title)

        # Expansion
        subtitle = QLabel(
            "Continuous Optimization Response Testing for EEG eXpertise")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf = QFont()
        sf.setPointSize(17)
        sf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        subtitle.setFont(sf)
        subtitle.setStyleSheet("color: #868b96; background: transparent;")
        root.addSpacing(12)
        root.addWidget(subtitle)

        # Begin button — minimal outlined
        root.addSpacing(56)
        self.begin_btn = QPushButton("BEGIN ASSESSMENT")
        self.begin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.begin_btn.setFixedSize(252, 50)
        bf = QFont()
        bf.setPointSize(11)
        bf.setWeight(QFont.Weight.Medium)
        bf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        self.begin_btn.setFont(bf)
        self.begin_btn.setStyleSheet(
            "QPushButton { background-color: transparent; color: #dde0e6;"
            " border: 1px solid #454a55; border-radius: 0px; }"
            " QPushButton:hover { border-color: #8a8f9b; color: #ffffff;"
            " background-color: #181a20; }"
            " QPushButton:pressed { background-color: #0e0f13; }"
            " QPushButton:disabled { color: #585c66;"
            " border-color: #2a2d35; }")
        root.addLayout(self._centered(self.begin_btn))

        root.addStretch(3)

        # Credit
        credit = QLabel(
            "Developed by:\nElijah W. Keldsen and M. Brandon Westover")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cf = QFont()
        cf.setPointSize(11)
        cf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        credit.setFont(cf)
        credit.setStyleSheet("color: #5c606a; background: transparent;")
        root.addWidget(credit)

        support = QLabel(
            "A project supported by the Clinical Data Animations Center")
        support.setAlignment(Qt.AlignmentFlag.AlignCenter)
        support.setFont(cf)
        support.setStyleSheet("color: #5c606a; background: transparent;")
        root.addSpacing(8)
        root.addWidget(support)
        root.addSpacing(34)

    @staticmethod
    def _load_backdrop_eeg():
        """One segment of bank EEG — downsampled + per-channel normalised —
        for the static backdrop. Returns an (n_ch, n) array or None."""
        try:
            raw = None
            with h5py.File(BANK_PATH, "r") as f:
                for domain in ("iiic", "spike"):
                    if domain in f and len(f[domain]):
                        sid = next(iter(f[domain]))
                        raw = np.asarray(f[domain][sid]["eeg30s"],
                                         dtype=np.float32)
                        break
            if raw is None:
                return None
        except Exception:
            return None
        eeg = raw[:16, ::3]
        eeg = eeg - eeg.mean(axis=1, keepdims=True)
        sd = eeg.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        return (eeg / sd).astype(np.float32)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        p.fillRect(r, QColor("#0b0d12"))
        if self._eeg is not None:
            self._paint_eeg(p, r)
        p.fillRect(r, QColor(11, 13, 18, 60))   # light scrim — depth + contrast

    def _paint_eeg(self, p, r):
        eeg = self._eeg
        n_ch, n = eeg.shape
        W, H = r.width(), r.height()
        span = 540
        idx = np.arange(span) % n
        xs = np.linspace(0.0, float(W), span)
        amp = (H / n_ch) * 0.40
        pen = QPen(QColor(46, 66, 94))
        pen.setWidthF(1.1)
        p.setPen(pen)
        for ch in range(n_ch):
            ys = (H * (ch + 0.5) / n_ch) - eeg[ch, idx] * amp
            path = QPainterPath()
            path.moveTo(float(xs[0]), float(ys[0]))
            for i in range(1, span):
                path.lineTo(float(xs[i]), float(ys[i]))
            p.drawPath(path)

    @staticmethod
    def _centered(widget):
        """Wrap a widget in an HBox with side stretches so it sits centred."""
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(widget)
        row.addStretch(1)
        return row


# ─────────────────── consent + registration pages ───────────────────

# Shared styling — same dark palette + minimal outlined buttons as the
# landing page, so the intro screens read as one piece.
_PAGE_BG = "#0b0d12"
_BTN_CSS = (
    "QPushButton { background-color: transparent; color: #dde0e6;"
    " border: 1px solid #454a55; border-radius: 0px; }"
    " QPushButton:hover { border-color: #8a8f9b; color: #ffffff;"
    " background-color: #181a20; }"
    " QPushButton:pressed { background-color: #0e0f13; }"
    " QPushButton:disabled { color: #585c66; border-color: #2a2d35; }"
)


def _btn_font():
    f = QFont()
    f.setPointSize(11)
    f.setWeight(QFont.Weight.Medium)
    f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
    return f


def _hcenter(widget):
    """An HBox with side stretches so `widget` sits horizontally centred."""
    row = QHBoxLayout()
    row.addStretch(1)
    row.addWidget(widget)
    row.addStretch(1)
    return row


class ConsentPage(QWidget):
    """Overview + consent screen, shown between the landing page and the
    registration form. `accept_btn` advances; Decline shows a thank-you."""

    _OVERVIEW = (
        "This is an adaptive assessment of EEG interpretation. The number "
        "of recordings you review is not fixed — the assessment adjusts to "
        "your responses and ends once it has gathered enough information.\n\n"
        "For each recording you will make a brief judgment. From your "
        "responses the assessment builds a profile of how you read these "
        "studies and classifies your level of expertise.\n\n"
        "Answer each recording as you would in routine clinical practice."
    )
    # Placeholder consent wording — replace with IRB-approved text.
    _TERMS = (
        "By selecting “I Accept” you confirm that you are "
        "participating voluntarily, that the information you provide and "
        "your responses may be recorded and used for research and "
        "credentialing purposes, and that you may withdraw at any time. "
        "Select “Decline” if you do not wish to participate."
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CORTEX")
        self.resize(980, 660)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"ConsentPage {{ background-color: {_PAGE_BG}; }}")

        self._stack = QStackedWidget()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)
        self._stack.addWidget(self._build_consent())   # index 0
        self._stack.addWidget(self._build_thanks())    # index 1

    def _build_consent(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(2)

        heading = QLabel("Before You Begin")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hf = QFont()
        hf.setPointSize(30)
        hf.setWeight(QFont.Weight.DemiBold)
        hf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        heading.setFont(hf)
        heading.setStyleSheet("color: #eef1f5; background: transparent;")
        root.addWidget(heading)

        overview = QLabel(self._OVERVIEW)
        overview.setWordWrap(True)
        overview.setFixedWidth(640)
        overview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        of = QFont()
        of.setPointSize(14)
        overview.setFont(of)
        overview.setStyleSheet("color: #aab0ba; background: transparent;")
        root.addSpacing(30)
        root.addLayout(_hcenter(overview))

        terms = QLabel(self._TERMS)
        terms.setWordWrap(True)
        terms.setFixedWidth(640)
        terms.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tfont = QFont()
        tfont.setPointSize(11)
        terms.setFont(tfont)
        terms.setStyleSheet("color: #767b87; background: transparent;")
        root.addSpacing(30)
        root.addLayout(_hcenter(terms))

        root.addSpacing(40)
        self.decline_btn = QPushButton("DECLINE")
        self.accept_btn = QPushButton("I ACCEPT")
        btns = QHBoxLayout()
        btns.addStretch(1)
        for b in (self.decline_btn, self.accept_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedSize(190, 50)
            b.setFont(_btn_font())
            b.setStyleSheet(_BTN_CSS)
        btns.addWidget(self.decline_btn)
        btns.addSpacing(22)
        btns.addWidget(self.accept_btn)
        btns.addStretch(1)
        root.addLayout(btns)
        root.addStretch(3)

        self.decline_btn.clicked.connect(
            lambda: self._stack.setCurrentIndex(1))
        return page

    def _build_thanks(self):
        page = QWidget()
        root = QVBoxLayout(page)
        root.addStretch(1)
        msg = QLabel("Thank you for your time.")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mf = QFont()
        mf.setPointSize(24)
        mf.setWeight(QFont.Weight.DemiBold)
        msg.setFont(mf)
        msg.setStyleSheet("color: #c8ccd3; background: transparent;")
        root.addWidget(msg)
        sub = QLabel("You have chosen not to participate — "
                     "you may close this window.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf = QFont()
        sf.setPointSize(12)
        sub.setFont(sf)
        sub.setStyleSheet("color: #767b87; background: transparent;")
        root.addSpacing(14)
        root.addWidget(sub)
        root.addStretch(1)
        return page


class RegistrationPage(QWidget):
    """Participant registration form. `commit()` validates the entry and
    appends it to results/registrations.csv; `continue_btn` advances."""

    _EXPERTISE = ["— select —", "Medical student", "Resident", "Nurse",
                  "Fellow", "Neurologist", "Epileptologist", "Other"]
    _FIELD_CSS = (
        "QLineEdit { background-color: #15171c; color: #dde0e6;"
        " border: 1px solid #3a3d45; border-radius: 0px; padding: 4px 8px; }"
        " QLineEdit:focus { border-color: #8a8f9b; }"
    )
    _COMBO_CSS = (
        "QComboBox { background-color: #15171c; color: #dde0e6;"
        " border: 1px solid #3a3d45; border-radius: 0px; padding: 4px 8px; }"
        " QComboBox:focus { border-color: #8a8f9b; }"
        " QComboBox QAbstractItemView { background-color: #15171c;"
        " color: #dde0e6; selection-background-color: #2c2f36; }"
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CORTEX")
        self.resize(980, 660)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"RegistrationPage {{ background-color: {_PAGE_BG}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(2)

        heading = QLabel("Participant Information")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hf = QFont()
        hf.setPointSize(26)
        hf.setWeight(QFont.Weight.DemiBold)
        hf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        heading.setFont(hf)
        heading.setStyleSheet("color: #eef1f5; background: transparent;")
        root.addWidget(heading)

        self.f_name = QLineEdit()
        self.f_age = QLineEdit()
        self.f_gender = QLineEdit()
        self.f_inst = QLineEdit()
        self.f_email = QLineEdit()
        self.f_creds = QLineEdit()
        self.f_expertise = QComboBox()
        self.f_expertise.addItems(self._EXPERTISE)

        form_box = QWidget()
        form_box.setFixedWidth(470)
        form = QFormLayout(form_box)
        form.setSpacing(13)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for label_text, field in (
            ("Name", self.f_name), ("Age", self.f_age),
            ("Gender", self.f_gender),
            ("Institutional affiliation", self.f_inst),
            ("Email", self.f_email), ("Expertise", self.f_expertise),
            ("Credentials", self.f_creds),
        ):
            field.setFixedHeight(34)
            field.setStyleSheet(
                self._COMBO_CSS if isinstance(field, QComboBox)
                else self._FIELD_CSS)
            form.addRow(self._flabel(label_text), field)
        root.addSpacing(26)
        root.addLayout(_hcenter(form_box))

        self.msg = QLabel("")
        self.msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mf = QFont()
        mf.setPointSize(11)
        self.msg.setFont(mf)
        self.msg.setStyleSheet("color: #d8806a; background: transparent;")
        root.addSpacing(10)
        root.addWidget(self.msg)

        root.addSpacing(16)
        self.continue_btn = QPushButton("CONTINUE")
        self.continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.continue_btn.setFixedSize(220, 50)
        self.continue_btn.setFont(_btn_font())
        self.continue_btn.setStyleSheet(_BTN_CSS)
        root.addLayout(_hcenter(self.continue_btn))
        root.addStretch(3)

    @staticmethod
    def _flabel(text):
        lbl = QLabel(text)
        f = QFont()
        f.setPointSize(11)
        lbl.setFont(f)
        lbl.setStyleSheet("color: #9aa0ab; background: transparent;")
        return lbl

    def commit(self):
        """Validate + save the registration. Returns True on success."""
        name = self.f_name.text().strip()
        email = self.f_email.text().strip()
        if not name or not email:
            self.msg.setText("Please enter at least your name and email.")
            return False
        row = {
            "timestamp_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "name": name,
            "age": self.f_age.text().strip(),
            "gender": self.f_gender.text().strip(),
            "institution": self.f_inst.text().strip(),
            "email": email,
            "expertise": (self.f_expertise.currentText()
                          if self.f_expertise.currentIndex() > 0 else ""),
            "credentials": self.f_creds.text().strip(),
        }
        try:
            self._save_row(row)
        except Exception as e:
            self.msg.setText(f"Could not save registration: {e}")
            return False
        return True

    @staticmethod
    def _save_row(row):
        path = (Path(__file__).resolve().parent.parent / "results"
                / "registrations.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if is_new:
                w.writeheader()
            w.writerow(row)


def main():
    # Stop Qt's macOS Cmd/Ctrl swap so the physical Control key actually
    # produces Qt.Key.Key_Control (matching morgoth's behavior on
    # Linux/Windows). Must be set before QApplication is constructed.
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_MacDontSwapCtrlAndMeta, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Palatino"))            # app-wide serif typeface

    # Page flow: landing → consent → registration → viewer. Each page is
    # its own window; `goto` shows the next at the previous one's geometry
    # then closes the previous (so the app never hits zero windows and
    # quits mid-transition). `state` holds the live reference.
    state = {}

    def goto(widget):
        prev = state.get("page")
        if prev is not None:
            widget.setGeometry(prev.geometry())
        state["page"] = widget
        widget.show()
        if prev is not None and prev is not widget:
            prev.close()

    def open_consent():
        consent = ConsentPage()
        consent.accept_btn.clicked.connect(open_registration)
        goto(consent)

    def open_registration():
        reg = RegistrationPage()

        def on_continue():
            if reg.commit():
                open_viewer(reg)

        reg.continue_btn.clicked.connect(on_continue)
        goto(reg)

    def open_viewer(reg):
        reg.continue_btn.setEnabled(False)
        reg.continue_btn.setText("LOADING…")
        app.processEvents()                  # let the button repaint first
        seg_ids, seg_meta, seg_domains = pick_examples(n_total=10)
        if not seg_ids:
            raise SystemExit("No segments to display")
        win = BankViewer(seg_ids, seg_meta, seg_domains)
        # Install app-wide event filter so combo boxes don't swallow arrow
        # keys / Ctrl before we see them.
        app.installEventFilter(win)
        win.show()
        reg.close()
        state["page"] = win
        # Coach-marks walkthrough on a dedicated example — a bank IIIC
        # segment that is not one of the scored test items — then the test.
        test_ids = set(seg_ids)
        tut_sid = next((s for s in sorted(seg_domains)
                        if seg_domains[s] == "iiic" and s not in test_ids),
                       None)
        if tut_sid is not None:
            app.processEvents()              # let the window lay out first
            win.start_tutorial(tut_sid)

    landing = LandingPage()
    landing.begin_btn.clicked.connect(open_consent)
    goto(landing)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
