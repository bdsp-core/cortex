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
import logging
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# Allow concurrent read while bank builds are still writing.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Engine reproducibility contract — pin BLAS single-thread before numpy.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import (
    Qt, QEvent, QTimer, QRect, QPoint, QThread, QObject, pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (QColor, QFont, QFontDatabase, QPainter, QPainterPath,
                         QPen, QPixmap)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QFrame, QPushButton, QLabel, QCheckBox, QSplitter,
    QGridLayout, QLineEdit, QFormLayout, QStackedWidget, QProgressBar,
)
from scipy import signal as sig


# Frozen PyInstaller bundles: __file__ for the ENTRY script resolves to
# <MEIPASS>/<entry-script-name>.py, so .parent.parent goes one level
# ABOVE the data unpack root — into the .app's Contents/ rather than
# Contents/Frameworks/. Anchor on sys._MEIPASS so bundled data files
# resolve correctly. (Dev mode: __file__ is scripts/eeg_bank_viewer.py
# and parent.parent is the repo root, which is what we want.)
if getattr(sys, "frozen", False):
    _REPO = Path(sys._MEIPASS)
else:
    _REPO = Path(__file__).resolve().parent.parent
BANK_PATH = _REPO / "data" / "eeg_bank.h5"
SPEC_PATH = _REPO / "data" / "eeg_bank_spec.h5"   # precomputed 10-min spectrograms

# v1.3.8 — IIIC viewing geometry. The 30s `eeg30s` clip is the CENTRAL 30s of
# the 10-min spectrogram (stimes ~2..598s, center ~300s); the expert-labeled
# epoch is the central 10s of the clip == clip-local [10, 20] s. The viewer
# defaults the IIIC window to that labeled epoch, pans in full-window 10s steps
# over {0-10, 10-20, 20-30} s, and boxes the labeled epoch on the EEG while
# marking the 30s-clip extent on the 10-min spectrogram.
IIIC_CLIP_S = 30.0          # IIIC eeg30s clip length (s)
IIIC_LABEL_START_S = 10.0   # labeled epoch start within the clip (s)
IIIC_LABEL_LEN_S = 10.0     # labeled epoch length (s)


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

    def __init__(self, controller, session_id, tutorial_sid, recorder=None):
        super().__init__()
        self.setWindowTitle("CORTEX")
        self.resize(1500, 950)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")

        self.bank = h5py.File(BANK_PATH, "r")
        self.spec_file = None
        if SPEC_PATH.exists():
            try:
                self.spec_file = h5py.File(SPEC_PATH, "r")
            except Exception as e:
                logger.warning("could not open spec file %s: %s",
                               SPEC_PATH, e)
                self.spec_file = None
        self.controller = controller
        self.session_id = session_id
        self.tutorial_sid = tutorial_sid
        self.recorder = recorder
        self.t_start = 0.0
        self.window_s = 10.0

        # live-test state
        self._awaiting_answer = False
        self._selected_choice = None
        self._answer_changes = 0
        self._interaction = []          # per-question interaction trace
        self._rt_t0 = None              # perf_counter at question paint
        self._cur_trial = -1
        self._cur_seg = None
        self._cur_k = None
        self.gui_trial_log = []         # per-question GUI metadata records
        self.session_result = None
        self._session_over = False
        self._n_correct = 0
        self._n_answered = 0

        # The viewer is the functional tool — native system font throughout;
        # the serif app font is kept for the intro / branding screens only.
        self.setFont(QFontDatabase.systemFont(
            QFontDatabase.SystemFont.GeneralFont))
        self._build_ui()

        # Engine -> GUI signals (queued onto this GUI thread).
        self.controller.itemReady.connect(self.show_item)
        self.controller.trialDone.connect(self._on_trial_done)
        self.controller.sessionComplete.connect(self._on_session_complete)
        self.controller.sessionFailed.connect(self._on_session_failed)

    def closeEvent(self, ev):
        # End the engine session cleanly if the window closes mid-test.
        if self.session_result is None:
            try:
                self.controller.abort()
            except Exception:
                pass
        if self.recorder is not None:
            try:
                self.recorder.close()
            except Exception:
                pass
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
             "what you see, then press Confirm. You can change your "
             "selection freely before confirming.\n\n"
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
             "right parasagittal chains.\n\n"
             "The spectrogram appears only for these pattern-"
             "classification recordings. The spike-present questions "
             "show the EEG on its own, with no spectrogram panel."),
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
        # Hand control to the engine — the first itemReady follows shortly.
        self.seg_info_lbl.setText("Preparing the assessment…")
        self.controller.start()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # Top bar: current-question readout + answer panel.
        top = QHBoxLayout()
        self.seg_info_lbl = QLabel("")
        top.addWidget(self.seg_info_lbl)

        # Answer panel: 6 IIIC pattern-class options. A click or number key
        # SELECTS (highlights) an option; the Confirm button or Enter commits.
        # The selection can be changed freely before confirming.
        top.addStretch(1)
        self.answer_buttons = []
        for i in range(6):
            btn = AnswerButton("")
            btn.setMinimumWidth(135)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda _=False, idx=i: self._select_answer(idx))
            top.addWidget(btn)
            self.answer_buttons.append(btn)
        self.confirm_btn = QPushButton("Confirm ⏎")
        self.confirm_btn.setMinimumWidth(120)
        self.confirm_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self._confirm_answer)
        top.addSpacing(14)
        top.addWidget(self.confirm_btn)
        self._region_top = QWidget()
        self._region_top.setLayout(top)
        outer.addWidget(self._region_top)

        # v1.3.8: instruction banner for IIIC recordings — tells the candidate
        # the red box marks the region scored by the panel and that panning
        # reveals neighbouring context. Hidden for spike (no box, full clip).
        self.classify_hint_lbl = QLabel(
            "Classify the pattern found within the red box. "
            "Pan left or right to gain context.")
        self.classify_hint_lbl.setStyleSheet(
            "color: #b03030; font-weight: 600; padding: 2px 0;")
        self.classify_hint_lbl.setVisible(False)
        outer.addWidget(self.classify_hint_lbl)

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
        self.pan_l_btn = QPushButton("◀ Pan")
        self.pan_l_btn.clicked.connect(lambda: self._pan(-1))
        ctrl.addWidget(self.pan_l_btn)
        self.pan_r_btn = QPushButton("Pan ▶")
        self.pan_r_btn.clicked.connect(lambda: self._pan(+1))
        ctrl.addWidget(self.pan_r_btn)
        # Interaction-trace logging — every display change during a question.
        for _box, _name in ((self.montage_box, "montage"),
                            (self.gain_box, "gain"), (self.bp_box, "bandpass"),
                            (self.notch_box, "notch"), (self.win_box, "window")):
            _box.currentTextChanged.connect(
                lambda v, n=_name: self._log_interaction(n, v))
        self.spec_cb.toggled.connect(
            lambda on: self._log_interaction("spectrogram", bool(on)))
        self._region_ctrl = QWidget()
        self._region_ctrl.setLayout(ctrl)
        outer.addWidget(self._region_ctrl)

    def _on_window_change(self, _):
        self.window_s = float(self.win_box.currentText())
        self._redraw()

    def _on_spec_toggle(self, on):
        # v1.2.5: a spike segment never shows the spectrogram. If somehow
        # the checkbox is toggled while a spike segment is showing (the
        # checkbox is disabled by _redraw, but defensively belt-and-braces),
        # keep the panel hidden and short-circuit.
        family = getattr(self, "_cur_family", "iiic")
        if family == "spike":
            self.spec_container.setVisible(False)
            return
        self.spec_container.setVisible(on)
        if on:
            self._redraw()

    def _pan(self, direction):
        # v1.3.8: pan by a full (non-overlapping) window so the IIIC default
        # 10s window steps over the three context windows {0-10,10-20,20-30}s.
        step = self.window_s
        new = self.t_start + direction * step
        max_t = max(0.0, self.duration - self.window_s)
        self.t_start = float(np.clip(new, 0.0, max_t))
        self._log_interaction("pan", direction)
        self._redraw()

    # IIIC pattern-class options on keys 1-6 — order matches the engine's
    # IIIC task list (sz, lpd, gpd, lrda, grda, iic).
    _IIIC_OPTIONS = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
    # Phase-9 K=7: spike Yes/No options on keys 1-2.
    _SPIKE_OPTIONS = ["Yes (spike present)", "No (no spike)"]

    def _refresh_answer_panel(self):
        """Reset the answer buttons based on the current segment family:
        spike → 2 buttons (Yes/No), IIIC → 6 buttons (pattern-class)."""
        family = getattr(self, "_cur_family", "iiic")
        if family == "spike":
            options = self._SPIKE_OPTIONS
        else:
            options = self._IIIC_OPTIONS
        n_opt = len(options)
        for i, btn in enumerate(self.answer_buttons):
            btn.set_flash(False)
            if i < n_opt:
                btn.setText(f"{i + 1}  ·  {options[i]}")
                btn.setVisible(True)
            else:
                btn.setText("")
                btn.setVisible(False)

    # ─────────────── engine-driven question flow ───────────────

    def show_item(self, item):
        """Slot for SessionController.itemReady — render the engine's chosen
        segment and arm the answer panel.

        Phase-9 K=7: dispatch the answer-panel and EEG renderer on the
        segment's family (spike vs iiic) via the controller's K=7 inputs.
        Spike segments → Yes/No (2-button) UI + 10s EEG; IIIC segments
        → 6-button pattern-class UI + 30s EEG (unchanged from K=6).
        """
        self._cur_trial = int(item["trial_index"])
        self._cur_seg = int(item["seg_id"])
        self._cur_k = int(item["task_k"])
        self._selected_choice = None
        self._answer_changes = 0
        self._interaction = []
        self._awaiting_answer = True
        self.confirm_btn.setEnabled(False)
        # Phase-9 K=7: determine the family from the engine's K=7 inputs.
        # The K=7 inputs expose .family(seg_id) → "spike" | "iiic".
        try:
            self._cur_family = self.controller.inputs.family(self._cur_seg)
        except Exception:
            # Back-compat: pre-K=7 inputs default to IIIC.
            self._cur_family = "iiic"
        self._render(self._cur_seg, self._cur_family)
        # Start the reaction-time clock AFTER the segment has painted.
        self._rt_t0 = None
        QTimer.singleShot(0, self._start_rt_clock)

    def _start_rt_clock(self):
        self._rt_t0 = time.perf_counter()

    def _log_interaction(self, action, detail):
        """Append a timestamped display action to the question's trace."""
        if not self._awaiting_answer or self._rt_t0 is None:
            return
        self._interaction.append({
            "t_ms": round((time.perf_counter() - self._rt_t0) * 1000.0, 1),
            "action": action, "detail": detail})

    def _select_answer(self, choice):
        """Select (highlight) an answer option without advancing. Validates
        the choice against the current family's option count (2 for spike;
        6 for IIIC). The choice can be changed freely until Confirm."""
        family = getattr(self, "_cur_family", "iiic")
        options = (self._SPIKE_OPTIONS if family == "spike"
                   else self._IIIC_OPTIONS)
        if not self._awaiting_answer or choice >= len(options):
            return
        if self._selected_choice is not None and self._selected_choice != choice:
            self._answer_changes += 1
        self._selected_choice = choice
        for i, btn in enumerate(self.answer_buttons):
            btn.set_flash(i == choice)
        self.confirm_btn.setEnabled(True)
        self._log_interaction("select", options[choice])

    def _confirm_answer(self):
        """Commit the selected answer — record per-question GUI metadata
        and hand the family-appropriate raw value to the engine.

        Phase-9 K=7 raw-value mapping (preserves the existing _y_source
        contract `int(raw == k)` in session_controller.EngineWorker):

          IIIC family — engine asks task k ∈ {1..6} (sz, lpd, gpd, lrda,
            grda, iic in the K=7 task list). User clicks button i ∈ {0..5}
            (Seizure, LPD, GPD, LRDA, GRDA, Other). Raw = i + 1 so that
            i==0→raw=1==k(sz) iff user-said-seizure and engine-asked-sz.

          Spike family — engine asks task k = 0 (spike). User clicks
            button 0 (Yes) or 1 (No). Raw = 0 (Yes → matches k=0 → y=1)
            or raw = -1 (No → does not match k=0 → y=0).
        """
        if not self._awaiting_answer or self._selected_choice is None:
            return
        rt_ms = (round((time.perf_counter() - self._rt_t0) * 1000.0, 1)
                 if self._rt_t0 is not None else None)
        choice = self._selected_choice
        family = getattr(self, "_cur_family", "iiic")
        if family == "spike":
            options = self._SPIKE_OPTIONS
            # Yes → 0 (matches engine task k=0); No → -1 (never matches)
            raw = 0 if choice == 0 else -1
        else:
            options = self._IIIC_OPTIONS
            # IIIC button i ∈ {0..5} → engine task k ∈ {1..6} (spike=0 takes slot 0)
            raw = choice + 1
        self._awaiting_answer = False
        self.confirm_btn.setEnabled(False)
        self.gui_trial_log.append({
            "trial_index": self._cur_trial,
            "seg_id": self._cur_seg,
            "task_k": self._cur_k,
            "family": family,
            "response_button": choice,
            "response_raw": raw,
            "response_label": options[choice],
            "reaction_time_ms": rt_ms,
            "answer_changes": self._answer_changes,
            "montage": self.montage_box.currentText(),
            "gain_uv": float(self.gain_box.currentText()),
            "bandpass": self.bp_box.currentText(),
            "notch": self.notch_box.currentText(),
            "window_s": float(self.win_box.currentText()),
            "pan_t_start": float(self.t_start),
            "interaction": list(self._interaction),
        })
        self.seg_info_lbl.setText("Selecting the next recording…")
        self.controller.submit_answer(raw)

    def _on_trial_done(self, telemetry):
        """Slot for SessionController.trialDone — merge the engine telemetry
        with this trial's GUI metadata, tally accuracy, and persist."""
        tix = telemetry.get("trial_index")
        gui = next((g for g in self.gui_trial_log
                    if g.get("trial_index") == tix), None)
        if gui is not None:
            self._n_answered += 1
            # v1.3.6: spike is BINARY — pattern_class_true is the placeholder
            # "spike", so the IIIC label-match below never fires for it. Grade
            # the spike response against sign(s_mean) (cortex_storage._spike_correct).
            if telemetry.get("task_code") == "spike":
                from cortex_storage import _spike_correct
                correct = _spike_correct(telemetry.get("response_y"),
                                         telemetry.get("s_mean"))
            else:
                correct = (str(gui.get("response_label", "")).lower()
                           == telemetry.get("pattern_class_true"))
            if correct:
                self._n_correct += 1
        if self.recorder is not None:
            self.recorder.write_trial(telemetry, gui)

    def _on_session_complete(self, result):
        """Slot for SessionController.sessionComplete.

        v1.1.4: finalize() runs on a background thread (_FinalizeWorker)
        so the UI stays responsive while files are written + (if the
        participant opted in to visualizations) the slow MP4s render.
        ComputingResultsPage is shown IMMEDIATELY to acknowledge the
        test ending; ResultsScreen swaps in once the worker emits
        `finished`.

        Pre-v1.1.4 the finalize was synchronous and the GUI was frozen
        for ~3 minutes on the opt-in path."""
        self._awaiting_answer = False
        self._session_over = True
        self.session_result = result
        if getattr(result, "aborted", False):
            return                       # window is closing — no transition

        opt_in = False
        if self.recorder is not None:
            opt_in = bool(getattr(self.recorder, "render_videos", False))
        # Show the transition page first, then yield so it paints
        # before we spin up the worker thread (avoids a brief blank
        # frame on slow machines).
        computing_page = ComputingResultsPage(opt_in=opt_in)
        self.setCentralWidget(computing_page)
        QApplication.processEvents()

        if self.recorder is None:
            # No recorder ⇒ nothing to finalize; transition straight
            # through to results on the next event-loop tick.
            QTimer.singleShot(0, lambda: self._show_results(result))
            return

        self._fin_thread = QThread(self)
        self._fin_worker = _FinalizeWorker(self.recorder, result)
        self._fin_worker.moveToThread(self._fin_thread)
        self._fin_thread.started.connect(self._fin_worker.run)
        # v1.2.9: live progress → the computing page's determinate bar.
        self._fin_worker.progress.connect(computing_page.set_progress)
        self._fin_worker.finished.connect(self._on_finalize_done)
        self._fin_worker.failed.connect(self._on_finalize_failed)
        # Tear the thread down once the worker emits either terminal.
        self._fin_worker.finished.connect(self._fin_thread.quit)
        self._fin_worker.failed.connect(self._fin_thread.quit)
        self._fin_thread.finished.connect(self._fin_worker.deleteLater)
        self._fin_thread.finished.connect(self._fin_thread.deleteLater)
        self._fin_thread.start()

    @pyqtSlot(object)
    def _on_finalize_done(self, result):
        """Worker emitted success — swap to ResultsScreen on the GUI
        thread."""
        self._show_results(result)

    @pyqtSlot(str)
    def _on_finalize_failed(self, err_msg):
        """Worker raised — log + still show results so the participant
        gets a verdict even when persistence partially failed."""
        logging.getLogger(__name__).warning(
            "background finalize failed: %s; showing results anyway",
            err_msg)
        self._show_results(self.session_result)

    def _show_results(self, result):
        # v1.1.5: hand the session dir to ResultsScreen so its
        # "Show results folder" button can open the right path.
        # getattr-with-None default: handles the no-recorder path
        # (test fixtures, audit paths) AND test stub recorders that
        # don't model the full SessionRecorder surface. ResultsScreen
        # hides the button when session_dir is None.
        session_dir = getattr(self.recorder, "dir", None)
        self.setCentralWidget(
            ResultsScreen(result, self._n_correct, self._n_answered,
                          session_dir=session_dir))

    def _on_session_failed(self, msg):
        """Slot for SessionController.sessionFailed."""
        self._awaiting_answer = False
        self._session_over = True
        self.seg_info_lbl.setText(f"Engine error — {msg}")

    def _render(self, sid, domain):
        """Load + draw an arbitrary bank segment. Also used for the
        tutorial example, which is not part of the test set.

        Phase-9 K=7: auto-detect EEG dataset name + sampling rate based on
        family. IIIC segments use `eeg30s` (20, 6000) @ 200 Hz (unchanged);
        spike segments use `eeg10s` (20, 1281) @ 128 Hz per the SN1 spike-
        paper methodology (Jing et al — 10s windows at 128 Hz × 20 ch).
        """
        self._cur_sid = sid
        self._cur_domain = domain
        try:
            grp = self.bank[f"{domain}/{sid}"]
            # K=7 spike family stores `eeg10s` (10s @ 128 Hz); IIIC family
            # stores `eeg30s` (30s @ 200 Hz). Auto-detect on the dataset name.
            if "eeg10s" in grp:
                dset = grp["eeg10s"]
                fs_default = 128.0
            else:
                dset = grp["eeg30s"]
                fs_default = 200.0
            self.data = np.asarray(dset, dtype=np.float32)
        except Exception as e:
            print(f"  WARN: seg {sid} unreadable ({e}); skipping", flush=True)
            self.data = None
            return
        self.fs = float(dset.attrs.get("fs_hz", fs_default))
        ch_attr = dset.attrs.get("channel_names")
        if ch_attr is not None:
            self.channel_names = [b.decode("utf-8", errors="ignore") if isinstance(b, (bytes, np.bytes_)) else str(b)
                                for b in np.asarray(ch_attr)]
        else:
            self.channel_names = CHANNELS_19[:self.data.shape[0]]
        self.duration = self.data.shape[1] / self.fs
        if self.window_s > self.duration:
            self.window_s = self.duration
            self.win_box.setCurrentText(str(int(round(self.window_s))))
        # v1.3.8: IIIC opens on the labeled epoch (clip-local [10, 20] s) so the
        # red box frames the scored region by default; spike opens at 0 (the
        # full 10s clip). Clamp to the last full window for short clips.
        max_t = max(0.0, self.duration - self.window_s)
        self.t_start = (float(min(IIIC_LABEL_START_S, max_t))
                        if domain == "iiic" else 0.0)
        self.filter_bank = FilterBank(self.fs)
        self._refresh_answer_panel()
        self._redraw()

    def _redraw(self):
        if not hasattr(self, "data"):
            return
        self._draw_eeg()
        # v1.2.5: family-aware spectrogram panel.
        # Spike segments do not get a spectrogram (the Jing spike-paper
        # methodology uses only the 10s by 128 Hz EEG window for the Yes
        # / No spike classification). Hide the spec_container outright
        # so (1) the tutorial's IIIC spectrogram does not bleed through
        # onto the spike question, and (2) the EEG plot expands to fill
        # the freed space (it carries stretch=1 in plots_row while
        # spec_container has stretch=0; Qt re-distributes when a
        # stretch-0 widget is hidden). Also disable the spec_cb checkbox
        # so the user cannot toggle on a panel that has no data anyway.
        family = getattr(self, "_cur_family", "iiic")
        # v1.3.8: the "classify within the red box" banner is IIIC-only.
        if getattr(self, "classify_hint_lbl", None) is not None:
            self.classify_hint_lbl.setVisible(family == "iiic")
        if family == "spike":
            self.spec_container.setVisible(False)
            self.spec_cb.setEnabled(False)
        else:
            self.spec_cb.setEnabled(True)
            if self.spec_cb.isChecked():
                self.spec_container.setVisible(True)
                self._draw_spectrogram()
            else:
                self.spec_container.setVisible(False)

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
        # v1.3.8: box the expert-labeled epoch (clip-local [10, 20] s) on IIIC
        # recordings — this is the region scored by the panel. The box is in
        # clip-time data coords, so it frames the whole plot in the default
        # 10-20s window and leaves view when the user pans for context. Added
        # after the (per-draw) clear() above so it never accumulates.
        if getattr(self, "_cur_family", "iiic") == "iiic":
            _lo = IIIC_LABEL_START_S
            _hi = IIIC_LABEL_START_S + IIIC_LABEL_LEN_S
            _box = pg.LinearRegionItem(
                values=(_lo, _hi), movable=False,
                brush=pg.mkBrush(255, 80, 80, 45),
                pen=pg.mkPen((215, 45, 45), width=2))
            _box.setZValue(-10)   # behind the EEG traces
            self.eeg_plot.addItem(_box)
        # UI shows the question number; the segment id is internal only.
        qlabel = ("Tutorial example" if self._tutorial_active
                  else f"Question {self._cur_trial + 1}")
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
        # v1.3.8: clear any prior 30s-clip markers — this method reuses the
        # ImageItems and never clear()s the plots, so InfiniteLines would
        # otherwise accumulate on every redraw.
        for _p, _ln in getattr(self, "_spec_marker_lines", []):
            try:
                _p.removeItem(_ln)
            except Exception:
                pass
        self._spec_marker_lines = []
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
        # v1.3.8: dotted white verticals marking where the 30s EEG clip sits
        # within the 10-min spectrogram. The clip is the central 30s of the
        # displayed span, so mid = span centre and the bounds are mid ± 15 s.
        # Guarded to the precomputed 10-min case (span >> clip); the on-the-fly
        # 30s fallback spectrogram has no wider context to annotate.
        if (t1 - t0) > IIIC_CLIP_S * 1.5:
            _mid = 0.5 * (t0 + t1)
            for _x in (_mid - IIIC_CLIP_S / 2.0, _mid + IIIC_CLIP_S / 2.0):
                for _p in self.spec_plots:
                    _ln = pg.InfiniteLine(
                        pos=_x, angle=90,
                        pen=pg.mkPen('w', width=1.5,
                                     style=Qt.PenStyle.DotLine))
                    _ln.setZValue(10)   # over the spectrogram image
                    _p.addItem(_ln)
                    self._spec_marker_lines.append((_p, _ln))

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
                Qt.Key.Key_Down, Qt.Key.Key_Control,
                Qt.Key.Key_Return, Qt.Key.Key_Enter, *_ANSWER_KEYS}

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
        """Keyboard shortcuts.

          ← / →  : pan ±10 s (or ±half window if window shorter than 10s)
          ↑ / ↓  : step through gain ladder (↑ = bigger traces)
          Ctrl   : cycle montage bipolar → average → laplacian → bipolar
          1-6    : select an IIIC answer option
          Enter  : confirm the selected answer
        """
        if self._tutorial_active or self._session_over:
            return            # viewer inert during the tutorial / after the test
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
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._confirm_answer()
        elif key in self._ANSWER_KEYS:
            # Number keys 1-6 select an IIIC option (same as a button click).
            self._select_answer(self._ANSWER_KEYS.index(key))
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


# --- Registration constants (module-level: imported by tests / smoke) ----

# CONSENT_VERSION
CONSENT_VERSION = "v1.3.6"
# IRB_PROTOCOL_ID
IRB_PROTOCOL_ID = "IRB 2016P000058 and IRB 2013P001024"

# Required fields gate Continue on each wizard page. _EXPERTISE order is
# stable — cortex_smoke.py:open_registration relies on index 4 == Fellow.
_EXPERTISE = ["— select —", "Medical student", "Resident", "Nurse",
              "Fellow", "Neurologist", "Epileptologist", "Other"]
_SEX = ["— select —", "Male", "Female", "Prefer not to say"]
# v1.1.3: gender_identity field removed (sex is the canonical
# demographic; the SAGER two-field convention was over-scoped for
# the credentialing dataset's analytic needs). REGISTRATION_FIELDS_V2
# kept frozen below for schema-migration detection.
_YEARS_EEG = ["— select —", "0–4", "5–9", "10–14", "15–19", "20–24",
              "25–29", "30+"]
_EEG_VOLUME = ["— select —", "Fewer than 5", "5–20", "21–50", "51–100",
               "More than 100"]
_PRACTICE = ["— select —", "Academic medical center", "Community hospital",
             "Tele-EEG service", "Private practice",
             "Training only / not yet in practice", "Other"]
_COLOR_VISION = ["Prefer not to say", "Normal color vision",
                 "Red–green deficiency", "Blue–yellow deficiency", "Unsure"]
_CONFIDENCE = ["Prefer not to say",
               "1 — Very low", "2", "3", "4 — Moderate", "5", "6",
               "7 — Very high"]
_PRIOR_TEST = ["Prefer not to say", "No", "Yes", "Unsure"]
# v1.1.4: per-session preference — controls whether the post-session
# MP4 renders (collapse / passfail / engine_explainer, ~3 min total)
# run at all. Default is "No" (fast path) per the v1.1.3 feedback that
# the 3-min render time blocked participants from seeing their
# results. Opt-in is the only path that triggers the renders; opt-out
# skips them entirely (no MP4 files produced in the session dir).
_VISUALIZATIONS = ["No — faster results",
                   "Yes — generate visualizations"]
# Short curated country list — broad geographic coverage for v1.1.1.
# Expand to full ISO 3166 once the dataset volume warrants it.
_COUNTRY = ["Prefer not to say", "United States", "Canada", "Mexico",
            "Brazil", "United Kingdom", "Ireland", "Germany", "France",
            "Italy", "Spain", "Netherlands", "Sweden", "Switzerland",
            "Israel", "Saudi Arabia", "United Arab Emirates", "South Africa",
            "India", "China", "Japan", "South Korea", "Singapore",
            "Australia", "New Zealand", "Other"]
# NIH categories with MENA pilot category (Federal Register 2024 OMB SPD 15).
_RACE = ["Prefer not to say", "American Indian or Alaska Native", "Asian",
         "Black or African American", "Hispanic or Latino/a/x",
         "Middle Eastern or North African",
         "Native Hawaiian or Pacific Islander", "White", "More than one",
         "Other"]

# CSV schema for registrations.csv. The canonical "current" name is
# REGISTRATION_FIELDS (always points at the most recent version);
# version-numbered constants are kept as frozen legacy snapshots so
# _save_row's migration logic can name the .bak file by the version
# that was on disk.
#
# History:
#   V1 — pre-v1.1.1, 9 columns (name + 6 free-text fields).
#   V2 — v1.1.1 / v1.1.2, 20 columns (wizard intake + gender_identity).
#   V3 — v1.1.3, 19 columns (gender_identity dropped; sex retained).
#   V4 — v1.1.4+, 20 columns (+ wants_visualizations session preference).
_REGISTRATION_FIELDS_V1 = [
    "session_id", "timestamp_utc",
    "name", "age", "gender", "institution", "email", "expertise",
    "credentials",
]
_REGISTRATION_FIELDS_V2 = [
    "session_id", "timestamp_utc", "consent_version", "irb_protocol_id",
    "eligibility_confirmed",
    "name", "age", "email",
    "institution", "expertise", "practice_setting",
    "years_reading_eeg", "eeg_volume_per_month",
    "self_rated_confidence", "color_vision", "prior_test_taken",
    "sex", "gender_identity", "country", "race_ethnicity",
]
_REGISTRATION_FIELDS_V3 = [
    "session_id", "timestamp_utc", "consent_version", "irb_protocol_id",
    "eligibility_confirmed",
    "name", "age", "email",
    "institution", "expertise", "practice_setting",
    "years_reading_eeg", "eeg_volume_per_month",
    "self_rated_confidence", "color_vision", "prior_test_taken",
    "sex", "country", "race_ethnicity",
]
REGISTRATION_FIELDS_V4 = [
    "session_id", "timestamp_utc", "consent_version", "irb_protocol_id",
    "eligibility_confirmed",
    # Identity
    "name", "age", "email",
    # Clinical background
    "institution", "expertise", "practice_setting",
    "years_reading_eeg", "eeg_volume_per_month",
    "self_rated_confidence", "color_vision", "prior_test_taken",
    # Demographics (gender_identity removed v1.1.3)
    "sex", "country", "race_ethnicity",
    # Session preferences (v1.1.4)
    "wants_visualizations",
]
# Canonical "always-current" alias. New code should import this name;
# version-suffixed constants are for migration / historical reference.
REGISTRATION_FIELDS = REGISTRATION_FIELDS_V4

# Migration: detect which legacy schema a non-current header matches
# so the .bak filename names the version that was rotated.
_LEGACY_SCHEMAS = (("v3", _REGISTRATION_FIELDS_V3),
                   ("v2", _REGISTRATION_FIELDS_V2),
                   ("v1", _REGISTRATION_FIELDS_V1))


def is_opt_in_for_visualizations(reg_row) -> bool:
    """Return True if the participant opted in to the slow per-session
    MP4 renders (v1.1.4). Centralised so the dropdown's label format
    can change without scattering string comparisons across the
    codebase. Missing / unknown value ⇒ opt-out (the default)."""
    val = (reg_row.get("wants_visualizations") or "").lower()
    return val.startswith("yes")


def _is_dropdown_set(combo: QComboBox) -> bool:
    """A QComboBox whose first item is the '— select —' sentinel counts as
    'set' only if the user picked a later entry."""
    if combo.count() == 0:
        return False
    first = combo.itemText(0)
    if first.startswith("—"):
        return combo.currentIndex() > 0
    return True


class RegistrationPage(QWidget):
    """Three-page participant-registration wizard. `commit()` validates
    the entry and appends it to results/registrations.csv;
    `continue_btn` (the page-3 advance button) hands control to the
    viewer once `commit()` returns True.

    Class-level API kept stable for the outer flow in
    eeg_bank_viewer.run() and for cortex_smoke / tests:
      - .continue_btn  : the final 'CONTINUE' button (page 3)
      - .commit()      : validates pages 1–3 and writes the CSV row
      - .registration  : the saved dict (read by SessionRecorder)
      - .session_id    : UUID4 generated at commit
      - .f_name, .f_email, .f_age, .f_inst, .f_expertise : preserved
        widget handles used by cortex_smoke.py / tests
    """

    _EXPERTISE = _EXPERTISE  # back-compat: tests / smoke refer to this

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
    _CHECK_CSS = (
        "QCheckBox { color: #dde0e6; background: transparent; }"
        " QCheckBox::indicator { width: 18px; height: 18px;"
        " border: 1px solid #5a5f6b; background: #15171c; }"
        " QCheckBox::indicator:checked { background: #4a7bd6;"
        " border-color: #4a7bd6; }"
    )

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CORTEX")
        self.resize(980, 720)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"RegistrationPage {{ background-color: {_PAGE_BG}; }}")

        # Build per-page widgets first so their handles are available
        # to validation + commit regardless of which page is showing.
        self._build_widgets()

        # Stacked container: page 1 → page 2 → page 3.
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_page1_identity())
        self._stack.addWidget(self._build_page2_clinical())
        self._stack.addWidget(self._build_page3_demographics())

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._stack)

        # Wire navigation. Continue/Back live on each page; commit fires
        # only on the page-3 advance.
        self._next1_btn.clicked.connect(self._advance_from_page1)
        self._next2_btn.clicked.connect(self._advance_from_page2)
        self._back2_btn.clicked.connect(
            lambda: self._stack.setCurrentIndex(0))
        self._back3_btn.clicked.connect(
            lambda: self._stack.setCurrentIndex(1))

    # ---- widget construction ------------------------------------------

    def _build_widgets(self):
        # Page 1 — eligibility + identity (required: gate + name + email).
        self.f_eligibility = QCheckBox(
            "I am a healthcare professional or student in a "
            "clinical/research role.")
        self.f_eligibility.setStyleSheet(self._CHECK_CSS)
        self.f_name = QLineEdit()
        self.f_email = QLineEdit()
        self.f_age = QLineEdit()
        for w in (self.f_name, self.f_email, self.f_age):
            w.setFixedHeight(34)
            w.setStyleSheet(self._FIELD_CSS)

        # Page 2 — clinical background (most fields required).
        self.f_inst = QLineEdit()
        self.f_inst.setFixedHeight(34)
        self.f_inst.setStyleSheet(self._FIELD_CSS)
        self.f_expertise = self._make_combo(_EXPERTISE)
        self.f_practice = self._make_combo(_PRACTICE)
        self.f_years_eeg = self._make_combo(_YEARS_EEG)
        self.f_eeg_volume = self._make_combo(_EEG_VOLUME)
        self.f_confidence = self._make_combo(_CONFIDENCE)
        self.f_color_vision = self._make_combo(_COLOR_VISION)
        self.f_prior_test = self._make_combo(_PRIOR_TEST)

        # Page 3 — demographics (all optional with 'Prefer not to say').
        # v1.1.3: gender_identity dropped; sex retained as the canonical
        # demographic field.
        self.f_sex = self._make_combo(_SEX)
        self.f_country = self._make_combo(_COUNTRY)
        self.f_race = self._make_combo(_RACE)
        # v1.1.4 — per-session preference. Default index 0 = "No" so
        # silent participants take the fast path; only those who
        # explicitly opt in wait the 2-3 min for visualizations.
        self.f_visualizations = self._make_combo(_VISUALIZATIONS)

    def _make_combo(self, items):
        c = QComboBox()
        c.addItems(items)
        c.setFixedHeight(34)
        c.setStyleSheet(self._COMBO_CSS)
        return c

    # ---- page layouts -------------------------------------------------

    def _page_chrome(self, title, step_text):
        """Returns (page_widget, root_layout, msg_label).
        Builds the heading + step indicator common to all three pages."""
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(1)

        heading = QLabel(title)
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hf = QFont()
        hf.setPointSize(26)
        hf.setWeight(QFont.Weight.DemiBold)
        hf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        heading.setFont(hf)
        heading.setStyleSheet("color: #eef1f5; background: transparent;")
        root.addWidget(heading)

        step = QLabel(step_text)
        step.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf = QFont()
        sf.setPointSize(10)
        step.setFont(sf)
        step.setStyleSheet("color: #767b87; background: transparent;")
        root.addSpacing(6)
        root.addWidget(step)

        msg = QLabel("")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mf = QFont()
        mf.setPointSize(11)
        msg.setFont(mf)
        msg.setStyleSheet("color: #d8806a; background: transparent;")
        return page, root, msg

    def _build_page1_identity(self):
        page, root, msg = self._page_chrome(
            "Participant Information", "Step 1 of 3 · Eligibility & identity")
        self._msg1 = msg

        form_box = QWidget()
        form_box.setFixedWidth(470)
        form = QFormLayout(form_box)
        form.setSpacing(13)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for label_text, field in (
            ("Name", self.f_name),
            ("Email", self.f_email),
            ("Age (optional)", self.f_age),
        ):
            form.addRow(self._flabel(label_text), field)

        root.addSpacing(26)
        root.addLayout(_hcenter(form_box))
        root.addSpacing(20)
        elig_row = QHBoxLayout()
        elig_row.addStretch(1)
        elig_row.addWidget(self.f_eligibility)
        elig_row.addStretch(1)
        root.addLayout(elig_row)
        root.addSpacing(10)
        root.addWidget(msg)
        root.addSpacing(16)

        self._next1_btn = QPushButton("NEXT")
        for b in (self._next1_btn,):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedSize(220, 50)
            b.setFont(_btn_font())
            b.setStyleSheet(_BTN_CSS)
        root.addLayout(_hcenter(self._next1_btn))
        root.addStretch(2)
        return page

    def _build_page2_clinical(self):
        page, root, msg = self._page_chrome(
            "Participant Information", "Step 2 of 3 · Clinical background")
        self._msg2 = msg

        form_box = QWidget()
        form_box.setFixedWidth(520)
        form = QFormLayout(form_box)
        form.setSpacing(11)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for label_text, field in (
            ("Institutional affiliation (optional)", self.f_inst),
            ("Expertise", self.f_expertise),
            ("Practice setting", self.f_practice),
            ("Years reading EEG", self.f_years_eeg),
            ("EEGs read per month", self.f_eeg_volume),
            ("Self-rated EEG-reading confidence (optional)",
             self.f_confidence),
            ("Color vision (optional)", self.f_color_vision),
            ("Have you taken this test before? (optional)",
             self.f_prior_test),
        ):
            form.addRow(self._flabel(label_text), field)

        root.addSpacing(20)
        root.addLayout(_hcenter(form_box))
        root.addSpacing(10)
        root.addWidget(msg)
        root.addSpacing(14)

        self._back2_btn = QPushButton("BACK")
        self._next2_btn = QPushButton("NEXT")
        nav = QHBoxLayout()
        nav.addStretch(1)
        for b in (self._back2_btn, self._next2_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedSize(180, 50)
            b.setFont(_btn_font())
            b.setStyleSheet(_BTN_CSS)
        nav.addWidget(self._back2_btn)
        nav.addSpacing(18)
        nav.addWidget(self._next2_btn)
        nav.addStretch(1)
        root.addLayout(nav)
        root.addStretch(1)
        return page

    def _build_page3_demographics(self):
        page, root, msg = self._page_chrome(
            "Participant Information", "Step 3 of 3 · Demographics (optional)")
        self._msg3 = msg
        # _msg is the public-facing message label that pre-wizard tests
        # used; alias it to the page-3 message so commit() errors show up.
        self.msg = msg

        sub = QLabel(
            "These fields support equity and generalizability analyses. "
            "Every option includes 'Prefer not to say'.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setFixedWidth(580)
        subf = QFont(); subf.setPointSize(10)
        sub.setFont(subf)
        sub.setStyleSheet("color: #9aa0ab; background: transparent;")
        root.addSpacing(12)
        root.addLayout(_hcenter(sub))

        form_box = QWidget()
        form_box.setFixedWidth(470)
        form = QFormLayout(form_box)
        form.setSpacing(13)
        form.setContentsMargins(0, 0, 0, 0)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for label_text, field in (
            ("Sex", self.f_sex),
            ("Country of practice", self.f_country),
            ("Race / ethnicity", self.f_race),
            ("Generate personalized visualizations?",
             self.f_visualizations),
        ):
            form.addRow(self._flabel(label_text), field)
        root.addSpacing(22)
        root.addLayout(_hcenter(form_box))
        root.addSpacing(10)
        root.addWidget(msg)
        root.addSpacing(14)

        self._back3_btn = QPushButton("BACK")
        self.continue_btn = QPushButton("CONTINUE")
        nav = QHBoxLayout()
        nav.addStretch(1)
        for b in (self._back3_btn, self.continue_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedSize(180, 50)
            b.setFont(_btn_font())
            b.setStyleSheet(_BTN_CSS)
        nav.addWidget(self._back3_btn)
        nav.addSpacing(18)
        nav.addWidget(self.continue_btn)
        nav.addStretch(1)
        root.addLayout(nav)
        root.addStretch(1)
        return page

    @staticmethod
    def _flabel(text):
        lbl = QLabel(text)
        f = QFont()
        f.setPointSize(11)
        lbl.setFont(f)
        lbl.setStyleSheet("color: #9aa0ab; background: transparent;")
        return lbl

    # ---- page-by-page validation --------------------------------------

    def _validate_page1(self):
        if not self.f_name.text().strip():
            return "Please enter your name."
        if not self.f_email.text().strip():
            return "Please enter your email."
        if "@" not in self.f_email.text() or "." not in self.f_email.text():
            return "Please enter a valid email address."
        if not self.f_eligibility.isChecked():
            return ("Please confirm you are a healthcare professional "
                    "or student.")
        return None

    def _validate_page2(self):
        if not _is_dropdown_set(self.f_expertise):
            return "Please select your expertise."
        if not _is_dropdown_set(self.f_practice):
            return "Please select your practice setting."
        if not _is_dropdown_set(self.f_years_eeg):
            return "Please select your years reading EEG."
        if not _is_dropdown_set(self.f_eeg_volume):
            return "Please select how many EEGs you read per month."
        return None

    def _validate_page3(self):
        # Page 3 fields are all optional (every dropdown defaults to a
        # "Prefer not to say" / first-item answer). Nothing to gate on.
        return None

    def _advance_from_page1(self):
        err = self._validate_page1()
        if err:
            self._msg1.setText(err)
            return
        self._msg1.setText("")
        self._stack.setCurrentIndex(1)

    def _advance_from_page2(self):
        err = self._validate_page2()
        if err:
            self._msg2.setText(err)
            return
        self._msg2.setText("")
        self._stack.setCurrentIndex(2)

    # ---- commit (page-3 advance) --------------------------------------

    def commit(self):
        """Re-validate all pages, build the schema-v2 row, persist it.
        Returns True on success; on failure, surfaces the error on the
        relevant page's message label."""
        # Re-validate in case the user bypassed the wizard (e.g., tests
        # that call commit() directly without clicking through).
        for idx, (label, validator) in enumerate((
            (self._msg1, self._validate_page1),
            (self._msg2, self._validate_page2),
            (self._msg3, self._validate_page3),
        )):
            err = validator()
            if err:
                self._stack.setCurrentIndex(idx)
                label.setText(err)
                return False

        self.session_id = str(uuid.uuid4())
        row = self._build_row()
        self.registration = row          # handed to the SessionRecorder
        try:
            self._save_row(row)
        except Exception as e:
            self._msg3.setText(f"Could not save registration: {e}")
            return False
        return True

    def _build_row(self):
        def _combo_val(combo):
            # Treat the "— select —" sentinel as empty.
            if combo.count() and combo.itemText(0).startswith("—") \
                    and combo.currentIndex() == 0:
                return ""
            return combo.currentText()

        return {
            "session_id": self.session_id,
            "timestamp_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(timespec="seconds"),
            "consent_version": CONSENT_VERSION,
            "irb_protocol_id": IRB_PROTOCOL_ID,
            "eligibility_confirmed": "yes"
                if self.f_eligibility.isChecked() else "no",
            "name": self.f_name.text().strip(),
            "age": self.f_age.text().strip(),
            "email": self.f_email.text().strip(),
            "institution": self.f_inst.text().strip(),
            "expertise": _combo_val(self.f_expertise),
            "practice_setting": _combo_val(self.f_practice),
            "years_reading_eeg": _combo_val(self.f_years_eeg),
            "eeg_volume_per_month": _combo_val(self.f_eeg_volume),
            "self_rated_confidence": _combo_val(self.f_confidence),
            "color_vision": _combo_val(self.f_color_vision),
            "prior_test_taken": _combo_val(self.f_prior_test),
            "sex": _combo_val(self.f_sex),
            "country": _combo_val(self.f_country),
            "race_ethnicity": _combo_val(self.f_race),
            # v1.1.4 — the f_visualizations combo has no "— select —"
            # sentinel so _combo_val always returns one of the two
            # explicit options; never "".
            "wants_visualizations": self.f_visualizations.currentText(),
        }

    @staticmethod
    def _save_row(row):
        # Route to the platform's user-data dir when running from a
        # frozen .app/.exe (the bundle is read-only — fatal under
        # macOS App Translocation). Dev runs keep using <repo>/results/.
        from cortex_storage import user_data_root  # local to avoid import cycle
        path = user_data_root() / "registrations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Schema migration: if the existing file's header doesn't match
        # REGISTRATION_FIELDS (the canonical current schema), rotate
        # it to a versioned .bak file. We name the backup by the
        # schema version that was actually on disk (v1, v2, or
        # 'legacy' for an unrecognised header), so an upgrade chain
        # stays auditable. DictWriter doesn't validate against an
        # existing header — silent column drift is the failure mode
        # without this guard.
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8", newline="") as rh:
                    existing_header = next(csv.reader(rh), [])
            except StopIteration:
                existing_header = []
            if existing_header != REGISTRATION_FIELDS:
                ver = "legacy"
                for tag, fields in _LEGACY_SCHEMAS:
                    if existing_header == fields:
                        ver = tag
                        break
                bak = path.with_suffix(f".{ver}.csv.bak")
                # Don't overwrite an earlier rotation.
                i = 1
                while bak.exists():
                    bak = path.with_suffix(f".{ver}.csv.bak{i}")
                    i += 1
                path.rename(bak)
        is_new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=REGISTRATION_FIELDS)
            if is_new:
                w.writeheader()
            w.writerow(row)


# ── v1.1.4: background-thread finalize + computing-results landing ──

class _FinalizeWorker(QObject):
    """QObject worker that runs SessionRecorder.finalize() off the GUI
    thread so the ComputingResultsPage stays interactive (spinner
    animates, window can be moved) while finalize does its file I/O
    and — if the participant opted in — the slow MP4 renders.

    The recorder is a plain Python object (no Qt internals) so calling
    its methods from a non-GUI thread is safe. Matplotlib's Agg
    backend (used by cortex_render_videos + render_engine_explainer)
    is also thread-safe.

    Signals:
      finished(object) — emitted on success; payload is the original
                         SessionResult so the main thread can hand it
                         to ResultsScreen.
      failed(str)      — emitted on exception; payload is the error
                         message for logging. ResultsScreen still
                         renders (partial data is better than none).
    """
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    # v1.2.9: (stage label, fraction 0..1, eta_seconds or None). Emitted
    # from this worker thread; connected to the GUI thread's
    # ComputingResultsPage.set_progress via Qt's queued cross-thread signal.
    progress = pyqtSignal(str, float, object)

    def __init__(self, recorder, result):
        super().__init__()
        self._recorder = recorder
        self._result = result

    def _emit_progress(self, stage, frac, eta):
        self.progress.emit(str(stage), float(frac), eta)

    @pyqtSlot()
    def run(self):
        try:
            if self._recorder is not None:
                self._recorder.finalize(self._result,
                                        progress=self._emit_progress)
        except Exception as e:                          # noqa: BLE001
            logging.getLogger(__name__).exception(
                "background finalize failed")
            self.failed.emit(str(e))
            return
        self.finished.emit(self._result)


class ComputingResultsPage(QWidget):
    """Brief transition screen shown between the last question and
    ResultsScreen. v1.1.4: the heavy MP4 renders now run on a
    background thread (_FinalizeWorker); this page is the
    user-visible 'please wait' covering that time. Always shown
    regardless of opt-in so the participant gets a clean
    acknowledgement of the test ending — just for a much shorter
    time on the fast (opt-out) path.

    Layout:
      heading: 'Thank you for participating.'
      sub:     dynamic by opt-in:
                opt-in → 'Computing your results and generating
                          personalized visualizations (this takes
                          2-3 minutes).'
                opt-out → 'Computing your results — this will take
                          just a few seconds.'
      spinner: QProgressBar in indeterminate mode (busy chase
               animation; no exact-progress signals needed from the
               worker)
      footer:  'Please don't close this window.'

    Class-level API: `.opt_in: bool` (read-only after construction)
    and the inherited QWidget surface. No buttons — closing is up to
    the user via the window decorations; the worker daemon will keep
    running if they do, so partial / no MP4 output is the failure
    mode (not a crash).
    """

    _PB_CSS = (
        "QProgressBar { background-color: #15171c; border: 1px solid"
        " #3a3d45; border-radius: 0px; height: 10px; }"
        " QProgressBar::chunk { background-color: #7ed391; }"
    )

    def __init__(self, opt_in: bool):
        super().__init__()
        self.opt_in = bool(opt_in)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"ComputingResultsPage {{ background-color: {_PAGE_BG}; }}")

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(2)

        heading = QLabel("Thank you for participating.")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hf = QFont()
        hf.setPointSize(28)
        hf.setWeight(QFont.Weight.DemiBold)
        hf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        heading.setFont(hf)
        heading.setStyleSheet("color: #eef1f5; background: transparent;")
        root.addWidget(heading)

        if self.opt_in:
            sub_text = ("Computing your results and generating "
                        "personalized visualizations.")
        else:
            sub_text = ("Computing your results.\n"
                        "This will take just a few seconds.")
        sub = QLabel(sub_text)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setFixedWidth(640)
        sf = QFont()
        sf.setPointSize(13)
        sub.setFont(sf)
        sub.setStyleSheet("color: #aab0ba; background: transparent;")
        root.addSpacing(18)
        row = QHBoxLayout()
        row.addStretch(1); row.addWidget(sub); row.addStretch(1)
        root.addLayout(row)

        # v1.2.9: starts indeterminate (busy-chase) so it animates during
        # the brief pre-render writes; switches to a determinate 0..1000 bar
        # on the first set_progress() call, then tracks real frame progress.
        self.spinner = QProgressBar()
        self.spinner.setMinimum(0)
        self.spinner.setMaximum(0)
        self.spinner.setTextVisible(False)
        self.spinner.setFixedSize(420, 10)
        self.spinner.setStyleSheet(self._PB_CSS)
        self._determinate = False
        root.addSpacing(34)
        row2 = QHBoxLayout()
        row2.addStretch(1); row2.addWidget(self.spinner); row2.addStretch(1)
        root.addLayout(row2)

        # v1.3.1: live stage label, fed by _FinalizeWorker.progress. The ETA
        # countdown was removed (the estimate was not accurate enough to be
        # useful); the determinate bar + stage still communicate progress.
        self.stage_lbl = QLabel("Starting…")
        self.stage_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stf = QFont(); stf.setPointSize(11)
        self.stage_lbl.setFont(stf)
        self.stage_lbl.setStyleSheet(
            "color: #cfd3da; background: transparent;")
        root.addSpacing(12)
        root.addWidget(self.stage_lbl)

        foot = QLabel("Please do not close this window.")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ff = QFont()
        ff.setPointSize(10)
        foot.setFont(ff)
        foot.setStyleSheet("color: #5c606a; background: transparent;")
        root.addSpacing(28)
        root.addWidget(foot)
        root.addStretch(3)

    @pyqtSlot(str, float, object)
    def set_progress(self, stage, frac, eta):
        """Update the determinate bar + stage label. Connected (queued,
        cross-thread) to _FinalizeWorker.progress. v1.3.1: the ``eta``
        argument is accepted for signal compatibility but no longer shown."""
        if not self._determinate:
            self.spinner.setRange(0, 1000)
            self._determinate = True
        frac = max(0.0, min(1.0, float(frac)))
        self.spinner.setValue(int(frac * 1000))
        if stage:
            self.stage_lbl.setText("Done" if frac >= 1.0 else f"{stage}…")


class ResultsScreen(QWidget):
    """Terminal results screen — per-IIIC-task PASS/FAIL/REFER verdict
    with a one-line skill+bias narrative and a collapsible 'technical
    details' grid showing the raw ℓ̂, ℓ*, θ̂, and pass-mass π values.

    Layout (v1.1.2):
      heading: "Assessment Complete"
      sub:     "N recordings reviewed. <stop reason>"
      verdict table (6 rows, 2 lines each):
        TASK    VERDICT          NARRATIVE (skill + bias)
      [ Show technical details ▾ ]  (toggles the details grid below)
      details grid (hidden by default):
        TASK  VERDICT  ℓ̂      ℓ*     θ̂      π
        + caption explaining the symbols.
      footer: "Your responses have been recorded."
      [ CLOSE ]

    `n_correct` and `n_answered` are accepted for back-compat with the
    BankViewer call site but no longer rendered; per-trial agreement
    is still recorded in cortex.log + the summary CSV for analysis.
    """

    _TASK_LABELS = {"spike": "Spike", "sz": "Seizure", "lpd": "LPD",
                    "gpd": "GPD", "lrda": "LRDA", "grda": "GRDA",
                    "iic": "Other"}
    _STOP_TEXT = {
        "all_resolved": "Each category reached a final assessment.",
        "delta_reached": "Skill was estimated to the target precision.",
        "bank_exhausted": "All available recordings were reviewed.",
        "aborted": "The assessment ended early.",
    }
    # Clinician-friendly verdict labels. Underlying state constants
    # (PASS / FAIL / REFER_BORDERLINE / REFER_UNINFORMATIVE / PENDING)
    # stay in cortex_policy + the persisted CSV/JSON — only the display
    # label is softened here.
    _VERDICT_LABEL = {
        "PASS": "Pass",
        "FAIL": "Did not pass",
        "REFER_BORDERLINE": "Refer (borderline)",
        "REFER_UNINFORMATIVE": "Refer (need more data)",
        "PENDING": "—",
    }
    _VERDICT_COLOR = {
        "PASS": "#7ed391",                  # soft green
        "FAIL": "#d8806a",                  # soft red-orange
        "REFER_BORDERLINE": "#d4b169",      # soft amber
        "REFER_UNINFORMATIVE": "#9aa0ab",   # muted grey
        "PENDING": "#5c606a",
    }

    def __init__(self, result, n_correct, n_answered, session_dir=None):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"ResultsScreen {{ background-color: {_PAGE_BG}; }}")

        # v1.1.5: session-output directory (where CORTEX wrote the
        # MP4s + CSVs + certificate.json). When set, a "Show results
        # folder" button appears next to CLOSE and opens this dir in
        # the native file manager. None ⇒ button is hidden, matches
        # the pre-v1.1.5 constructor contract.
        self._session_dir = Path(session_dir) if session_dir else None

        # Load per-task Youden thresholds for the narrative + details
        # panel. Failure here (missing cert_config, malformed entry)
        # is non-fatal — we drop the threshold context but still show
        # the verdict + raw ℓ̂/θ̂.
        codes = list(getattr(result, "task_codes", []) or [])
        self._ell_star = self._safe_load_ell_star(codes)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addStretch(1)

        heading = QLabel("Assessment Complete")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hf = QFont()
        hf.setPointSize(28)
        hf.setWeight(QFont.Weight.DemiBold)
        hf.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1)
        heading.setFont(hf)
        heading.setStyleSheet("color: #eef1f5; background: transparent;")
        root.addWidget(heading)

        n_q = int(getattr(result, "n_questions", n_answered) or n_answered)
        stop = self._STOP_TEXT.get(getattr(result, "stop_reason", ""), "")
        # v1.2.9: surface the spike vs IIIC-pattern split (n_q already sums
        # both families). Derived from per-trial task_code; falls back to the
        # plain total when trials are unavailable (older/stub results).
        trials = list(getattr(result, "trials", []) or [])
        n_spike = sum(1 for t in trials if t.get("task_code") == "spike")
        if trials:
            n_pattern = len(trials) - n_spike
            reviewed = (f"{n_q} recordings reviewed "
                        f"({n_spike} spike, {n_pattern} pattern).")
        else:
            reviewed = f"{n_q} recordings reviewed."
        sub = QLabel(f"{reviewed}   {stop}")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sf = QFont()
        sf.setPointSize(12)
        sub.setFont(sf)
        sub.setStyleSheet("color: #aab0ba; background: transparent;")
        root.addSpacing(10)
        root.addWidget(sub)

        root.addSpacing(22)
        root.addLayout(_hcenter(self._build_verdict_table(result)))

        # Collapsible details panel — hidden by default. The button text
        # toggles between "Show / Hide technical details".
        self._details_panel = self._build_details_panel(result)
        self._details_panel.setVisible(False)
        self.details_btn = QPushButton("Show technical details ▾")
        self.details_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_btn.setFixedHeight(28)
        df = QFont()
        df.setPointSize(10)
        self.details_btn.setFont(df)
        self.details_btn.setStyleSheet(
            "QPushButton { color: #9aa0ab; background: transparent;"
            " border: none; padding: 4px 10px; }"
            " QPushButton:hover { color: #eef1f5; }")
        self.details_btn.clicked.connect(self._toggle_details)
        root.addSpacing(14)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.details_btn)
        row.addStretch(1)
        root.addLayout(row)

        root.addSpacing(6)
        root.addWidget(self._details_panel)

        foot = QLabel("Your responses have been recorded.")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ff = QFont()
        ff.setPointSize(11)
        foot.setFont(ff)
        foot.setStyleSheet("color: #5c606a; background: transparent;")
        root.addSpacing(22)
        root.addWidget(foot)

        # v1.1.5: button row — [Show results folder]  [CLOSE]. The
        # "Show results folder" button is hidden when session_dir is
        # None (back-compat with tests + the no-recorder fast path)
        # so the screen still shows just CLOSE when there's no folder
        # to point at.
        self.show_folder_btn = QPushButton("SHOW RESULTS FOLDER")
        self.show_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_folder_btn.setFixedSize(240, 48)
        self.show_folder_btn.setFont(_btn_font())
        self.show_folder_btn.setStyleSheet(_BTN_CSS)
        self.show_folder_btn.clicked.connect(self._open_session_folder)
        self.show_folder_btn.setVisible(self._session_dir is not None)

        self.close_btn = QPushButton("CLOSE")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setFixedSize(200, 48)
        self.close_btn.setFont(_btn_font())
        self.close_btn.setStyleSheet(_BTN_CSS)
        self.close_btn.clicked.connect(lambda: self.window().close())

        root.addSpacing(16)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.show_folder_btn)
        btn_row.addSpacing(18)
        btn_row.addWidget(self.close_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)
        root.addStretch(2)

    def _open_session_folder(self):
        """Open the per-session output dir in the OS file manager.
        Dispatches by sys.platform; logs + no-ops if the folder is
        missing (defensive — shouldn't happen on a finalized
        session, but a participant clicking after manual file
        deletion shouldn't crash the app)."""
        if self._session_dir is None:
            return
        path = self._session_dir
        if not path.is_dir():
            logging.getLogger(__name__).warning(
                "ResultsScreen: cannot open results folder — %s "
                "does not exist", path)
            return
        try:
            if sys.platform == "darwin":
                cmd = ["open", str(path)]
            elif sys.platform == "win32":
                # Windows Explorer accepts the path as positional;
                # os.startfile would also work but subprocess is
                # uniform with the other platforms and easier to
                # mock in tests.
                cmd = ["explorer", str(path)]
            else:
                # Linux / *BSD — XDG handler dispatches to the
                # configured file manager (Nautilus / Dolphin / etc.).
                cmd = ["xdg-open", str(path)]
            # check=False: don't raise if the file manager exits
            # nonzero (some return 1 for "already open"). Don't
            # capture output — let the user see file-manager errors
            # in the spawning terminal during dev runs.
            subprocess.Popen(cmd, close_fds=True)
        except Exception as e:                          # noqa: BLE001
            logging.getLogger(__name__).warning(
                "ResultsScreen: open-results-folder failed: %s", e)

    @staticmethod
    def _safe_load_ell_star(codes):
        """Return [ℓ*_k] for the given task codes, or [None]*K if the
        config cannot be loaded. Non-fatal — keeps the screen visible
        even when cert_config is missing / malformed.

        Phase-9 K=7: defaults to cortex_policy_k7.load_ell_star_k7 which
        reads the K=7 cert_config v14 block (`ell_star_unified_v14`,
        7 tasks). Falls back to the K=6 loader if the K=7 loader cannot
        be imported (back-compat with K=6 builds)."""
        if not codes:
            return []
        try:
            from cortex_policy_k7 import load_ell_star_k7
            return list(load_ell_star_k7(codes))
        except Exception as e:                       # noqa: BLE001
            logging.getLogger(__name__).warning(
                "ResultsScreen: could not load ℓ* thresholds (%s); "
                "narrative will omit threshold context.", e)
            return [None] * len(codes)

    @staticmethod
    def _cell(text, color, size, bold=False,
              align=Qt.AlignmentFlag.AlignLeft):
        lbl = QLabel(text)
        f = QFont()
        f.setPointSize(size)
        if bold:
            f.setWeight(QFont.Weight.DemiBold)
        lbl.setFont(f)
        lbl.setStyleSheet(f"color: {color}; background: transparent;")
        lbl.setAlignment(align | Qt.AlignmentFlag.AlignVCenter)
        return lbl

    @staticmethod
    def _skill_narrative(l_hat, l_star):
        """One-clause skill summary: 'Above/Below threshold by 0.XX' or
        'Near the passing threshold'. Returns ('text', color)."""
        if l_hat is None:
            return ("Skill not estimated", "#9aa0ab")
        if l_star is None:
            return (f"Skill estimate ℓ̂ = {l_hat:+.2f}", "#dde0e6")
        diff = float(l_hat) - float(l_star)
        if abs(diff) < 0.05:
            return ("Near the passing threshold", "#d4b169")
        if diff >= 0.05:
            return (f"Above threshold by {diff:.2f}", "#7ed391")
        return (f"Below threshold by {abs(diff):.2f}", "#d8806a")

    @staticmethod
    def _bias_narrative(t_hat):
        """One-clause bias summary: 'liberal' = over-calls (θ̂<0),
        'conservative' = under-calls (θ̂>0). Returns ('text', color)."""
        if t_hat is None:
            return ("", "#9aa0ab")
        a = abs(float(t_hat))
        if a < 0.10:
            return ("Bias near neutral", "#aab0ba")
        direction = ("liberal (over-calls)" if float(t_hat) < 0
                     else "conservative (under-calls)")
        strength = "Slight" if a < 0.25 else "Strong"
        return (f"{strength} {direction}", "#aab0ba")

    def _build_verdict_table(self, result):
        """Per-task table: TASK | VERDICT (colored) | NARRATIVE (2 lines:
        skill summary above bias summary)."""
        codes = list(getattr(result, "task_codes", []) or [])
        verdicts = list(getattr(result, "verdicts", None) or [])
        lm = getattr(result, "final_l_mean", None)
        tm = getattr(result, "final_t_mean", None)
        lm = [] if lm is None else list(lm)
        tm = [] if tm is None else list(tm)

        box = QWidget()
        box.setFixedWidth(700)
        grid = QGridLayout(box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(14)
        grid.addWidget(self._cell("TASK", "#6b7280", 9, True), 0, 0)
        grid.addWidget(self._cell("RESULT", "#6b7280", 9, True), 0, 1)
        grid.addWidget(self._cell("SKILL & BIAS", "#6b7280", 9, True), 0, 2)
        for i, code in enumerate(codes):
            label = self._TASK_LABELS.get(code, code)
            grid.addWidget(self._cell(label, "#dde0e6", 13, True),
                           i + 1, 0)
            # Verdict cell — clinician-friendly label, color-coded
            v = verdicts[i] if i < len(verdicts) else "PENDING"
            v_text = self._VERDICT_LABEL.get(v, v)
            v_color = self._VERDICT_COLOR.get(v, self._VERDICT_COLOR["PENDING"])
            grid.addWidget(self._cell(v_text, v_color, 13, True),
                           i + 1, 1)
            # Narrative cell — two stacked lines (skill + bias)
            l_hat = float(lm[i]) if i < len(lm) else None
            t_hat = float(tm[i]) if i < len(tm) else None
            l_star = (self._ell_star[i] if i < len(self._ell_star) else None)
            skill_text, skill_color = self._skill_narrative(l_hat, l_star)
            bias_text, bias_color = self._bias_narrative(t_hat)
            narrative = QWidget()
            nv = QVBoxLayout(narrative)
            nv.setContentsMargins(0, 0, 0, 0)
            nv.setSpacing(2)
            nv.addWidget(self._cell(skill_text, skill_color, 12))
            if bias_text:
                nv.addWidget(self._cell(bias_text, bias_color, 10))
            grid.addWidget(narrative, i + 1, 2)
        return box

    def _build_details_panel(self, result):
        """Compact grid of raw posterior numbers: TASK | VERDICT | ℓ̂ |
        ℓ* | θ̂ | π. Hidden by default; toggled via the details button."""
        codes = list(getattr(result, "task_codes", []) or [])
        verdicts = list(getattr(result, "verdicts", None) or [])
        lm = getattr(result, "final_l_mean", None)
        tm = getattr(result, "final_t_mean", None)
        lm = [] if lm is None else list(lm)
        tm = [] if tm is None else list(tm)
        # π_k (posterior P(ℓ̂ > ℓ*)) is the AD6-policy quantity; available
        # only when the session ran under AD6 (otherwise dict is None).
        pd = getattr(result, "policy_diagnostics", None) or {}
        pi = list(pd.get("pi") or []) if isinstance(pd, dict) else []

        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)

        grid_box = QWidget()
        grid_box.setFixedWidth(700)
        grid = QGridLayout(grid_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(6)
        right = Qt.AlignmentFlag.AlignRight
        # Headers — small uppercase grey
        for col, (text, align) in enumerate((
            ("TASK", Qt.AlignmentFlag.AlignLeft),
            ("VERDICT", Qt.AlignmentFlag.AlignLeft),
            ("ℓ̂", right), ("ℓ*", right),
            ("θ̂", right), ("π", right),
        )):
            grid.addWidget(self._cell(text, "#6b7280", 9, True, align),
                           0, col)
        for i, code in enumerate(codes):
            grid.addWidget(self._cell(code, "#dde0e6", 11), i + 1, 0)
            v = verdicts[i] if i < len(verdicts) else "PENDING"
            grid.addWidget(self._cell(v, "#aab0ba", 10), i + 1, 1)
            l_hat = lm[i] if i < len(lm) else None
            t_hat = tm[i] if i < len(tm) else None
            l_star = self._ell_star[i] if i < len(self._ell_star) else None
            pi_k = pi[i] if i < len(pi) else None
            grid.addWidget(self._cell(
                f"{l_hat:.3f}" if l_hat is not None else "—",
                "#eef1f5", 11, False, right), i + 1, 2)
            grid.addWidget(self._cell(
                f"{l_star:.3f}" if l_star is not None else "—",
                "#aab0ba", 11, False, right), i + 1, 3)
            grid.addWidget(self._cell(
                f"{t_hat:+.3f}" if t_hat is not None else "—",
                "#eef1f5", 11, False, right), i + 1, 4)
            grid.addWidget(self._cell(
                f"{pi_k:.3f}" if pi_k is not None else "—",
                "#eef1f5", 11, False, right), i + 1, 5)
        pv.addLayout(_hcenter(grid_box))

        cap = QLabel(
            "ℓ̂ — posterior skill estimate · ℓ* — Youden-optimal passing "
            "threshold · θ̂ — bias (positive = conservative; negative = "
            "liberal) · π — posterior P(ℓ̂ > ℓ*).")
        cap.setWordWrap(True)
        cap.setFixedWidth(700)
        cf = QFont()
        cf.setPointSize(9)
        cap.setFont(cf)
        cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cap.setStyleSheet("color: #767b87; background: transparent;")
        pv.addLayout(_hcenter(cap))
        return panel

    def _toggle_details(self):
        # Use isHidden() — reflects the explicit setVisible() state
        # regardless of whether the top-level window is shown.
        # isVisible() returns False whenever the parent isn't on
        # screen, which would make this toggle stick in "show" mode.
        will_show = self._details_panel.isHidden()
        self._details_panel.setVisible(will_show)
        self.details_btn.setText(
            "Hide technical details ▴" if will_show
            else "Show technical details ▾")


def main():
    # Initialise CORTEX file-based logging first thing — PyInstaller
    # `console=False` macOS bundles silence stdout, so without this any
    # print() output (Dropbox status, MP4 render warnings, etc.) is
    # invisible post-session. setup_logging() writes to
    # ~/Library/Application Support/CORTEX/cortex.log.
    from cortex_storage import setup_logging  # local to keep PYZ tracing clean
    setup_logging()
    # Install the diagnostic excepthook BEFORE constructing QApplication so
    # any failure during Qt setup, engine-input load, or session start
    # ships a JSON diagnostic to Dropbox (same auth path as result CSVs).
    # Dev runs skip the upload unless CORTEX_DIAG_UPLOAD=1; frozen bundles
    # always upload. The original traceback is preserved either way.
    from cortex_diagnostics import install_excepthook
    install_excepthook()
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
        # Phase-9 K=7 (v1.2.1 hotfix): use the K=7 engine inputs builder so
        # the session sees all 7 tasks (spike + 6 IIIC). The v1.2.0 ship had
        # a leftover `from cortex_engine_inputs import build_iiic_engine_inputs`
        # here that returned K=6 IIIC-only inputs (300 segs) — spike segments
        # in the bundled bank were never asked. session_controller.py:67
        # aliases `build_iiic_engine_inputs = build_k7_engine_inputs` for
        # consumers that import from session_controller, but this entry
        # point imported from cortex_engine_inputs DIRECTLY, bypassing the
        # alias. Fixed in v1.2.1; regression-gated by
        # tests/test_cortex_viewer.py::test_main_open_viewer_uses_k7_engine_inputs.
        from cortex_engine_inputs_k7 import build_k7_engine_inputs
        from session_controller import (
            SessionController, N_PARTICLES, MAX_QUESTIONS_DEFAULT,
            MAX_CONSEC_SAME_DOMAIN_DEFAULT)
        from cortex_storage import SessionRecorder
        inputs = build_k7_engine_inputs()
        # Reserve one IIIC segment for the tutorial and exclude it from the
        # engine pool, so the engine never re-serves the practice segment.
        # Phase-9 K=7 (v1.2.4 fix): inputs.all_seg_ids[0] for the K=7
        # builder returns a SPIKE seg (spike comes first in the manifest),
        # but the tutorial UI is hard-coded for IIIC (6-button panel +
        # eeg30s renderer; v1.2.0-v1.2.3 bug: tutorial silently failed to
        # load because _render tried to fetch `/iiic/<spike_seg_id>` from
        # the bank which doesn't exist). Walk all_seg_ids and pick the
        # first IIIC seg explicitly. The K=6 IIICEngineInputs predecessor
        # has no .family attr so fall through to all_seg_ids[0] (which is
        # always IIIC there).
        if hasattr(inputs, "family"):
            tutorial_sid = next(
                s for s in inputs.all_seg_ids if inputs.family(s) == "iiic")
        else:
            tutorial_sid = inputs.all_seg_ids[0]
        engine_inputs = inputs.without([tutorial_sid])
        # v1.1.0: hard cap at MAX_QUESTIONS_DEFAULT — keeps the live-test
        # runway decoupled from bank size so future bank growth doesn't
        # implicitly lengthen the test. capture_clouds=True so the session
        # writes trajectory.npz.
        controller = SessionController(
            engine_inputs, reg.session_id,
            capture_clouds=True,
            max_questions=MAX_QUESTIONS_DEFAULT,
            # v1.3.5: break up long single-domain runs (sim-validated cap=12).
            max_consecutive_same_domain=MAX_CONSEC_SAME_DOMAIN_DEFAULT)
        # Record which termination policy actually drives this session —
        # AD6Policy in production, DeltaStopPolicy / NoStopPolicy on the
        # legacy/audit paths — so participant.json reflects what stopped
        # the session, not a stale module constant.
        # v1.1.4: visualizations are opt-in (default off). When the
        # participant opted out at registration, render_videos=False
        # makes finalize() skip the 3 MP4 renders entirely (~3 min
        # saved). When opted in, render_videos=True triggers all
        # three renders on the background-finalize worker so the
        # ComputingResultsPage covers the wait.
        recorder = SessionRecorder(
            reg.session_id, reg.registration,
            {"n_iiic_segments": len(engine_inputs.all_seg_ids),
             "policy": type(controller.session.policy).__name__,
             "n_particles": N_PARTICLES,
             "max_questions": MAX_QUESTIONS_DEFAULT,
             "tutorial_seg_id": int(tutorial_sid)},
            render_videos=is_opt_in_for_visualizations(reg.registration))
        win = BankViewer(controller, reg.session_id, tutorial_sid, recorder)
        # Install app-wide event filter so combo boxes don't swallow arrow
        # keys / Ctrl before we see them.
        app.installEventFilter(win)
        win.show()
        reg.close()
        state["page"] = win
        # Coach-marks walkthrough on the reserved practice segment; when it
        # finishes, _finish_tutorial starts the engine session.
        app.processEvents()                  # let the window lay out first
        win.start_tutorial(tutorial_sid)

    landing = LandingPage()
    landing.begin_btn.clicked.connect(open_consent)
    goto(landing)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
