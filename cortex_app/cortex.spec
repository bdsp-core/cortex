# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CORTEX — the clinician-facing adaptive EEG
certification test-taker app.

Builds an unsigned, standalone .app (macOS) / .exe folder (Windows).
Users don't need Python installed; first launch on macOS will show a
Gatekeeper warning (right-click -> Open the first time).

Build:
    cd cortex_app
    bash build_mac.sh         # produces dist/CORTEX.app + dist/CORTEX.dmg
    build_windows.bat         # produces dist\\CORTEX\\CORTEX.exe folder

Inputs (must be present in the REPO ROOT at build time):
    cortex_config.yaml        # copy from cortex_config.example.yaml + fill
    data/eeg_bank.h5          # run cortex_app/fetch_test_bank.sh

All source / data / config lives at repo root — matches the layout
Eli's scripts/build_internal_test_zip.py and runtime modules expect.
Nothing CORTEX-source-related is duplicated under cortex_app/.
"""
import sys
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH)                # cortex_app/
REPO = ROOT.parent                   # repo root

# Files bundled into the app and unpacked alongside the entry script at
# runtime. The viewer reads them via
#   BANK_PATH = Path(__file__).resolve().parent.parent / "data" / "eeg_bank.h5"
# and cortex_storage reads `_REPO / "cortex_config.yaml"` — both resolve
# to (unpack_root)/data/ and (unpack_root)/ respectively. The second
# tuple element is the in-bundle destination.
datas = [
    # the curated test bank (fetched by fetch_test_bank.sh; gitignored)
    (str(REPO / 'data' / 'eeg_bank.h5'),       'data'),
    # iiic_segment_signals.csv — read by session_controller for segment
    # metadata
    (str(REPO / 'data' / 'labels' / 'iiic_segment_signals.csv'),
                                                'data/labels'),
    # logo asset used on the welcome / consent screens
    (str(REPO / 'data' / 'Brain_Data_Science_Platform.png'), 'data'),
    # local config carrying the (gitignored) Dropbox credentials
    (str(REPO / 'cortex_config.yaml'),         '.'),
    # frozen prior used by the engine for the live test
    (str(REPO / 'Sigma_l_fitted.npy'),         '.'),
    # engine + calibration packages (small, vendored verbatim)
    (str(REPO / 'engine'),                      'engine'),
    (str(REPO / 'calibration'),                 'calibration'),
]

binaries = []

# Hidden imports that PyInstaller's static analysis can miss.
hiddenimports = [
    # PyQt6 plugins
    'PyQt6.QtSvg',
    'PyQt6.QtPrintSupport',
    # scipy submodules used at runtime
    'scipy.ndimage',
    'scipy.signal',
    'scipy.signal._signaltools',
    'scipy.signal._filter_design',
    'scipy.signal._fir_filter_design',
    # h5py inner pieces
    'h5py._hl',
    'h5py._hl.files',
    'h5py._hl.dataset',
    # matplotlib backends (used by cortex_render_videos for the
    # per-session collapse.mp4 / passfail.mp4 outputs)
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_svg',
    # pyqtgraph
    'pyqtgraph.graphicsItems',
    'pyqtgraph.graphicsItems.ViewBox',
    # cortex modules under repo's scripts/
    'cortex_engine_inputs',
    'cortex_policy',
    'cortex_render_videos',
    'cortex_storage',
    'session_controller',
    # vendored packages bundled via datas
    'engine',
    'engine.core_mcmc',
    'engine.auroc',
]

a = Analysis(
    [str(REPO / 'scripts' / 'eeg_bank_viewer.py')],
    pathex=[str(REPO / 'scripts'), str(REPO)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # GUI alternatives we don't use
        'tkinter', '_tkinter',
        # heavy science stacks not needed by the test-taker
        'IPython', 'jupyter',
        'pytest', 'sphinx',
        'numpyro', 'jax', 'jaxlib',
        'torch', 'torchvision', 'pytorch_lightning',
        'mne', 'neurokit2',
        'statsmodels',
        'tornado',
    ],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CORTEX',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,                # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,       # unsigned; user right-click->Open first time
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CORTEX',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='CORTEX.app',
        bundle_identifier='org.bdsp-core.cortex',
        info_plist={
            'CFBundleShortVersionString': '1.0.0',
            'CFBundleName': 'CORTEX',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '12.0',
            'NSHumanReadableCopyright': 'BDSP / Westover Lab',
        },
    )
