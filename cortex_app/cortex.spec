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

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

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
    # engine package — vendored verbatim. 180 KB total; cheap to ship the
    # whole tree. Includes diagnostics.py + core_mcmc_brute_k.py +
    # engine_mode_b.py + variants/ which are NOT exercised by the CORTEX
    # runtime path (CortexSession uses Mode-A SMC adaptive only), but
    # core_mcmc.py has lazy `from diagnostics import ...` and
    # `from core_mcmc_brute_k import ...` guarded by callback / method
    # kwargs — keeping the whole tree is the safe choice.
    (str(REPO / 'engine'),                      'engine'),
    # calibration: the runtime needs ONLY cert_config.yaml (read by
    # cortex_policy.load_ell_star_iiic). The wholesale include of the
    # `calibration/` tree in v1.0..1.0.5 dragged in 38 MB of dev-only
    # NUTS posterior .npz files under calibration/joint/ that are never
    # opened at runtime — plus the calibration orchestrator scripts
    # (_run_youden_joint.py, cli.py), the CALIBRATION_PROVENANCE.md, and
    # the youden_ell_star.json archival output. All dropped in v1.0.6.
    (str(REPO / 'calibration' / 'cert_config.yaml'),  'calibration'),
    # cortex_policy.load_ell_star_iiic() also has a fallback path that
    # reads a bundle-root copy of cert_config.yaml. Ship that fallback
    # (4 KB defensive insurance for the alt-path branch).
    (str(REPO / 'cert_config.yaml'),           '.'),
] + collect_data_files('certifi')  # certifi CA bundle is loaded via
                                   # importlib.resources by `requests` →
                                   # invisible to the AST tracer. Without
                                   # this, the dropbox HTTPS transport
                                   # would fail at session finalize.

# v1.1.2: bundle imageio-ffmpeg's per-platform ffmpeg binary so MP4
# rendering (collapse.mp4 / passfail.mp4) works on clinician machines
# without a system ffmpeg install. The binary lives inside the wheel
# at imageio_ffmpeg/binaries/ — collect_data_files picks it up
# regardless of the platform-specific filename
# (ffmpeg-linux-x86_64-v7.0.2, ffmpeg-macos-aarch64-v7.0.2,
# ffmpeg-win-x86_64-v7.0.2.exe, etc.). The wheel ships only ONE
# binary per platform, so the .dmg / .zip / .tar.gz each gain
# ~25-80 MB. License note: imageio-ffmpeg's bundled ffmpeg is
# GPL-licensed (includes libx264); CORTEX is CC BY-NC 4.0 and
# shells out to ffmpeg as a separate executable (mere aggregation,
# not linking) — compatible. See README / LICENSE attribution.
datas += collect_data_files('imageio_ffmpeg', include_py_files=False)

binaries = []

# Hidden imports that PyInstaller's static analysis can miss.
hiddenimports = [
    # PyQt6 plugins
    'PyQt6.QtSvg',
    'PyQt6.QtPrintSupport',
    # scipy submodules used at runtime. engine/core.py + core_mcmc.py +
    # auroc.py call scipy.stats.norm + scipy.special.{log_ndtr, logsumexp}
    # during engine update — these go through cython-compiled .so files
    # that scipy 1.13.x's bundled hook sometimes drops.
    'scipy.ndimage',
    'scipy.signal',
    'scipy.signal._signaltools',
    'scipy.signal._filter_design',
    'scipy.signal._fir_filter_design',
    'scipy.special',
    'scipy.special._ufuncs',
    'scipy.special.cython_special',
    'scipy.stats',
    'scipy.stats._continuous_distns',
    # h5py inner pieces
    'h5py._hl',
    'h5py._hl.files',
    'h5py._hl.dataset',
    # matplotlib backends (used by cortex_render_videos for the
    # per-session collapse.mp4 / passfail.mp4 outputs)
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_svg',
    # imageio_ffmpeg — v1.1.2 cross-platform ffmpeg shipping. Python
    # module is small (~10 KB); the platform binary is staged via
    # collect_data_files('imageio_ffmpeg') in datas above.
    'imageio_ffmpeg',
    # pyqtgraph — the viewer uses pg.{ColorMap, ImageItem, PlotWidget,
    # TextItem} (verified by grep against scripts/eeg_bank_viewer.py).
    # pyqtgraph's package __init__ resolves these via lazy proxies that
    # the AST tracer can miss; pin the concrete class modules.
    'pyqtgraph.graphicsItems',
    'pyqtgraph.graphicsItems.ViewBox',
    'pyqtgraph.graphicsItems.ImageItem',
    'pyqtgraph.graphicsItems.TextItem',
    'pyqtgraph.widgets',
    'pyqtgraph.widgets.PlotWidget',
    'pyqtgraph.colormap',
    # cortex modules under repo's scripts/
    'cortex_diagnostics',
    'cortex_engine_inputs',
    'cortex_policy',
    'cortex_render_videos',
    'cortex_storage',
    'render_engine_explainer',                # v1.1.3 third MP4 renderer
    'session_controller',
    # vendored packages bundled via datas
    'engine',
    'engine.core_mcmc',
    'engine.auroc',
    # PyYAML — cortex_storage / cortex_policy do function-body `import yaml`
    # to load cortex_config.yaml and cert_config.yaml. PyInstaller's PyYAML
    # hook usually catches this but listing it explicitly is cheap insurance.
    'yaml',
    '_yaml',
    # Dropbox SDK — cortex_storage._dropbox_upload does function-body
    # `import dropbox`. The SDK lazy-loads ~25 endpoint submodules from its
    # __init__; explicitly collecting everything below.
    'dropbox',
] + collect_submodules('dropbox') + [
    # numpy 2.0 pickle-compat shim. Sigma_l_fitted.npy's pickle references
    # numpy._core.multiarray._reconstruct (a numpy 2.x path); numpy 1.26.x
    # ships a `_core/` shim package whose submodules forward to numpy.core.*
    # lazily, which the static AST tracer cannot see. Without these, the
    # bundled app crashes with ModuleNotFoundError: 'numpy._core' on
    # load_fitted_Sigma() — the v1.0.2 shipped bug.
    'numpy._core',
    'numpy._core.multiarray',
    'numpy._core.umath',
    'numpy._core._multiarray_umath',
    'numpy._core._dtype',
    'numpy._core._dtype_ctypes',
    'numpy._core._internal',
] + collect_submodules('numpy._core')

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
            'CFBundleShortVersionString': '1.1.2',
            'CFBundleName': 'CORTEX',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '12.0',
            'NSHumanReadableCopyright': 'BDSP / Westover Lab',
        },
    )
