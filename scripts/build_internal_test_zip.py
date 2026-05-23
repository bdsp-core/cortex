"""Build the CORTEX internal-test distribution bundle.

Assembles the minimal runnable test-taker bundle — the CORTEX viewer +
the two engine modules it uses + the test bank + config + venv setup —
into dist/cortex-internal-test.zip. Run from anywhere:

    python scripts/build_internal_test_zip.py

The bundle deliberately excludes the methodology / calibration /
deployment source, the dev test-suite, and the large unused data CSVs
(verified against the runtime import + file-reference closure).
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
BUNDLE = "cortex-internal-test"
DIST = _REPO / "dist"

# The minimal runtime closure — every file the CORTEX test-taker app
# touches at run time, and nothing else.
FILES = [
    "run_cortex_test.sh",
    "venv_requirements.sh",
    "requirements-cortex.txt",
    ".python-version",
    "cortex_config.yaml",            # carries the Dropbox token — by design
    # AD6Policy reads ell_star_unified_v13 from this file. Source-of-truth
    # lives under calibration/; the repo-root cert_config.yaml is a stale
    # stub that lacks the v13 block — must ship the calibration/ copy.
    "calibration/cert_config.yaml",
    "Sigma_l_fitted.npy",
    "README_CORTEX_TEST.md",
    "engine/core_mcmc.py",
    "engine/auroc.py",
    "scripts/eeg_bank_viewer.py",
    "scripts/session_controller.py",
    "scripts/cortex_engine_inputs.py",
    "scripts/cortex_storage.py",
    "scripts/cortex_policy.py",      # TerminationPolicy + default_policy_for
    "scripts/cortex_render_videos.py",  # per-test-taker collapse + passfail MP4
    "data/eeg_bank.h5",
    "data/labels/iiic_segment_signals.csv",
]


def main():
    missing = [f for f in FILES if not (_REPO / f).exists()]
    if missing:
        if "cortex_config.yaml" in missing:
            print("ERROR: cortex_config.yaml not found. Copy "
                  "cortex_config.example.yaml to cortex_config.yaml and add\n"
                  "your Dropbox token first — see docs/CORTEX_DROPBOX_SETUP.md.")
        raise SystemExit(f"Cannot build — missing files: {missing}")

    DIST.mkdir(exist_ok=True)
    stage = DIST / BUNDLE
    if stage.exists():
        shutil.rmtree(stage)
    for rel in FILES:
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO / rel, dst)

    zip_path = DIST / f"{BUNDLE}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(DIST))   # arcname -> cortex-internal-test/...
    shutil.rmtree(stage)

    mb = zip_path.stat().st_size / 1e6
    print(f"built {zip_path}  ({mb:.0f} MB, {len(FILES)} files)")
    print(f"unzips to a single folder: {BUNDLE}/  "
          f"(the test-taker then runs: bash run_cortex_test.sh)")


if __name__ == "__main__":
    main()
