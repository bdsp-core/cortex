# Building & releasing CORTEX

How to produce the click-to-run CORTEX test-taker app and ship it as a
GitHub Release asset. Both macOS and Windows builds are **unsigned** —
users see a one-time Gatekeeper / SmartScreen warning on first launch
and dismiss it by right-clicking → Open (macOS) or "More info" → "Run
anyway" (Windows). **No Apple Developer Program subscription required.**

This is the standalone-app build (PyInstaller). Eli's
[`scripts/build_internal_test_zip.py`](../scripts/build_internal_test_zip.py)
is the complementary "bash-launcher zip" build for users willing to
install Python 3.11 themselves; this one is for those who aren't.

## What gets shipped

| Platform | Output | Distribute by | First-launch UX |
|---|---|---|---|
| macOS | `cortex_app/dist/CORTEX.dmg` | attach to GitHub Release | right-click → **Open** once |
| Windows | `cortex_app/dist/CORTEX/` (zip the whole folder) | attach the `.zip` to GitHub Release | "More info" → **Run anyway** |

The end user just downloads, double-clicks. No Python install, no
terminal, no command line — the unsigned-app warning is the only
friction.

## Where the source lives

All CORTEX source code, the config, the engine, and the test bank live
at the **repo root** — same layout that
[`scripts/build_internal_test_zip.py`](../scripts/build_internal_test_zip.py)
and the runtime modules expect:

| Path | Purpose | In git? |
|---|---|---|
| `scripts/eeg_bank_viewer.py` | the GUI entry point | yes |
| `scripts/cortex_*.py`, `scripts/session_controller.py` | engine + storage + policy | yes |
| `engine/`, `calibration/` | vendored byte-equivalent calibration code | yes |
| `cortex_config.example.yaml` | sanitized config template | yes |
| `Sigma_l_fitted.npy` | frozen ℓ-prior | yes |
| `data/labels/iiic_segment_signals.csv` | segment metadata | yes |
| `requirements-cortex.txt` | runtime deps (also used by the bash bundle) | yes |
| `cortex_config.yaml` | live config with Dropbox token | **gitignored** |
| `data/eeg_bank.h5` | curated 170 MB test bank | **gitignored** |

`cortex_app/` only holds the build infrastructure (spec, scripts,
docs) — nothing CORTEX-source-related is duplicated.

## Before you build (one-time setup)

You need **Python 3.11** on the build machine. Check with
`python3.11 --version`. If missing:

- macOS:  `brew install python@3.11`
- Windows: [python.org installer](https://www.python.org/downloads/release/python-3119/)

You also need two files in the **repo root** that are gitignored:

1. **`cortex_config.yaml`** — copy from the committed template and
   fill in the Dropbox credentials:

       cp cortex_config.example.yaml cortex_config.yaml
       # then edit cortex_config.yaml to add app_key, app_secret,
       # refresh_token (use an "App folder"-scoped Dropbox app with
       # ONLY files.content.write permission).

2. **`data/eeg_bank.h5`** — the curated 100 IIIC + 100 spike test bank
   (~170 MB). Fetch from S3:

       cd cortex_app && bash fetch_test_bank.sh

   That pulls `s3://bdsp-opendata-credentialed/eeg-test/test_h5.h5`
   into `data/eeg_bank.h5` at the repo root. Uses the `opendata`
   read-only AWS profile.

## Build for macOS

    cd cortex_app
    bash build_mac.sh

About 2–5 minutes. Produces:

- `cortex_app/dist/CORTEX.app` — standalone macOS app bundle (~450 MB)
- `cortex_app/dist/CORTEX.dmg` — distributable disk image (~300 MB compressed)

Test locally by double-clicking `dist/CORTEX.app`. First launch:
right-click → **Open** to bypass Gatekeeper.

## Build for Windows

On a Windows machine with Python 3.11:

    cd cortex_app
    build_windows.bat

Produces `cortex_app/dist/CORTEX/CORTEX.exe` plus its support files in
`cortex_app/dist/CORTEX/`. Zip the **whole `CORTEX/` folder** (not just
the .exe) before distributing — Right-click → Send to → Compressed
(zipped) folder.

## Cutting a release (the **automated** way — Mac + Windows, no local build needed)

The repo ships a GitHub Actions workflow at
[`.github/workflows/cortex-release.yml`](../.github/workflows/cortex-release.yml)
that:

1. Runs on `macos-latest` to build `CORTEX-mac.dmg`.
2. Runs on `windows-latest` to build `CORTEX-windows.zip`.
3. Attaches both to a new GitHub Release.

You don't need a Windows machine. You don't even need to run the local
build. **Pushing a `cortex-v*` tag does everything.**

### One-time setup: add three secrets

Go to **Settings → Secrets and variables → Actions → New repository
secret** and add:

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | access key for the `opendata` profile (read-only on `bdsp-opendata-credentialed`) |
| `AWS_SECRET_ACCESS_KEY` | matching secret key |
| `CORTEX_CONFIG_YAML` | paste the entire contents of your local `cortex_config.yaml` (the one with the Dropbox app_key / app_secret / refresh_token) |

These only need to be set once; subsequent releases reuse them.

### Cutting a release

    # Bump version in cortex.spec first (CFBundleShortVersionString)
    git commit -am "cortex: bump to v1.0"
    git tag cortex-v1.0
    git push origin main cortex-v1.0   # the tag push triggers the workflow

Then watch it run at **Actions → CORTEX Release Build**. About 10–15
min later there'll be a new release at **Releases** with both assets
attached and the release-notes body populated. Hand the release URL to
the test-takers.

To dry-run without creating a release: use **Actions → CORTEX Release
Build → Run workflow** (manual dispatch builds the artifacts but only
creates the release if it was triggered by a tag push).

## Cutting a release (the **manual** way — if you can't use CI)

If the CI workflow is unavailable, you can do it by hand:

1. Bump version in [`cortex.spec`](./cortex.spec)
   (`CFBundleShortVersionString` + `info_plist`).
2. Build on a Mac → rename `cortex_app/dist/CORTEX.dmg` → `CORTEX-vX.Y.dmg`.
3. Build on a Windows machine → zip `cortex_app/dist/CORTEX/` →
   `CORTEX-vX.Y-windows.zip`.
4. On GitHub: **Releases → Draft a new release** → tag `cortex-vX.Y` →
   attach both assets → publish.

CLI alternative using `gh`:

    gh release create cortex-v1.0 \
        cortex_app/dist/CORTEX-mac.dmg#"macOS (.dmg)" \
        CORTEX-v1.0-windows.zip#"Windows (.zip)" \
        --title "CORTEX v1.0" \
        --notes-file release-notes.md

## Release notes template

> **CORTEX v1.0 — internal pilot**
>
> Adaptive EEG certification test. Runs locally, uploads results
> automatically to a secure Dropbox folder.
>
> **Download:**
> - macOS: `CORTEX-v1.0.dmg`
> - Windows: `CORTEX-v1.0-windows.zip`
>
> **First launch:**
> - macOS: open the DMG, drag CORTEX to Applications, right-click
>   CORTEX → **Open** (only the first time — macOS remembers after).
> - Windows: unzip, open the folder, double-click `CORTEX.exe`. If
>   SmartScreen warns, click "More info" → "Run anyway" once.
>
> **Requirements:** macOS 12+ or Windows 10+. No Python install needed.
>
> **Reporting bugs:** anything that crashes, looks wrong, or feels
> slow — email the study team.

## Why unsigned (and what it would take to sign)

Code signing + Apple notarization removes the Gatekeeper warning
entirely. It needs an Apple Developer Program subscription (\$99/yr),
a Developer ID Certificate, and a notarization pass in the build
script. Windows: an EV code-signing certificate (\$100s/yr) removes
SmartScreen.

For an internal pilot the unsigned route with right-click-Open is
fine — the [sleep-yoda](https://github.com/bdsp-core/sleep-yoda)
project ships the same way and that pattern was the model here.

## Troubleshooting

**`pyinstaller: command not found`** — the build script installs it
inside `build_venv`. If something failed silently, run
`pip install pyinstaller` manually inside the activated venv.

**`ModuleNotFoundError` at runtime** — PyInstaller missed a hidden
import. Add the missing module to `hiddenimports=[...]` in
[`cortex.spec`](./cortex.spec) and re-build.

**App opens then crashes immediately** — run from terminal to surface
the real error:

    ./dist/CORTEX.app/Contents/MacOS/CORTEX           # macOS
    dist\CORTEX\CORTEX.exe                            # Windows

**App is much bigger than expected** — audit `excludes=[...]` in
`cortex.spec`. The big offenders (torch, jax, mne, neurokit2,
statsmodels) are already excluded; if you add a new dep, check its
transitive bloat.

**`Errno 13: Permission denied` on `dist/`** — old build files locked;
`rm -rf cortex_app/dist cortex_app/build` and retry.
