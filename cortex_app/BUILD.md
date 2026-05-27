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

2. **`data/eeg_bank.h5`** — the curated 300 IIIC + 100 spike test bank
   (~450 MB; v1.1.0). Fetch from the repo's own `build-data-v2` release:

       cd cortex_app && bash fetch_test_bank.sh

   No AWS needed — uses your normal `gh` CLI auth.

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

### One-time setup: add ONE secret

Go to **Settings → Secrets and variables → Actions → New repository
secret** and add:

| Secret name | Value |
|---|---|
| `CORTEX_CONFIG_YAML` | paste the entire contents of your local `cortex_config.yaml` (the one with the Dropbox app_key / app_secret / refresh_token) |

That's it. No AWS keys needed.

The 170 MB test bank lives as an asset on the `build-data-v1` release
of this repo; CI pulls it via `gh release download` using the built-in
`GITHUB_TOKEN`. End-users never touch S3 or the build-data release —
the bank is fully embedded inside the .app/.exe by PyInstaller.

To refresh the test bank for a new pilot wave: upload the new
`eeg_bank.h5` as `build-data-v2`, then change the tag reference in
[`.github/workflows/cortex-release.yml`](../.github/workflows/cortex-release.yml)
(two `gh release download build-data-v1` lines).

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

The CI workflow (`.github/workflows/cortex-release.yml`) already
populates the release body with the text below — keep both in sync if
you edit it. The macOS instructions reflect the **Sequoia+** flow
(Apple removed the inline right-click → Open workaround in Sequoia;
users now have to authorize via System Settings).

```markdown
## CORTEX adaptive EEG certification test

### Downloads
- **macOS:** `CORTEX-mac.dmg` — see "Opening CORTEX on macOS" below
- **Windows:** `CORTEX-windows.zip` — see "Opening CORTEX on Windows" below

### Opening CORTEX on macOS

CORTEX is not Apple-signed (this is an internal pilot, not a commercial app),
so the **first launch needs one-time approval**. On macOS Sequoia and later,
Apple removed the inline "Open Anyway" button from the warning dialog — you
have to authorize it from System Settings.

**You MUST drag CORTEX to /Applications first — don't double-click it inside
the disk-image window.** macOS quietly runs apps launched from a downloaded
location in a read-only "translocated" copy and *caches* that copy, so a
buggy old version can keep getting re-run even after you download a fixed
update. Installing to /Applications avoids the whole class of problem.

1. Double-click `CORTEX-mac.dmg` to open it. A window appears with **CORTEX**
   on the left and an **Applications** shortcut on the right.
2. **Drag the CORTEX icon onto the Applications shortcut.** Don't double-click
   CORTEX inside this window.
3. Eject the disk image (Finder → arrow next to "CORTEX" in the sidebar).
   Open **Applications**, double-click **CORTEX**.
4. macOS shows *"CORTEX Not Opened — Apple could not verify CORTEX is free of
   malware..."* Click **Done**.
5. Open **System Settings → Privacy & Security**. Scroll to the **Security**
   section near the bottom.
6. You'll see *"CORTEX was blocked to protect your Mac."* Click **Open Anyway**
   (enter your password if prompted).
7. Try opening **CORTEX** from /Applications again. This time you get a new
   dialog *with* an **Open** button — click **Open**.
8. macOS remembers — no further warnings on subsequent launches.

If you're comfortable with Terminal, this single command run after dragging
to /Applications also works (skips steps 4–7):

    xattr -d com.apple.quarantine /Applications/CORTEX.app

### Opening CORTEX on Windows

1. Extract the `.zip`, open the `CORTEX/` folder, double-click `CORTEX.exe`.
2. If Windows SmartScreen warns *"Windows protected your PC"*, click
   **More info** then **Run anyway**. (Only the first launch.)

### Requirements

macOS 12 (Monterey) or later, or Windows 10/11. No Python install needed —
everything is bundled.

### Reporting bugs

Anything that crashes, looks visually wrong, is confusing, or feels slow —
note it and email the study team. That feedback is the whole point of this
internal round.
```

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
