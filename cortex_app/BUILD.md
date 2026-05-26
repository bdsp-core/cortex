# Building & releasing CORTEX

How to produce the click-to-run CORTEX test-taker app and ship it as a
GitHub Release asset. Both macOS and Windows builds are **unsigned** —
users see a one-time Gatekeeper / SmartScreen warning on first launch
and dismiss it by right-clicking → Open (macOS) or "More info" → "Run
anyway" (Windows). No Apple Developer Program subscription required.

## What gets shipped

| Platform | Output | Distribute by | First-launch UX |
|---|---|---|---|
| macOS | `dist/CORTEX.dmg` | attach to GitHub Release | right-click → **Open** once |
| Windows | `dist/CORTEX/` (zip the folder) | attach the `.zip` to GitHub Release | "More info" → **Run anyway** |

The end user just downloads, double-clicks. No Python install. No
terminal. The unsigned-app warning is the only friction.

## Before you build (one-time setup)

You need **Python 3.11** on the build machine. Check with
`python3 --version`. If not, install from
<https://www.python.org/downloads/release/python-3119/>.

You also need two files in this directory that are gitignored
(deliberately — they carry secrets / are too big for git):

1. **`cortex_config.yaml`** — copy from the committed template and
   fill in the Dropbox credentials:

       cp cortex_app/cortex_config.yaml.template cortex_app/cortex_config.yaml
       # then edit cortex_app/cortex_config.yaml to add app_key,
       # app_secret, refresh_token (Dropbox app must be "App folder"
       # scoped with only files.content.write permission).

2. **`data/eeg_bank.h5`** — the curated 100 IIIC + 100 spike test bank
   (~170 MB). Fetch from S3:

       cd cortex_app && bash fetch_test_bank.sh

   That pulls `s3://bdsp-opendata-credentialed/eeg-test/test_h5.h5` and
   stages it as `cortex_app/data/eeg_bank.h5`. Uses the `opendata`
   (read-only) AWS profile.

## Build for macOS

    cd cortex_app
    bash build_mac.sh

About 2–5 minutes. Produces:

- `dist/CORTEX.app` — standalone macOS app bundle (~300–500 MB)
- `dist/CORTEX.dmg` — distributable disk image (compressed, smaller)

Test locally by double-clicking `dist/CORTEX.app`. First launch:
right-click → **Open** to bypass Gatekeeper.

## Build for Windows

On a Windows machine with Python 3.11:

    cd cortex_app
    build_windows.bat

Produces `dist/CORTEX/CORTEX.exe` plus its support files in
`dist/CORTEX/`. Zip the **whole `CORTEX/` folder** (not just the .exe)
before distributing — `Right-click → Send to → Compressed (zipped) folder`.

## Cutting a release

1. Bump the version in [`cortex.spec`](./cortex.spec) (`CFBundleShortVersionString`)
   and update the bundle identifier if it's a major rev.
2. Build on macOS → `CORTEX-vX.Y.dmg` (rename `dist/CORTEX.dmg`).
3. Build on Windows → `CORTEX-vX.Y-windows.zip` (zip `dist/CORTEX/`).
4. On GitHub: **Releases → Draft a new release** → tag `cortex-vX.Y` →
   title "CORTEX vX.Y" → attach both assets → publish.
5. Hand the release URL to the test-takers. They click the asset that
   matches their OS, download, double-click.

## Release notes template

> **CORTEX v1.0 — internal pilot**
>
> Adaptive EEG certification test. Runs locally, uploads results
> automatically.
>
> **Download:**
> - macOS: `CORTEX-v1.0.dmg`
> - Windows: `CORTEX-v1.0-windows.zip`
>
> **First launch:**
> - macOS: open the DMG, drag CORTEX to Applications, right-click
>   CORTEX → **Open** (only the first time — macOS will remember).
> - Windows: unzip, open the folder, double-click `CORTEX.exe`. If
>   SmartScreen warns, click "More info" → "Run anyway" once.
>
> **Requirements:** macOS 12+ or Windows 10+. No Python install needed.
>
> **Reporting bugs:** anything that crashes, looks wrong, or feels
> slow → email the study team.

## Why unsigned (and what it would take to sign)

Code signing + Apple notarization removes the Gatekeeper warning
entirely. It needs an Apple Developer Program subscription
(\$99/yr), a Developer ID Certificate, and a notarization pass in
the build script. Same Windows side: an EV code-signing certificate
(\$100s/yr) removes SmartScreen.

For an internal pilot, the unsigned route with right-click-Open is
fine — the [sleep-yoda project](https://github.com/bdsp-core/sleep-yoda)
ships the same way.

## Troubleshooting

**`pyinstaller: command not found`** — `pip install pyinstaller` is in
the build scripts. If it failed silently, run it manually inside
`build_venv`.

**`ModuleNotFoundError` at runtime** — PyInstaller missed a hidden
import. Add the missing module to the `hiddenimports=[...]` list in
[`cortex.spec`](./cortex.spec), re-build.

**App opens but crashes immediately** — run from terminal to see the
real error:

    ./dist/CORTEX.app/Contents/MacOS/CORTEX           # macOS
    dist\CORTEX\CORTEX.exe                            # Windows

**App is much bigger than expected** — check `excludes=[...]` in
`cortex.spec`. The big offenders (torch, jax, mne, neurokit2) are
already excluded; if you add a new dep, audit its transitive bloat.

**`Errno 13: Permission denied` on `dist/`** — old build left files
locked; `rm -rf cortex_app/dist cortex_app/build` and retry.
