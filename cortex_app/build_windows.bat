@echo off
REM Build the CORTEX test-taker as a standalone Windows .exe folder.
REM
REM Result lands in:
REM     cortex_app\dist\CORTEX\CORTEX.exe
REM
REM Distribution: zip the entire cortex_app\dist\CORTEX folder and
REM attach the .zip as a GitHub Release asset. First-time Windows users
REM may see a SmartScreen "Windows protected your PC" warning because
REM the .exe is unsigned — they click "More info" then "Run anyway".
cd /d "%~dp0"

echo === CORTEX build (Windows) ===

REM Python 3.11 specifically.
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" 2>nul
if errorlevel 1 (
    echo ERROR: need Python 3.11 to build CORTEX.
    echo Install from https://www.python.org/downloads/release/python-3119/
    pause
    exit /b 1
)
for /f "delims=" %%i in ('python --version') do echo Using %%i

if not exist cortex_config.yaml (
    echo ERROR: cortex_app\cortex_config.yaml is missing.
    echo Copy cortex_config.yaml.template to cortex_config.yaml and fill
    echo in the Dropbox credentials, then re-run this script.
    pause
    exit /b 1
)
if not exist data\eeg_bank.h5 (
    echo data\eeg_bank.h5 not found.
    echo Run `bash fetch_test_bank.sh` from Git Bash or WSL, or copy the
    echo file manually from s3://bdsp-opendata-credentialed/eeg-test/test_h5.h5
    pause
    exit /b 1
)

if not exist build_venv (
    echo Creating build_venv ...
    python -m venv build_venv
)
call build_venv\Scripts\activate.bat
echo Installing build deps ...
pip install --quiet --upgrade pip
pip install --quiet -r requirements-cortex.txt
pip install --quiet pyinstaller

echo Running PyInstaller ...
pyinstaller --clean --noconfirm cortex.spec
if errorlevel 1 (
    pause
    exit /b 1
)

echo.
echo === build done ===
echo Distribute the entire folder: dist\CORTEX\
echo Zip it before sending (Right-click -> Send to -> Compressed (zipped) folder).
pause
