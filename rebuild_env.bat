@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================================
echo Rebuild Python Environment for ai-quinter
echo Project: %cd%
echo ============================================================

echo.
echo [1/7] Checking dangerous filename conflicts...

if exist requests.py (
    echo [ERROR] Found requests.py in project root.
    echo This file will shadow the real requests package.
    echo Please rename requests.py to something else, for example request_client.py
    pause
    exit /b 1
)

if exist requests (
    echo [ERROR] Found folder named requests in project root.
    echo This folder will shadow the real requests package.
    echo Please rename it.
    pause
    exit /b 1
)

echo [OK] No requests.py or requests folder found.

echo.
echo [2/7] Removing old broken .venv...

if exist .venv (
    attrib -R .venv\* /S /D >nul 2>nul
    rmdir /S /Q .venv
)

if exist .venv (
    echo [ERROR] Failed to delete .venv.
    echo Close VS Code, PyCharm, Python, and all terminals using this environment.
    echo Then run this script again.
    pause
    exit /b 1
)

echo [OK] Old .venv removed.

echo.
echo [3/7] Creating new virtual environment...

py -3 -m venv .venv

if errorlevel 1 (
    echo [WARN] py -3 failed. Trying python -m venv...
    python -m venv .venv
)

if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    echo Please check whether Python is installed correctly.
    pause
    exit /b 1
)

echo [OK] New .venv created.

echo.
echo [4/7] Activating environment...

call .venv\Scripts\activate.bat

if errorlevel 1 (
    echo [ERROR] Failed to activate .venv.
    pause
    exit /b 1
)

echo [OK] Environment activated.

echo.
echo [5/7] Upgrading pip, setuptools, wheel...

python -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip tools.
    pause
    exit /b 1
)

echo.
echo [6/7] Installing project dependencies...

if exist requirements.txt (
    python -m pip install -r requirements.txt
) else (
    echo [WARN] requirements.txt not found. Installing minimum packages...
    python -m pip install requests python-dotenv
)

if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [7/7] Testing requests package...

python -c "import requests; print('requests file:', requests.__file__); print('requests version:', requests.__version__); print('has Session:', hasattr(requests, 'Session')); assert hasattr(requests, 'Session')"

if errorlevel 1 (
    echo [ERROR] requests is still broken.
    echo Check whether there is another file named requests.py somewhere in your project.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] Environment rebuilt successfully.
echo ============================================================
echo.
echo To run your project:
echo call .venv\Scripts\activate.bat
echo python "version 0.0.2.py"
echo.
pause