@echo off
setlocal

echo === UGAI Agent venv rebuild ===

if not exist config.json (
  echo ERROR: config.json not found.
  echo Put this file in the project root, for example D:\ugai_agent_langgraph_v0
  pause
  exit /b 1
)

echo.
echo Checking Python...

set PYTHON_CMD=

py -3 --version >nul 2>nul
if %errorlevel%==0 (
  set PYTHON_CMD=py -3
  goto PY_FOUND
)

python --version >nul 2>nul
if %errorlevel%==0 (
  set PYTHON_CMD=python
  goto PY_FOUND
)

echo ERROR: No normal Python found.
echo Install Python 3.10 or 3.11 and check "Add python.exe to PATH".
pause
exit /b 1

:PY_FOUND
echo Using: %PYTHON_CMD%
%PYTHON_CMD% --version

echo.
echo Python executable:
%PYTHON_CMD% -c "import sys; print(sys.executable)"

%PYTHON_CMD% -c "import sys; p=sys.executable.lower(); raise SystemExit(10 if 'lumerical' in p else 0)"
if %errorlevel%==10 (
  echo.
  echo ERROR: Your Python points to Lumerical embedded Python.
  echo Do not use it for this project.
  echo Install normal Python 3.10 or 3.11, then run this script again.
  pause
  exit /b 1
)

echo.
echo Removing old .venv...
if exist .venv rmdir /s /q .venv

echo.
echo Creating new .venv...
%PYTHON_CMD% -m venv .venv
if not exist .venv\Scripts\python.exe (
  echo ERROR: venv creation failed.
  pause
  exit /b 1
)

echo.
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo Installing requirements...
.venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo Running v1 fix if available...
if exist apply_v1_ascii_indent_fix.py (
  .venv\Scripts\python.exe apply_v1_ascii_indent_fix.py
) else (
  if exist apply_v1_fix_indentation.py (
    .venv\Scripts\python.exe apply_v1_fix_indentation.py
  )
)

echo.
echo Syntax check...
.venv\Scripts\python.exe -m py_compile main_qq.py
for /r core %%f in (*.py) do .venv\Scripts\python.exe -m py_compile "%%f"
for /r adapters %%f in (*.py) do .venv\Scripts\python.exe -m py_compile "%%f"
for /r storage %%f in (*.py) do .venv\Scripts\python.exe -m py_compile "%%f"

echo.
echo DONE.
echo.
echo Start with:
echo .venv\Scripts\python.exe main_qq.py
echo.
pause
