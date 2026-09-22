@echo off
REM DrowsyGuard AI - one-click setup (Windows). Needs Python 3.10.
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
  py -3.10 --version >nul 2>&1
  if %errorlevel%==0 (
    set "PYCMD=py -3.10"
  ) else (
    goto NEED310
  )
) else (
  python --version >nul 2>&1
  if errorlevel 1 goto NOPYTHON
  python -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,10) else 1)"
  if errorlevel 1 goto NEED310
  set "PYCMD=python"
)

echo [1/3] Creating virtual environment...
%PYCMD% -m venv .venv-drowsy
call .venv-drowsy\Scripts\activate.bat
echo [2/3] Installing libraries (light, no torch)...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo [3/3] Done! Double-click run.bat to start the Dashboard.
pause
exit /b 0

:NEED310
echo.
echo  You need Python 3.10 (project is verified on 3.10).
echo  Opening download page - install it, check "Add python to PATH", then run setup.bat again.
start https://www.python.org/downloads/release/python-31011/
pause
exit /b 1

:NOPYTHON
echo.
echo  No Python found. Install Python 3.10 first:
echo  https://www.python.org/downloads/release/python-31011/
echo  (check "Add python to PATH" during install)
pause
exit /b 1
