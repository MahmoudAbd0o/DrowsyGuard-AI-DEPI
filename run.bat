@echo off
REM DrowsyGuard AI - one-click run (Windows). Run setup.bat first.
cd /d "%~dp0"
if not exist ".venv-drowsy\Scripts\activate.bat" (
  echo Run setup.bat first (double-click it).
  pause
  exit /b 1
)
call .venv-drowsy\Scripts\activate.bat
python dashboard.py
pause
