@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Run: powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
  exit /b 1
)
".venv\Scripts\python.exe" -m render_check %*
exit /b %ERRORLEVEL%

