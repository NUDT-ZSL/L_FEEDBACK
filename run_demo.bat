@echo off
setlocal
set ROOT=%~dp0
set EXE=%ROOT%build\windows\Release\character_demo.exe
if not exist "%EXE%" (
  echo [ERROR] Build the project first with build_windows.bat
  exit /b 10
)
"%EXE%" --seed 1337 --equipment iron --switch-time 1.25 --report "%ROOT%reports\window-report.json" --snapshot "%ROOT%reports\window-snapshot.json"
endlocal

