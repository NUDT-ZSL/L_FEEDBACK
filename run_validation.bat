@echo off
setlocal
set ROOT=%~dp0
set EXE=%ROOT%build\windows\Release\character_headless.exe
if not exist "%EXE%" (
  echo [ERROR] Build the project first with build_windows.bat
  exit /b 10
)

"%EXE%" --headless --seed 1337 --logic-frames 360 --render-fps 120 --max-render-frames 720 --switch-time 1.25 --equipment iron --report "%ROOT%reports\validation-report.json" --snapshot "%ROOT%reports\state-snapshot.json"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Report:   reports\validation-report.json
echo Snapshot: reports\state-snapshot.json
endlocal

