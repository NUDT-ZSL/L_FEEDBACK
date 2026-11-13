@echo off
setlocal
set ROOT=%~dp0
set IN=%ROOT%scratch\demo\in
set OUT=%ROOT%scratch\demo\out

echo [1/3] Generate deterministic sample dataset
python -m voiceforge generate "%IN%"
if errorlevel 1 exit /b 1

echo [2/3] Synthesize unified voice track and timeline
python -m voiceforge synthesize "%IN%" "%OUT%"
if errorlevel 1 exit /b 1

echo [3/3] Re-run to prove identical summary
python -m voiceforge synthesize "%IN%" "%OUT%.repeat"
if errorlevel 1 exit /b 1

echo Done. Artifacts are under %ROOT%scratch\demo
endlocal

