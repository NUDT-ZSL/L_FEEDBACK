@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" tools\make_samples.py --directory samples
if errorlevel 1 exit /b 1

".venv\Scripts\python.exe" -m unittest discover -v
if errorlevel 1 exit /b 1

if not exist "verification_run" mkdir "verification_run"
".venv\Scripts\python.exe" bpm_grain.py --input samples\skill_120_mono.wav --percussion samples\percussion_120_stereo.wav --output verification_run\mix.wav --seed 4242 --log-level INFO
exit /b %ERRORLEVEL%

