@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
if not exist "artifacts\acceptance" mkdir "artifacts\acceptance"
if not exist "%ROOT%artifacts\acceptance\fixture" mkdir "%ROOT%artifacts\acceptance\fixture"
python "%ROOT%generate_samples.py" --seed 0 --out "%ROOT%artifacts\acceptance\fixture"
if errorlevel 1 exit /b %errorlevel%
python "%ROOT%audit_cli.py" --before "%ROOT%artifacts\acceptance\fixture\patch_before.bin" --after "%ROOT%artifacts\acceptance\fixture\patch_after.bin" --before-manifest "%ROOT%artifacts\acceptance\fixture\before_manifest.json" --after-manifest "%ROOT%artifacts\acceptance\fixture\after_manifest.json" --defects "%ROOT%artifacts\acceptance\fixture\defects.json" --entries "%ROOT%artifacts\acceptance\fixture\entries.json" --output "%ROOT%artifacts\acceptance\audit_report.json"
if errorlevel 1 exit /b %errorlevel%
python -m unittest discover -s "%ROOT%tests" -v
exit /b %errorlevel%
