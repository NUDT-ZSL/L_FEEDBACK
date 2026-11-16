@echo off
setlocal
cd /d "%~dp0"
python tools\generate_samples.py --out samples
exit /b %errorlevel%
