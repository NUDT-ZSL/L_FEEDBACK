@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
    echo Usage: run_mixer.bat path\to\project.json
    exit /b 2
)
python -m mediamixer "%~1"
exit /b %errorlevel%
