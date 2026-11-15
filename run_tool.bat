@echo off
setlocal
if not exist .venv\Scripts\python.exe (
    echo Create the virtual environment first:
    echo   py -3.11 -m venv .venv
    echo   .venv\Scripts\python -m pip install -e .[test]
    exit /b 1
)
.venv\Scripts\python -m material_tool %*
exit /b %ERRORLEVEL%

