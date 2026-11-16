@echo off
setlocal
set "PYTHONPATH=%~dp0;%PYTHONPATH%"
python -m broadcast_master %*
exit /b %ERRORLEVEL%

