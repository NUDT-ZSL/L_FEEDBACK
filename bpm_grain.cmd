@echo off
setlocal
py -3 "%~dp0bpm_grain.py" %*
exit /b %ERRORLEVEL%

