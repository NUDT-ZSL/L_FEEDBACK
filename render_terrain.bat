@echo off
setlocal
py -3.11 "%~dp0render_terrain.py" %*
exit /b %ERRORLEVEL%
