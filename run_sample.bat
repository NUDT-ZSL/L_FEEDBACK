@echo off
setlocal
pushd "%~dp0"
python render_terrain.py --config sample_config.json %*
set "RESULT=%ERRORLEVEL%"
popd
exit /b %RESULT%

