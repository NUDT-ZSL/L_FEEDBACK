@echo off
REM Windows cmd entry point for the locdub pipeline.
REM All arguments after the script name are forwarded to the Python module,
REM e.g. run_locdub.bat --timeline timeline.json --video source.mp4 ...
python -m locdub %*
echo locdub exit code: %ERRORLEVEL%
exit /b %ERRORLEVEL%
