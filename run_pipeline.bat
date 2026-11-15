@echo off
REM Windows command line entry point for the audio alignment pipeline.
REM Usage: run_pipeline.bat manifest.json output.wav [-c config.json] [-r report.json]
setlocal
python -m audiopipeline.cli %*
exit /b %ERRORLEVEL%
