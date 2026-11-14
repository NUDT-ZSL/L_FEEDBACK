@echo off
REM ===========================================================================
REM Windows 10/11 end-to-end demo:
REM   1. configure with Visual Studio 2022 and FFmpeg dev libraries
REM   2. build the CLI and tests
REM   3. generate deterministic samples (needs Python 3.10+, optional)
REM   4. run the pipeline and ctest
REM
REM Set FFMPEG_ROOT to your FFmpeg 5.1 shared build, e.g.
REM   set FFMPEG_ROOT=C:\ffmpeg-5.1-full-build-shared
REM ===========================================================================
setlocal

if "%FFMPEG_ROOT%"=="" (
  echo [ERROR] Please set FFMPEG_ROOT to the FFmpeg shared build root.
  exit /b 1
)

set ROOT=%~dp0..
set BUILD=%ROOT%\build

cmake -S "%ROOT%" -B "%BUILD%" -G "Visual Studio 17 2022" -A x64 ^
  -DFFmpeg_ROOT="%FFMPEG_ROOT%"
if errorlevel 1 exit /b 1

cmake --build "%BUILD%" --config Release
if errorlevel 1 exit /b 1

python "%ROOT%\tools\generate_samples.py" --output "%BUILD%\samples"
if errorlevel 1 exit /b 1

"%BUILD%\Release\lipsync.exe" ^
  --input "%BUILD%\samples" ^
  --output "%BUILD%\output" ^
  --config "%ROOT%\config\default.json"
if errorlevel 1 exit /b 1

ctest --test-dir "%BUILD%" --build-config Release --output-on-failure
exit /b %errorlevel%
