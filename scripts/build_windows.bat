@echo off
setlocal

set "ROOT=%~dp0.."
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSPATH="
set "VSVERSION="

if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%I"
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion`) do set "VSVERSION=%%I"
)

if not defined VSPATH (
    set "VSPATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools"
    set "VSVERSION=16.0"
)

set "VSDEV=%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
set "CMAKE=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

if not exist "%VSDEV%" (
    echo error: Visual Studio C++ build tools were not found. 1>&2
    exit /b 1
)
if not exist "%CMAKE%" (
    where cmake >nul 2>nul
    if errorlevel 1 (
        echo error: CMake was not found. 1>&2
        exit /b 1
    )
    set "CMAKE=cmake"
)

for /f "tokens=1 delims=." %%V in ("%VSVERSION%") do set "VSMAJOR=%%V"
if "%VSMAJOR%"=="17" (
    set "GENERATOR=Visual Studio 17 2022"
) else (
    set "GENERATOR=Visual Studio 16 2019"
)

call "%VSDEV%" >nul
if errorlevel 1 exit /b 1

"%CMAKE%" -S "%ROOT%" -B "%ROOT%\build" -G "%GENERATOR%" -A x64
if errorlevel 1 exit /b 1

"%CMAKE%" --build "%ROOT%\build" --config Release --parallel
if errorlevel 1 exit /b 1

echo Built: %ROOT%\build\Release\deterministic_csm.exe
endlocal

