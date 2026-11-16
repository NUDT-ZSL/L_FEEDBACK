@echo off
setlocal

if "%VCPKG_ROOT%"=="" (
  echo [ERROR] VCPKG_ROOT is not set.
  echo Install vcpkg, run its bootstrap script, then set VCPKG_ROOT.
  exit /b 10
)

where cmake >nul 2>nul
if errorlevel 1 (
  echo [ERROR] CMake 3.20+ is not on PATH.
  exit /b 10
)

cmake --preset windows-vcpkg
if errorlevel 1 exit /b %errorlevel%

cmake --build --preset windows-release
if errorlevel 1 exit /b %errorlevel%

ctest --preset windows-tests
if errorlevel 1 exit /b %errorlevel%

echo.
echo Build and tests completed.
echo Demo:       build\windows\Release\character_demo.exe
echo Headless:   build\windows\Release\character_headless.exe
endlocal

