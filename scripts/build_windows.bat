@echo off
setlocal
cmake -S . -B build\windows-vs2022 -G "Visual Studio 17 2022" -A x64 || exit /b 1
cmake --build build\windows-vs2022 --config Release || exit /b 1
ctest --test-dir build\windows-vs2022 -C Release --output-on-failure || exit /b 1
echo.
echo Build complete:
echo   build\windows-vs2022\bin\Release\skinned_csm_renderer.exe
endlocal
