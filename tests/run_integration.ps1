param(
    [string]$TestExe = "build\windows\Release\character_renderer.exe",
    [string]$TestRoot = "build\windows\integration_data"
)

$ErrorActionPreference = "Stop"
cmake "-DTEST_EXE=$TestExe" "-DTEST_ROOT=$TestRoot" -P tests/run_integration.cmake
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

