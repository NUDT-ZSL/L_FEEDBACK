param(
    [string]$BuildDir = "build",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

cmake -S . -B $BuildDir
cmake --build $BuildDir --config $Config -j

$candidate = Join-Path $BuildDir "$Config/canyon_renderer.exe"
if (-not (Test-Path $candidate)) {
    $candidate = Join-Path $BuildDir "canyon_renderer.exe"
}
if (-not (Test-Path $candidate)) {
    throw "canyon_renderer.exe was not produced in $BuildDir"
}

& $candidate --scene scenes/canyon_demo.json --out "$BuildDir/out_full" --mode full
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $candidate --scene scenes/canyon_demo.json --out "$BuildDir/out_incremental" --mode incremental
exit $LASTEXITCODE
