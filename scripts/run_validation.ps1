param(
    [switch]$AllowGL45,
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"
$RootDir = Split-Path -Parent $PSScriptRoot
$CaptureDir = Join-Path $RootDir "artifacts\auto_test"
$OverlayDir = Join-Path $CaptureDir "overlay"

cmake -S $RootDir -B (Join-Path $RootDir $BuildDir) `
    -G "Visual Studio 17 2022" -A x64
cmake --build (Join-Path $RootDir $BuildDir) --config Release

$Exe = Get-ChildItem (Join-Path $RootDir $BuildDir) `
    -Recurse -Filter "pcss_shadow_renderer.exe" |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $Exe) {
    throw "pcss_shadow_renderer.exe was not found under $BuildDir"
}

New-Item -ItemType Directory -Force -Path $CaptureDir, $OverlayDir | Out-Null
$Common = @("--auto-test", "--width", "960", "--height", "540", "--no-vsync")
$Gl45 = @()
if ($AllowGL45) {
    $Gl45 = @("--allow-gl45")
}

& $Exe @Common --frames 84 --hide-overlay @Gl45 `
    --capture-dir $CaptureDir 2>&1 |
    Tee-Object -FilePath (Join-Path $CaptureDir "stdout.log")
if ($LASTEXITCODE -ne 0) {
    throw "Auto-test failed with exit code $LASTEXITCODE"
}

& $Exe @Common --frames 20 @Gl45 `
    --capture-dir $OverlayDir 2>&1 |
    Tee-Object -FilePath (Join-Path $OverlayDir "stdout.log")
if ($LASTEXITCODE -ne 0) {
    throw "Overlay auto-test failed with exit code $LASTEXITCODE"
}

$Log = Get-Content (Join-Path $CaptureDir "stdout.log") -Raw
if ($Log -notmatch "\[RESULT\]" -or
    $Log -notmatch "pcss_light_size_delta_percent" -or
    $Log -notmatch "allocated resolution=4096") {
    throw "Validation markers missing from stdout log"
}

$PngCount = (Get-ChildItem $CaptureDir -Filter *.png).Count
if ($PngCount -lt 10) {
    throw "Expected at least 10 scene captures, found $PngCount"
}

Write-Host "Validation complete: $PngCount scene captures + overlay capture."

