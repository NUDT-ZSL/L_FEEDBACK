param(
    [string]$BuildDir = (Join-Path $PSScriptRoot "..\build\Release"),
    [string]$WorkDir = (Join-Path $PSScriptRoot "..\work\two-runs")
)

$ErrorActionPreference = "Stop"
$baker = Join-Path $BuildDir "baker.exe"
$sceneGen = Join-Path $BuildDir "scene_gen.exe"

if (Test-Path $WorkDir) {
    Remove-Item -Recurse -Force $WorkDir
}
New-Item -ItemType Directory -Force $WorkDir | Out-Null

& $sceneGen --out (Join-Path $WorkDir "scene.bin") --seed 22
& $sceneGen --out (Join-Path $WorkDir "scene_again.bin") --seed 22
if ((Get-FileHash (Join-Path $WorkDir "scene.bin")).Hash -ne
    (Get-FileHash (Join-Path $WorkDir "scene_again.bin")).Hash) {
    throw "scene generator is not deterministic"
}

& $baker --input (Join-Path $WorkDir "scene.bin") `
    --output (Join-Path $WorkDir "out1.rgba") `
    --snapshot (Join-Path $WorkDir "snapshot.bin") |
    Tee-Object -FilePath (Join-Path $WorkDir "run1.log")
& $baker --input (Join-Path $WorkDir "scene.bin") `
    --output (Join-Path $WorkDir "out2.rgba") `
    --snapshot (Join-Path $WorkDir "snapshot.bin") |
    Tee-Object -FilePath (Join-Path $WorkDir "run2.log")

$hash1 = (Get-FileHash (Join-Path $WorkDir "out1.rgba")).Hash
$hash2 = (Get-FileHash (Join-Path $WorkDir "out2.rgba")).Hash
if ($hash1 -ne $hash2) {
    throw "two-run output differs"
}
Write-Host "two-run output and scene generation: identical"
Write-Host "artifacts: $WorkDir"
