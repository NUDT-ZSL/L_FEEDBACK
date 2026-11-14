param(
    [string]$BuildDir = (Join-Path $PSScriptRoot "..\build\Release"),
    [string]$WorkDir = (Join-Path $PSScriptRoot "..\work\kill-during-write")
)

$ErrorActionPreference = "Stop"
$baker = Join-Path $BuildDir "baker.exe"
$sceneGen = Join-Path $BuildDir "scene_gen.exe"

if (Test-Path $WorkDir) {
    Remove-Item -Recurse -Force $WorkDir
}
New-Item -ItemType Directory -Force $WorkDir | Out-Null

& $sceneGen --out (Join-Path $WorkDir "scene.bin") --seed 22 | Out-Null
& $baker --input (Join-Path $WorkDir "scene.bin") `
    --output (Join-Path $WorkDir "baseline.rgba") `
    --snapshot (Join-Path $WorkDir "baseline.snap") --sync-io | Out-Null

Copy-Item (Join-Path $WorkDir "baseline.rgba") (Join-Path $WorkDir "out.rgba")
Copy-Item (Join-Path $WorkDir "baseline.snap") (Join-Path $WorkDir "snapshot.bin")

$oldDelay = $env:BAKER_IO_DELAY_MS
$env:BAKER_IO_DELAY_MS = "1000"
$process = Start-Process -FilePath $baker -PassThru -NoNewWindow -ArgumentList @(
    "--input", (Join-Path $WorkDir "scene.bin"),
    "--output", (Join-Path $WorkDir "out.rgba"),
    "--snapshot", (Join-Path $WorkDir "snapshot.bin")
)

$temp = $null
for ($i = 0; $i -lt 500; ++$i) {
    $temp = Get-ChildItem -Path $WorkDir -Filter "out.rgba.tmp.*" -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($null -ne $temp) {
        break
    }
    Start-Sleep -Milliseconds 20
}
if ($null -eq $temp) {
    if (!$process.HasExited) { Stop-Process -Id $process.Id -Force }
    throw "temporary output was not observed"
}

Stop-Process -Id $process.Id -Force
$process.WaitForExit()
if ($null -eq $oldDelay) {
    Remove-Item Env:BAKER_IO_DELAY_MS -ErrorAction SilentlyContinue
} else {
    $env:BAKER_IO_DELAY_MS = $oldDelay
}

if ((Get-FileHash (Join-Path $WorkDir "out.rgba")).Hash -ne
    (Get-FileHash (Join-Path $WorkDir "baseline.rgba")).Hash) {
    throw "SIGKILL path damaged the previous output"
}

& $baker --input (Join-Path $WorkDir "scene.bin") `
    --output (Join-Path $WorkDir "out.rgba") `
    --snapshot (Join-Path $WorkDir "snapshot.bin") |
    Tee-Object -FilePath (Join-Path $WorkDir "resume.log")

if ((Get-FileHash (Join-Path $WorkDir "out.rgba")).Hash -ne
    (Get-FileHash (Join-Path $WorkDir "baseline.rgba")).Hash) {
    throw "rerun output differs from the sequential baseline"
}
Write-Host "forced kill preserved the old output and recovery reproduced the baseline"
