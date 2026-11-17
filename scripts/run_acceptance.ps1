param(
  [string]$Configuration = "Release",
  [string]$BuildDirectory = "build",
  [string]$OutputDirectory = "outputs\acceptance"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $Root "$BuildDirectory\bin\$Configuration\terrain_renderer.exe"
$Output = Join-Path $Root $OutputDirectory

if (-not (Test-Path $Executable)) {
  throw "Renderer not found: $Executable. Run scripts\build_windows.ps1 first."
}

New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Data = Join-Path $Output "data"

& $Executable angles `
  --data-dir $Data `
  --out-dir (Join-Path $Output "angles") `
  --snapshot (Join-Path $Output "angles\snapshot.json")

& $Executable incremental `
  --data-dir $Data `
  --out-dir (Join-Path $Output "incremental") `
  --yaw 0 `
  --step 0.65 `
  --shadow soft `
  --bias-texels 2 `
  --snapshot (Join-Path $Output "incremental\snapshot.json")

python (Join-Path $Root "tests\verify_acceptance.py") `
  --exe $Executable `
  --work-dir (Join-Path $Root "outputs\acceptance_test")

