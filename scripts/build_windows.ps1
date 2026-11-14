param(
  [switch]$Verify
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

if (-not $env:VCPKG_ROOT) {
  throw "VCPKG_ROOT is not set. Point it to the vcpkg checkout."
}

$Toolchain = Join-Path $env:VCPKG_ROOT "scripts/buildsystems/vcpkg.cmake"
if (-not (Test-Path $Toolchain)) {
  throw "vcpkg toolchain not found: $Toolchain"
}

Push-Location $Root
try {
  cmake --preset windows-x64-release
  cmake --build --preset windows-x64-release
  ctest --preset windows-x64-release

  $Exe = Join-Path $Root "build/windows-x64/bin/lf_character_engine.exe"
  if (-not (Test-Path $Exe)) {
    throw "application executable not found: $Exe"
  }

  if ($Verify) {
    & $Exe --verify --data-dir (Join-Path $Root "artifacts/windows-verify") `
      --width 640 --height 480
    if ($LASTEXITCODE -ne 0) {
      throw "pixel verification failed with exit code $LASTEXITCODE"
    }
  }
} finally {
  Pop-Location
}

