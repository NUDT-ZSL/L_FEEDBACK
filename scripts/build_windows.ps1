param(
  [string]$Generator = "Visual Studio 17 2022",
  [string]$Configuration = "Release",
  [string]$BuildDirectory = "build"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot

cmake -S $Root -B (Join-Path $Root $BuildDirectory) -G $Generator -A x64
cmake --build (Join-Path $Root $BuildDirectory) --config $Configuration

$Executable = Join-Path $Root "$BuildDirectory\bin\$Configuration\terrain_renderer.exe"
Write-Host "Built: $Executable"
& $Executable --version

