param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv")) {
    & $PythonLauncher $PythonVersion -m venv .venv
}

$venvPython = Join-Path (Get-Location) ".venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --no-deps -r requirements.txt

Write-Host ""
Write-Host "Installed. Run:"
Write-Host "  .\.venv\Scripts\python.exe -m render_check --instances 6 --seed 1337"

