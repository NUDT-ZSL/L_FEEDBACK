param(
    [string]$OutputRoot = "artifacts/validation"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repo ".venv\Scripts\python.exe"
$python = if (Test-Path $venvPython) { $venvPython } else { "python" }
$runDir = Join-Path $repo $OutputRoot
$repeatDir = Join-Path $runDir "repeat"
$config = Join-Path $repo "config\default.json"

Push-Location $repo
try {
    & $python -m radio_pipeline run --config $config --output-dir $runDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python -m radio_pipeline report --run-dir $runDir --output (Join-Path $runDir "report.rebuilt.json")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python -m radio_pipeline verify --run-dir $runDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python -m radio_pipeline run --config $config --output-dir $repeatDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $python scripts\check_binary_determinism.py $runDir $repeatDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "Validation passed: run, report, verify, and binary determinism."
    exit 0
}
finally {
    Pop-Location
}

