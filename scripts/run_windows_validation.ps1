param(
    [string]$BuildDir = "build",
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

ctest --test-dir $BuildDir -C $Config --output-on-failure

$sim = Join-Path $BuildDir "$Config\lightcluster_sim.exe"
$gpu = Join-Path $BuildDir "$Config\lightcluster.exe"

python scripts\validate_cli.py $sim

& $gpu --frames 120 --sync deferred --delay 5 --pool-capacity 8 `
    --seed 20261115 --debug --headless | Tee-Object -FilePath gpu_deferred.jsonl
if ($LASTEXITCODE -ne 0) {
    throw "GPU deferred validation failed with exit code $LASTEXITCODE"
}

& $gpu --frames 3 --sync immediate --debug --headless --inject-debug-fault |
    Tee-Object -FilePath gpu_debug_fault.jsonl
if ($LASTEXITCODE -ne 75) {
    throw "Expected debug mismatch exit code 75, got $LASTEXITCODE"
}

Write-Host "Windows validation passed (sim, GPU deferred, GPU debug exit code)."
