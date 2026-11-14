param(
    [string]$BuildDir = "$PSScriptRoot/../build",
    [string]$OutputDir = "$PSScriptRoot/../artifacts/offline",
    [string]$Config = "$PSScriptRoot/../config/render.json",
    [switch]$Gpu,
    [switch]$SkipConfigure
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path "$PSScriptRoot/..").Path
$BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$Config = [System.IO.Path]::GetFullPath($Config)

if (-not $SkipConfigure) {
    cmake -S $Root -B $BuildDir -DDEFERRED_BUILD_TESTS=ON
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }
}
cmake --build $BuildDir --config Release
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

function Find-Binary([string]$Name) {
    $candidates = @(
        (Join-Path $BuildDir "Release/$Name.exe"),
        (Join-Path $BuildDir "$Name.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw "Cannot find $Name.exe under $BuildDir"
}

$Fog = Find-Binary "deferred_fog"
$Verify = Find-Binary "deferred_verify"
$Tests = Find-Binary "deferred_core_tests"

& $Tests
if ($LASTEXITCODE -ne 0) { throw "Core tests failed" }

$Backend = if ($Gpu) { "opengl" } else { "cpu" }
& $Fog --config $Config --offline --backend $Backend --output $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Offline rendering failed" }

if ($Gpu) {
    $CpuDir = Join-Path $Root "artifacts/offline_cpu"
    & $Fog --config $Config --offline --backend cpu --output $CpuDir
    & $Verify $OutputDir $CpuDir | Set-Content (Join-Path $Root "artifacts/verification_report.json")
} else {
    & $Verify $OutputDir | Set-Content (Join-Path $Root "artifacts/verification_report.json")
}
if ($LASTEXITCODE -ne 0) { throw "Artifact verification failed" }

function Assert-Error {
    param([string]$Name, [int]$Expected, [string[]]$Arguments)
    $stdout = Join-Path $Root "artifacts/$Name.stdout"
    $stderr = Join-Path $Root "artifacts/$Name.stderr"
    & $Fog @Arguments 1>$stdout 2>$stderr
    $actual = $LASTEXITCODE
    $message = Get-Content $stderr -Raw
    if ($actual -ne $Expected) {
        throw "$Name expected exit $Expected, got $actual ($message)"
    }
    Write-Host "$Name exit=$actual $message"
}

Assert-Error "invalid_steps" 2 @("--config", (Join-Path $Root "config/invalid_steps.json"), "--offline", "--backend", "cpu")
Assert-Error "invalid_syntax" 2 @("--config", (Join-Path $Root "config/invalid_syntax.json"), "--offline", "--backend", "cpu")
Assert-Error "invalid_type" 2 @("--config", (Join-Path $Root "config/invalid_type.json"), "--offline", "--backend", "cpu")
Assert-Error "gbuffer_failure" 3 @("--config", $Config, "--offline", "--backend", "cpu", "--inject-gbuffer-failure")
Assert-Error "shader_failure" 4 @("--config", $Config, "--offline", "--backend", "cpu", "--inject-shader-failure")

$ErrorReport = @{
    schema = "deferred-fog-error-codes/v1"
    results = @(
        @{ case = "invalid_steps"; code = "E_CONFIG_PARSE"; exit = 2 },
        @{ case = "invalid_syntax"; code = "E_CONFIG_PARSE"; exit = 2 },
        @{ case = "invalid_type"; code = "E_CONFIG_PARSE"; exit = 2 },
        @{ case = "gbuffer_failure"; code = "E_GBUFFER_FAIL"; exit = 3 },
        @{ case = "shader_failure"; code = "E_SHADER_COMPILE"; exit = 4 }
    )
}
$ErrorReport | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $Root "artifacts/error_codes.json")
Write-Host "validation complete"
