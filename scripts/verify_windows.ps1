param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$OutputDir = "artifacts/verify-windows"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Root "$BuildDir/$Config/skeletal_material_demo.exe"
$CpuTests = Join-Path $Root "$BuildDir/$Config/cpu_tests.exe"
$Out = Join-Path $Root $OutputDir

if (-not (Test-Path $Exe) -or -not (Test-Path $CpuTests)) {
    throw "Build executables not found. Build the Release configuration first."
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Invoke-App {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $stdout = Join-Path $Out "$Name.stdout"
    $stderr = Join-Path $Out "$Name.stderr"
    & $Exe @Arguments 1> $stdout 2> $stderr
    return @{
        Code = $LASTEXITCODE
        Out = (Get-Content -Raw -Path $stdout)
        Err = (Get-Content -Raw -Path $stderr)
    }
}

function Assert-Contains {
    param([string]$Text, [string]$Needle, [string]$Context)
    if (-not $Text.Contains($Needle)) {
        throw "$Context did not contain: $Needle"
    }
}

Write-Host "[1/8] CPU scene/UBO contract"
& $CpuTests
if ($LASTEXITCODE -ne 0) {
    throw "cpu_tests failed with exit code $LASTEXITCODE"
}

Write-Host "[2/8] GPU self-test and screenshot"
$shot = Join-Path $Out "self-test.ppm"
$result = Invoke-App "self-test" @("--self-test", "--screenshot", $shot)
if ($result.Code -ne 0) {
    throw "GPU self-test failed with exit code $($result.Code): $($result.Err)"
}
Assert-Contains $result.Out "[SELFTEST] PASS" "GPU self-test"
Assert-Contains $result.Out "Visible character pixels" "GPU self-test"
Assert-Contains $result.Out "Shader mode: CustomLighting" "GPU self-test"

Write-Host "[3/8] Shader compile failure fallback"
$fallbackShot = Join-Path $Out "compile-fallback.ppm"
$result = Invoke-App "compile-fallback" @(
    "--self-test", "--inject-shader-compile-failure", "--screenshot", $fallbackShot)
if ($result.Code -ne 0) {
    throw "Compile fallback test failed with exit code $($result.Code)"
}
Assert-Contains $result.Err "E2001/SHADER_COMPILE_FALLBACK" "Compile fallback"
Assert-Contains $result.Out "Shader mode: SafeFallback" "Compile fallback"
Assert-Contains $result.Out "[SELFTEST] PASS" "Compile fallback"

Write-Host "[4/8] Shader link failure fallback"
$result = Invoke-App "link-fallback" @("--self-test", "--inject-shader-link-failure")
if ($result.Code -ne 0) {
    throw "Link fallback test failed with exit code $($result.Code)"
}
Assert-Contains $result.Err "E2002/SHADER_LINK_FALLBACK" "Link fallback"
Assert-Contains $result.Out "Shader mode: SafeFallback" "Link fallback"
Assert-Contains $result.Out "[SELFTEST] PASS" "Link fallback"

Write-Host "[5/8] Context creation failure path"
$result = Invoke-App "context-failure" @("--simulate-context-failure")
if ($result.Code -ne 1003) {
    throw "Expected context failure exit code 1003, got $($result.Code)"
}
Assert-Contains $result.Err "E1003/OPENGL_45_UNAVAILABLE" "Context failure"

Write-Host "[6/8] Procedural resource failure path"
$result = Invoke-App "resource-failure" @("--simulate-resource-failure")
if ($result.Code -ne 3001) {
    throw "Expected resource failure exit code 3001, got $($result.Code)"
}
Assert-Contains $result.Err "E3001/SCENE_RESOURCE_INVALID" "Resource failure"

Write-Host "[7/8] Three-consecutive-frame GL error path"
$result = Invoke-App "gl-error" @("--self-test", "--inject-gl-error")
if ($result.Code -ne 1010) {
    throw "Expected GL error exit code 1010, got $($result.Code)"
}
Assert-Contains $result.Err "E1010/PERSISTENT_GL_ERROR" "GL error watchdog"

Write-Host "[8/8] VSync-off 200 FPS cap"
$result = Invoke-App "fps-cap" @("--benchmark-frames", "120", "--vsync-off")
if ($result.Code -ne 0) {
    throw "FPS cap benchmark failed with exit code $($result.Code)"
}
Assert-Contains $result.Out "vsync=OFF" "FPS cap benchmark"
$match = [regex]::Match($result.Out, "fps=([0-9.]+)")
if (-not $match.Success) {
    throw "FPS cap benchmark did not report an FPS value"
}
$fps = [double]$match.Groups[1].Value
if ($fps -le 0.0 -or $fps -gt 205.0) {
    throw "VSync-off FPS cap exceeded 205 FPS: $fps"
}

Write-Host "All Windows verification checks passed. Outputs: $Out"
