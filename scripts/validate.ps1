param(
    [string]$Exe = "build/windows-vs2022/bin/Release/skinned_csm_renderer.exe",
    [string]$TestExe = "build/windows-vs2022/bin/Release/sr_core_tests.exe",
    [string]$OutDir = "artifacts/validation-windows"
)

$ErrorActionPreference = "Stop"
$failures = 0

function Add-Failure([string]$Message) {
    Write-Host "FAIL: $Message" -ForegroundColor Red
    $script:failures++
}

function Expect-Exit([int]$Expected, [string]$Name, [string[]]$Arguments) {
    $stdout = Join-Path $OutDir "$Name.stdout.txt"
    $stderr = Join-Path $OutDir "$Name.stderr.txt"
    & $Exe @Arguments 1> $stdout 2> $stderr
    $actual = $LASTEXITCODE
    if ($actual -ne $Expected) {
        Add-Failure "$Name expected exit $Expected, got $actual"
    } elseif ((Get-Item $stderr).Length -eq 0) {
        Add-Failure "$Name produced no readable stderr"
    } else {
        $first = Get-Content $stderr -TotalCount 1
        Write-Host "PASS: $Name exited $actual`: $first"
    }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Running CPU tests..."
& $TestExe
if ($LASTEXITCODE -ne 0) {
    Add-Failure "CPU unit tests"
}

$runA = Join-Path $OutDir "run_a"
$runB = Join-Path $OutDir "run_b"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $runA, $runB

$common = @(
    "--start-frame", "0",
    "--frames", "4",
    "--width", "640",
    "--height", "360",
    "--shadow-size", "512",
    "--cascades", "2",
    "--regenerate-frame", "2",
    "--regenerate-variant", "1",
    "--context", "glfw"
)

Write-Host "Rendering deterministic run A..."
& $Exe --output $runA @common 1> (Join-Path $OutDir "run_a.stdout.txt")
if ($LASTEXITCODE -ne 0) { Add-Failure "run A" }

Write-Host "Rendering deterministic run B..."
& $Exe --output $runB @common 1> (Join-Path $OutDir "run_b.stdout.txt")
if ($LASTEXITCODE -ne 0) { Add-Failure "run B" }

Write-Host "Comparing PNG hashes and bytes..."
$hashes = @()
Get-ChildItem (Join-Path $runA "*.png") | ForEach-Object {
    $other = Join-Path $runB $_.Name
    if (-not (Test-Path $other)) {
        Add-Failure "missing file in run B: $($_.Name)"
        return
    }
    $hashA = (Get-FileHash $_.FullName -Algorithm SHA256).Hash
    $hashB = (Get-FileHash $other -Algorithm SHA256).Hash
    if ($hashA -ne $hashB) {
        Add-Failure "PNG mismatch: $($_.Name)"
    }
    $hashes += "$hashA  $($_.Name)"
}

$blocked = Join-Path $OutDir "blocked-output"
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $blocked
Set-Content -Path $blocked -Value "not-a-directory"
Expect-Exit 3 "output_io_error" @("--output", (Join-Path $blocked "child"))
Expect-Exit 5 "shader_error" @(
    "--output", (Join-Path $OutDir "err_shader"),
    "--test-error", "shader", "--context", "glfw")
Expect-Exit 6 "framebuffer_error" @(
    "--output", (Join-Path $OutDir "err_fbo"),
    "--test-error", "framebuffer", "--context", "glfw")
Expect-Exit 7 "cascade_error" @(
    "--output", (Join-Path $OutDir "err_cascade"),
    "--cascades", "1", "--context", "glfw")

$summary = Join-Path $OutDir "validation_summary.txt"
$hashes | Set-Content $summary
Add-Content $summary ""
if ($failures -eq 0) {
    Add-Content $summary "RESULT: PASS"
    Get-Content $summary
    exit 0
}

Add-Content $summary "RESULT: FAIL ($failures failures)"
Get-Content $summary
exit 1
