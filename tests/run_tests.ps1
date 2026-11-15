param(
    [string]$Exe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($Exe)) {
    $candidates = @(
        ".\build\Release\pbr_renderer.exe",
        ".\build\pbr_renderer.exe",
        ".\build2\pbr_renderer.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            $Exe = $candidate
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($Exe) -or -not (Test-Path $Exe)) {
    throw "Renderer executable not found. Pass -Exe <path>."
}
$Exe = (Resolve-Path $Exe).Path
$Tmp = Join-Path $ProjectRoot ("build\test-temp-" + $PID)
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
$PassCount = 0

function Invoke-Renderer {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][int]$Expected
    )
    $stdout = Join-Path $Tmp ($Name + ".stdout.txt")
    $stderr = Join-Path $Tmp ($Name + ".stderr.txt")
    & $Exe @Arguments > $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne $Expected) {
        Get-Content $stdout, $stderr -ErrorAction SilentlyContinue | Write-Host
        throw "$Name expected exit $Expected, got $exitCode"
    }
    $script:PassCount++
    Write-Host "PASS: $Name exit $exitCode"
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Path
    )
    $json = $Object | ConvertTo-Json -Depth 30
    [System.IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding($false)))
}

$Base = Get-Content "assets\scene.json" -Raw | ConvertFrom-Json

$Missing = $Base | ConvertTo-Json -Depth 30 | ConvertFrom-Json
$Missing.camera.PSObject.Properties.Remove("f_number")
Write-JsonFile -Object $Missing -Path (Join-Path $Tmp "missing_field.json")

$BadType = $Base | ConvertTo-Json -Depth 30 | ConvertFrom-Json
$BadType.materials[0].roughness = "smooth"
Write-JsonFile -Object $BadType -Path (Join-Path $Tmp "bad_type.json")

$BadId = $Base | ConvertTo-Json -Depth 30 | ConvertFrom-Json
$BadId.spheres[0].material_id = 99
Write-JsonFile -Object $BadId -Path (Join-Path $Tmp "bad_material_id.json")

Invoke-Renderer -Name "invalid_samples" -Expected 1 `
    -Arguments @("--samples", "0", "--output", (Join-Path $Tmp "unused"))
Invoke-Renderer -Name "invalid_tonemap" -Expected 1 `
    -Arguments @("--tonemap", "filmic", "--output", (Join-Path $Tmp "unused"))
Invoke-Renderer -Name "missing_scene" -Expected 2 `
    -Arguments @("--scene", (Join-Path $Tmp "does_not_exist.json"), "--output", (Join-Path $Tmp "unused"))
Invoke-Renderer -Name "missing_field" -Expected 2 `
    -Arguments @("--scene", (Join-Path $Tmp "missing_field.json"), "--output", (Join-Path $Tmp "unused"))
Invoke-Renderer -Name "wrong_type" -Expected 2 `
    -Arguments @("--scene", (Join-Path $Tmp "bad_type.json"), "--output", (Join-Path $Tmp "unused"))
Invoke-Renderer -Name "material_id_out_of_range" -Expected 2 `
    -Arguments @("--scene", (Join-Path $Tmp "bad_material_id.json"), "--output", (Join-Path $Tmp "unused"))
Invoke-Renderer -Name "missing_reference" -Expected 3 `
    -Arguments @("--scene", "assets\scene.json", "--output", (Join-Path $Tmp "missing_reference"),
        "--mode", "validate", "--samples", "2", "--subframes", "1",
        "--reference", (Join-Path $Tmp "no-reference.json"))

$Blocked = Join-Path $Tmp "blocked_parent"
[System.IO.File]::WriteAllText($Blocked, "not a directory")
Invoke-Renderer -Name "output_io_failure" -Expected 3 `
    -Arguments @("--scene", "assets\scene.json", "--output", (Join-Path $Blocked "render"))

Invoke-Renderer -Name "deterministic_a" -Expected 0 `
    -Arguments @("--scene", "assets\scene.json", "--output", (Join-Path $Tmp "det_a"),
        "--samples", "8", "--subframes", "2", "--exposure", "0", "--tonemap", "aces")
Invoke-Renderer -Name "deterministic_b" -Expected 0 `
    -Arguments @("--scene", "assets\scene.json", "--output", (Join-Path $Tmp "det_b"),
        "--samples", "8", "--subframes", "2", "--exposure", "0", "--tonemap", "aces")
$hashA = (Get-FileHash (Join-Path $Tmp "det_a_ev0_aces.hdr") -Algorithm SHA256).Hash
$hashB = (Get-FileHash (Join-Path $Tmp "det_b_ev0_aces.hdr") -Algorithm SHA256).Hash
if ($hashA -ne $hashB) {
    throw "Repeated HDR renders are not byte-identical"
}
$PassCount++
Write-Host "PASS: repeated HDR renders are byte-identical ($hashA)"

$OutputDirectory = Join-Path $Tmp "output_directory"
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
Invoke-Renderer -Name "output_directory" -Expected 0 `
    -Arguments @("--scene", "assets\scene.json", "--output", $OutputDirectory,
        "--samples", "2", "--subframes", "1", "--exposure", "0", "--tonemap", "aces")
if (-not (Test-Path (Join-Path $OutputDirectory "render_ev0_aces.hdr"))) {
    throw "Directory output did not use render as the file prefix"
}
$PassCount++
Write-Host "PASS: directory --output uses render as the file prefix"

foreach ($exposure in @("-1", "0", "1")) {
    foreach ($curve in @("reinhard", "aces", "uncharted2")) {
        Invoke-Renderer -Name ("color_" + $exposure + "_" + $curve) -Expected 0 `
            -Arguments @("--scene", "assets\scene.json", "--output", (Join-Path $Tmp "color"),
                "--samples", "2", "--subframes", "1", "--exposure", $exposure, "--tonemap", $curve)
    }
}

$pngFiles = Get-ChildItem $Tmp -Filter "color_ev*_*.png"
$hdrFiles = Get-ChildItem $Tmp -Filter "color_ev*_*.hdr"
if ($pngFiles.Count -ne 9 -or $hdrFiles.Count -ne 9) {
    throw "Expected 9 PNG and 9 HDR color-pipeline outputs"
}
foreach ($file in $pngFiles) {
    $bytes = [System.IO.File]::ReadAllBytes($file.FullName)
    $width = [System.Net.IPAddress]::NetworkToHostOrder([BitConverter]::ToInt32($bytes, 16))
    $height = [System.Net.IPAddress]::NetworkToHostOrder([BitConverter]::ToInt32($bytes, 20))
    if ($width -ne 512 -or $height -ne 512) {
        throw "Wrong PNG dimensions in $($file.Name): ${width}x${height}"
    }
}
foreach ($file in $hdrFiles) {
    $header = Get-Content $file.FullName -TotalCount 14
    if (-not ($header -contains "-Y 512 +X 512")) {
        throw "Wrong HDR resolution header in $($file.Name)"
    }
    if (-not ($header -match "exposure_ev=") -or -not ($header -match "tonemap=")) {
        throw "Missing HDR parameter comments in $($file.Name)"
    }
}
$PassCount++
Write-Host "PASS: all 3 exposures x 3 tone curves produce 512x512 HDR/PNG"

$LowReport = Join-Path $Tmp "low_report.txt"
Invoke-Renderer -Name "validation_over_threshold" -Expected 0 `
    -Arguments @("--scene", "assets\scene.json", "--output", (Join-Path $Tmp "low"),
        "--mode", "validate", "--samples", "8", "--subframes", "2",
        "--reference", "assets\reference_stats.json", "--report", $LowReport)
if (-not (Select-String -Path $LowReport -Pattern "SUMMARY: FAIL" -Quiet)) {
    throw "Low-sample validation did not report threshold failure"
}
$PassCount++
Write-Host "PASS: out-of-threshold validation writes a FAIL report and exits 0"

$BaselineReport = Join-Path $Tmp "baseline_report.txt"
Invoke-Renderer -Name "validation_baseline" -Expected 0 `
    -Arguments @("--scene", "assets\scene.json", "--output", (Join-Path $Tmp "baseline"),
        "--mode", "validate", "--samples", "256", "--subframes", "8",
        "--reference", "assets\reference_stats.json", "--report", $BaselineReport)
if (-not (Select-String -Path $BaselineReport -Pattern "SUMMARY: PASS" -Quiet)) {
    throw "Baseline validation did not pass"
}
$PassCount++
Write-Host "PASS: baseline validation passes"

Write-Host "ALL TESTS PASSED ($PassCount checks); logs: $Tmp"
