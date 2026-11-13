param(
    [Parameter(Mandatory = $false)]
    [string]$Rigskin = "$PSScriptRoot\..\build\Release\rigskin.exe",
    [Parameter(Mandatory = $false)]
    [string]$UnitTests = "$PSScriptRoot\..\build\Release\rigskin_unit_tests.exe"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Work = Join-Path $Root ("build\integration-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Work | Out-Null

Add-Type @"
using System.Runtime.InteropServices;
public static class ConsoleControl {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool GenerateConsoleCtrlEvent(uint ctrlEvent, uint processGroupId);
}
"@

function Fail([string]$Message) {
    Write-Error "FAIL: $Message`nIntegration work directory: $Work"
    exit 1
}

function Pass([string]$Message) {
    Write-Host "PASS: $Message"
}

function Invoke-Rigskin([string]$Snapshot, [string[]]$Arguments) {
    & $Rigskin --snapshot-dir $Snapshot @Arguments | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Fail "rigskin failed for $Snapshot with exit code $LASTEXITCODE"
    }
}

function Get-TreeHashes([string]$Directory) {
    Get-ChildItem -LiteralPath $Directory -Recurse -File |
        Sort-Object FullName |
        ForEach-Object {
            $Relative = $_.FullName.Substring($Directory.Length + 1)
            $Hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
            "$Relative $Hash"
        }
}

function Assert-SameTree([string]$Left, [string]$Right, [string]$Label) {
    $LeftHashes = Get-TreeHashes $Left
    $RightHashes = Get-TreeHashes $Right
    if (Compare-Object $LeftHashes $RightHashes) {
        Fail "$Label trees differ"
    }
}

function Assert-Summary(
    [string]$Snapshot,
    [string]$Status,
    [string]$Reused,
    [string]$Recomputed,
    [string]$Codes) {
    $Summary = Get-Content -LiteralPath (Join-Path $Snapshot "summary.txt") -Raw
    if ($Summary -notmatch "status=$Status") { Fail "summary status in $Snapshot" }
    if ($Summary -notmatch "frames_reused=$Reused") { Fail "summary reused in $Snapshot" }
    if ($Summary -notmatch "frames_recomputed=$Recomputed") { Fail "summary recomputed in $Snapshot" }
    if ($Summary -notmatch "error_codes=$Codes") { Fail "summary codes in $Snapshot" }
}

if (-not (Test-Path -LiteralPath $Rigskin)) { Fail "rigskin not found: $Rigskin" }
if (-not (Test-Path -LiteralPath $UnitTests)) { Fail "unit tests not found: $UnitTests" }

Write-Host "work=$Work"
& $UnitTests
if ($LASTEXITCODE -ne 0) { Fail "unit tests" }
Pass "unit tests"

$Serial = Join-Path $Work "serial"
$Parallel = Join-Path $Work "parallel"
Invoke-Rigskin $Serial @("--threads", "1", "--frames", "16", "--width", "640", "--height", "480")
Invoke-Rigskin $Parallel @("--threads", "8", "--frames", "16", "--width", "640", "--height", "480")
Assert-SameTree $Serial $Parallel "serial/parallel"
Pass "serial/parallel byte-identical output trees"

$ThreeSerial = Join-Path $Work "three-serial"
$ThreeWide = Join-Path $Work "three-wide"
Invoke-Rigskin $ThreeSerial @("--threads", "1", "--frames", "3", "--width", "320", "--height", "240")
Invoke-Rigskin $ThreeWide @("--threads", "64", "--frames", "3", "--width", "320", "--height", "240")
Assert-SameTree $ThreeSerial $ThreeWide "oversubscribed thread"
Pass "threads greater than frames produces no empty or duplicate frames"

Invoke-Rigskin $Parallel @("--threads", "3", "--frames", "16", "--width", "640", "--height", "480")
Assert-Summary $Parallel "complete" "16" "0" "none"
Pass "clean resume reuses all frames"

$Corrupt = Join-Path $Work "corrupt"
Copy-Item -LiteralPath $Serial -Destination $Corrupt -Recurse
$Mesh = Join-Path $Corrupt "frames\frame_0005.mesh"
$Stream = [IO.File]::Open($Mesh, [IO.FileMode]::Append, [IO.FileAccess]::Write)
$Stream.WriteByte(0)
$Stream.Dispose()
Invoke-Rigskin $Corrupt @("--threads", "4", "--frames", "16", "--width", "640", "--height", "480")
Assert-Summary $Corrupt "complete" "5" "11" "E2002"
Assert-SameTree (Join-Path $Serial "frames") (Join-Path $Corrupt "frames") "corrupt frames"
if ((Get-FileHash (Join-Path $Serial "manifest.txt")).Hash -ne
    (Get-FileHash (Join-Path $Corrupt "manifest.txt")).Hash) {
    Fail "corrupt manifest"
}
Pass "corrupt frame and subsequent frames recomputed"

$BadVersion = Join-Path $Work "badver"
Copy-Item -LiteralPath $Serial -Destination $BadVersion -Recurse
$Manifest = Join-Path $BadVersion "manifest.txt"
$Text = Get-Content -LiteralPath $Manifest -Raw
$Text = $Text.Replace("FMSNAP_MANIFEST 2", "FMSNAP_MANIFEST 999")
$Text = $Text.Replace("manifest_version 2", "manifest_version 999")
Set-Content -LiteralPath $Manifest -Value $Text -NoNewline
Invoke-Rigskin $BadVersion @("--threads", "5", "--frames", "16", "--width", "640", "--height", "480")
Assert-Summary $BadVersion "complete" "0" "16" "E1003"
Assert-SameTree (Join-Path $Serial "frames") (Join-Path $BadVersion "frames") "bad-version frames"
if (-not (Test-Path (Join-Path $BadVersion "quarantine\invalid_manifest_version\frames"))) {
    Fail "bad version payload was not quarantined"
}
Pass "unsupported version quarantined and rebuilt"

$HashChange = Join-Path $Work "hashchange"
Copy-Item -LiteralPath $Serial -Destination $HashChange -Recurse
Invoke-Rigskin $HashChange @("--threads", "4", "--frames", "16", "--width", "641", "--height", "480")
Assert-Summary $HashChange "complete" "0" "16" "E1004"
if (-not (Test-Path (Join-Path $HashChange "quarantine\input_hash_changed\frames"))) {
    Fail "input hash change payload was not quarantined"
}
Pass "input hash change rebuilds the snapshot"

$KillReference = Join-Path $Work "kill-ref"
Invoke-Rigskin $KillReference @("--threads", "1", "--frames", "24", "--width", "640", "--height", "480")

$Kill = Join-Path $Work "kill"
$KillArguments = "--snapshot-dir `"$Kill`" --threads 4 --frames 24 " +
    "--width 640 --height 480 --commit-delay-ms 200"
$KillProcess = Start-Process -FilePath $Rigskin -PassThru -NoNewWindow -ArgumentList @(
    $KillArguments
)
for ($Attempt = 0; $Attempt -lt 300; $Attempt++) {
    $ManifestPath = Join-Path $Kill "manifest.txt"
    $Count = 0
    if (Test-Path $ManifestPath) {
        $Count = @(Select-String -LiteralPath $ManifestPath -Pattern "^entry ").Count
    }
    if ($Count -ge 3) { break }
    Start-Sleep -Milliseconds 20
}
Stop-Process -Id $KillProcess.Id -Force -ErrorAction SilentlyContinue
Wait-Process -Id $KillProcess.Id -ErrorAction SilentlyContinue
if (Test-Path (Join-Path $Kill "summary.txt")) {
    Fail "forced-kill process unexpectedly wrote a summary"
}
Invoke-Rigskin $Kill @("--threads", "5", "--frames", "24", "--width", "640", "--height", "480")
Assert-Summary $Kill "complete" "4" "20" "E2101"
Assert-SameTree (Join-Path $KillReference "frames") (Join-Path $Kill "frames") "kill-resume frames"
if ((Get-FileHash (Join-Path $KillReference "manifest.txt")).Hash -ne
    (Get-FileHash (Join-Path $Kill "manifest.txt")).Hash) {
    Fail "kill-resume manifest"
}
Pass "SIGKILL recovery matches clean execution"

$Sigint = Join-Path $Work "sigint"
$SigintArguments = "--snapshot-dir `"$Sigint`" --threads 4 --frames 24 " +
    "--width 640 --height 480 --commit-delay-ms 200"
$SigintProcess = Start-Process -FilePath $Rigskin -PassThru -NoNewWindow -ArgumentList @(
    $SigintArguments
)
for ($Attempt = 0; $Attempt -lt 300; $Attempt++) {
    $ManifestPath = Join-Path $Sigint "manifest.txt"
    $Count = 0
    if (Test-Path $ManifestPath) {
        $Count = @(Select-String -LiteralPath $ManifestPath -Pattern "^entry ").Count
    }
    if ($Count -ge 2) { break }
    Start-Sleep -Milliseconds 20
}
$PreviousTreatControlCAsInput = [Console]::TreatControlCAsInput
[Console]::TreatControlCAsInput = $true
try {
    [ConsoleControl]::GenerateConsoleCtrlEvent(0, 0) | Out-Null
} finally {
    [Console]::TreatControlCAsInput = $PreviousTreatControlCAsInput
}
$SigintProcess.WaitForExit()
$Summary = Get-Content -LiteralPath (Join-Path $Sigint "summary.txt") -Raw
if ($Summary -notmatch "status=interrupted") { Fail "SIGINT summary" }
if ($Summary -notmatch "error_codes=E9001") { Fail "SIGINT error code" }
Invoke-Rigskin $Sigint @("--threads", "3", "--frames", "24", "--width", "640", "--height", "480")
Assert-SameTree (Join-Path $KillReference "frames") (Join-Path $Sigint "frames") "SIGINT-resume frames"
Pass "SIGINT summary and resume"

Write-Host "ALL INTEGRATION TESTS PASSED"

