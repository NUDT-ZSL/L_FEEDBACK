param(
    [string]$Directory = "",
    [int]$Duration = 20,
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Directory)) {
    $Directory = Join-Path $Root ("artifacts\e2e-" + [DateTime]::UtcNow.ToString("yyyyMMddHHmmss"))
}
New-Item -ItemType Directory -Force -Path $Directory | Out-Null
$Project = Join-Path $Root "src\ReplayForge\ReplayForge.csproj"
$Dll = Join-Path $Root "src\ReplayForge\bin\$Configuration\net8.0\win-x64\replayforge.dll"

Write-Host "workspace: $Directory"
& dotnet build $Project -c $Configuration -r win-x64 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "build failed" }

& dotnet $Dll generate --dir $Directory --duration $Duration --seed 1337 --overwrite |
    Set-Content (Join-Path $Directory "generate.json") -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "generation failed" }

$OutFile = Join-Path $Directory "crashed-run.json"
$ErrFile = Join-Path $Directory "crashed-run.err"
$Process = Start-Process dotnet -ArgumentList @(
    $Dll, "run", "--manifest", (Join-Path $Directory "manifest.json")
) -PassThru -NoNewWindow -RedirectStandardOutput $OutFile -RedirectStandardError $ErrFile

$SnapshotPath = Join-Path $Directory "work\resume.json"
$Completed = 0
$Deadline = [DateTime]::UtcNow.AddSeconds(120)
while ([DateTime]::UtcNow -lt $Deadline) {
    if (Test-Path $SnapshotPath) {
        try {
            $Snapshot = Get-Content $SnapshotPath -Raw | ConvertFrom-Json
            $Completed = [int]$Snapshot.completedSegments
            if ($Completed -ge 3) { break }
        } catch { }
    }
    if ($Process.HasExited) { break }
    Start-Sleep -Milliseconds 10
}

if ($Completed -lt 3) {
    throw "render finished before an interruption could be injected"
}

& taskkill.exe /PID $Process.Id /T /F | Out-Null
$Process.WaitForExit()

$OutputPath = Join-Path $Directory "output\final.mp4"
$Stream = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
try {
    $Partial = [System.Text.Encoding]::ASCII.GetBytes("SIMULATED-PARTIAL-GOP" * 256)
    $Stream.Write($Partial, 0, $Partial.Length)
} finally {
    $Stream.Dispose()
}

& dotnet $Dll run --manifest (Join-Path $Directory "manifest.json") |
    Set-Content (Join-Path $Directory "resume.json") -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "resume failed" }

$Report = Get-Content (Join-Path $Directory "resume.json") -Raw | ConvertFrom-Json
if (-not $Report.ok) { throw "resume report is not ok" }
if ($Report.recovery.mode -ne "forced-termination-recovery") { throw "wrong recovery mode" }
if ($Report.recovery.previousCleanShutdown -ne $false) { throw "unclean shutdown was not detected" }
if ($Report.recovery.rolledBackBytes -le 0) { throw "partial bytes were not rolled back" }
if (-not $Report.validation.ok) { throw "output validation failed" }
if ($Report.totalFrames -ne ($Duration * 30)) { throw "wrong frame count" }

& dotnet $Dll run --manifest (Join-Path $Directory "manifest.json") |
    Set-Content (Join-Path $Directory "normal-complete.json") -Encoding utf8
$Normal = Get-Content (Join-Path $Directory "normal-complete.json") -Raw | ConvertFrom-Json
if ($Normal.recovery.mode -ne "normal-already-complete") { throw "normal completion path changed" }

$Manifest = Get-Content (Join-Path $Directory "manifest.json") -Raw | ConvertFrom-Json
$Manifest.timeline.masterStartPtsSeconds = 10.1
$Manifest.summaryPath = "sync-error-summary.json"
$Manifest | ConvertTo-Json -Depth 100 |
    Set-Content (Join-Path $Directory "bad-sync.json") -Encoding utf8
& dotnet $Dll run --manifest (Join-Path $Directory "bad-sync.json") |
    Set-Content (Join-Path $Directory "bad-sync-result.json") -Encoding utf8
$BadSync = Get-Content (Join-Path $Directory "bad-sync-result.json") -Raw | ConvertFrom-Json
if ($BadSync.error.code -ne "E_SYNC_FAILED") { throw "sync failure code is wrong" }

Write-Host ("crash recovery: mode={0} frames={1} duration={2} keyframes={3} rolledBack={4}" -f `
    $Report.recovery.mode, $Report.totalFrames, $Report.totalDurationSeconds,
    $Report.validation.keyframes, $Report.recovery.rolledBackBytes)
Write-Host "normal exit: $($Normal.recovery.mode)"
Write-Host "business error: $($BadSync.error.code)"
Write-Host "PASS: $Directory"
