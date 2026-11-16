param(
    [Parameter(Mandatory = $true)]
    [string]$Exe,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path -LiteralPath $Exe).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path (Split-Path -Parent $Exe) "acceptance"
}
if (Test-Path -LiteralPath $OutputRoot) {
    Remove-Item -LiteralPath $OutputRoot -Recurse -Force
}
$runA = Join-Path $OutputRoot "run_a"
$runB = Join-Path $OutputRoot "run_b"
$runNoLoss = Join-Path $OutputRoot "run_no_loss"
New-Item -ItemType Directory -Force -Path $runA, $runB, $runNoLoss | Out-Null

function Invoke-Renderer {
    param([string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "renderer failed with exit code ${LASTEXITCODE}: $Arguments"
    }
}

$commonArguments = @(
    "--frames", "60",
    "--simulate-device-loss-at", "30"
)
Invoke-Renderer ($commonArguments + @("--output-dir", $runA))
Invoke-Renderer ($commonArguments + @("--output-dir", $runB))
$noLossArguments = @("--frames", "60", "--output-dir", $runNoLoss)
Invoke-Renderer -Arguments $noLossArguments

$filesA = Get-ChildItem -LiteralPath $runA -File | Sort-Object Name
$filesB = Get-ChildItem -LiteralPath $runB -File | Sort-Object Name
if ($filesA.Count -ne 121 -or $filesB.Count -ne 121) {
    throw "expected 121 output files per run, got $($filesA.Count) and $($filesB.Count)"
}
if ($filesA.Count -ne $filesB.Count) {
    throw "run file counts differ"
}

for ($index = 0; $index -lt $filesA.Count; $index++) {
    if ($filesA[$index].Name -ne $filesB[$index].Name) {
        throw "output filename mismatch: $($filesA[$index].Name)"
    }
    $hashA = (Get-FileHash -Algorithm SHA256 -LiteralPath $filesA[$index].FullName).Hash
    $hashB = (Get-FileHash -Algorithm SHA256 -LiteralPath $filesB[$index].FullName).Hash
    if ($hashA -ne $hashB) {
        throw "non-deterministic output: $($filesA[$index].Name)"
    }
}

for ($frame = 0; $frame -lt 60; $frame++) {
    $base = "frame_{0:D4}" -f $frame
    foreach ($extension in @(".png", ".json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $runA ($base + $extension)))) {
            throw "missing frame output: $base$extension"
        }
    }
}

$stats = Get-Content -Raw -LiteralPath (Join-Path $runA "stats.json") |
    ConvertFrom-Json
if ($stats.frameCount -ne 60) { throw "frameCount is not 60" }
if ($stats.deviceLossInjectionAfterFrame -ne 30) {
    throw "device loss injection frame is not 30"
}
if ($stats.deviceLossRecoveries -ne 1) {
    throw "expected exactly one device loss recovery"
}
if ($stats.drawCallsPerFrame -ne 4) { throw "draw call count is not 4" }
if ($stats.shadowMapSize -ne 1024) { throw "shadow map size is not 1024" }
if (($stats.frames | Measure-Object).Count -ne 60) {
    throw "aggregate frame array length is not 60"
}
for ($frame = 0; $frame -lt 60; $frame++) {
    $entry = $stats.frames[$frame]
    if ($entry.frame -ne $frame) { throw "frame sequence is not contiguous" }
    $expectedTime = $frame / 60.0
    if ([math]::Abs($entry.animationTimeSeconds - $expectedTime) -gt 0.000001) {
        throw "animation time mismatch at frame $frame"
    }
    if (($entry.shadowMapSizes -join ",") -ne "1024,1024,1024") {
        throw "shadow map sizes mismatch at frame $frame"
    }
    $expectedRecoveries = if ($frame -gt 30) { 1 } else { 0 }
    if ($entry.deviceLossRecoveries -ne $expectedRecoveries) {
        throw "recovery counter mismatch at frame $frame"
    }
}

$uniquePngs = @(
    Get-ChildItem -LiteralPath $runA -Filter "frame_*.png" |
        ForEach-Object { (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash } |
        Sort-Object -Unique
)
if ($uniquePngs.Count -ne 60) {
    throw "expected 60 unique PNG frames, got $($uniquePngs.Count)"
}

for ($frame = 0; $frame -lt 60; $frame++) {
    $name = "frame_{0:D4}.png" -f $frame
    $lossHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $runA $name)).Hash
    $noLossHash = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $runNoLoss $name)).Hash
    if ($lossHash -ne $noLossHash) {
        throw "device recovery changed rendered pixels at $name"
    }
}
$noLossStats = Get-Content -Raw -LiteralPath (Join-Path $runNoLoss "stats.json") |
    ConvertFrom-Json
if ($null -ne $noLossStats.deviceLossInjectionAfterFrame) {
    throw "no-loss run unexpectedly reports an injection frame"
}
if ($noLossStats.deviceLossRecoveries -ne 0) {
    throw "no-loss run unexpectedly reports a recovery"
}

$invalidCases = @(
    @{ Name = "negative frames"; Arguments = @("--frames", "-1") },
    @{ Name = "split count"; Arguments = @("--frames", "1", "--cascade-splits", "0.1,0.3") },
    @{ Name = "non-increasing splits"; Arguments = @("--frames", "1", "--cascade-splits", "0.4,0.3,0.8") },
    @{ Name = "split range"; Arguments = @("--frames", "1", "--cascade-splits", "0,0.3,0.8") },
    @{ Name = "loss frame range"; Arguments = @("--frames", "2", "--simulate-device-loss-at", "2") },
    @{ Name = "negative bias"; Arguments = @("--frames", "1", "--const-offset", "-0.1") },
    @{ Name = "unknown option"; Arguments = @("--frames", "1", "--not-an-option", "1") }
)
$invalidIndex = 0
foreach ($case in $invalidCases) {
    $arguments = [string[]]$case.Arguments
    $standardOutput = Join-Path $OutputRoot ("invalid_{0}.stdout.txt" -f $invalidIndex)
    $standardError = Join-Path $OutputRoot ("invalid_{0}.stderr.txt" -f $invalidIndex)
    $process = Start-Process -FilePath $Exe -ArgumentList $arguments -Wait -PassThru `
        -NoNewWindow -RedirectStandardOutput $standardOutput `
        -RedirectStandardError $standardError
    $invalidIndex++
    if ($process.ExitCode -eq 0) {
        throw "invalid input was accepted: $($case.Name)"
    }
    if ((Get-Item -LiteralPath $standardError).Length -eq 0) {
        throw "invalid input did not write to stderr: $($case.Name)"
    }
    Write-Host ("OK invalid input rejected: {0} (exit {1})" -f $case.Name, $process.ExitCode)
}

Write-Host "OK: 60 frames x 2 runs are byte-identical"
Write-Host "OK: all PNGs match the no-loss control run"
Write-Host "OK: frame sequence, timestamps, CSM metadata, and recovery count"
Write-Host "OK: all invalid parameter cases returned non-zero"

