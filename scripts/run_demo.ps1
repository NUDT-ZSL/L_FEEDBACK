[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$PythonVersionArgs = "-3.11",
    [string]$SampleDir = "demo",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$prefix = @()
if ($PythonVersionArgs) {
    $prefix = $PythonVersionArgs.Split(
        " ",
        [System.StringSplitOptions]::RemoveEmptyEntries
    )
}

$generateArgs = @("tools\generate_sample.py", "--output", $SampleDir)
if ($Force) {
    $generateArgs += "--force"
}

& $Python @prefix @generateArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$configPath = Join-Path $SampleDir "replay.json"
& $Python @prefix -m replay_mixer --config $configPath
exit $LASTEXITCODE
