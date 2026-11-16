param(
    [switch]$Headless,
    [int]$Frames = 2,
    [int]$Width = 960,
    [int]$Height = 540
)

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -3.11 -m venv .venv
}

& $Python -m pip install -r requirements.txt

if (-not (Test-Path "scene.json")) {
    & $Python -m proc_gi.generate --output scene.json
}

$Arguments = @("-m", "proc_gi.app", "--width", "$Width", "--height", "$Height")
if ($Headless) {
    $Arguments += @("--headless", "--frames", "$Frames", "--exercise-update")
}
& $Python @Arguments

