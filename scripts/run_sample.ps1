param(
    [string]$BuildDir = "build",
    [string]$RunDir = "run"
)

$ErrorActionPreference = "Stop"

function Find-Executable([string]$name) {
    $candidates = @(
        (Join-Path $BuildDir "$name.exe"),
        (Join-Path $BuildDir "RelWithDebInfo/$name.exe"),
        (Join-Path $BuildDir "Release/$name.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    throw "Cannot find $name.exe under $BuildDir"
}

$renderer = Find-Executable "volume_renderer"
$sceneGen = Find-Executable "scene_gen"
$shaderDir = (Resolve-Path (Join-Path $BuildDir "shaders")).Path
$runPath = Join-Path (Get-Location) $RunDir
$sampleDir = Join-Path $runPath "sample"
New-Item -ItemType Directory -Force -Path $runPath, $sampleDir | Out-Null

& $sceneGen $sampleDir
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $renderer `
    --scene (Join-Path $sampleDir "scene.json") `
    --out (Join-Path $runPath "render.png") `
    --stats (Join-Path $runPath "stats.json") `
    --reference-out (Join-Path $runPath "reference.png") `
    --recovery-dir (Join-Path $runPath "recovery") `
    --shader-dir $shaderDir `
    --validation on
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Get-FileHash (Join-Path $runPath "render.png") -Algorithm SHA256
Get-FileHash (Join-Path $runPath "reference.png") -Algorithm SHA256
