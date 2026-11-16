$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    py -3.11 -m venv .venv
}

& $Python -m pip install -r requirements.txt
if (-not (Test-Path "scene.json")) {
    & $Python -m proc_gi.generate --output scene.json --summary artifacts/scene_summary.json
}
& $Python -m unittest discover -s tests -v
& $Python -m proc_gi.validate

