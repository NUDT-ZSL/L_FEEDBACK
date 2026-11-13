[CmdletBinding()]
param(
    [string]$Python = "py",
    [string]$PythonVersionArgs = "-3.11"
)

$ErrorActionPreference = "Stop"
$prefix = @()
if ($PythonVersionArgs) {
    $prefix = $PythonVersionArgs.Split(
        " ",
        [System.StringSplitOptions]::RemoveEmptyEntries
    )
}

& $Python @prefix -m unittest discover -s tests -v
exit $LASTEXITCODE
