[CmdletBinding()]
param(
    [string]$TestRoot = (Join-Path $env:TEMP "cfaudit_uid10_accept")
)

$ErrorActionPreference = "Stop"
$SourceRepo = $PSScriptRoot | Split-Path -Parent
$script:BaseProcessPath = $env:PATH

function Check([bool]$Condition, [string]$Message) {
    if (-not $Condition) {
        throw "CHECK FAILED: $Message"
    }
    Write-Host "PASS: $Message"
}

function Read-JsonFile([string]$Path) {
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Import-VsEnvironment([string]$Arch) {
    $env:PATH = $script:BaseProcessPath
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsPath) {
        throw "Visual Studio C++ build tools were not found"
    }
    $batchName = if ($Arch -eq "x86") { "vcvars32.bat" } else { "vcvars64.bat" }
    $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\$batchName"
    $batch = Join-Path $TestRoot "_vcvars.cmd"
    Set-Content -LiteralPath $batch -Encoding ASCII -Value @(
        "call `"$vcvars`" >nul",
        "set"
    )
    $environment = & cmd.exe /d /c $batch
    foreach ($line in $environment) {
        $index = $line.IndexOf("=")
        if ($index -lt 1) {
            continue
        }
        $name = $line.Substring(0, $index)
        $value = $line.Substring($index + 1)
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Invoke-Compiler([string]$FileName, [string[]]$Arguments) {
    & $FileName @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FileName failed with exit code $LASTEXITCODE"
    }
}

function Build-Samples([string]$Arch, [string]$Suffix) {
    Import-VsEnvironment $Arch
    Invoke-Compiler "cl.exe" @(
        "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
        "sample.c", "/Fo:sample$Suffix.obj"
    )
    Invoke-Compiler "cl.exe" @(
        "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
        "sample_safe.c", "/Fo:sample_safe$Suffix.obj"
    )
    Invoke-Compiler "link.exe" @(
        "/nologo", "/debug", "/pdb:old$Suffix.pdb", "/out:old$Suffix.exe",
        "sample$Suffix.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
    )
    Invoke-Compiler "link.exe" @(
        "/nologo", "/debug", "/pdb:new$Suffix.pdb", "/out:new$Suffix.exe",
        "sample_safe$Suffix.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
    )
}

function Invoke-CFA([string[]]$Arguments, [string]$StderrFile) {
    $null = & $CFAudit @Arguments 2> $StderrFile
    return $LASTEXITCODE
}

if (Test-Path -LiteralPath $TestRoot) {
    Remove-Item -LiteralPath $TestRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $TestRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $SourceRepo "cfaudit") -Destination $TestRoot -Recurse
Copy-Item -LiteralPath (Join-Path $SourceRepo "samples") -Destination $TestRoot -Recurse
Copy-Item -LiteralPath (Join-Path $SourceRepo "tests") -Destination $TestRoot -Recurse
Copy-Item -LiteralPath (Join-Path $SourceRepo "cfaudit.cmd") -Destination $TestRoot
$SampleDir = Join-Path $TestRoot "samples"
$Tests = Join-Path $TestRoot "tests"
Set-Location -LiteralPath $SampleDir

$CFAudit = Join-Path $TestRoot "cfaudit.cmd"
$Python = "py"
$Configs = Join-Path $TestRoot "configs"
& $Python -3 (Join-Path $SampleDir "generate_configs.py") --outdir $Configs
Check ($LASTEXITCODE -eq 0) "config generation script completed"

# Generate both PE architectures from the same C source.
Import-VsEnvironment "x64"
Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "minimal.c", "/Fo:pe64.obj"
)
Invoke-Compiler "link.exe" @(
    "/nologo", "/debug", "/pdb:pe64.pdb", "/out:pe64.exe",
    "pe64.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
)
Import-VsEnvironment "x86"
Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "minimal.c", "/Fo:pe32.obj"
)
Invoke-Compiler "link.exe" @(
    "/nologo", "/debug", "/pdb:pe32.pdb", "/out:pe32.exe",
    "pe32.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
)
& $Python -3 (Join-Path $Tests "patch_pair_sections.py") pe64.exe
Check ($LASTEXITCODE -eq 0) "x64 read-only section layout patched after cl/link"
& $Python -3 (Join-Path $Tests "patch_pair_sections.py") pe32.exe
Check ($LASTEXITCODE -eq 0) "x86 read-only section layout patched after cl/link"

Build-Samples "x64" "64"
Build-Samples "x86" "32"

Import-VsEnvironment "x64"
Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "summary.c", "/Fo:summary.obj"
)
Invoke-Compiler "link.exe" @(
    "/nologo", "/debug", "/pdb:summary.pdb", "/out:summary.exe",
    "summary.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
)

Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "warnings_main.c", "/Fo:warnings_main.obj"
)
Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "warnings_conflict.c", "/Fo:warnings_conflict.obj"
)
Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "warnings_conv.c", "/Fo:warnings_conv.obj"
)
Invoke-Compiler "link.exe" @(
    "/nologo", "/debug", "/force:multiple", "/pdb:warnings.pdb", "/out:warnings.exe",
    "warnings_main.obj", "warnings_conflict.obj", "warnings_conv.obj",
    "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
)

Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "regress_old.c", "/Fo:regress_old.obj"
)
Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "regress_new.c", "/Fo:regress_new.obj"
)
Invoke-Compiler "link.exe" @(
    "/nologo", "/debug", "/pdb:regress_old.pdb", "/out:regress_old.exe",
    "regress_old.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
)
Invoke-Compiler "link.exe" @(
    "/nologo", "/debug", "/pdb:regress_new.pdb", "/out:regress_new.exe",
    "regress_new.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
)

$errorFile = Join-Path $TestRoot "stderr.txt"
$exit = Invoke-CFA @("parse", "--pe", "pe64.exe", "--out", "p64.json") $errorFile
Check ($exit -eq 0) "x64 parse exits 0"
$p64 = Read-JsonFile "p64.json"
$dumpText = (& dumpbin.exe /nologo /unwindinfo pe64.exe | Out-String)
if ($dumpText -notmatch "Function Table \((\d+)\)") {
    throw "dumpbin did not report a function-table count"
}
$dumpCount = [int]$Matches[1]
Check ($p64.functions.Count -eq $dumpCount) "parse function count matches dumpbin /unwindinfo ($dumpCount)"
$createFile = $p64.imports | Where-Object { $_.name -eq "CreateFileW" } | Select-Object -First 1
Check ($null -ne $createFile) "x64 imports contain CreateFileW"
$rdata = $p64.pe_sections | Where-Object { $_.name -eq ".rdata" } | Select-Object -First 1
Check ($null -ne $rdata) "x64 PE has a .rdata section"
Check ($createFile.iat_rva -ge $rdata.rva -and $createFile.iat_rva -lt ($rdata.rva + $rdata.raw_size)) "CreateFileW IAT RVA is inside .rdata"
$rvas = @($p64.pe_sections | ForEach-Object { [int]$_.rva })
Check (((($rvas | Sort-Object) -join ",") -eq ($rvas -join ","))) "x64 sections are sorted by RVA"
Check (@($p64.pe_sections | Where-Object { $_.name.Contains([char]0) }).Count -eq 0) "section names contain no NUL"

$exit = Invoke-CFA @("parse", "--pe", "pe32.exe", "--out", "p32.json") $errorFile
Check ($exit -eq 2) "x86 parse exits 2"
$p32 = Read-JsonFile "p32.json"
Check ($p32.functions.Count -eq 0) "x86 has no .pdata functions"
Check (@($p32.errors | Where-Object { $_.code -eq "MISSING_PDATA" }).Count -ge 1) "x86 reports MISSING_PDATA"
$rvas32 = @($p32.pe_sections | ForEach-Object { [int]$_.rva })
Check (((($rvas32 | Sort-Object) -join ",") -eq ($rvas32 -join ","))) "x86 sections are sorted by RVA"
Check (@($p32.pe_sections | Where-Object { $_.name.Contains([char]0) }).Count -eq 0) "x86 section names contain no NUL"

$exit = Invoke-CFA @("parse", "--pe", "pe64.exe", "--pdb", "pe64.pdb", "--out", "p64pdb.json") $errorFile
Check ($exit -eq 0) "matched PE/PDB parse exits 0"
$p64pdb = Read-JsonFile "p64pdb.json"
$badNames = @($p64pdb.functions | Where-Object {
    -not ($_.PSObject.Properties.Name -contains "rva") -or
    -not ($_.PSObject.Properties.Name -contains "size") -or
    -not ($_.PSObject.Properties.Name -contains "name")
})
Check ($badNames.Count -eq 0) "every PDB-augmented function has rva, size, and name"

$exit = Invoke-CFA @(
    "parse", "--pe", "old64.exe", "--pdb", "new64.pdb", "--out", "mismatch.json"
) $errorFile
Check ($exit -eq 2) "PE/PDB mismatch exits 2"
$mismatch = Read-JsonFile "mismatch.json"
Check (@($mismatch.errors | Where-Object { $_.code -eq "PDB_MISMATCH" }).Count -eq 1) "mismatch reports PDB_MISMATCH"
Check ($null -ne $mismatch.pe_guid -and $null -ne $mismatch.pdb_guid -and $null -ne $mismatch.pe_age -and $null -ne $mismatch.pdb_age) "mismatch reports GUID and Age fields"
Check (-not ($mismatch.PSObject.Properties.Name -contains "sinks")) "mismatch output has no sinks field"

$exit = Invoke-CFA @(
    "analyze", "--pe", "old64.exe", "--pdb", "old64.pdb",
    "--config", (Join-Path $Configs "config.sample.json"), "--out", "sample64.json"
) $errorFile
Check ($exit -eq 0) "x64 sample analysis exits 0"
$sample64 = Read-JsonFile "sample64.json"
$sinkMemcpy = $sample64.sinks | Where-Object { $_.symbol -eq "sink_memcpy" }
$sinkNetwork = $sample64.sinks | Where-Object { $_.symbol -eq "sink_network" }
$sinkWinExec = $sample64.sinks | Where-Object { $_.symbol -eq "sink_win_exec" }
$sinkCreate = $sample64.sinks | Where-Object { $_.symbol -eq "sink_create_file" }
Check ($sinkMemcpy.exploitability -eq "DIRECT") "sink_memcpy is DIRECT"
Check (($sinkMemcpy.path -join ">") -eq "source_a>sink_memcpy") "sink_memcpy path is source_a>sink_memcpy"
Check ($sinkMemcpy.tainted_params -contains 1) "sink_memcpy reports the tainted parameter"
Check ($sinkNetwork.exploitability -eq "CONDITIONAL") "sink_network is CONDITIONAL"
Check (($sinkNetwork.path -join ">") -eq "source_a>relay>sink_network") "sink_network path traverses relay"
Check ($sinkNetwork.tainted_params -contains 0) "sink_network taints parameter 0"
Check ($sinkWinExec.exploitability -eq "CONDITIONAL") "sink_win_exec is CONDITIONAL"
Check (@($sinkWinExec.path).Count -eq 4) "sink_win_exec path has four functions"
Check ($sinkCreate.exploitability -eq "UNREACHABLE") "sanitized sink_create_file is UNREACHABLE"
Check ([int]$sample64.analysis_stats.unknown_propagation -ge 0) "unknown_propagation is non-negative"

$exit = Invoke-CFA @(
    "analyze", "--pe", "old64.exe", "--pdb", "old64.pdb",
    "--config", (Join-Path $Configs "config.sample.param0.json"), "--out", "param0.json"
) $errorFile
Check ($exit -eq 0) "param_index=0 analysis exits 0"
$param0 = Read-JsonFile "param0.json"
$param0Network = $param0.sinks | Where-Object { $_.symbol -eq "sink_network" }
Check (@($param0Network.tainted_params).Count -eq 0) "param_index=0 leaves sink_network untainted"
Check ($param0Network.exploitability -eq "UNREACHABLE") "param_index=0 makes sink_network UNREACHABLE"
Check ([int]$param0.analysis_stats.unknown_propagation -ge 0) "param0 unknown_propagation is non-negative"

$exit = Invoke-CFA @(
    "analyze", "--pe", "old64.exe", "--pdb", "old64.pdb",
    "--config", (Join-Path $Configs "config.sample.json"), "--out", "deterministic2.json"
) $errorFile
$deterministic2 = Read-JsonFile "deterministic2.json"
$pathsA = $sample64.sinks | ForEach-Object { ($_.symbol + ":" + ($_.path -join ">")) }
$pathsB = $deterministic2.sinks | ForEach-Object { ($_.symbol + ":" + ($_.path -join ">")) }
Check (($pathsA -join "|") -eq ($pathsB -join "|")) "sink paths are byte-stable across repeated runs"

$exit = Invoke-CFA @(
    "analyze", "--pe", "old32.exe", "--pdb", "old32.pdb",
    "--config", (Join-Path $Configs "config.sample.json"), "--out", "sample32.json"
) $errorFile
Check ($exit -eq 0) "x86 sample analysis exits 0"
$sample32 = Read-JsonFile "sample32.json"
$sink32 = $sample32.sinks | Where-Object { $_.symbol -eq "sink_memcpy" }
Check ($sink32.exploitability -eq "DIRECT" -and ($sink32.path -join ">") -eq "source_a>sink_memcpy") "x86 cdecl propagation reaches sink_memcpy"

$exit = Invoke-CFA @(
    "diff", "--before", "old64.exe", "--after", "new64.exe",
    "--pdb", "old64.pdb", "--config", (Join-Path $Configs "config.sample.json"),
    "--out", "diff_improved.json"
) $errorFile
Check ($exit -eq 0) "patch diff exits 0"
$diffImproved = Read-JsonFile "diff_improved.json"
$improvedMemcpy = $diffImproved.improved | Where-Object { $_.symbol -eq "sink_memcpy" }
Check ($null -ne $improvedMemcpy) "diff marks sink_memcpy as improved"
Check ($improvedMemcpy.before.exploitability -eq "DIRECT" -and $improvedMemcpy.after.exploitability -eq "UNREACHABLE") "diff includes before/after exploitability evidence"
Check (@($diffImproved.regressed).Count -eq 0 -and $diffImproved.passed -eq $true) "diff has no regression and passes"

$exit = Invoke-CFA @(
    "diff", "--baseline", "sample64.json", "--after", "new64.exe",
    "--pdb", "new64.pdb", "--config", (Join-Path $Configs "config.sample.json"),
    "--out", "diff_baseline.json"
) $errorFile
Check ($exit -eq 0) "baseline diff exits 0"
$diffBaseline = Read-JsonFile "diff_baseline.json"
Check ($diffBaseline.schema_version -eq "1.0") "baseline diff keeps schema_version 1.0"
Check ([int]$diffBaseline.analysis_stats.scanned_functions -gt 0) "baseline diff has scanned functions"

$baselineBad = Read-JsonFile "sample64.json"
$baselineBad.schema_version = "0.9"
$baselineBad | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath "baseline_bad.json" -Encoding UTF8
$exit = Invoke-CFA @(
    "diff", "--baseline", "baseline_bad.json", "--after", "new64.exe",
    "--pdb", "new64.pdb", "--config", (Join-Path $Configs "config.sample.json"),
    "--out", "diff_baseline_bad.json"
) $errorFile
Check ($exit -eq 2) "incompatible baseline exits 2"
$diffBaselineBad = Read-JsonFile "diff_baseline_bad.json"
Check (@($diffBaselineBad.errors | Where-Object { $_.code -eq "BASELINE_VERSION_MISMATCH" }).Count -eq 1) "incompatible baseline reports BASELINE_VERSION_MISMATCH"
Check (-not ($diffBaselineBad.PSObject.Properties.Name -contains "improved") -and -not ($diffBaselineBad.PSObject.Properties.Name -contains "regressed")) "incompatible baseline omits improved/regressed"

$exit = Invoke-CFA @(
    "diff", "--before", "regress_old.exe", "--after", "regress_new.exe",
    "--pdb", "regress_old.pdb", "--config", (Join-Path $Configs "config.regress.json"),
    "--out", "diff_regressed.json"
) $errorFile
Check ($exit -eq 1) "regression diff exits 1"
$diffRegressed = Read-JsonFile "diff_regressed.json"
$regressedSink = $diffRegressed.regressed | Where-Object { $_.symbol -eq "sink_win_exec" }
Check ($null -ne $regressedSink) "regression diff contains sink_win_exec"
Check ($regressedSink.before.exploitability -eq "CONDITIONAL" -and $regressedSink.after.exploitability -eq "DIRECT") "regression is CONDITIONAL to DIRECT"
Check ($diffRegressed.passed -eq $false) "regression report passed is false"

$exit = Invoke-CFA @(
    "analyze", "--pe", "summary.exe", "--pdb", "summary.pdb",
    "--config", (Join-Path $Configs "config.summary.json"), "--out", "summary.json"
) $errorFile
Check ($exit -eq 0) "summary/recursion analysis exits 0"
$summary = Read-JsonFile "summary.json"
$summarySink = $summary.sinks | Where-Object { $_.symbol -eq "sink_memcpy" }
Check (($summarySink.tainted_sources -join ">") -eq "source_a>source_b") "summary records both tainted sources"
Check ($summarySink.shortest_path_source -eq "source_b") "summary chooses the shortest source"
Check (($summarySink.shortest_path -join ">") -eq "source_b>sink_memcpy") "summary shortest path is source_b>sink_memcpy"
Check (@($summary.warnings | Where-Object { $_.code -eq "RECURSION_CUT" }).Count -ge 1) "summary reports RECURSION_CUT"

$exit = Invoke-CFA @(
    "analyze", "--pe", "warnings.exe", "--pdb", "warnings.pdb",
    "--config", (Join-Path $Configs "config.warnings.json"), "--out", "warnings.json"
) $errorFile
Check ($exit -eq 0) "warning-only analysis exits 0"
$warnings = Read-JsonFile "warnings.json"
$sourceMissing = $warnings.warnings | Where-Object { $_.code -eq "SOURCE_NOT_FOUND" -and $_.function -eq "missing_func" }
$nameConflict = $warnings.warnings | Where-Object { $_.code -eq "NAME_CONFLICT" -and $_.pdb_name -and $_.import_name } | Select-Object -First 1
$convMismatch = $warnings.warnings | Where-Object { $_.code -eq "CONV_MISMATCH" -and $_.function -eq "CreateFileW" }
Check ($null -ne $sourceMissing) "warnings include SOURCE_NOT_FOUND for missing_func"
Check ($null -ne $nameConflict) "warnings include NAME_CONFLICT with both names"
Check ($null -ne $convMismatch) "warnings include CONV_MISMATCH"

# Malformed inputs and read-only behavior.
$corruptDir = Join-Path $TestRoot "corrupt"
& $Python -3 (Join-Path $Tests "make_corrupt.py") --pe pe64.exe --pdb pe64.pdb --outdir $corruptDir
Check ($LASTEXITCODE -eq 0) "corrupt-input generator completed"

$beforeFiles = @(Get-ChildItem -LiteralPath $TestRoot -Recurse -File | ForEach-Object { $_.FullName } | Sort-Object)
$exit = Invoke-CFA @("parse", "--pe", (Join-Path $corruptDir "bad_section.exe"), "--out", "bad_section.json") $errorFile
$afterFiles = @(Get-ChildItem -LiteralPath $TestRoot -Recurse -File | ForEach-Object { $_.FullName } | Sort-Object)
$allowedOut = (Resolve-Path "bad_section.json").Path
$newFiles = @($afterFiles | Where-Object { $beforeFiles -notcontains $_ -and $_ -ne $allowedOut })
Check ($newFiles.Count -eq 0) "parse writes no path other than --out"
Check ($exit -eq 2) "bad section exits 2"
$badSection = Read-JsonFile "bad_section.json"
Check (@($badSection.errors | Where-Object { $_.code -eq "INVALID_PE" }).Count -ge 1) "bad section reports INVALID_PE"
Check ((Get-Item $errorFile).Length -eq 0) "bad section stderr is empty"

$exit = Invoke-CFA @(
    "parse", "--pe", "pe64.exe", "--pdb", (Join-Path $corruptDir "bad_stream.pdb"),
    "--out", "bad_stream.json"
) $errorFile
Check ($exit -eq 2) "0xFFFFFFFF PDB stream size exits 2"
$badStream = Read-JsonFile "bad_stream.json"
Check (@($badStream.errors | Where-Object { $_.code -eq "INVALID_PDB" }).Count -ge 1) "bad stream reports INVALID_PDB"
Check ((Get-Item $errorFile).Length -eq 0) "bad stream stderr is empty"

$exit = Invoke-CFA @(
    "parse", "--pe", "pe64.exe", "--pdb", (Join-Path $corruptDir "bad_dbi.pdb"),
    "--out", "bad_dbi.json"
) $errorFile
Check ($exit -eq 2) "DBI self-reference exits 2"
$badDbi = Read-JsonFile "bad_dbi.json"
Check (@($badDbi.errors | Where-Object { $_.code -eq "INVALID_PDB" }).Count -ge 1) "DBI self-reference reports INVALID_PDB"
$stderrText = Get-Content -LiteralPath $errorFile -Raw
Check ($null -eq $stderrText -or $stderrText -notmatch "traceback") "stderr contains no Python traceback"

# Real Windows job-object limits.
$exit = Invoke-CFA @(
    "analyze", "--pe", "old64.exe", "--pdb", "old64.pdb",
    "--config", (Join-Path $Configs "config.sample.json"),
    "--mem-limit", "16777216", "--out", "mem_limit.json"
) $errorFile
Check ($exit -eq 3) "memory limit exits 3"
$memLimit = Read-JsonFile "mem_limit.json"
$memError = $memLimit.errors | Where-Object { $_.code -eq "RESOURCE_LIMIT" }
Check ($null -ne $memError -and $memError.resource -eq "memory") "memory limit reports RESOURCE_LIMIT memory"
Check ($null -eq (Get-Process cfaudit -ErrorAction SilentlyContinue)) "no cfaudit process remains after memory limit"

& $Python -3 (Join-Path $Tests "generate_large.py") large.c --count 20001
Check ($LASTEXITCODE -eq 0) "large-function source generator completed"
Invoke-Compiler "cl.exe" @(
    "/nologo", "/Zi", "/Od", "/GS-", "/Oy-", "/Oi-", "/c",
    "large.c", "/Fo:large.obj"
)
Invoke-Compiler "link.exe" @(
    "/nologo", "/debug", "/pdb:large.pdb", "/out:large.exe",
    "large.obj", "kernel32.lib", "ucrt.lib", "legacy_stdio_definitions.lib"
)
Check ((Get-Item "large.exe").Length -le 30MB) "large PE stays within 30MB"
Check ((Get-Item "large.pdb").Length -le 60MB) "large PDB stays within 60MB"
$exit = Invoke-CFA @(
    "analyze", "--pe", "large.exe", "--pdb", "large.pdb",
    "--config", (Join-Path $Configs "config.large.json"),
    "--cpu-limit-ms", "100", "--out", "time_limit.json"
) $errorFile
Check ($exit -eq 3) "CPU limit on >20000 functions exits 3"
$timeLimit = Read-JsonFile "time_limit.json"
$timeError = $timeLimit.errors | Where-Object { $_.code -eq "RESOURCE_LIMIT" }
Check ($null -ne $timeError -and $timeError.resource -eq "time") "CPU limit reports RESOURCE_LIMIT time"
Check ($null -eq (Get-Process cfaudit -ErrorAction SilentlyContinue)) "no cfaudit process remains after CPU limit"
$leftovers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "cfaudit" -and $_.CommandLine -match "_worker"
})
Check ($leftovers.Count -eq 0) "no isolated cfaudit worker process remains"

Write-Host ""
Write-Host "ALL ACCEPTANCE CHECKS PASSED"
Write-Host "Test artifacts: $TestRoot"
