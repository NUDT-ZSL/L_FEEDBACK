param(
    [Parameter(Mandatory = $true)]
    [string]$Exe
)

$ErrorActionPreference = "Stop"
$Exe = (Resolve-Path $Exe).Path
$Root = Join-Path ([System.IO.Path]::GetTempPath()) ("voice-replay-acceptance-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Root | Out-Null

function Fail([string]$Message) {
    Write-Error "FAIL: $Message"
    exit 1
}

function Invoke-Tool {
    param([string[]]$Arguments, [int]$ExpectedExit = 0, [string]$ExpectedCode = "")
    $log = Join-Path $Root ("run-" + [Guid]::NewGuid().ToString("N") + ".log")
    & $Exe @Arguments *> $log
    $code = $LASTEXITCODE
    $text = Get-Content -Raw -LiteralPath $log
    if ($code -ne $ExpectedExit) {
        Write-Host $text
        Fail "exit $code, expected $ExpectedExit"
    }
    if ($ExpectedCode -ne "" -and $text -notmatch [regex]::Escape("error=$ExpectedCode")) {
        Write-Host $text
        Fail "missing $ExpectedCode"
    }
    return @{ Code = $code; Text = $text }
}

function Copy-Input([string]$Target) {
    New-Item -ItemType Directory -Force -Path $Target | Out-Null
    Copy-Item -Path (Join-Path $Root "in\*") -Destination $Target -Recurse -Force
}

try {
    $input = Join-Path $Root "in"
    New-Item -ItemType Directory -Path $input | Out-Null
    Invoke-Tool @("generate", "--output", $input) | Out-Null

    $full = Join-Path $Root "full"
    $result = Invoke-Tool @("export", "--input", $input, "--output", $full)
    if (($result.Text | Select-String "voice-replay: SHA256 " -AllMatches).Matches.Count -ne 5) {
        Fail "expected five WAV digest lines"
    }
    $master = Join-Path $full "master.wav"
    $manifest = Join-Path $full "manifest.json"
    if ((Get-Item $master).Length -ne 1555244) { Fail "wrong master WAV size" }
    $stem = Join-Path $full "tracks\track_0001_caster_alpha.wav"
    if ((Get-Item $stem).Length -ne 777644) { Fail "wrong stem WAV size" }
    $manifestText = Get-Content -Raw $manifest
    if ($manifestText -notmatch '"total_frames": 388800') { Fail "wrong manifest frames" }
    if ($manifestText -notmatch '"alignment_offset_frames": 48000') { Fail "missing alignment offset" }
    if ($manifestText -notmatch '"used_frames": 0') { Fail "missing zero-sample track marker" }
    $masterHash = (Get-FileHash $master -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($manifestText -notmatch $masterHash) { Fail "manifest master digest mismatch" }

    $resumed = Join-Path $Root "resumed"
    Invoke-Tool @("export", "--input", $input, "--output", $resumed, "--abort-after-blocks", "17") 75 | Out-Null
    if (-not (Test-Path (Join-Path $resumed ".voice-replay-state.json"))) {
        Fail "abort did not write state"
    }
    Invoke-Tool @("export", "--input", $input, "--output", $resumed, "--resume") | Out-Null
    foreach ($relative in @("master.wav", "manifest.json",
                            "tracks\track_0001_caster_alpha.wav",
                            "tracks\track_0002_caster_beta.wav",
                            "tracks\track_0003_caster_gamma.wav",
                            "tracks\track_0004_caster_silent.wav")) {
        $a = Join-Path $full $relative
        $b = Join-Path $resumed $relative
        if ((Get-FileHash $a -Algorithm SHA256).Hash -ne
            (Get-FileHash $b -Algorithm SHA256).Hash) {
            Fail "resume differs at $relative"
        }
    }

    $masterTime = (Get-Item $master).LastWriteTimeUtc
    $manifestTime = (Get-Item $manifest).LastWriteTimeUtc
    Start-Sleep -Seconds 1
    $reuse = Invoke-Tool @("export", "--input", $input, "--output", $full)
    if ($reuse.Text -notmatch "REUSE existing byte-identical outputs") {
        Fail "repeat run did not reuse outputs"
    }
    if ((Get-Item $master).LastWriteTimeUtc -ne $masterTime) { Fail "master timestamp changed" }
    if ((Get-Item $manifest).LastWriteTimeUtc -ne $manifestTime) { Fail "manifest timestamp changed" }

    Copy-Input (Join-Path $Root "negative\in")
    $negativeMeta = Join-Path $Root "negative\in\caster_alpha.json"
    (Get-Content -Raw $negativeMeta).Replace('"start_frame": 0', '"start_frame": -1') |
        Set-Content -NoNewline $negativeMeta
    Invoke-Tool @("export", "--input", (Join-Path $Root "negative\in"),
                  "--output", (Join-Path $Root "negative\out")) 4 "E_META_INVALID" | Out-Null
    if (Test-Path (Join-Path $Root "negative\out\master.wav")) {
        Fail "invalid metadata left a partial output"
    }

    Copy-Input (Join-Path $Root "nan\in")
    $nanMeta = Join-Path $Root "nan\in\caster_alpha.json"
    (Get-Content -Raw $nanMeta).Replace('"gain_db": -6.0', '"gain_db": NaN') |
        Set-Content -NoNewline $nanMeta
    Invoke-Tool @("export", "--input", (Join-Path $Root "nan\in"),
                  "--output", (Join-Path $Root "nan\out")) 4 "E_META_INVALID" | Out-Null

    Copy-Input (Join-Path $Root "rate\in")
    $rateEvents = Join-Path $Root "rate\in\events.json"
    (Get-Content -Raw $rateEvents).Replace('"sample_rate": 48000', '"sample_rate": 44100') |
        Set-Content -NoNewline $rateEvents
    Invoke-Tool @("export", "--input", (Join-Path $Root "rate\in"),
                  "--output", (Join-Path $Root "rate\out")) 5 "E_RATE_MISMATCH" | Out-Null

    Invoke-Tool @() 2 "E_USAGE" | Out-Null

    Copy-Input (Join-Path $Root "statechange\in")
    Invoke-Tool @("export", "--input", (Join-Path $Root "statechange\in"),
                  "--output", (Join-Path $Root "statechange\out"),
                  "--abort-after-blocks", "11") 75 | Out-Null
    $changedMeta = Join-Path $Root "statechange\in\caster_alpha.json"
    (Get-Content -Raw $changedMeta).Replace('"gain_db": -6.0', '"gain_db": -7.0') |
        Set-Content -NoNewline $changedMeta
    Invoke-Tool @("export", "--input", (Join-Path $Root "statechange\in"),
                  "--output", (Join-Path $Root "statechange\out"),
                  "--resume") 8 "E_STATE_CORRUPT" | Out-Null

    Write-Host "ALL ACCEPTANCE TESTS PASSED"
}
finally {
    Remove-Item -Recurse -Force -Path $Root -ErrorAction SilentlyContinue
}

