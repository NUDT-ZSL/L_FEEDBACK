$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

dotnet restore SpatialAudioPipeline.sln --configfile NuGet.Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

dotnet build SpatialAudioPipeline.sln -c Release --no-restore
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

dotnet tests/AudioPipeline.Tests/bin/Release/net8.0/AudioPipeline.Tests.dll
exit $LASTEXITCODE
