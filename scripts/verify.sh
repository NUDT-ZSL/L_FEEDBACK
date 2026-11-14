#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

dotnet restore SpatialAudioPipeline.sln --configfile NuGet.Config
dotnet build SpatialAudioPipeline.sln -c Release --no-restore
dotnet tests/AudioPipeline.Tests/bin/Release/net8.0/AudioPipeline.Tests.dll
