# Third-party notices

ReplayForge uses FFmpeg as its codec and synthetic-media engine. The Windows x64
and Linux x64 builds restore platform FFmpeg executables and shared libraries
from the `TqkLibrary.FFmpeg.Gpl2.*` NuGet packages. Those binaries are distributed
under the terms supplied in the corresponding NuGet package (`docs/LICENSE.txt`).

No network access is required at runtime. All validation media is generated
locally by the built-in deterministic generator.
