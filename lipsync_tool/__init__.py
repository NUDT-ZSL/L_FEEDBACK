"""Game localization lip-sync alignment toolkit.

Modules:
    config   - configuration loading with validation / default fallback
    wavio    - deterministic PCM WAV IO, resampling and mono downmix
    wsola    - WSOLA based time-scale modification
    align    - energy matching + dynamic-programming event alignment
    mux      - FFmpeg MP4 remuxing with failure recovery
    samples  - deterministic local sample construction
    pipeline - per-language orchestration and report generation
    cli      - command line interface
"""

__version__ = "1.0.0"
