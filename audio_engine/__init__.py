"""Offline rhythm-game audio engine.

Modules:
    errors    - business error codes
    loader    - WAV loading + band-limited resampling to 48 kHz
    quantize  - BPM grid quantization of skill trigger times
    ducking   - BGM ducking envelope
    looping   - whole-bar loop extension with equal-length crossfade
    layers    - combo-state driven layer mixing
    sfx       - skill SFX scheduling, panning, soft limiting
    engine    - end-to-end pipeline orchestration
"""

ENGINE_SAMPLE_RATE = 48000
