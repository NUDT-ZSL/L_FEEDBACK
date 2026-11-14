#!/usr/bin/env python3
"""Generate a deterministic offline sample dataset for the lipsync pipeline.

Creates 6 voice-over files covering English, Japanese and Spanish:
  - 4 well-formed files (WAV and OGG inputs, including an unknown phoneme)
  - 1 file whose phoneme intervals overflow the audio duration (>100ms)
  - 1 file with a corrupted sidecar JSON

All samples are synthesized locally from fixed seeds. No network access and no
external datasets are required. OGG samples are produced with the local
FFmpeg CLI when available; otherwise only WAV inputs are generated.
"""

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import wave


SAMPLE_RATE = 24000
DURATION_SECONDS = 2.0


def synth_pcm16(duration_seconds: float, frequency: float) -> bytes:
    """Deterministic sine tone with a fixed amplitude envelope."""
    frame_count = int(round(duration_seconds * SAMPLE_RATE))
    frames = bytearray()
    for index in range(frame_count):
        envelope = 0.35 * (1.0 - index / max(1, frame_count)) + 0.15
        sample = int(32767 * envelope *
                     math.sin(2.0 * math.pi * frequency * index / SAMPLE_RATE))
        frames += struct.pack("<h", sample)
    return bytes(frames)


def write_wav(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)


def write_ogg_via_ffmpeg(wav_path: str, ogg_path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "wav", "-i", wav_path,
            "-c:a", "libvorbis", "-q:a", "4",
            ogg_path,
        ],
        check=True,
    )
    return True


def write_phoneme_json(path: str, language: str, phonemes) -> None:
    payload = {
        "language": language,
        "phonemes": [
            {"phoneme": phoneme, "start": start, "end": end}
            for phoneme, start, end in phonemes
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic samples")
    parser.add_argument("--output", required=True, help="sample output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    base_pcm = synth_pcm16(DURATION_SECONDS, 220.0)
    base_wav = os.path.join(args.output, "_base_24k.wav")
    write_wav(base_wav, base_pcm)

    # 1) English WAV: normal phoneme timeline fully inside [0, 2.0s].
    write_wav(os.path.join(args.output, "voice_en_line01.wav"), base_pcm)
    write_phoneme_json(
        os.path.join(args.output, "voice_en_line01.json"),
        "en",
        [
            ("HH", 0.05, 0.18),
            ("AH", 0.18, 0.40),
            ("L", 0.40, 0.55),
            ("OW", 0.55, 0.85),
            ("SIL", 0.85, 1.00),
            ("W", 1.00, 1.20),
            ("ER", 1.20, 1.45),
            ("L", 1.45, 1.62),
            ("D", 1.62, 1.85),
        ],
    )

    # 2) Japanese OGG: includes an intentionally unknown phoneme "XX"
    #    (mapper must warn and fall back to silence without interrupting).
    ogg_created = write_ogg_via_ffmpeg(
        base_wav, os.path.join(args.output, "voice_ja_line02.ogg")
    )
    ja_audio = "voice_ja_line02.ogg" if ogg_created else "voice_ja_line02.wav"
    if not ogg_created:
        write_wav(os.path.join(args.output, ja_audio), base_pcm)
    write_phoneme_json(
        os.path.join(args.output, "voice_ja_line02.json"),
        "ja",
        [
            ("K", 0.10, 0.25),
            ("O", 0.25, 0.50),
            ("N", 0.50, 0.70),
            ("N", 0.70, 0.90),
            ("I", 0.90, 1.10),
            ("CH", 1.10, 1.35),
            ("W", 1.35, 1.55),
            ("A", 1.55, 1.80),
            ("XX", 1.80, 1.95),
        ],
    )

    # 3) Spanish WAV: normal timeline.
    write_wav(os.path.join(args.output, "voice_es_line03.wav"), base_pcm)
    write_phoneme_json(
        os.path.join(args.output, "voice_es_line03.json"),
        "es",
        [
            ("B", 0.08, 0.22),
            ("U", 0.22, 0.42),
            ("E", 0.42, 0.62),
            ("N", 0.62, 0.80),
            ("O", 0.80, 1.05),
            ("S", 1.05, 1.30),
            ("D", 1.30, 1.48),
            ("I", 1.48, 1.68),
            ("A", 1.68, 1.90),
        ],
    )

    # 4) English WAV: another well-formed file to reach >=5 good/bad entries.
    write_wav(os.path.join(args.output, "voice_en_line04.wav"), base_pcm)
    write_phoneme_json(
        os.path.join(args.output, "voice_en_line04.json"),
        "en",
        [
            ("S", 0.05, 0.20),
            ("P", 0.20, 0.34),
            ("IY", 0.34, 0.58),
            ("K", 0.58, 0.76),
            ("SIL", 0.76, 0.95),
            ("AE", 0.95, 1.20),
            ("N", 1.20, 1.40),
            ("D", 1.40, 1.62),
        ],
    )

    # 5) Overflow file: last phoneme ends at 2.6s -> exceeds 2.0s audio by
    #     600ms (>100ms tolerance) => E_PHONEME_OVERFLOW, lip sync skipped.
    write_wav(os.path.join(args.output, "voice_es_overflow.wav"), base_pcm)
    write_phoneme_json(
        os.path.join(args.output, "voice_es_overflow.json"),
        "es",
        [
            ("K", 0.10, 0.30),
            ("A", 0.30, 0.55),
            ("S", 0.55, 0.80),
            ("A", 0.80, 1.10),
            ("R", 1.10, 1.40),
            ("O", 1.40, 1.75),
            ("S", 1.75, 2.60),
        ],
    )

    # 6) Corrupted JSON sidecar: malformed syntax => E_PARSE_JSON.
    write_wav(os.path.join(args.output, "voice_ja_broken.wav"), base_pcm)
    with open(os.path.join(args.output, "voice_ja_broken.json"), "w",
              encoding="utf-8") as handle:
        handle.write('{\n  "language": "ja",\n  "phonemes": [\n')
        handle.write('    {"phoneme": "A", "start": 0.1, "end": 0.3,}\n')
        handle.write("  // unterminated array and trailing comma above\n")

    os.remove(base_wav)

    audio_files = [
        name for name in sorted(os.listdir(args.output))
        if name.endswith((".wav", ".ogg"))
    ]
    print(f"Generated {len(audio_files)} audio samples in {args.output}")
    for name in audio_files:
        print(" -", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
