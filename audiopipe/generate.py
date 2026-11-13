"""Deterministic PCM16 WAV + event-CSV material generator.

Everything derives from ``random.Random(seed)`` (the stable Mersenne Twister
that is part of the language spec) and integer math, so the same seed and
parameters always reproduce byte-identical files. No network, no third party.
"""

import csv
import hashlib
import math
import os
import random

from . import wavio

DEFAULT_LANGUAGES = ("en", "ja", "fr", "de")
DEFAULT_CLIP_COUNT = 4

CLIP_DURATIONS_MS = (1500, 2000, 3000)
EVENT_DURATIONS_MS = (120, 240, 360, 480, 640)
GAIN_DB_CHOICES = (-6.0, -3.0, 0.0, 2.0, 4.0)
SAMPLE_RATES = (44100, 48000)


def _seed_rng(seed, *parts):
    rng = random.Random()
    # Derive a stable integer seed from deterministic, string-joinable parts.
    key = "%s|%s" % (seed, "|".join(str(p) for p in parts))
    derived = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng.seed(derived, version=2)
    return rng


def _tone_samples(rng, n, rate, freq, amp=12000):
    """Deterministic low-frequency tone plus a small deterministic jitter."""
    out = []
    phase0 = rng.uniform(0.0, 2.0 * math.pi)
    jitter_seed = rng.randrange(1 << 30)
    jrng = random.Random(jitter_seed)
    for i in range(n):
        tone = amp * math.sin(2.0 * math.pi * freq * (i / rate) + phase0)
        jitter = jrng.uniform(-256.0, 256.0)
        v = int(round(tone + jitter))
        if v > 32767:
            v = 32767
        elif v < -32768:
            v = -32768
        out.append(v)
    return out


def _wav_name(clip_id, language, track, event_idx):
    return "%s_%s_t%02d_e%02d.wav" % (clip_id, language, track, event_idx)


def generate(root, seed=20261114, languages=DEFAULT_LANGUAGES, clip_count=DEFAULT_CLIP_COUNT):
    """Generate all material under ``root``; return summary dict."""
    wav_dir = os.path.join(root, "wav")
    csv_dir = os.path.join(root, "events")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    written_wav = 0
    written_csv = 0
    for language in languages:
        for clip_index in range(clip_count):
            clip_id = "clip%03d" % clip_index
            rng = _seed_rng(seed, language, clip_index)
            clip_ms = CLIP_DURATIONS_MS[rng.randrange(len(CLIP_DURATIONS_MS))]
            rate = SAMPLE_RATES[rng.randrange(len(SAMPLE_RATES))]
            track_count = 2 + rng.randrange(3)  # 2..4 tracks

            rows = []
            for track in range(1, track_count + 1):
                event_count = 1 + rng.randrange(3)  # 1..3 events
                cursor = rng.randrange(-80, 60)  # allow negative start offsets
                for event_idx in range(1, event_count + 1):
                    dur_ms = EVENT_DURATIONS_MS[rng.randrange(len(EVENT_DURATIONS_MS))]
                    start_ms = cursor
                    gain_db = GAIN_DB_CHOICES[rng.randrange(len(GAIN_DB_CHOICES))]
                    missing = 1 if rng.random() < 0.12 else 0
                    force_mute = 1 if (not missing and rng.random() < 0.10) else 0

                    # ~15% of present material deliberately mismatches duration.
                    mismatch_pct = 0
                    if not missing and rng.random() < 0.15:
                        mismatch_pct = rng.choice((-25, 40, 70))

                    fname = ""
                    actual_samples = 0
                    if not missing:
                        fname = _wav_name(clip_id, language, track, event_idx)
                        wanted_ms = max(1, dur_ms + mismatch_pct)
                        n = int(round(wanted_ms * rate / 1000.0))
                        if mismatch_pct < -99:
                            n = 0  # exercise zero-length material branch
                        freq = 180.0 + 60.0 * track + 25.0 * event_idx
                        samples = _tone_samples(rng, n, rate, freq) if n > 0 else []
                        wavio.write_wav(os.path.join(wav_dir, fname), rate, samples)
                        actual_samples = n
                        written_wav += 1

                    rows.append(
                        {
                            "clip_id": clip_id,
                            "language": language,
                            "track": track,
                            "event_index": event_idx,
                            "start_ms": start_ms,
                            "duration_ms": dur_ms,
                            "gain_db": gain_db,
                            "missing": missing,
                            "force_mute": force_mute,
                            "audio_file": fname,
                            "declared_rate": 0 if missing else rate,
                            "actual_samples": actual_samples,
                            "duration_mismatch_pct": mismatch_pct,
                        }
                    )
                    # Advance cursor; small gap or deliberate overlap.
                    step = dur_ms + rng.choice((-120, -60, 0, 80, 160))
                    cursor = start_ms + max(40, step)

            csv_path = os.path.join(csv_dir, "%s_%s_events.csv" % (clip_id, language))
            fieldnames = [
                "clip_id", "language", "track", "event_index", "start_ms",
                "duration_ms", "gain_db", "missing", "force_mute",
                "audio_file", "declared_rate", "actual_samples",
                "duration_mismatch_pct",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            written_csv += 1

    return {
        "seed": seed,
        "languages": list(languages),
        "clip_count": clip_count,
        "wav_files": written_wav,
        "csv_files": written_csv,
        "wav_dir": wav_dir,
        "events_dir": csv_dir,
    }
