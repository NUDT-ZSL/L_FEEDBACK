"""Deterministic local sample generation.

No external audio assets, network access, or unseeded randomness are used.
The same seed and language list produce byte-identical sample files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .errors import LipAlignError
from .timeline import round_half_up
from .wavio import sha256_file, write_pcm16_wav


PathLike = Union[str, os.PathLike]

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_LANGUAGES = ("en", "zh", "es")

LANGUAGE_SPECS: Dict[str, Dict[str, Any]] = {
    "en": {"speed": 1.00, "pitch": 145.0, "formant_scale": 1.00},
    "zh": {"speed": 0.87, "pitch": 178.0, "formant_scale": 1.05},
    "es": {"speed": 1.16, "pitch": 158.0, "formant_scale": 0.98},
}

MOUTH_PHONEMES: Dict[str, Tuple[str, ...]] = {
    "closed": ("M", "B", "P"),
    "open": ("AA", "AO", "AH"),
    "wide": ("IY", "EH", "EY"),
    "round": ("OW", "UW", "W"),
    "neutral": ("ER", "AX", "UH"),
    "labiodental": ("F", "V"),
    "fricative": ("S", "SH", "Z"),
}

MOUTH_FORMANTS: Dict[str, Tuple[float, float]] = {
    "closed": (300.0, 920.0),
    "open": (760.0, 1240.0),
    "wide": (420.0, 2160.0),
    "round": (355.0, 720.0),
    "neutral": (520.0, 1480.0),
    "labiodental": (310.0, 1880.0),
    "fricative": (410.0, 2460.0),
}

SENTENCE_TEXTS = (
    "The quick brown fox",
    "A small blue lantern",
    "Voices across the room",
    "We meet before sunrise",
    "Seven silver bells",
    "The river turns north",
    "Open the bright gate",
    "A quiet afternoon",
)


def _seed_sequence(seed: int, stream: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(stream)])


def _raised_cosine(length: int) -> np.ndarray:
    if length <= 1:
        return np.ones(max(length, 0), dtype=np.float64)
    phase = np.linspace(0.0, np.pi, length, endpoint=True)
    return 0.5 - 0.5 * np.cos(phase)


def _categorize_phonemes(rng: np.random.Generator, count: int) -> List[Tuple[str, str]]:
    categories = list(MOUTH_PHONEMES.keys())
    result: List[Tuple[str, str]] = []
    previous = ""
    for _ in range(count):
        candidates = [category for category in categories if category != previous]
        category = candidates[int(rng.integers(0, len(candidates)))]
        names = MOUTH_PHONEMES[category]
        name = names[int(rng.integers(0, len(names)))]
        result.append((name, category))
        previous = category
    return result


def build_timeline_payload(
    seed: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    sentence_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a deterministic timeline dictionary for the synthetic corpus."""

    rng = np.random.default_rng(_seed_sequence(seed, 101))
    count = sentence_count if sentence_count is not None else int(rng.integers(4, 7))
    if count < 1:
        raise LipAlignError("E_SYNTH", "sentence_count must be positive", exit_code=2)
    cursor = round_half_up(0.30 * sample_rate)
    sentences: List[Dict[str, Any]] = []
    for index in range(count):
        duration_sec = 1.30 + float(rng.random()) * 0.82
        duration_samples = max(round_half_up(duration_sec * sample_rate), 1000)
        start = cursor
        end = start + duration_samples

        phone_count = int(rng.integers(8, 15))
        categories = _categorize_phonemes(rng, phone_count)
        weights = 0.45 + rng.random(phone_count)
        weights /= float(np.sum(weights))
        offsets = np.concatenate(([0], np.cumsum(weights)))
        boundaries = [
            start + int(round_half_up(float(offsets[i]) * duration_samples))
            for i in range(phone_count + 1)
        ]
        boundaries[0] = start
        boundaries[-1] = end
        for i in range(1, len(boundaries)):
            if boundaries[i] <= boundaries[i - 1]:
                boundaries[i] = boundaries[i - 1] + 1
        boundaries[-1] = end

        phonemes: List[Dict[str, Any]] = []
        for phone_index, (name, mouth) in enumerate(categories):
            phone_start = boundaries[phone_index]
            phone_end = boundaries[phone_index + 1]
            phonemes.append(
                {
                    "phoneme": name,
                    "mouth": mouth,
                    "start_sample": phone_start,
                    "end_sample": phone_end,
                    "start_sec": phone_start / float(sample_rate),
                    "end_sec": phone_end / float(sample_rate),
                }
            )
        sentences.append(
            {
                "id": "s{:03d}".format(index + 1),
                "text": SENTENCE_TEXTS[index % len(SENTENCE_TEXTS)],
                "start_sample": start,
                "end_sample": end,
                "start_sec": start / float(sample_rate),
                "end_sec": end / float(sample_rate),
                "phonemes": phonemes,
            }
        )
        gap_sec = 0.52 + float(rng.random()) * 0.24
        cursor = end + round_half_up(gap_sec * sample_rate)

    duration_samples = cursor + round_half_up(0.72 * sample_rate)
    return {
        "schema_version": 1,
        "kind": "lipalign-timeline",
        "sample_rate": int(sample_rate),
        "seed": int(seed),
        "duration_samples": int(duration_samples),
        "duration_sec": duration_samples / float(sample_rate),
        "mouth_categories": list(MOUTH_PHONEMES.keys()),
        "rounding": "nearest integer sample, ties upward",
        "sentences": sentences,
    }


def _mouth_track(
    phoneme_payload: Sequence[Dict[str, Any]],
    start: int,
    end_sentence: int,
) -> np.ndarray:
    """Return a per-sample mouth-category index, with short transitions."""

    length = end_sentence - start
    result = np.zeros(length, dtype=np.int16)
    categories = list(MOUTH_PHONEMES.keys())
    category_index = {name: i for i, name in enumerate(categories)}
    for phone in phoneme_payload:
        local_start = max(0, int(phone["start_sample"]) - start)
        local_end = min(length, int(phone["end_sample"]) - start)
        if local_end <= local_start:
            continue
        result[local_start:local_end] = category_index[phone["mouth"]]
    return result


def _smooth_formants(
    mouth_indices: np.ndarray,
    sample_rate: int,
    formant_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a categorical mouth track to smooth F1/F2 contours."""

    categories = list(MOUTH_PHONEMES.keys())
    f1_table = np.array([MOUTH_FORMANTS[name][0] for name in categories], dtype=np.float64)
    f2_table = np.array([MOUTH_FORMANTS[name][1] for name in categories], dtype=np.float64)
    raw_f1 = f1_table[mouth_indices] * formant_scale
    raw_f2 = f2_table[mouth_indices] * formant_scale
    smooth = max(2, int(round(0.010 * sample_rate)))
    if smooth % 2 == 0:
        smooth += 1
    kernel = np.hanning(smooth)
    kernel /= float(np.sum(kernel))
    pad = smooth // 2
    f1 = np.convolve(np.pad(raw_f1, pad, mode="edge"), kernel, mode="valid")
    f2 = np.convolve(np.pad(raw_f2, pad, mode="edge"), kernel, mode="valid")
    return f1, f2


def _synthesize_sentence(
    sample_count: int,
    phoneme_payload: Sequence[Dict[str, Any]],
    timeline_sentence_start: int,
    timeline_sentence_end: int,
    sample_rate: int,
    language_spec: Dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize a voiced, formant-shaped phrase of an exact sample count."""

    n = int(max(sample_count, 1))
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    base_f0 = float(language_spec["pitch"]) * (0.97 + 0.06 * float(rng.random()))
    f0 = base_f0 * (
        1.0
        + 0.030 * np.sin(2.0 * np.pi * 0.73 * t + float(rng.random()) * 2.0 * np.pi)
        + 0.014 * np.sin(2.0 * np.pi * 2.11 * t + 0.4)
    )
    phase = 2.0 * np.pi * np.cumsum(f0) / float(sample_rate)

    target_duration = max(1, timeline_sentence_end - timeline_sentence_start)
    scale = float(n) / float(target_duration)
    scaled_phones: List[Dict[str, Any]] = []
    for phone in phoneme_payload:
        scaled_phones.append(
            {
                "mouth": phone["mouth"],
                "start_sample": int(round((int(phone["start_sample"]) - timeline_sentence_start) * scale)),
                "end_sample": int(round((int(phone["end_sample"]) - timeline_sentence_start) * scale)),
            }
        )
    if scaled_phones:
        scaled_phones[0]["start_sample"] = 0
        scaled_phones[-1]["end_sample"] = n
    mouth_indices = _mouth_track(scaled_phones, 0, n)
    f1, f2 = _smooth_formants(mouth_indices, sample_rate, float(language_spec["formant_scale"]))

    wave = np.zeros(n, dtype=np.float64)
    harmonic_count = 18
    for harmonic in range(1, harmonic_count + 1):
        frequency = harmonic * f0
        weight = 1.0 / (float(harmonic) ** 0.82)
        f1_weight = 1.75 * np.exp(-0.5 * ((frequency - f1) / 155.0) ** 2)
        f2_weight = 1.10 * np.exp(-0.5 * ((frequency - f2) / 230.0) ** 2)
        wave += weight * (1.0 + f1_weight + f2_weight) * np.sin(harmonic * phase)

    noise = rng.standard_normal(n)
    high_noise = np.diff(noise, prepend=noise[0])
    texture = np.zeros(n, dtype=np.float64)
    for phone in scaled_phones:
        local_start = max(0, int(phone["start_sample"]))
        local_end = min(n, int(phone["end_sample"]))
        if local_end <= local_start:
            continue
        mouth = str(phone["mouth"])
        strength = 0.0
        if mouth == "fricative":
            strength = 0.20
        elif mouth == "labiodental":
            strength = 0.11
        elif mouth == "closed":
            strength = 0.05
        if strength > 0.0:
            phone_len = local_end - local_start
            burst_len = min(phone_len, max(1, int(round(0.030 * sample_rate))))
            envelope = np.linspace(1.0, 0.0, burst_len, endpoint=False) ** 1.7
            texture[local_start : local_start + burst_len] += (
                strength * envelope * high_noise[local_start : local_start + burst_len]
            )

    amplitude = 0.86 + 0.14 * np.sin(2.0 * np.pi * 3.7 * t + 0.2)
    waveform = wave + texture
    waveform *= amplitude
    waveform -= float(np.mean(waveform))

    fade = min(int(round(0.012 * sample_rate)), n // 2)
    if fade > 0:
        ramp = _raised_cosine(fade)
        waveform[:fade] *= ramp
        waveform[-fade:] *= ramp[::-1]

    peak = float(np.max(np.abs(waveform)))
    if peak > 1e-12:
        waveform *= 0.34 / peak
    return waveform


def synthesize_language_track(
    timeline: Dict[str, Any],
    language: str,
    seed: int,
) -> np.ndarray:
    """Return a full mono float track for one language."""

    if language not in LANGUAGE_SPECS:
        raise LipAlignError(
            "E_SYNTH", "Unknown built-in language: {}".format(language), exit_code=2
        )
    sample_rate = int(timeline["sample_rate"])
    total_samples = int(timeline["duration_samples"])
    track = np.zeros(total_samples, dtype=np.float64)
    spec = LANGUAGE_SPECS[language]
    language_stream = 200 + DEFAULT_LANGUAGES.index(language) * 37
    rng = np.random.default_rng(_seed_sequence(seed, language_stream))

    for sentence in timeline["sentences"]:
        start = int(sentence["start_sample"])
        end = int(sentence["end_sample"])
        target_samples = end - start
        speech_count = max(
            32,
            round_half_up(target_samples / float(spec["speed"])),
        )
        speech = _synthesize_sentence(
            speech_count,
            sentence["phonemes"],
            start,
            end,
            sample_rate,
            spec,
            rng,
        )
        write_start = start
        write_end = min(total_samples, write_start + len(speech))
        if write_end <= write_start:
            continue
        track[write_start:write_end] += speech[: write_end - write_start]
    return track


def generate_pink_noise(
    sample_count: int,
    sample_rate: int,
    seed: int,
    target_rms: float = 0.0055,
) -> np.ndarray:
    """Generate low-level deterministic pink noise in the time domain via FFT."""

    n = int(sample_count)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(_seed_sequence(seed, 991))
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    scale = 1.0 / np.sqrt(np.maximum(frequencies, 1.0))
    scale[0] = 0.0
    pink = np.fft.irfft(spectrum * scale, n=n)
    pink -= float(np.mean(pink))
    rms = float(np.sqrt(np.mean(pink * pink)))
    if rms > 1e-12:
        pink *= target_rms / rms
    peak = float(np.max(np.abs(pink)))
    if peak > 0.035:
        pink *= 0.035 / peak
    return pink


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert float samples in [-1, 1] to deterministic int16 PCM."""

    clipped = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype(np.int16)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_samples(
    out_dir: PathLike,
    seed: int,
    languages: Optional[Sequence[str]] = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Dict[str, Any]:
    """Generate timeline, language WAVs, background WAV, and manifest."""

    output_root = Path(out_dir)
    language_list = list(languages) if languages is not None else list(DEFAULT_LANGUAGES)
    if len(language_list) < 3:
        raise LipAlignError(
            "E_SYNTH", "At least three languages are required", exit_code=2
        )
    if len(set(language_list)) != len(language_list):
        raise LipAlignError("E_SYNTH", "Duplicate language names", exit_code=2)
    unknown = [name for name in language_list if name not in LANGUAGE_SPECS]
    if unknown:
        raise LipAlignError(
            "E_SYNTH",
            "Unknown built-in language(s): {}".format(", ".join(unknown)),
            exit_code=2,
        )
    if int(sample_rate) < 8000:
        raise LipAlignError("E_SYNTH", "sample_rate must be at least 8000", exit_code=2)

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        audio_root = output_root / "audio"
        audio_root.mkdir(parents=True, exist_ok=True)
        timeline_payload = build_timeline_payload(seed, int(sample_rate))
        timeline_path = output_root / "timeline.json"
        _write_json(timeline_path, timeline_payload)

        generated_files: Dict[str, str] = {"timeline": str(timeline_path)}
        language_manifest: List[Dict[str, Any]] = []
        for language in language_list:
            track = synthesize_language_track(timeline_payload, language, int(seed))
            pcm = float_to_pcm16(track)
            wav_path = audio_root / "{}.wav".format(language)
            write_pcm16_wav(wav_path, pcm, int(sample_rate), channels=1)
            generated_files["audio/{}".format(wav_path.name)] = str(wav_path)
            language_manifest.append(
                {
                    "language": language,
                    "file": "audio/{}".format(wav_path.name),
                    "speech_speed": LANGUAGE_SPECS[language]["speed"],
                    "pitch_hz": LANGUAGE_SPECS[language]["pitch"],
                    "sample_count": int(len(pcm)),
                }
            )

        background = generate_pink_noise(
            int(timeline_payload["duration_samples"]), int(sample_rate), int(seed)
        )
        background_pcm = float_to_pcm16(background)
        background_path = output_root / "background.wav"
        write_pcm16_wav(background_path, background_pcm, int(sample_rate), channels=1)
        generated_files["background"] = str(background_path)

        hashes = {
            "timeline.json": sha256_file(timeline_path),
            "background.wav": sha256_file(background_path),
        }
        for language in language_list:
            name = "audio/{}.wav".format(language)
            hashes[name] = sha256_file(output_root / name)
        manifest = {
            "schema_version": 1,
            "kind": "lipalign-sample-manifest",
            "seed": int(seed),
            "sample_rate": int(sample_rate),
            "timeline": "timeline.json",
            "background": "background.wav",
            "languages": language_manifest,
            "sha256": hashes,
        }
        manifest_path = output_root / "manifest.json"
        _write_json(manifest_path, manifest)
        generated_files["manifest"] = str(manifest_path)
    except LipAlignError:
        raise
    except OSError as exc:
        raise LipAlignError(
            "E_OUTPUT", "Cannot write generated samples: {}".format(exc), exit_code=2
        ) from exc

    return {
        "seed": int(seed),
        "sample_rate": int(sample_rate),
        "languages": language_list,
        "sentence_count": len(timeline_payload["sentences"]),
        "duration_sec": float(timeline_payload["duration_sec"]),
        "output_dir": str(output_root),
        "files": generated_files,
        "manifest": manifest,
    }

