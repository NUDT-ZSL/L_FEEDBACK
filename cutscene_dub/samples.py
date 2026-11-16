from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .audio import SAMPLE_RATE, float_to_pcm16
from .errors import BusinessError
from .io_utils import (
    AtomicWavWriter,
    sha256_file,
)
from .model import Dialogue, Timeline, canonical_json_bytes
from .state import SampleState, sample_state_path


SAMPLE_GENERATOR_VERSION = 1


def dialogue_sample_path(audio_root: Path, language: str, dialogue_id: str) -> Path:
    return audio_root / language / f"{dialogue_id}.wav"


def bgm_sample_path(audio_root: Path) -> Path:
    return audio_root / "bgm.wav"


def _unit(*parts: Any) -> float:
    payload = b"\0".join(str(part).encode("utf-8") for part in parts)
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _waveform_phase(*parts: Any) -> float:
    return 2.0 * math.pi * _unit("phase", *parts)


def _speech_samples(
    dialogue: Dialogue,
    *,
    seed: int,
    sample_rate: int,
) -> np.ndarray:
    window_s = dialogue.window_s
    context = (seed, dialogue.target_language, dialogue.id)
    if dialogue.source_duration_s is not None:
        duration_s = dialogue.source_duration_s
    else:
        factor = 1.02 + 0.20 * _unit(*context, "duration-factor")
        duration_s = window_s * factor
    total_frames = max(1, int(round(duration_s * sample_rate)))
    lead_ratio = 0.05 + 0.13 * _unit(*context, "lead")
    tail_ratio = 0.06 + 0.17 * _unit(*context, "tail")
    lead_frames = min(
        max(0, int(round(total_frames * lead_ratio))),
        max(0, total_frames - 2),
    )
    tail_frames = min(
        max(0, int(round(total_frames * tail_ratio))),
        max(0, total_frames - lead_frames - 1),
    )
    active_frames = total_frames - lead_frames - tail_frames
    if active_frames < max(1, int(0.01 * sample_rate)):
        raise BusinessError(
            code="E_SAMPLE_GENERATION",
            message="source_duration_s is too short to contain a generated utterance.",
            entry_id=dialogue.id,
            details={"source_duration_s": duration_s},
        )

    t = np.arange(active_frames, dtype=np.float64) / sample_rate
    f0 = 105.0 + 55.0 * _unit(*context, "f0")
    f0 += 4.0 * np.sin(2.0 * math.pi * 0.71 * t + _waveform_phase(*context, "vib"))
    phase = 2.0 * math.pi * np.cumsum(f0) / sample_rate
    phase += _waveform_phase(*context, "carrier")
    syllable_rate = 3.0 + 1.9 * _unit(*context, "syllable-rate")
    syllable_phase = _waveform_phase(*context, "syllable")
    syllable = 0.30 + 0.70 * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * syllable_rate * t + syllable_phase)
    ) ** 1.8
    word_gate = 0.62 + 0.38 * (
        0.5 + 0.5 * np.sin(2.0 * math.pi * 0.63 * t + syllable_phase * 0.37)
    )
    harmonics = (
        0.78 * np.sin(phase)
        + 0.30 * np.sin(2.0 * phase + 0.4)
        + 0.15 * np.sin(3.0 * phase + 1.1)
        + 0.07 * np.sin(5.0 * phase + 0.7)
    )
    breath_phase = _waveform_phase(*context, "breath")
    breath = (
        0.042 * np.sin(2.0 * math.pi * 2330.0 * t + breath_phase)
        + 0.028 * np.sin(2.0 * math.pi * 3710.0 * t + breath_phase * 1.7)
    )
    signal = (harmonics + breath) * syllable * word_gate
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal * (0.52 / peak)
    edge = min(active_frames, max(1, int(round(0.006 * sample_rate))))
    ramp = np.linspace(0.0, 1.0, edge, endpoint=False)
    signal[:edge] *= ramp
    signal[-edge:] *= ramp[::-1]

    output = np.zeros(total_frames, dtype=np.float64)
    output[lead_frames : lead_frames + active_frames] = signal
    return output


def _bgm_samples(
    timeline: Timeline,
    *,
    seed: int,
    sample_rate: int,
) -> np.ndarray:
    total_frames = int(round(timeline.duration_s * sample_rate))
    if total_frames <= 0:
        raise BusinessError(
            code="E_SAMPLE_GENERATION",
            message="Timeline duration is too short to generate background music.",
        )
    t = np.arange(total_frames, dtype=np.float64) / sample_rate
    base = 52.0 + 18.0 * _unit(seed, timeline.project_id, "bgm-base")
    chord = (1.0, 1.25, 1.5, 2.0)
    phases = [
        _waveform_phase(seed, timeline.project_id, "bgm", index)
        for index in range(len(chord))
    ]
    pad = np.zeros(total_frames, dtype=np.float64)
    for index, (ratio, phase) in enumerate(zip(chord, phases, strict=True)):
        detune = 1.0 + 0.0012 * np.sin(
            2.0 * math.pi * (0.13 + index * 0.03) * t + phase
        )
        pad += (0.42 / (index + 1)) * np.sin(
            2.0 * math.pi * base * ratio * t * detune + phase
        )
    slow = 0.72 + 0.28 * np.sin(
        2.0 * math.pi * 0.11 * t + _waveform_phase(seed, timeline.project_id, "bgm-slow")
    )
    beat_period = 0.48
    beat_pos = np.mod(t, beat_period)
    pulse = np.exp(-beat_pos * 9.0) * np.sin(
        2.0 * math.pi * base * 3.0 * t
        + _waveform_phase(seed, timeline.project_id, "bgm-pulse")
    )
    signal = (pad * slow + 0.16 * pulse) * 0.20
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal = signal * (0.24 / peak)
    edge_in = min(total_frames, max(1, int(round(0.20 * sample_rate))))
    edge_out = min(total_frames, max(1, int(round(0.35 * sample_rate))))
    fade_in = np.linspace(0.0, 1.0, edge_in, endpoint=False)
    fade_out = np.linspace(0.0, 1.0, edge_out, endpoint=False)
    signal[:edge_in] *= fade_in
    signal[-edge_out:] *= fade_out[::-1]
    return signal


def _write_sample(
    path: Path,
    samples: np.ndarray,
    *,
    sample_rate: int,
) -> dict[str, Any]:
    pcm = float_to_pcm16(samples)
    writer = AtomicWavWriter(
        path,
        sample_rate=sample_rate,
        channels=1,
        failpoint="sample_partial",
    )
    try:
        writer.write_pcm(pcm)
        writer.commit()
    finally:
        if not writer._committed:
            writer.abort(keep_partial=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "frames": int(pcm.shape[0]),
        "duration_s": pcm.shape[0] / float(sample_rate),
        "peak_dbfs": 20.0 * math.log10(max(1.0, float(np.max(np.abs(pcm)))) / 32767.0),
    }


def _sample_fingerprint(
    timeline: Timeline,
    *,
    seed: int,
    sample_rate: int,
) -> str:
    value = {
        "tool_version": __version__,
        "generator_version": SAMPLE_GENERATOR_VERSION,
        "timeline_sha256": timeline.source_sha256,
        "seed": seed,
        "target_language": timeline.target_language,
        "sample_rate_hz": sample_rate,
        "dialogues": [
            {
                "id": dialogue.id,
                "start_s": dialogue.start_s,
                "end_s": dialogue.end_s,
                "text": dialogue.text,
                "source_duration_s": dialogue.source_duration_s,
            }
            for dialogue in timeline.dialogues
        ],
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def generate_samples(
    timeline: Timeline,
    *,
    audio_root: str | Path,
    seed: int,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(audio_root)
    root.mkdir(parents=True, exist_ok=True)
    expected_paths = [bgm_sample_path(root)] + [
        dialogue_sample_path(root, timeline.target_language, dialogue.id)
        for dialogue in timeline.dialogues
    ]
    fingerprint = _sample_fingerprint(timeline, seed=seed, sample_rate=SAMPLE_RATE)
    state = SampleState(sample_state_path(root))
    reused = False
    if not force and state.is_reusable(fingerprint=fingerprint, expected_paths=expected_paths):
        reused = True
    else:
        state.begin(fingerprint=fingerprint, expected_paths=expected_paths)
        records: list[dict[str, Any]] = []
        try:
            bgm = _bgm_samples(timeline, seed=seed, sample_rate=SAMPLE_RATE)
            bgm_record = _write_sample(
                bgm_sample_path(root), bgm, sample_rate=SAMPLE_RATE
            )
            bgm_record.update({"kind": "bgm", "dialogue_id": None, "language": None})
            records.append(bgm_record)
            state.file_published(Path(bgm_record["path"]), bgm_record["sha256"])

            for dialogue in timeline.dialogues:
                speech = _speech_samples(
                    dialogue, seed=seed, sample_rate=SAMPLE_RATE
                )
                path = dialogue_sample_path(
                    root, timeline.target_language, dialogue.id
                )
                record = _write_sample(path, speech, sample_rate=SAMPLE_RATE)
                record.update(
                    {
                        "kind": "dialogue",
                        "dialogue_id": dialogue.id,
                        "language": timeline.target_language,
                    }
                )
                records.append(record)
                state.file_published(path, record["sha256"])
            state.complete()
        except KeyboardInterrupt:
            state.interrupted("sample_generation")
            raise
        except BusinessError as exc:
            state.failed(exc)
            raise
        except Exception as exc:
            wrapped = BusinessError(
                code="E_SAMPLE_GENERATION",
                message=f"Unexpected sample generation failure: {exc}",
            )
            state.failed(wrapped)
            raise wrapped from exc

    if reused:
        records = []
        for path in expected_paths:
            record = {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            if path == bgm_sample_path(root):
                record.update({"kind": "bgm", "dialogue_id": None, "language": None})
            else:
                dialogue_id = path.stem
                record.update(
                    {
                        "kind": "dialogue",
                        "dialogue_id": dialogue_id,
                        "language": timeline.target_language,
                    }
                )
            records.append(record)

    return {
        "ok": True,
        "command": "generate-samples",
        "tool_version": __version__,
        "reused": reused,
        "seed": seed,
        "timeline_sha256": timeline.source_sha256,
        "audio_root": str(root),
        "bgm": str(bgm_sample_path(root)),
        "files": records,
    }

