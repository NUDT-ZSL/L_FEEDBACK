from __future__ import annotations

import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    audio_io_error,
    channel_count_error,
    invalid_config,
    unsupported_bit_depth,
    unsupported_sample_rate,
)
from .schema import InputMetadata, ProjectConfig, TrackSpec, build_config_digest
from .util import ensure_existing_file, sha256_file


@dataclass(frozen=True)
class LoadedTrack:
    spec: TrackSpec
    sample_rate: int
    channels: int
    pcm: array
    frames: int
    metadata: InputMetadata


@dataclass(frozen=True)
class LoadedProject:
    config: ProjectConfig
    commentary: tuple[LoadedTrack, ...]
    engine: LoadedTrack
    ambience: LoadedTrack
    config_digest: str


def _read_wav(path: Path, spec: TrackSpec, expected_sample_rate: int) -> tuple[array, int, int]:
    try:
        wav = wave.open(str(path), "rb")
    except (wave.Error, OSError) as exc:
        raise audio_io_error(
            "cannot open input WAV",
            path=spec.path_value,
            role=spec.role,
            error=str(exc),
        ) from None
    try:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        compression = wav.getcomptype()
        if compression != "NONE":
            raise audio_io_error(
                "compressed WAV input is not supported",
                path=spec.path_value,
                role=spec.role,
                compression=compression,
            )
        if sample_width != 2:
            raise unsupported_bit_depth(
                "input WAV must be 16-bit PCM",
                path=spec.path_value,
                role=spec.role,
                bits=sample_width * 8,
            )
        if channels not in (1, 2):
            raise channel_count_error(
                "input WAV must be mono or stereo",
                path=spec.path_value,
                role=spec.role,
                channels=channels,
            )
        if channels != spec.channels:
            raise audio_io_error(
                "declared track channels do not match WAV header",
                path=spec.path_value,
                role=spec.role,
                declared=spec.channels,
                wav=channels,
            )
        if sample_rate != expected_sample_rate:
            raise unsupported_sample_rate(
                "input WAV sample rate does not match configuration sample_rate",
                path=spec.path_value,
                role=spec.role,
                expected=expected_sample_rate,
                wav=sample_rate,
            )
        raw = wav.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise audio_io_error(
            "cannot decode input WAV",
            path=spec.path_value,
            role=spec.role,
            error=str(exc),
        ) from None
    finally:
        wav.close()

    expected_bytes = frame_count * channels * 2
    if len(raw) != expected_bytes:
        raise audio_io_error(
            "input WAV data length does not match its header",
            path=spec.path_value,
            role=spec.role,
            expected_bytes=expected_bytes,
            actual_bytes=len(raw),
        )
    pcm = array("h")
    pcm.frombytes(raw)
    if sys.byteorder == "big":
        pcm.byteswap()
    return pcm, channels, sample_rate


def load_project(config: ProjectConfig) -> LoadedProject:
    loaded: list[LoadedTrack] = []
    metadata: list[InputMetadata] = []
    specs = (*config.commentary, config.engine, config.ambience)
    for spec in specs:
        ensure_existing_file(spec.path, spec.role, spec.path_value)
        pcm, channels, sample_rate = _read_wav(spec.path, spec, config.sample_rate)
        frames = len(pcm) // channels
        wav_bytes = spec.path.stat().st_size
        digest = sha256_file(spec.path)
        meta = InputMetadata(
            role=spec.role,
            name=spec.name,
            path_value=spec.path_value,
            sha256=digest,
            wav_bytes=wav_bytes,
            sample_rate=sample_rate,
            channels=channels,
            frames=frames,
        )
        metadata.append(meta)
        track = LoadedTrack(
            spec=spec,
            sample_rate=sample_rate,
            channels=channels,
            pcm=pcm,
            frames=frames,
            metadata=meta,
        )
        loaded.append(track)
        if spec.role == "commentary":
            source_duration_ms = frames * 1000.0 / sample_rate
            for index, anchor in enumerate(spec.anchors):
                if anchor.source_ms > source_duration_ms + 1.0e-9:
                    raise invalid_config(
                        "commentary anchor source_ms exceeds source WAV duration",
                        path=spec.path_value,
                        track=spec.name,
                        index=index,
                        source_ms=anchor.source_ms,
                        source_duration_ms=source_duration_ms,
                    )

    commentary_count = len(config.commentary)
    commentary = tuple(loaded[:commentary_count])
    engine = loaded[commentary_count]
    ambience = loaded[commentary_count + 1]
    return LoadedProject(
        config=config,
        commentary=commentary,
        engine=engine,
        ambience=ambience,
        config_digest=build_config_digest(config, metadata),
    )
