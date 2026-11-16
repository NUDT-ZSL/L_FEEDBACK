"""End-to-end alignment, caching, assembly, and report generation."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .cache import FragmentCache
from .errors import LipAlignError
from .jsonio import atomic_write_json
from .stretch import StretchSettings, time_stretch
from .synth import float_to_pcm16, generate_pink_noise
from .timeline import Timeline, load_timeline
from .vad import SplitSettings, split_language
from .wavio import (
    read_wav,
    sha256_file,
    sha256_int16,
    write_pcm16_wav,
)


PathLike = Union[str, os.PathLike]


@dataclass
class AlignOptions:
    timeline_path: PathLike
    audio_inputs: Sequence[Tuple[str, PathLike]]
    output_path: PathLike
    report_path: PathLike
    cache_dir: PathLike
    algorithm: str = "wsola"
    split_settings: Optional[SplitSettings] = None
    stretch_settings: Optional[StretchSettings] = None
    background_path: Optional[PathLike] = None
    crossfade_ms: float = 10.0


@dataclass
class AlignResult:
    report: Dict[str, Any]
    partial: bool
    stats: Dict[str, Any]


def _seconds(samples: int, sample_rate: int) -> float:
    return samples / float(sample_rate)


def _rounded(value: float, digits: int = 9) -> float:
    return float(round(value, digits))


def _error_record(
    language: str,
    sentence_id: str,
    code: str,
    message: str,
) -> Dict[str, Any]:
    return {
        "language": language,
        "sentence_id": sentence_id,
        "code": code,
        "message": message,
    }


def _place_with_crossfade(
    track: np.ndarray,
    fragment_pcm: np.ndarray,
    start_sample: int,
    crossfade_samples: int,
) -> None:
    """Place a fragment on the fixed timeline with 10 ms equal-power edges.

    The head and tail equal-power crossfade into whatever already occupies the
    channel (the previous fragment tail when sentences touch, or timeline
    silence when they have a gap).  Placement itself remains sample-anchored,
    so the final track never changes length.
    """

    piece = np.asarray(fragment_pcm, dtype=np.float64).reshape(-1) / 32768.0
    if len(piece) == 0:
        return
    start = max(0, int(start_sample))
    end = min(len(track), start + len(piece))
    if end <= start:
        return
    piece = piece[: end - start]
    fade = min(int(crossfade_samples), len(piece) // 2)
    if fade <= 0:
        track[start:end] += piece
        return

    ramp = (np.arange(fade, dtype=np.float64) + 1.0) / float(fade + 1)
    fade_in = np.sin(0.5 * np.pi * ramp)
    fade_out = np.cos(0.5 * np.pi * ramp)

    head_existing = track[start : start + fade].copy()
    track[start : start + fade] = (
        head_existing * fade_out + piece[:fade] * fade_in
    )
    interior_end = len(piece) - fade
    if interior_end > fade:
        track[start + fade : start + interior_end] += piece[fade:interior_end]
    room_tail = min(fade, end - (start + interior_end))
    if room_tail > 0:
        tail_start = end - room_tail
        tail_existing = track[tail_start:end].copy()
        track[tail_start:end] = (
            tail_existing * fade_in[:room_tail]
            + piece[-room_tail:] * fade_out[:room_tail]
        )


def _load_background(
    background_path: Optional[PathLike],
    timeline: Timeline,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if background_path is None:
        samples = generate_pink_noise(
            timeline.duration_samples,
            timeline.sample_rate,
            timeline.seed,
        )
        pcm = float_to_pcm16(samples)
        return pcm, {
            "source": "built-in deterministic pink noise",
            "path": None,
            "sha256": None,
            "target_rms": 0.0055,
        }

    audio = read_wav(background_path)
    if audio.channels != 1:
        raise LipAlignError(
            "E_CHANNEL_MISMATCH",
            "Background WAV must be mono: {}".format(background_path),
            exit_code=2,
        )
    if audio.sample_rate != timeline.sample_rate:
        raise LipAlignError(
            "E_RATE_MISMATCH",
            "Background sample rate {} Hz does not match timeline {} Hz".format(
                audio.sample_rate, timeline.sample_rate
            ),
            exit_code=2,
            details={
                "expected_rate": timeline.sample_rate,
                "actual_rate": audio.sample_rate,
                "background": str(background_path),
            },
        )
    if audio.sample_count < timeline.duration_samples:
        raise LipAlignError(
            "E_AUDIO_LENGTH",
            "Background is shorter than the timeline",
            exit_code=2,
            details={
                "background_samples": audio.sample_count,
                "required_samples": timeline.duration_samples,
            },
        )
    pcm = audio.samples[: timeline.duration_samples, 0].copy()
    return pcm, {
        "source": "provided WAV",
        "path": str(background_path),
        "sha256": sha256_file(background_path),
        "target_rms": None,
    }


def align(options: AlignOptions) -> AlignResult:
    started = time.perf_counter()
    split_settings = options.split_settings or SplitSettings()
    split_settings.validate()
    if options.stretch_settings is None:
        stretch_settings = StretchSettings(algorithm=options.algorithm)
    else:
        stretch_settings = options.stretch_settings
        if stretch_settings.algorithm != options.algorithm:
            stretch_settings = StretchSettings(
                algorithm=options.algorithm,
                wsola_frame_ms=stretch_settings.wsola_frame_ms,
                wsola_overlap_ms=stretch_settings.wsola_overlap_ms,
                wsola_search_ms=stretch_settings.wsola_search_ms,
                wsola_search_step=stretch_settings.wsola_search_step,
                pv_frame_size=stretch_settings.pv_frame_size,
                pv_overlap=stretch_settings.pv_overlap,
            )
    stretch_settings.validate()
    if options.crossfade_ms < 0:
        raise LipAlignError(
            "E_PARAMETER", "crossfade_ms must be >= 0", exit_code=2
        )

    timeline = load_timeline(options.timeline_path)
    timeline_sha256 = sha256_file(timeline.source_path)
    cache = FragmentCache(options.cache_dir, timeline.sample_rate, stretch_settings)
    crossfade_samples = int(
        round(options.crossfade_ms * timeline.sample_rate / 1000.0)
    )

    seen_languages = set()
    normalized_inputs: List[Tuple[str, Path]] = []
    for language, path in options.audio_inputs:
        if not language or language in seen_languages:
            raise LipAlignError(
                "E_PARAMETER",
                "Language names must be non-empty and unique",
                exit_code=2,
            )
        seen_languages.add(language)
        normalized_inputs.append((language, Path(path)))
    if not normalized_inputs:
        raise LipAlignError(
            "E_PARAMETER", "At least one language WAV is required", exit_code=2
        )

    language_reports: List[Dict[str, Any]] = []
    language_pcm_tracks: List[np.ndarray] = []
    errors: List[Dict[str, Any]] = []
    cache_hits = 0
    cache_recomputes = 0
    sentences_processed = 0
    sentences_failed = 0

    for language, audio_path in normalized_inputs:
        audio = read_wav(audio_path)
        if audio.channels != 1:
            raise LipAlignError(
                "E_CHANNEL_MISMATCH",
                "Language WAV must be mono: {}".format(audio_path),
                exit_code=2,
                details={"language": language, "channels": audio.channels},
            )
        if audio.sample_rate != timeline.sample_rate:
            raise LipAlignError(
                "E_RATE_MISMATCH",
                "Language sample rate {} Hz does not match timeline {} Hz".format(
                    audio.sample_rate, timeline.sample_rate
                ),
                exit_code=2,
                details={
                    "language": language,
                    "expected_rate": timeline.sample_rate,
                    "actual_rate": audio.sample_rate,
                },
            )
        if audio.sample_count < timeline.duration_samples:
            raise LipAlignError(
                "E_AUDIO_LENGTH",
                "Language WAV is shorter than the timeline",
                exit_code=2,
                details={
                    "language": language,
                    "source_samples": audio.sample_count,
                    "required_samples": timeline.duration_samples,
                },
            )

        source_pcm = audio.samples[: timeline.duration_samples, 0].copy()
        source_float = source_pcm.astype(np.float64) / 32768.0
        split = split_language(
            source_float,
            timeline.sample_rate,
            timeline.sentences,
            split_settings,
        )
        voice_track = np.zeros(timeline.duration_samples, dtype=np.float64)
        sentence_reports: List[Dict[str, Any]] = []
        language_hits = 0
        language_recomputes = 0

        for index, sentence in enumerate(timeline.sentences):
            target_samples = sentence.duration_samples
            target_start_sec = _seconds(sentence.start_sample, timeline.sample_rate)
            target_end_sec = _seconds(sentence.end_sample, timeline.sample_rate)
            assignment = split.segments[index]
            report_item: Dict[str, Any] = {
                "index": index,
                "id": sentence.sentence_id,
                "text": sentence.text,
                "phoneme_count": len(sentence.phonemes),
                "mouth_sequence": [phone.mouth for phone in sentence.phonemes],
                "target_start_sample": sentence.start_sample,
                "target_end_sample": sentence.end_sample,
                "target_start_sec": _rounded(target_start_sec),
                "target_end_sec": _rounded(target_end_sec),
                "target_duration_sec": _rounded(
                    _seconds(target_samples, timeline.sample_rate)
                ),
                "target_samples": target_samples,
                "actual_start_sample": None,
                "actual_end_sample": None,
                "actual_start_sec": None,
                "actual_end_sec": None,
                "source_start_sample": None,
                "source_end_sample": None,
                "source_samples": None,
                "source_duration_sec": None,
                "source_content_sha256": None,
                "stretch_factor": None,
                "stretch_factor_fraction": None,
                "output_samples": None,
                "algorithm": stretch_settings.algorithm,
                "rounding_policy": (
                    "target samples are integer timeline samples; stretched output "
                    "is cropped/padded to target_samples; PCM uses rint(x*32767)"
                ),
                "cache": None,
                "cache_key": None,
                "cache_write_error": None,
                "output_sha256": None,
                "status": "pending",
                "error": None,
            }

            if assignment is None:
                code = "E_SEGMENT_NOT_FOUND"
                message = "No active speech segment could be assigned to this sentence"
                report_item["status"] = "failed"
                report_item["error"] = {"code": code, "message": message}
                sentence_reports.append(report_item)
                errors.append(
                    _error_record(language, sentence.sentence_id, code, message)
                )
                sentences_failed += 1
                continue

            source_start, source_end = assignment
            source_start = max(0, min(int(source_start), timeline.duration_samples - 1))
            source_end = max(source_start + 1, min(int(source_end), timeline.duration_samples))
            clip_pcm = source_pcm[source_start:source_end].copy()
            source_samples = len(clip_pcm)
            source_hash = sha256_int16(clip_pcm)
            factor = float(target_samples) / float(source_samples)
            key = cache.key_for(source_hash, source_samples, target_samples)
            report_item.update(
                {
                    "source_start_sample": source_start,
                    "source_end_sample": source_end,
                    "source_samples": source_samples,
                    "source_duration_sec": _rounded(
                        _seconds(source_samples, timeline.sample_rate)
                    ),
                    "source_content_sha256": source_hash,
                    "stretch_factor": _rounded(factor),
                    "stretch_factor_fraction": "{}/{}".format(
                        target_samples, source_samples
                    ),
                    "cache_key": key,
                }
            )

            try:
                cached = cache.get(key, source_hash, source_samples, target_samples)
                if cached is not None:
                    output_pcm = cached
                    report_item["cache"] = "hit"
                    cache_hits += 1
                    language_hits += 1
                else:
                    cache_recomputes += 1
                    language_recomputes += 1
                    clip_float = clip_pcm.astype(np.float64) / 32768.0
                    stretched = time_stretch(
                        clip_float,
                        target_samples,
                        timeline.sample_rate,
                        stretch_settings,
                    )
                    if len(stretched) != target_samples:
                        raise LipAlignError(
                            "E_STRETCH_LENGTH",
                            "Stretcher returned {} samples, expected {}".format(
                                len(stretched), target_samples
                            ),
                            exit_code=9,
                        )
                    output_pcm = float_to_pcm16(stretched)
                    report_item["cache"] = "recomputed"
                    try:
                        cache.put(
                            key,
                            source_hash,
                            source_samples,
                            target_samples,
                            output_pcm,
                        )
                    except Exception as exc:
                        report_item["cache_write_error"] = "{}: {}".format(
                            type(exc).__name__, exc
                        )

                output_hash = sha256_int16(output_pcm)
                report_item["output_sha256"] = output_hash
                report_item["output_samples"] = int(len(output_pcm))
                _place_with_crossfade(
                    voice_track,
                    output_pcm,
                    sentence.start_sample,
                    crossfade_samples,
                )
                report_item["actual_start_sample"] = sentence.start_sample
                report_item["actual_end_sample"] = sentence.end_sample
                report_item["actual_start_sec"] = _rounded(target_start_sec)
                report_item["actual_end_sec"] = _rounded(target_end_sec)
                report_item["status"] = "ok"
                sentences_processed += 1
            except LipAlignError as exc:
                report_item["status"] = "failed"
                report_item["error"] = {"code": exc.code, "message": exc.message}
                errors.append(
                    _error_record(language, sentence.sentence_id, exc.code, exc.message)
                )
                sentences_failed += 1
            except Exception as exc:
                code = "E_STRETCH"
                message = "{}: {}".format(type(exc).__name__, exc)
                report_item["status"] = "failed"
                report_item["error"] = {"code": code, "message": message}
                errors.append(
                    _error_record(language, sentence.sentence_id, code, message)
                )
                sentences_failed += 1
            sentence_reports.append(report_item)

        track_pcm = float_to_pcm16(voice_track)
        language_pcm_tracks.append(track_pcm)
        language_reports.append(
            {
                "language": language,
                "source_path": str(audio_path),
                "source_sha256": sha256_file(audio_path),
                "source_sample_rate": audio.sample_rate,
                "source_channels": audio.channels,
                "source_sample_count": audio.sample_count,
                "source_content_sha256": sha256_int16(source_pcm),
                "split": {
                    "settings": split_settings.as_dict(),
                    "active_run_count": split.active_run_count,
                    "silence_run_count": split.silence_run_count,
                    "detected_runs": [
                        {"start_sample": start, "end_sample": end}
                        for start, end in split.runs
                    ],
                },
                "cache_hits": language_hits,
                "cache_recomputes": language_recomputes,
                "track_sha256": sha256_int16(track_pcm),
                "sentences": sentence_reports,
            }
        )

    background_pcm, background_info = _load_background(
        options.background_path, timeline
    )
    if len(background_pcm) != timeline.duration_samples:
        raise LipAlignError(
            "E_BACKGROUND_LENGTH",
            "Background track length is not the timeline length",
            exit_code=2,
        )

    channel_names = ["language:{}".format(pair[0]) for pair in normalized_inputs]
    channel_names.append("background:pink_noise")
    stacked = np.stack(language_pcm_tracks + [background_pcm], axis=1).astype(
        np.int16, copy=False
    )
    output_path = Path(options.output_path)
    try:
        write_pcm16_wav(
            output_path,
            stacked,
            timeline.sample_rate,
            channels=len(channel_names),
        )
    except OSError as exc:
        raise LipAlignError(
            "E_OUTPUT", "Cannot write multitrack WAV: {}".format(exc), exit_code=2
        ) from exc

    written = read_wav(output_path)
    if (
        written.sample_rate != timeline.sample_rate
        or written.channels != len(channel_names)
        or written.sample_count != timeline.duration_samples
    ):
        raise LipAlignError(
            "E_OUTPUT_VERIFY",
            "Written multitrack WAV failed container verification",
            exit_code=10,
        )

    channel_hashes = [
        sha256_int16(stacked[:, index]) for index in range(stacked.shape[1])
    ]
    elapsed = time.perf_counter() - started
    partial = sentences_failed > 0
    considered = len(normalized_inputs) * len(timeline.sentences)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "lipalign-alignment-report",
        "status": "partial" if partial else "ok",
        "timeline": {
            "path": str(timeline.source_path),
            "sha256": timeline_sha256,
            "seed": timeline.seed,
            "sample_rate": timeline.sample_rate,
            "duration_samples": timeline.duration_samples,
            "duration_sec": _rounded(timeline.duration_sec),
            "sentence_count": len(timeline.sentences),
            "rounding": "nearest integer sample, ties upward",
        },
        "settings": {
            "algorithm": stretch_settings.algorithm,
            "stretch": stretch_settings.as_dict(),
            "crossfade_ms": options.crossfade_ms,
            "crossfade_method": (
                "10 ms equal-power crossfade at each fragment head/tail "
                "(with the previous fragment tail or timeline silence)"
            ),
            "cache_dir": str(Path(options.cache_dir).resolve()),
        },
        "languages": language_reports,
        "background": background_info,
        "output": {
            "path": str(output_path),
            "format": "WAV PCM 16-bit",
            "sample_rate": timeline.sample_rate,
            "sample_count": timeline.duration_samples,
            "duration_sec": _rounded(timeline.duration_sec),
            "channel_count": len(channel_names),
            "channels": channel_names,
            "channel_sha256": channel_hashes,
            "file_sha256": sha256_file(output_path),
            "all_channels_same_sample_count": True,
            "duration_error_samples": written.sample_count - timeline.duration_samples,
            "duration_error_sec": 0.0,
            "duration_error_limit_sec": 0.005,
        },
        "totals": {
            "sentences_considered": considered,
            "sentences_processed": sentences_processed,
            "sentences_failed": sentences_failed,
            "cache_hits": cache_hits,
            "cache_recomputes": cache_recomputes,
            "elapsed_sec": _rounded(elapsed, 6),
        },
        "errors": errors,
    }
    try:
        atomic_write_json(options.report_path, report)
    except OSError as exc:
        raise LipAlignError(
            "E_OUTPUT", "Cannot write alignment report: {}".format(exc), exit_code=2
        ) from exc
    stats = report["totals"]
    return AlignResult(report=report, partial=partial, stats=dict(stats))

