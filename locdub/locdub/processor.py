"""Per-language processing pipeline and multi-language orchestration."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf

from . import audioio, ffmpeg_utils, mixer
from .constants import (
    DEFAULT_SAMPLE_RATE,
    SEGMENT_OK,
    SEGMENT_SKIPPED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    STATUS_SUCCESS,
)
from .logging_utils import RunLogger
from .model import LanguageResult, Segment, SegmentResult


class ProcessConfig:
    def __init__(self, *, video: Path, output_dir: Path,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 channels: int = 2, gain_db: float = 0.0,
                 max_segment_seconds: float = 600.0,
                 block_seconds: float = 60.0) -> None:
        self.video = video
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.channels = channels
        self.gain_db = gain_db
        self.max_block_frames = int(sample_rate * max_segment_seconds)
        self.block_seconds = block_seconds


def process_language(language: str, dub_dir: Path,
                     segments: List[Segment], skipped_ids: Dict[str, str],
                     config: ProcessConfig,
                     logger: RunLogger) -> LanguageResult:
    """Adapt every segment for one language and produce its MP4."""
    lang_dir = config.output_dir / language
    adapted_dir = lang_dir / "adapted"
    adapted_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[{language}] starting ({len(segments)} segments)")

    results: List[SegmentResult] = []
    jobs: List[mixer.MixJob] = []
    ok_count = 0

    for segment in segments:
        result = SegmentResult(segment_id=segment.segment_id, status=SEGMENT_OK)
        target_wav = adapted_dir / f"{segment.segment_id}.wav"
        source_wav = dub_dir / segment.audio_path

        if segment.segment_id in skipped_ids:
            _mark_skipped(result, jobs, segment, target_wav, logger, language,
                          reason=skipped_ids[segment.segment_id])
            results.append(result)
            continue

        try:
            adaptation = audioio.adapt_segment(
                source_wav,
                segment.duration_ms,
                segment.min_ratio,
                segment.max_ratio,
                config.max_block_frames,
            )
        except audioio.AudioError as exc:
            _mark_skipped(result, jobs, segment, target_wav, logger, language,
                          reason=str(exc))
            results.append(result)
            continue

        raw_wav = adapted_dir / f"{segment.segment_id}.raw.wav"
        _write_adapted(raw_wav, adaptation.audio, adaptation.sample_rate)
        # Normalise sample rate/channel count so the mixer can place clips on
        # a single sample-accurate timeline.
        ffmpeg_utils.normalise_wav(
            raw_wav, target_wav, config.sample_rate, config.channels
        )
        raw_wav.unlink(missing_ok=True)
        result.original_duration_ms = adaptation.original_duration_ms
        result.adapted_duration_ms = adaptation.adapted_duration_ms
        result.requested_ratio = adaptation.requested_ratio
        result.applied_ratio = adaptation.applied_ratio
        result.boundary_clamped = adaptation.boundary_clamped
        if adaptation.boundary_clamped:
            warning = logger.warning(
                "requested ratio outside allowed range; clamped to boundary",
                language=language, segment_id=segment.segment_id,
                requested_ratio=round(adaptation.requested_ratio, 4),
                applied_ratio=round(adaptation.applied_ratio, 4),
                min_ratio=segment.min_ratio, max_ratio=segment.max_ratio,
            )
            result.warnings.append(_brief(warning))

        jobs.append(mixer.MixJob(
            segment_id=segment.segment_id,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            wav_path=target_wav,
        ))
        result.status = SEGMENT_OK
        ok_count += 1
        results.append(result)
        logger.info(
            f"[{language}] {segment.segment_id} ratio="
            f"{adaptation.applied_ratio:.3f}{' clamped' if adaptation.boundary_clamped else ''}"
        )

    output_file: Optional[Path] = None
    status = STATUS_SUCCESS
    try:
        output_file = _assemble(language, jobs, config, logger)
    except (ffmpeg_utils.FFmpegError, audioio.AudioError, OSError) as exc:
        logger.error("language assembly failed", language=language, error=str(exc))
        return LanguageResult(
            language=language, status=STATUS_FAILED, output_file=None,
            segments=results, total_segments=len(segments),
            ok_segments=ok_count, skipped_segments=len(segments) - ok_count,
        )

    if ok_count < len(segments):
        status = STATUS_PARTIAL
    return LanguageResult(
        language=language,
        status=status,
        output_file=str(output_file),
        segments=results,
        total_segments=len(segments),
        ok_segments=ok_count,
        skipped_segments=len(segments) - ok_count,
    )


def run_all(languages: Dict[str, Path], segments: List[Segment],
            skipped_ids: Dict[str, str], config: ProcessConfig,
            logger: RunLogger, workers: int) -> List[LanguageResult]:
    """Process languages in parallel, bounded by ``workers``."""
    workers = max(1, min(workers, len(languages)))
    if workers == 1:
        return [
            process_language(lang, path, segments, skipped_ids, config, logger)
            for lang, path in languages.items()
        ]
    outcomes: List[LanguageResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_language, lang, path, segments,
                        skipped_ids, config, logger): lang
            for lang, path in languages.items()
        }
        for future in as_completed(futures):
            outcomes.append(future.result())
    order = {lang: idx for idx, lang in enumerate(languages)}
    outcomes.sort(key=lambda item: order[item.language])
    return outcomes


def _assemble(language: str, jobs: List[mixer.MixJob],
              config: ProcessConfig, logger: RunLogger) -> Path:
    lang_dir = config.output_dir / language
    with tempfile.TemporaryDirectory(prefix=f"locdub_{language}_",
                                     dir=lang_dir) as tmp_name:
        tmp_dir = Path(tmp_name)
        background = tmp_dir / "background.wav"
        mixdown = tmp_dir / "mixdown.wav"
        ffmpeg_utils.extract_background(
            config.video, background,
            sample_rate=config.sample_rate, channels=config.channels,
        )
        mixer.mix_stream(
            background, jobs, mixdown,
            sample_rate=config.sample_rate,
            channels=config.channels,
            gain_db=config.gain_db,
            block_seconds=config.block_seconds,
        )
        duration_ms = ffmpeg_utils.wav_duration_ms(mixdown)
        out_mp4 = lang_dir / f"{language}.mp4"
        ffmpeg_utils.mux_mixdown(config.video, mixdown, out_mp4, duration_ms)
        logger.info(f"[{language}] wrote {out_mp4}")
        return out_mp4


def _write_adapted(path: Path, audio, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


def _mark_skipped(result: SegmentResult, jobs: List[mixer.MixJob],
                  segment: Segment, target_wav: Path,
                  logger: RunLogger, language: str, *, reason: str) -> None:
    result.status = SEGMENT_SKIPPED
    result.silence_placeholder = True
    error = logger.error("segment skipped; silence placeholder retained",
                         language=language, segment_id=segment.segment_id,
                         reason=reason)
    result.errors.append(_brief(error))
    jobs.append(mixer.MixJob(
        segment_id=segment.segment_id,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        wav_path=None,
    ))


def _brief(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items()
            if key in ("level", "message", "reason", "requested_ratio",
                       "applied_ratio", "min_ratio", "max_ratio")}
