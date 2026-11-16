"""End-to-end per-language lip-sync processing pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import align as align_mod
from . import mux as mux_mod
from . import wavio
from .config import Config
from .wsola import clamp_stretch_factor, wsola_time_stretch


@dataclass
class LanguageResult:
    language: str
    status: str = "ok"
    stretch_factor: Optional[float] = None
    requested_stretch_factor: Optional[float] = None
    event_count: int = 0
    mean_offset_ms: float = 0.0
    max_offset_ms: float = 0.0
    coverage_ratio: float = 0.0
    silence_segments: List[Dict[str, Any]] = field(default_factory=list)
    uncovered_segments: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    outputs: Dict[str, str] = field(default_factory=dict)

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "status": self.status,
            "stretch_factor": round(self.stretch_factor, 6) if self.stretch_factor is not None else None,
            "requested_stretch_factor": (
                round(self.requested_stretch_factor, 6)
                if self.requested_stretch_factor is not None
                else None
            ),
            "event_count": self.event_count,
            "alignment_stats": {
                "mean_offset_ms": round(self.mean_offset_ms, 4),
                "max_offset_ms": round(self.max_offset_ms, 4),
                "event_count": self.event_count,
            },
            "coverage_ratio": round(self.coverage_ratio, 4),
            "silence_segments": self.silence_segments,
            "uncovered_segments": self.uncovered_segments,
            "warnings": self.warnings,
            "errors": self.errors,
            "notes": self.notes,
            "outputs": self.outputs,
        }


def _find_wav(audio_dir: Path, language: str) -> Optional[Path]:
    candidates = [
        audio_dir / f"dub_{language}.wav",
        audio_dir / f"{language}.wav",
        audio_dir / language / "dub.wav",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(audio_dir.glob(f"*{language}*.wav"))
    return matches[0] if matches else None


def process_language(
    language: str,
    metadata: Dict[str, Any],
    audio_dir: Path,
    output_dir: Path,
    video_path: Path,
    config: Config,
) -> LanguageResult:
    """Process a single dub language; never raises for expected failures."""
    result = LanguageResult(language=language)
    frame_rate = float(metadata["frame_rate"])
    target_rate = int(metadata.get("target_sample_rate", 32000))
    animation_duration = float(metadata["duration_seconds"])

    wav_path = _find_wav(audio_dir, language)
    if wav_path is None:
        result.status = "failed"
        result.errors.append(f"no WAV dub found for language '{language}' in {audio_dir}")
        return result

    try:
        wav = wavio.read_wav(wav_path)
        mono_signal, _, normalize_notes = wavio.normalize_to_mono(wav, target_rate)
        result.notes.extend(normalize_notes)
        original_sample_width = wav.sample_width
    except Exception as exc:  # corrupted/unreadable audio
        result.status = "failed"
        result.errors.append(f"failed to read/normalize {wav_path.name}: {exc}")
        return result

    # Stage 1: map frame-numbered lip events onto the dub audio timeline.
    mapped_events = align_mod.map_events_to_audio_time(metadata["lip_events"], frame_rate)

    # Stage 2: WSOLA time scaling to match the original animation duration.
    input_duration = mono_signal.shape[0] / float(target_rate)
    requested_factor = animation_duration / input_duration if input_duration > 0 else 1.0
    result.requested_stretch_factor = requested_factor
    factor, clamp_warning = clamp_stretch_factor(
        requested_factor, config.stretch_min, config.stretch_max
    )
    result.stretch_factor = factor
    if clamp_warning:
        result.warnings.append(clamp_warning)

    stretched = wsola_time_stretch(mono_signal, factor, target_rate)
    stretched_duration = stretched.shape[0] / float(target_rate)

    # Stage 3: energy search + dynamic programming alignment (max 50 ms shift).
    aligned_events, stats = align_mod.align_events(
        mapped_events,
        stretched,
        target_rate,
        tolerance_ms=config.align_tolerance_ms,
        overlap_penalty_weight=config.overlap_penalty_weight,
    )
    result.event_count = len(aligned_events)
    result.mean_offset_ms = stats["mean_offset_ms"]
    result.max_offset_ms = stats["max_offset_ms"]

    # Stage 4: silence and coverage diagnostics, never interrupting processing.
    silence = align_mod.detect_silence(
        stretched,
        target_rate,
        threshold_db=config.silence_threshold_db,
        min_silence_ms=config.silence_min_ms,
    )
    result.silence_segments = [segment.to_dict() for segment in silence]
    if silence:
        result.warnings.append(
            f"detected {len(silence)} silence segment(s) longer than "
            f"{config.silence_min_ms:.0f} ms; silence markers added to alignment JSON"
        )

    coverage = align_mod.coverage_ratio(aligned_events, stretched_duration)
    result.coverage_ratio = coverage
    result.uncovered_segments = []
    if coverage < config.coverage_min_ratio:
        message = (
            f"lip event coverage {coverage:.2%} below minimum "
            f"{config.coverage_min_ratio:.2%}; uncovered-gap markers added"
        )
        result.warnings.append(message)
        result.uncovered_segments = align_mod.uncovered_segments(
            aligned_events, stretched_duration
        )

    # Stage 5: always persist the stretched WAV and aligned lip JSON.
    output_dir.mkdir(parents=True, exist_ok=True)
    stretched_wav = output_dir / f"dub_{language}_stretched.wav"
    lip_json = output_dir / f"lipsync_{language}.json"
    wavio.write_wav(stretched_wav, stretched, target_rate, original_sample_width)

    lip_payload = {
        "language": language,
        "source_wav": str(wav_path.name),
        "sample_rate": target_rate,
        "sample_width_bytes": original_sample_width,
        "frame_rate": frame_rate,
        "stretch_factor": factor,
        "requested_stretch_factor": requested_factor,
        "audio_duration_seconds": round(stretched_duration, 4),
        "event_count": len(aligned_events),
        "lip_events": aligned_events,
        "silence_markers": result.silence_segments,
        "uncovered_gap_markers": result.uncovered_segments,
    }
    lip_json.write_text(json.dumps(lip_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    result.outputs["stretched_wav"] = str(stretched_wav)
    result.outputs["lip_json"] = str(lip_json)

    # Stage 6: MP4 remux with JSON/WAV fallback on failure.
    mp4_path = output_dir / f"cutscene_{language}.mp4"
    try:
        mux_result = mux_mod.remux_mp4(
            video_path, stretched_wav, mp4_path, simulate_failure=config.simulate_mux_failure
        )
        result.outputs["mp4"] = str(mux_result.output_path)
        result.notes.append(
            f"mp4 remuxed: video={mux_result.video_codec} copied, "
            f"audio={mux_result.audio_codec}, duration={mux_result.duration_seconds:.3f}s"
        )
    except mux_mod.MuxError as exc:
        result.status = "mux_fallback"
        result.errors.append(f"MP4 remux failed; fallback to WAV + JSON: {exc}")

    return result


def run_pipeline(
    metadata_path: Path,
    audio_dir: Path,
    output_dir: Path,
    languages: List[str],
    config: Config,
    config_notes: Optional[List[str]] = None,
    video_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Process every requested language and write the consolidated report."""
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if video_path is None:
        video_path = output_dir.parent / "video" / "cutscene_silent.mp4"

    language_results: List[LanguageResult] = []
    for language in languages:
        language_results.append(
            process_language(
                language=language,
                metadata=metadata,
                audio_dir=Path(audio_dir),
                output_dir=output_dir,
                video_path=Path(video_path),
                config=config,
            )
        )

    failed_segments = [
        result.language for result in language_results if result.status == "failed"
    ]
    fallback_segments = [
        result.language for result in language_results if result.status == "mux_fallback"
    ]
    report = {
        "tool": "lipsync_tool",
        "metadata_path": str(metadata_path),
        "audio_dir": str(audio_dir),
        "video_path": str(video_path),
        "languages": languages,
        "config": config.to_dict(),
        "config_notes": config_notes or [],
        "failed_segments": failed_segments,
        "mux_fallback_segments": fallback_segments,
        "language_reports": [result.to_report_dict() for result in language_results],
    }
    report_path = output_dir / "processing_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
