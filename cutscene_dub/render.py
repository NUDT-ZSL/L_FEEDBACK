from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .adapt import DialogueAdaptation, adapt_dialogues
from .audio import (
    SAMPLE_RATE,
    float_to_pcm16,
    load_mono_wav,
    peak_dbfs,
    rms_dbfs,
)
from .config import RenderConfig
from .ducking import apply_ducking
from .errors import BusinessError
from .io_utils import (
    AtomicWavWriter,
    maybe_test_interrupt,
    sha256_file,
    write_json_atomic,
)
from .model import Timeline, canonical_json_bytes
from .samples import bgm_sample_path, dialogue_sample_path
from .state import RenderState, render_state_path


_CENTER_GAIN = 1.0 / math.sqrt(2.0)


def _safe_db(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return float(value)


def _input_sha(path: Path, *, entry_id: str | None = None) -> str:
    if not path.is_file():
        raise BusinessError(
            code="E_AUDIO_READ",
            message="Required WAV input file does not exist.",
            entry_id=entry_id,
            path=str(path),
        )
    try:
        return sha256_file(path)
    except BusinessError as exc:
        exc.entry_id = entry_id
        raise


def resolve_input_paths(
    timeline: Timeline,
    *,
    audio_root: str | Path | None,
    bgm_path: str | Path | None,
) -> tuple[Path, Path, list[tuple[str, Path]]]:
    root = (
        Path(audio_root)
        if audio_root is not None
        else timeline.source_path.parent / "samples"
    )
    bgm = Path(bgm_path) if bgm_path is not None else bgm_sample_path(root)
    dialogue_paths = [
        (dialogue.id, dialogue_sample_path(root, dialogue.target_language, dialogue.id))
        for dialogue in timeline.dialogues
    ]
    return root, bgm, dialogue_paths


def _render_fingerprint(
    timeline: Timeline,
    *,
    bgm_path: Path,
    bgm_sha256: str,
    dialogue_paths: list[tuple[str, Path]],
    dialogue_hashes: dict[str, str],
    config: RenderConfig,
) -> str:
    value = {
        "tool_version": __version__,
        "render_schema_version": "1.0",
        "timeline_sha256": timeline.source_sha256,
        "target_language": timeline.target_language,
        "bgm": {"path": str(bgm_path.resolve()), "sha256": bgm_sha256},
        "dialogues": [
            {
                "id": dialogue_id,
                "path": str(path.resolve()),
                "sha256": dialogue_hashes[dialogue_id],
            }
            for dialogue_id, path in dialogue_paths
        ],
        "config": config.to_dict(),
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _timeline_report(timeline: Timeline, *, total_frames: int) -> dict[str, Any]:
    return {
        "project_id": timeline.project_id,
        "target_language": timeline.target_language,
        "duration_s": timeline.duration_s,
        "duration_frames": total_frames,
        "shots": [
            {
                "id": shot.id,
                "start_s": shot.start_s,
                "end_s": shot.end_s,
                "events": [
                    {
                        "id": event.id,
                        "type": event.type,
                        "at_s": event.at_s,
                        "end_s": event.end_s,
                    }
                    for event in shot.events
                ],
                "subtitles": [
                    {
                        "id": subtitle.id,
                        "start_s": subtitle.start_s,
                        "end_s": subtitle.end_s,
                        "text": subtitle.text,
                    }
                    for subtitle in shot.subtitles
                ],
                "dialogue_ids": [dialogue.id for dialogue in shot.dialogues],
            }
            for shot in timeline.shots
        ],
    }


def _ducking_report(
    *,
    config: RenderConfig,
    intervals: list[dict[str, Any]] | None,
    max_jump_db: float | None,
    measured_attenuation_db: float | None,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "depth_db": -config.duck_depth_db,
        "depth_positive_db": config.duck_depth_db,
        "fade_in_ms": config.duck_fade_in_ms,
        "fade_out_ms": config.duck_fade_out_ms,
        "max_gain_jump_db": config.max_gain_jump_db,
        "max_observed_gain_jump_db": max_jump_db,
        "measured_dialogue_attenuation_db": measured_attenuation_db,
        "outside_gain_db": 0.0,
        "merge_policy": (
            "Adjacent dialogue intervals with a gap shorter than fade_out+fade_in "
            "are merged into one hold-down ducking interval."
        ),
        "intervals": intervals or [],
    }


def _failure_report(
    timeline: Timeline,
    *,
    analyses: list[DialogueAdaptation],
    config: RenderConfig,
    bgm_path: Path,
    bgm_sha256: str,
    total_frames: int,
    error: BusinessError,
) -> dict[str, Any]:
    report = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "status": "failed",
        "source": {
            "timeline": str(timeline.source_path),
            "timeline_sha256": timeline.source_sha256,
            "bgm": str(bgm_path),
            "bgm_sha256": bgm_sha256,
            "sample_rate_hz": SAMPLE_RATE,
            "bit_depth": 16,
            "output_channels": 2,
        },
        "timeline": _timeline_report(timeline, total_frames=total_frames),
        "config": config.to_dict(),
        "alignment": {
            "dialogues": [analysis.as_report() for analysis in analyses],
            "unadaptable_count": sum(1 for item in analyses if item.unadaptable),
        },
        "ducking": _ducking_report(
            config=config,
            intervals=[],
            max_jump_db=None,
            measured_attenuation_db=None,
            status="not_rendered",
        ),
        "music": {
            "path": str(bgm_path),
            "sha256": bgm_sha256,
            "status": "not_rendered",
        },
        "mix": None,
        "output": None,
        "errors": [error.as_dict()],
    }
    return report


def _build_mix(
    analyses: list[DialogueAdaptation],
    *,
    bgm: np.ndarray,
    total_frames: int,
    config: RenderConfig,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
]:
    dialogue_mono = np.zeros(total_frames, dtype=np.float64)
    active = np.zeros(total_frames, dtype=bool)
    intervals: list[tuple[int, int, str]] = []
    dialogue_gain = 10.0 ** (config.dialogue_gain_db / 20.0)
    for analysis in analyses:
        if analysis.final_samples is None or analysis.final_frames <= 0:
            continue
        start = analysis.final_start_frame
        end = start + analysis.final_frames
        if start < 0 or end > total_frames:
            raise BusinessError(
                code="E_INTERNAL",
                message="Adapted dialogue falls outside the timeline.",
                entry_id=analysis.dialogue.id,
            )
        dialogue_mono[start:end] += analysis.final_samples * dialogue_gain
        active[start:end] = True
        intervals.append((start, end, analysis.dialogue.id))

    ducking = apply_ducking(bgm, dialogue_intervals=intervals, config=config)
    music_gain = 10.0 ** (config.music_gain_db / 20.0)
    ducked_music = bgm * ducking.gain_envelope * music_gain

    mixed_mono = dialogue_mono + ducked_music
    stereo = np.empty((total_frames, 2), dtype=np.float64)
    stereo[:, 0] = mixed_mono * _CENTER_GAIN
    stereo[:, 1] = mixed_mono * _CENTER_GAIN
    pre_peak = peak_dbfs(stereo)
    ceiling = config.peak_ceiling_linear
    pre_peak_linear = float(np.max(np.abs(stereo))) if stereo.size else 0.0
    if pre_peak_linear > ceiling and pre_peak_linear > 0:
        attenuation_linear = ceiling / pre_peak_linear
        global_attenuation_db = 20.0 * math.log10(attenuation_linear)
    else:
        attenuation_linear = 1.0
        global_attenuation_db = 0.0
    stereo *= attenuation_linear

    if active.any():
        dialogue_rms = rms_dbfs(dialogue_mono[active])
        music_rms = rms_dbfs(ducked_music[active])
        ratio = (
            dialogue_rms - music_rms
            if math.isfinite(dialogue_rms) and math.isfinite(music_rms)
            else None
        )
    else:
        dialogue_rms = None
        music_rms = None
        ratio = None

    pcm = float_to_pcm16(stereo)
    quantized_peak_linear = (
        float(np.max(np.abs(pcm.astype(np.float64)))) / 32768.0
        if pcm.size
        else 0.0
    )
    quantized_peak = (
        20.0 * math.log10(quantized_peak_linear)
        if quantized_peak_linear > 0
        else None
    )
    if quantized_peak is not None and quantized_peak > config.peak_ceiling_dbfs + 1e-6:
        raise BusinessError(
            code="E_INTERNAL",
            message="Peak ceiling was not enforced after PCM quantization.",
            details={
                "peak_dbfs": quantized_peak,
                "ceiling_dbfs": config.peak_ceiling_dbfs,
            },
        )

    mix_report = {
        "pan": {
            "dialogue": {"left": _CENTER_GAIN, "right": _CENTER_GAIN},
            "music": {"left": _CENTER_GAIN, "right": _CENTER_GAIN},
            "rule": "constant-power center (1/sqrt(2) to each channel)",
        },
        "gains": {
            "dialogue_db": config.dialogue_gain_db,
            "music_db": config.music_gain_db,
        },
        "peak_before_limit_dbfs": _safe_db(pre_peak),
        "peak_ceiling_dbfs": config.peak_ceiling_dbfs,
        "global_attenuation_db": global_attenuation_db,
        "peak_after_limit_dbfs": _safe_db(
            20.0 * math.log10(pre_peak_linear * attenuation_linear)
            if pre_peak_linear > 0
            else None
        ),
        "quantized_peak_dbfs": _safe_db(quantized_peak),
        "clipping_detected": False,
        "dialogue_rms_dbfs": _safe_db(dialogue_rms),
        "music_rms_during_dialogue_dbfs": _safe_db(music_rms),
        "dialogue_to_music_ratio_db": _safe_db(ratio),
    }
    duck_info = {
        "intervals": ducking.intervals_report(),
        "max_jump_db": ducking.max_gain_jump_db,
        "measured_attenuation_db": ducking.measured_dialogue_attenuation_db,
    }
    return pcm, mix_report, duck_info, ducking.gain_envelope


def _write_output(
    output_path: Path,
    pcm: np.ndarray,
    *,
    total_frames: int,
) -> None:
    writer = AtomicWavWriter(
        output_path,
        sample_rate=SAMPLE_RATE,
        channels=2,
        failpoint="render_audio_partial",
    )
    try:
        writer.write_pcm(pcm)
        writer.commit()
    finally:
        if not writer._committed:
            writer.abort(keep_partial=True)
    if pcm.shape[0] != total_frames:
        raise BusinessError(
            code="E_INTERNAL",
            message="Internal output frame count mismatch.",
            details={"expected_frames": total_frames, "actual": int(pcm.shape[0])},
        )


def render_project(
    timeline: Timeline,
    *,
    audio_root: str | Path,
    bgm_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    config: RenderConfig,
    force: bool = False,
) -> dict[str, Any]:
    config.validate()
    root = Path(audio_root)
    bgm = Path(bgm_path)
    output = Path(output_path)
    report_path_obj = Path(report_path)
    if output.resolve() == report_path_obj.resolve():
        raise BusinessError(
            code="E_CONFIG",
            message="Output WAV and report JSON must be different paths.",
        )
    if output.suffix.lower() != ".wav":
        raise BusinessError(
            code="E_CONFIG",
            message="Output audio path must use the .wav extension.",
            path=str(output),
        )
    if report_path_obj.suffix.lower() != ".json":
        raise BusinessError(
            code="E_CONFIG",
            message="Report path must use the .json extension.",
            path=str(report_path_obj),
        )
    total_frames = int(round(timeline.duration_s * SAMPLE_RATE))
    if total_frames <= 0:
        raise BusinessError(
            code="E_TIMELINE_RANGE",
            message="Timeline duration is shorter than one output sample frame.",
        )

    _, _, dialogue_paths = resolve_input_paths(
        timeline, audio_root=root, bgm_path=bgm
    )
    input_paths = [p for _, p in dialogue_paths] + [bgm, timeline.source_path]
    for protected in input_paths:
        if output.resolve() == protected.resolve() or report_path_obj.resolve() == protected.resolve():
            raise BusinessError(
                code="E_CONFIG",
                message="Output paths must not overwrite an input file.",
                path=str(protected),
            )

    bgm_sha = _input_sha(bgm)
    dialogue_hashes: dict[str, str] = {}
    for dialogue_id, path in dialogue_paths:
        dialogue_hashes[dialogue_id] = _input_sha(path, entry_id=dialogue_id)
    fingerprint = _render_fingerprint(
        timeline,
        bgm_path=bgm,
        bgm_sha256=bgm_sha,
        dialogue_paths=dialogue_paths,
        dialogue_hashes=dialogue_hashes,
        config=config,
    )

    state = RenderState(render_state_path(output))
    if not force and state.is_reusable(
        fingerprint=fingerprint,
        output_path=output,
        report_path=report_path_obj,
    ):
        summary = state.summary
        return {
            "ok": True,
            "command": "render",
            "tool_version": __version__,
            "reused": True,
            "audio": str(output),
            "report": str(report_path_obj),
            "audio_sha256": summary.get("audio_sha256"),
            "frame_count": summary.get("frame_count", total_frames),
            "duration_s": total_frames / float(SAMPLE_RATE),
        }

    state.begin(fingerprint=fingerprint, output_path=output, report_path=report_path_obj)
    try:
        analyses = adapt_dialogues(timeline, audio_root=root, config=config)
        unadaptable = [item for item in analyses if item.unadaptable]
        if unadaptable:
            first = unadaptable[0]
            error = BusinessError(
                code="E_DIALOGUE_UNFIT",
                message=(
                    f"{len(unadaptable)} dialogue(s) cannot fit within their "
                    "subtitle windows under the configured speed limit."
                ),
                entry_id=first.dialogue.id,
                details={
                    "entry_ids": [item.dialogue.id for item in unadaptable],
                    "report_path": str(report_path_obj),
                },
            )
            failure_report = _failure_report(
                timeline,
                analyses=analyses,
                config=config,
                bgm_path=bgm,
                bgm_sha256=bgm_sha,
                total_frames=total_frames,
                error=error,
            )
            write_json_atomic(report_path_obj, failure_report)
            report_sha = sha256_file(report_path_obj)
            state.report_published(report_sha)
            state.failed(error, report_published=True)
            raise error

        music_input = load_mono_wav(bgm)
        if music_input.frames >= total_frames:
            music = music_input.samples[:total_frames].copy()
            padded_frames = 0
            truncated_frames = music_input.frames - total_frames
        else:
            music = np.pad(music_input.samples, (0, total_frames - music_input.frames))
            padded_frames = total_frames - music_input.frames
            truncated_frames = 0

        pcm, mix_report, duck_info, _ = _build_mix(
            analyses,
            bgm=music,
            total_frames=total_frames,
            config=config,
        )
        _write_output(output, pcm, total_frames=total_frames)
        audio_sha = sha256_file(output)
        state.audio_published(audio_sha, total_frames)
        maybe_test_interrupt("render_after_audio_commit")

        output_info = {
            "audio": str(output),
            "audio_sha256": audio_sha,
            "report": str(report_path_obj),
            "channels": 2,
            "interleaved": True,
            "sample_rate_hz": SAMPLE_RATE,
            "bit_depth": 16,
            "frame_count": total_frames,
            "duration_s": total_frames / float(SAMPLE_RATE),
            "timeline_duration_error_frames": 0,
        }
        music_report = {
            "path": str(bgm),
            "sha256": bgm_sha,
            "original_frames": music_input.frames,
            "original_duration_s": music_input.frames / float(SAMPLE_RATE),
            "used_frames": total_frames,
            "padded_frames": padded_frames,
            "truncated_frames": truncated_frames,
        }
        final_report = {
            "schema_version": "1.0",
            "tool_version": __version__,
            "status": "ok",
            "source": {
                "timeline": str(timeline.source_path),
                "timeline_sha256": timeline.source_sha256,
                "bgm": str(bgm),
                "bgm_sha256": bgm_sha,
                "sample_rate_hz": SAMPLE_RATE,
                "bit_depth": 16,
                "output_channels": 2,
            },
            "timeline": _timeline_report(timeline, total_frames=total_frames),
            "config": config.to_dict(),
            "alignment": {
                "dialogues": [analysis.as_report() for analysis in analyses],
                "unadaptable_count": 0,
            },
            "music": music_report,
            "ducking": _ducking_report(
                config=config,
                intervals=duck_info["intervals"],
                max_jump_db=duck_info["max_jump_db"],
                measured_attenuation_db=duck_info["measured_attenuation_db"],
                status="applied" if duck_info["intervals"] else "not_needed",
            ),
            "mix": mix_report,
            "output": output_info,
            "errors": [],
        }
        write_json_atomic(report_path_obj, final_report)
        report_sha = sha256_file(report_path_obj)
        state.report_published(report_sha)
        maybe_test_interrupt("render_after_report_commit")
        state.complete(
            frame_count=total_frames,
            audio_sha256=audio_sha,
            report_sha256=report_sha,
        )
        return {
            "ok": True,
            "command": "render",
            "tool_version": __version__,
            "reused": False,
            "audio": str(output),
            "report": str(report_path_obj),
            "audio_sha256": audio_sha,
            "report_sha256": report_sha,
            "frame_count": total_frames,
            "duration_s": total_frames / float(SAMPLE_RATE),
        }
    except KeyboardInterrupt:
        state.interrupted("render")
        raise
    except BusinessError as exc:
        if exc.code != "E_DIALOGUE_UNFIT":
            state.failed(exc)
        raise
    except Exception as exc:
        wrapped = BusinessError(
            code="E_INTERNAL",
            message=f"Unexpected render failure: {exc}",
        )
        state.failed(wrapped)
        raise wrapped from exc

