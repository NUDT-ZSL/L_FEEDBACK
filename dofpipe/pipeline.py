from __future__ import annotations

import dataclasses
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig, format_frame_name
from .depth import (
    load_color,
    load_depth_raw,
    parse_depth,
    preflight_sequence,
)
from .errors import ConfigurationError, FocusTrackError, InputDataError, PipelineError
from .focus import FocusTrack, load_focus_track
from .render import DOFRenderer, image_difference_8bit
from .report import build_report, frame_report_entry, write_json_atomic

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class RenderSummary:
    report: dict[str, Any]
    report_path: Path
    output_files: list[Path]


def _noop_progress(_: str) -> None:
    return


def _resolve_focus_track(config: PipelineConfig, input_dir: Path) -> Path:
    candidate = Path(config.focus_track)
    if candidate.is_absolute():
        return candidate
    input_relative = input_dir / candidate
    if input_relative.is_file():
        return input_relative
    if config.source_path is not None:
        config_relative = config.source_path.parent / candidate
        if config_relative.is_file():
            return config_relative
    return input_relative


def _write_png_atomic(path: Path, image: np.ndarray, compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.png")
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, int(compression)]
    success = cv2.imwrite(str(temporary), image, parameters)
    if not success:
        raise InputDataError(f"failed to encode PNG output {path}")
    os.replace(temporary, path)


def _error_payload(exc: Exception, frame_index: int | None) -> dict[str, Any]:
    reason = getattr(exc, "reason", None)
    if reason is None:
        reason = str(exc)
    return {
        "type": type(exc).__name__,
        "frame_index": None if frame_index is None else int(frame_index),
        "reason": str(reason),
        "message": str(exc),
    }


def _write_failure_reports(
    output_dir: Path,
    config: PipelineConfig,
    width: int,
    height: int,
    frames: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    exc: Exception,
    frame_index: int | None,
) -> None:
    report = build_report(
        config=config,
        width=width,
        height=height,
        frames=frames,
        warnings=warnings,
        status="failed",
        error=_error_payload(exc, frame_index),
    )
    # Keep both names: report.partial.json is the crash-safe working copy and
    # report.json is the conventional artifact a caller will look for.
    write_json_atomic(output_dir / "report.partial.json", report)
    write_json_atomic(output_dir / "report.json", report)


def run_pipeline(
    config: PipelineConfig,
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    depth_slices: int | None = None,
    gather_rounds: int | None = None,
    progress: ProgressCallback | None = None,
) -> RenderSummary:
    progress = progress or _noop_progress
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").unlink(missing_ok=True)
    (output_dir / "report.partial.json").unlink(missing_ok=True)

    if depth_slices is not None or gather_rounds is not None:
        render_config = dataclasses.replace(
            config.render,
            depth_slices=config.render.depth_slices
            if depth_slices is None
            else int(depth_slices),
            gather_rounds=config.render.gather_rounds
            if gather_rounds is None
            else int(gather_rounds),
        )
        config = dataclasses.replace(config, render=render_config)
    if config.render.depth_slices < 1:
        raise ConfigurationError("depth_slices must be >= 1")
    if config.render.gather_rounds < 1:
        raise ConfigurationError("gather_rounds must be >= 1")

    cv2.setNumThreads(1)
    cv2.setRNGSeed(int(config.seed) & 0x7FFFFFFF)

    image_shape, color_paths, depth_paths = preflight_sequence(config, input_dir)
    height, width = image_shape
    output_names = [
        format_frame_name(config.render.output_pattern, sequence_index, frame_index)
        for sequence_index, frame_index in enumerate(
            range(config.frame_start, config.frame_start + config.frame_count)
        )
    ]
    for output_name in output_names:
        output_path = output_dir / output_name
        try:
            output_path.relative_to(output_dir)
        except ValueError as exc:
            raise ConfigurationError(
                f"render.output_pattern escapes output directory: {output_name}"
            ) from exc
        output_path.unlink(missing_ok=True)
    renderer = DOFRenderer(
        config.lens,
        config.aperture,
        config.render,
        width=width,
        height=height,
    )

    frames: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    output_files: list[Path] = []
    current_frame_index: int | None = None

    try:
        focus_path = _resolve_focus_track(config, input_dir)
        keyframes = load_focus_track(focus_path, config.frame_rate)
        track = FocusTrack(
            keyframes,
            config.frame_rate,
            config.frame_count,
            frame_start=config.frame_start,
        )
        warnings = track.warning_dicts()

        partial_report = build_report(
            config=config,
            width=width,
            height=height,
            frames=frames,
            warnings=warnings,
            status="in_progress",
            error=None,
        )
        write_json_atomic(output_dir / "report.partial.json", partial_report)

        for sequence_index, (color_path, depth_path) in enumerate(
            zip(color_paths, depth_paths, strict=True)
        ):
            frame_index = config.frame_start + sequence_index
            current_frame_index = frame_index
            started_ns = time.perf_counter_ns()

            focus_distance_m = track.sample(frame_index)
            raw_depth = load_depth_raw(depth_path)
            depth_frame = parse_depth(raw_depth, config.lens, path=depth_path)
            color = load_color(color_path, (height, width))
            try:
                coc_radius = renderer.lens.coc_radius_px(
                    depth_frame.meters, focus_distance_m
                )
            except ConfigurationError as exc:
                raise FocusTrackError(frame_index, str(exc)) from exc
            result = renderer.render(color, depth_frame, focus_distance_m)

            output_name = output_names[sequence_index]
            output_path = output_dir / output_name
            _write_png_atomic(
                output_path,
                result.image_bgr,
                config.render.png_compression,
            )
            output_files.append(output_path)

            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            frames.append(
                frame_report_entry(
                    frame_index=frame_index,
                    sequence_index=sequence_index,
                    output_file=output_name,
                    focus_distance_m=focus_distance_m,
                    coc_radius_px=coc_radius,
                    invalid_depth_pixels=depth_frame.invalid_count,
                    clamped_depth_pixels=depth_frame.clamped_count,
                    blades=config.aperture.blades,
                    max_blur_radius_px=config.lens.max_blur_radius_px,
                    render_time_ms=elapsed_ms,
                )
            )
            partial_report = build_report(
                config=config,
                width=width,
                height=height,
                frames=frames,
                warnings=warnings,
                status="in_progress",
                error=None,
            )
            write_json_atomic(output_dir / "report.partial.json", partial_report)
            progress(
                f"rendered frame {frame_index} "
                f"({sequence_index + 1}/{config.frame_count})"
            )

        track.validate_after_frames()
        final_report = build_report(
            config=config,
            width=width,
            height=height,
            frames=frames,
            warnings=warnings,
            status="completed",
            error=None,
        )
        report_path = output_dir / "report.json"
        write_json_atomic(report_path, final_report)
        (output_dir / "report.partial.json").unlink(missing_ok=True)
        return RenderSummary(
            report=final_report,
            report_path=report_path,
            output_files=output_files,
        )
    except FocusTrackError as exc:
        _write_failure_reports(
            output_dir,
            config,
            width,
            height,
            frames,
            warnings,
            exc,
            exc.frame_index,
        )
        raise
    except Exception as exc:
        frame_index = (
            current_frame_index
            if isinstance(exc, (PipelineError, InputDataError))
            else None
        )
        _write_failure_reports(
            output_dir,
            config,
            width,
            height,
            frames,
            warnings,
            exc,
            frame_index,
        )
        raise


def verify_slice_consistency(
    config: PipelineConfig,
    input_dir: str | Path,
) -> dict[str, Any]:
    input_dir = Path(input_dir).resolve()
    tile_shape, color_paths, depth_paths = preflight_sequence(config, input_dir)
    height, width = tile_shape
    full_render_config = dataclasses.replace(config.render, depth_slices=1)
    full_renderer = DOFRenderer(
        config.lens, config.aperture, full_render_config, width=width, height=height
    )
    sliced_renderer = DOFRenderer(
        config.lens, config.aperture, config.render, width=width, height=height
    )
    focus_path = _resolve_focus_track(config, input_dir)
    keyframes = load_focus_track(focus_path, config.frame_rate)
    track = FocusTrack(
        keyframes,
        config.frame_rate,
        config.frame_count,
        frame_start=config.frame_start,
    )
    frames: list[dict[str, Any]] = []
    overall_max = 0.0
    overall_mean = 0.0
    for sequence_index, (color_path, depth_path) in enumerate(
        zip(color_paths, depth_paths, strict=True)
    ):
        frame_index = config.frame_start + sequence_index
        focus_distance_m = track.sample(frame_index)
        depth_frame = parse_depth(load_depth_raw(depth_path), config.lens, path=depth_path)
        color = load_color(color_path, (height, width))
        full_image = full_renderer.render(color, depth_frame, focus_distance_m).image_bgr
        sliced_image = sliced_renderer.render(
            color, depth_frame, focus_distance_m
        ).image_bgr
        difference = image_difference_8bit(full_image, sliced_image)
        overall_max = max(overall_max, difference["max_abs_error_8bit"])
        overall_mean += difference["mean_abs_error_8bit"]
        frames.append(
            {
                "frame_index": frame_index,
                "focus_distance_m": float(focus_distance_m),
                **difference,
            }
        )
    track.validate_after_frames()
    overall_mean /= max(1, len(frames))
    passed = overall_max <= config.validation.slice_max_abs_error_8bit
    return {
        "status": "passed" if passed else "failed",
        "threshold_max_abs_error_8bit": float(
            config.validation.slice_max_abs_error_8bit
        ),
        "max_abs_error_8bit": float(overall_max),
        "mean_abs_error_8bit": float(overall_mean),
        "depth_slices": int(config.render.depth_slices),
        "gather_rounds": int(config.render.gather_rounds),
        "total_frames": len(frames),
        "frames": frames,
    }
