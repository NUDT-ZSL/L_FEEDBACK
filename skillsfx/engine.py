"""Batch generation, atomic publication, and manifest-only replay checks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
from typing import Any, Mapping
import uuid

import numpy as np

from . import __version__
from .dsp import (
    apply_channel_map,
    apply_tail_fade,
    build_tail_harmonics,
    clipping_sample_count,
    lookahead_limit,
    measure_rms_dbfs,
    ms_to_frames,
    normalize_loudness,
    peak_stats,
    remove_dc,
    resample_audio,
    resampled_length,
    resolve_channel_map,
    soft_compress,
    true_peak_stats,
)
from .errors import SkillSFXError
from .recipe import (
    BusParams,
    CompressorParams,
    EventSpec,
    LimiterParams,
    NormalizationParams,
    Recipe,
    load_recipe,
)
from .wavio import (
    encode_wav,
    encoded_sha256,
    read_wav,
    sha256_file,
    verify_wav,
)


_GAIN_STREAM = 0x600D
_TIME_STREAM = 0x71AE
_TAIL_STREAM = 0x7A11


def _signed_ms_to_frames(milliseconds: float, sample_rate: int) -> int:
    frames = int(
        math.floor(abs(milliseconds) * sample_rate / 1000.0 + 0.5)
    )
    return frames if milliseconds >= 0.0 else -frames


def derive_variant_seed(base_seed: int, variant_index: int) -> int:
    state = np.random.SeedSequence([base_seed, variant_index]).generate_state(
        1,
        dtype=np.uint64,
    )
    return int(state[0])


def _rng(base_seed: int, variant_index: int, event_index: int, stream: int):
    sequence = np.random.SeedSequence(
        [base_seed, variant_index, event_index, stream]
    )
    return np.random.default_rng(sequence)


def _event_context(event: EventSpec) -> dict[str, Any]:
    return {
        "event_index": event.index,
        "source_path": event.source_ref,
        "resolved_source_path": str(event.source_path),
    }


def _load_source_assets(
    recipe: Recipe,
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for event in recipe.events:
        key = str(event.source_path)
        if key in cache:
            continue
        try:
            audio, info = read_wav(event.source_path)
        except SkillSFXError as exc:
            context = dict(exc.context)
            context.update(_event_context(event))
            raise SkillSFXError(exc.code, exc.message, context) from exc
        audio.setflags(write=False)
        cache[key] = {
            "audio": audio,
            "info": info,
            "sha256": sha256_file(event.source_path),
        }
    return cache


def _sample_tail_harmonics(
    event: EventSpec,
    variant_index: int,
    base_seed: int,
    output: Recipe,
    channels: int,
) -> dict[str, Any] | None:
    config = output.randomization.tail_harmonics
    if event.role != "tail" or not config.enabled:
        return None
    rng = _rng(base_seed, variant_index, event.index, _TAIL_STREAM)
    count = int(rng.integers(1, len(config.ratios) + 1))
    selected_indices = np.sort(
        rng.choice(len(config.ratios), size=count, replace=False)
    )
    partials: list[dict[str, Any]] = []
    for selection_order, ratio_index in enumerate(selected_indices):
        ratio = float(config.ratios[int(ratio_index)])
        frequency = config.fundamental_hz * ratio
        if frequency >= output.output.sample_rate * 0.49:
            raise SkillSFXError(
                "SFX-E3001",
                "tail harmonic frequency exceeds the output Nyquist limit",
                {
                    **_event_context(event),
                    "fundamental_hz": config.fundamental_hz,
                    "ratio": ratio,
                    "frequency_hz": frequency,
                    "sample_rate": output.output.sample_rate,
                },
            )
        gain_db = float(rng.uniform(config.gain_db_min, config.gain_db_max))
        phase = float(rng.uniform(0.0, 2.0 * math.pi))
        partials.append(
            {
                "ratio": ratio,
                "frequency_hz": frequency,
                "gain_db": gain_db,
                "phase_rad": phase,
                "selection_order": selection_order,
            }
        )
    if channels == 1:
        channel_gains = [1.0]
    else:
        pan = float(rng.uniform(-0.85, 0.85))
        left = math.sqrt((1.0 - pan) / 2.0)
        right = math.sqrt((1.0 + pan) / 2.0)
        channel_gains = [left, right]
    return {
        "fundamental_hz": config.fundamental_hz,
        "attack_ms": config.attack_ms,
        "decay_ms": config.decay_ms,
        "partials": partials,
        "channel_gains": channel_gains,
        "stream_seed": [base_seed, variant_index, event.index, _TAIL_STREAM],
    }


def _build_timeline(
    recipe: Recipe,
    level: int,
    variant_index: int,
    manifest_directory: Path,
    source_cache: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = recipe.output
    timeline: list[dict[str, Any]] = []
    for event in recipe.events:
        source_entry = source_cache[str(event.source_path)]
        info = source_entry["info"]
        channel_map, descriptor = resolve_channel_map(
            event.channel_map,
            info.channels,
            output.channels,
            _event_context(event),
        )
        gain_rng = _rng(
            recipe.seed,
            variant_index,
            event.index,
            _GAIN_STREAM,
        )
        time_rng = _rng(
            recipe.seed,
            variant_index,
            event.index,
            _TIME_STREAM,
        )
        gain_jitter_db = (
            float(
                gain_rng.uniform(
                    -recipe.randomization.gain_jitter_db,
                    recipe.randomization.gain_jitter_db,
                )
            )
            if recipe.randomization.gain_jitter_db > 0.0
            else 0.0
        )
        time_jitter_ms = (
            float(
                time_rng.uniform(
                    -recipe.randomization.time_jitter_ms,
                    recipe.randomization.time_jitter_ms,
                )
            )
            if recipe.randomization.time_jitter_ms > 0.0
            else 0.0
        )
        base_offset_frames = ms_to_frames(event.offset_ms, output.sample_rate)
        time_jitter_frames = _signed_ms_to_frames(
            time_jitter_ms,
            output.sample_rate,
        )
        offset_frames = base_offset_frames + time_jitter_frames
        if offset_frames < 0 or offset_frames >= output.duration_frames:
            raise SkillSFXError(
                "SFX-E2003",
                "randomized event offset is outside the output timeline",
                {
                    **_event_context(event),
                    "variant_index": variant_index,
                    "offset_frames": offset_frames,
                    "offset_ms": 1000.0 * offset_frames / output.sample_rate,
                    "output_frames": output.duration_frames,
                    "output_duration_ms": output.duration_ms,
                    "time_jitter_ms": time_jitter_ms,
                },
            )
        resampled_frames = resampled_length(
            info.frame_count,
            info.sample_rate,
            output.sample_rate,
        )
        span_frames = (
            ms_to_frames(event.duration_ms, output.sample_rate)
            if event.duration_ms is not None
            else resampled_frames
        )
        if span_frames <= 0:
            raise SkillSFXError(
                "SFX-E1003",
                "event duration resolves to zero samples",
                {
                    **_event_context(event),
                    "duration_ms": event.duration_ms,
                },
            )
        placed_frames = min(
            span_frames,
            max(0, output.duration_frames - offset_frames),
        )
        tail = _sample_tail_harmonics(
            event,
            variant_index,
            recipe.seed,
            recipe,
            output.channels,
        )
        try:
            source_reference = os.path.relpath(
                event.source_path,
                start=manifest_directory,
            )
        except ValueError:
            # Windows raises when source and batch live on different volumes.
            source_reference = str(event.source_path)
        source_relative = source_reference.replace(os.sep, "/")
        timeline.append(
            {
                "event_index": event.index,
                "role": event.role,
                "source": {
                    "path": source_relative,
                    "resolved_path": str(event.source_path),
                    "sha256": source_entry["sha256"],
                    "sample_rate": info.sample_rate,
                    "channels": info.channels,
                    "frames": info.frame_count,
                    "sample_format": info.sample_format,
                    "bits_per_sample": info.bits_per_sample,
                },
                "configured_offset_ms": event.offset_ms,
                "base_offset_frames": base_offset_frames,
                "offset_frames": offset_frames,
                "offset_seconds": offset_frames / output.sample_rate,
                "time_jitter_ms": time_jitter_ms,
                "time_jitter_frames": time_jitter_frames,
                "gain_linear_configured": event.gain,
                "gain_jitter_db": gain_jitter_db,
                "gain_linear_applied": event.gain
                * (10.0 ** (gain_jitter_db / 20.0)),
                "span_frames": span_frames,
                "placed_frames": placed_frames,
                "duration_ms_configured": event.duration_ms,
                "source_resampled_frames": resampled_frames,
                "channel_map": descriptor,
                "channel_matrix": channel_map.tolist(),
                "tail_harmonics": tail,
                "seed_streams": {
                    "gain": [
                        recipe.seed,
                        variant_index,
                        event.index,
                        _GAIN_STREAM,
                    ],
                    "time": [
                        recipe.seed,
                        variant_index,
                        event.index,
                        _TIME_STREAM,
                    ],
                },
            }
        )
    return timeline


def _parse_bus(value: BusParams | Mapping[str, Any]) -> BusParams:
    if isinstance(value, BusParams):
        return value
    compressor = value["compressor"]
    limiter = value["limiter"]
    return BusParams(
        compressor=CompressorParams(
            threshold_db=float(compressor["threshold_db"]),
            ratio=float(compressor["ratio"]),
            knee_db=float(compressor["knee_db"]),
            attack_ms=float(compressor["attack_ms"]),
            release_ms=float(compressor["release_ms"]),
        ),
        tail_fade_ms=float(value["tail_fade_ms"]),
        limiter=LimiterParams(
            ceiling_dbtp=float(limiter["ceiling_dbtp"]),
            lookahead_ms=float(limiter["lookahead_ms"]),
            attack_ms=float(limiter["attack_ms"]),
            release_ms=float(limiter["release_ms"]),
        ),
    )


def _parse_normalization(
    value: NormalizationParams | Mapping[str, Any],
) -> NormalizationParams:
    if isinstance(value, NormalizationParams):
        return value
    return NormalizationParams(
        target_lufs=(
            float(value["target_lufs"])
            if value.get("target_lufs") is not None
            else None
        ),
        target_rms_dbfs=(
            float(value["target_rms_dbfs"])
            if value.get("target_rms_dbfs") is not None
            else None
        ),
        min_gain_db=float(value.get("min_gain_db", -24.0)),
        max_gain_db=float(value.get("max_gain_db", 24.0)),
    )


def _load_timeline_source(
    event: Mapping[str, Any],
    source_cache: dict[str, dict[str, Any]] | None,
    *,
    replay: bool,
) -> tuple[np.ndarray, Mapping[str, Any]]:
    source = event["source"]
    resolved = source.get("resolved_path") or source["path"]
    key = str(resolved)
    if source_cache is not None and key in source_cache:
        cached = source_cache[key]
        if cached["sha256"] != source["sha256"]:
            code = "SFX-E5001" if replay else "SFX-E2002"
            raise SkillSFXError(
                code,
                "source WAV hash changed after timeline construction",
                {
                    "event_index": event["event_index"],
                    "source_path": source["path"],
                    "expected_sha256": source["sha256"],
                    "actual_sha256": cached["sha256"],
                },
            )
        _verify_source_metadata(event, cached["info"], replay=replay)
        return cached["audio"], cached["info"]
    try:
        audio, info = read_wav(resolved)
    except SkillSFXError as exc:
        code = "SFX-E5001" if replay else exc.code
        raise SkillSFXError(
            code,
            "timeline source WAV could not be read",
            {
                "event_index": event["event_index"],
                "source_path": source["path"],
                "resolved_source_path": str(resolved),
                "cause": exc.to_dict(),
            },
        ) from exc
    actual_sha = sha256_file(resolved)
    if actual_sha != source["sha256"]:
        code = "SFX-E5001" if replay else "SFX-E2002"
        raise SkillSFXError(
            code,
            "source WAV hash does not match the timeline",
            {
                "event_index": event["event_index"],
                "source_path": source["path"],
                "expected_sha256": source["sha256"],
                "actual_sha256": actual_sha,
            },
        )
    _verify_source_metadata(event, info, replay=replay)
    if source_cache is not None:
        audio.setflags(write=False)
        source_cache[key] = {
            "audio": audio,
            "info": info,
            "sha256": actual_sha,
        }
    return audio, info


def _verify_source_metadata(
    event: Mapping[str, Any],
    info: Any,
    *,
    replay: bool,
) -> None:
    source = event["source"]
    comparisons = (
        ("sample_rate", info.sample_rate),
        ("channels", info.channels),
        ("frames", info.frame_count),
        ("sample_format", info.sample_format),
        ("bits_per_sample", info.bits_per_sample),
    )
    for field, actual in comparisons:
        expected = source.get(field)
        if expected is not None and expected != actual:
            raise SkillSFXError(
                "SFX-E5001" if replay else "SFX-E2002",
                "source metadata does not match the timeline",
                {
                    "event_index": event["event_index"],
                    "source_path": source["path"],
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                },
            )


def render_timeline(
    timeline: list[dict[str, Any]],
    output: Mapping[str, Any],
    bus: BusParams | Mapping[str, Any],
    normalization: NormalizationParams | Mapping[str, Any],
    *,
    source_cache: dict[str, dict[str, Any]] | None = None,
    replay: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_rate = int(output["sample_rate"])
    channels = int(output["channels"])
    frames = int(output.get("duration_frames") or 0)
    if frames <= 0:
        frames = ms_to_frames(float(output["duration_ms"]), sample_rate)
    mix = np.zeros((frames, channels), dtype=np.float64)
    last_event_end = 0
    for event in timeline:
        source_audio, info = _load_timeline_source(
            event,
            source_cache,
            replay=replay,
        )
        resampled = resample_audio(
            source_audio,
            int(info.sample_rate),
            sample_rate,
        )
        matrix = np.asarray(event["channel_matrix"], dtype=np.float64)
        expected_shape = (channels, int(info.channels))
        if matrix.shape != expected_shape:
            raise SkillSFXError(
                "SFX-E5001" if replay else "SFX-E2004",
                "timeline channel matrix has the wrong shape",
                {
                    "event_index": event["event_index"],
                    "source_path": event["source"]["path"],
                    "actual_shape": list(matrix.shape),
                    "expected_shape": list(expected_shape),
                },
            )
        mapped = apply_channel_map(resampled, matrix)
        span_frames = int(event["span_frames"])
        if mapped.shape[0] < span_frames:
            block = np.zeros((span_frames, channels), dtype=np.float64)
            block[: mapped.shape[0]] = mapped
        else:
            block = mapped[:span_frames].copy()
        if event.get("tail_harmonics") is not None:
            block += build_tail_harmonics(
                event["tail_harmonics"],
                span_frames,
                sample_rate,
                channels,
            )
        block *= float(event["gain_linear_applied"])
        offset_frames = int(event["offset_frames"])
        if offset_frames < 0 or offset_frames >= frames:
            raise SkillSFXError(
                "SFX-E2003" if not replay else "SFX-E5001",
                "timeline event offset is outside the output duration",
                {
                    "event_index": event["event_index"],
                    "source_path": event["source"]["path"],
                    "offset_frames": offset_frames,
                    "output_frames": frames,
                    "offset_ms": 1000.0 * offset_frames / sample_rate,
                },
            )
        placed_frames = min(span_frames, frames - offset_frames)
        mix[offset_frames : offset_frames + placed_frames] += block[:placed_frames]
        last_event_end = max(last_event_end, offset_frames + placed_frames)

    bus_params = _parse_bus(bus)
    normalization_params = _parse_normalization(normalization)
    mixed_peak = peak_stats(mix)
    mixed_true_peak = true_peak_stats(mix, sample_rate)

    mix, dc_info = remove_dc(mix)
    mix, compressor_info = soft_compress(
        mix,
        sample_rate,
        bus_params.compressor,
    )
    tail_margin_frames = max(0, frames - last_event_end)
    mix, fade_info = apply_tail_fade(
        mix,
        sample_rate,
        bus_params.tail_fade_ms,
        tail_margin_frames,
    )
    mix, normalization_info = normalize_loudness(
        mix,
        sample_rate,
        normalization_params,
    )
    mix, limiter_info = lookahead_limit(
        mix,
        sample_rate,
        bus_params.limiter,
    )

    warnings: list[dict[str, Any]] = []
    if fade_info["warning"] is not None:
        warnings.append(fade_info["warning"])
    if normalization_info["warning"] is not None:
        warnings.append(normalization_info["warning"])
    if abs(float(limiter_info["safety_trim_db"])) > 1e-9:
        warnings.append(
            {
                "code": "limiter_safety_trim",
                "message": "a constant safety trim followed the smoothed limiter",
                "trim_db": limiter_info["safety_trim_db"],
            }
        )
    clip_count = clipping_sample_count(mix)
    if clip_count:
        raise SkillSFXError(
            "SFX-E4002",
            "limiter output contains clipping samples",
            {"clipping_sample_count": clip_count},
        )
    float_peak = peak_stats(mix)
    float_true_peak = true_peak_stats(mix, sample_rate)
    report = {
        "mixed_input": {
            "peak": mixed_peak,
            "true_peak": mixed_true_peak,
        },
        "dc_removal": dc_info,
        "compressor": compressor_info,
        "tail_fade": fade_info,
        "normalization": normalization_info,
        "limiter": limiter_info,
        "rendered_float": {
            **float_peak,
            **float_true_peak,
            "clipping_sample_count": clip_count,
        },
        "warnings": warnings,
    }
    return mix, report


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (result or "skill")[:64]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _generate_one_variant(
    *,
    recipe: Recipe,
    level: int,
    bus: BusParams,
    variant_index: int,
    stage_directory: Path,
    source_cache: Mapping[str, Mapping[str, Any]],
    cancel_event: threading.Event,
) -> dict[str, Any]:
    if cancel_event.is_set():
        raise SkillSFXError(
            "SFX-E6001",
            "batch was cancelled before variant generation",
            {"variant_index": variant_index},
        )
    timeline = _build_timeline(
        recipe,
        level,
        variant_index,
        stage_directory,
        source_cache,
    )
    if cancel_event.is_set():
        raise SkillSFXError(
            "SFX-E6001",
            "batch was cancelled before variant rendering",
            {"variant_index": variant_index},
        )
    audio, report = render_timeline(
        timeline,
        recipe.output.to_dict(),
        bus,
        recipe.normalization,
        source_cache=dict(source_cache),
    )
    if cancel_event.is_set():
        raise SkillSFXError(
            "SFX-E6001",
            "batch was cancelled before variant encoding",
            {"variant_index": variant_index},
        )
    filename = (
        f"{_slug(recipe.skill_name)}_l{level}_v{variant_index:04d}.wav"
    )
    path = stage_directory / filename
    encoded = encode_wav(audio, recipe.output.sample_rate, recipe.output.bit_depth)
    path.write_bytes(encoded)
    output_sha = encoded_sha256(encoded)
    try:
        decoded, info = verify_wav(
            path,
            sample_rate=recipe.output.sample_rate,
            channels=recipe.output.channels,
            bit_depth=recipe.output.bit_depth,
            frame_count=recipe.output.duration_frames,
        )
    except SkillSFXError as exc:
        raise SkillSFXError(
            "SFX-E4001",
            "post-write WAV verification failed",
            {
                "variant_index": variant_index,
                "output_path": str(path),
                "cause": exc.to_dict(),
            },
        ) from exc
    decoded_peak = peak_stats(decoded)
    decoded_true_peak = true_peak_stats(decoded, recipe.output.sample_rate)
    decoded_clip_count = clipping_sample_count(decoded)
    if decoded_clip_count:
        raise SkillSFXError(
            "SFX-E4002",
            "encoded WAV contains clipping samples",
            {
                "variant_index": variant_index,
                "output_path": str(path),
                "clipping_sample_count": decoded_clip_count,
            },
        )
    if float(decoded_true_peak["true_peak_dbtp"]) > bus.limiter.ceiling_dbtp + 1e-4:
        raise SkillSFXError(
            "SFX-E4002",
            "encoded WAV exceeds the true-peak ceiling",
            {
                "variant_index": variant_index,
                "true_peak_dbtp": decoded_true_peak["true_peak_dbtp"],
                "ceiling_dbtp": bus.limiter.ceiling_dbtp,
            },
        )
    analysis = {
        "normalization_gain_db": report["normalization"]["applied_gain_db"],
        "limiter_trigger_count": report["limiter"]["trigger_count"],
        "limiter_trigger_intervals": report["limiter"]["trigger_intervals"],
        "rendered_peak_dbfs": report["rendered_float"]["peak_dbfs"],
        "rendered_true_peak_dbtp": report["rendered_float"]["true_peak_dbtp"],
        "decoded_peak_dbfs": decoded_peak["peak_dbfs"],
        "decoded_true_peak_dbtp": decoded_true_peak["true_peak_dbtp"],
        "clipping_sample_count": decoded_clip_count,
        "warnings": report["warnings"],
    }
    return {
        "index": variant_index,
        "seed": derive_variant_seed(recipe.seed, variant_index),
        "seed_sequence": [recipe.seed, variant_index],
        "output": {
            "path": filename,
            "sha256": output_sha,
            "frames": recipe.output.duration_frames,
            "duration_seconds": (
                recipe.output.duration_frames / recipe.output.sample_rate
            ),
            "wav": info.to_dict(),
        },
        "timeline": timeline,
        "bus": bus.to_dict(),
        "normalization_params": recipe.normalization.to_dict(),
        "processing": report,
        "analysis": analysis,
    }


def _prepare_stage(output_directory: Path) -> Path:
    parent = output_directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_directory.exists():
        if not output_directory.is_dir():
            raise SkillSFXError(
                "SFX-E6001",
                "output path exists and is not a directory",
                {"output_path": str(output_directory)},
            )
        try:
            nonempty = any(output_directory.iterdir())
        except OSError as exc:
            raise SkillSFXError(
                "SFX-E6001",
                "output directory cannot be inspected",
                {"output_path": str(output_directory), "reason": str(exc)},
            ) from exc
        if nonempty:
            raise SkillSFXError(
                "SFX-E6001",
                "output directory must be new or empty for atomic publication",
                {"output_path": str(output_directory)},
            )
        output_directory.rmdir()
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.stage-",
            dir=parent,
        )
    )


def generate_batch(
    recipe_path: str | Path,
    *,
    level: int,
    output_directory: str | Path,
    workers: int = 1,
) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int):
        raise SkillSFXError(
            "SFX-E6001",
            "worker count must be an integer",
            {"workers": workers},
        )
    if workers < 1 or workers > 64:
        raise SkillSFXError(
            "SFX-E6001",
            "worker count must be between 1 and 64",
            {"workers": workers},
        )
    recipe = load_recipe(recipe_path)
    bus = recipe.bus_for_level(level)
    output_directory = Path(output_directory).resolve()
    source_cache = _load_source_assets(recipe)
    stage_directory = _prepare_stage(output_directory)
    run_material = (
        recipe.sha256
        + "\n"
        + str(level)
        + "\n"
        + json.dumps(recipe.output.to_dict(), sort_keys=True)
    )
    run_id = hashlib.sha256(run_material.encode("utf-8")).hexdigest()
    cancel_event = threading.Event()
    variants: list[dict[str, Any]] = []
    failure: SkillSFXError | None = None
    unexpected_failure: BaseException | None = None

    try:
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="skillsfx",
        ) as executor:
            futures: dict[Future[dict[str, Any]], int] = {
                executor.submit(
                    _generate_one_variant,
                    recipe=recipe,
                    level=level,
                    bus=bus,
                    variant_index=index,
                    stage_directory=stage_directory,
                    source_cache=source_cache,
                    cancel_event=cancel_event,
                ): index
                for index in range(recipe.variant_count)
            }
            for future in as_completed(futures):
                try:
                    variants.append(future.result())
                except SkillSFXError as exc:
                    if failure is None:
                        failure = exc
                    cancel_event.set()
                    for pending in futures:
                        pending.cancel()
                except BaseException as exc:
                    if unexpected_failure is None:
                        unexpected_failure = exc
                    cancel_event.set()
                    for pending in futures:
                        pending.cancel()
        if failure is not None:
            context = dict(failure.context)
            context["batch_failed"] = True
            context["staged_output_removed"] = True
            raise SkillSFXError(failure.code, failure.message, context) from failure
        if unexpected_failure is not None:
            raise SkillSFXError(
                "SFX-E6001",
                "worker thread failed unexpectedly",
                {
                    "exception_type": type(unexpected_failure).__name__,
                    "reason": str(unexpected_failure),
                    "staged_output_removed": True,
                },
            ) from unexpected_failure
        if len(variants) != recipe.variant_count:
            raise SkillSFXError(
                "SFX-E6001",
                "not all variants completed",
                {
                    "expected": recipe.variant_count,
                    "completed": len(variants),
                    "staged_output_removed": True,
                },
            )
        variants.sort(key=lambda item: item["index"])
        manifest = {
            "manifest_version": 1,
            "generator": {
                "name": "skillsfx",
                "version": __version__,
            },
            "run_id": run_id,
            "recipe": recipe.to_manifest_summary(level),
            "batch": {
                "status": "complete",
                "workers": workers,
                "variant_count": recipe.variant_count,
            },
            "variants": variants,
        }
        manifest = _json_safe(manifest)
        manifest_text = json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        manifest_path = stage_directory / "manifest.json"
        manifest_path.write_text(manifest_text, encoding="utf-8")
        json.loads(manifest_path.read_text(encoding="utf-8"))
        os.replace(stage_directory, output_directory)
    except BaseException as exc:
        shutil.rmtree(stage_directory, ignore_errors=True)
        if isinstance(exc, SkillSFXError):
            raise
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise SkillSFXError(
            "SFX-E6001",
            "batch pipeline failed and staged output was removed",
            {
                "output_path": str(output_directory),
                "exception_type": type(exc).__name__,
                "reason": str(exc),
                "staged_output_removed": True,
            },
        ) from exc

    return {
        "status": "ok",
        "run_id": run_id,
        "level": level,
        "variant_count": recipe.variant_count,
        "workers": workers,
        "output_directory": str(output_directory),
        "manifest": str(output_directory / "manifest.json"),
    }


def _manifest_error(
    message: str,
    context: Mapping[str, Any] | None = None,
) -> SkillSFXError:
    return SkillSFXError("SFX-E5001", message, dict(context or {}))


def verify_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is not allowed: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except OSError as exc:
        raise _manifest_error(
            "manifest could not be read",
            {"manifest_path": str(path), "reason": str(exc)},
        ) from exc
    except json.JSONDecodeError as exc:
        raise _manifest_error(
            "manifest is not valid JSON",
            {
                "manifest_path": str(path),
                "line": exc.lineno,
                "column": exc.colno,
            },
        ) from exc
    except ValueError as exc:
        raise _manifest_error(
            "manifest is not strict JSON",
            {"manifest_path": str(path), "reason": str(exc)},
        ) from exc
    if not isinstance(manifest, dict):
        raise _manifest_error("manifest root must be an object")
    if manifest.get("manifest_version") != 1:
        raise _manifest_error(
            "unsupported manifest version",
            {"manifest_version": manifest.get("manifest_version")},
        )
    variants = manifest.get("variants")
    if not isinstance(variants, list) or not variants:
        raise _manifest_error("manifest has no variants")
    recipe_summary = manifest.get("recipe")
    if not isinstance(recipe_summary, dict):
        raise _manifest_error("manifest is missing recipe summary")
    base_seed = recipe_summary.get("seed")
    output = recipe_summary.get("output")
    if not isinstance(base_seed, int) or not isinstance(output, dict):
        raise _manifest_error("manifest recipe summary is incomplete")
    if output.get("sample_rate") not in (44100, 48000):
        raise _manifest_error(
            "manifest output sample rate is unsupported",
            {"sample_rate": output.get("sample_rate")},
        )
    run_material = (
        str(recipe_summary.get("sha256", ""))
        + "\n"
        + str(recipe_summary.get("level", ""))
        + "\n"
        + json.dumps(output, sort_keys=True)
    )
    expected_run_id = hashlib.sha256(run_material.encode("utf-8")).hexdigest()
    if manifest.get("run_id") != expected_run_id:
        raise _manifest_error(
            "manifest run ID does not match its recipe summary",
            {
                "manifest_run_id": manifest.get("run_id"),
                "expected_run_id": expected_run_id,
            },
        )

    source_cache: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for variant in variants:
        index = variant.get("index")
        try:
            if not isinstance(index, int):
                raise _manifest_error("variant index is not an integer")
            expected_seed = derive_variant_seed(base_seed, index)
            if variant.get("seed") != expected_seed:
                raise _manifest_error(
                    "variant seed does not match its seed sequence",
                    {
                        "variant_index": index,
                        "manifest_seed": variant.get("seed"),
                        "expected_seed": expected_seed,
                    },
                )
            timeline = variant.get("timeline")
            bus = variant.get("bus")
            normalization = variant.get("normalization_params")
            output_entry = variant.get("output")
            if (
                not isinstance(timeline, list)
                or not isinstance(bus, dict)
                or not isinstance(normalization, dict)
                or not isinstance(output_entry, dict)
            ):
                raise _manifest_error(
                    "variant manifest is incomplete",
                    {"variant_index": index},
                )
            for event in timeline:
                source = event.get("source") if isinstance(event, dict) else None
                if not isinstance(source, dict):
                    raise _manifest_error(
                        "event source entry is missing",
                        {"variant_index": index},
                    )
                candidate = source.get("path")
                if not isinstance(candidate, str):
                    raise _manifest_error(
                        "event source path is invalid",
                        {"variant_index": index},
                    )
                source_path = Path(candidate)
                if not source_path.is_absolute():
                    source_path = path.parent / source_path
                if not source_path.exists() and source.get("resolved_path"):
                    source_path = Path(source["resolved_path"])
                source_path = source_path.resolve()
                source["resolved_path"] = str(source_path)
                key = str(source_path)
                if key not in source_cache:
                    try:
                        audio, info = read_wav(source_path)
                    except SkillSFXError as exc:
                        raise _manifest_error(
                            "manifest source WAV could not be read",
                            {
                                "variant_index": index,
                                "event_index": event.get("event_index"),
                                "source_path": str(source_path),
                                "cause": exc.to_dict(),
                            },
                        ) from exc
                    actual_hash = sha256_file(source_path)
                    if actual_hash != source.get("sha256"):
                        raise _manifest_error(
                            "manifest source WAV hash mismatch",
                            {
                                "variant_index": index,
                                "event_index": event.get("event_index"),
                                "source_path": str(source_path),
                                "expected_sha256": source.get("sha256"),
                                "actual_sha256": actual_hash,
                            },
                        )
                    audio.setflags(write=False)
                    source_cache[key] = {
                        "audio": audio,
                        "info": info,
                        "sha256": actual_hash,
                    }
            rendered, report = render_timeline(
                timeline,
                output,
                bus,
                normalization,
                source_cache=source_cache,
                replay=True,
            )
            encoded = encode_wav(
                rendered,
                int(output["sample_rate"]),
                output["bit_depth"],
            )
            reconstructed_sha = encoded_sha256(encoded)
            output_path = Path(str(output_entry.get("path", "")))
            if not output_path.is_absolute():
                output_path = path.parent / output_path
            output_path = output_path.resolve()
            try:
                actual_bytes = output_path.read_bytes()
            except OSError as exc:
                raise _manifest_error(
                    "variant output file could not be read",
                    {
                        "variant_index": index,
                        "output_path": str(output_path),
                        "reason": str(exc),
                    },
                ) from exc
            actual_sha = encoded_sha256(actual_bytes)
            if actual_sha != output_entry.get("sha256"):
                raise _manifest_error(
                    "output file hash does not match the manifest",
                    {
                        "variant_index": index,
                        "output_path": str(output_path),
                        "expected_sha256": output_entry.get("sha256"),
                        "actual_sha256": actual_sha,
                    },
                )
            if reconstructed_sha != actual_sha or encoded != actual_bytes:
                raise _manifest_error(
                    "output file is not reproducible from the manifest",
                    {
                        "variant_index": index,
                        "output_path": str(output_path),
                        "manifest_sha256": output_entry.get("sha256"),
                        "reconstructed_sha256": reconstructed_sha,
                    },
                )
            try:
                decoded, info = verify_wav(
                    output_path,
                    sample_rate=int(output["sample_rate"]),
                    channels=int(output["channels"]),
                    bit_depth=output["bit_depth"],
                    frame_count=int(output_entry["frames"]),
                )
            except SkillSFXError as exc:
                raise _manifest_error(
                    "output WAV header verification failed during replay",
                    {
                        "variant_index": index,
                        "cause": exc.to_dict(),
                    },
                ) from exc
            decoded_clip_count = clipping_sample_count(decoded)
            if decoded_clip_count:
                raise _manifest_error(
                    "replayed output contains clipping samples",
                    {
                        "variant_index": index,
                        "clipping_sample_count": decoded_clip_count,
                    },
                )
            analysis = variant.get("analysis", {})
            recorded_gain = analysis.get("normalization_gain_db")
            actual_gain = report["normalization"]["applied_gain_db"]
            if recorded_gain is not None and abs(
                float(recorded_gain) - float(actual_gain)
            ) > 1e-6:
                raise _manifest_error(
                    "normalization gain is not reproducible",
                    {
                        "variant_index": index,
                        "manifest_gain_db": recorded_gain,
                        "reproduced_gain_db": actual_gain,
                    },
                )
            recorded_triggers = analysis.get("limiter_trigger_count")
            actual_triggers = report["limiter"]["trigger_count"]
            if (
                recorded_triggers is not None
                and int(recorded_triggers) != int(actual_triggers)
            ):
                raise _manifest_error(
                    "limiter trigger count is not reproducible",
                    {
                        "variant_index": index,
                        "manifest_trigger_count": recorded_triggers,
                        "reproduced_trigger_count": actual_triggers,
                    },
                )
            recorded_intervals = analysis.get("limiter_trigger_intervals")
            actual_intervals = report["limiter"]["trigger_intervals"]
            if (
                recorded_intervals is not None
                and recorded_intervals != actual_intervals
            ):
                raise _manifest_error(
                    "limiter trigger intervals are not reproducible",
                    {
                        "variant_index": index,
                        "manifest_trigger_intervals": recorded_intervals,
                        "reproduced_trigger_intervals": actual_intervals,
                    },
                )
        except SkillSFXError as exc:
            failures.append(
                {
                    "variant_index": index,
                    "error_code": exc.code,
                    "message": exc.message,
                    "context": exc.context,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "variant_index": index,
                    "error_code": "SFX-E5001",
                    "message": "unexpected replay failure",
                    "context": {
                        "exception_type": type(exc).__name__,
                        "reason": str(exc),
                    },
                }
            )
    if failures:
        raise _manifest_error(
            "manifest replay consistency check failed",
            {
                "manifest_path": str(path),
                "failure_count": len(failures),
                "failures": failures,
            },
        )
    return {
        "status": "ok",
        "manifest": str(path),
        "verified_variants": len(variants),
        "run_id": manifest.get("run_id"),
    }
