from __future__ import annotations

import math
import sys
import time
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import LoadedProject, load_project
from .checkpoint import (
    PHASE_FINAL,
    PHASE_MEASURE,
    BlockLayout,
    CheckpointState,
    RawPcmStore,
    read_checkpoint,
    write_checkpoint,
)
from .dsp import (
    PreparedProject,
    commentary_effective_frames,
    float_array_from_le_bytes,
    float_array_to_le_bytes,
    int16_array_to_le_bytes,
    prepare_project,
    quantize_with_gain,
    render_premix_block,
)
from .errors import (
    AppError,
    CheckpointValidationError,
    checkpoint_invalid,
    output_error,
)
from .schema import ProjectConfig, load_config
from .util import (
    atomic_replace,
    db_to_lin,
    fsync_handle,
    lin_to_db,
    rms_to_db,
    sha256_bytes,
    sha256_file,
)


@dataclass(frozen=True)
class ExportOptions:
    strict_checkpoint: bool = False
    stop_after_blocks: int | None = None
    stop_after_phase: str = "any"


@dataclass(frozen=True)
class ArtifactPaths:
    checkpoint: Path
    measure_raw: Path
    final_raw: Path
    temporary_wav: Path


class SimulatedInterrupt(Exception):
    def __init__(self, phase: str, completed_blocks: int) -> None:
        super().__init__("simulated export interruption")
        self.phase = phase
        self.completed_blocks = completed_blocks


def artifact_paths(output: Path) -> ArtifactPaths:
    return ArtifactPaths(
        checkpoint=Path(str(output) + ".ckpt.json"),
        measure_raw=Path(str(output) + ".measure.raw"),
        final_raw=Path(str(output) + ".final.raw"),
        temporary_wav=Path(str(output) + ".tmp.wav"),
    )


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _reset_artifacts(paths: ArtifactPaths) -> None:
    for path in (
        paths.checkpoint,
        Path(str(paths.checkpoint) + ".tmp"),
        paths.measure_raw,
        paths.final_raw,
        paths.temporary_wav,
    ):
        _remove_if_exists(path)


def _normalization_matches_config(
    normalization: dict[str, Any],
    config: ProjectConfig,
) -> bool:
    return (
        math.isclose(
            normalization["target_rms_dbfs"],
            config.mix.target_rms_dbfs,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            normalization["peak_limit_dbfs"],
            config.mix.peak_limit_dbfs,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )


def _scan_measure_stream(
    path: Path,
    layout: BlockLayout,
    soft_knee_ratio: float,
    peak_limit_dbfs: float,
) -> tuple[float, float, bool]:
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise output_error("measure data stream is unavailable", error=str(exc)) from None
    if actual_size != layout.measure_bytes:
        raise output_error(
            "measure data stream length is inconsistent",
            expected_bytes=layout.measure_bytes,
            actual_bytes=actual_size,
        )
    sum_squares = 0.0
    peak = 0.0
    sample_count = 0
    with path.open("rb") as handle:
        for frame_count in layout.frame_counts:
            byte_count = frame_count * layout.channels * 4
            data = handle.read(byte_count)
            if len(data) != byte_count:
                raise output_error("measure data stream ended unexpectedly")
            values = float_array_from_le_bytes(data)
            for value in values:
                numeric = float(value)
                sum_squares += numeric * numeric
                magnitude = abs(numeric)
                if magnitude > peak:
                    peak = magnitude
            sample_count += len(values)
    expected_samples = layout.total_frames * layout.channels
    if sample_count != expected_samples:
        raise output_error(
            "measure sample count is inconsistent",
            expected=expected_samples,
            actual=sample_count,
        )
    rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
    ceiling = db_to_lin(peak_limit_dbfs)
    knee = ceiling * soft_knee_ratio
    soft_triggered = peak > knee + 1.0e-12
    return rms, peak, soft_triggered


def _derive_normalization(
    config: ProjectConfig,
    measured_rms: float,
    measured_peak: float,
    soft_triggered: bool,
) -> dict[str, Any]:
    target = config.mix.target_rms_dbfs
    peak_limit = config.mix.peak_limit_dbfs
    measured_rms_db = rms_to_db(measured_rms)
    measured_peak_db = lin_to_db(measured_peak, floor=1.0e-9)
    max_gain_triggered = False
    peak_limit_triggered = False

    if measured_rms <= 0.0:
        applied_gain_db = 0.0
    else:
        desired_gain_db = target - measured_rms_db
        applied_gain_db = desired_gain_db
        if applied_gain_db > config.mix.max_gain_db:
            applied_gain_db = config.mix.max_gain_db
            max_gain_triggered = True
        peak_headroom_db = peak_limit - measured_peak_db
        if applied_gain_db > peak_headroom_db:
            applied_gain_db = peak_headroom_db
            peak_limit_triggered = True

    return {
        "target_rms_dbfs": target,
        "measured_rms_dbfs": measured_rms_db,
        "measured_peak_dbfs": measured_peak_db,
        "applied_gain_db": applied_gain_db,
        "peak_limit_triggered": peak_limit_triggered,
        "soft_limit_triggered": soft_triggered,
        "peak_limit_dbfs": peak_limit,
        "max_gain_triggered": max_gain_triggered,
    }


def _make_measure_state(
    writer: RawPcmStore,
    config_digest: str,
    layout: BlockLayout,
) -> CheckpointState:
    return CheckpointState(
        phase=PHASE_MEASURE,
        completed_blocks=writer.completed_blocks,
        data_bytes=writer.data_bytes,
        data_sha256=writer.digest_hex,
        config_digest=config_digest,
        block_frames=layout.block_frames,
        total_blocks=layout.total_blocks,
        total_frames=layout.total_frames,
        channels=layout.channels,
        normalization=None,
    )


def _make_final_state(
    writer: RawPcmStore,
    config_digest: str,
    layout: BlockLayout,
    normalization: dict[str, Any],
) -> CheckpointState:
    return CheckpointState(
        phase=PHASE_FINAL,
        completed_blocks=writer.completed_blocks,
        data_bytes=writer.data_bytes,
        data_sha256=writer.digest_hex,
        config_digest=config_digest,
        block_frames=layout.block_frames,
        total_blocks=layout.total_blocks,
        total_frames=layout.total_frames,
        channels=layout.channels,
        normalization=normalization,
    )


def _maybe_interrupt(
    options: ExportOptions,
    phase: str,
    processed_new_blocks: int,
    completed_blocks: int,
    total_blocks: int,
) -> None:
    if options.stop_after_blocks is None:
        return
    if options.stop_after_phase not in ("any", phase):
        return
    if (
        processed_new_blocks >= options.stop_after_blocks
        and completed_blocks < total_blocks
    ):
        raise SimulatedInterrupt(phase, completed_blocks)


def _scan_final_stream(path: Path, layout: BlockLayout) -> tuple[float, float]:
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        raise output_error("final PCM stream is unavailable", error=str(exc)) from None
    if actual_size != layout.final_bytes:
        raise output_error(
            "final PCM stream length is inconsistent",
            expected_bytes=layout.final_bytes,
            actual_bytes=actual_size,
        )
    sum_squares = 0.0
    peak = 0.0
    sample_count = 0
    with path.open("rb") as handle:
        for frame_count in layout.frame_counts:
            byte_count = frame_count * layout.channels * 2
            data = handle.read(byte_count)
            if len(data) != byte_count:
                raise output_error("final PCM stream ended unexpectedly")
            values = array("h")
            values.frombytes(data)
            if sys.byteorder == "big":
                values.byteswap()
            for value in values:
                normalized = float(value) / 32767.0
                sum_squares += normalized * normalized
                peak = max(peak, abs(normalized))
            sample_count += len(values)
    expected_samples = layout.total_frames * layout.channels
    if sample_count != expected_samples:
        raise output_error(
            "final sample count is inconsistent",
            expected=expected_samples,
            actual=sample_count,
        )
    rms = math.sqrt(sum_squares / sample_count) if sample_count else 0.0
    return rms, peak


def _finalize_wav(
    final_raw: Path,
    output: Path,
    temporary: Path,
    layout: BlockLayout,
    sample_rate: int,
    bit_depth: int,
) -> str:
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        actual_size = final_raw.stat().st_size
        if actual_size != layout.final_bytes:
            raise output_error(
                "final PCM stream length is inconsistent before WAV assembly",
                expected_bytes=layout.final_bytes,
                actual_bytes=actual_size,
            )
        raw_handle = final_raw.open("rb")
        try:
            with wave.open(str(temporary), "wb") as wav:
                wav.setnchannels(layout.channels)
                wav.setsampwidth(bit_depth // 8)
                wav.setframerate(sample_rate)
                remaining = layout.final_bytes
                while remaining:
                    chunk = raw_handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise output_error("final PCM stream ended during WAV assembly")
                    wav.writeframesraw(chunk)
                    remaining -= len(chunk)
        finally:
            raw_handle.close()
        with temporary.open("rb+") as handle:
            fsync_handle(handle)
        atomic_replace(temporary, output)
        with wave.open(str(output), "rb") as wav:
            if (
                wav.getnchannels() != layout.channels
                or wav.getsampwidth() != bit_depth // 8
                or wav.getframerate() != sample_rate
                or wav.getnframes() != layout.total_frames
                or wav.getnframes() * layout.channels * (bit_depth // 8)
                != layout.final_bytes
            ):
                raise output_error(
                    "final WAV header is inconsistent with configured/output PCM",
                    path=str(output),
                )
        if output.stat().st_size != 44 + layout.final_bytes:
            raise output_error(
                "final WAV byte length is inconsistent with PCM data",
                path=str(output),
                expected_bytes=44 + layout.final_bytes,
                actual_bytes=output.stat().st_size,
            )
    except AppError:
        _remove_if_exists(temporary)
        raise
    except (OSError, wave.Error) as exc:
        _remove_if_exists(temporary)
        raise output_error(
            "cannot finalize WAV output",
            path=str(output),
            error=str(exc),
        ) from None
    return sha256_file(output)


def _commentary_stats(prepared: PreparedProject) -> list[dict[str, Any]]:
    config = prepared.config
    rows: list[dict[str, Any]] = []
    for commentary in prepared.commentary:
        loaded = commentary.loaded
        active_frames, effective_frames = commentary_effective_frames(
            commentary,
            config.total_frames,
        )
        trimmed_frames = sum(
            interval.end_frame - interval.start_frame
            for interval in commentary.trim_intervals
        )
        trimmed_ms = trimmed_frames * 1000.0 / loaded.sample_rate
        rows.append(
            {
                "name": loaded.spec.name,
                "source_frames": loaded.frames,
                "source_samples": loaded.frames * loaded.channels,
                "active_output_frames": active_frames,
                "effective_frames": effective_frames,
                "effective_samples": effective_frames * config.output_channels,
                "trimmed_intervals": len(commentary.trim_intervals),
                "trimmed_source_frames": trimmed_frames,
                "trimmed_total_ms": trimmed_ms,
            }
        )
    return rows


def run_export(config_path: str, options: ExportOptions) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    loaded: LoadedProject = load_project(config)
    prepared = prepare_project(loaded)
    layout = BlockLayout.build(
        total_frames=config.total_frames,
        block_frames=config.mix.block_frames,
        channels=config.output_channels,
    )
    paths = artifact_paths(config.output_path)
    reset_reason: str | None = None
    state: CheckpointState | None = None

    try:
        state = read_checkpoint(
            paths.checkpoint,
            loaded.config_digest,
            layout,
            paths.measure_raw,
            paths.final_raw,
        )
        if state is not None and state.phase == PHASE_FINAL:
            assert state.normalization is not None
            if not _normalization_matches_config(state.normalization, config):
                raise CheckpointValidationError(
                    "normalization_mismatch",
                    "checkpoint normalization does not match configuration",
                )
    except CheckpointValidationError as exc:
        if options.strict_checkpoint:
            raise checkpoint_invalid(
                exc.detail,
                reason=exc.reason,
                **exc.context,
            ) from None
        reset_reason = f"{exc.reason}: {exc.detail}"
        _reset_artifacts(paths)
        state = None

    if state is None and any(
        path.exists()
        for path in (
            paths.checkpoint,
            paths.measure_raw,
            paths.final_raw,
            paths.temporary_wav,
        )
    ):
        _reset_artifacts(paths)

    resumed_phase: str | None = state.phase if state is not None else None
    resumed_blocks = state.completed_blocks if state is not None else 0
    measure_resumed_blocks = (
        state.completed_blocks
        if state is not None and state.phase == PHASE_MEASURE
        else 0
    )
    final_resumed_blocks = (
        state.completed_blocks
        if state is not None and state.phase == PHASE_FINAL
        else 0
    )

    if state is None or state.phase == PHASE_MEASURE:
        measure_completed = state.completed_blocks if state is not None else 0
        measure_digest = (
            state.data_sha256 if state is not None else sha256_bytes(b"")
        )
        processed_measure = 0
        with RawPcmStore(
            paths.measure_raw,
            layout,
            PHASE_MEASURE,
            measure_completed,
            measure_digest,
        ) as writer:
            if measure_completed == 0:
                write_checkpoint(
                    paths.checkpoint,
                    _make_measure_state(writer, loaded.config_digest, layout),
                )
            for block_index in range(measure_completed, layout.total_blocks):
                start_frame = block_index * layout.block_frames
                frame_count = layout.frame_counts[block_index]
                premix = render_premix_block(prepared, start_frame, frame_count)
                writer.append(float_array_to_le_bytes(premix))
                write_checkpoint(
                    paths.checkpoint,
                    _make_measure_state(writer, loaded.config_digest, layout),
                )
                processed_measure += 1
                _maybe_interrupt(
                    options,
                    PHASE_MEASURE,
                    processed_measure,
                    writer.completed_blocks,
                    layout.total_blocks,
                )

        measured_rms, measured_peak, soft_triggered = _scan_measure_stream(
            paths.measure_raw,
            layout,
            config.mix.soft_knee_ratio,
            config.mix.peak_limit_dbfs,
        )
        normalization = _derive_normalization(
            config,
            measured_rms,
            measured_peak,
            soft_triggered,
        )
        _remove_if_exists(paths.final_raw)
        final_initial = RawPcmStore(
            paths.final_raw,
            layout,
            PHASE_FINAL,
            0,
            sha256_bytes(b""),
        )
        final_initial.close()
        write_checkpoint(
            paths.checkpoint,
            _make_final_state(
                final_initial,
                loaded.config_digest,
                layout,
                normalization,
            ),
        )
        _remove_if_exists(paths.measure_raw)
        final_completed = 0
        final_digest = sha256_bytes(b"")
    else:
        assert state.normalization is not None
        normalization = dict(state.normalization)
        final_completed = state.completed_blocks
        final_digest = state.data_sha256
        _remove_if_exists(paths.measure_raw)

    processed_final = 0
    with RawPcmStore(
        paths.final_raw,
        layout,
        PHASE_FINAL,
        final_completed,
        final_digest,
    ) as final_writer:
        for block_index in range(final_completed, layout.total_blocks):
            start_frame = block_index * layout.block_frames
            frame_count = layout.frame_counts[block_index]
            premix = render_premix_block(prepared, start_frame, frame_count)
            quantized = quantize_with_gain(
                premix,
                db_to_lin(normalization["applied_gain_db"]),
            )
            final_writer.append(int16_array_to_le_bytes(quantized))
            write_checkpoint(
                paths.checkpoint,
                _make_final_state(
                    final_writer,
                    loaded.config_digest,
                    layout,
                    normalization,
                ),
            )
            processed_final += 1
            _maybe_interrupt(
                options,
                PHASE_FINAL,
                processed_final,
                final_writer.completed_blocks,
                layout.total_blocks,
            )

    final_rms, final_peak = _scan_final_stream(paths.final_raw, layout)
    output_sha256 = _finalize_wav(
        paths.final_raw,
        config.output_path,
        paths.temporary_wav,
        layout,
        config.sample_rate,
        config.bit_depth,
    )
    commentary = _commentary_stats(prepared)
    total_trimmed_ms = sum(
        row["trimmed_total_ms"] for row in commentary
    )
    elapsed_seconds = time.perf_counter() - started

    _remove_if_exists(paths.final_raw)
    _remove_if_exists(paths.checkpoint)
    _remove_if_exists(paths.measure_raw)

    public_normalization = dict(normalization)
    public_normalization["final_rms_dbfs"] = rms_to_db(final_rms)
    public_normalization["final_peak_dbfs"] = lin_to_db(final_peak, floor=1.0e-9)
    return {
        "ok": True,
        "output_path": config.output_path_value,
        "sample_rate": config.sample_rate,
        "channels": config.output_channels,
        "bit_depth": config.bit_depth,
        "duration_ms": config.duration_ms,
        "output_frames": config.total_frames,
        "output_samples": config.total_frames * config.output_channels,
        "output_duration_ms": (
            config.total_frames * 1000.0 / config.sample_rate
        ),
        "data_bytes": layout.final_bytes,
        "output_sha256": output_sha256,
        "commentary": commentary,
        "normalization": public_normalization,
        "applied_normalization_gain_db": normalization["applied_gain_db"],
        "peak_limit_triggered": normalization["peak_limit_triggered"],
        "soft_limit_triggered": normalization["soft_limit_triggered"],
        "any_peak_limiting_triggered": (
            normalization["peak_limit_triggered"]
            or normalization["soft_limit_triggered"]
        ),
        "total_trimmed_ms": total_trimmed_ms,
        "resumed_blocks": resumed_blocks,
        "resumed_phase": resumed_phase,
        "measure_resumed_blocks": measure_resumed_blocks,
        "final_resumed_blocks": final_resumed_blocks,
        "checkpoint_reset_reason": reset_reason,
        "config_digest": loaded.config_digest,
        "total_elapsed_ms": elapsed_seconds * 1000.0,
        "elapsed_seconds": elapsed_seconds,
    }
