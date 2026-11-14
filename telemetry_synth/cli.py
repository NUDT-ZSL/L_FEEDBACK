"""Command-line entry point for the deterministic synthesis pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tracemalloc
import uuid
from pathlib import Path

from . import BITS_PER_SAMPLE, CHANNELS, SAMPLE_RATE, __version__
from .errors import SynthError
from .schema import ParamsModel, TelemetryModel, load_params, load_telemetry
from .synth import AudioSynthesizer
from .util import make_temp_file, remove_quietly, sha256_file
from .wav_io import validate_pcm16_wav, write_pcm16_wav


class BusinessArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise SynthError("E_USAGE", message)


def _seed_value(text: str) -> int:
    try:
        value = int(text, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be an integer") from exc
    if value < 0 or value > (1 << 64) - 1:
        raise argparse.ArgumentTypeError("seed must be in [0, 2^64-1]")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = BusinessArgumentParser(
        prog="python -m telemetry_synth.cli",
        description="Synthesize deterministic stereo racing audio from telemetry.",
    )
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--params", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--seed", required=True, type=_seed_value)
    parser.add_argument("--block-size", required=True, type=_positive_int)
    return parser


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _process_peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return peak
        return peak * 1024
    except (ImportError, OSError, ValueError):
        return None


def estimate_peak_memory_bytes(
    telemetry: TelemetryModel,
    params: ParamsModel,
    block_size: int,
) -> int:
    """Conservative model of the largest streaming working set."""

    harmonic_count = len(params.engine.harmonics)
    # Engine phase trajectory (n*H*2 doubles), waveform and control arrays,
    # tire filter/noise arrays, event buffers, and PCM conversion scratch.
    block_bytes = block_size * (
        384
        + 160 * harmonic_count
        + 128
    )
    telemetry_bytes = telemetry.frame_count * (8 * 4 + 48)
    event_bytes = len(telemetry.events) * 256
    return int(2_000_000 + block_bytes + telemetry_bytes + event_bytes)


def _write_report(path: Path, report: dict[str, object]) -> None:
    try:
        text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    except OSError as exc:
        raise SynthError("E_REPORT_IO", f"cannot write report {path}: {exc}") from exc


def _finalize_outputs(
    temp_wav: Path,
    temp_report: Path,
    output_wav: Path,
    output_report: Path,
) -> None:
    """Replace both targets only after both temporary files are complete."""

    backups: dict[Path, Path] = {}
    placed: list[Path] = []
    try:
        for target in (output_wav, output_report):
            if target.exists():
                backup = target.with_name(
                    f".{target.name}.bak.{os.getpid()}.{uuid.uuid4().hex}"
                )
                os.replace(target, backup)
                backups[target] = backup
        os.replace(temp_wav, output_wav)
        placed.append(output_wav)
        os.replace(temp_report, output_report)
        placed.append(output_report)
    except OSError as exc:
        for path in placed:
            remove_quietly(path)
        for target, backup in backups.items():
            try:
                os.replace(backup, target)
            except OSError:
                pass
        raise SynthError(
            "E_OUTPUT_IO",
            f"failed to publish output files: {exc}",
        ) from exc
    else:
        for backup in backups.values():
            remove_quietly(backup)


def _build_report(
    *,
    telemetry_path: Path,
    params_path: Path,
    output_path: Path,
    report_path: Path,
    telemetry: TelemetryModel,
    params: ParamsModel,
    seed: int,
    block_size: int,
    synth: AudioSynthesizer,
    written_samples: int,
    wav_details: dict[str, int | str],
    wav_sha256: str,
    elapsed_seconds: float,
    peak_memory_estimate_bytes: int,
    tracemalloc_peak_bytes: int,
    process_peak_rss_bytes: int | None,
) -> dict[str, object]:
    stats = synth.stats
    if written_samples != telemetry.total_samples:
        raise SynthError(
            "E_ALIGNMENT",
            f"wrote {written_samples} samples; telemetry expects {telemetry.total_samples}",
        )
    return {
        "schema_version": 1,
        "status": "ok",
        "generator": {
            "name": "telemetry_synth",
            "version": __version__,
        },
        "inputs": {
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": sha256_file(telemetry_path),
            "params_path": str(params_path),
            "params_sha256": sha256_file(params_path),
            "model_id": params.model_id,
            "seed": seed,
            "block_size": block_size,
        },
        "outputs": {
            "wav_path": str(output_path),
            "report_path": str(report_path),
            "wav_sha256": wav_sha256,
        },
        "timeline": {
            "frame_interval_ms": telemetry.frame_interval_ms,
            "frame_count": telemetry.frame_count,
            "samples_per_frame": telemetry.samples_per_frame,
            "duration_seconds": telemetry.frame_count
            * telemetry.frame_interval_s,
            "sample_rate": SAMPLE_RATE,
            "channels": CHANNELS,
            "bits_per_sample": BITS_PER_SAMPLE,
            "expected_sample_count": telemetry.total_samples,
            "written_sample_count": written_samples,
            "sample_count_match": written_samples == telemetry.total_samples,
        },
        "wav_header": wav_details,
        "bus_peaks": {
            key: float(value) for key, value in stats.bus_peaks.items()
        },
        "mix": {
            "pre_limit_peak": float(stats.pre_limit_peak),
            "post_limit_peak": float(stats.post_limit_peak),
            "soft_limit_threshold": params.mix.soft_limit_threshold,
            "master_gain": params.mix.master_gain,
            "bus_gains": {
                "engine": params.mix.engine_gain,
                "tire": params.mix.tire_gain,
                "events": params.mix.event_gain,
            },
        },
        "events": {
            "trigger_count": stats.event_trigger_count,
            "attenuated_count": stats.events_attenuated,
            "counts_by_type": stats.event_counts_by_type,
            "engine_shift_count": stats.engine_shift_count,
            "tire_state_switch_count": stats.tire_state_switch_count,
        },
        "memory": {
            "peak_memory_estimate_bytes": peak_memory_estimate_bytes,
            "tracemalloc_peak_bytes": tracemalloc_peak_bytes,
            "process_peak_rss_bytes": process_peak_rss_bytes,
            "method": "modeled_streaming_working_set",
        },
        "timing": {
            "elapsed_seconds": elapsed_seconds,
        },
    }


def run(args: argparse.Namespace) -> int:
    telemetry_path: Path = args.telemetry
    params_path: Path = args.params
    output_path: Path = args.output
    report_path: Path = (
        args.report
        if args.report is not None
        else output_path.with_suffix(".report.json")
    )

    if _same_path(output_path, report_path):
        raise SynthError("E_USAGE", "output WAV and report paths must differ")
    for protected in (telemetry_path, params_path):
        if _same_path(output_path, protected):
            raise SynthError("E_USAGE", "output WAV must not overwrite an input file")
        if _same_path(report_path, protected):
            raise SynthError("E_USAGE", "report must not overwrite an input file")

    face_start = time.perf_counter()
    tracemalloc.start()
    params = load_params(params_path)
    telemetry = load_telemetry(telemetry_path, params)
    synth = AudioSynthesizer(
        telemetry=telemetry,
        params=params,
        seed=args.seed,
        block_size=args.block_size,
    )

    temp_wav: Path | None = None
    temp_report: Path | None = None
    try:
        temp_wav = make_temp_file(output_path, "wav")
        temp_report = make_temp_file(report_path, "report")
        written = write_pcm16_wav(
            temp_wav,
            telemetry.total_samples,
            synth.iter_float_blocks(),
        )
        if written != telemetry.total_samples:
            raise SynthError(
                "E_ALIGNMENT",
                f"wrote {written} samples; expected {telemetry.total_samples}",
            )
        wav_details = validate_pcm16_wav(temp_wav, telemetry.total_samples)
        elapsed_seconds = time.perf_counter() - face_start
        peak_estimate = estimate_peak_memory_bytes(
            telemetry,
            params,
            args.block_size,
        )
        traced_current, traced_peak = tracemalloc.get_traced_memory()
        del traced_current
        process_peak = _process_peak_rss_bytes()
        wav_sha256 = sha256_file(temp_wav)
        report = _build_report(
            telemetry_path=telemetry_path,
            params_path=params_path,
            output_path=output_path,
            report_path=report_path,
            telemetry=telemetry,
            params=params,
            seed=args.seed,
            block_size=args.block_size,
            synth=synth,
            written_samples=written,
            wav_details=wav_details,
            wav_sha256=wav_sha256,
            elapsed_seconds=elapsed_seconds,
            peak_memory_estimate_bytes=peak_estimate,
            tracemalloc_peak_bytes=int(traced_peak),
            process_peak_rss_bytes=process_peak,
        )
        _write_report(temp_report, report)
        json.loads(temp_report.read_text(encoding="utf-8"))
        _finalize_outputs(
            temp_wav,
            temp_report,
            output_path,
            report_path,
        )
        temp_wav = None
        temp_report = None
    finally:
        remove_quietly(temp_wav)
        remove_quietly(temp_report)
        tracemalloc.stop()

    print(
        "peak_memory_estimate_bytes="
        f"{peak_estimate} "
        f"tracemalloc_peak_bytes={int(traced_peak)} "
        f"process_peak_rss_bytes={process_peak} "
        "method=modeled_streaming_working_set",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        return run(args)
    except SynthError as exc:
        print(
            f"error_code={exc.code} exit_code={exc.exit_code} message={exc.message}",
            file=sys.stderr,
        )
        return int(exc.exit_code)
    except KeyboardInterrupt:
        print(
            "error_code=E_INTERNAL exit_code=99 message=interrupted",
            file=sys.stderr,
        )
        return 99
    except Exception as exc:
        print(
            "error_code=E_INTERNAL exit_code=99 "
            f"message={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 99


if __name__ == "__main__":
    raise SystemExit(main())

