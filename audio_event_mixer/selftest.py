"""End-to-end acceptance checks with deterministic local fixtures."""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any, Union

import numpy as np

from .dsp import panning_gains
from .errors import BusinessError, ErrorCode
from .generator import generate_demo_dataset
from .processor import process_events
from .wavio import read_wav, write_pcm_wav


def _record(
    checks: list[dict], name: str, passed: bool, details: Any = None
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "details": details,
        }
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _error_case(
    checks: list[dict],
    name: str,
    root: Path,
    expected_code: str,
    events_payload: dict,
    output_path: Path,
    expected_event_index: Any = 0,
) -> None:
    events_path = root / f"{name}.json"
    report_path = root / f"{name}.report.json"
    _write_json(events_path, events_payload)
    observed_code = None
    observed_index = None
    report_payload = None
    try:
        process_events(events_path, output_path, report_path)
    except BusinessError as exc:
        observed_code = exc.code
        observed_index = exc.event_index
    if report_path.is_file():
        try:
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report_payload = None
    passed = (
        observed_code == expected_code
        and observed_index == expected_event_index
        and isinstance(report_payload, dict)
        and report_payload.get("error_code") == expected_code
        and report_payload.get("failure_event_index") == expected_event_index
    )
    _record(
        checks,
        name,
        passed,
        {
            "expected_code": expected_code,
            "observed_code": observed_code,
            "expected_event_index": expected_event_index,
            "observed_event_index": observed_index,
            "report_error_code": (
                report_payload.get("error_code")
                if isinstance(report_payload, dict)
                else None
            ),
        },
    )


def run_self_test(out_dir: Union[str, Path] = "validation") -> dict:
    """Run deterministic acceptance checks and write a JSON result."""

    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []
    result: dict = {
        "schema_version": "1.0",
        "status": "failed",
        "root": str(root),
        "checks": checks,
        "report_path": str(root / "selftest_report.json"),
    }

    try:
        manifest = generate_demo_dataset(root)
        events_path = Path(manifest["events_path"])
        output_dir = root / "output"
        mix_path = output_dir / "mix.wav"
        processing_report_path = output_dir / "mix.report.json"
        report = process_events(
            events_path, mix_path, processing_report_path
        )
        decoded = read_wav(mix_path)

        supported_files = [item for item in manifest["files"] if item["supported"]]
        encodings = {item["encoding"] for item in supported_files}
        _record(
            checks,
            "multi_format_dataset",
            len(supported_files) == 4
            and {
                "pcm_s16le",
                "pcm_s24le",
                "pcm_s32le",
                "float_s32le",
            }.issubset(encodings)
            and any(item["supported"] is False for item in manifest["files"]),
            {
                "supported": sorted(encodings),
                "unsupported_fixture": [
                    item["path"]
                    for item in manifest["files"]
                    if not item["supported"]
                ],
            },
        )

        output_info = decoded.info
        _record(
            checks,
            "output_wav_format",
            output_info.sample_rate == 48000
            and output_info.channels == 2
            and output_info.bits_per_sample == 16
            and decoded.samples.shape == (report["timeline_frames"], 2),
            output_info.as_dict(),
        )

        parsed_report = json.loads(
            processing_report_path.read_text(encoding="utf-8")
        )
        _record(
            checks,
            "report_json_parseable",
            parsed_report.get("status") == "ok"
            and len(parsed_report.get("events", [])) == 4
            and parsed_report.get("output_path") == str(mix_path.resolve()),
            {
                "status": parsed_report.get("status"),
                "event_count": len(parsed_report.get("events", [])),
                "output_path": parsed_report.get("output_path"),
            },
        )

        timing_ok = True
        max_timing_error = 0.0
        for event in report["events"]:
            expected = int(
                math.floor(event["trigger_seconds"] * 48000 + 0.5)
            )
            timing_ok &= event["trigger_sample"] == expected
            max_timing_error = max(
                max_timing_error, abs(event["timing_error_samples"])
            )
        _record(
            checks,
            "sample_timeline_rounding",
            timing_ok and max_timing_error <= 0.5000001,
            {
                "max_abs_error_samples": max_timing_error,
                "max_allowed_samples": 1,
            },
        )

        duration_ok = True
        max_duration_error = 0.0
        for event in report["events"]:
            source_duration = event["source_duration_seconds"]
            output_duration = event["duration_seconds"]
            error = abs(output_duration - source_duration)
            max_duration_error = max(max_duration_error, error)
            duration_ok &= error <= 0.001
        _record(
            checks,
            "resampling_duration",
            duration_ok,
            {
                "max_abs_error_seconds": max_duration_error,
                "max_allowed_seconds": 0.001,
            },
        )

        loop_ok = True
        loop_details = []
        boundary_after_max = 0.0
        for event in report["events"]:
            if event["loop_cleanup"] is None:
                continue
            loop = event["loop_cleanup"]
            loop_frames = loop["end_sample"] - loop["start_sample"]
            expected_fade = min(
                math.floor(loop_frames * 0.10),
                math.floor(48000 * 0.005),
            )
            boundary_after = loop["boundary_jump_after"]
            boundary_after_max = max(boundary_after_max, boundary_after)
            loop_ok &= (
                loop["crossfade_samples"] == expected_fade
                and loop["frame_count_preserved"] is True
                and boundary_after <= 1e-9
            )
            loop_details.append(
                {
                    "id": event["id"],
                    "start_sample": loop["start_sample"],
                    "end_sample": loop["end_sample"],
                    "crossfade_samples": loop["crossfade_samples"],
                    "expected_crossfade_samples": expected_fade,
                    "boundary_jump_after": boundary_after,
                }
            )
        _record(
            checks,
            "loop_crossfade",
            loop_ok and len(loop_details) == 2,
            {
                "loops": loop_details,
                "max_boundary_jump_after": boundary_after_max,
            },
        )

        left_gain, right_gain = panning_gains(
            1000, 48000, 3.7, 0.0, start_frame=123
        )
        equal_center = bool(
            np.allclose(left_gain, right_gain)
            and np.allclose(left_gain, math.sqrt(0.5))
        )
        lfo_left, lfo_right = panning_gains(1000, 48000, 1.3, 0.8)
        constant_power = bool(
            np.max(np.abs(lfo_left**2 + lfo_right**2 - 1.0)) <= 1e-12
        )
        _record(
            checks,
            "equal_power_panning",
            equal_center and constant_power,
            {
                "depth_zero_center_gain": float(left_gain[0]),
                "depth_zero_left_equals_right": equal_center,
                "max_constant_power_error": float(
                    np.max(np.abs(lfo_left**2 + lfo_right**2 - 1.0))
                ),
            },
        )

        peak = float(np.max(np.abs(decoded.samples)))
        dc = np.mean(decoded.samples, axis=0)
        _record(
            checks,
            "output_level_and_dc",
            peak <= 1.0
            and float(np.max(np.abs(dc))) <= 0.001
            and np.all(np.isfinite(decoded.samples)),
            {
                "peak": peak,
                "dc_left": float(dc[0]),
                "dc_right": float(dc[1]),
                "dc_limit": 0.001,
            },
        )

        mix24_path = output_dir / "mix_24bit.wav"
        report24 = process_events(
            events_path,
            mix24_path,
            output_dir / "mix_24bit.report.json",
            output_bit_depth=24,
        )
        info24 = read_wav(mix24_path).info
        _record(
            checks,
            "24bit_output",
            info24.bits_per_sample == 24
            and info24.encoding == "pcm_s24le"
            and read_wav(mix24_path).samples.shape
            == (report24["timeline_frames"], 2),
            info24.as_dict(),
        )

        errors_root = root / "errors"
        errors_root.mkdir(parents=True, exist_ok=True)
        unsupported_path = root / "samples" / "source_unsupported_mulaw.wav"
        _error_case(
            checks,
            "unsupported_format_error",
            errors_root,
            ErrorCode.UNSUPPORTED_FORMAT,
            {
                "events": [
                    {
                        "file": str(unsupported_path),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 0.5,
                    }
                ]
            },
            errors_root / "unsupported.wav",
        )
        _error_case(
            checks,
            "missing_file_error",
            errors_root,
            ErrorCode.FILE_NOT_FOUND,
            {
                "events": [
                    {
                        "file": str(root / "missing.wav"),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 0.5,
                    }
                ]
            },
            errors_root / "missing.wav",
        )
        _error_case(
            checks,
            "invalid_event_error",
            errors_root,
            ErrorCode.INVALID_EVENT,
            {
                "events": [
                    {
                        "file": str(root / "samples" / "source_44k_s16_mono.wav"),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 1.5,
                    }
                ]
            },
            errors_root / "invalid.wav",
        )
        _error_case(
            checks,
            "invalid_loop_range_error",
            errors_root,
            ErrorCode.INVALID_EVENT,
            {
                "events": [
                    {
                        "file": str(root / "samples" / "source_44k_s16_mono.wav"),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 0.5,
                        "loop": {"start": 0.60, "end": 0.90},
                    }
                ]
            },
            errors_root / "bad_loop.wav",
        )

        denied_parent = errors_root / "not_a_directory"
        denied_parent.write_text("sentinel", encoding="utf-8")
        _error_case(
            checks,
            "output_denied_error",
            errors_root,
            ErrorCode.OUTPUT_DENIED,
            {
                "events": [
                    {
                        "file": str(root / "samples" / "source_44k_s16_mono.wav"),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 0.5,
                    }
                ]
            },
            denied_parent / "denied.wav",
            expected_event_index=None,
        )

        sentinel_path = errors_root / "sentinel.wav"
        sentinel = b"existing-output-must-survive"
        sentinel_path.write_bytes(sentinel)
        atomic_events = errors_root / "atomic_events.json"
        _write_json(
            atomic_events,
            {
                "events": [
                    {
                        "file": str(root / "samples" / "source_44k_s16_mono.wav"),
                        "time": 0.0,
                        "pan_frequency_hz": 1.0,
                        "pan_depth": 9.0,
                    }
                ]
            },
        )
        try:
            process_events(atomic_events, sentinel_path, errors_root / "atomic.report.json")
        except BusinessError:
            pass
        _record(
            checks,
            "atomic_output_preserved",
            sentinel_path.read_bytes() == sentinel,
            {"bytes_unchanged": sentinel_path.read_bytes() == sentinel},
        )

        spill_source = root / "samples" / "memmap_source.wav"
        spill_frames = 5 * 48000
        spill_t = np.arange(spill_frames, dtype=np.float64) / 48000.0
        write_pcm_wav(
            spill_source,
            0.10 * np.sin(2.0 * math.pi * 440.0 * spill_t),
            sample_rate=48000,
            bit_depth=16,
        )
        spill_events = root / "memmap_events.json"
        _write_json(
            spill_events,
            {
                "events": [
                    {
                        "file": str(spill_source),
                        "time": 0.0,
                        "pan_frequency_hz": 0.5,
                        "pan_depth": 0.4,
                    }
                ]
            },
        )
        spill_report = process_events(
            spill_events,
            root / "output" / "memmap_mix.wav",
            root / "output" / "memmap_mix.report.json",
            memory_limit_bytes=8 * 1024 * 1024,
        )
        _record(
            checks,
            "memmap_spill_path",
            spill_report["processing"]["mix_storage"] == "memmap"
            and spill_report["timeline_frames"] == spill_frames,
            {
                "mix_storage": spill_report["processing"]["mix_storage"],
                "timeline_frames": spill_report["timeline_frames"],
                "memory_limit_bytes": 8 * 1024 * 1024,
            },
        )

        result["manifest"] = {
            "events_path": manifest["events_path"],
            "files": manifest["files"],
        }
        result["primary_output"] = str(mix_path)
        result["primary_report"] = str(processing_report_path)
        result["summary"] = {
            "events": report["event_count"],
            "timeline_frames": report["timeline_frames"],
            "sample_rate": report["output_sample_rate"],
            "bit_depth": report["output_bit_depth"],
            "peak_before_soft_clip": report["processing"][
                "peak_before_soft_clip"
            ],
            "peak_after_soft_clip": report["processing"][
                "peak_after_soft_clip"
            ],
        }
    except Exception:
        _record(
            checks,
            "unexpected_exception",
            False,
            {"traceback": traceback.format_exc()},
        )

    result["status"] = (
        "passed"
        if checks and all(check["passed"] for check in checks)
        else "failed"
    )
    _write_json(Path(result["report_path"]), result)
    return result
