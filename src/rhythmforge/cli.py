"""Command line interface for rhythmforge.

Subcommands:

* ``generate`` - create deterministic sample WAV files and a demo chart,
* ``build``    - render master/stems/markers/manifest into an output folder,
* ``verify``   - run internal consistency checks and optional ffmpeg checks.

All failures print a single JSON document on stderr and return the matching
:class:`~rhythmforge.errors.ExitCode`; success prints JSON on stdout and 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from .assembly import assemble_sections
from .errors import BuildError, ExitCode
from .ffmpeg_check import ffmpeg_available, validate_wav
from .generator import generate_project
from .hashing import sha256_file
from .locking import OutputLock
from .pipeline import build
from .renderer import parse_layers
from .timing import load_chart
from .waveio import read_wav


def _emit_error(error: BuildError) -> int:
    print(error.to_json(), file=sys.stderr)
    return error.code


def _cmd_generate(args: argparse.Namespace) -> int:
    chart = generate_project(args.samples_dir, args.chart)
    payload = {
        "ok": True,
        "chart": os.path.abspath(args.chart),
        "samples_dir": os.path.abspath(args.samples_dir),
        "layer_count": len(chart["layers"]),
        "source_hashes": {
            layer["file"]: sha256_file(os.path.join(args.samples_dir, layer["file"]))
            for layer in chart["layers"]
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return ExitCode.OK


def _cmd_build(args: argparse.Namespace) -> int:
    manifest = build(args.chart, args.samples_dir, args.output, force=args.force)
    print(json.dumps({"ok": True, "manifest": manifest}, ensure_ascii=False,
                     indent=2, sort_keys=True))
    return ExitCode.OK


def verify_build(chart_path: str, samples_dir: str, output_dir: str) -> dict:
    """Programmatic verification used by both the CLI and the test suite."""
    chart, time_map, signatures, sections = load_chart(chart_path)
    layers = parse_layers(chart, samples_dir)
    layer_order = [layer.layer_id for layer in layers]
    enabled = {layer.layer_id for layer in layers if layer.enabled}

    master, master_info = read_wav(os.path.join(output_dir, "master.wav"))
    if master_info.sample_rate != time_map.sample_rate or master_info.channels != 2:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "master format does not match internal representation",
                         {"sample_rate": master_info.sample_rate,
                          "channels": master_info.channels})

    stem_sum = np.zeros_like(master, dtype=np.float64)
    stem_details = {}
    for layer_id in layer_order:
        stem, info = read_wav(os.path.join(output_dir, "stems", f"{layer_id}.wav"))
        if info.frames != master_info.frames:
            raise BuildError(ExitCode.DURATION_MISMATCH,
                             f"stem {layer_id} length differs from master",
                             {"stem_frames": info.frames,
                              "master_frames": master_info.frames})
        stem_details[layer_id] = info.frames
        if layer_id in enabled:
            stem_sum += stem.astype(np.float64)

    diff = np.abs(stem_sum - master.astype(np.float64))
    max_error = float(diff.max())
    tolerance = 2e-4
    if max_error > tolerance:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "master is not the linear sum of enabled stems",
                         {"max_abs_error": max_error, "tolerance": tolerance})

    peak = float(np.max(np.abs(master)))
    peak_limit = float(10.0 ** (float(chart["true_peak_dbfs"]) / 20.0))
    # Quantized 24-bit output may sit one LSB above the floating limit.
    if peak > peak_limit + 1e-6:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "master exceeds declared peak protection",
                         {"peak": peak, "limit": peak_limit})

    with open(os.path.join(output_dir, "markers.json"), "r", encoding="utf-8") as handle:
        marker_payload = json.load(handle)
    if marker_payload["total_frames"] != master_info.frames:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "marker total_frames differs from master frames")
    expected_frames = sum(time_map.section_frames(section.start_beat, section.end_beat)
                          for section in sections)
    if expected_frames != master_info.frames:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "master length differs from section-declared duration",
                         {"expected_frames": expected_frames,
                          "actual_frames": master_info.frames})

    section_markers = [item for item in marker_payload["markers"]
                       if item["type"] == "section"]
    bpm_markers = [item for item in marker_payload["markers"]
                   if item["type"] == "bpm"]
    if len(section_markers) != len(sections) + 1:
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "section marker count mismatch",
                         {"expected": len(sections) + 1,
                          "actual": len(section_markers)})
    if len(bpm_markers) != len(time_map.beats):
        raise BuildError(ExitCode.DURATION_MISMATCH,
                         "bpm marker count mismatch")

    ffmpeg_results = []
    if ffmpeg_available():
        ffmpeg_results.append(validate_wav(os.path.join(output_dir, "master.wav")))
        ffmpeg_results.append(
            validate_wav(os.path.join(output_dir, "stems", f"{layer_order[0]}.wav")))

    return {
        "ok": True,
        "frames": master_info.frames,
        "expected_frames": expected_frames,
        "stem_max_abs_error": max_error,
        "tolerance": tolerance,
        "sample_peak": peak,
        "peak_limit": peak_limit,
        "section_marker_count": len(section_markers),
        "bpm_marker_count": len(bpm_markers),
        "ffmpeg": ffmpeg_results,
        "stem_frames": stem_details,
    }


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_build(args.chart, args.samples_dir, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return ExitCode.OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rhythmforge",
        description="Deterministic offline rhythm-game audio track builder")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic samples and chart")
    gen.add_argument("--samples-dir", required=True)
    gen.add_argument("--chart", required=True)
    gen.set_defaults(handler=_cmd_generate)

    bld = sub.add_parser("build", help="build master, stems and markers")
    bld.add_argument("--chart", required=True)
    bld.add_argument("--samples-dir", required=True)
    bld.add_argument("--output", required=True)
    bld.add_argument("--force", action="store_true",
                     help="ignore incremental cache and rebuild every section")
    bld.set_defaults(handler=_cmd_build)

    ver = sub.add_parser("verify", help="verify a completed build")
    ver.add_argument("--chart", required=True)
    ver.add_argument("--samples-dir", required=True)
    ver.add_argument("--output", required=True)
    ver.set_defaults(handler=_cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return args.handler(args)
    except BuildError as error:
        return _emit_error(error)
    except FileNotFoundError as error:
        return _emit_error(BuildError(ExitCode.MISSING_SOURCE, str(error)))
    except argparse.ArgumentError as error:
        return _emit_error(BuildError(ExitCode.ARGUMENT_ERROR, str(error)))


if __name__ == "__main__":
    raise SystemExit(main())
