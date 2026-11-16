from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .config import RenderConfig
from .errors import BusinessError, error_payload
from .model import load_timeline
from .render import render_project
from .samples import generate_samples


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise BusinessError(
            code="E_CLI_USAGE",
            message=message,
            path=self.prog,
        )


def _add_common_timeline(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--timeline",
        required=True,
        type=Path,
        help="Path to the cutscene timeline JSON.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="cutscene-dub",
        description=(
            "Deterministically generate dub samples, conform dialogue to subtitle "
            "windows, duck background music, and mix a stereo cutscene WAV."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    samples = subparsers.add_parser(
        "generate-samples",
        help="Generate deterministic 48 kHz/16-bit mono dialogue and BGM WAVs.",
    )
    _add_common_timeline(samples)
    samples.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Audio root. Defaults to <timeline-dir>/samples.",
    )
    samples.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Integer random seed; identical seed and timeline produce identical WAV bytes.",
    )
    samples.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when persisted state and hashes are complete.",
    )

    render = subparsers.add_parser(
        "render",
        help="Conform dialogue, duck music, mix stereo WAV, and write a JSON report.",
    )
    _add_common_timeline(render)
    render.add_argument(
        "-a",
        "--audio-root",
        type=Path,
        default=None,
        help="Root containing <target-language>/<dialogue-id>.wav. Default <timeline-dir>/samples.",
    )
    render.add_argument(
        "--bgm",
        type=Path,
        default=None,
        help="Background music WAV. Default <audio-root>/bgm.wav.",
    )
    render.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Stereo output WAV. Default <timeline-dir>/build/cutscene.wav.",
    )
    render.add_argument(
        "-r",
        "--report",
        type=Path,
        default=None,
        help="JSON alignment report. Default alongside the output WAV.",
    )
    render.add_argument(
        "--silence-threshold-dbfs",
        type=float,
        default=-45.0,
        help="Peak threshold used to trim leading/trailing silence (default -45).",
    )
    render.add_argument(
        "--silence-margin-ms",
        type=float,
        default=5.0,
        help="Silence retained around detected speech after trimming (default 5 ms).",
    )
    render.add_argument(
        "--max-speedup",
        type=float,
        default=1.15,
        help="Maximum atempo speed-up ratio; entries above it are unadaptable (default 1.15).",
    )
    render.add_argument(
        "--duck-depth-db",
        type=float,
        default=16.0,
        help="Positive music attenuation during dialogue in dB (default 16).",
    )
    render.add_argument(
        "--duck-fade-in-ms",
        type=float,
        default=120.0,
        help="Music attenuation ramp-in duration in ms (default 120).",
    )
    render.add_argument(
        "--duck-fade-out-ms",
        type=float,
        default=120.0,
        help="Music gain recovery duration in ms (default 120).",
    )
    render.add_argument(
        "--max-gain-jump-db",
        type=float,
        default=0.5,
        help="Maximum allowed per-sample gain change in the ducking envelope (default 0.5 dB).",
    )
    render.add_argument(
        "--peak-ceiling-dbfs",
        type=float,
        default=-1.0,
        help="Output peak ceiling; values above -1 are rejected (default -1).",
    )
    render.add_argument(
        "--dialogue-gain-db",
        type=float,
        default=0.0,
        help="Pre-mix dialogue gain in dB (default 0).",
    )
    render.add_argument(
        "--music-gain-db",
        type=float,
        default=0.0,
        help="Pre-ducking music gain in dB (default 0).",
    )
    render.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable name or path (default ffmpeg).",
    )
    render.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when persisted state and hashes are complete.",
    )
    return parser


def _emit_json(value: dict[str, Any], *, stream: Any) -> None:
    stream.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    stream.flush()


def _emit_error(command: str, exc: BusinessError) -> None:
    payload = error_payload(command, exc)
    try:
        _emit_json(payload, stream=sys.stderr)
    except (TypeError, ValueError):
        fallback = {
            "ok": False,
            "command": command,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "entry_id": exc.entry_id,
            },
        }
        _emit_json(fallback, stream=sys.stderr)


def _run_render(args: argparse.Namespace) -> dict[str, Any]:
    timeline = load_timeline(args.timeline)
    audio_root = args.audio_root or (timeline.source_path.parent / "samples")
    bgm = args.bgm or (audio_root / "bgm.wav")
    output = args.output or (timeline.source_path.parent / "build" / "cutscene.wav")
    report = args.report or (output.parent / "alignment_report.json")
    config = RenderConfig(
        silence_threshold_dbfs=args.silence_threshold_dbfs,
        silence_margin_ms=args.silence_margin_ms,
        max_speedup=args.max_speedup,
        duck_depth_db=args.duck_depth_db,
        duck_fade_in_ms=args.duck_fade_in_ms,
        duck_fade_out_ms=args.duck_fade_out_ms,
        max_gain_jump_db=args.max_gain_jump_db,
        peak_ceiling_dbfs=args.peak_ceiling_dbfs,
        dialogue_gain_db=args.dialogue_gain_db,
        music_gain_db=args.music_gain_db,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    return render_project(
        timeline,
        audio_root=audio_root,
        bgm_path=bgm,
        output_path=output,
        report_path=report,
        config=config,
        force=args.force,
    )


def _run_generate(args: argparse.Namespace) -> dict[str, Any]:
    timeline = load_timeline(args.timeline)
    root = args.output_dir or (timeline.source_path.parent / "samples")
    return generate_samples(
        timeline,
        audio_root=root,
        seed=args.seed,
        force=args.force,
    )


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    command = "unknown"
    try:
        parser = build_parser()
        args = parser.parse_args(argv)
        command = args.command
        if args.command == "generate-samples":
            result = _run_generate(args)
        elif args.command == "render":
            result = _run_render(args)
        else:
            raise BusinessError(
                code="E_CLI_USAGE",
                message=f"Unsupported command: {args.command}",
            )
        _emit_json(result, stream=sys.stdout)
        return 0
    except KeyboardInterrupt:
        exc = BusinessError(
            code="E_INTERRUPTED",
            message=(
                "Interrupted. Temporary .part artifacts and persisted state are "
                "incomplete; rerun without --force to recover."
            ),
        )
        _emit_error(command, exc)
        return exc.status
    except BusinessError as exc:
        if os.environ.get("CUTSCENE_DUB_DEBUG"):
            traceback.print_exc()
        _emit_error(command, exc)
        return exc.status
    except SystemExit:
        raise
    except Exception as exc:
        if os.environ.get("CUTSCENE_DUB_DEBUG"):
            traceback.print_exc()
        wrapped = BusinessError(
            code="E_INTERNAL",
            message=f"Unexpected failure: {type(exc).__name__}: {exc}",
        )
        _emit_error(command, wrapped)
        return wrapped.status

