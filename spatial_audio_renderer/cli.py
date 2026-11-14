"""Command-line interface for deterministic offline spatial audio rendering."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from typing import Any, Dict, Optional, Sequence

from .errors import BusinessError, report_error
from .paths import ensure_output_dir, ensure_snapshot_dir, normalize_path
from .renderer import OfflineRenderer, RenderOptions, output_filenames
from .scene import load_scene


class BusinessArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise BusinessError(
            "INVALID_ARGUMENT",
            message,
            {"usage": self.format_usage().strip()},
        )


def build_parser() -> argparse.ArgumentParser:
    parser = BusinessArgumentParser(
        prog="spatial-audio-renderer",
        description="Deterministic offline spatial-audio renderer with crash-resumable snapshots.",
    )
    parser.add_argument("--scene", "--scene-file", required=True,
                        help="Path to the scene JSON file")
    parser.add_argument("--output-dir", "--output", required=True,
                        help="Directory for mix.wav, dry tracks, and manifest")
    parser.add_argument("--snapshot-dir", "--snapshots", required=True,
                        help="Directory for versioned render snapshots")
    parser.add_argument("--snapshot-interval", type=int, default=30,
                        help="Write a snapshot every N render frames (default: 30)")
    parser.add_argument("--frame-size", type=int, default=None,
                        help="Override the scene audio block size in samples")
    parser.add_argument("--max-snapshots", type=int, default=8,
                        help="Retain at most this many snapshot manifests")
    parser.add_argument("--resume", action="store_true",
                        help="Require restoration from the latest compatible snapshot")
    parser.add_argument(
        "--simulate-crash-after-frames",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _install_signal_handlers(renderer: OfflineRenderer):
    previous = {}
    if threading.current_thread() is not threading.main_thread():
        return previous

    def request_stop(signum: int, frame: Any) -> None:
        renderer.request_interrupt = True

    for signal_name in ("SIGINT", "SIGTERM"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is None:
            continue
        try:
            previous[signal_value] = signal.getsignal(signal_value)
            signal.signal(signal_value, request_stop)
        except (OSError, RuntimeError, ValueError):
            pass
    return previous


def _restore_signal_handlers(previous: Dict[int, Any]) -> None:
    for signal_value, handler in previous.items():
        try:
            signal.signal(signal_value, handler)
        except (OSError, RuntimeError, ValueError):
            pass


def _success_payload(result: Any) -> Dict[str, Any]:
    return {
        "status": "complete",
        "exit_code": 0,
        "mix_file": str(result.mix_path),
        "dry_track_files": [str(path) for path in result.dry_track_paths],
        "manifest_file": str(result.manifest_path),
        "resumed_from": str(result.resumed_from) if result.resumed_from else None,
        "resumed_from_frame": result.resumed_from_frame,
        "total_frames": result.total_frames,
        "total_samples": result.total_samples,
        "snapshots_written": result.snapshots_written,
    }


def run(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    previous_handlers: Dict[int, Any] = {}
    try:
        args = parser.parse_args(argv)
        if args.snapshot_interval <= 0:
            raise BusinessError(
                "INVALID_ARGUMENT",
                "--snapshot-interval must be a positive integer",
            )
        if args.max_snapshots <= 0:
            raise BusinessError(
                "INVALID_ARGUMENT",
                "--max-snapshots must be a positive integer",
            )
        if args.frame_size is not None and args.frame_size <= 0:
            raise BusinessError(
                "INVALID_ARGUMENT",
                "--frame-size must be a positive integer",
            )
        if (args.simulate_crash_after_frames is not None
                and args.simulate_crash_after_frames <= 0):
            raise BusinessError(
                "INVALID_ARGUMENT",
                "--simulate-crash-after-frames must be a positive integer",
            )

        scene_path = normalize_path(args.scene)
        scene = load_scene(scene_path)
        if args.frame_size is not None:
            scene = scene.with_frame_size(args.frame_size)

        planned_files = ["mix.wav", "render_manifest.json"] + output_filenames(scene)
        output_dir = ensure_output_dir(args.output_dir, planned_files)
        snapshot_dir = ensure_snapshot_dir(args.snapshot_dir)

        options = RenderOptions(
            snapshot_interval=args.snapshot_interval,
            max_snapshots=args.max_snapshots,
            force_resume=bool(args.resume),
            simulate_crash_after_frames=args.simulate_crash_after_frames,
        )
        renderer = OfflineRenderer(scene, output_dir, snapshot_dir, options)
        previous_handlers = _install_signal_handlers(renderer)
        result = renderer.run()
        sys.stdout.write(
            json.dumps(_success_payload(result), ensure_ascii=False, sort_keys=True) + "\n"
        )
        sys.stdout.flush()
        return 0
    except BusinessError as error:
        report_error(error)
        return error.exit_code
    except KeyboardInterrupt:
        error = BusinessError(
            "RENDER_INTERRUPTED",
            "rendering was interrupted before a snapshot could be guaranteed",
        )
        report_error(error)
        return error.exit_code
    except Exception as exc:
        error = BusinessError(
            "INTERNAL_ERROR",
            "an unexpected internal failure occurred",
            {"exception_type": type(exc).__name__},
        )
        report_error(error)
        return error.exit_code
    finally:
        _restore_signal_handlers(previous_handlers)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())

