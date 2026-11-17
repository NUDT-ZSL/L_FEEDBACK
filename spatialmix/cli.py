"""Command line interface for the offline spatial mixer."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from typing import Any, Sequence

from . import __version__
from .engine import RenderOptions, render
from .errors import (
    BusinessError,
    E_INTERNAL,
    E_INTERRUPTED,
    ProcessingInterrupted,
)
from .hrtf import generate_hrtf


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatialmix",
        description=(
            "Deterministic offline HRTF spatial mixer with streaming block "
            "processing and resumable progress."
        ),
    )
    parser.add_argument("input_wav", nargs="?", help="Input multichannel WAV file")
    parser.add_argument("source_config", nargs="?", help="Source coordinate JSON file")
    parser.add_argument(
        "output_wav", nargs="?", help="Destination multichannel WAV file"
    )
    parser.add_argument(
        "--output-format",
        default=None,
        help=("Output sample format: pcm8, pcm16, pcm24, pcm32, float32 or float64"),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Samples per streaming block (default: 256)",
    )
    parser.add_argument(
        "--listener-pos",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Listener position in meters (default: 0 0 0)",
    )
    parser.add_argument(
        "--listener-forward",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Listener forward vector (default: 0 0 1)",
    )
    parser.add_argument(
        "--listener-up",
        nargs=3,
        type=float,
        default=(0.0, 1.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Listener up hint vector (default: 0 1 0)",
    )
    parser.add_argument(
        "--filter-order",
        choices=("before", "after"),
        default="before",
        help="Apply air-absorption low-pass before or after HRTF convolution",
    )
    parser.add_argument(
        "--speed-of-sound",
        type=float,
        default=343.0,
        help="Speed of sound in m/s used by the Doppler model (default: 343)",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="Progress state path (default: <output>.state.json)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard an existing state file and intermediate work directory",
    )
    parser.add_argument(
        "--expected-hrtf-version",
        default=None,
        help="Require a specific local HRTF dataset version",
    )
    parser.add_argument(
        "--stop-after-blocks",
        type=int,
        default=None,
        help=(
            "Gracefully stop after N committed source blocks while preserving "
            "resume state (useful for interruption tests)"
        ),
    )
    parser.add_argument(
        "--stop-after-mix-blocks",
        type=int,
        default=None,
        help=(
            "Gracefully stop after N committed final-mix blocks while "
            "preserving resume state (useful for interruption tests)"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON to stdout/stderr",
    )
    parser.add_argument(
        "--hrtf-info",
        action="store_true",
        help=(
            "Print the generated HRTF manifest and exit "
            "(positional arguments may be omitted)"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _error_payload(error: BusinessError) -> dict[str, Any]:
    return error.to_dict()


def _emit_error(error: BusinessError, use_json: bool) -> None:
    if use_json:
        print(json.dumps(_error_payload(error), ensure_ascii=False), file=sys.stderr)
    else:
        print(
            f"[{error.code}] {error.definition.error_type}: {error.message}",
            file=sys.stderr,
        )
        if error.details:
            print(
                "details: "
                + json.dumps(error.details, ensure_ascii=False, sort_keys=True),
                file=sys.stderr,
            )


def _emit_interrupted(message: str, use_json: bool) -> None:
    if use_json:
        print(
            json.dumps(
                {
                    "ok": False,
                    "code": E_INTERRUPTED.code,
                    "error_code": E_INTERRUPTED.code,
                    "exit_code": E_INTERRUPTED.exit_code,
                    "type": E_INTERRUPTED.error_type,
                    "error_type": E_INTERRUPTED.error_type,
                    "message": message,
                    "detail": {},
                    "details": {},
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
    else:
        print(f"[{E_INTERRUPTED.code}] interrupted: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    known, unknown = parser.parse_known_args(argv)
    use_json = bool(known.json)

    if known.hrtf_info:
        sample_rate = 48000
        info = generate_hrtf(sample_rate).metadata()
        if use_json:
            print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(json.dumps(info, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if unknown:
        parser.error("unrecognized arguments: " + " ".join(unknown))
    if not known.input_wav or not known.source_config or not known.output_wav:
        parser.error("input_wav, source_config and output_wav are required")
    if not known.output_format:
        parser.error("--output-format is required")
    args = known

    started = time.perf_counter()
    options = RenderOptions(
        input_path=args.input_wav,
        config_path=args.source_config,
        output_path=args.output_wav,
        output_format=args.output_format,
        block_size=args.block_size,
        listener_position=tuple(args.listener_pos),
        listener_forward=tuple(args.listener_forward),
        listener_up=tuple(args.listener_up),
        filter_order=args.filter_order,
        state_path=args.state,
        speed_of_sound=args.speed_of_sound,
        stop_after_blocks=args.stop_after_blocks,
        stop_after_mix_blocks=args.stop_after_mix_blocks,
        fresh=args.fresh,
        expected_hrtf_version=args.expected_hrtf_version,
    )

    def _handle_sigterm(signum: int, frame: Any) -> None:
        raise ProcessingInterrupted(
            f"Processing received signal {signum}; committed blocks can be resumed"
        )

    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
    except (AttributeError, ValueError, OSError):
        pass
    try:
        result = render(options)
    except ProcessingInterrupted as exc:
        _emit_interrupted(str(exc), args.json)
        return E_INTERRUPTED.exit_code
    except BusinessError as exc:
        _emit_error(exc, args.json)
        return exc.exit_code
    except KeyboardInterrupt:
        _emit_interrupted(
            "Processing was interrupted; committed blocks can be resumed", args.json
        )
        return E_INTERRUPTED.exit_code
    except Exception as exc:
        error = BusinessError(
            E_INTERNAL,
            "Unexpected internal failure",
            {"exception_type": type(exc).__name__, "reason": str(exc)},
        )
        _emit_error(error, args.json)
        return error.exit_code

    elapsed = time.perf_counter() - started
    payload = result.to_dict()
    payload["elapsed_seconds"] = round(elapsed, 6)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "spatialmix: wrote "
            f"{result.frames} frames @ {result.sample_rate} Hz, "
            f"{result.output_format}, {result.source_count} sources -> "
            f"{result.output_path}"
        )
        print(
            f"peak={result.peak_abs:.6f}, block={result.block_size}, "
            f"resumed={result.resumed}, "
            f"working_set_estimate={result.peak_working_bytes_estimate} bytes"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
