from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .engine import ExportOptions, SimulatedInterrupt, run_export
from .errors import (
    EXIT_INTERNAL,
    EXIT_INTERRUPTED,
    AppError,
    interrupted_export,
    invalid_config,
)


class MixArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise invalid_config("invalid command-line arguments", error=message)


def build_parser() -> argparse.ArgumentParser:
    parser = MixArgumentParser(
        prog="python -m replay_mixer",
        description="Deterministic offline replay commentary mixer.",
    )
    parser.add_argument("--config", required=True, help="Path to the replay JSON configuration.")
    parser.add_argument(
        "--strict-checkpoint",
        action="store_true",
        help="Return CHECKPOINT_INVALID instead of automatically restarting a bad checkpoint.",
    )
    parser.add_argument(
        "--stop-after-blocks",
        type=int,
        default=None,
        help="Testing hook: stop after N newly written blocks in the selected phase.",
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=("any", "measure", "final"),
        default="any",
        help="Testing hook: phase in which --stop-after-blocks applies.",
    )
    return parser


def _emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _emit_error(error: AppError) -> None:
    sys.stderr.write(error.to_json_line() + "\n")
    sys.stderr.flush()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = build_parser().parse_args(argv)
        if args.stop_after_blocks is not None and args.stop_after_blocks < 1:
            raise invalid_config("--stop-after-blocks must be positive")
        result = run_export(
            args.config,
            ExportOptions(
                strict_checkpoint=args.strict_checkpoint,
                stop_after_blocks=args.stop_after_blocks,
                stop_after_phase=args.stop_after_phase,
            ),
        )
        _emit(result)
        return 0
    except AppError as exc:
        _emit_error(exc)
        return exc.exit_code
    except SimulatedInterrupt as exc:
        error = interrupted_export(
            "export stopped intentionally after writing a checkpoint",
            phase=exc.phase,
            completed_blocks=exc.completed_blocks,
        )
        _emit_error(error)
        return EXIT_INTERRUPTED
    except KeyboardInterrupt:
        error = interrupted_export("export interrupted by the user")
        _emit_error(error)
        return EXIT_INTERRUPTED
    except Exception as exc:  # Deliberately do not expose a traceback contract.
        error = AppError(
            "INTERNAL_ERROR",
            EXIT_INTERNAL,
            "unexpected internal error",
            {"error_type": type(exc).__name__, "error": str(exc)},
        )
        _emit_error(error)
        return EXIT_INTERNAL
