"""Command line interface: ``python -m cinemixer project.json``."""

from __future__ import annotations

import argparse
import sys

from .errors import CineMixerError
from .logging_util import StructuredLogger
from .pipeline import run_project


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinemixer",
        description="Deterministic offline multi-track cutscene audio renderer.",
    )
    parser.add_argument("project", help="path to the project description JSON file")
    parser.add_argument(
        "--fault-after",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # test-only deterministic crash injection
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    log = StructuredLogger(sys.stderr)
    try:
        result = run_project(arguments.project, log=log, fault_after=arguments.fault_after)
    except CineMixerError as exc:
        log.error(exc.code, exc.message)
        return exc.exit_code
    except Exception as exc:  # defensive: keep the exit-code contract
        log.error("E_INTERNAL", f"unexpected failure: {exc!r}")
        return 1

    summary = {
        "output": str(result.output_path),
        "index": str(result.index_path),
        "length_samples": result.length_samples,
        "measured_lufs": (
            round(result.measured_lufs, 3)
            if result.measured_lufs == result.measured_lufs
            else None
        ),
        "loudness_adjusted": result.loudness_adjusted,
        "failed_tracks": list(result.failed_tracks),
    }
    log.info("render finished successfully", "I_OK", **summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
