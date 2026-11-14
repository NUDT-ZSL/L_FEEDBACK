from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from .dsp import integrated_loudness, true_peak
from .errors import CinemixError, Diagnostics
from .processor import process_project, write_report
from .wavio import read_pcm_wav


def _json_line(payload: dict, *, stream: TextIO = sys.stdout) -> None:
    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stream,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cinemix",
        description="Deterministic offline multitrack mixer with BS.1770 loudness and true-peak limiting",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Process a multitrack JSON project")
    process.add_argument("config", type=Path, help="Project JSON file")
    process.add_argument("--output", type=Path, help="Override project.output")
    process.add_argument("--report", type=Path, help="Write a deterministic JSON report")
    process.add_argument(
        "--print-report",
        action="store_true",
        help="Print the full JSON report to stdout",
    )

    measure = subparsers.add_parser("measure", help="Measure a PCM WAV file")
    measure.add_argument("wav", type=Path)

    return parser


def _command_process(args: argparse.Namespace, diagnostics: Diagnostics) -> int:
    result = process_project(
        args.config,
        output_override=args.output,
        diagnostics=diagnostics,
    )
    if args.report is not None:
        write_report(args.report, result.report)
    if args.print_report:
        print(
            json.dumps(
                result.report,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
        )
    else:
        measurement = result.report["measurement"]
        _json_line(
            {
                "ok": True,
                "output": result.report["output"],
                "integrated_lufs": measurement["integrated_lufs"],
                "true_peak_dbtp": measurement["true_peak_dbtp"],
                "loudness_error_lu": measurement["loudness_error_lu"],
                "warnings": sum(
                    issue["level"] == "warning"
                    for issue in result.report["diagnostics"]
                ),
            }
        )
    return 0


def _command_measure(args: argparse.Namespace) -> int:
    samples, info = read_pcm_wav(args.wav)
    loudness = integrated_loudness(samples, info.sample_rate)
    peak = true_peak(samples, info.sample_rate)
    values = {
        "integrated_lufs": (
            round(loudness.lufs, 6) if loudness.lufs != float("-inf") else None
        ),
        "true_peak_dbtp": (
            round(peak.dbtp, 6) if peak.dbtp != float("-inf") else None
        ),
        "sample_peak_dbfs": None,
    }
    import numpy as np

    sample_peak = (
        float(np.max(np.abs(samples))) if samples.size else 0.0
    )
    if sample_peak > 0.0:
        import math

        values["sample_peak_dbfs"] = round(
            20.0 * math.log10(sample_peak), 6
        )
    _json_line(values)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    diagnostics = Diagnostics()
    try:
        if args.command == "process":
            code = _command_process(args, diagnostics)
        elif args.command == "measure":
            code = _command_measure(args)
        else:
            parser.error("unknown command")
            return 2
    except CinemixError as exc:
        diagnostics.emit()
        _json_line(exc.as_issue().as_dict(), stream=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        diagnostics.emit()
        _json_line(
            {
                "level": "error",
                "code": "E_INTERNAL",
                "message": "Interrupted",
            },
            stream=sys.stderr,
        )
        return 130
    except Exception as exc:  # noqa: BLE001 - turn unexpected failures into structured diagnostics
        diagnostics.emit()
        _json_line(
            {
                "level": "error",
                "code": "E_INTERNAL",
                "message": f"{type(exc).__name__}: {exc}",
            },
            stream=sys.stderr,
        )
        return 99
    diagnostics.emit()
    return code
