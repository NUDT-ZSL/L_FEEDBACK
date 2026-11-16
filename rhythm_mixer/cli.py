"""Command line entry point.

Single-threaded execution is forced before NumPy is imported so the tool
behaves identically on Windows and Linux, including byte-identical output.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse
import json
import sys

from .engine import run
from .errors import MixerError


def build_parser():
    parser = argparse.ArgumentParser(
        prog="rhythm-mixer",
        description=(
            "Mix beat-aligned dynamic WAV layers driven by a BPM chart and "
            "control events."
        ),
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="directory containing chart.json, events.json and layer WAVs",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="directory receiving mix_output.wav and beat_alignment_report.json",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        wav_path, report_path = run(args.input_dir, args.output_dir)
    except MixerError as exc:
        payload = exc.to_payload()
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return exc.exit_code
    except (OSError, ValueError) as exc:
        payload = {"error": "E_INTERNAL", "message": str(exc)}
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "wav": os.path.basename(wav_path),
                "report": os.path.basename(report_path),
                "output_dir": args.output_dir,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
