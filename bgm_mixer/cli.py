"""Command-line entry points and stable one-line JSON reporting."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence

import numpy as np
import soundfile as sf

from . import __version__
from .config import load_settings
from .core import run_mix
from .errors import E_INVALID_INPUT, E_IO, E_MIX, MixerError
from .synthesis import synthesize_signal


class JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise MixerError(E_INVALID_INPUT, f"command-line error: {message}")


def _json_line(payload: Dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=False,
    )


def _mixer_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        prog="mixer.py",
        description="Loop-aware, drift-free WAV BGM mixer.",
    )
    parser.add_argument("--input", required=True, help="source WAV used by tracks without a path")
    parser.add_argument("--target-rate", required=True, type=int, help="output sample rate in Hz")
    parser.add_argument("--config", required=True, help="playback configuration JSON")
    parser.add_argument("--output", required=True, help="output WAV path")
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def _synth_parser() -> argparse.ArgumentParser:
    parser = JsonArgumentParser(
        prog="mixer.py synth",
        description="Generate deterministic local sine test audio.",
    )
    parser.add_argument("--output", required=True, help="output WAV path")
    parser.add_argument("--sample-rate", required=True, type=int, help="source sample rate in Hz")
    parser.add_argument("--duration", required=True, type=float, help="duration in seconds")
    parser.add_argument(
        "--frequency",
        action="append",
        type=float,
        default=None,
        help="sine frequency in Hz; repeat to sum components",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.5,
        help="amplitude applied to every generated component",
    )
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    return parser


def _atomic_write_samples(path: str, samples: np.ndarray, sample_rate: int) -> str:
    output_path = os.path.abspath(path)
    parent = os.path.dirname(output_path) or os.curdir
    if not os.path.isdir(parent):
        raise MixerError(E_IO, f"output directory does not exist: {parent}")
    try:
        descriptor, temp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            dir=parent,
        )
        os.close(descriptor)
    except OSError as exc:
        raise MixerError(E_IO, f"output directory is not writable: {exc}") from exc

    try:
        with sf.SoundFile(
            temp_path,
            mode="w",
            samplerate=sample_rate,
            channels=samples.shape[1],
            format="WAV",
            subtype="PCM_24",
        ) as handle:
            handle.write(samples)
        os.replace(temp_path, output_path)
        temp_path = ""
    except MixerError:
        raise
    except OSError as exc:
        raise MixerError(E_IO, f"cannot write output WAV: {exc}") from exc
    except (RuntimeError, ValueError) as exc:
        raise MixerError(E_IO, f"cannot write output WAV: {exc}") from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    return output_path


def _run_synth(argv: Sequence[str]) -> Dict[str, Any]:
    args = _synth_parser().parse_args(list(argv))
    if args.sample_rate <= 0:
        raise MixerError(E_INVALID_INPUT, "sample rate must be positive")
    if not math.isfinite(args.duration) or args.duration <= 0.0:
        raise MixerError(E_INVALID_INPUT, "duration must be finite and positive")
    if not math.isfinite(args.amplitude) or args.amplitude < 0.0:
        raise MixerError(E_INVALID_INPUT, "amplitude must be finite and non-negative")
    frequencies = args.frequency if args.frequency else [1000.0]
    try:
        samples = synthesize_signal(
            sample_rate=args.sample_rate,
            duration_seconds=args.duration,
            frequencies=frequencies,
            amplitudes=args.amplitude,
            channels=args.channels,
        )
    except ValueError as exc:
        raise MixerError(E_INVALID_INPUT, str(exc)) from exc

    output_path = _atomic_write_samples(args.output, samples, args.sample_rate)
    return {
        "status": "ok",
        "error_code": None,
        "output": output_path,
        "output_sample_rate": args.sample_rate,
        "output_frames": int(samples.shape[0]),
        "output_channels": int(samples.shape[1]),
        "output_duration_seconds": round(samples.shape[0] / float(args.sample_rate), 9),
        "frequencies": [float(value) for value in frequencies],
        "amplitude": float(args.amplitude),
    }


def _run_mixer(argv: Sequence[str]) -> Dict[str, Any]:
    args = _mixer_parser().parse_args(list(argv))
    input_path = os.path.abspath(os.path.expanduser(args.input))
    config_path = os.path.abspath(os.path.expanduser(args.config))
    settings = load_settings(config_path, input_path)
    return run_mix(settings, args.target_rate, args.output)


def _error_summary(error: MixerError) -> Dict[str, Any]:
    return {
        "status": "error",
        "error_code": error.code,
        "exit_code": error.exit_code,
        "message": error.message,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "synth":
            summary = _run_synth(arguments[1:])
        else:
            summary = _run_mixer(arguments)
        exit_code = 0
    except MixerError as exc:
        summary = _error_summary(exc)
        exit_code = exc.exit_code
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # Defensive boundary: never leak a traceback to CLI users.
        error = MixerError(E_MIX, f"unexpected mixer failure: {type(exc).__name__}: {exc}")
        summary = _error_summary(error)
        exit_code = error.exit_code

    print(_json_line(summary), flush=True)
    return exit_code
