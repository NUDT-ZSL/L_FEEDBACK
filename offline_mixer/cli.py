"""Command-line interface for the offline mixer."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .engine import run_mix
from .errors import AllTracksUnreadable, ManifestError, MixerError, OutputError


class BusinessArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise ManifestError("MIX1000", f"command-line argument error: {message}")


def build_parser() -> argparse.ArgumentParser:
    parser = BusinessArgumentParser(
        prog="offline-mixer",
        description=(
            "Offline 48 kHz / 24-bit dialogue-aware mixer with BS.1770 "
            "loudness normalization and sample-aligned ducking."
        ),
    )
    parser.add_argument(
        "manifest",
        nargs="?",
        help="input JSON manifest",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="output directory",
    )
    parser.add_argument(
        "-m",
        "--manifest",
        dest="manifest_option",
        help="input JSON manifest (alternative to positional form)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_option",
        help="output directory (alternative to positional form)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        manifest = args.manifest_option or args.manifest
        output_dir = args.output_option or args.output_dir
        if not manifest:
            raise ManifestError("MIX1000", "missing manifest path")
        if not output_dir:
            raise ManifestError("MIX1000", "missing output directory")
        if args.manifest_option and args.manifest:
            raise ManifestError(
                "MIX1000", "manifest supplied both positionally and with --manifest"
            )
        if args.output_option and args.output_dir:
            raise ManifestError(
                "MIX1000",
                "output directory supplied both positionally and with --output-dir",
            )

        result = run_mix(manifest, output_dir)
        if result.exit_code != 0:
            code = "MIX2001"
            if result.failure_message and "segment" in result.failure_message:
                code = "MIX3001"
            print(
                f"{code}: {result.failure_message or 'mixing failed'}",
                file=sys.stderr,
            )
            if result.metadata_path is not None:
                print(f"diagnostic metadata: {result.metadata_path}", file=sys.stderr)
            return result.exit_code

        metadata_path = result.metadata_path
        status = result.metadata.get("summary", {}).get("status", "ok")
        print(
            f"{status.upper()}: wrote {len(result.output_files)} WAV segment(s); "
            f"metadata={metadata_path}"
        )
        skipped = result.metadata.get("summary", {}).get("tracks_skipped", 0)
        errors = result.metadata.get("summary", {}).get("tracks_errors", 0)
        if skipped or errors:
            print(
                f"WARNING: {skipped} track(s) skipped, {errors} track(s) isolated "
                "after processing errors; see metadata.",
                file=sys.stderr,
            )
        return 0
    except ManifestError as exc:
        print(exc.as_stderr(), file=sys.stderr)
        return 2
    except (AllTracksUnreadable, OutputError) as exc:
        print(exc.as_stderr(), file=sys.stderr)
        return 1
    except MixerError as exc:
        print(exc.as_stderr(), file=sys.stderr)
        return 1 if exc.code.startswith("MIX3") else 2
    except KeyboardInterrupt:
        print("MIX9001: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"MIX9000: unexpected internal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
