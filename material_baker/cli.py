"""Command-line interface for material-baker."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .bake import run_bake
from .errors import ToolError, invalid_config


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise invalid_config(f"{self.prog}: {message}")


def _bake_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="material-baker bake",
        description="Bake PBR material maps from a JSON configuration.",
    )
    parser.add_argument("config", nargs="?", help="JSON configuration file")
    parser.add_argument(
        "--config", dest="config_option", help="JSON configuration file"
    )
    parser.add_argument(
        "--input",
        "--input-dir",
        dest="input_dir",
        help="Directory containing albedo/roughness/metallic/normal PNG files",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        help="Directory for baked PNG files and metadata.json",
    )
    return parser


def _selftest_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="material-baker selftest",
        description="Run the deterministic end-to-end self-test.",
    )
    parser.add_argument(
        "--work-dir",
        default=".material_baker_selftest",
        help="Writable directory used for generated samples and outputs",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep generated sample and output files",
    )
    return parser


def _print_error(error: ToolError) -> None:
    print(f"ERROR {error.code}: {error.message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)

    if arguments and arguments[0] in (
        "selftest",
        "self-test",
        "--selftest",
        "--self-test",
    ):
        if arguments[0] in ("--selftest", "--self-test"):
            arguments = arguments[1:]
        else:
            arguments = arguments[1:]
        parser = _selftest_parser()
        args = parser.parse_args(arguments)
        from .selftest import run_selftest

        try:
            return run_selftest(args.work_dir, keep=args.keep)
        except ToolError as exc:
            _print_error(exc)
            return exc.exit_code
        except Exception as exc:  # pragma: no cover - last-resort CLI boundary
            print(f"ERROR E_INTERNAL: {exc}", file=sys.stderr)
            return 1

    if arguments and arguments[0] == "bake":
        arguments = arguments[1:]
    parser = _bake_parser()
    args = parser.parse_args(arguments)
    config_path = args.config_option or args.config

    try:
        if not config_path:
            raise invalid_config("a JSON configuration file is required")
        summary = run_bake(
            config_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, ensure_ascii=False))
        return 0
    except ToolError as exc:
        _print_error(exc)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - last-resort CLI boundary
        print(f"ERROR E_INTERNAL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

