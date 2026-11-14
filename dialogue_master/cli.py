from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from dialogue_master.config import ConfigError, load_config
from dialogue_master.output import OutputError, publish_outputs
from dialogue_master.pipeline import process


class CliConfigError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliConfigError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="dialogue-master",
        description=(
            "Normalize dialogue clips to target LUFS and place them on a "
            "sample-accurate multichannel timeline."
        ),
    )
    parser.add_argument("config_path", nargs="?", help="input JSON config")
    parser.add_argument(
        "output_directory", nargs="?", help="output directory"
    )
    parser.add_argument(
        "-c", "--config", dest="config_option", help="input JSON config"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_option",
        help="output directory",
    )
    return parser


def _choose_path(
    positional: Optional[str],
    option: Optional[str],
    field: str,
) -> str:
    if positional and option and positional != option:
        raise CliConfigError(
            "{} specified twice with different values".format(field)
        )
    result = option or positional
    if not result:
        raise CliConfigError("{} is required".format(field))
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        config_argument = _choose_path(
            args.config_path, args.config_option, "config path"
        )
        output_argument = _choose_path(
            args.output_directory, args.output_option, "output directory"
        )
        config = load_config(Path(config_argument))
    except (CliConfigError, ConfigError) as exc:
        print("E_CFG: {}".format(exc), file=sys.stderr)
        return 1

    try:
        result = process(config)
    except MemoryError:
        print(
            "E_AUDIO: insufficient memory to build master track",
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            "E_AUDIO: unrecoverable processing failure: {}".format(exc),
            file=sys.stderr,
        )
        return 2

    for message in result.messages:
        print(message, file=sys.stderr)

    try:
        publish_outputs(
            Path(output_argument),
            result.master,
            result.log,
            config.sample_rate,
            config.bit_depth,
            config.channel_labels,
            write_master=result.exit_code == 0,
        )
    except OutputError as exc:
        print("E_AUDIO: {}".format(exc), file=sys.stderr)
        return 2
    return result.exit_code
