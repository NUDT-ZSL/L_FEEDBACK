"""Command-line interface for validation and rendering."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from .automation import AutomationError
from .generate import SCENARIOS, generate_project
from .pipeline import render_project
from .project import ProjectError, load_output_config, load_project, validate_headers
from .report import print_json, write_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postaudio",
        description="Deterministic multi-track WAV post-processing",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="create a deterministic sample project")
    generate_parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    generate_parser.add_argument("--out", required=True, help="project directory to create")
    generate_parser.add_argument(
        "--force", action="store_true", help="replace an existing output directory"
    )

    validate_parser = subparsers.add_parser(
        "validate",
        aliases=["load"],
        help="validate manifest declarations against WAV headers",
    )
    validate_parser.add_argument("project", help="project directory containing manifest.json")

    render_parser = subparsers.add_parser(
        "render",
        aliases=["export"],
        help="render a project to WAV and JSON report",
    )
    render_parser.add_argument("project", help="project directory containing manifest.json")
    render_parser.add_argument(
        "--config", help="output config JSON (default: manifest.output_config or project/output_config.json)"
    )
    render_parser.add_argument("--output", help="output WAV path (default: <project>/out/output.wav)")
    render_parser.add_argument("--report", help="report path (default: <project>/out/report.json)")
    return parser


def _print_project_error(exc: Exception) -> None:
    code = getattr(exc, "code", "PROJECT_ERROR")
    detail = getattr(exc, "detail", str(exc))
    print(f"code={code} {detail}", file=sys.stderr)


def _run_generate(args: argparse.Namespace) -> int:
    if os.path.exists(args.out) and not args.force:
        print(
            f"code=OUTPUT_EXISTS directory={args.out} (use --force to replace)",
            file=sys.stderr,
        )
        return 1
    if args.force and os.path.isdir(args.out):
        import shutil

        shutil.rmtree(args.out)
    try:
        root = generate_project(args.scenario, args.out)
    except (OSError, ValueError) as exc:
        print(f"code=GENERATION_FAILED {exc}", file=sys.stderr)
        return 1
    print_json({"generated": root, "scenario": args.scenario})
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    try:
        project = load_project(args.project)
    except ProjectError as exc:
        _print_project_error(exc)
        return 1
    summaries, errors = validate_headers(project)
    if errors:
        for error in errors:
            print(
                f"track={error['track']} code={error['code']} detail={error['detail']}",
                file=sys.stderr,
            )
        return 1
    unique_rates = sorted({item["sample_rate"] for item in summaries})
    unique_depths = sorted({item["bit_depth"] for item in summaries})
    print_json(
        {
            "valid": True,
            "track_count": len(summaries),
            "manifest_track_count": len(project.tracks),
            "sample_rates": unique_rates,
            "bit_depths": unique_depths,
            "tracks": summaries,
        }
    )
    return 0


def _run_render(args: argparse.Namespace) -> int:
    try:
        project = load_project(args.project)
        config_path = args.config or project.config_path
        config = load_output_config(config_path)
    except ProjectError as exc:
        _print_project_error(exc)
        return 1
    except AutomationError as exc:
        print(
            f"code={exc.code} node_index={exc.node_index} {exc.detail}",
            file=sys.stderr,
        )
        return 1
    output_path = args.output or os.path.join(project.root, "out", "output.wav")
    report_path = args.report or os.path.join(project.root, "out", "report.json")
    try:
        result = render_project(project, config, output_path, report_path)
    except (ProjectError, OSError, ValueError) as exc:
        print(f"code=RENDER_FAILED {exc}", file=sys.stderr)
        return 1
    for warning in result.report.get("warnings", []):
        print(
            f"track={warning['track']} code={warning['code']} detail={warning['detail']}",
            file=sys.stderr,
        )
    for error in result.report.get("errors", []):
        print(
            f"track={error['track']} code={error['code']} detail={error['detail']}",
            file=sys.stderr,
        )
    print_json(
        {
            "exit_code": result.exit_code,
            "output": output_path,
            "report": report_path,
            "tracks_rendered": len(result.report.get("tracks", [])),
            "errors": len(result.report.get("errors", [])),
        }
    )
    return result.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        return _run_generate(args)
    if args.command in {"validate", "load"}:
        return _run_validate(args)
    if args.command in {"render", "export"}:
        return _run_render(args)
    parser.error("unknown command")
    return 1
