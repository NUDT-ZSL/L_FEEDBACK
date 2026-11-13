from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .assemble import AssemblyError, analyze_workdir, assemble, verify_sequence
from .scheduler import Scheduler, SchedulerInterrupted
from .spec import SpecError, load_spec


EXIT_INPUT_ERROR = 2
EXIT_RENDER_INCOMPLETE = 3
EXIT_ASSEMBLY_INTEGRITY = 4
EXIT_INTERRUPTED = 130


def _emit(value: dict[str, Any], stream: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), file=stream)


def _check_disjoint(work_dir: Path, output_dir: Path | None) -> None:
    if output_dir is None:
        return
    work = work_dir.resolve()
    output = output_dir.resolve()
    if output == Path(output.anchor) or work == Path(work.anchor):
        raise ValueError("work and output directories may not be filesystem roots")
    if output == work or output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError("work directory and output directory must be disjoint")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="deterministic scene JSON")
    parser.add_argument("--work-dir", required=True, help="chunk/checkpoint work directory")
    parser.add_argument("--chunk-size", type=int, default=24, help="frames per chunk")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frame-render",
        description="Offline deterministic ray-sphere frame renderer",
    )
    parser.add_argument("--version", action="version", version="offline-frame-renderer 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render = subparsers.add_parser("render", help="render/resume frame chunks only")
    _add_common(render)
    render.add_argument("--jobs", type=int, default=4)
    render.add_argument("--max-retries", type=int, default=1)
    render.add_argument("--verbose", action="store_true")

    run = subparsers.add_parser("run", help="render/resume chunks and assemble sequence")
    _add_common(run)
    run.add_argument("--output", required=True, help="final sequence directory")
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--max-retries", type=int, default=1)
    run.add_argument("--verbose", action="store_true")

    assembly = subparsers.add_parser("assemble", help="assemble an already-rendered workdir")
    _add_common(assembly)
    assembly.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify", help="verify a final sequence directory")
    verify.add_argument("--output", required=True)
    return parser


def _render_summary_or_error(
    spec: Any, work_dir: Path, chunk_size: int, jobs: int, max_retries: int, verbose: bool
) -> tuple[dict[str, Any], bool]:
    scheduler = Scheduler(
        spec,
        work_dir,
        chunk_size=chunk_size,
        jobs=jobs,
        max_retries=max_retries,
        verbose=verbose,
    )
    try:
        summary = scheduler.run()
    except SchedulerInterrupted as exc:
        summary = exc.summary
        summary["error"] = "scheduler_interrupted"
        return summary, False
    if summary["status"] == "ok":
        return summary, True
    analysis = analyze_workdir(work_dir, spec, chunk_size)
    error_summary = analysis.summary("render_incomplete")
    error_summary.update(
        {
            "render_status": summary["status"],
            "unfinished_chunks": summary["unfinished_chunks"],
            "retry_counts": summary["retry_counts"],
            "attempts": summary["attempts"],
            "recoverable_failures": summary["recoverable_failures"],
            "forced_terminations": summary["forced_terminations"],
            "invalid_checkpoints": summary["invalid_checkpoints"],
        }
    )
    return error_summary, False


def _run(args: argparse.Namespace) -> int:
    spec = load_spec(args.input)
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output).resolve()
    _check_disjoint(work_dir, output_dir)
    render_summary, ok = _render_summary_or_error(
        spec,
        work_dir,
        args.chunk_size,
        args.jobs,
        args.max_retries,
        args.verbose,
    )
    if not ok:
        _emit(render_summary, sys.stderr)
        return (
            EXIT_INTERRUPTED
            if render_summary.get("render_status") == "interrupted"
            else EXIT_RENDER_INCOMPLETE
        )
    try:
        assembly_summary = assemble(work_dir, output_dir, spec, args.chunk_size)
    except AssemblyError as exc:
        _emit(exc.summary, sys.stderr)
        return EXIT_ASSEMBLY_INTEGRITY
    result = {
        "status": "ok",
        "command": "run",
        "render": render_summary,
        "assembly": assembly_summary,
    }
    _emit(result, sys.stdout)
    return 0


def _render(args: argparse.Namespace) -> int:
    spec = load_spec(args.input)
    work_dir = Path(args.work_dir).resolve()
    summary, ok = _render_summary_or_error(
        spec,
        work_dir,
        args.chunk_size,
        args.jobs,
        args.max_retries,
        args.verbose,
    )
    if ok:
        _emit(summary, sys.stdout)
        return 0
    _emit(summary, sys.stderr)
    return (
        EXIT_INTERRUPTED
        if summary.get("render_status") == "interrupted"
        else EXIT_RENDER_INCOMPLETE
    )


def _assemble(args: argparse.Namespace) -> int:
    spec = load_spec(args.input)
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output).resolve()
    _check_disjoint(work_dir, output_dir)
    try:
        summary = assemble(work_dir, output_dir, spec, args.chunk_size)
    except AssemblyError as exc:
        _emit(exc.summary, sys.stderr)
        return EXIT_ASSEMBLY_INTEGRITY
    _emit(summary, sys.stdout)
    return 0


def _verify(args: argparse.Namespace) -> int:
    summary = verify_sequence(args.output)
    if summary["ok"]:
        summary["status"] = "ok"
        summary["command"] = "verify"
        _emit(summary, sys.stdout)
        return 0
    summary["status"] = "error"
    summary["command"] = "verify"
    _emit(summary, sys.stderr)
    return EXIT_ASSEMBLY_INTEGRITY


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "render":
            return _render(args)
        if args.command == "assemble":
            return _assemble(args)
        if args.command == "verify":
            return _verify(args)
        parser.error(f"unknown command: {args.command}")
    except (SpecError, ValueError, OSError) as exc:
        _emit(
            {
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc),
            },
            sys.stderr,
        )
        return EXIT_INPUT_ERROR
    except Exception as exc:
        _emit(
            {
                "status": "error",
                "error": type(exc).__name__,
                "message": str(exc),
            },
            sys.stderr,
        )
        return EXIT_INPUT_ERROR
    return EXIT_INPUT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())

