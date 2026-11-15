from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .compiler import compile_graph, load_graph_json
from .errors import PipelineError, graph_error
from .limits import DEFAULT_LIMITS
from .outputs import write_atomic_file
from .renderer import CHANNEL_FILES, render_scene, write_render
from .samples import generate_samples
from .scene import load_scene


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise graph_error("E_IO", f"cannot hash input file: {type(exc).__name__}")
    return digest.hexdigest()


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _emit_error(error: PipelineError, error_out: Optional[Path]) -> None:
    payload = _json_bytes(error.as_envelope(), pretty=True)
    sys.stderr.write(payload.decode("utf-8"))
    if error_out is not None:
        try:
            write_atomic_file(error_out, payload)
        except Exception as exc:
            fallback = graph_error(
                "E_ERROR_OUTPUT",
                f"cannot write structured error file: {type(exc).__name__}",
            )
            sys.stderr.write(_json_bytes(fallback.as_envelope(), pretty=False).decode("utf-8"))


def _ir_payload(compiled) -> Dict[str, Any]:
    payload = compiled.to_dict()
    payload["ir_sha256"] = compiled.sha256()
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matgraph",
        description="Deterministic offline material graph compiler and CPU renderer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser(
        "generate-samples",
        help="write deterministic sample graph, scene, and PNG textures",
    )
    sample_parser.add_argument("--out", type=Path, default=Path("samples"))
    sample_parser.add_argument("--error-out", type=Path, default=None)

    compile_parser = subparsers.add_parser(
        "compile",
        help="compile a material graph and write its stable IR JSON",
    )
    compile_parser.add_argument("--graph", type=Path, required=True)
    compile_parser.add_argument("--out", type=Path, required=True)
    compile_parser.add_argument("--error-out", type=Path, default=None)

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate and compile a material graph without rendering",
    )
    validate_parser.add_argument("--graph", type=Path, required=True)
    validate_parser.add_argument("--error-out", type=Path, default=None)

    render_parser = subparsers.add_parser(
        "render",
        help="compile material, rasterize scene, shade, and publish all outputs",
    )
    render_parser.add_argument("--graph", type=Path, required=True)
    render_parser.add_argument("--scene", type=Path, required=True)
    render_parser.add_argument("--out", type=Path, required=True)
    render_parser.add_argument("--width", type=int, default=None)
    render_parser.add_argument("--height", type=int, default=None)
    render_parser.add_argument("--error-out", type=Path, default=None)
    return parser


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.command == "generate-samples":
        generate_samples(args.out)
        return {
            "ok": True,
            "command": "generate-samples",
            "out": str(args.out.resolve()),
            "files": [
                "material_graph.json",
                "scene.json",
                "textures/checker.png",
                "textures/bumps.png",
            ],
        }

    graph = load_graph_json(args.graph, DEFAULT_LIMITS)
    compiled = compile_graph(graph, args.graph.resolve().parent, DEFAULT_LIMITS)
    if args.command == "compile":
        write_atomic_file(args.out, _json_bytes(_ir_payload(compiled)))
        return {
            "ok": True,
            "command": "compile",
            "out": str(args.out.resolve()),
            "instruction_count": len(compiled.instructions),
            "ir_sha256": compiled.sha256(),
        }
    if args.command == "validate":
        return {
            "ok": True,
            "command": "validate",
            "graph": compiled.name,
            "node_count": compiled.node_count,
            "instruction_count": len(compiled.instructions),
            "folded_node_count": len(compiled.folded_nodes),
            "instruction_order": compiled.instruction_order(),
            "ir_sha256": compiled.sha256(),
        }

    scene = load_scene(
        args.scene,
        DEFAULT_LIMITS,
        width_override=args.width,
        height_override=args.height,
    )
    result = render_scene(
        compiled,
        scene,
        graph_sha256=_sha256_file(args.graph),
        scene_sha256=_sha256_file(args.scene),
    )
    write_render(result, args.out)
    return {
        "ok": True,
        "command": "render",
        "out": str(args.out.resolve()),
        "report": str((args.out / CHANNEL_FILES["report"]).resolve()),
        "resolution": result.report["resolution"],
        "counts": result.report["counts"],
        "ir_sha256": compiled.sha256(),
        "files": sorted(result.files),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        summary = _run(args)
    except PipelineError as exc:
        _emit_error(exc, getattr(args, "error_out", None))
        return 2
    except Exception as exc:
        internal = PipelineError(
            code="E_INTERNAL",
            message=f"unexpected failure: {type(exc).__name__}: {exc}",
            stage="cli",
        )
        _emit_error(internal, getattr(args, "error_out", None))
        return 3
    sys.stdout.write(_json_bytes(summary).decode("utf-8"))
    return 0
