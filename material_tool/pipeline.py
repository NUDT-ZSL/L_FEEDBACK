"""Batch orchestration, hot reload probes and summary generation."""

from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .consistency import check_consistency
from .errors import MaterialToolError, StructuredError
from .graph import (
    CHANNELS,
    ParsedGraph,
    load_graph_file,
    parse_graph_document,
)
from .renderer import ChannelRenderResult, MaterialSession, Renderer
from .samples import builtin_sample_documents


@dataclass
class RunSettings:
    output_dir: Path
    channels: list[str] = field(default_factory=lambda: list(CHANNELS))
    size: int = 512
    tolerance: float = 1e-4
    reload_check: bool = True
    input_dir: Path | None = None


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or fallback


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _structured_from_exception(
    exc: Exception, graph_id: str | None
) -> StructuredError:
    if isinstance(exc, MaterialToolError):
        details = {
            key: value
            for key, value in exc.details.items()
            if key not in {"source", "vertex_source", "fragment_source"}
        }
        return StructuredError(
            code=exc.code,
            message=exc.message,
            graph_id=graph_id,
            details=details,
        )
    return StructuredError(
        code="UNEXPECTED_ERROR",
        message=f"{type(exc).__name__}: {exc}",
        graph_id=graph_id,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_inputs(
    input_dir: Path | None,
) -> tuple[list[ParsedGraph], list[StructuredError]]:
    graphs: list[ParsedGraph] = []
    errors: list[StructuredError] = []
    if input_dir is None:
        for name, document in builtin_sample_documents().items():
            graphs.append(parse_graph_document(document, source_name=f"builtin:{name}"))
        return graphs, errors
    if not input_dir.exists() or not input_dir.is_dir():
        errors.append(
            StructuredError(
                code="INPUT_DIRECTORY_INVALID",
                message=f"Material graph directory does not exist: {input_dir}",
            )
        )
        return graphs, errors
    paths = sorted(input_dir.rglob("*.json"))
    if not paths:
        errors.append(
            StructuredError(
                code="INPUT_NO_JSON",
                message=f"No JSON material graphs found under {input_dir}.",
            )
        )
    for path in paths:
        graphs.append(load_graph_file(path))
    return graphs, errors


def _reload_document(
    graph: ParsedGraph, revision: int
) -> dict[str, Any]:
    document = copy.deepcopy(graph.document)
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        document["metadata"] = metadata
    metadata["reload_probe_revision"] = revision
    return document


def _save_compile_log(
    graph_dir: Path, error: StructuredError
) -> None:
    raw_log = error.details.get("raw_log")
    if not isinstance(raw_log, str) or not raw_log:
        return
    shader_dir = graph_dir / "shaders"
    shader_dir.mkdir(parents=True, exist_ok=True)
    (shader_dir / "compile.log").write_text(raw_log, encoding="utf-8")


def _channel_summary(
    result: ChannelRenderResult, output_root: Path
) -> dict[str, Any]:
    return {
        "image": _relative(result.image_path, output_root),
        "width": int(result.roughness.shape[1]),
        "height": int(result.roughness.shape[0]),
        "stats": result.stats,
    }


def _run_graph(
    renderer: Renderer | None,
    graph: ParsedGraph,
    graph_dir: Path,
    settings: RunSettings,
) -> tuple[dict[str, Any], list[StructuredError]]:
    errors: list[StructuredError] = list(graph.errors)
    entry: dict[str, Any] = {
        "id": graph.graph_id,
        "source": graph.source_name,
        "valid": graph.valid,
        "status": "invalid" if not graph.valid else "pending",
        "inputs": graph.to_summary_inputs(),
        "errors": [error.to_dict() for error in errors],
        "channels": {},
        "consistency": None,
        "hot_reload": {"attempted": False},
        "resources": {},
    }
    if not graph.valid:
        return entry, errors
    if renderer is None:
        error = StructuredError(
            code="GL_CONTEXT_ERROR",
            message="No OpenGL renderer is available for this material graph.",
            graph_id=graph.graph_id,
        )
        errors.append(error)
        entry["status"] = "failed"
        entry["errors"] = [item.to_dict() for item in errors]
        return entry, errors

    graph_dir.mkdir(parents=True, exist_ok=True)
    _write_json(graph_dir / "material.json", graph.document)
    entry["artifact_graph"] = _relative(graph_dir / "material.json", settings.output_dir)
    entry["shaders"] = {
        "vertex": _relative(graph_dir / "shaders" / "material.vert", settings.output_dir),
        "fragment": _relative(
            graph_dir / "shaders" / "material.frag", settings.output_dir
        ),
    }

    baseline = renderer.resource_counts()
    entry["resources"]["before_session"] = baseline
    session: MaterialSession | None = None
    try:
        session = renderer.create_session(
            graph, output_dir=graph_dir, size=settings.size
        )
        initial_results: dict[str, ChannelRenderResult] = {}
        for channel in settings.channels:
            initial_results[channel] = session.render_channel(channel)
        results = dict(initial_results)
        consistency = check_consistency(
            graph, results, tolerance=settings.tolerance
        )

        reload_report: dict[str, Any] = {
            "attempted": settings.reload_check,
            "changed": False,
            "resource_counts_equal": None,
            "rendered_after_reload": [],
            "image_hashes_equal": None,
        }
        if settings.reload_check:
            changed_document = _reload_document(graph, 1)
            reloaded_graph = parse_graph_document(
                changed_document, source_name=graph.source_name + "#reload"
            )
            if not reloaded_graph.valid:
                reload_error = StructuredError(
                    code="HOT_RELOAD_PARSE_ERROR",
                    message="Reloaded graph failed validation.",
                    graph_id=graph.graph_id,
                    details={
                        "errors": [error.to_dict() for error in reloaded_graph.errors]
                    },
                )
                errors.append(reload_error)
                reload_report["error"] = reload_error.to_dict()
            else:
                before_reload = renderer.resource_counts()
                try:
                    reload_counts = session.reload(reloaded_graph)
                    reload_report.update(reload_counts)
                    reload_report["changed"] = True
                    reload_report["resource_counts_equal"] = (
                        reload_counts["resource_counts_before"]
                        == reload_counts["resource_counts_after"]
                    )
                    reloaded_results = {
                        channel: session.render_channel(channel)
                        for channel in settings.channels
                    }
                    reload_report["rendered_after_reload"] = list(settings.channels)
                    reload_report["image_hashes_equal"] = all(
                        initial_results[channel].stats["sha256"]
                        == reloaded_results[channel].stats["sha256"]
                        for channel in settings.channels
                    )
                    results = reloaded_results
                    graph = reloaded_graph
                    consistency = check_consistency(
                        graph, results, tolerance=settings.tolerance
                    )
                    after_reload_render = renderer.resource_counts()
                    reload_report["resource_counts_after_reload_render"] = (
                        after_reload_render
                    )
                    reload_report["resource_counts_stable_after_render"] = (
                        before_reload == after_reload_render
                    )
                except Exception as exc:
                    reload_error = _structured_from_exception(exc, graph.graph_id)
                    reload_error.message = "Hot reload failed: " + reload_error.message
                    errors.append(reload_error)
                    reload_report["error"] = reload_error.to_dict()
                    _save_compile_log(graph_dir, reload_error)

        for channel, result in results.items():
            entry["channels"][channel] = _channel_summary(
                result, settings.output_dir
            )
        entry["consistency"] = consistency
        entry["hot_reload"] = reload_report
        if consistency["conclusion"] == "fail":
            for difference in consistency["differences"]:
                errors.append(
                    StructuredError(
                        code="CONSISTENCY_MISMATCH",
                        message=(
                            f"{difference['parameter']} differs between "
                            f"{difference['reference_channel']} and "
                            f"{difference['actual_channel']} by "
                            f"{difference['max_abs_diff']:.9g} "
                            f"(tolerance {settings.tolerance:.9g})."
                        ),
                        graph_id=graph.graph_id,
                        channel=difference["actual_channel"],
                        path=difference["node_path"],
                        details=difference,
                    )
                )
        entry["errors"] = [error.to_dict() for error in errors]
        entry["status"] = "rendered"
    except Exception as exc:
        error = _structured_from_exception(exc, graph.graph_id)
        errors.append(error)
        _save_compile_log(graph_dir, error)
        entry["errors"] = [item.to_dict() for item in errors]
        entry["status"] = "failed"
    finally:
        if session is not None:
            session.release()
        after_release = renderer.resource_counts()
        entry["resources"]["after_release"] = after_release
        entry["resources"]["session_leak_free"] = baseline == after_release
        if baseline != after_release:
            error = StructuredError(
                code="RESOURCE_LEAK",
                message="GPU resource counts did not return to the pre-session baseline.",
                graph_id=graph.graph_id,
                details={
                    "before_session": baseline,
                    "after_release": after_release,
                },
            )
            errors.append(error)
            entry["errors"] = [item.to_dict() for item in errors]
            entry["status"] = "failed"
    return entry, errors


def run(settings: RunSettings) -> tuple[dict[str, Any], int]:
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    graphs, load_errors = _load_inputs(settings.input_dir)
    all_errors: list[StructuredError] = list(load_errors)
    entries: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    renderer: Renderer | None = None
    context_info: dict[str, str] = {}
    global_baseline: dict[str, int] = {}
    final_counts: dict[str, int] = {}
    final_counts_after_close: dict[str, int] = {}
    try:
        if any(graph.valid for graph in graphs):
            renderer = Renderer(size=settings.size)
            context_info = renderer.context_info
            global_baseline = renderer.resource_counts()
        for index, graph in enumerate(graphs):
            graph_id = graph.graph_id or f"unnamed_{index}"
            safe = _safe_name(graph_id, f"graph_{index}")
            if graph_id in seen_ids:
                duplicate = StructuredError(
                    code="GRAPH_ID_DUPLICATE",
                    message=f"Duplicate graph id {graph_id!r}; output uses a unique suffix.",
                    graph_id=graph_id,
                )
                all_errors.append(duplicate)
                safe = f"{safe}__{index + 1}"
            seen_ids[graph_id] = index
            graph_dir = settings.output_dir / safe
            entry, errors = _run_graph(renderer, graph, graph_dir, settings)
            entries.append(entry)
            all_errors.extend(errors)
    except Exception as exc:
        all_errors.append(_structured_from_exception(exc, None))
    finally:
        if renderer is not None:
            final_counts_preclose = renderer.resource_counts()
            renderer.close()
            final_counts = final_counts_preclose
            final_counts_after_close = renderer.resource_counts()

    rendered = sum(1 for entry in entries if entry["status"] == "rendered")
    failed = sum(1 for entry in entries if entry["status"] == "failed")
    invalid = sum(1 for entry in entries if entry["status"] == "invalid")
    consistency_passed = sum(
        1
        for entry in entries
        if (entry.get("consistency") or {}).get("conclusion") == "pass"
    )
    consistency_failed = sum(
        1
        for entry in entries
        if (entry.get("consistency") or {}).get("conclusion") == "fail"
    )
    summary = {
        "schema_version": 1,
        "tool": {
            "name": "material-graph-renderer",
            "version": __version__,
        },
        "settings": {
            "input_dir": str(settings.input_dir) if settings.input_dir else "builtin",
            "output_dir": str(settings.output_dir),
            "width": settings.size,
            "height": settings.size,
            "channels": settings.channels,
            "consistency_tolerance": settings.tolerance,
            "reload_check": settings.reload_check,
        },
        "gl": context_info,
        "resource_baseline": global_baseline,
        "resource_counts_before_close": final_counts,
        "leak_free_before_close": final_counts == global_baseline,
        "resource_counts_after_close": final_counts_after_close,
        "leak_free_after_close": final_counts_after_close == {},
        "graphs": entries,
        "errors": [error.to_dict() for error in all_errors],
        "summary": {
            "total_graphs": len(entries),
            "rendered": rendered,
            "failed": failed,
            "invalid": invalid,
            "consistency_passed": consistency_passed,
            "consistency_failed": consistency_failed,
            "error_count": len(all_errors),
        },
    }
    _write_json(settings.output_dir / "summary.json", summary)
    for error in all_errors:
        print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
    return summary, 0

