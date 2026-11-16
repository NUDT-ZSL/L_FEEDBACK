"""High-level entry point wiring parsing, checkpoints and rendering together."""

from __future__ import annotations

import json
from pathlib import Path

from .logging_util import StructuredLogger
from .project import Project, parse_project
from .renderer import Renderer, RenderResult
from .state import CheckpointStore, checksum_bytes, state_path_for


def project_checksum(project: Project) -> str:
    """Deterministic checksum covering every project field that affects output."""
    payload = json.dumps(
        {
            "project_id": project.project_id,
            "sample_rate": project.sample_rate,
            "channels": project.channels,
            "duration_ms": project.duration_ms,
            "target_lufs": project.target_lufs,
            "output": project.output.__dict__,
            "compressor": project.compressor.__dict__,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "source": track.source,
                    "gain_db": track.gain_db,
                    "muted": track.muted,
                    "events": [event.__dict__ for event in track.events],
                }
                for track in project.tracks
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return checksum_bytes(payload)


def run_project(
    project_path: str | Path,
    log: StructuredLogger | None = None,
    fault_after: int | None = None,
) -> RenderResult:
    project = parse_project(project_path)
    project_root = Path(project_path).resolve().parent
    output_path = Renderer(project, project_root)._resolve_output(project.output.path)

    store = CheckpointStore(output_path, project_checksum(project), log or _stderr_logger())
    resumed = store.load()
    if resumed:
        store.log.info("valid checkpoint found; resuming from cached tracks", "I_STATE_RESUME")

    renderer = Renderer(project, project_root, log=store.log, fault_after=fault_after)
    result = renderer.run(store)
    store.cleanup()
    store.log.info(
        "render completed",
        "I_RENDER_DONE",
        output=str(result.output_path),
        index=str(result.index_path),
        measured_lufs=(
            round(result.measured_lufs, 3)
            if result.measured_lufs == result.measured_lufs
            else None
        ),
    )
    return result


def _stderr_logger() -> StructuredLogger:
    from .logging_util import logger

    return logger


def checkpoint_file(project_path: str | Path) -> Path:
    """Utility for tests: expected checkpoint location for a project."""
    project = parse_project(project_path)
    project_root = Path(project_path).resolve().parent
    output_path = Renderer(project, project_root)._resolve_output(project.output.path)
    return state_path_for(output_path)
