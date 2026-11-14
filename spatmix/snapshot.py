"""Snapshot save/restore for bit-identical resume."""

import hashlib
import json
import os

from . import SNAPSHOT_VERSION
from .errors import PipelineError

SNAPSHOT_NAME = "snapshot.json"


def scene_digest(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def save_snapshot(engine, scene_path, path=None):
    """Serialize the full processing state to a versioned JSON file."""
    if path is None:
        path = os.path.join(engine.out_dir, SNAPSHOT_NAME)
    state = {
        "version": SNAPSHOT_VERSION,
        "scene_path": os.path.abspath(scene_path),
        "scene_sha256": scene_digest(scene_path),
        "out_dir": os.path.abspath(engine.out_dir),
        "frames_completed": engine.frames_completed,
        "total_frames": engine.total_frames,
        "sample_rate": engine.sr,
        "frame_size": engine.frame_size,
        "layout": engine.layout,
        "mix_buffer_position": engine.frames_completed * engine.frame_size,
        "sources": {s.spec["id"]: s.state_dict() for s in engine.sources},
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, path)
    engine.emit(type="snapshot_written", file=os.path.basename(path),
                frames_completed=engine.frames_completed)
    return path


def load_snapshot(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except json.JSONDecodeError as exc:
        raise PipelineError("E_SCHEMA", f"{path}: invalid snapshot JSON: {exc}") from exc
    except OSError as exc:
        raise PipelineError("E_IO", f"{path}: {exc}") from exc
    if not isinstance(state, dict) or "version" not in state:
        raise PipelineError("E_SCHEMA", f"{path}: not a snapshot file")
    if state["version"] != SNAPSHOT_VERSION:
        raise PipelineError(
            "E_SNAPSHOT_VERSION",
            f"{path}: snapshot version {state['version']!r} does not match "
            f"supported version {SNAPSHOT_VERSION}",
        )
    return state


def verify_scene_match(state, scene_path):
    digest = scene_digest(scene_path)
    if digest != state["scene_sha256"]:
        raise PipelineError(
            "E_SNAPSHOT_MISMATCH",
            f"scene file {scene_path} does not match the scene hash stored "
            f"in the snapshot",
        )
