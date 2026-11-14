"""Multithreaded bake job queue with crash-safe state recovery.

Tasks are submitted in a deterministic order and complete out of order.
Every state transition is persisted atomically (tmp file + os.replace) to
``<output>/_state/bake_state.json`` so a killed process leaves a consistent
snapshot.  On ``--resume``:

* ``done`` tasks are skipped (their outputs are not re-baked),
* ``running`` tasks are treated as interrupted and re-queued,
* ``failed`` tasks are retried.

Fault injection for testing (documented, not part of the public API):

* ``TERRAIN_BAKER_FAIL_TASKS`` = "cx,cy,lod;cx,cy,lod" -> those tasks raise.
* ``TERRAIN_BAKER_CRASH_AT``  = "cx,cy,lod" -> os._exit(3) mid-task.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .consistency import validate_chunk
from .errors import ErrorCode
from .heightfield import HeightField
from .lod_mesh import build_chunk_mesh, select_lod_for_distance
from .normalmap import bake_normal_map

STATE_DIR = "_state"
STATE_FILE = "bake_state.json"


def task_key(cx: int, cy: int, lod: int) -> str:
    return f"{cx},{cy},{lod}"


def config_fingerprint(cfg: Config) -> str:
    payload = {
        "seed": cfg.seed,
        "heightfield": {
            "resolution": cfg.heightfield.resolution,
            "chunk_size": cfg.heightfield.chunk_size,
            "world_size": cfg.heightfield.world_size,
            "min_height": cfg.heightfield.min_height,
            "max_height": cfg.heightfield.max_height,
            "noise": vars(cfg.heightfield.noise),
        },
        "lod_levels": cfg.lod_levels,
        "normal_threshold_deg": cfg.normal_threshold_deg,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class StateStore:
    """Thread-safe, atomically persisted task state."""

    def __init__(self, output_dir: Path):
        self._dir = Path(output_dir) / STATE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / STATE_FILE
        self._lock = threading.Lock()
        self._state = {"version": 1, "config_hash": None, "tasks": {}}

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict:
        with self._lock:
            if self._path.exists():
                try:
                    self._state = json.loads(self._path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    # Corrupt state (e.g. killed mid-write): start clean
                    # rather than crash.
                    self._state = {"version": 1, "config_hash": None, "tasks": {}}
            return self._state

    def _flush(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=1, sort_keys=True),
                       encoding="utf-8")
        os.replace(tmp, self._path)

    def set_config_hash(self, fingerprint: str) -> None:
        with self._lock:
            self._state["config_hash"] = fingerprint
            self._flush()

    def reset(self) -> None:
        with self._lock:
            self._state = {"version": 1, "config_hash": None, "tasks": {}}
            self._flush()

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._state["tasks"].get(key)

    def update(self, key: str, **fields) -> None:
        with self._lock:
            entry = self._state["tasks"].setdefault(key, {})
            entry.update(fields)
            self._flush()

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._state))


def _chunk_dir(output_dir: Path, cx: int, cy: int) -> Path:
    return Path(output_dir) / f"chunk_{cx:03d}_{cy:03d}"


def execute_task(cfg: Config, field: HeightField, output_dir: Path,
                 cx: int, cy: int, lod: int) -> dict:
    """Bake one (chunk, lod): mesh + normal map + consistency check.

    Returns the state-file record for the task.  Raises on unexpected
    failure (the caller records it as a failed task).
    """
    key = task_key(cx, cy, lod)
    # Deterministic fault injection for recovery testing.
    if key in os.environ.get("TERRAIN_BAKER_FAIL_TASKS", "").split(";"):
        raise RuntimeError(f"injected failure for task {key}")
    if os.environ.get("TERRAIN_BAKER_CRASH_AT") == key:
        os._exit(3)

    step = 1 << lod
    mesh = build_chunk_mesh(field, cx, cy, lod)
    image = bake_normal_map(field, cx, cy, step)
    rgb = np.asarray(image)
    result = validate_chunk(mesh, rgb, cfg.normal_threshold_deg)

    chunk_dir = _chunk_dir(output_dir, cx, cy)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = chunk_dir / f"mesh_lod{lod}.npz"
    normal_path = chunk_dir / f"normal_lod{lod}.png"
    mesh.save(mesh_path)
    image.save(normal_path)

    consistency = {
        "max_angle_deg": result.max_angle_deg,
        "mean_angle_deg": result.mean_angle_deg,
        "bad_vertices": result.bad_vertices,
        "threshold_deg": result.threshold_deg,
        "ok": result.ok,
    }
    error = None
    if not result.ok:
        error = {"code": ErrorCode.E_NORMAL_MISMATCH,
                 "message": f"{result.bad_vertices} vertices exceed "
                            f"{result.threshold_deg} deg"}
    return {
        "status": "done",
        "outputs": {"mesh": str(mesh_path), "normal_map": str(normal_path)},
        "consistency": consistency,
        "error": error,
    }


def plan_tasks(cfg: Config) -> list[tuple[int, int, int]]:
    """Deterministic submission order: row-major chunks, ascending LOD."""
    tasks = []
    chunks = cfg.chunks_per_side
    for cy in range(chunks):
        for cx in range(chunks):
            for lod in range(cfg.lod_levels):
                tasks.append((cx, cy, lod))
    return tasks


def selected_lods(cfg: Config) -> dict:
    """Viewpoint-based LOD choice per chunk, recorded in the report."""
    chunks = cfg.chunks_per_side
    cells = cfg.heightfield.chunk_size - 1
    chunk_world = cells * cfg.spacing
    vx, vy, vz = cfg.view.position
    result = {}
    for cy in range(chunks):
        for cx in range(chunks):
            center_x = (cx + 0.5) * chunk_world
            center_y = (cy + 0.5) * chunk_world
            center_z = 0.5 * (cfg.heightfield.min_height + cfg.heightfield.max_height)
            dist = float(np.sqrt((center_x - vx) ** 2
                                 + (center_y - vy) ** 2
                                 + (center_z - vz) ** 2))
            result[f"{cx},{cy}"] = select_lod_for_distance(
                dist, chunk_world, cfg.lod_levels - 1,
                cfg.view.screen_error_threshold_px,
                cfg.view.viewport_height_px, cfg.view.fov_y_deg)
    return result


def run_bake(cfg: Config, output_dir: Path, jobs: int, resume: bool,
             log=print) -> tuple[int, dict]:
    """Run the bake.  Returns (exit_code, final_state)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = StateStore(output_dir)
    state = store.load()
    fingerprint = config_fingerprint(cfg)

    if not resume:
        store.reset()
        state = store.snapshot()
    elif state.get("config_hash") not in (None, fingerprint):
        log("state file belongs to a different configuration; starting fresh")
        store.reset()
        state = store.snapshot()
    store.set_config_hash(fingerprint)

    tasks = plan_tasks(cfg)
    pending: list[tuple[int, int, int]] = []
    for cx, cy, lod in tasks:
        key = task_key(cx, cy, lod)
        entry = store.get(key)
        status = entry.get("status") if entry else None
        if status == "done":
            continue  # never re-bake completed work
        if status in ("running", "failed") and resume:
            note = "interrupted" if status == "running" else "failed"
            log(f"re-queuing {note} task {key}")
        pending.append((cx, cy, lod))

    log(f"tasks: {len(tasks)} total, {len(tasks) - len(pending)} already done, "
        f"{len(pending)} to bake with {jobs} worker(s)")

    # One shared, read-only heightfield: deterministic and thread-safe.
    field = HeightField(cfg)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {}
            for cx, cy, lod in pending:  # ordered submission
                key = task_key(cx, cy, lod)
                store.update(key, status="running", error=None)
                fut = pool.submit(execute_task, cfg, field, output_dir, cx, cy, lod)
                futures[fut] = (cx, cy, lod)
            for fut in as_completed(futures):  # unordered completion
                cx, cy, lod = futures[fut]
                key = task_key(cx, cy, lod)
                try:
                    record = fut.result()
                except Exception as exc:  # noqa: BLE001 - recorded, not raised
                    store.update(key, status="failed",
                                 error={"code": ErrorCode.E_TASK_EXECUTION,
                                        "message": f"{type(exc).__name__}: {exc}"})
                    log(f"task {key} FAILED: {exc}")
                else:
                    store.update(key, **record)
                    log(f"task {key} done")

    final = store.snapshot()
    mismatch = any(
        (e or {}).get("status") == "done"
        and (e.get("consistency") or {}).get("ok") is False
        for e in final["tasks"].values())
    remaining_failed = any((e or {}).get("status") == "failed"
                           for e in final["tasks"].values())
    if remaining_failed:
        return 3, final
    if mismatch:
        return 2, final
    return 0, final
