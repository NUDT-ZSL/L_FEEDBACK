"""Bounded-concurrency frame execution."""

from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from .errors import (
    EXIT_GL_CONTEXT,
    EXIT_GL_RESOURCE,
    FRAME_RENDER,
    GL_CONTEXT,
    GL_RESOURCE,
    BusinessError,
)
from .renderer import FrameArtifact, RenderPipeline, frame_failure
from .scene import Scene


def _context_failure(exception: BaseException) -> BusinessError:
    if isinstance(exception, BusinessError):
        return exception
    return BusinessError(
        GL_CONTEXT,
        f"could not initialize a render worker context: {exception}",
        EXIT_GL_CONTEXT,
        {"exception_type": type(exception).__name__},
    )


def iter_frames(
    scene: Scene,
    *,
    backend: str,
    parallel_frames: int,
) -> Iterator[FrameArtifact]:
    """Yield frame artifacts as they finish.

    At most ``parallel_frames`` standalone GL contexts exist.  Frame indices
    beyond that bound remain queued.
    """

    frame_count = len(scene.frame_times)
    if parallel_frames < 1 or parallel_frames > 32:
        raise BusinessError(
            FRAME_RENDER,
            "parallel_frames must be in [1, 32]",
            EXIT_GL_RESOURCE,
            {"parallel_frames": parallel_frames},
        )

    if parallel_frames == 1:
        pipeline = RenderPipeline(scene, backend)
        try:
            for frame_index in range(frame_count):
                try:
                    yield pipeline.render_frame(frame_index)
                except Exception as exc:
                    raise frame_failure(frame_index, exc) from exc
        finally:
            pipeline.close()
        return

    worker_count = min(parallel_frames, frame_count)
    task_queue: queue.Queue[int] = queue.Queue()
    result_queue: queue.Queue[tuple[str, int | None, object]] = queue.Queue()
    stop = threading.Event()
    for frame_index in range(frame_count):
        task_queue.put(frame_index)

    def worker() -> None:
        try:
            pipeline = RenderPipeline(scene, backend)
        except BaseException as exc:
            result_queue.put(("context_error", None, exc))
            stop.set()
            return
        try:
            while not stop.is_set():
                try:
                    frame_index = task_queue.get_nowait()
                except queue.Empty:
                    break
                if stop.is_set():
                    break
                try:
                    artifact = pipeline.render_frame(frame_index)
                except BaseException as exc:
                    result_queue.put(("frame_error", frame_index, exc))
                    stop.set()
                    break
                result_queue.put(("frame", frame_index, artifact))
        finally:
            pipeline.close()

    threads = [threading.Thread(target=worker, name=f"fogcity-render-{i}") for i in range(worker_count)]
    for thread in threads:
        thread.start()

    remaining = frame_count
    first_error: BusinessError | None = None
    try:
        while remaining > 0:
            kind, frame_index, payload = result_queue.get()
            if kind == "context_error":
                first_error = first_error or _context_failure(payload)  # type: ignore[arg-type]
                stop.set()
                break
            if kind == "frame_error":
                assert frame_index is not None
                first_error = first_error or frame_failure(frame_index, payload)  # type: ignore[arg-type]
                stop.set()
                break
            assert frame_index is not None
            remaining -= 1
            yield payload  # type: ignore[misc]
    finally:
        stop.set()
        for thread in threads:
            thread.join()

    if first_error is not None:
        raise first_error

