"""Batch orchestration: manifest parsing, concurrency, isolation, recovery."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .errors import DubReplaceError, InternalError
from .processor import process_animation
from .state import JobState, STATUS_FAILED, STATUS_RUNNING, STATUS_SUCCESS
from .util import atomic_write_json


@dataclass
class JobItem:
    animation_id: str
    meta_path: Path
    output_path: Path
    report_path: Path


def load_manifest(manifest_path: str | Path) -> tuple[list[JobItem], int, Path]:
    """Parse the job manifest. Returns (items, concurrency, job_dir)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        from .errors import MissingAudioError

        raise MissingAudioError(f"作业清单不存在: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        from .errors import MetadataError

        raise MetadataError(f"作业清单 JSON 解析失败: {manifest_path} ({exc})") from exc

    job_dir = Path(raw.get("job_dir", manifest_path.parent))
    if not job_dir.is_absolute():
        job_dir = (manifest_path.parent / job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)

    concurrency = int(raw.get("concurrency", 1))
    if concurrency < 1:
        concurrency = 1

    items: list[JobItem] = []
    seen_ids: set[str] = set()
    for entry in raw.get("animations", []):
        animation_id = str(entry["animation_id"])
        if animation_id in seen_ids:
            from .errors import MetadataError

            raise MetadataError(f"作业清单中 animation_id 重复: {animation_id}")
        seen_ids.add(animation_id)
        meta_rel = Path(str(entry["meta"]))
        meta_path = meta_rel if meta_rel.is_absolute() else (manifest_path.parent / meta_rel)
        output_rel = entry.get("output")
        report_rel = entry.get("report")
        item_dir = job_dir / animation_id
        output_path = (
            (manifest_path.parent / output_rel)
            if output_rel
            else item_dir / f"{animation_id}_dialect.wav"
        )
        report_path = (
            (manifest_path.parent / report_rel)
            if report_rel
            else job_dir / "reports" / f"{animation_id}.json"
        )
        items.append(
            JobItem(
                animation_id=animation_id,
                meta_path=meta_path.resolve(),
                output_path=Path(output_path),
                report_path=Path(report_path),
            )
        )
    if not items:
        from .errors import MetadataError

        raise MetadataError("作业清单 animations 为空，没有可处理的动画目录")
    return items, concurrency, job_dir


def _failure_report(item: JobItem, error: DubReplaceError) -> dict:
    return {
        "animation_id": item.animation_id,
        "status": "failed",
        "input": {"original_track": None},
        "output": None,
        "sentences": [],
        "discarded_clips": [],
        "events": [],
        "summary": {"total_lines": 0, "replaced": 0, "fallback": 0, "failed_lines": 0, "warnings": 0},
        "error": error.to_dict(),
    }


def _run_one(item: JobItem, job_dir: Path) -> tuple[str, str, dict | None, dict | None, Path]:
    """Worker body. Never raises: failures are classified and returned."""
    state = JobState.load(job_dir)
    if state.is_success(item.animation_id):
        return item.animation_id, "skipped", None, None, item.report_path

    state.mark(item.animation_id, STATUS_RUNNING)
    work_dir = job_dir / "tmp" / item.animation_id
    try:
        report = process_animation(
            item.meta_path, item.output_path, item.report_path, work_dir
        )
    except DubReplaceError as exc:
        state.record_failure(item.animation_id, exc.to_dict())
        failure = _failure_report(item, exc)
        atomic_write_json(item.report_path, failure)
        print(
            f"[FAIL] {item.animation_id}: {exc.code} {exc.message} | 提示: {exc.hint}",
            file=sys.stderr,
        )
        return item.animation_id, "failed", None, exc.to_dict(), item.report_path
    except Exception as exc:  # defensive: keep batch alive on unexpected errors
        wrapped = InternalError(f"未预期异常: {type(exc).__name__}: {exc}")
        state.record_failure(item.animation_id, wrapped.to_dict())
        failure = _failure_report(item, wrapped)
        atomic_write_json(item.report_path, failure)
        print(f"[FAIL] {item.animation_id}: {wrapped.message}", file=sys.stderr)
        return item.animation_id, "failed", None, wrapped.to_dict(), item.report_path
    else:
        state.record_success(
            item.animation_id,
            report_path=str(item.report_path),
            output_path=str(item.output_path),
        )
        return item.animation_id, STATUS_SUCCESS, report, None, item.report_path


def run_job(manifest_path: str | Path) -> tuple[int, dict]:
    """Execute the whole job. Returns (exit code, job_summary)."""
    items, concurrency, job_dir = load_manifest(manifest_path)
    state = JobState.load(job_dir)

    results: list[dict] = []
    succeeded = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="dub") as pool:
        futures = {pool.submit(_run_one, item, job_dir): item for item in items}
        for future in as_completed(futures):
            animation_id, status, report, error, report_path = future.result()
            if status == "skipped":
                skipped += 1
            elif status == STATUS_SUCCESS:
                succeeded += 1
            else:
                failed += 1
            results.append(
                {
                    "animation_id": animation_id,
                    "status": status,
                    "report_path": str(report_path),
                    "error": error,
                }
            )

    summary = {
        "job_dir": str(job_dir),
        "total": len(items),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "state_version": state.version,
        "results": results,
    }
    atomic_write_json(job_dir / "job_summary.json", summary)
    exit_code = 0 if failed == 0 else 1
    return exit_code, summary
