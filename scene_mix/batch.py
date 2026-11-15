from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import SCHEMA_VERSION
from .io_utils import read_json, write_json
from .pipeline import mix_group


@dataclass(frozen=True)
class BatchJob:
    name: str
    input_dir: str
    output_dir: str


def _read_progress(path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = read_json(path)
        if isinstance(payload, dict):
            return payload
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return None


def _write_progress(path: Path, payload: Dict[str, object]) -> None:
    write_json(path, payload)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _completed(output_dir: Path) -> bool:
    return (
        (output_dir / "report.json").is_file()
        and (output_dir / "mix.wav").is_file()
        and not (output_dir / ".progress").exists()
    )


def _run_job(job: BatchJob) -> Dict[str, object]:
    input_dir = Path(job.input_dir)
    output_dir = Path(job.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / ".progress"
    previous = _read_progress(progress_path)
    prior_attempts = int(previous.get("attempts", 0)) if previous else 0
    attempts = prior_attempts + 1
    retry_count = attempts - 1
    progress = {
        "schema_version": SCHEMA_VERSION,
        "group": job.name,
        "status": "running",
        "attempts": attempts,
        "retry_count": retry_count,
        "pid": os.getpid(),
        "started_at_utc": _utc_now(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
    _write_progress(progress_path, progress)

    # This hook is only for deterministic recovery tests; it defaults to zero.
    test_delay_ms = int(os.environ.get("SCENE_MIX_TEST_DELAY_MS", "0") or "0")
    if test_delay_ms > 0:
        time.sleep(test_delay_ms / 1000.0)

    try:
        report = mix_group(
            input_dir / "timeline.json",
            output_dir,
            retry_count=retry_count,
        )
        # mix_group is deliberately retry-agnostic so a normal single-group
        # invocation is deterministic; the batch layer owns the attempt count.
        if report.get("retry_count") != retry_count:
            raise RuntimeError("report retry count mismatch")
        progress_path.unlink(missing_ok=True)
        return {
            "name": job.name,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "status": "success",
            "retry_count": retry_count,
            "warnings": report.get("warnings", []),
        }
    except BaseException as exc:
        failed = dict(progress)
        failed["status"] = "failed"
        failed["finished_at_utc"] = _utc_now()
        failed["error"] = f"{type(exc).__name__}: {exc}"
        failed["traceback"] = traceback.format_exc(limit=8)
        _write_progress(progress_path, failed)
        return {
            "name": job.name,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "status": "failed",
            "retry_count": retry_count,
            "error": failed["error"],
        }


def discover_groups(input_root: Path) -> List[BatchJob]:
    input_root = Path(input_root)
    if not input_root.exists():
        raise ValueError(f"input root does not exist: {input_root}")
    if (input_root / "timeline.json").is_file():
        return [
            BatchJob(
                name=input_root.name,
                input_dir=str(input_root),
                output_dir="",
            )
        ]

    jobs: List[BatchJob] = []
    for timeline in sorted(input_root.rglob("timeline.json")):
        group_dir = timeline.parent
        relative = group_dir.relative_to(input_root)
        jobs.append(
            BatchJob(
                name=str(relative).replace(os.sep, "/"),
                input_dir=str(group_dir),
                output_dir=str(relative),
            )
        )
    return jobs


def run_batch(
    input_root: Path,
    output_root: Path,
    jobs_requested: int = 4,
) -> Dict[str, object]:
    input_root = Path(input_root)
    output_root = Path(output_root)
    cpu_count = max(1, os.cpu_count() or 1)
    if jobs_requested < 1:
        raise ValueError("jobs must be at least 1")
    jobs_used = min(int(jobs_requested), cpu_count)

    discovered = discover_groups(input_root)
    if not discovered:
        raise ValueError(f"no timeline.json found under {input_root}")

    pending: List[BatchJob] = []
    results: List[Dict[str, object]] = []
    for job in discovered:
        if (input_root / "timeline.json").is_file():
            output_dir = output_root
        else:
            output_dir = output_root / job.output_dir
        concrete = BatchJob(job.name, job.input_dir, str(output_dir))
        if _completed(output_dir):
            report = read_json(output_dir / "report.json")
            results.append(
                {
                    "name": concrete.name,
                    "input_dir": concrete.input_dir,
                    "output_dir": concrete.output_dir,
                    "status": "skipped_completed",
                    "retry_count": int(report.get("retry_count", 0)),
                    "warnings": report.get("warnings", []),
                }
            )
        else:
            pending.append(concrete)

    output_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=jobs_used) as executor:
        futures = {executor.submit(_run_job, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(
                    {
                        "name": job.name,
                        "input_dir": job.input_dir,
                        "output_dir": job.output_dir,
                        "status": "failed",
                        "retry_count": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    results.sort(key=lambda item: str(item["name"]))
    success = sum(
        1 for item in results if item["status"] in ("success", "skipped_completed")
    )
    failed = len(results) - success
    summary = {
        "schema_version": SCHEMA_VERSION,
        "jobs_requested": int(jobs_requested),
        "jobs_used": int(jobs_used),
        "cpu_logical_cores": int(cpu_count),
        "groups_total": len(results),
        "success": success,
        "failed": failed,
        "groups": results,
    }
    write_json(output_root / "batch_summary.json", summary)
    return summary
