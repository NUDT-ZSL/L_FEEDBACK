"""Deterministic parallel processing of independent WAV/timeline jobs."""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from .errors import (
    BATCH_INVALID,
    INPUT_MISSING,
    RepairError,
)
from .pipeline import run_job


def _json_bytes(document: Dict[str, Any]) -> bytes:
    text = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _atomic_bytes(path: str, payload: bytes) -> None:
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".audiorepair-batch-", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def load_batch(
    manifest_path: str,
    output_dir: Optional[str] = None,
) -> List[Dict[str, str]]:
    if not os.path.isfile(manifest_path):
        raise RepairError(
            INPUT_MISSING, "batch manifest not found: {}".format(manifest_path)
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RepairError(
            BATCH_INVALID,
            "invalid batch JSON at line {} column {}: {}".format(
                exc.lineno, exc.colno, exc.msg
            ),
        )
    except UnicodeDecodeError as exc:
        raise RepairError(
            BATCH_INVALID, "batch manifest is not valid UTF-8: {}".format(exc)
        )
    except OSError as exc:
        raise RepairError(INPUT_MISSING, "cannot read batch manifest: {}".format(exc))
    if not isinstance(document, dict) or not isinstance(document.get("jobs"), list):
        raise RepairError(BATCH_INVALID, "batch manifest must contain a jobs array")
    if not document["jobs"]:
        raise RepairError(BATCH_INVALID, "batch jobs array must not be empty")

    base = os.path.dirname(os.path.abspath(manifest_path))
    jobs: List[Dict[str, str]] = []
    outputs = set()
    reports = set()
    for index, item in enumerate(document["jobs"]):
        if not isinstance(item, dict):
            raise RepairError(
                BATCH_INVALID, "jobs[{}] must be an object".format(index)
            )
        job_id = str(item.get("id", "job-{}".format(index)))
        input_value = item.get("input")
        timeline_value = item.get("timeline")
        if not isinstance(input_value, str) or not isinstance(timeline_value, str):
            raise RepairError(
                BATCH_INVALID,
                "jobs[{}] requires string input and timeline paths".format(index),
            )
        input_path = (
            input_value
            if os.path.isabs(input_value)
            else os.path.join(base, input_value)
        )
        timeline_path = (
            timeline_value
            if os.path.isabs(timeline_value)
            else os.path.join(base, timeline_value)
        )

        if output_dir is not None:
            target_dir = os.path.abspath(output_dir)
            output_path = os.path.join(target_dir, "{}.wav".format(job_id))
            report_path = os.path.join(
                target_dir, "{}.report.json".format(job_id)
            )
        else:
            output_value = item.get("output", "out/{}.wav".format(job_id))
            report_value = item.get(
                "report", "out/{}.report.json".format(job_id)
            )
            if not isinstance(output_value, str) or not isinstance(report_value, str):
                raise RepairError(
                    BATCH_INVALID,
                    "jobs[{}] output and report must be strings".format(index),
                )
            output_path = (
                output_value
                if os.path.isabs(output_value)
                else os.path.join(base, output_value)
            )
            report_path = (
                report_value
                if os.path.isabs(report_value)
                else os.path.join(base, report_value)
            )
        output_key = os.path.abspath(output_path)
        report_key = os.path.abspath(report_path)
        if output_key in outputs or report_key in reports or output_key == report_key:
            raise RepairError(
                BATCH_INVALID,
                "jobs[{}] has a duplicate output/report path".format(index),
            )
        outputs.add(output_key)
        reports.add(report_key)
        jobs.append(
            {
                "id": job_id,
                "input": input_path,
                "timeline": timeline_path,
                "output": output_path,
                "report": report_path,
            }
        )
    return jobs


def _run_one(job: Dict[str, str]) -> Dict[str, Any]:
    try:
        result = run_job(
            job["input"], job["timeline"], job["output"], job["report"]
        )
        return {
            "id": job["id"],
            "status": "ok",
            "error_code": None,
            "message": "",
            "output": os.path.abspath(job["output"]),
            "report": os.path.abspath(job["report"]),
            "integrated_loudness_lufs": result["integrated_loudness_lufs"],
            "max_boundary_error_ms": result["max_boundary_error_ms"],
        }
    except RepairError as exc:
        return {
            "id": job["id"],
            "status": "error",
            "error_code": exc.code,
            "message": exc.message,
            "details": exc.details,
            "output": None,
            "report": None,
        }
    except Exception as exc:  # isolate unexpected failures to one job
        return {
            "id": job["id"],
            "status": "error",
            "error_code": "PROCESSING_FAILED",
            "message": "{}: {}".format(type(exc).__name__, exc),
            "details": {},
            "output": None,
            "report": None,
        }


def run_batch(
    jobs: List[Dict[str, str]],
    thread_count: int,
    summary_path: str,
) -> Dict[str, Any]:
    """Execute jobs concurrently and always wait for clean pool shutdown."""
    workers = max(1, int(thread_count))
    results: List[Optional[Dict[str, Any]]] = [None] * len(jobs)
    executor = ThreadPoolExecutor(
        max_workers=min(workers, len(jobs)),
        thread_name_prefix="audiorepair",
    )
    futures: Dict[Future, int] = {}
    try:
        for index, job in enumerate(jobs):
            futures[executor.submit(_run_one, job)] = index
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    ordered = [result for result in results if result is not None]
    succeeded = sum(1 for result in ordered if result["status"] == "ok")
    failed = len(ordered) - succeeded
    summary = {
        "version": 1,
        "status": "complete" if failed == 0 else "completed_with_errors",
        "thread_count": workers,
        "summary": {
            "total": len(jobs),
            "succeeded": succeeded,
            "failed": failed,
        },
        "jobs": ordered,
    }
    try:
        _atomic_bytes(summary_path, _json_bytes(summary))
    except OSError as exc:
        raise RepairError(
            "PROCESSING_FAILED",
            "cannot write batch summary: {}".format(exc),
        )
    return summary

