"""Concurrent mix requests with per-task memory isolation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import load_config
from .engine import MixEngine, RenderInterrupted
from .errors import MixerError

MAX_CONCURRENT_JOBS = 4


@dataclass
class BatchJob:
    job_id: str
    config_path: str
    output_path: str
    state_file: Optional[str] = None


def _run_job(job: BatchJob) -> Dict[str, Any]:
    # Each job builds its own engine instance, hence independent track
    # buffers and FFT workspaces; a failure here stays inside this call.
    config_path = Path(job.config_path)
    config = load_config(config_path)
    engine = MixEngine(config, base_dir=config_path.parent,
                       output_path=job.output_path,
                       state_file=job.state_file)
    result = engine.render()
    return {"job_id": job.job_id, "ok": True,
            "output_path": result.output_path,
            "resumed": result.resumed, "report": result.report}


def process_batch(jobs: List[BatchJob],
                  max_workers: int = MAX_CONCURRENT_JOBS
                  ) -> Dict[str, Any]:
    workers = max(1, min(max_workers, MAX_CONCURRENT_JOBS))
    summaries: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_safe_run, job): job for job in jobs}
        for future in as_completed(future_map):
            summaries.append(future.result())
    order = {job.job_id: i for i, job in enumerate(jobs)}
    summaries.sort(key=lambda item: order[item["job_id"]])
    succeeded = sum(1 for item in summaries if item["ok"])
    return {
        "status": "completed" if succeeded == len(summaries)
        else "completed_with_errors",
        "max_workers": workers,
        "total_jobs": len(jobs),
        "succeeded": succeeded,
        "failed": len(jobs) - succeeded,
        "jobs": summaries,
    }


def _safe_run(job: BatchJob) -> Dict[str, Any]:
    try:
        return _run_job(job)
    except RenderInterrupted as exc:
        return {"job_id": job.job_id, "ok": False,
                "code": "E_INTERNAL",
                "message": f"任务被中断钩子暂停: {exc.scene_id}",
                "details": {"scene_id": exc.scene_id,
                            "state_file": exc.state_file}}
    except MixerError as exc:
        return {"job_id": job.job_id, "ok": False, **exc.to_dict()}
    except Exception as exc:  # defensive: never break sibling jobs
        return {"job_id": job.job_id, "ok": False,
                "code": "E_INTERNAL", "message": str(exc),
                "details": {"type": type(exc).__name__}}
