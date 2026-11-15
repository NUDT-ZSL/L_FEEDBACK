"""Deterministic JSON report rendering."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import __version__
from .pipeline import FileResult, OUT_BIT_DEPTH, OUT_CHANNELS, OUT_RATE, Params


def build_report(
    results: list[FileResult], params: Params, jobs: int
) -> dict:
    files = []
    for r in results:
        entry = {
            "index": r.index,
            "input": r.path,
            "sha256_16": r.processing.get("sha256_16"),
            "status": r.status,
            "flags": sorted(r.flags),
            "metadata": r.metadata,
            "measured": r.measured or None,
            "processing": {
                k: v for k, v in r.processing.items() if k != "sha256_16"
            },
            "output": r.output,
            "error": r.error,
        }
        files.append(entry)
    return {
        "tool": "loudnorm-batch",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "target_lufs": params.target_lufs,
            "true_peak_ceiling_dbtp": params.ceiling_dbtp,
            "gain_range_db": [params.gain_min_db, params.gain_max_db],
            "total_duration_ms": params.total_ms,
            "jobs": jobs,
            "per_file_timeout_s": params.timeout_s,
            "output_format": {
                "container": "WAV",
                "codec": "PCM",
                "sample_rate": OUT_RATE,
                "bit_depth": OUT_BIT_DEPTH,
                "channels": OUT_CHANNELS,
            },
        },
        "files": files,
    }


def render(report: dict) -> str:
    """Serialize deterministically (byte-identical except generated_at)."""
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
