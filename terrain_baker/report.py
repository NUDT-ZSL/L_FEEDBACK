"""JSON bake report generation."""

from __future__ import annotations

import json
from pathlib import Path

from . import __version__
from .config import Config
from .errors import ErrorCode
from .jobs import plan_tasks, selected_lods, task_key


def build_report(cfg: Config, state: dict) -> dict:
    """Assemble the report from the final task state."""
    lod_selection = selected_lods(cfg)
    chunks: dict[tuple[int, int], dict] = {}
    for cx, cy, lod in plan_tasks(cfg):
        key = task_key(cx, cy, lod)
        entry = (state.get("tasks") or {}).get(key) or {}
        chunk = chunks.setdefault((cx, cy), {
            "chunk": [cx, cy],
            "selected_lod": lod_selection[f"{cx},{cy}"],
            "lods": [],
            "status": ErrorCode.OK,
        })
        lod_entry = {
            "level": lod,
            "status": entry.get("status", "pending"),
            "mesh": (entry.get("outputs") or {}).get("mesh"),
            "normal_map": (entry.get("outputs") or {}).get("normal_map"),
            "consistency": entry.get("consistency"),
            "error": entry.get("error"),
        }
        chunk["lods"].append(lod_entry)
        if entry.get("status") == "failed":
            chunk["status"] = (entry.get("error") or {}).get(
                "code", ErrorCode.E_TASK_EXECUTION)
        elif (entry.get("consistency") or {}).get("ok") is False:
            chunk["status"] = ErrorCode.E_NORMAL_MISMATCH

    chunk_list = [chunks[k] for k in sorted(chunks)]
    for chunk in chunk_list:
        chunk["lods"].sort(key=lambda e: e["level"])

    all_entries = list((state.get("tasks") or {}).values())
    summary = {
        "total_tasks": len(plan_tasks(cfg)),
        "done": sum(1 for e in all_entries if (e or {}).get("status") == "done"),
        "failed": sum(1 for e in all_entries if (e or {}).get("status") == "failed"),
        "consistency_errors": sum(
            1 for e in all_entries
            if (e or {}).get("status") == "done"
            and (e.get("consistency") or {}).get("ok") is False),
        "max_angle_deg": max(
            ((e.get("consistency") or {}).get("max_angle_deg", 0.0)
             for e in all_entries if e),
            default=0.0),
    }
    return {
        "generator": f"terrain_baker {__version__}",
        "seed": cfg.seed,
        "lod_levels": cfg.lod_levels,
        "normal_threshold_deg": cfg.normal_threshold_deg,
        "summary": summary,
        "chunks": chunk_list,
    }


def write_report(cfg: Config, state: dict, output_dir: Path) -> Path:
    report = build_report(cfg, state)
    path = Path(output_dir) / "report.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return path
