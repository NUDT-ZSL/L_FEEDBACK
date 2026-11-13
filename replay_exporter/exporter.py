"""High-level facade tying planning, snapshots, rendering and validation together."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from .engine import render_master
from .errors import ExporterError
from .samplegen import generate_project
from .validator import validate_master


def export(
    timeline_path: str,
    output_dir: str,
    *,
    checkpoint_every: int = 5,
    stop_after_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """Run (or resume) one master export. Raises ExporterError on failure."""
    return render_master(
        timeline_path,
        output_dir,
        checkpoint_every=checkpoint_every,
        stop_after_blocks=stop_after_blocks,
    )


def safe_export(timeline_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """Export that always returns a JSON-serializable result, never raises."""
    try:
        return render_master(timeline_path, output_dir, **kwargs)
    except ExporterError as error:
        return error.to_dict()


def validate(output_dir: str, timeline_path: Optional[str] = None) -> Dict[str, Any]:
    return validate_master(output_dir, timeline_path=timeline_path)


def generate_sample_project(root: str, **kwargs) -> Dict[str, Any]:
    return generate_project(root, **kwargs)


def dump_json(payload: Dict[str, Any], path: Optional[str] = None) -> str:
    text = json.dumps(payload, sort_keys=True, indent=2, default=str)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return text
