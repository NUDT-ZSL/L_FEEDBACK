"""End-to-end terrain generation pipeline and parameter sweep orchestration."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import TerrainConfig
from .erosion import simulate
from .io_utils import (
    encode_height16,
    encode_material16,
    encode_normal_rgb,
    write_json,
    write_png,
)
from .materials import MATERIAL_NAMES, classify_materials, material_statistics
from .noise import generate_heightfield
from .normals import compute_normals, encode_normals_rgb


def _height_statistics(heightfield: np.ndarray) -> dict[str, float]:
    return {
        "min": float(heightfield.min()),
        "max": float(heightfield.max()),
        "mean": float(heightfield.mean()),
        "std": float(heightfield.std()),
        "sum": float(heightfield.sum()),
    }


def run_pipeline(config: TerrainConfig, iterations: int | None = None) -> dict:
    """Generate every artifact for one configuration and write them to disk."""
    iterations = config.iterations if iterations is None else iterations
    initial = generate_heightfield(config.width, config.height, config.noise)
    eroded, diagnostics = simulate(
        initial, config.erosion, iterations, fault_at=config.fault_at
    )

    normals = compute_normals(eroded)
    normal_rgb = encode_normals_rgb(normals)
    material = classify_materials(eroded)
    materials = material_statistics(material)

    prefix = Path(config.output_prefix)
    height_path = prefix.with_name(prefix.name + "_height.png")
    normal_path = prefix.with_name(prefix.name + "_normal.png")
    material_path = prefix.with_name(prefix.name + "_material.png")
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")

    write_png(encode_height16(eroded), height_path)
    write_png(encode_normal_rgb(normal_rgb), normal_path)
    write_png(encode_material16(material), material_path)

    metadata = {
        "config": asdict(config),
        "iterations_executed": iterations,
        "height_stats_initial": _height_statistics(initial),
        "height_stats_final": _height_statistics(eroded),
        "convergence": diagnostics,
        "material_names": list(MATERIAL_NAMES),
        "material_stats": materials,
        "outputs": {
            "height_png": str(height_path),
            "normal_png": str(normal_path),
            "material_png": str(material_path),
            "metadata_json": str(metadata_path),
        },
    }
    write_json(metadata, metadata_path)
    return {
        "ok": True,
        "iterations": iterations,
        "output_prefix": str(prefix),
        "outputs": metadata["outputs"],
        "metadata": metadata,
    }


def run_sweep(config: TerrainConfig) -> dict:
    """Run one pipeline per iteration count, isolating per-run failures.

    Iteration counts are executed in strictly ascending order. Any exception
    raised by a run is captured with its type and message, then the sweep
    continues. The overall exit code is 0 only when every run succeeds.
    """
    counts = sorted(config.sweep_iterations)
    base_prefix = Path(config.output_prefix)
    entries = []
    failures = 0

    for count in counts:
        run_prefix = base_prefix.with_name(f"{base_prefix.name}_it{count}")
        run_config = TerrainConfig(
            width=config.width,
            height=config.height,
            iterations=count,
            output_prefix=str(run_prefix),
            noise=config.noise,
            erosion=config.erosion,
            sweep_iterations=(),
            fault_at=config.fault_at,
        )
        try:
            result = run_pipeline(run_config)
            entries.append(
                {"iterations": count, "status": "success", "outputs": result["outputs"]}
            )
        except Exception as exc:  # noqa: BLE001 - failure isolation is the feature
            failures += 1
            entries.append(
                {
                    "iterations": count,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    summary = {
        "config": asdict(config),
        "sweep_iterations": counts,
        "total_runs": len(counts),
        "failed_runs": failures,
        "exit_code": 1 if failures else 0,
        "results": entries,
    }
    summary_path = base_prefix.with_name(base_prefix.name + "_sweep_summary.json")
    write_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary
