"""End-to-end pipeline, output transaction, and metadata construction."""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import OutputError
from .model import SceneSpec, load_scene, validate_runtime_options
from .raster import RasterResult, rasterize_scene
from .shading import DEFAULT_NORMAL_RGB, bake_tangent_normal, coverage_to_height


OUTPUT_NAMES = ("albedo.png", "height.png", "normal.png", "metadata.json")


@dataclass(frozen=True)
class PipelineResult:
    outputs: dict[str, str]
    summary: dict[str, Any]
    metadata: dict[str, Any]


class OutputTransaction:
    """Write all outputs to a staging directory, then replace final files."""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.staging_directory: Path | None = None
        self.backup_directory: Path | None = None
        self._committed = False

    def __enter__(self) -> "OutputTransaction":
        try:
            if self.output_directory.exists() and not self.output_directory.is_dir():
                raise OutputError(
                    "Output path exists but is not a directory.",
                    field="output_dir",
                    details={"path": str(self.output_directory)},
                )
            self.output_directory.mkdir(parents=True, exist_ok=True)
            suffix = uuid.uuid4().hex
            self.staging_directory = self.output_directory / f".terrain-raster-{suffix}.tmp"
            self.staging_directory.mkdir()
            self.backup_directory = self.staging_directory / ".backup"
            self.backup_directory.mkdir()
        except OutputError:
            raise
        except OSError as exc:
            raise OutputError(
                "Could not prepare output directory.",
                field="output_dir",
                details={"path": str(self.output_directory), "reason": str(exc)},
            ) from None
        return self

    def staged_path(self, name: str) -> Path:
        if self.staging_directory is None:
            raise RuntimeError("Output transaction is not active")
        return self.staging_directory / name

    def commit(self) -> dict[str, str]:
        if self.staging_directory is None:
            raise RuntimeError("Output transaction is not active")
        installed: list[Path] = []
        backups: dict[Path, Path] = {}
        try:
            for name in OUTPUT_NAMES:
                source = self.staged_path(name)
                if not source.is_file():
                    raise OSError(f"Staged output is missing: {name}")
                final = self.output_directory / name
                if final.exists():
                    backup = self.backup_directory / name
                    os.replace(final, backup)
                    backups[final] = backup
                os.replace(source, final)
                installed.append(final)
        except OSError as exc:
            for final in reversed(installed):
                try:
                    final.unlink(missing_ok=True)
                except OSError:
                    pass
            for final, backup in backups.items():
                try:
                    if backup.exists():
                        os.replace(backup, final)
                except OSError:
                    pass
            raise OutputError(
                "Could not commit output files.",
                field="output_dir",
                details={"path": str(self.output_directory), "reason": str(exc)},
            ) from None

        self._committed = True
        try:
            shutil.rmtree(self.staging_directory)
        except OSError:
            pass
        return {
            name: str((self.output_directory / name).resolve()) for name in OUTPUT_NAMES
        }

    def abort(self) -> None:
        if self.staging_directory is not None:
            shutil.rmtree(self.staging_directory, ignore_errors=True)
            self.staging_directory = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if not self._committed:
            self.abort()
        return False


def _write_png(path: Path, array: Any, mode: str) -> None:
    try:
        Image.fromarray(array, mode=mode).save(path, format="PNG", optimize=False)
    except OSError as exc:
        raise OutputError(
            "Could not write PNG output.",
            field="output_dir",
            details={"path": str(path), "reason": str(exc)},
        ) from None


def _build_metadata(
    scene: SceneSpec,
    raster: RasterResult,
    height: Any,
    *,
    width: int,
    height_pixels: int,
    lod: str,
    grid_size: int,
    normal_strength: float,
    outputs: dict[str, str],
) -> dict[str, Any]:
    by_id = {layer.id: layer for layer in scene.layers}
    layers: list[dict[str, Any]] = []
    for result in raster.layers:
        spec = by_id[result.id]
        layers.append(
            {
                "id": result.id,
                "order": result.order,
                "blend": "over",
                "alpha": spec.alpha,
                "color": list(spec.color),
                "bbox": list(result.bbox) if result.bbox is not None else None,
                "nontransparent_pixels": result.nontransparent_pixels,
                "self_intersecting": result.self_intersecting,
                "lod": result.lod,
                "vertex_counts": {
                    "high": result.vertex_counts["high"],
                    "medium": result.vertex_counts["medium"],
                    "low": result.vertex_counts["low"],
                    "selected": sum(
                        int(shape["vertex_count"]) for shape in result.shapes
                    ),
                },
                "shapes": list(result.shapes),
            }
        )

    return {
        "schema_version": 1,
        "pipeline": "terrain-pbr-vector-rasterizer",
        "scene_name": scene.name,
        "input": str(scene.source_path.resolve()),
        "lod": lod,
        "canvas": {"width": width, "height": height_pixels},
        "grid": {
            "size": grid_size,
            "tiles_x": raster.tiles_x,
            "tiles_y": raster.tiles_y,
            "tiles_processed": raster.stats.tiles_processed,
        },
        "peak_rasterized_pixels": raster.stats.peak_rasterized_pixels,
        "total_rasterized_pixels": raster.stats.total_rasterized_pixels,
        "peak_candidate_pixels": raster.stats.peak_candidate_pixels,
        "layer_order": [layer.id for layer in scene.layers],
        "layers": layers,
        "shading": {
            "height_source": "maximum geometric raster coverage over all layers",
            "height_range": [0, 255],
            "sobel_kernel": "3x3",
            "normal_strength": normal_strength,
            "normal_encoding": "RGB8, tangent-space, +Y follows image rows",
            "default_normal_rgb": list(DEFAULT_NORMAL_RGB),
            "nonzero_height_pixels": int(
                (height > 0).sum() if hasattr(height, "sum") else 0
            ),
        },
        "outputs": {
            "albedo": outputs["albedo.png"],
            "height": outputs["height.png"],
            "normal": outputs["normal.png"],
            "metadata": outputs["metadata.json"],
        },
    }


def run_pipeline(
    *,
    input_path: str | Path,
    output_directory: str | Path,
    width: Any,
    height: Any,
    lod: Any,
    grid_size: Any,
) -> PipelineResult:
    started = time.perf_counter()
    width_int, height_int, lod_value, grid_int = validate_runtime_options(
        width=width,
        height=height,
        lod=lod,
        grid_size=grid_size,
    )
    scene = load_scene(input_path)
    raster = rasterize_scene(
        scene,
        width=width_int,
        height=height_int,
        lod=lod_value,
        grid_size=grid_int,
    )
    height_map = coverage_to_height(raster.height_coverage)
    normal_map, gradient_x, gradient_y = bake_tangent_normal(
        height_map, strength=scene.normal_strength
    )

    with OutputTransaction(output_directory) as transaction:
        _write_png(transaction.staged_path("albedo.png"), raster.albedo_rgba, "RGBA")
        _write_png(transaction.staged_path("height.png"), height_map, "L")
        _write_png(transaction.staged_path("normal.png"), normal_map, "RGB")

        # Paths are stable output names at this point; final absolute paths are
        # filled into the metadata once the transaction commits.
        provisional_paths = {
            name: str((Path(output_directory) / name).resolve())
            for name in OUTPUT_NAMES
        }
        metadata = _build_metadata(
            scene,
            raster,
            height_map,
            width=width_int,
            height_pixels=height_int,
            lod=lod_value,
            grid_size=grid_int,
            normal_strength=scene.normal_strength,
            outputs=provisional_paths,
        )
        metadata["rasterization"] = {
            "algorithm": "scanline interior + fixed 4x4 edge coverage",
            "coverage_levels": 17,
            "line_antialiasing": "fixed 4x4 capsule coverage",
            "peak_rasterized_pixels_definition": (
                "largest per-shape tile pixel buffer allocated during rasterization"
            ),
        }
        metadata["normal_gradient"] = {
            "zero_gradient_pixels": int(
                ((gradient_x == 0.0) & (gradient_y == 0.0)).sum()
            )
        }
        with transaction.staged_path("metadata.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        outputs = transaction.commit()

    elapsed = time.perf_counter() - started
    summary = {
        "ok": True,
        "outputs": outputs,
        "elapsed_seconds": round(elapsed, 6),
        "canvas": {"width": width_int, "height": height_int},
        "lod": lod_value,
        "grid_size": grid_int,
        "peak_rasterized_pixels": raster.stats.peak_rasterized_pixels,
        "nontransparent_pixels": int(
            (raster.albedo_rgba[:, :, 3] > 0).sum()
        ),
    }
    return PipelineResult(outputs=outputs, summary=summary, metadata=metadata)
