"""Threaded tiled deferred renderer with snapshots and incremental recovery."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from typing import Any

from .errors import E_TILE_FAILED, PipelineError, SnapshotCorruptError
from .gbuffer import GBufferTile, material_shading, rasterize_tile
from .lighting import TileShading, shade_tile
from .scene import Scene, ShadowConfig, TileSpec, load_scene
from .shadows import ShadowMaps, build_shadow_maps, estimate_tile_shadow_bytes
from .snapshot import Snapshot, SnapshotStore, load_snapshot


@dataclass
class PipelineOptions:
    quality: str | None = None
    tile_width: int | None = None
    tile_height: int | None = None
    workers: int = 4
    snapshot_dir: str | Path | None = None
    incremental: bool = True
    report_path: str | Path | None = None
    memory_limit_bytes: int | None = None
    memory_limit_mb: int | None = None
    pcf_kernel: int | None = None
    shadow_resolution: int | None = None
    directional_resolution: int | None = None
    point_resolution: int | None = None

    def __post_init__(self) -> None:
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.quality is not None and self.quality not in {"high", "fast"}:
            raise ValueError("quality must be 'high' or 'fast'")
        for name in (
            "tile_width",
            "tile_height",
            "memory_limit_bytes",
            "memory_limit_mb",
            "shadow_resolution",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pcf_kernel is not None and self.pcf_kernel <= 0:
            raise ValueError("pcf_kernel must be positive")


@dataclass
class RenderResult:
    scene: Scene
    frame: list[float]
    gbuffer_tiles: list[GBufferTile]
    report: dict[str, Any]
    ppm_bytes: bytes

    @property
    def failed_tiles(self) -> list[dict[str, Any]]:
        return list(self.report.get("failed_tiles", []))


@dataclass
class ReferenceResult:
    scene: Scene
    gbuffer: GBufferTile
    shadow_maps: ShadowMaps
    shading: TileShading
    frame: list[float]


@dataclass(frozen=True)
class _TilePlan:
    tile: TileSpec
    reuse_gbuffer: bool
    reuse_shadow_maps: bool
    cached: Snapshot | None
    reasons: tuple[str, ...]


@dataclass
class _TileResult:
    plan: _TilePlan
    gbuffer: GBufferTile | None = None
    shadow_maps: ShadowMaps | None = None
    shading: TileShading | None = None
    material_dependencies: tuple[str, ...] = ()
    light_dependencies: tuple[str, ...] = ()
    error: dict[str, str] | None = None
    snapshot_checksum: str | None = None
    snapshot_path: str | None = None
    snapshot_written: bool = False
    snapshot_error: dict[str, str] | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.gbuffer is not None and self.shading is not None


def _validate_quality(quality: str) -> str:
    normalized = quality.lower()
    if normalized in {"hq", "full", "high_quality"}:
        normalized = "high"
    if normalized in {"low", "single", "fast_mode"}:
        normalized = "fast"
    if normalized not in {"high", "fast"}:
        raise ValueError("quality must be 'high' or 'fast'")
    return normalized


def apply_options(scene: Scene, options: PipelineOptions) -> Scene:
    tile_width = scene.tile_width if options.tile_width is None else min(scene.width, options.tile_width)
    tile_height = scene.tile_height if options.tile_height is None else min(scene.height, options.tile_height)
    quality = scene.shadow.quality if options.quality is None else _validate_quality(options.quality)
    directional_resolution = scene.shadow.directional_resolution
    point_resolution = scene.shadow.point_resolution
    if options.shadow_resolution is not None:
        directional_resolution = options.shadow_resolution
        point_resolution = options.shadow_resolution
    if options.directional_resolution is not None:
        directional_resolution = options.directional_resolution
    if options.point_resolution is not None:
        point_resolution = options.point_resolution
    pcf_kernel = scene.shadow.pcf_kernel
    if options.pcf_kernel is not None:
        pcf_kernel = options.pcf_kernel if options.pcf_kernel % 2 == 1 else options.pcf_kernel + 1
    shadow = ShadowConfig(
        quality=quality,
        directional_resolution=directional_resolution,
        point_resolution=point_resolution,
        pcf_kernel=pcf_kernel,
        bias=scene.shadow.bias,
        normal_bias=scene.shadow.normal_bias,
        depth_epsilon=scene.shadow.depth_epsilon,
    )
    if options.memory_limit_bytes is not None:
        memory_limit = options.memory_limit_bytes
    elif options.memory_limit_mb is not None:
        memory_limit = options.memory_limit_mb * 1024 * 1024
    else:
        memory_limit = scene.memory_limit_bytes
    return replace(
        scene,
        tile_width=tile_width,
        tile_height=tile_height,
        shadow=shadow,
        memory_limit_bytes=memory_limit,
    )


def _memory_limit_error(scene: Scene, tile: TileSpec) -> PipelineError | None:
    if scene.memory_limit_bytes is None:
        return None
    required = tile.width * tile.height * 64 + estimate_tile_shadow_bytes(scene)
    if required > scene.memory_limit_bytes:
        return PipelineError(
            E_TILE_FAILED,
            f"tile requires at least {required} bytes but limit is {scene.memory_limit_bytes}",
            "memory",
        )
    return None


def _as_error(error: BaseException) -> dict[str, str]:
    if isinstance(error, PipelineError):
        return error.as_dict()
    if isinstance(error, MemoryError):
        return {
            "code": E_TILE_FAILED,
            "message": "tile memory allocation failed",
            "stage": "memory",
        }
    return {
        "code": E_TILE_FAILED,
        "message": f"{type(error).__name__}: {error}",
        "stage": "tile",
    }


def _dependency_names(scene: Scene, material_indices: tuple[int, ...], light_ids: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    material_names = []
    for material_index in material_indices:
        material = scene.material(material_index)
        if material is not None:
            material_names.append(material.material_id)
    return tuple(sorted(set(material_names))), tuple(light_ids)


def _run_tile_plan(scene: Scene, plan: _TilePlan, snapshot_store: SnapshotStore | None) -> _TileResult:
    result = _TileResult(plan=plan)
    try:
        memory_error = _memory_limit_error(scene, plan.tile)
        if memory_error is not None:
            raise memory_error
        cached = plan.cached
        if plan.reuse_gbuffer:
            if cached is None:
                raise PipelineError(E_TILE_FAILED, "reuse requested without a snapshot", "snapshot_restore")
            gbuffer = cached.gbuffer
            if plan.reuse_shadow_maps:
                shadow_maps = cached.shadow_maps
            else:
                shadow_maps = build_shadow_maps(scene)
            material_indices = tuple(gbuffer.material_usage())
            material_dependencies, _unused = _dependency_names(scene, material_indices, ())
        else:
            gbuffer = rasterize_tile(scene, plan.tile)
            material_indices = material_shading(scene, gbuffer)
            material_dependencies, _unused = _dependency_names(scene, material_indices, ())
            if plan.reuse_shadow_maps and cached is not None:
                shadow_maps = cached.shadow_maps
            else:
                shadow_maps = build_shadow_maps(scene)
        shading = shade_tile(scene, gbuffer, shadow_maps)
        result.gbuffer = gbuffer
        result.shadow_maps = shadow_maps
        result.shading = shading
        result.material_dependencies = material_dependencies
        result.light_dependencies = shading.light_usage
        if snapshot_store is not None and not plan.reuse_gbuffer:
            path = snapshot_store.path_for(plan.tile)
            result.snapshot_path = str(path)
            try:
                result.snapshot_checksum = snapshot_store.save(
                    scene,
                    plan.tile,
                    gbuffer,
                    shadow_maps,
                    material_dependencies,
                    shading.light_usage,
                )
                result.snapshot_written = True
            except PipelineError as error:
                result.snapshot_error = error.as_dict()
        elif snapshot_store is not None and cached is not None:
            result.snapshot_path = cached.path
            result.snapshot_checksum = cached.checksum
    except Exception as error:
        result.error = _as_error(error)
        result.gbuffer = None
        result.shadow_maps = None
        result.shading = None
        if snapshot_store is not None:
            path = snapshot_store.path_for(plan.tile)
            result.snapshot_path = str(path)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as unlink_error:
                result.snapshot_error = {
                    "code": "E_SNAPSHOT_IO",
                    "message": f"failed to remove stale snapshot: {unlink_error}",
                    "stage": "snapshot",
                }
    return result


def _snapshot_dependencies(snapshot: Snapshot) -> tuple[set[str], set[str], bool]:
    material_values = snapshot.dependencies.get("materials")
    light_values = snapshot.dependencies.get("lights")
    has_coverage = snapshot.dependencies.get("has_coverage")
    if not isinstance(material_values, list) or not all(isinstance(item, str) for item in material_values):
        raise SnapshotCorruptError("snapshot material dependencies are invalid")
    if not isinstance(light_values, list) or not all(isinstance(item, str) for item in light_values):
        raise SnapshotCorruptError("snapshot light dependencies are invalid")
    if not isinstance(has_coverage, bool):
        raise SnapshotCorruptError("snapshot coverage dependency is invalid")
    return set(material_values), set(light_values), has_coverage


def _validate_loaded_snapshot(scene: Scene, tile: TileSpec, snapshot: Snapshot) -> None:
    if snapshot.tile != tile:
        raise SnapshotCorruptError(f"snapshot tile does not match requested tile {tile.tile_id}")
    if snapshot.gbuffer.width != tile.width or snapshot.gbuffer.height != tile.height:
        raise SnapshotCorruptError("snapshot G-buffer size does not match tile")
    material_dependencies, _light_dependencies, _has_coverage = _snapshot_dependencies(snapshot)
    if snapshot.scene.get("base_signature") != scene.base_signature():
        return
    light_ids = {light.light_id for light in scene.lights}
    if set(snapshot.shadow_maps) != light_ids:
        raise SnapshotCorruptError("snapshot shadow cache does not match current light set")
    if set(snapshot.lights) != light_ids:
        raise SnapshotCorruptError("snapshot light state does not match current light set")
    expected_materials = {
        scene.materials[index].material_id
        for index in snapshot.gbuffer.material_usage()
    }
    if material_dependencies != expected_materials:
        raise SnapshotCorruptError("snapshot material dependencies do not match its G-buffer")
    for material_id in snapshot.gbuffer.material_usage():
        if material_id >= len(scene.materials):
            raise SnapshotCorruptError("snapshot material index is out of range")


def _difference_keys(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    keys = set(left) | set(right)
    return {key for key in keys if left.get(key) != right.get(key)}


def _make_plans(
    scene: Scene,
    options: PipelineOptions,
    snapshots: dict[int, Snapshot],
    snapshot_errors: list[dict[str, Any]],
) -> tuple[list[_TilePlan], list[str], dict[str, set[str]]]:
    tiles = scene.tiles
    changed_materials: set[str] = set()
    changed_light_colors: set[str] = set()
    changed_light_shadows: set[str] = set()
    global_reasons: list[str] = []
    if not options.incremental:
        global_reasons.append("incremental_disabled")
    if options.incremental and not snapshots:
        global_reasons.append("no_snapshots")

    base_mismatch = False
    if options.incremental and snapshots:
        for snapshot in snapshots.values():
            old_base = snapshot.scene.get("base_signature")
            old_version = snapshot.scene.get("version")
            old_quality = snapshot.render.get("quality")
            if old_base != scene.base_signature():
                base_mismatch = True
                if old_version != scene.version:
                    global_reasons.append("scene_version_changed")
                elif old_quality != scene.shadow.quality:
                    global_reasons.append("quality_changed")
                else:
                    global_reasons.append("base_signature_changed")
                break
            if old_version != scene.version:
                base_mismatch = True
                global_reasons.append("scene_version_changed")
                break
            if old_quality != scene.shadow.quality:
                base_mismatch = True
                global_reasons.append("quality_changed")
                break
    if base_mismatch:
        global_reasons = list(dict.fromkeys(global_reasons + ["full_rerender"]))

    if options.incremental and snapshots and not base_mismatch:
        current_materials = scene.material_signatures()
        current_colors = scene.light_color_signatures()
        current_shadows = scene.light_shadow_signatures()
        for snapshot in snapshots.values():
            old_materials = snapshot.scene.get("materials")
            old_colors = snapshot.scene.get("light_colors")
            old_shadows = snapshot.scene.get("light_shadows")
            if not isinstance(old_materials, dict) or not isinstance(old_colors, dict) or not isinstance(old_shadows, dict):
                raise SnapshotCorruptError("snapshot mutable signatures are missing")
            changed_materials.update(_difference_keys(old_materials, current_materials))
            changed_light_colors.update(_difference_keys(old_colors, current_colors))
            changed_light_shadows.update(_difference_keys(old_shadows, current_shadows))

    plans: list[_TilePlan] = []
    for tile in tiles:
        snapshot = snapshots.get(tile.index)
        if not options.incremental:
            plans.append(_TilePlan(tile, False, False, None, ("incremental_disabled",)))
            continue
        if snapshot is None:
            plans.append(_TilePlan(tile, False, False, None, ("no_snapshot",)))
            continue
        if base_mismatch:
            plans.append(_TilePlan(tile, False, False, snapshot, tuple(global_reasons)))
            continue
        old_material_dependencies, old_light_dependencies, has_coverage = _snapshot_dependencies(snapshot)
        affected_materials = sorted(old_material_dependencies & changed_materials)
        affected_colors = sorted(old_light_dependencies & changed_light_colors)
        reasons: list[str] = []
        if affected_materials:
            reasons.append("material_changed:" + ",".join(affected_materials))
        if affected_colors:
            reasons.append("light_color_changed:" + ",".join(affected_colors))
        shadow_affected = bool(changed_light_shadows) and (has_coverage or bool(old_light_dependencies))
        if shadow_affected:
            reasons.append("light_shadow_changed:" + ",".join(sorted(changed_light_shadows)))
        if reasons:
            reuse_shadow = not shadow_affected
            plans.append(_TilePlan(tile, False, reuse_shadow, snapshot, tuple(reasons)))
        else:
            plans.append(_TilePlan(tile, True, True, snapshot, ("snapshot_reused",)))
    change_sets = {
        "materials": changed_materials,
        "light_colors": changed_light_colors,
        "light_shadows": changed_light_shadows,
    }
    return plans, list(dict.fromkeys(global_reasons)), change_sets


def _tile_report(scene: Scene, result: _TileResult) -> dict[str, Any]:
    plan = result.plan
    entry: dict[str, Any] = {
        "index": plan.tile.index,
        "id": plan.tile.tile_id,
        "x": plan.tile.x,
        "y": plan.tile.y,
        "width": plan.tile.width,
        "height": plan.tile.height,
        "status": "ok" if result.succeeded else "failed",
        "success": result.succeeded,
        "reused": plan.reuse_gbuffer,
        "gbuffer_reused": plan.reuse_gbuffer,
        "shadow_cache_reused": plan.reuse_shadow_maps,
        "reasons": list(plan.reasons),
        "materials": list(result.material_dependencies),
        "lights": list(result.light_dependencies),
        "snapshot_path": result.snapshot_path,
        "snapshot_written": result.snapshot_written,
        "snapshot_checksum": result.snapshot_checksum,
    }
    if result.error is not None:
        entry["error"] = result.error
        entry["code"] = result.error["code"]
        entry["message"] = result.error["message"]
        entry["stage"] = result.error["stage"]
    if result.snapshot_error is not None:
        entry["snapshot_error"] = result.snapshot_error
    if result.gbuffer is not None:
        entry["coverage_pixels"] = sum(1 for depth in result.gbuffer.depth if depth != float("inf"))
    if result.shadow_maps is not None:
        entry["shadow_maps"] = {
            light_id: {
                "kind": "point" if hasattr(shadow_map, "faces") else "directional",
                "resolution": shadow_map.resolution,
            }
            for light_id, shadow_map in sorted(result.shadow_maps.items())
        }
    return entry


def _ppm_bytes(scene: Scene, frame: list[float]) -> bytes:
    header = f"P6\n{scene.width} {scene.height}\n255\n".encode("ascii")
    pixels = bytearray(scene.width * scene.height * 3)
    output_index = 0
    for value in frame:
        quantized = int(round(min(1.0, max(0.0, value)) * 255.0))
        pixels[output_index] = quantized
        output_index += 1
    return header + bytes(pixels)


def write_ppm(path: str | Path, ppm_bytes: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(ppm_bytes)
    os.replace(temporary, destination)


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


class RenderPipeline:
    def __init__(
        self,
        scene: Scene | str | Path | dict[str, Any],
        options: PipelineOptions | dict[str, Any] | None = None,
    ) -> None:
        loaded = scene if isinstance(scene, Scene) else load_scene(scene)
        if isinstance(options, dict):
            options = PipelineOptions(**options)
        self.options = options or PipelineOptions()
        self.scene = apply_options(loaded, self.options)
        self.snapshot_store = (
            SnapshotStore(self.options.snapshot_dir) if self.options.snapshot_dir is not None else None
        )

    def _load_snapshots(self) -> tuple[dict[int, Snapshot], list[dict[str, Any]]]:
        snapshots: dict[int, Snapshot] = {}
        errors: list[dict[str, Any]] = []
        if not self.options.incremental or self.snapshot_store is None:
            return snapshots, errors
        for tile in self.scene.tiles:
            path = self.snapshot_store.path_for(tile)
            if not path.exists():
                continue
            try:
                snapshot = load_snapshot(path)
                if snapshot.tile != tile:
                    continue
                _validate_loaded_snapshot(self.scene, tile, snapshot)
                snapshots[tile.index] = snapshot
            except PipelineError as error:
                errors.append(
                    {
                        "tile_index": tile.index,
                        "tile_id": tile.tile_id,
                        "path": str(path),
                        **error.as_dict(),
                    }
                )
        return snapshots, errors

    def run(self) -> RenderResult:
        snapshots, snapshot_errors = self._load_snapshots()
        plans, global_reasons, change_sets = _make_plans(
            self.scene, self.options, snapshots, snapshot_errors
        )
        worker_count = min(self.options.workers, max(1, len(plans)))
        results_by_index: dict[int, _TileResult] = {}
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="render-tile") as executor:
            futures: dict[int, Future[_TileResult]] = {
                plan.tile.index: executor.submit(_run_tile_plan, self.scene, plan, self.snapshot_store)
                for plan in plans
            }
            for tile in self.scene.tiles:
                future = futures[tile.index]
                try:
                    results_by_index[tile.index] = future.result()
                except Exception as error:
                    results_by_index[tile.index] = _TileResult(
                        plan=next(plan for plan in plans if plan.tile.index == tile.index),
                        error=_as_error(error),
                    )

        ordered_results = [results_by_index[tile.index] for tile in self.scene.tiles]
        if self.snapshot_store is not None:
            expected_names = {
                self.snapshot_store.path_for(tile).name for tile in self.scene.tiles
            }
            for path in sorted(self.snapshot_store.directory.glob("*.snapshot.json")):
                if path.name in expected_names:
                    continue
                try:
                    path.unlink()
                except OSError as error:
                    snapshot_errors.append(
                        {
                            "tile_index": None,
                            "tile_id": None,
                            "path": str(path),
                            "code": "E_SNAPSHOT_IO",
                            "message": f"failed to remove orphan snapshot: {error}",
                            "stage": "snapshot",
                        }
                    )
        gbuffer_by_coordinate: dict[tuple[int, int], GBufferTile] = {}
        shading_by_coordinate: dict[tuple[int, int], TileShading] = {}
        successful_tiles: list[GBufferTile] = []
        for result in ordered_results:
            if result.succeeded and result.gbuffer is not None and result.shading is not None:
                gbuffer_by_coordinate[(result.gbuffer.x, result.gbuffer.y)] = result.gbuffer
                shading_by_coordinate[(result.gbuffer.x, result.gbuffer.y)] = result.shading
                successful_tiles.append(result.gbuffer)
        frame = _compose_at_coordinates(self.scene, successful_tiles, shading_by_coordinate)
        ppm_bytes = _ppm_bytes(self.scene, frame)
        tile_reports = [_tile_report(self.scene, result) for result in ordered_results]
        failed_tiles = [entry for entry in tile_reports if entry["status"] == "failed"]
        succeeded_count = len(tile_reports) - len(failed_tiles)
        reused_count = sum(1 for result in ordered_results if result.succeeded and result.plan.reuse_gbuffer)
        rendered_count = succeeded_count - reused_count
        business_codes = sorted(
            {
                entry["error"]["code"]
                for entry in failed_tiles
                if isinstance(entry.get("error"), dict) and "code" in entry["error"]
            }
            | {
                error["code"]
                for error in snapshot_errors
                if isinstance(error.get("code"), str)
            }
        )
        report: dict[str, Any] = {
            "schema_version": 1,
            "scene": {
                "id": self.scene.scene_id,
                "version": self.scene.version,
                "summary": self.scene.summary(),
                "base_signature": self.scene.base_signature(),
            },
            "frame": {
                "width": self.scene.width,
                "height": self.scene.height,
                "tile_width": self.scene.tile_width,
                "tile_height": self.scene.tile_height,
            },
            "quality": self.scene.shadow.quality,
            "pcf_kernel": 1 if self.scene.shadow.quality == "fast" else self.scene.shadow.pcf_kernel,
            "configured_pcf_kernel": self.scene.shadow.pcf_kernel,
            "shadow": {
                "quality": self.scene.shadow.quality,
                "resolution_scale": 0.5 if self.scene.shadow.quality == "fast" else 1.0,
                "effective": {
                    light.light_id: {
                        "resolution": self.scene.shadow.effective_resolution(light),
                        "pcf_kernel": self.scene.shadow.effective_pcf_kernel(light),
                    }
                    for light in self.scene.lights
                },
            },
            "status": "ok" if not failed_tiles else "partial",
            "complete": not failed_tiles,
            "business_codes": business_codes,
            "summary": {
                "total_tiles": len(tile_reports),
                "succeeded_tiles": succeeded_count,
                "failed_tiles": len(failed_tiles),
                "reused_tiles": reused_count,
                "rendered_tiles": rendered_count,
            },
            "incremental": {
                "enabled": self.options.incremental,
                "global_reasons": global_reasons,
                "changed_materials": sorted(change_sets["materials"]),
                "changed_light_colors": sorted(change_sets["light_colors"]),
                "changed_light_shadows": sorted(change_sets["light_shadows"]),
            },
            "snapshot_errors": snapshot_errors,
            "failed_tiles": failed_tiles,
            "failed_tile_ids": [entry["id"] for entry in failed_tiles],
            "tiles": tile_reports,
            "output_sha256": __import__("hashlib").sha256(ppm_bytes).hexdigest(),
        }
        result = RenderResult(
            scene=self.scene,
            frame=frame,
            gbuffer_tiles=successful_tiles,
            report=report,
            ppm_bytes=ppm_bytes,
        )
        if self.options.report_path is not None:
            write_report(self.options.report_path, report)
        return result

    def render_to(self, output_path: str | Path, report_path: str | Path | None = None) -> RenderResult:
        result = self.run()
        write_ppm(output_path, result.ppm_bytes)
        if report_path is not None:
            write_report(report_path, result.report)
        return result


def _compose_at_coordinates(
    scene: Scene,
    tiles: list[GBufferTile],
    shading_by_coordinate: dict[tuple[int, int], TileShading],
) -> list[float]:
    frame = [component for _pixel in range(scene.width * scene.height) for component in scene.background]
    for tile in tiles:
        tile_shading = shading_by_coordinate.get((tile.x, tile.y))
        if tile_shading is None:
            continue
        for local_y in range(tile.height):
            frame_y = tile.y + local_y
            for local_x in range(tile.width):
                frame_x = tile.x + local_x
                frame_index = (frame_y * scene.width + frame_x) * 3
                tile_index = tile.pixel_index(local_x, local_y) * 3
                frame[frame_index] = tile_shading.color[tile_index]
                frame[frame_index + 1] = tile_shading.color[tile_index + 1]
                frame[frame_index + 2] = tile_shading.color[tile_index + 2]
    return frame


def render_reference(scene: Scene) -> ReferenceResult:
    tile = TileSpec(index=0, x=0, y=0, width=scene.width, height=scene.height)
    gbuffer = rasterize_tile(scene, tile)
    material_shading(scene, gbuffer)
    shadow_maps = build_shadow_maps(scene)
    shading = shade_tile(scene, gbuffer, shadow_maps)
    frame = _compose_at_coordinates(scene, [gbuffer], {(0, 0): shading})
    return ReferenceResult(scene=scene, gbuffer=gbuffer, shadow_maps=shadow_maps, shading=shading, frame=frame)


def render_scene(
    scene: Scene | str | Path | dict[str, Any],
    options: PipelineOptions | dict[str, Any] | None = None,
) -> RenderResult:
    return RenderPipeline(scene, options).run()
