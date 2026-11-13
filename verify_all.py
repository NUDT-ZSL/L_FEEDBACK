"""Self-contained regression entry point for the offline terrain pipeline.

Five contract checks run in a fixed, deterministic order:

1. ``heightfield_determinism``  - splitmix64 bits and sampling are reproducible;
2. ``frustum_conservative``     - no visible AABB is ever culled (no false neg);
3. ``stitch_consistency``       - shared edges carry identical vertex sequences;
4. ``backend_consistency``      - ModernGL and NumPy outputs match within tolerance;
5. ``business_codes``           - E_RES_TOO_HIGH / E_BAD_FRUSTUM / E_NO_GL rules.

The stitch switch is read from configuration only.  Running with a config that
disables stitching makes check 3 fail with a non-zero exit code and the failing
check name printed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from terrain.camera import Frustum
from terrain.config import PipelineConfig, write_json
from terrain.errors import E_BAD_FRUSTUM, E_NO_GL, E_RES_TOO_HIGH, PipelineError
from terrain.heightfield import Heightfield, splitmix64
from terrain.lod import (
    HeightBoundsCache,
    enforce_restricted_quadtree,
    recount_leaves,
    select_leaves,
)
from terrain.mesh import (
    _opposite,
    build_tile_mesh_safe,
    tile_axis_coordinates,
)
from terrain.pipeline import run_pipeline
from terrain.quadtree import TileKey, adjacent_leaves

DEFAULT_CONFIG = "configs/default.json"
BACKEND_PIXEL_TOLERANCE = 6
BACKEND_MATCH_RATIO = 0.95


def _build_scene(config: PipelineConfig):
    heightfield = Heightfield(
        config.seed, config.height_scale, config.noise_octaves
    )
    cache = HeightBoundsCache(config, heightfield)
    leaves, counters = select_leaves(config, cache)
    leaves = enforce_restricted_quadtree(leaves, config.max_depth)
    meshes = [
        build_tile_mesh_safe(key, leaves, heightfield, config, cache.grid(key))
        for key in sorted(leaves, key=lambda tile: (tile.level, tile.iz, tile.ix))
    ]
    return heightfield, cache, leaves, counters, meshes


def check_heightfield_determinism(config: PipelineConfig) -> None:
    first = Heightfield(config.seed, config.height_scale, config.noise_octaves)
    second = Heightfield(config.seed, config.height_scale, config.noise_octaves)
    expected = np.uint64(0xE220A8397B1DCDAF)
    if splitmix64(np.uint64(0)) != expected:
        raise AssertionError("splitmix64(0) disagrees with known test vector")
    points = [(3.5, -7.25), (0.0, 0.0), (-11.75, 4.5), (15.999, 15.999)]
    samples_a = [first.sample(x, z) for x, z in points]
    samples_b = [second.sample(x, z) for x, z in points]
    if samples_a != samples_b:
        raise AssertionError("heightfield samples differ across instances")
    grid_a = first.tile_grid(-8.0, -8.0, 8.0, 8.0, 16)
    grid_b = second.tile_grid(-8.0, -8.0, 8.0, 8.0, 16)
    if not np.array_equal(grid_a, grid_b):
        raise AssertionError("tile grids differ across instances")
    if np.any(~np.isfinite(grid_a)):
        raise AssertionError("non-finite heightfield values")
    # Arbitrary-point interface must agree with the vectorised sampler.
    probe_x = np.linspace(-16.0, 16.0, 37)
    probe_z = np.linspace(-16.0, 16.0, 37)
    vectorised = first.sample_array(probe_x, probe_z)
    scalar = np.array(
        [first.sample(float(x), float(z)) for x, z in zip(probe_x, probe_z)]
    )
    if not np.allclose(vectorised, scalar, rtol=0.0, atol=1e-12):
        raise AssertionError("scalar and vectorised samplers disagree")


def check_frustum_conservative(config: PipelineConfig) -> None:
    heightfield = Heightfield(
        config.seed, config.height_scale, config.noise_octaves
    )
    cache = HeightBoundsCache(config, heightfield)
    frustum = Frustum(config.camera)
    false_negatives = 0
    tested = 0
    for level in range(config.max_depth + 1):
        for ix in range(1 << level):
            for iz in range(1 << level):
                key = TileKey(level, ix, iz)
                aabb = cache.aabb(key)
                tested += 1
                # Ground truth: any of the 8 corners inside means visible.
                x0, y0, z0, x1, y1, z1 = aabb
                corners = np.array(
                    [
                        [cx, cy, cz]
                        for cx in (x0, x1)
                        for cy in (y0, y1)
                        for cz in (z0, z1)
                    ]
                )
                distances = corners @ frustum.planes[:, :3].T + frustum.planes[:, 3]
                corner_visible = bool(np.any(np.all(distances >= 0.0, axis=1)))
                if corner_visible and not frustum.intersects_aabb(aabb):
                    false_negatives += 1
    if false_negatives:
        raise AssertionError(
            f"{false_negatives}/{tested} visible AABBs were culled"
        )
    leaves, counters = select_leaves(config, cache)
    restricted = enforce_restricted_quadtree(leaves, config.max_depth)
    if sum(recount_leaves(restricted, config.max_depth)) != len(restricted):
        raise AssertionError("visible counts do not match leaf selection")
    if not all(
        tested == visible + culled
        for tested, visible, culled in zip(
            counters.tested_per_level,
            counters.visible_per_level,
            counters.culled_per_level,
        )
    ):
        raise AssertionError("per-level tested != visible + culled")


def check_stitch_consistency(config: PipelineConfig) -> None:
    if not config.enable_stitching:
        raise AssertionError(
            "stitching disabled in config; shared-edge invariants cannot hold"
        )
    heightfield, _cache, leaves, _counters, meshes = _build_scene(config)
    mesh_by_key = {mesh.key: mesh for mesh in meshes}
    patched = sum(
        len(mesh.stitched_edges)
        for mesh in meshes
        if mesh.stitched_edges != ("ROLLBACK",)
    )
    if patched == 0:
        raise AssertionError("no stitched edges were produced to verify")
    checked = 0
    for key, mesh in mesh_by_key.items():
        for direction in ("N", "S", "W", "E"):
            neighbours = adjacent_leaves(leaves, key, direction)
            if not neighbours:
                continue
            axis = 0 if direction in ("N", "S") else 1
            own_x, own_z = tile_axis_coordinates(key, config)
            if direction in ("N", "S"):
                fixed = own_z[0] if direction == "N" else own_z[-1]
                own_edge = mesh.vertices[np.isclose(mesh.vertices[:, 2], fixed)]
            else:
                fixed = own_x[0] if direction == "W" else own_x[-1]
                own_edge = mesh.vertices[np.isclose(mesh.vertices[:, 0], fixed)]
            own_points = set(map(tuple, np.round(own_edge[:, [0, 2]], 6)))
            for neighbour in neighbours:
                nx0, nz0, nx1, nz1 = neighbour.bounds(
                    config.root_min, config.root_size
                )
                ox0, oz0, ox1, oz1 = key.bounds(
                    config.root_min, config.root_size
                )
                if axis == 0:
                    low, high = max(ox0, nx0), min(ox1, nx1)
                else:
                    low, high = max(oz0, nz0), min(oz1, nz1)
                neighbour_mesh = mesh_by_key[neighbour]
                nx, nz = tile_axis_coordinates(neighbour, config)
                if direction in ("N", "S"):
                    fixed = nz[0] if _opposite(direction) == "N" else nz[-1]
                    fine_edge = neighbour_mesh.vertices[
                        np.isclose(neighbour_mesh.vertices[:, 2], fixed)
                    ]
                else:
                    fixed = nx[0] if _opposite(direction) == "W" else nx[-1]
                    fine_edge = neighbour_mesh.vertices[
                        np.isclose(neighbour_mesh.vertices[:, 0], fixed)
                    ]
                fine_points = set(
                    map(tuple, np.round(fine_edge[:, [0, 2]], 6))
                )
                within = {
                    point
                    for point in fine_points
                    if low - 1e-6 <= point[axis] <= high + 1e-6
                }
                own_within = {
                    point
                    for point in own_points
                    if low - 1e-6 <= point[axis] <= high + 1e-6
                }
                if within != own_within:
                    raise AssertionError(
                        f"edge {key}/{direction} vs {neighbour} not identical"
                    )
                checked += 1
    if checked == 0:
        raise AssertionError("no shared edges were checked")


def check_backend_consistency(config: PipelineConfig) -> tuple[str, str | None]:
    software_config = config.with_overrides(backend="software")
    software_result = run_pipeline(software_config)
    software_image = software_result.image

    gl_config = config.with_overrides(backend="moderngl")
    try:
        gl_result = run_pipeline(gl_config)
    except PipelineError as exc:
        if exc.code != E_NO_GL:
            raise
        return software_result.backend_used, E_NO_GL
    gl_image = gl_result.image
    if gl_image.shape != software_image.shape:
        raise AssertionError("backend output shapes differ")
    difference = np.abs(
        software_image.astype(np.int16) - gl_image.astype(np.int16)
    )
    match_ratio = float(np.mean(difference <= BACKEND_PIXEL_TOLERANCE))
    if match_ratio < BACKEND_MATCH_RATIO:
        raise AssertionError(
            f"backends match only {match_ratio:.3f} of pixels"
        )
    if gl_result.contract["total_triangle_count"] != (
        software_result.contract["total_triangle_count"]
    ):
        raise AssertionError("backend triangle counts differ")
    return gl_result.backend_used, None


def check_business_codes(config: PipelineConfig) -> None:
    try:
        config.with_overrides(tile_resolution=2049).validate()
    except PipelineError as exc:
        if exc.code != E_RES_TOO_HIGH:
            raise AssertionError(f"expected E_RES_TOO_HIGH, got {exc.code}")
    else:
        raise AssertionError("resolution 2049 was not rejected")

    try:
        from dataclasses import asdict

        config.with_overrides(
            camera={**asdict(config.camera), "near": -1.0}
        ).validate()
    except PipelineError as exc:
        if exc.code != E_BAD_FRUSTUM:
            raise AssertionError(f"expected E_BAD_FRUSTUM, got {exc.code}")
    else:
        raise AssertionError("near <= 0 was not rejected")

    boundary = config.with_overrides(tile_resolution=2048)
    boundary.validate()

    # CLI contract: business rejection exits 0 and writes the code to stderr.
    bad_resolution = Path("artifacts/_verify_res.json")
    bad_frustum = Path("artifacts/_verify_frustum.json")
    payload = config.to_dict()
    write_json(bad_resolution, {**payload, "tile_resolution": 4096})
    write_json(
        bad_frustum,
        {**payload, "camera": {**payload["camera"], "near": 0.0}},
    )
    for path, code in (
        (bad_resolution, E_RES_TOO_HIGH),
        (bad_frustum, E_BAD_FRUSTUM),
    ):
        proc = subprocess.run(
            [sys.executable, "-m", "terrain.main", "--config", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise AssertionError(f"{code}: CLI exit code {proc.returncode}")
        if code not in proc.stderr:
            raise AssertionError(f"{code}: missing code on stderr")


def main() -> int:
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    config = PipelineConfig.load(config_path)
    checks = [
        ("heightfield_determinism", lambda: check_heightfield_determinism(config)),
        ("frustum_conservative", lambda: check_frustum_conservative(config)),
        ("stitch_consistency", lambda: check_stitch_consistency(config)),
        (
            "backend_consistency",
            lambda: check_backend_consistency(config),
        ),
        ("business_codes", lambda: check_business_codes(config)),
    ]
    failures = []
    for name, check in checks:
        try:
            detail = check()
        except Exception as exc:  # noqa: BLE001 - reported as a named failure
            failures.append(name)
            print(f"[FAIL] {name}: {exc}")
        else:
            suffix = f" ({detail})" if detail else ""
            print(f"[PASS] {name}{suffix}")
    if failures:
        print("FAILED_CHECKS=" + ",".join(failures))
        return 1
    print("ALL_CHECKS_PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
