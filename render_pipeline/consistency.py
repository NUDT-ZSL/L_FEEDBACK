from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ErrorCode, diagnostic
from .material import MaterialMaps
from .renderer import ViewRender
from .scene import LightSpec, SceneSpec
from .shading import integrate_lighting
from .util import rounded


UV_LANDMARKS: tuple[tuple[float, float], ...] = (
    (0.13, 0.25),
    (0.13, 0.50),
    (0.13, 0.75),
)


def _nearest_covered(view: ViewRender, uv: tuple[float, float]) -> tuple[int, int, np.ndarray, np.ndarray] | None:
    coverage = view.gbuffer.coverage.reshape(-1)
    if not np.any(coverage):
        return None
    uv_flat = view.gbuffer.uv.reshape(-1, 2)
    distance = np.sum((uv_flat - np.asarray(uv, dtype=np.float64)) ** 2.0, axis=1)
    distance[~coverage] = np.inf
    index = int(np.argmin(distance))
    if not np.isfinite(distance[index]):
        return None
    y, x = divmod(index, view.camera.width)
    return x, y, view.gbuffer.uv[y, x].copy(), view.gbuffer.normal[y, x].copy()


def _cross_view_uv_check(
    scene: SceneSpec,
    views: dict[str, ViewRender],
    uv_threshold: float = 0.02,
    normal_threshold: float = 0.08,
) -> dict[str, Any]:
    camera_ids = [camera.id for camera in scene.cameras[:2]]
    if len(camera_ids) < 2:
        return {
            "id": "uv_cross_view_consistency",
            "status": "fail",
            "reason": "at least two cameras are required",
            "thresholds": {"uv_l2": uv_threshold, "normal_l2": normal_threshold},
            "samples": [],
            "max_uv_error": None,
            "max_normal_error": None,
        }
    first_id, second_id = camera_ids
    first_view = views[first_id]
    second_view = views[second_id]
    samples: list[dict[str, Any]] = []
    max_uv_error = 0.0
    max_normal_error = 0.0
    missing = 0
    for target in UV_LANDMARKS:
        first = _nearest_covered(first_view, target)
        second = _nearest_covered(second_view, target)
        if first is None or second is None:
            missing += 1
            continue
        x0, y0, uv0, normal0 = first
        x1, y1, uv1, normal1 = second
        uv_error = float(np.linalg.norm(uv0 - uv1))
        normal_error = float(np.linalg.norm(normal0 - normal1))
        material_error = float(
            np.max(
                np.abs(
                    first_view.shading.albedo[y0, x0]
                    - second_view.shading.albedo[y1, x1]
                )
            )
        )
        max_uv_error = max(max_uv_error, uv_error)
        max_normal_error = max(max_normal_error, normal_error)
        samples.append(
            {
                "target_uv": list(target),
                "front_pixel": [x0, y0],
                "side_pixel": [x1, y1],
                "front_uv": rounded(uv0),
                "side_uv": rounded(uv1),
                "uv_error": rounded(uv_error),
                "normal_error": rounded(normal_error),
                "albedo_error": rounded(material_error),
                "status": "pass" if uv_error <= uv_threshold and normal_error <= normal_threshold else "fail",
            }
        )
    passed = missing == 0 and max_uv_error <= uv_threshold and max_normal_error <= normal_threshold
    return {
        "id": "uv_cross_view_consistency",
        "status": "pass" if passed else "fail",
        "thresholds": {"uv_l2": uv_threshold, "normal_l2": normal_threshold},
        "missing_landmarks": missing,
        "max_uv_error": rounded(max_uv_error),
        "max_normal_error": rounded(max_normal_error),
        "samples": samples,
    }


def _boundary_pairs(mask: np.ndarray, threshold: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    indices: list[tuple[int, int, int, int]] = []
    for y in range(mask.shape[0]):
        row = mask[y]
        crossing = ((row[:-1] - threshold) * (row[1:] - threshold) <= 0.0) & (np.abs(row[:-1] - row[1:]) > 1e-9)
        for x in np.nonzero(crossing)[0]:
            indices.append((y, int(x), y, int(x) + 1))
    for x in range(mask.shape[1]):
        column = mask[:, x]
        crossing = ((column[:-1] - threshold) * (column[1:] - threshold) <= 0.0) & (
            np.abs(column[:-1] - column[1:]) > 1e-9
        )
        for y in np.nonzero(crossing)[0]:
            indices.append((int(y), x, int(y) + 1, x))
    if not indices:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    array = np.asarray(indices, dtype=np.int64)
    return array[:, :2], array[:, 2:]


def _metallic_boundary_check(
    material: MaterialMaps,
    preset: LightSpec,
    threshold: float = 0.06,
) -> dict[str, Any]:
    first, second = _boundary_pairs(material.metallic)
    if first.size == 0:
        return {
            "id": "metallic_boundary_shading",
            "status": "fail",
            "reason": "metallic mask has no crossing at 0.35",
            "threshold": threshold,
            "sample_count": 0,
            "max_linear_color_delta": None,
        }

    albedo_a = material.albedo_linear[first[:, 0], first[:, 1]]
    albedo_b = material.albedo_linear[second[:, 0], second[:, 1]]
    albedo_delta = np.max(np.abs(albedo_a - albedo_b), axis=1)
    roughness_a = material.roughness[first[:, 0], first[:, 1]]
    roughness_b = material.roughness[second[:, 0], second[:, 1]]
    roughness_delta = np.abs(roughness_a - roughness_b)
    isolated = (albedo_delta <= 0.05) & (roughness_delta <= 0.05)
    if np.any(isolated):
        chosen_a = albedo_a[isolated]
        chosen_b = albedo_b[isolated]
        chosen_ra = roughness_a[isolated]
        chosen_rb = roughness_b[isolated]
        chosen_ma = material.metallic[first[isolated, 0], first[isolated, 1]]
        chosen_mb = material.metallic[second[isolated, 0], second[isolated, 1]]
        chosen_first = first[isolated]
        chosen_second = second[isolated]
    else:
        chosen_a, chosen_b = albedo_a, albedo_b
        chosen_ra, chosen_rb = roughness_a, roughness_b
        chosen_ma = material.metallic[first[:, 0], first[:, 1]]
        chosen_mb = material.metallic[second[:, 0], second[:, 1]]
        chosen_first, chosen_second = first, second

    count = len(chosen_a)
    normals = np.broadcast_to(np.array((0.0, 0.0, 1.0)), (count, 3))
    view = np.array((0.35, 0.20, 0.915))
    view /= np.linalg.norm(view)
    _, _, color_a, _, _ = integrate_lighting(normals, tuple(view), chosen_a, chosen_ra, chosen_ma, preset)
    _, _, color_b, _, _ = integrate_lighting(normals, tuple(view), chosen_b, chosen_rb, chosen_mb, preset)
    delta = np.max(np.abs(color_a - color_b), axis=1)
    max_index = int(np.argmax(delta))
    max_delta = float(delta[max_index])
    passed = max_delta <= threshold
    return {
        "id": "metallic_boundary_shading",
        "status": "pass" if passed else "fail",
        "threshold": threshold,
        "sample_count": count,
        "max_linear_color_delta": rounded(max_delta),
        "worst_pair": {
            "first_xy": chosen_first[max_index].astype(int).tolist(),
            "second_xy": chosen_second[max_index].astype(int).tolist(),
            "first_metallic": rounded(float(chosen_ma[max_index])),
            "second_metallic": rounded(float(chosen_mb[max_index])),
            "first_color": rounded(color_a[max_index]),
            "second_color": rounded(color_b[max_index]),
        },
    }


def _energy_conservation_check(
    material: MaterialMaps,
    preset: LightSpec,
    limit: float = 1.0,
) -> dict[str, Any]:
    y_step = max(1, material.height // 16)
    x_step = max(1, material.width // 16)
    albedo_base = material.albedo_linear[::y_step, ::x_step].reshape(-1, 3)
    roughness_base = material.roughness[::y_step, ::x_step].reshape(-1)
    metallic_levels = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    albedo = np.repeat(albedo_base, len(metallic_levels), axis=0)
    roughness = np.repeat(roughness_base, len(metallic_levels), axis=0)
    metallic = np.tile(metallic_levels, len(albedo_base))
    normals = np.broadcast_to(np.array((0.0, 0.0, 1.0)), albedo.shape).copy()
    reflectance, _, _, weights, _ = integrate_lighting(
        normals,
        (0.25, 0.2, 0.947),
        albedo,
        roughness,
        metallic,
        preset,
    )
    max_reflectance = float(np.max(reflectance))
    max_weight = float(np.max(weights))
    peak = max(max_reflectance, max_weight)
    return {
        "id": "energy_conservation",
        "status": "pass" if peak <= limit + 1e-9 else "fail",
        "limit": limit,
        "max_reflectance": rounded(max_reflectance),
        "max_lobe_weight_sum": rounded(max_weight),
        "sample_count": int(len(albedo)),
        "metallic_range": [0.0, 1.0],
    }


def run_consistency(
    scene: SceneSpec,
    material: MaterialMaps,
    preset: LightSpec,
    views: dict[str, ViewRender],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checks = [
        _cross_view_uv_check(scene, views),
        _metallic_boundary_check(material, preset),
        _energy_conservation_check(material, preset),
    ]
    errors: list[dict[str, Any]] = []
    for check in checks:
        if check["status"] == "pass":
            continue
        if check["id"] == "uv_cross_view_consistency":
            errors.append(
                diagnostic(
                    ErrorCode.CONSISTENCY_UV_MISMATCH,
                    "UV/normal landmark sampling differs between the two camera views",
                    "consistency",
                    preset_id=preset.id,
                    check=check["id"],
                    max_uv_error=check.get("max_uv_error"),
                    max_normal_error=check.get("max_normal_error"),
                )
            )
        elif check["id"] == "metallic_boundary_shading":
            errors.append(
                diagnostic(
                    ErrorCode.CONSISTENCY_METALLIC_JUMP,
                    "metallic mask boundary produces a shading jump above threshold",
                    "consistency",
                    preset_id=preset.id,
                    check=check["id"],
                    measured=check.get("max_linear_color_delta"),
                    threshold=check.get("threshold"),
                )
            )
        else:
            errors.append(
                diagnostic(
                    ErrorCode.CONSISTENCY_ENERGY_VIOLATION,
                    "shading reflectance exceeded the conservation boundary",
                    "consistency",
                    preset_id=preset.id,
                    check=check["id"],
                    max_reflectance=check.get("max_reflectance"),
                )
            )
    overall = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return (
        {
            "schema_version": 1,
            "scene_id": scene.scene_id,
            "preset_id": preset.id,
            "status": overall,
            "checks": checks,
            "diagnostic_codes": [item["code"] for item in errors],
        },
        errors,
    )
