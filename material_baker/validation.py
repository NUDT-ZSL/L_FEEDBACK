"""Normal-space consistency, seam checks and localization reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .assets import LoadedAssets
from .scene import Scene
from .tangent import TangentField
from .util import REPORT_SCHEMA

SEVERITY_NAMES = ("none", "warning", "error", "critical")
SEVERITY_LEVEL = {"none": 0, "warning": 1, "error": 2, "critical": 3}


@dataclass
class ValidationResult:
    report: dict[str, Any]
    mask: np.ndarray


def _round_vector(vector: np.ndarray, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in np.asarray(vector).ravel()]


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))[..., 0]


def _stratified_indices(count: int, limit: int, seed: int) -> np.ndarray:
    if count <= 0 or limit <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    strata = int(limit)
    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    boundaries = (np.arange(strata + 1, dtype=np.int64) * count) // strata
    lengths = boundaries[1:] - boundaries[:-1]
    offsets = rng.integers(0, lengths, dtype=np.int64)
    return np.sort(boundaries[:-1] + offsets)


def _random_indices(count: int, limit: int, seed: int) -> np.ndarray:
    if count <= 0 or limit <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    return np.sort(rng.choice(count, size=limit, replace=False))


def _sample_indices(
    count: int, limit: int, seed: int, stratified: bool
) -> np.ndarray:
    if stratified:
        return _stratified_indices(count, limit, seed)
    return _random_indices(count, limit, seed)


def _severity(angle: np.ndarray, threshold: float) -> np.ndarray:
    severity = np.zeros(angle.shape, dtype=np.uint8)
    if threshold <= 0.0:
        anomaly = angle > 0.0
        severity[anomaly] = 3
        return severity
    severity[(angle > threshold) & (angle <= threshold * 1.5)] = 1
    severity[(angle > threshold * 1.5) & (angle <= threshold * 2.5)] = 2
    severity[angle > threshold * 2.5] = 3
    return severity


def _reference_normal(
    base_color: np.ndarray,
    tangent: TangentField,
    gain: float,
    minimum_gradient: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    luminance = (
        base_color[:, :, 0] * np.float32(0.2126)
        + base_color[:, :, 1] * np.float32(0.7152)
        + base_color[:, :, 2] * np.float32(0.0722)
    )
    blurred = cv2.GaussianBlur(
        luminance.astype(np.float32),
        (0, 0),
        1.05,
        borderType=cv2.BORDER_REFLECT_101,
    )
    gradient_x = np.zeros_like(blurred)
    gradient_y = np.zeros_like(blurred)
    if blurred.shape[1] > 1:
        gradient_x[:, 1:-1] = (blurred[:, 2:] - blurred[:, :-2]) * 0.5
        gradient_x[:, 0] = blurred[:, 1] - blurred[:, 0]
        gradient_x[:, -1] = blurred[:, -1] - blurred[:, -2]
    if blurred.shape[0] > 1:
        gradient_y[1:-1, :] = (blurred[2:, :] - blurred[:-2, :]) * 0.5
        gradient_y[0, :] = blurred[1, :] - blurred[0, :]
        gradient_y[-1, :] = blurred[-1, :] - blurred[-2, :]

    reference_ts = np.stack(
        [
            -gain * gradient_x,
            -gain * gradient_y,
            np.ones_like(gradient_x),
        ],
        axis=2,
    )
    reference_ts /= np.maximum(
        np.linalg.norm(reference_ts, axis=2, keepdims=True),
        np.float32(1e-8),
    )
    reference_object = (
        tangent.tangent * reference_ts[:, :, 0:1]
        + tangent.bitangent * reference_ts[:, :, 1:2]
        + tangent.geometry_normal[None, None, :] * reference_ts[:, :, 2:3]
    )
    reference_object /= np.maximum(
        np.linalg.norm(reference_object, axis=2, keepdims=True),
        np.float32(1e-8),
    )
    usable = np.hypot(gradient_x, gradient_y) >= minimum_gradient
    return (
        reference_object.astype(np.float32),
        usable,
        gradient_x.astype(np.float32),
        gradient_y.astype(np.float32),
    )


def _normal_check(
    scene: Scene,
    assets: LoadedAssets,
    tangent: TangentField,
    object_normals: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    validation = scene.validation
    threshold = float(validation["angle_threshold_deg"])
    reference, usable, gradient_x, gradient_y = _reference_normal(
        assets.base_color,
        tangent,
        float(validation["reference_gain"]),
        float(validation["reference_min_gradient"]),
    )
    angle = _angle_degrees(object_normals, reference)
    angle[~usable] = 0.0
    severity_map = _severity(angle, threshold)
    angle_anomaly = (angle > threshold if threshold > 0.0 else angle > 0.0) & usable

    normal_ts = assets.normal_tangent
    nx = normal_ts[:, :, 0]
    ny = normal_ts[:, :, 1]
    nz = normal_ts[:, :, 2]
    tangent_xy = np.stack([nx, ny], axis=2)
    tangent_tilt = np.linalg.norm(tangent_xy, axis=2)
    normalized_xy = tangent_xy / np.maximum(
        tangent_tilt[:, :, None], np.float32(1e-8)
    )
    reference_xy = np.stack([-gradient_x, -gradient_y], axis=2)
    reference_xy /= np.maximum(
        np.linalg.norm(reference_xy, axis=2, keepdims=True),
        np.float32(1e-8),
    )
    alignment = np.sum(normalized_xy * reference_xy, axis=2)
    direction_usable = (np.hypot(gradient_x, gradient_y) >= float(
        validation["reference_min_gradient"]
    )) & (
        tangent_tilt >= float(validation["reference_tilt_min"])
    )
    direction_mismatch = direction_usable & (alignment < -0.75)
    direction_severity = np.zeros(alignment.shape, dtype=np.uint8)
    direction_severity[direction_usable & (alignment < -0.75)] = 1
    direction_severity[direction_usable & (alignment < -0.90)] = 2
    direction_severity[direction_usable & (alignment < -0.98)] = 3
    severity_map = np.maximum(severity_map, direction_severity)
    anomaly_mask = angle_anomaly | direction_mismatch

    def transform(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        vector = (
            tangent.tangent * x_values[:, :, None]
            + tangent.bitangent * y_values[:, :, None]
            + tangent.geometry_normal[None, None, :] * nz[:, :, None]
        )
        vector /= np.maximum(
            np.linalg.norm(vector, axis=2, keepdims=True), np.float32(1e-8)
        )
        return vector

    object_normal_guess = np.stack([nx, ny, nz], axis=2).astype(np.float32)
    object_normal_guess /= np.maximum(
        np.linalg.norm(object_normal_guess, axis=2, keepdims=True),
        np.float32(1e-8),
    )
    angle_x_flip = _angle_degrees(transform(-nx, ny), reference)
    angle_y_flip = _angle_degrees(transform(nx, -ny), reference)
    angle_object_guess = _angle_degrees(object_normal_guess, reference)
    align_x_flip = np.sum(
        (
            np.stack([-nx, ny], axis=2)
            / np.maximum(
                np.linalg.norm(np.stack([-nx, ny], axis=2), axis=2, keepdims=True),
                np.float32(1e-8),
            )
        )
        * reference_xy,
        axis=2,
    )
    align_y_flip = np.sum(
        (
            np.stack([nx, -ny], axis=2)
            / np.maximum(
                np.linalg.norm(np.stack([nx, -ny], axis=2), axis=2, keepdims=True),
                np.float32(1e-8),
            )
        )
        * reference_xy,
        axis=2,
    )

    anomaly_indices = np.flatnonzero(anomaly_mask)
    total_anomalies = int(anomaly_indices.size)
    seed = int(validation["seed"])
    max_entries = min(
        int(validation["max_entries"]), int(validation["max_samples"])
    )
    sampled_indices = _sample_indices(
        total_anomalies,
        max_entries,
        seed,
        bool(validation["stratified"]),
    )
    samples: list[dict[str, Any]] = []
    tile_size = int(validation["tile_size"])
    classification_margin = 4.0
    for flat_index in anomaly_indices[sampled_indices]:
        y, x = np.unravel_index(int(flat_index), anomaly_mask.shape)
        current_angle = float(angle[y, x])
        candidates = {
            "x_flip": float(angle_x_flip[y, x]),
            "y_flip": float(angle_y_flip[y, x]),
            "object_guess": float(angle_object_guess[y, x]),
        }
        current_alignment = float(alignment[y, x])
        alignment_candidates = {
            "x_flip": float(align_x_flip[y, x]),
            "y_flip": float(align_y_flip[y, x]),
        }
        best_alignment_name = max(
            alignment_candidates, key=alignment_candidates.get
        )
        if (
            alignment_candidates[best_alignment_name]
            > current_alignment + 0.20
            and direction_mismatch[y, x]
        ):
            anomaly_type = "uv_mismatch"
            related = best_alignment_name
        elif min(candidates["x_flip"], candidates["y_flip"]) + classification_margin < current_angle:
            anomaly_type = "uv_mismatch"
            related = (
                "x_flip"
                if candidates["x_flip"] <= candidates["y_flip"]
                else "y_flip"
            )
        elif candidates["object_guess"] + classification_margin < current_angle:
            anomaly_type = "object_space_mismatch"
            related = "direct_object_interpretation"
        else:
            anomaly_type = "tspace_flip"
            related = "tangent_frame"
        severity = SEVERITY_NAMES[int(severity_map[y, x])]
        samples.append(
            {
                "x": int(x),
                "y": int(y),
                "tile": [int(x) // tile_size, int(y) // tile_size],
                "type": anomaly_type,
                "related_hypothesis": related,
                "severity": severity,
                "angle_deg": round(current_angle, 6),
                "tangent_direction_alignment": round(current_alignment, 6),
                "object_normal": _round_vector(object_normals[y, x]),
                "reference_normal": _round_vector(reference[y, x]),
            }
        )
    anomaly_type_counts: dict[str, int] = {}
    for sample in samples:
        key = str(sample["type"])
        anomaly_type_counts[key] = anomaly_type_counts.get(key, 0) + 1

    severity_counts = {
        SEVERITY_NAMES[level]: int(np.count_nonzero(severity_map == level))
        for level in range(1, 4)
    }
    seam_band_count = int(np.count_nonzero(tangent.seam_band_mask))
    seam_band_anomalies = int(
        np.count_nonzero(anomaly_mask & tangent.seam_band_mask)
    )

    all_pixels = int(anomaly_mask.size)
    quality_count = min(int(validation["max_samples"]), all_pixels)
    quality_sample = _sample_indices(
        all_pixels,
        quality_count,
        seed ^ 0x5EED,
        bool(validation["stratified"]),
    )
    quality_flat = quality_sample
    quality_anomalies = int(
        np.count_nonzero(anomaly_mask.ravel()[quality_flat])
    )
    max_flat = int(np.argmax(angle))
    max_y, max_x = np.unravel_index(max_flat, angle.shape)
    direction_indices = np.flatnonzero(direction_mismatch)
    if direction_indices.size:
        min_alignment_selected = int(
            direction_indices[np.argmin(alignment.ravel()[direction_indices])]
        )
        min_alignment_y, min_alignment_x = np.unravel_index(
            min_alignment_selected, alignment.shape
        )
        min_alignment_value = float(alignment[min_alignment_y, min_alignment_x])
    else:
        min_alignment_y, min_alignment_x = max_y, max_x
        min_alignment_value = float(np.min(alignment[direction_usable])) if np.any(
            direction_usable
        ) else 1.0

    detail = {
        "reference_method": (
            "low-frequency base-color luminance gradient combined with the "
            "geometric normal and the UV-derived tangent frame"
        ),
        "usable_reference_pixels": int(np.count_nonzero(usable)),
        "pixels_with_anomaly": total_anomalies,
        "angle_anomaly_pixels": int(np.count_nonzero(angle_anomaly)),
        "direction_mismatch_pixels": int(np.count_nonzero(direction_mismatch)),
        "anomaly_ratio": round(
            total_anomalies / max(int(np.count_nonzero(usable)), 1), 9
        ),
        "angle_threshold_deg": threshold,
        "direction_alignment_threshold": -0.75,
        "max_angle_deg": round(float(angle[max_y, max_x]), 6),
        "max_angle_location": {"x": int(max_x), "y": int(max_y)},
        "min_direction_alignment": round(min_alignment_value, 6),
        "min_direction_alignment_location": {
            "x": int(min_alignment_x),
            "y": int(min_alignment_y),
        },
        "direction_usable_pixels": int(np.count_nonzero(direction_usable)),
        "severity_counts": severity_counts,
        "seam_band_pixels": seam_band_count,
        "seam_band_anomaly_pixels": seam_band_anomalies,
        "seam_band_anomaly_ratio": round(
            seam_band_anomalies / max(seam_band_count, 1), 9
        ),
        "sampling": {
            "seed": seed,
            "stratified": bool(validation["stratified"]),
            "anomaly_total": total_anomalies,
            "anomaly_sampled": int(sampled_indices.size),
            "quality_pixel_sample_count": int(quality_sample.size),
            "quality_pixel_sample_anomalies": quality_anomalies,
        },
        "sample_type_counts": anomaly_type_counts,
        "samples": samples,
    }
    return detail, severity_map


def _seam_check(
    scene: Scene,
    assets: LoadedAssets,
    tangent: TangentField,
) -> dict[str, Any]:
    island = tangent.island_id
    pair_x0: list[np.ndarray] = []
    pair_y0: list[np.ndarray] = []
    pair_x1: list[np.ndarray] = []
    pair_y1: list[np.ndarray] = []
    if island.shape[1] > 1:
        y, x = np.nonzero(island[:, 1:] != island[:, :-1])
        pair_x0.append(x)
        pair_y0.append(y)
        pair_x1.append(x + 1)
        pair_y1.append(y)
    if island.shape[0] > 1:
        y, x = np.nonzero(island[1:, :] != island[:-1, :])
        pair_x0.append(x)
        pair_y0.append(y)
        pair_x1.append(x)
        pair_y1.append(y + 1)
    if not pair_x0:
        return {
            "boundary_pairs": 0,
            "anomalous_pairs": 0,
            "anomaly_ratio": 0.0,
            "mirror_consistent_pairs": 0,
            "classification_counts": {},
            "tiles": [],
            "samples": [],
            "sampling": {
                "seed": int(scene.validation["seed"]),
                "stratified": bool(scene.validation["stratified"]),
                "anomaly_total": 0,
                "anomaly_sampled": 0,
            },
        }
    x0 = np.concatenate(pair_x0)
    y0 = np.concatenate(pair_y0)
    x1 = np.concatenate(pair_x1)
    y1 = np.concatenate(pair_y1)
    normal0 = tangent.object_normal_pixels(
        assets.normal_tangent,
        x0,
        y0,
        float(scene.baking["normal_strength"]),
    )
    normal1 = tangent.object_normal_pixels(
        assets.normal_tangent,
        x1,
        y1,
        float(scene.baking["normal_strength"]),
    )
    angle = _angle_degrees(normal0, normal1)
    tangent_angle = _angle_degrees(
        tangent.tangent[y0, x0], tangent.tangent[y1, x1]
    )
    handedness_change = tangent.handedness[y0, x0] != tangent.handedness[y1, x1]
    color_contrast = np.linalg.norm(
        assets.base_color[y0, x0] - assets.base_color[y1, x1], axis=1
    )
    threshold = float(scene.validation["seam_angle_threshold_deg"])
    anomaly = angle > threshold
    classification = np.full(angle.shape, "consistent", dtype=object)
    classification[handedness_change & ~anomaly] = "mirror_consistent"
    classification[anomaly & handedness_change] = "mirror_seam"
    classification[
        anomaly
        & ~handedness_change
        & (color_contrast >= float(scene.validation["hard_edge_color_threshold"]))
    ] = "hard_edge_seam"
    classification[
        anomaly
        & ~handedness_change
        & (color_contrast < float(scene.validation["hard_edge_color_threshold"]))
    ] = "flipped_seam"

    anomaly_indices = np.flatnonzero(anomaly)
    total_anomalies = int(anomaly_indices.size)
    max_entries = min(
        int(scene.validation["max_entries"]),
        int(scene.validation["max_samples"]),
    )
    sampled = _sample_indices(
        total_anomalies,
        max_entries,
        int(scene.validation["seed"]) ^ 0xA5A5,
        bool(scene.validation["stratified"]),
    )
    tile_size = int(scene.validation["tile_size"])
    severity = _severity(angle, threshold)
    samples: list[dict[str, Any]] = []
    tile_counts: dict[tuple[int, int], int] = {}
    for pair_index in anomaly_indices:
        tx = int(x0[pair_index]) // tile_size
        ty = int(y0[pair_index]) // tile_size
        tile_counts[(tx, ty)] = tile_counts.get((tx, ty), 0) + 1
    for pair_index in anomaly_indices[sampled]:
        samples.append(
            {
                "tile": [
                    int(x0[pair_index]) // tile_size,
                    int(y0[pair_index]) // tile_size,
                ],
                "boundary": [
                    [int(x0[pair_index]), int(y0[pair_index])],
                    [int(x1[pair_index]), int(y1[pair_index])],
                ],
                "islands": [
                    int(island[y0[pair_index], x0[pair_index]]),
                    int(island[y1[pair_index], x1[pair_index]]),
                ],
                "classification": str(classification[pair_index]),
                "severity": SEVERITY_NAMES[int(severity[pair_index])],
                "normal_angle_deg": round(float(angle[pair_index]), 6),
                "tangent_angle_deg": round(
                    float(tangent_angle[pair_index]), 6
                ),
                "handedness_change": bool(handedness_change[pair_index]),
                "base_color_contrast": round(
                    float(color_contrast[pair_index]), 6
                ),
                "normal_a": _round_vector(normal0[pair_index]),
                "normal_b": _round_vector(normal1[pair_index]),
            }
        )
    classes, counts = np.unique(classification, return_counts=True)
    classification_counts = {
        str(key): int(value)
        for key, value in sorted(zip(classes, counts), key=lambda item: str(item[0]))
    }
    return {
        "boundary_pairs": int(angle.size),
        "anomalous_pairs": total_anomalies,
        "anomaly_ratio": round(total_anomalies / max(int(angle.size), 1), 9),
        "angle_threshold_deg": threshold,
        "mirror_consistent_pairs": int(
            np.count_nonzero(classification == "mirror_consistent")
        ),
        "classification_counts": classification_counts,
        "tiles": [
            {"tile": [tx, ty], "anomaly_count": count}
            for (tx, ty), count in sorted(tile_counts.items())
        ],
        "sampling": {
            "seed": int(scene.validation["seed"]) ^ 0xA5A5,
            "stratified": bool(scene.validation["stratified"]),
            "anomaly_total": total_anomalies,
            "anomaly_sampled": int(sampled.size),
        },
        "samples": samples,
    }


def validate_normals(
    scene: Scene,
    assets: LoadedAssets,
    tangent: TangentField,
    object_normals: np.ndarray,
) -> ValidationResult:
    normal_detail, severity_map = _normal_check(
        scene, assets, tangent, object_normals
    )
    seam_detail = _seam_check(scene, assets, tangent)

    mask = np.zeros((*severity_map.shape, 3), dtype=np.float32)
    mask[:, :] = np.array([0.10, 0.12, 0.14], dtype=np.float32)
    colors = {
        1: [0.95, 0.78, 0.16],
        2: [0.95, 0.40, 0.10],
        3: [0.92, 0.08, 0.14],
    }
    for level, color in colors.items():
        mask[severity_map == level] = color
    mask[tangent.boundary_mask] = [0.15, 0.45, 0.95]
    for sample in seam_detail["samples"]:
        x, y = sample["boundary"][0]
        mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = [0.70, 0.16, 0.92]

    anomalies = int(normal_detail["pixels_with_anomaly"])
    seam_anomalies = int(seam_detail["anomalous_pairs"])
    if anomalies or seam_anomalies:
        status = "warning"
    else:
        status = "ok"
    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "normal_space": normal_detail,
        "seams": seam_detail,
        "summary": {
            "normal_anomaly_pixels": anomalies,
            "seam_anomaly_pairs": seam_anomalies,
            "seam_mirror_consistent_pairs": seam_detail[
                "mirror_consistent_pairs"
            ],
            "fatal_error": False,
        },
    }
    return ValidationResult(report=report, mask=mask)


def error_report(error: Any) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "status": "error",
        "error": error.as_dict(),
        "summary": {"fatal_error": True},
    }
