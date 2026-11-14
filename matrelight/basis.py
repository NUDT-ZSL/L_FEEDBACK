"""切线空间基重建与不连续修正。

初始切线来自合成法线的局部屏幕空间差分；从左上像素开始以固定光栅顺序
扫描，相邻切线夹角超过阈值时翻转符号并对法线重投影。修正只改变切线基，
合成法线的方向与长度保持不变。
"""

from __future__ import annotations

import numpy as np

from .deterministic import DeterministicContext


def _initial_tangents(normals: np.ndarray) -> np.ndarray:
    """由法线在屏幕空间的方向变化构造未修正切线 (H,W,3)。

    取法线切向投影 (nx,ny,0) 作为表面流向，再保证与法线正交；
    法线方向发生符号翻转时该流向随之反向，从而显式暴露不连续。
    """
    height, width = normals.shape[:2]
    tangents = np.zeros_like(normals)
    tangents[..., 0] = normals[..., 0]
    tangents[..., 1] = normals[..., 1]
    tangents[..., 2] = 1.0
    # 逐像素先对法线重投影，退化处在重建阶段用世界轴兜底
    for y in range(height):
        for x in range(width):
            normal = normals[y, x]
            tangent = tangents[y, x] - np.dot(tangents[y, x], normal) * normal
            if np.linalg.norm(tangent) < 1e-9:
                tangent = np.array([1.0, 0.0, 0.0]) - normal[0] * normal
            tangents[y, x] = tangent
    return tangents


def _reorthogonalize(tangent: np.ndarray, normal: np.ndarray) -> np.ndarray:
    tangent = tangent - np.dot(tangent, normal) * normal
    length = float(np.linalg.norm(tangent))
    if length < 1e-12:
        # 退化时选择与法线最不平行的世界轴重建切线
        axis = min(range(3), key=lambda k: abs(float(normal[k])))
        fallback = np.zeros(3, dtype=np.float64)
        fallback[axis] = 1.0
        tangent = fallback - np.dot(fallback, normal) * normal
        length = float(np.linalg.norm(tangent))
    return tangent / length


def reconstruct_tangent_basis(
    normals: np.ndarray,
    threshold_deg: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """返回 (tangents, bitangents, metadata)。

    所有向量输出前统一走种子控制的归一化，保证归约顺序可复现。
    """
    height, width = normals.shape[:2]
    raw_tangents = _initial_tangents(normals)
    tangents = np.zeros_like(raw_tangents)
    bitangents = np.zeros_like(raw_tangents)
    corrected = np.zeros((height, width), dtype=bool)
    references: list[list[str | None]] = [[None] * width for _ in range(height)]
    cos_threshold = float(np.cos(np.deg2rad(threshold_deg)))

    correction_count = 0
    max_angle_seen = 0.0
    for y in range(height):
        for x in range(width):
            normal = normals[y, x]
            candidate = raw_tangents[y, x]
            ref = None
            if x > 0:
                ref = tangents[y, x - 1]
                references[y][x] = "left"
            elif y > 0:
                ref = tangents[y - 1, x]
                references[y][x] = "up"

            if ref is not None:
                ref_xy = ref - np.dot(ref, normal) * normal
                ref_norm = float(np.linalg.norm(ref_xy))
                cand_xy = candidate - np.dot(candidate, normal) * normal
                cand_norm = float(np.linalg.norm(cand_xy))
                if ref_norm > 1e-12 and cand_norm > 1e-12:
                    cos_angle = float(np.dot(ref_xy, cand_xy) / (ref_norm * cand_norm))
                    cos_angle = min(1.0, max(-1.0, cos_angle))
                    angle = float(np.rad2deg(np.arccos(cos_angle)))
                    max_angle_seen = max(max_angle_seen, angle)
                    if cos_angle < cos_threshold:
                        candidate = -candidate
                        corrected[y, x] = True
                        correction_count += 1

            tangent = _reorthogonalize(candidate, normal)
            tangent = DeterministicContext.seeded_normalize(rng, tangent, axis=0)
            # 右手系副切线：B = normalize(cross(N, T))
            bitangent = np.cross(normal, tangent)
            bitangent = DeterministicContext.seeded_normalize(rng, bitangent, axis=0)
            tangents[y, x] = tangent
            bitangents[y, x] = bitangent

    metadata = {
        "threshold_deg": float(threshold_deg),
        "cos_threshold": cos_threshold,
        "correction_enabled": True,
        "correction_count": int(correction_count),
        "correction_ratio": float(correction_count / float(height * width)),
        "max_adjacent_angle_deg": float(max_angle_seen),
        "sweep_order": "raster-left-to-right-then-top-to-bottom",
        "correction_marker_shape": [height, width],
    }
    return tangents, bitangents, metadata, corrected
