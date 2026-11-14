"""层权重归一化与分层合成（法线/基础色/粗糙度）。

权重在本模块只计算一次，归一化结果同时驱动法线合成、颜色粗糙度合并
与反射渲染，并原样写入缓存，保证三处使用的是同一份数据。
"""

from __future__ import annotations

import numpy as np

from .deterministic import DeterministicContext
from .errors import MaterialError
from .material import LayerSpec
from .sampler import LayerTextures


def normalized_weights(layers: list[LayerSpec]) -> np.ndarray:
    """标量权重归一化；总权重非正按 E_LAYER_INVALID 处理。"""
    weights = np.asarray([layer.weight for layer in layers], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise MaterialError(
            "E_LAYER_INVALID",
            "全部层权重之和非正，无法归一化",
            {"weights": weights.tolist(), "sum": total},
        )
    return weights / total


def _overlay_normal(
    accum: np.ndarray, layer_normal: np.ndarray, weight: float
) -> np.ndarray:
    """叠加：围绕平面法线 z=(0,0,1) 做带权细节提升。"""
    detail = layer_normal.copy()
    detail[..., 2] = 0.0
    return accum + weight * detail


def composite_normals(
    textures: list[LayerTextures],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """按层顺序合成切线空间法线，末尾统一重归一化。"""
    modes = [tex.spec.mode for tex in textures]
    if all(mode == "weighted" for mode in modes):
        # 纯加权快速路径：全局加权求和，仅做一次重归一化
        weighted = [
            weights[i] * textures[i].normal.astype(np.float64, copy=False)
            for i in range(len(textures))
        ]
        summed = DeterministicContext.ordered_sum(rng, weighted)
        return DeterministicContext.seeded_normalize(rng, summed, axis=-1)

    first = textures[0].normal.astype(np.float64, copy=False)
    accum = first.copy()
    for i in range(1, len(textures)):
        tex = textures[i]
        n = tex.normal.astype(np.float64, copy=False)
        w = float(weights[i])
        if tex.spec.mode == "overlay":
            accum = _overlay_normal(accum, n, w)
        elif tex.spec.mode == "mix":
            accum = (1.0 - w) * accum + w * n
        else:  # weighted：在顺序路径中按权重并入
            accum = accum + w * n
        accum = DeterministicContext.seeded_normalize(rng, accum, axis=-1)
    return DeterministicContext.seeded_normalize(rng, accum, axis=-1)


def _overlay_channel(base: np.ndarray, incoming: np.ndarray, weight: float) -> np.ndarray:
    """Photoshop 风格通道叠加（<=0.5 变暗，否则变亮），再按权重混入。"""
    low = 2.0 * base * incoming
    high = 1.0 - 2.0 * (1.0 - base) * (1.0 - incoming)
    blended = np.where(base <= 0.5, low, high)
    return (1.0 - weight) * base + weight * blended


def composite_scalar_fields(
    textures: list[LayerTextures],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """合并基础色 (H,W,3) 与粗糙度 (H,W)，返回 (base_color, roughness)。"""
    modes = [tex.spec.mode for tex in textures]
    if all(mode == "weighted" for mode in modes):
        color_terms = [
            weights[i] * textures[i].color.astype(np.float64, copy=False)
            for i in range(len(textures))
        ]
        rough_terms = [
            weights[i] * textures[i].roughness.astype(np.float64, copy=False)
            for i in range(len(textures))
        ]
        color = np.clip(DeterministicContext.ordered_sum(rng, color_terms), 0.0, 1.0)
        roughness = np.clip(DeterministicContext.ordered_sum(rng, rough_terms), 0.0, 1.0)
        return color, roughness

    color = textures[0].color.astype(np.float64, copy=False).copy()
    roughness = textures[0].roughness.astype(np.float64, copy=False).copy()
    for i in range(1, len(textures)):
        tex = textures[i]
        w = float(weights[i])
        incoming_color = tex.color.astype(np.float64, copy=False)
        incoming_rough = tex.roughness.astype(np.float64, copy=False)
        if tex.spec.mode == "overlay":
            color = _overlay_channel(color, incoming_color, w)
            roughness = _overlay_channel(roughness, incoming_rough, w)
        else:  # mix / weighted 在顺序路径中均为线性插值
            color = (1.0 - w) * color + w * incoming_color
            roughness = (1.0 - w) * roughness + w * incoming_rough
    return np.clip(color, 0.0, 1.0), np.clip(roughness, 0.0, 1.0)


def per_layer_contributions(
    textures: list[LayerTextures], weights: np.ndarray
) -> list[dict]:
    """写入缓存的每层贡献元数据，归一化权重即实际渲染使用的权重。"""
    contributions = []
    for i, tex in enumerate(textures):
        contributions.append(
            {
                "index": tex.spec.index,
                "name": tex.spec.name,
                "mode": tex.spec.mode,
                "raw_weight": float(tex.spec.weight),
                "normalized_weight": float(weights[i]),
                "uv_scale": list(tex.spec.uv_scale),
                "uv_offset": list(tex.spec.uv_offset),
                "normal_path": str(tex.spec.normal_path),
                "base_color_path": str(tex.spec.color_path),
                "roughness_path": str(tex.spec.roughness_path),
            }
        )
    return contributions
