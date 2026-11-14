"""方向光反射渲染，GGX (Trowbridge-Reitz) 微表面模型。"""

from __future__ import annotations

import numpy as np

from .deterministic import DeterministicContext

F0_DIELECTRIC = 0.04
def _ggx_ndf(n_dot_h: np.ndarray, alpha_sq: np.ndarray) -> np.ndarray:
    denom_term = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0
    return alpha_sq / (np.pi * denom_term * denom_term)


def _schlick_ggx_g1(n_dot_x: np.ndarray, k: np.ndarray) -> np.ndarray:
    return n_dot_x / (n_dot_x * (1.0 - k) + k)


def render_lights(
    normals: np.ndarray,
    base_color: np.ndarray,
    roughness: np.ndarray,
    light_directions: list[np.ndarray],
    view_direction: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """对每盏方向光渲染一张线性 HDR 反射图 (H,W,3)。

    所有方向向量进入计算前单位化；半角向量归一化与各分量归约均走
    DeterministicContext，种子固定时浮点中间顺序一致。
    """
    view = DeterministicContext.seeded_normalize(rng, view_direction, axis=0)
    rough = np.clip(roughness.astype(np.float64, copy=False), 1e-3, 1.0)
    alpha = rough * rough
    alpha_sq = alpha * alpha
    # 直接光的 Smith 几何修正系数
    k = ((rough + 1.0) * (rough + 1.0)) / 8.0
    f0 = np.broadcast_to(F0_DIELECTRIC, base_color.shape).astype(np.float64)

    results: list[dict] = []
    for index, raw_light in enumerate(light_directions):
        light = DeterministicContext.seeded_normalize(rng, raw_light, axis=0)
        half_vec = DeterministicContext.seeded_normalize(rng, light + view, axis=0)

        n_dot_l = np.clip(np.sum(normals * light, axis=-1), 0.0, 1.0)
        n_dot_v = np.clip(np.sum(normals * view, axis=-1), 0.0, 1.0)
        n_dot_h = np.clip(np.sum(normals * half_vec, axis=-1), 0.0, 1.0)
        v_dot_h = np.clip(np.dot(view, half_vec), 0.0, 1.0)

        distribution = _ggx_ndf(n_dot_h, alpha_sq)
        g1_v = _schlick_ggx_g1(np.maximum(n_dot_v, 1e-6), k)
        g1_l = _schlick_ggx_g1(np.maximum(n_dot_l, 1e-6), k)
        geometry = g1_v * g1_l
        fresnel = f0 + (1.0 - f0) * ((1.0 - v_dot_h)[..., None] ** 5)

        denom = 4.0 * np.maximum(n_dot_v, 1e-6) * np.maximum(n_dot_l, 1e-6) + 1e-6
        specular = (distribution[..., None] * geometry[..., None] * fresnel) / denom[..., None]

        # Lambert 漫反射：保留 (1-F) 能量，与镜面反射互补
        diffuse_kd = (1.0 - fresnel) * (1.0 - F0_DIELECTRIC)
        diffuse = diffuse_kd * base_color / np.pi

        radiance = (diffuse + specular) * n_dot_l[..., None]
        radiance = np.nan_to_num(radiance, nan=0.0, posinf=0.0, neginf=0.0)
        results.append(
            {
                "index": index,
                "light_direction": light.tolist(),
                "view_direction": view.tolist(),
                "hdr": radiance.astype(np.float32),
                "stats": {
                    "mean": float(np.mean(radiance)),
                    "max": float(np.max(radiance)),
                },
            }
        )
    return results
