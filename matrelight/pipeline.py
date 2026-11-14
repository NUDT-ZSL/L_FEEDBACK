"""单材质处理管线与原子输出管理。"""

from __future__ import annotations
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import image_io
from .basis import reconstruct_tangent_basis
from .blend import (
    composite_normals,
    composite_scalar_fields,
    normalized_weights,
    per_layer_contributions,
)
from .cache import write_cache
from .deterministic import DeterministicContext
from .material import parse_material
from .sampler import LayerTextures, sample_uv
from .shading import render_lights


@dataclass
class ProcessResult:
    input_path: str
    output_dir: str
    status: str
    seed: int
    files: list[str]


def _atomic_directory(final_dir: Path):
    """在目标旁创建临时目录，成功后整体替换，失败自动清理，杜绝半成品。"""
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.", dir=final_dir.parent))
    return tmp


def _commit_directory(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


def process_material(
    input_path: str | Path,
    output_root: str | Path,
    seed: int,
) -> ProcessResult:
    """执行完整管线，业务错误以 MaterialError 抛出。"""
    input_path = Path(input_path)
    material = parse_material(input_path)
    ctx = DeterministicContext(seed, tag=material.material_name)

    final_dir = Path(output_root) / material.material_name
    work_dir = _atomic_directory(final_dir)
    try:
        textures = [
            LayerTextures(spec, material.width, material.height)
            for spec in material.layers
        ]
        uv_samples = sample_uv(textures, material)

        # 权重只归一化一次，法线/颜色/渲染/缓存共用同一数组
        weights = normalized_weights(material.layers)
        rng_normal = ctx.stage_rng("normal_composite")
        normals = composite_normals(textures, weights, rng_normal)

        rng_fields = ctx.stage_rng("scalar_fields")
        base_color, roughness = composite_scalar_fields(textures, weights, rng_fields)

        rng_basis = ctx.stage_rng("tangent_basis")
        tangents, bitangents, basis_meta, correction_marker = reconstruct_tangent_basis(
            normals, material.tangent_threshold_deg, rng_basis
        )

        # 基修正不得改动法线：断言长度与方向不变
        assert np.allclose(np.linalg.norm(normals, axis=-1), 1.0, atol=1e-9)

        rng_render = ctx.stage_rng("ggx_render")
        light_results = render_lights(
            normals,
            base_color,
            roughness,
            material.lights,
            material.view_dir,
            rng_render,
        )

        normal_png = work_dir / "normal_composite.png"
        image_io.save_normal_png(normal_png, normals)

        files: list[str] = [normal_png.name]
        light_manifests = []
        for result in light_results:
            stem = f"reflection_light_{result['index']:02d}"
            npy_path = work_dir / f"{stem}.npy"
            png_path = work_dir / f"{stem}.png"
            image_io.save_npy(npy_path, result["hdr"])
            image_io.save_rgb_png(png_path, image_io.tone_map(result["hdr"]))
            files.extend([npy_path.name, png_path.name])
            light_manifests.append(
                {
                    "index": result["index"],
                    "light_direction": result["light_direction"],
                    "view_direction": result["view_direction"],
                    "npy": npy_path.name,
                    "png": png_path.name,
                    "stats": result["stats"],
                }
            )

        metadata = {
            "material_name": material.material_name,
            "source_yaml": str(material.source_path),
            "seed": int(seed),
            "resolution": [material.width, material.height],
            "normal_encoding": "opengl_plus_y",
            "brdf": "ggx_trowbridge_reitz",
            "f0_dielectric": 0.04,
            "uv_wrap": material.uv_wrap,
            "normalized_weights": weights.tolist(),
            "layer_contributions": per_layer_contributions(textures, weights),
            "tangent_basis": basis_meta,
            "lights": light_manifests,
            "uv_samples": uv_samples,
            "final_brdf_params": {
                "roughness_min": float(np.min(roughness)),
                "roughness_max": float(np.max(roughness)),
                "roughness_mean": float(np.mean(roughness)),
                "base_color_mean": np.mean(base_color, axis=(0, 1)).tolist(),
            },
        }
        arrays = {
            "normal": normals,
            "tangent": tangents,
            "bitangent": bitangents,
            "base_color": base_color,
            "roughness": roughness,
            "correction_marker": correction_marker.astype(np.uint8).astype(np.float32),
        }
        cache_path = work_dir / "material_cache.mzc"
        write_cache(cache_path, metadata, arrays)
        files.append(cache_path.name)

        manifest = {
            "ok": True,
            "material_name": material.material_name,
            "seed": int(seed),
            "status": "OK",
            "files": sorted(files),
        }
        (work_dir / "result.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        files.append("result.json")

        _commit_directory(work_dir, final_dir)
        return ProcessResult(
            input_path=str(input_path),
            output_dir=str(final_dir),
            status="OK",
            seed=int(seed),
            files=sorted(files),
        )
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        # 若输出根目录是本次创建且已空，则一并删除，避免留下半成品根
        root = Path(output_root)
        try:
            if root.exists() and not any(root.iterdir()):
                root.rmdir()
        except OSError:
            pass
        raise
