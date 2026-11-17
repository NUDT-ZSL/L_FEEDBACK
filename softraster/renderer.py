"""End-to-end offline render pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .postprocess import postprocess
from .raster import GBuffer, mesh_frustum_visibility, rasterize_gbuffer
from .scene import Scene, load_scene
from .shading import shade_scene
from .shadows import ShadowMap, build_shadow_maps


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray
    gbuffer: GBuffer
    shadow_maps: tuple[ShadowMap | None, ...]
    mesh_visibility: np.ndarray


def render_scene(scene: Scene) -> RenderResult:
    visibility = mesh_frustum_visibility(scene.camera, scene.meshes)
    gbuffer = rasterize_gbuffer(scene.camera, scene.meshes, visibility)
    shadow_maps = build_shadow_maps(scene)
    image = shade_scene(scene, gbuffer, shadow_maps)
    return RenderResult(
        image=image,
        gbuffer=gbuffer,
        shadow_maps=shadow_maps,
        mesh_visibility=visibility,
    )


def render_postprocessed(
    scene: Scene,
    *,
    tonemap: str | None = None,
    gamma: bool = False,
) -> RenderResult:
    result = render_scene(scene)
    return RenderResult(
        image=postprocess(result.image, tonemap=tonemap, gamma=gamma),
        gbuffer=result.gbuffer,
        shadow_maps=result.shadow_maps,
        mesh_visibility=result.mesh_visibility,
    )


def render_path(
    path: str | Path,
    *,
    tonemap: str | None = None,
    gamma: bool = False,
) -> tuple[Scene, RenderResult]:
    scene = load_scene(path)
    return scene, render_postprocessed(scene, tonemap=tonemap, gamma=gamma)
