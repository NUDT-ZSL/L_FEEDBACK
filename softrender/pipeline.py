"""Render pipeline orchestration: load -> shadows -> raster -> tonemap."""

from __future__ import annotations

import time

import numpy as np

from . import geometry, raster, scene as scene_mod
from .errors import RenderError
from .normalmap import NormalMap
from .obj_loader import load_obj
from .shading import Shader
from .shadow import CubeShadowMap, DirShadowMap


def _load_world(scene) -> tuple:
    """Load all models; return world-space triangle arrays + materials."""
    materials = []
    worlds, norms, uvs, mats = [], [], [], []
    for spec in scene.models:
        obj = load_obj(spec.obj_path)
        if obj["uv"] is None:
            raise RenderError(f"{spec.obj_path}: model has no texture coordinates")
        material = scene_mod.load_material(spec.material_path)
        mat_idx = len(materials)
        materials.append(material)
        pos = obj["pos"]
        world = pos.reshape(-1, 3) @ spec.transform[:3, :3].T + spec.transform[:3, 3]
        world = world.reshape(pos.shape)
        if obj["nrm"] is not None:
            nrm = obj["nrm"].reshape(-1, 3) @ spec.transform[:3, :3].T
            nrm = nrm.reshape(obj["nrm"].shape)
        else:
            e1 = world[:, 1] - world[:, 0]
            e2 = world[:, 2] - world[:, 0]
            fn = np.cross(e1, e2)
            fn /= np.maximum(np.linalg.norm(fn, axis=2, keepdims=True), 1e-12)
            nrm = np.broadcast_to(fn[:, None, :], world.shape).copy()
        worlds.append(world)
        norms.append(nrm)
        uvs.append(obj["uv"])
        mats.append(np.full(pos.shape[0], mat_idx, dtype=np.int32))
    return (np.concatenate(worlds), np.concatenate(norms),
            np.concatenate(uvs), np.concatenate(mats), materials)


def render_scene(scene_path: str, width: int, height: int, nthreads: int,
                 shadow_res: int) -> tuple:
    """Render a scene. Returns (HDR float32 HxWx3, timings dict)."""
    timings = {}
    t0 = time.perf_counter()
    scene = scene_mod.load_scene(scene_path)
    world, nrm, uv, mat, materials = _load_world(scene)
    normal_maps = [NormalMap.from_file(m.normal_map) if m.normal_map else None
                   for m in materials]
    timings["load"] = time.perf_counter() - t0
    bmin = world.reshape(-1, 3).min(axis=0)
    bmax = world.reshape(-1, 3).max(axis=0)
    # ---- shadow maps ----
    t1 = time.perf_counter()
    dir_shadows = []
    for light in scene.dir_lights:
        rig = geometry.directional_light_camera(light.direction, bmin, bmax)
        stris = geometry.assemble_triangles(world, nrm, uv, mat, rig, shadow_res, shadow_res)
        _, depth = raster.rasterize(stris, shadow_res, shadow_res, nthreads)
        dir_shadows.append(DirShadowMap(depth, rig, light.direction))
    cube_shadow = None
    for light in scene.point_lights:
        diag = float(np.linalg.norm(bmax - bmin))
        near = max(0.02, diag * 0.005)
        far = (float(np.linalg.norm(bmax - light.position))
               + float(np.linalg.norm(bmin - light.position)) + diag)
        faces = np.empty((6, shadow_res, shadow_res), dtype=np.float32)
        for face in range(6):
            rig = geometry.point_light_face_camera(light.position, face, near, far)
            stris = geometry.assemble_triangles(world, nrm, uv, mat, rig, shadow_res, shadow_res)
            _, depth = raster.rasterize(stris, shadow_res, shadow_res, nthreads,
                                        radial_depth_from=light.position)
            faces[face] = depth
        cube_shadow = CubeShadowMap(faces, light.position)
    timings["shadow_maps"] = time.perf_counter() - t1
    # ---- main pass ----
    t2 = time.perf_counter()
    rig = geometry.main_camera(bmin, bmax, width, height)
    stris = geometry.assemble_triangles(world, nrm, uv, mat, rig, width, height)
    geometry.compute_tangents(stris)
    shader = Shader(
        materials=materials,
        normal_maps=normal_maps,
        eye=rig.eye,
        dir_lights=scene.dir_lights,
        point_lights=scene.point_lights,
        dir_shadows=dir_shadows,
        cube_shadow=cube_shadow,
        ambient=scene.ambient,
        tangents=stris["tangent"],
        bitangents=stris["bitangent"],
        tri_mat=stris["mat"],
    )
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    background = (np.array([0.055, 0.065, 0.09], dtype=np.float32)[None, None, :] * (1 - ramp)
                  + np.array([0.16, 0.19, 0.26], dtype=np.float32)[None, None, :] * ramp)
    background = np.broadcast_to(background, (height, width, 3)).copy()
    color, _ = raster.rasterize(stris, width, height, nthreads,
                                shader=shader, background=background)
    timings["main_pass"] = time.perf_counter() - t2
    timings["total"] = time.perf_counter() - t0
    return color, timings
