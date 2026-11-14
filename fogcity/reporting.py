"""Frame and aggregate statistics with public reproduction rules."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import __version__
from .interpolation import CameraState, camera_stats_with_segment_times
from .ordering import PanelOrder, ordering_stats
from .output import frame_json_name, frame_png_name
from .scene import Scene
from .util import quantize, quantize_vec


def make_frame_stats(
    scene: Scene,
    camera: CameraState,
    orders: tuple[PanelOrder, ...],
    image_sha256: str,
    *,
    building_triangle_count: int,
) -> dict[str, Any]:
    panel_lookup = {panel.id: panel for panel in scene.glass_panels}
    sigma_t = scene.fog.density * scene.fog.extinction
    sigma_s = sigma_t * scene.fog.scattering
    step_length = (camera.far - camera.near) / scene.fog.steps
    interpolation = camera_stats_with_segment_times(camera, scene)
    drawn = [order for order in orders if order.rendered]
    return {
        "schema_version": 1,
        "frame_index": camera.frame_index,
        "time": quantize(camera.time),
        "image": frame_png_name(camera.frame_index),
        "stats_file": frame_json_name(camera.frame_index),
        "image_sha256": image_sha256,
        "camera": interpolation,
        "volume": {
            "ray_space": "camera_view_distance",
            "step_rule": "fixed_near_to_far_step_with_ray_end_clamp",
            "step_count": scene.fog.steps,
            "step_length": quantize(step_length),
            "near": quantize(camera.near),
            "far": quantize(camera.far),
            "density": quantize(scene.fog.density),
            "extinction": quantize(scene.fog.extinction),
            "scattering": quantize(scene.fog.scattering),
            "sigma_t": quantize(sigma_t),
            "sigma_s": quantize(sigma_s),
            "anisotropy": quantize(scene.fog.anisotropy),
            "ambient": quantize(scene.fog.ambient),
            "integral": (
                "inscatter += exp(-sigma_t*t) * sigma_s * HG(-dot(ray,L),g) "
                "* light_color * intensity * shadow_visibility * dt"
            ),
            "composite": "base_rgb * transmittance + inscribed_fog_rgb",
        },
        "transparent_sort": ordering_stats(orders, panel_lookup),
        "counts": {
            "building_count": len(scene.buildings),
            "building_triangle_count": building_triangle_count,
            "panel_count": len(scene.glass_panels),
            "panels_drawn": len(drawn),
            "panels_backface_culled": sum(
                1 for order in orders if not order.front_facing
            ),
            "panels_behind_camera": sum(
                1 for order in orders if order.behind_camera
            ),
        },
        "light_direction": quantize_vec(scene.light.direction),
        "light_color": quantize_vec(scene.light.color),
        "light_intensity": quantize(scene.light.intensity),
        "fog_color": quantize_vec(scene.fog.color),
        "output_quantization": "floor(clamp(rgb,0,1)*255+0.5) uint8",
    }


def make_manifest(
    scene: Scene,
    frame_stats: list[dict[str, Any]],
    gl_info: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "renderer": {
            "name": "fogcity-preview",
            "version": __version__,
            "required_opengl": "4.1 core",
            "opengl_vendor": gl_info.get("GL_VENDOR", ""),
            "opengl_renderer": gl_info.get("GL_RENDERER", ""),
            "opengl_version": gl_info.get("GL_VERSION", ""),
        },
        "scene": {
            "name": scene.name,
            "sha256": scene.source_sha256,
            "resolution": [scene.width, scene.height],
            "fps": quantize(scene.fps),
            "start_time": quantize(scene.start_time),
            "end_time": quantize(scene.end_time),
            "frame_count": len(scene.frame_times),
            "keyframe_count": len(scene.keyframes),
        },
        "reproduction": {
            "interpolation": (
                "alpha=(time-t0)/(t1-t0); position=(1-alpha)*p0+alpha*p1; "
                "orientation=normalize(shortest_path_slerp(q0,q1,alpha)); "
                "fov/near/far are linear; quaternion is xyzw"
            ),
            "panel_sort": (
                "view_depth=dot(center-camera_position,camera_forward); "
                "round to 1e-6; sort desc; tie by panel id asc"
            ),
            "render_order": (
                "opaque prepass -> shadow ID/depth prepass -> base color -> "
                "volume composite -> transparent panels far-to-near"
            ),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "frames": frame_stats,
    }

