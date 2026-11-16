from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import glfw

from .config import DiagnosticLog, load_scene_config
from .diagnostics_cli import configure_logging
from .renderer import Renderer
from .scene_generator import DEFAULT_SEED, generate_scene, write_generated_scene

LOGGER = logging.getLogger("proc_gi.app")


def _ensure_scene(path: Path) -> None:
    if not path.exists():
        LOGGER.info("scene file missing; generating deterministic scene at %s", path)
        write_generated_scene(path, DEFAULT_SEED)


def _create_window(width: int, height: int, *, visible: bool):
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.SAMPLES, 0)
    window = glfw.create_window(width, height, "proc_gi - OpenGL 3.3 VPL GI", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("failed to create an OpenGL 3.3 core-profile window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    return window


def _color_palette(index: int) -> tuple[float, float, float]:
    palette = (
        (1.0, 0.66, 0.35),
        (0.48, 0.68, 1.0),
        (1.0, 0.94, 0.82),
        (0.35, 1.0, 0.62),
        (1.0, 0.36, 0.52),
    )
    return palette[index % len(palette)]


def run(args: argparse.Namespace) -> int:
    configure_logging()
    scene_path = Path(args.scene)
    _ensure_scene(scene_path)
    diagnostics = DiagnosticLog()
    config = load_scene_config(scene_path, diagnostics)
    _, mesh = generate_scene(config.seed, diagnostics)
    width = args.width or config.render.width
    height = args.height or config.render.height
    window = _create_window(width, height, visible=not args.headless)
    renderer = Renderer(config, mesh, diagnostics, width=width, height=height)
    if args.max_vpls is not None or args.vpl_capacity is not None:
        renderer.set_clustering(
            max_vpls=args.max_vpls,
            vpl_capacity=args.vpl_capacity,
        )

    selected_light = 0
    tonemap = True
    dirty = True
    last_stats = renderer._stats
    budget_cycle = (
        (16, 16),
        (12, 12),
        (16, 8),
        (8, 8),
        (4, 4),
    )
    budget_index = 0

    def select_light(delta: int) -> None:
        nonlocal selected_light
        selected_light = (selected_light + delta) % len(config.lights)
        light = config.lights[selected_light]
        print(
            f"selected {light.id} type={light.type} "
            f"pos={tuple(round(v, 2) for v in light.position)} "
            f"intensity={light.intensity:.2f} color={tuple(round(v, 2) for v in light.color)}"
        )

    def key_callback(_window, key, _scancode, action, _mods) -> None:
        nonlocal selected_light, tonemap, dirty, budget_index, last_stats
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(_window, True)
            return
        if key == glfw.KEY_TAB:
            select_light(1)
            return
        light = config.lights[selected_light]
        x, y, z = light.position
        changed = False
        if key == glfw.KEY_LEFT:
            renderer.update_light(light.id, position=[x - 0.12, y, z])
            changed = True
        elif key == glfw.KEY_RIGHT:
            renderer.update_light(light.id, position=[x + 0.12, y, z])
            changed = True
        elif key == glfw.KEY_UP:
            renderer.update_light(light.id, position=[x, y, z - 0.12])
            changed = True
        elif key == glfw.KEY_DOWN:
            renderer.update_light(light.id, position=[x, y, z + 0.12])
            changed = True
        elif key == glfw.KEY_PAGE_UP:
            renderer.update_light(light.id, position=[x, min(3.0, y + 0.10), z])
            changed = True
        elif key == glfw.KEY_PAGE_DOWN:
            renderer.update_light(light.id, position=[x, max(0.25, y - 0.10), z])
            changed = True
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            renderer.update_light(light.id, intensity=light.intensity + 2.0)
            changed = True
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            renderer.update_light(light.id, intensity=max(0.1, light.intensity - 2.0))
            changed = True
        elif key == glfw.KEY_C:
            nearest = min(
                range(5),
                key=lambda index: sum(
                    abs(a - b) for a, b in zip(light.color, _color_palette(index))
                ),
            )
            renderer.update_light(light.id, color=list(_color_palette(nearest + 1)))
            changed = True
        elif key == glfw.KEY_T:
            tonemap = not tonemap
            dirty = True
            print(f"tonemap={tonemap}")
        elif key == glfw.KEY_S:
            color_path = renderer.save_screenshot(tonemap=True)
            linear_path = renderer.save_screenshot(tonemap=False)
            diagnostics.write(config.render.warning_path)
            print(f"saved {color_path} and {linear_path}")
        elif key == glfw.KEY_G:
            budget_index = (budget_index + 1) % len(budget_cycle)
            max_vpls, capacity = budget_cycle[budget_index]
            renderer.set_clustering(max_vpls=max_vpls, vpl_capacity=capacity)
            print(f"clustering max_vpls={max_vpls} vpl_capacity={capacity}")
            dirty = True
        elif key == glfw.KEY_R:
            renderer._force_full_rebuild = True
            dirty = True
        if changed:
            dirty = True
            print(
                f"updated {light.id}: "
                f"pos={tuple(round(v, 2) for v in config.lights[selected_light].position)} "
                f"intensity={config.lights[selected_light].intensity:.2f}"
            )

    glfw.set_key_callback(window, key_callback)
    frame_limit = args.frames if args.headless else None
    rendered_frames = 0
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if dirty or (frame_limit is not None and rendered_frames < frame_limit):
            last_stats = renderer.render()
            if rendered_frames == 0 and args.exercise_update:
                light = config.lights[0]
                renderer.update_light(light.id, intensity=light.intensity + 4.0)
                dirty = True
            renderer.present(tonemap=tonemap)
            renderer.blit_to_default_framebuffer()
            glfw.swap_buffers(window)
            rendered_frames += 1
            if not (args.exercise_update and rendered_frames == 1):
                dirty = False
            if not args.headless:
                glfw.set_window_title(
                    window,
                    f"proc_gi | VPLs {last_stats.active_vpl_count}/{last_stats.cluster_count} | "
                    f"cube rebuilds {last_stats.cube_builds} | frame {last_stats.frame}",
                )
            if frame_limit is not None and rendered_frames >= frame_limit:
                break
        elif args.headless and frame_limit is not None:
            break
        elif not args.headless:
            glfw.wait_events_timeout(0.05)
    if args.headless or args.save_on_exit:
        color_path = renderer.save_screenshot(tonemap=True)
        linear_path = renderer.save_screenshot(tonemap=False)
        diagnostics.write(config.render.warning_path)
        summary = {
            "frames_rendered": rendered_frames,
            "stats": last_stats.as_dict(),
            "screenshots": [str(color_path), str(linear_path)],
            "warning_path": config.render.warning_path,
            "warnings": diagnostics.records,
        }
        report_path = Path("artifacts/app_run.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    renderer.close()
    glfw.destroy_window(window)
    glfw.terminate()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenGL 3.3 deterministic VPL global-illumination previewer."
    )
    parser.add_argument("--scene", default="scene.json")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--exercise-update", action="store_true")
    parser.add_argument("--max-vpls", type=int)
    parser.add_argument("--vpl-capacity", type=int)
    parser.add_argument("--save-on-exit", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        LOGGER.exception("fatal renderer error")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

