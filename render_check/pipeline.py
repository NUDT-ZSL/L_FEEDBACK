"""GLSL compilation, linking, and interface validation."""

from __future__ import annotations

import moderngl

from .errors import gl_compile_error, gl_link_error
from .materials import Material


REQUIRED_PROGRAM_UNIFORMS = frozenset(
    {
        "u_model",
        "u_view",
        "u_proj",
        "u_camera_pos",
        "u_light_pos",
        "u_light_color",
        "u_base_color",
        "u_metallic",
        "u_roughness",
        "u_emissive_strength",
        "u_uv_scale",
        "u_object_id",
        "u_material_index",
        "u_shape_id",
        "u_albedo_tex",
    }
)
REQUIRED_ATTRIBUTES = frozenset({"in_position", "in_normal", "in_uv"})


def active_uniforms(program: moderngl.Program) -> dict[str, moderngl.Uniform]:
    result: dict[str, moderngl.Uniform] = {}
    for name in program:
        member = program[name]
        if isinstance(member, moderngl.Uniform):
            result[name] = member
    return result


def _preflight_stage(
    ctx: moderngl.Context,
    material: Material,
    stage: str,
    source: str,
    shader_path: str,
) -> None:
    shader = None
    try:
        shader = ctx.program(**{f"{stage}_shader": source})
    except Exception as exc:
        message = str(exc)
        if "compil" in message.lower():
            raise gl_compile_error(
                f"material '{material.name}': {stage} shader compilation failed: {message}",
                material=material.name,
                stage=stage,
                path=shader_path,
                gl_message=message,
            ) from exc
        return
    finally:
        if shader is not None:
            shader.release()


def _validate_interface(program: moderngl.Program, material: Material) -> None:
    uniforms = active_uniforms(program)
    missing_uniforms = sorted(REQUIRED_PROGRAM_UNIFORMS.difference(uniforms))
    if missing_uniforms:
        raise gl_link_error(
            f"material '{material.name}': linked program is missing required uniforms: "
            f"{', '.join(missing_uniforms)}",
            material=material.name,
            missing_uniforms=missing_uniforms,
        )

    attributes = {
        name
        for name in program
        if isinstance(program[name], moderngl.Attribute)
    }
    missing_attributes = sorted(REQUIRED_ATTRIBUTES.difference(attributes))
    if missing_attributes:
        raise gl_link_error(
            f"material '{material.name}': linked program is missing required attributes: "
            f"{', '.join(missing_attributes)}",
            material=material.name,
            missing_attributes=missing_attributes,
        )


def compile_material_programs(
    ctx: moderngl.Context,
    materials: list[Material],
) -> dict[str, moderngl.Program]:
    programs: dict[str, moderngl.Program] = {}
    try:
        for material in materials:
            _preflight_stage(
                ctx,
                material,
                "vertex",
                material.vertex_source,
                str(material.vertex_shader),
            )
            _preflight_stage(
                ctx,
                material,
                "fragment",
                material.fragment_source,
                str(material.fragment_shader),
            )
            try:
                program = ctx.program(
                    vertex_shader=material.vertex_source,
                    fragment_shader=material.fragment_source,
                )
            except Exception as exc:
                message = str(exc)
                if "compiler failed" in message.lower():
                    raise gl_compile_error(
                        f"material '{material.name}': shader compilation failed: {message}",
                        material=material.name,
                        gl_message=message,
                    ) from exc
                raise gl_link_error(
                    f"material '{material.name}': program link failed: {message}",
                    material=material.name,
                    gl_message=message,
                ) from exc
            _validate_interface(program, material)
            programs[material.name] = program
    except Exception:
        for program in programs.values():
            program.release()
        raise
    return programs

