from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .compiler import CompiledMaterial, Instruction, Operand
from .errors import PipelineError
from .raster import GBuffer
from .texture import sample_texture


@dataclass(frozen=True)
class TangentFrame:
    tangent: np.ndarray
    bitangent: np.ndarray
    degenerate: np.ndarray


@dataclass(frozen=True)
class MaterialBuffers:
    base_color: np.ndarray
    normal: np.ndarray
    roughness: np.ndarray
    metallic: np.ndarray
    emissive: np.ndarray
    degenerate_tangent: np.ndarray
    normal_fallback: np.ndarray


def fresnel_schlick(cos_theta: np.ndarray, ior: np.ndarray) -> np.ndarray:
    cosine = np.clip(np.asarray(cos_theta, dtype=np.float64), 0.0, 1.0)
    index = np.asarray(ior, dtype=np.float64)
    f0 = np.square((index - 1.0) / (index + 1.0))
    return np.clip(f0 + (1.0 - f0) * np.power(1.0 - cosine, 5.0), 0.0, 1.0)


def _fallback_frame(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    axis = (
        np.asarray([0.0, 1.0, 0.0])
        if abs(float(normal[1])) < 0.9
        else np.asarray([1.0, 0.0, 0.0])
    )
    tangent = np.cross(axis, normal)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    bitangent /= np.linalg.norm(bitangent)
    return tangent, bitangent


def compute_tangent_frames(gbuffer: GBuffer) -> TangentFrame:
    height = gbuffer.height
    width = gbuffer.width
    tangent = np.zeros((height, width, 3), dtype=np.float64)
    bitangent = np.zeros((height, width, 3), dtype=np.float64)
    degenerate = np.zeros((height, width), dtype=bool)

    def screen_derivative(y: int, x: int, dy: int, dx: int):
        y2 = y + dy
        x2 = x + dx
        if not (0 <= y2 < height and 0 <= x2 < width):
            return None
        if not gbuffer.valid[y2, x2]:
            return None
        if gbuffer.object_id[y2, x2] != gbuffer.object_id[y, x]:
            return None
        delta_p = gbuffer.world[y2, x2] - gbuffer.world[y, x]
        delta_uv = gbuffer.uv[y2, x2] - gbuffer.uv[y, x]
        delta_uv -= np.rint(delta_uv)
        return delta_p, delta_uv

    rows, columns = np.nonzero(gbuffer.valid)
    for y, x in zip(rows.tolist(), columns.tolist()):
        normal = gbuffer.geometric_normal[y, x]
        dx_value = screen_derivative(y, x, 0, 1)
        if dx_value is None:
            dx_value = screen_derivative(y, x, 0, -1)
            if dx_value is not None:
                dx_value = (-dx_value[0], -dx_value[1])
        dy_value = screen_derivative(y, x, 1, 0)
        if dy_value is None:
            dy_value = screen_derivative(y, x, -1, 0)
            if dy_value is not None:
                dy_value = (-dy_value[0], -dy_value[1])

        valid_derivatives = dx_value is not None and dy_value is not None
        if valid_derivatives:
            delta_px, delta_uvx = dx_value
            delta_py, delta_uvy = dy_value
            du_dx, dv_dx = delta_uvx
            du_dy, dv_dy = delta_uvy
            determinant = du_dx * dv_dy - dv_dx * du_dy
            valid_derivatives = (
                np.isfinite(determinant) and abs(float(determinant)) > 1e-12
            )

        if valid_derivatives:
            dp_du = (dv_dy * delta_px - dv_dx * delta_py) / determinant
            dp_dv = (-du_dy * delta_px + du_dx * delta_py) / determinant
            tangent_u = dp_du - normal * float(np.dot(dp_du, normal))
            length_u = float(np.linalg.norm(tangent_u))
            length_v = float(np.linalg.norm(dp_dv))
            valid_derivatives = (
                np.isfinite(length_u)
                and np.isfinite(length_v)
                and length_u > 1e-12
                and length_v > 1e-12
            )

        if valid_derivatives:
            t = tangent_u / length_u
            b = np.cross(normal, t)
            b_length = float(np.linalg.norm(b))
            if not np.isfinite(b_length) or b_length <= 1e-12:
                valid_derivatives = False
            else:
                b /= b_length

        if not valid_derivatives:
            t, b = _fallback_frame(normal)
            degenerate[y, x] = True
        tangent[y, x] = t
        bitangent[y, x] = b

    return TangentFrame(
        tangent=np.ascontiguousarray(tangent),
        bitangent=np.ascontiguousarray(bitangent),
        degenerate=np.ascontiguousarray(degenerate),
    )


def _render_error(instruction: Instruction, code: str, message: str) -> PipelineError:
    return PipelineError(
        code=code,
        message=message,
        stage="render",
        node_index=instruction.node_index,
        node_id=instruction.node_id,
    )


def _as_array(operand: Operand, values: Dict[int, np.ndarray], count: int) -> np.ndarray:
    if operand.kind == "temp":
        dst = int(operand.name[1:])
        return values[dst]
    if operand.dim == 1:
        return np.full((count, 1), float(operand.value), dtype=np.float64)
    array = np.asarray(operand.value, dtype=np.float64)
    return np.broadcast_to(array[None, :], (count, operand.dim)).copy()


def _finite_checked(instruction: Instruction, value: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(value)):
        raise _render_error(
            instruction,
            "E_NUMERIC",
            "instruction produced a non-finite value",
        )
    return value


def evaluate_material(
    compiled: CompiledMaterial,
    gbuffer: GBuffer,
    camera_position: np.ndarray,
) -> MaterialBuffers:
    height = gbuffer.height
    width = gbuffer.width
    base_color = np.zeros((height, width, 3), dtype=np.float64)
    normal = np.zeros((height, width, 3), dtype=np.float64)
    normal[..., 2] = 1.0
    roughness = np.ones((height, width), dtype=np.float64)
    metallic = np.zeros((height, width), dtype=np.float64)
    emissive = np.zeros((height, width, 3), dtype=np.float64)
    normal_fallback = np.zeros((height, width), dtype=bool)

    valid_rows, valid_columns = np.nonzero(gbuffer.valid)
    count = int(valid_rows.size)
    has_normal_map = any(instruction.op == "normal_map" for instruction in compiled.instructions)
    if has_normal_map:
        frame = compute_tangent_frames(gbuffer)
    else:
        frame = TangentFrame(
            tangent=np.zeros((height, width, 3), dtype=np.float64),
            bitangent=np.zeros((height, width, 3), dtype=np.float64),
            degenerate=np.zeros((height, width), dtype=bool),
        )
    degenerate_tangent = np.ascontiguousarray(frame.degenerate.copy())

    if count == 0:
        return MaterialBuffers(
            base_color=base_color,
            normal=normal,
            roughness=roughness,
            metallic=metallic,
            emissive=emissive,
            degenerate_tangent=degenerate_tangent,
            normal_fallback=normal_fallback,
        )

    uv = gbuffer.uv[valid_rows, valid_columns]
    geometric_normal = gbuffer.geometric_normal[valid_rows, valid_columns]
    world = gbuffer.world[valid_rows, valid_columns]
    view = camera_position[None, :] - world
    view /= np.linalg.norm(view, axis=-1, keepdims=True)
    tangent = frame.tangent[valid_rows, valid_columns]
    bitangent = frame.bitangent[valid_rows, valid_columns]

    values: Dict[int, np.ndarray] = {}
    for instruction in compiled.instructions:
        operands = {
            key: _as_array(operand, values, count)
            for key, operand in instruction.operands
        }
        attrs = dict(instruction.attrs)
        if instruction.op == "uv":
            result = uv
        elif instruction.op == "texture_sample":
            texture_id = attrs["texture"]
            try:
                result = sample_texture(compiled.textures[texture_id], operands["uv"])
            except ValueError as exc:
                raise _render_error(instruction, "E_NUMERIC", str(exc))
        elif instruction.op == "add":
            result = operands["a"] + operands["b"]
        elif instruction.op == "multiply":
            result = operands["a"] * operands["b"]
        elif instruction.op == "mix":
            factor = operands["factor"]
            result = operands["a"] * (1.0 - factor) + operands["b"] * factor
        elif instruction.op == "fresnel":
            normal_value = operands["normal"]
            lengths = np.linalg.norm(normal_value, axis=-1)
            bad = (~np.isfinite(lengths)) | (lengths < 1e-12)
            normals = np.array(normal_value, copy=True)
            good = ~bad
            normals[good] /= lengths[good, None]
            normals[bad] = geometric_normal[bad]
            ior = operands["ior"][:, 0]
            if np.any(~np.isfinite(ior)) or np.any(ior < 1.0) or np.any(ior > 4.0):
                raise _render_error(
                    instruction,
                    "E_NUMERIC",
                    "fresnel.ior must be within 1..4",
                )
            result = fresnel_schlick(
                np.sum(normals * view, axis=1),
                ior,
            )[:, None]
        elif instruction.op == "normal_map":
            encoded = operands["normal"]
            strength = operands["strength"][:, 0]
            if np.any(~np.isfinite(strength)) or np.any(strength < 0.0) or np.any(strength > 10.0):
                raise _render_error(
                    instruction,
                    "E_NUMERIC",
                    "normal_map.strength must be within 0..10",
                )
            tangent_normal = encoded * 2.0 - 1.0
            tangent_normal[:, 0] *= strength
            tangent_normal[:, 1] *= strength
            lengths = np.linalg.norm(tangent_normal, axis=1)
            empty = (~np.isfinite(lengths)) | (lengths < 1e-12)
            tangent_normal[empty] = np.asarray([0.0, 0.0, 1.0])
            lengths[empty] = 1.0
            tangent_normal /= lengths[:, None]
            result = (
                tangent * tangent_normal[:, 0, None]
                + bitangent * tangent_normal[:, 1, None]
                + geometric_normal * tangent_normal[:, 2, None]
            )
            lengths = np.linalg.norm(result, axis=1)
            empty = (~np.isfinite(lengths)) | (lengths < 1e-12)
            result[empty] = geometric_normal[empty]
            lengths[empty] = 1.0
            result /= lengths[:, None]
        else:
            raise _render_error(
                instruction,
                "E_INTERNAL",
                f"unsupported instruction {instruction.op!r}",
            )
        values[instruction.dst] = _finite_checked(instruction, np.asarray(result, dtype=np.float64))

    base = _as_array(compiled.outputs["base_color"], values, count)
    normal_value = _as_array(compiled.outputs["normal"], values, count)
    roughness_value = _as_array(compiled.outputs["roughness"], values, count)[:, 0]
    metallic_value = _as_array(compiled.outputs["metallic"], values, count)[:, 0]
    emissive_value = _as_array(compiled.outputs["emissive"], values, count)

    normal_value, bad_normal = normalize_final_normals(normal_value, geometric_normal)
    normal_fallback[valid_rows, valid_columns] = bad_normal
    base = np.clip(base, 0.0, 1.0)
    roughness_value = np.clip(roughness_value, 0.0, 1.0)
    metallic_value = np.clip(metallic_value, 0.0, 1.0)
    emissive_value = np.clip(emissive_value, 0.0, 16.0)

    base_color[valid_rows, valid_columns] = base
    normal[valid_rows, valid_columns] = normal_value
    roughness[valid_rows, valid_columns] = roughness_value
    metallic[valid_rows, valid_columns] = metallic_value
    emissive[valid_rows, valid_columns] = emissive_value
    return MaterialBuffers(
        base_color=base_color,
        normal=normal,
        roughness=roughness,
        metallic=metallic,
        emissive=emissive,
        degenerate_tangent=degenerate_tangent,
        normal_fallback=normal_fallback,
    )


def normalize_final_normals(
    normal_value: np.ndarray,
    fallback: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(normal_value, axis=1)
    bad = (~np.isfinite(lengths)) | (lengths < 1e-12)
    result = np.array(normal_value, dtype=np.float64, copy=True)
    good = ~bad
    result[good] /= lengths[good, None]
    result[bad] = fallback[bad]
    fallback_lengths = np.linalg.norm(result[bad], axis=1)
    needs_second_fallback = bad.copy()
    if np.any(bad):
        secondary = (~np.isfinite(fallback_lengths)) | (fallback_lengths < 1e-12)
        if np.any(secondary):
            secondary_rows = np.nonzero(bad)[0][secondary]
            result[secondary_rows] = np.asarray([0.0, 0.0, 1.0])
        else:
            result[bad] /= fallback_lengths[:, None]
    return result, needs_second_fallback
