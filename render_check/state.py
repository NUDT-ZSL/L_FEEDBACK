"""Uniform reset, snapshots, and deterministic state verification."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import moderngl
import numpy as np

from .batch import InstanceSpec, Scene
from .errors import material_error, state_error
from .materials import Material
from .pipeline import active_uniforms


TEXTURE_UNIFORM = "u_albedo_tex"
MATRIX_GL_TYPES = frozenset({35674, 35675, 35676})
SAMPLER_GL_TYPES = frozenset({35678, 35679, 35680, 35681, 35682})


def _is_matrix(uniform: moderngl.Uniform) -> bool:
    return int(uniform.gl_type) in MATRIX_GL_TYPES


def _is_integer(uniform: moderngl.Uniform) -> bool:
    return int(uniform.gl_type) in SAMPLER_GL_TYPES or str(uniform.fmt).endswith("i")


def _expected_inputs(
    program: moderngl.Program,
    material: Material,
    instance: InstanceSpec,
    scene: Scene,
) -> dict[str, Any]:
    invariants: dict[str, Any] = {
        "u_model": instance.model,
        "u_view": scene.view,
        "u_proj": scene.projection,
        "u_camera_pos": scene.camera_position,
        "u_light_pos": scene.light_position,
        "u_light_color": scene.light_color,
        "u_object_id": float(instance.object_id),
        "u_material_index": float(instance.material_index),
        "u_shape_id": 0.0 if instance.shape == "cube" else 1.0,
        TEXTURE_UNIFORM: int(material.texture.unit),
    }
    expected: dict[str, Any] = {}
    for name in active_uniforms(program):
        if name in invariants:
            expected[name] = invariants[name]
        elif name in material.uniforms:
            expected[name] = material.uniforms[name]
        elif name.startswith("u_") and name[2:] in material.uniforms:
            expected[name] = material.uniforms[name[2:]]
        else:
            raise material_error(
                f"material '{material.name}': active uniform '{name}' has no default value",
                material=material.name,
                field=f"uniforms.{name}",
            )
    return expected


def _write_uniform(uniform: moderngl.Uniform, value: Any) -> None:
    if _is_integer(uniform):
        uniform.value = int(value)
        return
    array = np.asarray(value, dtype="<f4")
    order = "F" if _is_matrix(uniform) else "C"
    uniform.write(array.reshape(-1, order=order).tobytes())


def apply_expected_uniforms(
    program: moderngl.Program,
    material: Material,
    instance: InstanceSpec,
    scene: Scene,
) -> dict[str, Any]:
    expected = _expected_inputs(program, material, instance, scene)
    uniforms = active_uniforms(program)
    for name, value in expected.items():
        _write_uniform(uniforms[name], value)
    return expected


def _canonical_expected(uniform: moderngl.Uniform, value: Any) -> dict[str, Any]:
    if _is_integer(uniform):
        return {
            "kind": "int32",
            "gl_type": int(uniform.gl_type),
            "fmt": str(uniform.fmt),
            "dimension": int(uniform.dimension),
            "data": int(value),
        }
    array = np.asarray(value, dtype="<f4")
    order = "F" if _is_matrix(uniform) else "C"
    flat = array.reshape(-1, order=order)
    return {
        "kind": "float32",
        "gl_type": int(uniform.gl_type),
        "fmt": str(uniform.fmt),
        "dimension": int(uniform.dimension),
        "data": flat.tobytes().hex(),
    }


def _canonical_actual(uniform: moderngl.Uniform) -> dict[str, Any]:
    if _is_integer(uniform):
        return {
            "kind": "int32",
            "gl_type": int(uniform.gl_type),
            "fmt": str(uniform.fmt),
            "dimension": int(uniform.dimension),
            "data": int(uniform.value),
        }
    array = np.asarray(uniform.value, dtype="<f4").reshape(-1)
    return {
        "kind": "float32",
        "gl_type": int(uniform.gl_type),
        "fmt": str(uniform.fmt),
        "dimension": int(uniform.dimension),
        "data": array.tobytes().hex(),
    }


def make_snapshot(
    program: moderngl.Program,
    expected_inputs: dict[str, Any],
    instance: InstanceSpec,
    material: Material,
) -> dict[str, Any]:
    uniforms = active_uniforms(program)
    actual = {name: _canonical_actual(uniforms[name]) for name in sorted(uniforms)}
    expected = {
        name: _canonical_expected(uniforms[name], expected_inputs[name])
        for name in sorted(expected_inputs)
    }
    payload = {
        "instance_index": instance.index,
        "material": material.name,
        "uniforms": actual,
    }
    expected_payload = {
        "instance_index": instance.index,
        "material": material.name,
        "uniforms": expected,
    }
    actual_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    expected_hash = hashlib.sha256(
        json.dumps(expected_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        **payload,
        "state_hash": actual_hash,
        "expected_state_hash": expected_hash,
        "matches_expected": actual == expected,
        "_actual_uniforms": actual,
        "_expected_uniforms": expected,
    }


def verify_snapshot(record: dict[str, Any]) -> None:
    if record["matches_expected"]:
        return
    actual = record["_actual_uniforms"]
    expected = record["_expected_uniforms"]
    differing = sorted(
        name
        for name in set(actual).union(expected)
        if actual.get(name) != expected.get(name)
    )
    raise state_error(
        f"instance {record['instance_index']} material '{record['material']}': "
        "uniform state differs from the independently recomputed expected state "
        f"({', '.join(differing)})",
        instance_index=record["instance_index"],
        material=record["material"],
        expected_hash=record["expected_state_hash"],
        actual_hash=record["state_hash"],
        differing_uniforms=differing,
    )


def public_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}

