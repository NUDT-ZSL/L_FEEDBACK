"""Material graph parser, structural validation and type inference.

The parser deliberately accumulates independent errors instead of failing on
the first malformed node or edge. A graph is compiled only when the complete
error list is empty.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import StructuredError


SCHEMA_VERSION = 1
CHANNELS = ("base_color", "roughness", "metallic", "normal", "emissive")
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNELS)}
SEMANTIC_OUTPUT_TYPES = {
    "base_color": "vec3",
    "roughness": "float",
    "metallic": "float",
    "normal": "vec3",
    "emissive": "vec3",
}
TYPE_COMPONENTS = {"float": 1, "vec2": 2, "vec3": 3, "vec4": 4}
VECTOR_TYPES = {"vec2", "vec3", "vec4"}
BUILTIN_TEXTURES = ("checker", "noise", "normal_tile", "uv_grid")
GLSL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
KNOWN_NODE_TYPES = {
    "constant_float",
    "constant_vec2",
    "constant_vec3",
    "constant_vec4",
    "constant_color",
    "parameter",
    "uv",
    "world_position",
    "world_normal",
    "texture",
    "normal_map",
    "quantize",
    "combine",
    "swizzle",
    "channel_switch",
    "math",
}


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    glsl_type: str
    value: tuple[float, ...]
    raw_value: float | list[float]


@dataclass(frozen=True)
class Node:
    node_id: str
    node_type: str
    params: dict[str, Any]
    source_index: int


@dataclass(frozen=True)
class Edge:
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    edge_id: str


@dataclass
class NodeInfo:
    input_types: dict[str, str] = field(default_factory=dict)
    input_expected: dict[str, str] = field(default_factory=dict)
    output_types: dict[str, str] = field(default_factory=dict)
    required_inputs: set[str] = field(default_factory=set)
    op: str | None = None


@dataclass
class CameraSpec:
    position: tuple[float, float, float] = (0.0, 0.0, 3.0)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_degrees: float = 45.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fov_degrees": self.fov_degrees,
        }


@dataclass
class LightSpec:
    direction: tuple[float, float, float] = (-0.45, -0.65, -0.60)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": list(self.direction),
            "color": list(self.color),
            "intensity": self.intensity,
        }


@dataclass
class ParsedGraph:
    graph_id: str
    source_name: str
    document: dict[str, Any]
    parameters: dict[str, ParameterSpec] = field(default_factory=dict)
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    outputs: dict[str, tuple[str, str]] = field(default_factory=dict)
    node_info: dict[str, NodeInfo] = field(default_factory=dict)
    camera: CameraSpec = field(default_factory=CameraSpec)
    light: LightSpec = field(default_factory=LightSpec)
    errors: list[StructuredError] = field(default_factory=list)
    topo_order: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def add_error(
        self,
        code: str,
        message: str,
        *,
        node_id: str | None = None,
        channel: str | None = None,
        path: Iterable[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.errors.append(
            StructuredError(
                code=code,
                message=message,
                graph_id=self.graph_id or None,
                node_id=node_id,
                channel=channel,
                path=list(path or []),
                details=details or {},
            )
        )

    def parameter_uniform(self, name: str) -> str:
        return f"u_p_{name}"

    def texture_uniform(self, name: str) -> str:
        return f"u_tex_{name}"

    def output_path(self, channel: str) -> list[str]:
        binding = self.outputs.get(channel)
        if binding is None:
            return [f"graph.outputs.{channel}"]
        path: list[str] = [f"graph.outputs.{channel}", binding[0]]
        seen = {binding[0]}
        frontier = [binding[0]]
        incoming: dict[str, list[str]] = {}
        for edge in self.edges:
            incoming.setdefault(edge.to_node, []).append(edge.from_node)
        while frontier and len(path) < 32:
            current = frontier.pop(0)
            for source in incoming.get(current, []):
                if source not in seen:
                    seen.add(source)
                    path.append(source)
                    frontier.append(source)
        return path

    def to_summary_inputs(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "source": self.source_name,
            "camera": self.camera.to_dict(),
            "light": self.light.to_dict(),
            "parameters": {
                name: {"type": spec.glsl_type, "value": spec.raw_value}
                for name, spec in sorted(self.parameters.items())
            },
        }


def _error(
    code: str,
    message: str,
    *,
    graph_id: str | None = None,
    node_id: str | None = None,
    channel: str | None = None,
    path: Iterable[str] | None = None,
    details: dict[str, Any] | None = None,
) -> StructuredError:
    return StructuredError(
        code=code,
        message=message,
        graph_id=graph_id,
        node_id=node_id,
        channel=channel,
        path=list(path or []),
        details=details or {},
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted):
        return None
    return converted


def _triple(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    converted = tuple(_finite_number(item) for item in value)
    if any(item is None for item in converted):
        return None
    return converted  # type: ignore[return-value]


def _parse_parameter(
    name: str, raw: Any, graph_id: str
) -> tuple[ParameterSpec | None, StructuredError | None]:
    if not isinstance(name, str) or not GLSL_IDENTIFIER.match(name):
        return None, _error(
            "PARAMETER_BAD_NAME",
            f"Parameter name {name!r} is not a valid GLSL identifier.",
            graph_id=graph_id,
        )
    if isinstance(raw, dict):
        glsl_type = raw.get("type")
        value = raw.get("value")
    elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
        glsl_type, value = "float", raw
    elif isinstance(raw, (list, tuple)):
        glsl_type = {2: "vec2", 3: "vec3", 4: "vec4"}.get(len(raw))
        value = raw
    else:
        glsl_type, value = None, None
    if glsl_type not in TYPE_COMPONENTS:
        return None, _error(
            "PARAMETER_INVALID_TYPE",
            f"Parameter {name!r} has unsupported type {glsl_type!r}.",
            graph_id=graph_id,
            details={"parameter": name, "type": glsl_type},
        )
    components = TYPE_COMPONENTS[glsl_type]
    if components == 1:
        converted = _finite_number(value)
        if converted is None:
            return None, _error(
                "PARAMETER_INVALID_VALUE",
                f"Parameter {name!r} must contain one finite number.",
                graph_id=graph_id,
                details={"parameter": name, "value": value},
            )
        return ParameterSpec(name, glsl_type, (converted,), converted), None
    if not isinstance(value, (list, tuple)) or len(value) != components:
        return None, _error(
            "PARAMETER_INVALID_VALUE",
            f"Parameter {name!r} requires {components} finite numbers.",
            graph_id=graph_id,
            details={"parameter": name, "value": value},
        )
    converted_items = tuple(_finite_number(item) for item in value)
    if any(item is None for item in converted_items):
        return None, _error(
            "PARAMETER_INVALID_VALUE",
            f"Parameter {name!r} contains a non-finite or non-numeric value.",
            graph_id=graph_id,
            details={"parameter": name, "value": value},
        )
    plain_value = [float(item) for item in converted_items]  # type: ignore[arg-type]
    return (
        ParameterSpec(name, glsl_type, tuple(plain_value), plain_value),
        None,
    )


def _parse_ref(
    raw: Any, *, field_name: str, graph_id: str, owner: str
) -> tuple[tuple[str, str | None] | None, StructuredError | None]:
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            node_id, port = None, None
        elif "." in text:
            node_id, port = text.rsplit(".", 1)
        else:
            node_id, port = text, None
    elif isinstance(raw, dict):
        node_id = raw.get("node")
        port = raw.get("port")
    else:
        node_id, port = None, None
    if not isinstance(node_id, str) or not node_id:
        return None, _error(
            "REFERENCE_INVALID",
            f"{owner} field {field_name!r} must reference a node.",
            graph_id=graph_id,
            details={"owner": owner, "field": field_name, "value": raw},
        )
    if port is not None and (not isinstance(port, str) or not port):
        return None, _error(
            "REFERENCE_INVALID",
            f"{owner} field {field_name!r} has an invalid port.",
            graph_id=graph_id,
            details={"owner": owner, "field": field_name, "value": raw},
        )
    return (node_id, port), None


def _resolve_ref_port(
    graph: ParsedGraph, node_id: str, port: str | None, *, use_outputs: bool
) -> str | None:
    node = graph.nodes.get(node_id)
    if node is None:
        return None
    ports = (
        graph.node_info.get(node_id, NodeInfo()).output_types
        if use_outputs
        else graph.node_info.get(node_id, NodeInfo()).input_expected
    )
    if port:
        return port
    if "out" in ports:
        return "out"
    if len(ports) == 1:
        return next(iter(ports))
    return None


def _static_node_outputs(
    node: Node,
    parameters: dict[str, ParameterSpec],
    graph_id: str,
    errors: list[StructuredError],
) -> dict[str, str]:
    kind = node.node_type
    params = node.params
    if kind == "constant_float":
        return {"out": "float"}
    if kind in {"constant_vec2", "constant_vec3", "constant_vec4"}:
        return {"out": kind.removeprefix("constant_")}
    if kind == "constant_color":
        return {"out": "vec3"}
    if kind == "parameter":
        name = params.get("name")
        spec = parameters.get(name) if isinstance(name, str) else None
        if spec is None:
            errors.append(
                _error(
                    "PARAMETER_MISSING",
                    f"Node {node.node_id!r} references unknown parameter {name!r}.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"parameter": name},
                )
            )
            return {}
        return {"out": spec.glsl_type}
    if kind == "uv":
        return {"out": "vec2"}
    if kind in {"world_position", "world_normal"}:
        return {"out": "vec3"}
    if kind == "texture":
        name = params.get("texture")
        if name not in BUILTIN_TEXTURES:
            errors.append(
                _error(
                    "TEXTURE_UNKNOWN",
                    f"Node {node.node_id!r} references unknown built-in texture {name!r}.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"texture": name, "available": list(BUILTIN_TEXTURES)},
                )
            )
        return {"rgba": "vec4", "color": "vec3", "alpha": "float"}
    if kind == "normal_map":
        return {"out": "vec3"}
    if kind == "quantize":
        return {"out": "float"}
    if kind == "combine":
        size = params.get("size")
        if size not in (2, 3, 4):
            errors.append(
                _error(
                    "NODE_PARAM_INVALID",
                    f"combine node {node.node_id!r} requires size 2, 3 or 4.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"size": size},
                )
            )
            return {"out": "vec3"}
        return {"out": f"vec{size}"}
    if kind == "swizzle":
        components = params.get("components")
        if not isinstance(components, str) or not 1 <= len(components) <= 4:
            errors.append(
                _error(
                    "SWIZZLE_INVALID",
                    f"swizzle node {node.node_id!r} needs 1 to 4 components.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"components": components},
                )
            )
            return {"out": "float"}
        output_type = "float" if len(components) == 1 else f"vec{len(components)}"
        return {"out": output_type}
    if kind == "channel_switch":
        value_type = params.get("type", "float")
        if value_type not in TYPE_COMPONENTS:
            errors.append(
                _error(
                    "NODE_PARAM_INVALID",
                    f"channel_switch node {node.node_id!r} has invalid type {value_type!r}.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"type": value_type},
                )
            )
            return {"out": "float"}
        return {"out": value_type}
    return {}


def _math_ops() -> set[str]:
    return {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "min",
        "max",
        "dot",
        "cross",
        "normalize",
        "length",
        "abs",
        "sin",
        "cos",
        "tan",
        "fract",
        "floor",
        "ceil",
        "round",
        "sqrt",
        "exp",
        "log",
        "negate",
        "clamp",
        "mix",
        "smoothstep",
        "step",
    }


def _math_input_ports(op: str) -> list[str]:
    if op in {"add", "sub", "mul", "div", "pow", "min", "max", "dot", "cross"}:
        return ["a", "b"]
    if op in {"clamp"}:
        return ["a", "b", "c"]
    if op in {"mix"}:
        return ["a", "b", "c"]
    if op in {"smoothstep", "step"}:
        return ["a", "b", "c"] if op == "smoothstep" else ["a", "b"]
    return ["a"]


def _resolve_node(
    node: Node,
    input_types: dict[str, str],
    parameters: dict[str, ParameterSpec],
    graph_id: str,
    errors: list[StructuredError],
) -> NodeInfo:
    kind = node.node_type
    params = node.params
    info = NodeInfo(
        input_types=dict(input_types),
        output_types=_static_node_outputs(node, parameters, graph_id, errors),
    )
    if kind in {
        "constant_float",
        "constant_vec2",
        "constant_vec3",
        "constant_vec4",
        "constant_color",
    }:
        if not _validate_constant_node(node, graph_id, errors):
            info.output_types = {}
    elif kind == "texture":
        info.input_expected = {"uv": "vec2"}
        info.required_inputs = {"uv"}
    elif kind == "normal_map":
        info.input_expected = {"normal": "vec3", "strength": "float"}
        info.required_inputs = {"normal"}
    elif kind == "quantize":
        info.input_expected = {"value": "float"}
        info.required_inputs = {"value"}
        steps = _finite_number(params.get("steps", 255.0))
        if steps is None or steps <= 0:
            errors.append(
                _error(
                    "NODE_PARAM_INVALID",
                    f"quantize node {node.node_id!r} needs a positive finite steps value.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"steps": params.get("steps")},
                )
            )
    elif kind == "combine":
        size = params.get("size")
        ports = ["x", "y", "z", "w"][: size if size in (2, 3, 4) else 3]
        info.input_expected = {port: "float" for port in ports}
        info.required_inputs = set(ports)
    elif kind == "swizzle":
        info.input_expected = {"in": "vec2|vec3|vec4"}
        info.required_inputs = {"in"}
        components = params.get("components")
        size = len(components) if isinstance(components, str) else 1
        info.output_types = {
            "out": "float" if size == 1 else f"vec{min(size, 4)}"
        }
        if isinstance(components, str):
            if any(char not in "xyzw" for char in components):
                errors.append(
                    _error(
                        "SWIZZLE_INVALID",
                        f"swizzle node {node.node_id!r} may only use x/y/z/w components.",
                        graph_id=graph_id,
                        node_id=node.node_id,
                        details={"components": components},
                    )
                )
            else:
                source_type = input_types.get("in")
                if source_type in VECTOR_TYPES:
                    source_size = TYPE_COMPONENTS[source_type]
                    required = max("xyzw".index(char) for char in components) + 1
                    if required > source_size:
                        errors.append(
                            _error(
                                "SWIZZLE_INVALID",
                                f"swizzle node {node.node_id!r} selects {components!r} "
                                f"from {source_type}.",
                                graph_id=graph_id,
                                node_id=node.node_id,
                                details={
                                    "components": components,
                                    "source_type": source_type,
                                },
                            )
                        )
    elif kind == "channel_switch":
        value_type = info.output_types.get("out", "float")
        if not _validate_channel_switch(node, value_type, graph_id, errors):
            info.output_types = {}
    elif kind == "math":
        op = params.get("op")
        if op not in _math_ops():
            errors.append(
                _error(
                    "NODE_PARAM_INVALID",
                    f"math node {node.node_id!r} has unknown op {op!r}.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"op": op, "supported": sorted(_math_ops())},
                )
            )
            info.output_types = {}
            return info
        info.op = op
        ports = _math_input_ports(op)
        info.required_inputs = set(ports)
        info.output_types = _infer_math_output(node, op, input_types, graph_id, errors)
        if op in {"add", "sub", "mul", "div", "pow", "min", "max"}:
            info.input_expected = {
                port: "float or matching vector" for port in ("a", "b")
            }
        elif op in {"dot"}:
            info.input_expected = {"a": "vec2|vec3|vec4", "b": "same as a"}
        elif op == "cross":
            info.input_expected = {"a": "vec3", "b": "vec3"}
        elif op in {"clamp"}:
            info.input_expected = {
                "a": "numeric",
                "b": "float or same as a",
                "c": "float or same as a",
            }
        elif op == "mix":
            info.input_expected = {
                "a": "numeric",
                "b": "float or same as a",
                "c": "float or same as a",
            }
        elif op in {"smoothstep", "step"}:
            info.input_expected = {
                "a": "float",
                "b": "float",
                "c": "float or matching vector",
            }
            if op == "step":
                info.input_expected = {
                    "a": "float",
                    "b": "float or matching vector",
                }
        else:
            info.input_expected = {"a": "numeric"}
    elif kind in {
        "constant_float",
        "constant_vec2",
        "constant_vec3",
        "constant_vec4",
        "constant_color",
        "parameter",
        "uv",
        "world_position",
        "world_normal",
    }:
        info.input_expected = {}
    return info


def _validate_channel_switch(
    node: Node,
    value_type: str,
    graph_id: str,
    errors: list[StructuredError],
) -> bool:
    params = node.params
    values = params.get("values", {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        errors.append(
            _error(
                "NODE_PARAM_INVALID",
                f"channel_switch node {node.node_id!r} values must be an object.",
                graph_id=graph_id,
                node_id=node.node_id,
            )
        )
        return False
    components = TYPE_COMPONENTS[value_type]
    for channel, value in values.items():
        if channel not in CHANNELS:
            errors.append(
                _error(
                    "NODE_PARAM_INVALID",
                    f"channel_switch node {node.node_id!r} has unknown channel {channel!r}.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                    details={"channel": channel},
                )
            )
            return False
        if components == 1:
            if _finite_number(value) is None:
                errors.append(
                    _error(
                        "NODE_PARAM_INVALID",
                        f"channel_switch node {node.node_id!r} value for {channel!r} is invalid.",
                        graph_id=graph_id,
                        node_id=node.node_id,
                    )
                )
                return False
        elif not isinstance(value, (list, tuple)) or len(value) != components:
            errors.append(
                _error(
                    "NODE_PARAM_INVALID",
                    f"channel_switch node {node.node_id!r} value for {channel!r} "
                    f"requires {components} numbers.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                )
            )
            return False
        elif any(_finite_number(item) is None for item in value):
            errors.append(
                _error(
                    "NODE_PARAM_INVALID",
                    f"channel_switch node {node.node_id!r} value for {channel!r} is non-finite.",
                    graph_id=graph_id,
                    node_id=node.node_id,
                )
            )
            return False
    default = params.get("default", 0.0)
    if components == 1:
        valid = _finite_number(default) is not None
    else:
        valid = (
        isinstance(default, (list, tuple))
        and len(default) == components
        and all(_finite_number(item) is not None for item in default)
        )
    if not valid:
        errors.append(
            _error(
                "NODE_PARAM_INVALID",
                f"channel_switch node {node.node_id!r} has an invalid default value.",
                graph_id=graph_id,
                node_id=node.node_id,
                details={"default": default, "type": value_type},
            )
        )
    return valid


def _validate_constant_node(
    node: Node, graph_id: str, errors: list[StructuredError]
) -> bool:
    kind = node.node_type
    value = node.params.get("value")
    if kind == "constant_color" and value is None:
        value = node.params.get("color")
    if kind == "constant_float":
        valid = _finite_number(value) is not None
        expected = 1
    else:
        glsl_type = (
            "vec3" if kind == "constant_color" else kind.removeprefix("constant_")
        )
        expected = TYPE_COMPONENTS[glsl_type]
        valid = (
            isinstance(value, (list, tuple))
            and len(value) == expected
            and all(_finite_number(item) is not None for item in value)
        )
    if not valid:
        errors.append(
            _error(
                "NODE_PARAM_INVALID",
                f"{kind} node {node.node_id!r} requires {expected} finite value(s).",
                graph_id=graph_id,
                node_id=node.node_id,
                details={"value": value},
            )
        )
    return valid


def _infer_math_output(
    node: Node,
    op: str,
    input_types: dict[str, str],
    graph_id: str,
    errors: list[StructuredError],
) -> dict[str, str]:
    def bad_inputs(message: str) -> dict[str, str]:
        errors.append(
            _error(
                "NODE_INPUT_TYPE_MISMATCH",
                f"math node {node.node_id!r} ({op}): {message}",
                graph_id=graph_id,
                node_id=node.node_id,
                details={"op": op, "input_types": dict(input_types)},
            )
        )
        return {}

    if op in {"add", "sub", "mul", "div", "pow", "min", "max"}:
        a, b = input_types.get("a"), input_types.get("b")
        if a is None or b is None:
            return {}
        if a not in TYPE_COMPONENTS or b not in TYPE_COMPONENTS:
            return bad_inputs("both inputs must be numeric.")
        if a == "float":
            return {"out": b}
        if b == "float":
            return {"out": a}
        if a != b:
            return bad_inputs(f"cannot combine {a} with {b}.")
        return {"out": a}
    if op == "dot":
        a, b = input_types.get("a"), input_types.get("b")
        if a not in VECTOR_TYPES or b != a:
            return bad_inputs("dot requires two inputs of the same vector type.")
        return {"out": "float"}
    if op == "cross":
        a, b = input_types.get("a"), input_types.get("b")
        if a != "vec3" or b != "vec3":
            return bad_inputs("cross requires two vec3 inputs.")
        return {"out": "vec3"}
    if op in {
        "abs",
        "sin",
        "cos",
        "tan",
        "fract",
        "floor",
        "ceil",
        "round",
        "sqrt",
        "exp",
        "log",
        "negate",
        "normalize",
    }:
        a = input_types.get("a")
        if a not in TYPE_COMPONENTS:
            return {}
        if op == "normalize" and a == "float":
            return bad_inputs("normalize requires a vector input.")
        return {"out": a}
    if op == "length":
        a = input_types.get("a")
        if a not in VECTOR_TYPES:
            return bad_inputs("length requires a vector input.")
        return {"out": "float"}
    if op == "clamp":
        a, b, c = input_types.get("a"), input_types.get("b"), input_types.get("c")
        if a is None:
            return {}
        if b is not None and b not in {"float", a}:
            return bad_inputs(f"clamp lower bound {b} is not compatible with {a}.")
        if c is not None and c not in {"float", a}:
            return bad_inputs(f"clamp upper bound {c} is not compatible with {a}.")
        return {"out": a}
    if op == "mix":
        a, b, c = input_types.get("a"), input_types.get("b"), input_types.get("c")
        if a is None or b is None:
            return {}
        if a == "float":
            result = b
        elif b == "float":
            result = a
        elif a == b:
            result = a
        else:
            return bad_inputs(f"mix operands {a} and {b} are incompatible.")
        if c is not None and c not in {"float", result}:
            return bad_inputs(f"mix factor {c} is incompatible with {result}.")
        return {"out": result}
    if op == "smoothstep":
        a, b, c = input_types.get("a"), input_types.get("b"), input_types.get("c")
        if a not in {None, "float"} or b not in {None, "float"}:
            return bad_inputs("smoothstep edges must be float.")
        if c is None:
            return {}
        if c not in {"float", "vec2", "vec3", "vec4"}:
            return bad_inputs("smoothstep x must be numeric.")
        if c != "float" and (a is not None or b is not None):
            return bad_inputs("smoothstep vector x requires vector edges in this tool.")
        return {"out": c}
    if op == "step":
        a, b = input_types.get("a"), input_types.get("b")
        if a not in {None, "float"}:
            return bad_inputs("step edge must be float.")
        return {"out": b} if b else {}
    return {}


def _find_cycles(
    node_ids: Iterable[str], edges: list[Edge], graph_id: str
) -> list[list[str]]:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        if edge.from_node in adjacency and edge.to_node in adjacency:
            adjacency[edge.from_node].append(edge.to_node)
    state: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[list[str]] = []
    signatures: set[tuple[str, ...]] = set()

    def visit(node_id: str) -> None:
        state[node_id] = 1
        stack.append(node_id)
        for target in adjacency[node_id]:
            if state.get(target, 0) == 0:
                visit(target)
            elif state.get(target) == 1:
                start = stack.index(target)
                cycle = stack[start:] + [target]
                signature = tuple(sorted(set(cycle)))
                if signature not in signatures:
                    signatures.add(signature)
                    cycles.append(cycle)
        stack.pop()
        state[node_id] = 2

    for node_id in sorted(adjacency):
        if state.get(node_id, 0) == 0:
            visit(node_id)
    return cycles


def _kahn_order(nodes: dict[str, Node], edges: list[Edge]) -> list[str]:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    indegree: dict[str, int] = {node_id: 0 for node_id in nodes}
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.from_node not in nodes or edge.to_node not in nodes:
            continue
        key = (edge.from_node, edge.to_node)
        if key in seen:
            continue
        seen.add(key)
        adjacency[edge.from_node].append(edge.to_node)
        indegree[edge.to_node] += 1
    ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    return order


def parse_graph_document(
    document: Any, *, source_name: str = "<memory>"
) -> ParsedGraph:
    graph_id = ""
    if isinstance(document, dict):
        raw_id = document.get("id", document.get("material_id", ""))
        if isinstance(raw_id, str):
            graph_id = raw_id
    graph = ParsedGraph(
        graph_id=graph_id,
        source_name=source_name,
        document=copy.deepcopy(document) if isinstance(document, dict) else {},
    )
    if not isinstance(document, dict):
        graph.add_error(
            "GRAPH_ROOT_NOT_OBJECT", "Material graph root must be a JSON object."
        )
        return graph
    if not graph_id:
        graph.add_error("GRAPH_ID_MISSING", "Material graph requires a non-empty string id.")
    version = document.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        graph.add_error(
            "GRAPH_SCHEMA_UNSUPPORTED",
            f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION}.",
            details={"actual": version, "expected": SCHEMA_VERSION},
        )

    raw_parameters = document.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        graph.add_error("PARAMETERS_NOT_OBJECT", "parameters must be an object.")
        raw_parameters = {}
    for name, raw in raw_parameters.items():
        spec, error = _parse_parameter(name, raw, graph_id)
        if error is not None:
            graph.errors.append(error)
        elif spec is not None:
            graph.parameters[name] = spec

    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list):
        graph.add_error("NODES_MISSING", "Material graph requires a nodes array.")
        raw_nodes = []
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            graph.add_error(
                "NODE_NOT_OBJECT", f"nodes[{index}] must be an object.",
                details={"index": index},
            )
            continue
        node_id = raw_node.get("id")
        node_type = raw_node.get("type")
        if not isinstance(node_id, str) or not node_id:
            graph.add_error(
                "NODE_ID_MISSING", f"nodes[{index}] requires a non-empty string id.",
                details={"index": index},
            )
            continue
        if node_id in graph.nodes:
            graph.add_error(
                "NODE_ID_DUPLICATE", f"Duplicate node id {node_id!r}.", node_id=node_id
            )
            continue
        if not isinstance(node_type, str):
            graph.add_error(
                "NODE_TYPE_MISSING", f"Node {node_id!r} requires a string type.",
                node_id=node_id,
            )
            continue
        if node_type not in KNOWN_NODE_TYPES:
            graph.add_error(
                "NODE_TYPE_UNKNOWN",
                f"Node {node_id!r} has unknown type {node_type!r}.",
                node_id=node_id,
                details={"type": node_type, "supported": sorted(KNOWN_NODE_TYPES)},
            )
        params = raw_node.get("params", {})
        if not isinstance(params, dict):
            graph.add_error(
                "NODE_PARAM_INVALID", f"Node {node_id!r} params must be an object.",
                node_id=node_id,
            )
            params = {}
        graph.nodes[node_id] = Node(node_id, node_type, params, index)

    raw_edges = document.get("edges", [])
    if not isinstance(raw_edges, list):
        graph.add_error("EDGES_NOT_ARRAY", "edges must be an array.")
        raw_edges = []
    for index, raw_edge in enumerate(raw_edges):
        if not isinstance(raw_edge, dict):
            graph.add_error(
                "EDGE_INVALID", f"edges[{index}] must be an object.",
                details={"edge_index": index},
            )
            continue
        from_ref, from_error = _parse_ref(
            raw_edge.get("from"),
            field_name="from",
            graph_id=graph_id,
            owner=f"edges[{index}]",
        )
        to_ref, to_error = _parse_ref(
            raw_edge.get("to"),
            field_name="to",
            graph_id=graph_id,
            owner=f"edges[{index}]",
        )
        if from_error is not None:
            graph.errors.append(from_error)
        if to_error is not None:
            graph.errors.append(to_error)
        if from_ref is None or to_ref is None:
            continue
        from_node, from_port = from_ref
        to_node, to_port = to_ref
        assert from_node is not None and to_node is not None
        missing = False
        if from_node not in graph.nodes:
            graph.add_error(
                "EDGE_NODE_MISSING",
                f"edges[{index}].from references missing node {from_node!r}.",
                details={"edge_index": index, "missing_node": from_node},
            )
            missing = True
        if to_node not in graph.nodes:
            graph.add_error(
                "EDGE_NODE_MISSING",
                f"edges[{index}].to references missing node {to_node!r}.",
                details={"edge_index": index, "missing_node": to_node},
            )
            missing = True
        if missing:
            continue
        edge = Edge(
            from_node=from_node,
            from_port=from_port or "",
            to_node=to_node,
            to_port=to_port or "",
            edge_id=f"edge[{index}]",
        )
        graph.edges.append(edge)

    cycles = _find_cycles(graph.nodes.keys(), graph.edges, graph_id)
    for cycle in cycles:
        graph.add_error(
            "GRAPH_CYCLE",
            "Cycle detected: " + " -> ".join(cycle),
            node_id=cycle[0] if cycle else None,
            path=cycle,
            details={"cycle": cycle},
        )

    graph.topo_order = _kahn_order(graph.nodes, graph.edges)
    incoming: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.to_node, []).append(edge)

    for node_id in graph.topo_order:
        node = graph.nodes[node_id]
        input_types: dict[str, str] = {}
        for edge in incoming.get(node_id, []):
            source_info = graph.node_info.get(edge.from_node)
            if source_info is None:
                continue
            source_port = edge.from_port or _resolve_ref_port(
                graph, edge.from_node, None, use_outputs=True
            )
            if source_port and source_port in source_info.output_types:
                input_types[edge.to_port or _default_input_port(node)] = source_info.output_types[
                    source_port
                ]
        info = _resolve_node(
            node, input_types, graph.parameters, graph_id, graph.errors
        )
        graph.node_info[node_id] = info

    for node_id, node in graph.nodes.items():
        info = graph.node_info.setdefault(node_id, NodeInfo())
        for edge in incoming.get(node_id, []):
            source_info = graph.node_info.get(edge.from_node)
            if source_info is None:
                continue
            from_port = edge.from_port or _resolve_ref_port(
                graph, edge.from_node, None, use_outputs=True
            )
            if not from_port:
                graph.add_error(
                    "EDGE_PORT_AMBIGUOUS",
                    f"{edge.edge_id} source {edge.from_node!r} needs an explicit output port.",
                    node_id=edge.from_node,
                    details={"edge": edge.edge_id},
                )
            elif from_port not in source_info.output_types:
                graph.add_error(
                    "EDGE_PORT_UNKNOWN",
                    f"{edge.edge_id} source port {edge.from_node}.{from_port} does not exist.",
                    node_id=edge.from_node,
                    details={
                        "edge": edge.edge_id,
                        "available_ports": sorted(source_info.output_types),
                    },
                )
            to_port = edge.to_port or _default_input_port(node)
            if not to_port:
                graph.add_error(
                    "EDGE_PORT_AMBIGUOUS",
                    f"{edge.edge_id} target {edge.to_node!r} needs an explicit input port.",
                    node_id=edge.to_node,
                    details={"edge": edge.edge_id},
                )
            elif to_port not in info.input_expected:
                graph.add_error(
                    "EDGE_PORT_UNKNOWN",
                    f"{edge.edge_id} target port {edge.to_node}.{to_port} does not exist.",
                    node_id=edge.to_node,
                    details={
                        "edge": edge.edge_id,
                        "available_ports": sorted(info.input_expected),
                    },
                )
            elif from_port in source_info.output_types:
                source_type = source_info.output_types[from_port]
                if not _input_type_allowed(node, to_port, source_type, info):
                    graph.add_error(
                        "EDGE_TYPE_MISMATCH",
                        f"{edge.edge_id} connects {edge.from_node}.{from_port} "
                        f"({source_type}) to {edge.to_node}.{to_port} "
                        f"(expected {info.input_expected.get(to_port, '?')}).",
                        node_id=edge.to_node,
                        details={
                            "edge": edge.edge_id,
                            "source_type": source_type,
                            "expected": info.input_expected.get(to_port),
                            "source": f"{edge.from_node}.{from_port}",
                            "target": f"{edge.to_node}.{to_port}",
                        },
                    )
        for required in sorted(info.required_inputs):
            if required not in info.input_types:
                graph.add_error(
                    "NODE_INPUT_MISSING",
                    f"Node {node_id!r} is missing required input {required!r}.",
                    node_id=node_id,
                    details={"input": required},
                )

    input_targets: dict[tuple[str, str], str] = {}
    for edge in graph.edges:
        to_port = edge.to_port or _default_input_port(graph.nodes[edge.to_node])
        key = (edge.to_node, to_port)
        if key in input_targets:
            graph.add_error(
                "EDGE_DUPLICATE_INPUT",
                f"Input {edge.to_node}.{to_port} is connected more than once.",
                node_id=edge.to_node,
                details={"edge": edge.edge_id, "previous_edge": input_targets[key]},
            )
        else:
            input_targets[key] = edge.edge_id

    raw_outputs = document.get("outputs", {})
    if not isinstance(raw_outputs, dict):
        graph.add_error("OUTPUTS_NOT_OBJECT", "outputs must be an object.")
        raw_outputs = {}
    for channel, expected_type in SEMANTIC_OUTPUT_TYPES.items():
        if channel not in raw_outputs:
            graph.add_error(
                "OUTPUT_MISSING",
                f"Required semantic output {channel!r} is not bound.",
                channel=channel,
            )
            continue
        ref, error = _parse_ref(
            raw_outputs[channel],
            field_name=channel,
            graph_id=graph_id,
            owner="outputs",
        )
        if error is not None:
            graph.errors.append(error)
            continue
        assert ref is not None
        node_id, port = ref
        node = graph.nodes.get(node_id) if node_id is not None else None
        if node is None:
            graph.add_error(
                "OUTPUT_NODE_MISSING",
                f"Output {channel!r} references missing node {node_id!r}.",
                channel=channel,
                details={"node": node_id},
            )
            continue
        info = graph.node_info.get(node_id, NodeInfo())
        resolved_port = port or _resolve_ref_port(graph, node_id, None, use_outputs=True)
        if not resolved_port:
            graph.add_error(
                "OUTPUT_PORT_AMBIGUOUS",
                f"Output {channel!r} from node {node_id!r} needs an explicit port.",
                node_id=node_id,
                channel=channel,
            )
            continue
        if resolved_port not in info.output_types:
            graph.add_error(
                "OUTPUT_PORT_UNKNOWN",
                f"Output {channel!r} references missing port {node_id}.{resolved_port}.",
                node_id=node_id,
                channel=channel,
                details={"available_ports": sorted(info.output_types)},
            )
            continue
        actual_type = info.output_types[resolved_port]
        if actual_type != expected_type:
            graph.add_error(
                "OUTPUT_TYPE_MISMATCH",
                f"Output {channel!r} expects {expected_type}, got {actual_type} "
                f"from {node_id}.{resolved_port}.",
                node_id=node_id,
                channel=channel,
                details={
                    "expected_type": expected_type,
                    "actual_type": actual_type,
                    "output": f"{node_id}.{resolved_port}",
                },
            )
            continue
        graph.outputs[channel] = (node_id, resolved_port)

    graph.camera = _parse_camera(document.get("camera"), graph)
    graph.light = _parse_light(document.get("light"), graph)
    return graph


def _default_input_port(node: Node) -> str:
    if node.node_type == "texture":
        return "uv"
    if node.node_type == "normal_map":
        return "normal"
    if node.node_type == "quantize":
        return "value"
    if node.node_type == "swizzle":
        return "in"
    if node.node_type == "math":
        return "a"
    if node.node_type == "combine":
        return "x"
    return ""


def _input_type_allowed(
    node: Node, port: str, source_type: str, info: NodeInfo
) -> bool:
    expected = info.input_expected.get(port)
    if expected is None:
        return False
    if expected in TYPE_COMPONENTS:
        return source_type == expected
    if expected == "vec2|vec3|vec4":
        return source_type in VECTOR_TYPES
    if expected == "numeric":
        return source_type in TYPE_COMPONENTS
    if node.node_type == "math":
        op = node.params.get("op")
        input_types = info.input_types
        if op in {"add", "sub", "mul", "div", "pow", "min", "max"}:
            other_port = "b" if port == "a" else "a"
            other = input_types.get(other_port)
            if other is None or source_type == "float" or other == "float":
                return source_type in TYPE_COMPONENTS
            return source_type == other
        if op == "dot":
            a, b = input_types.get("a"), input_types.get("b")
            if port == "b":
                return source_type in VECTOR_TYPES and (a is None or source_type == a)
            return source_type in VECTOR_TYPES and (b is None or source_type == b)
        if op in {"clamp", "mix"}:
            a = input_types.get("a")
            if port in {"b", "c"} and a is not None:
                return source_type in {"float", a}
            return source_type in TYPE_COMPONENTS
        if op == "smoothstep":
            if port in {"a", "b"}:
                return source_type == "float"
            return source_type in TYPE_COMPONENTS
        if op == "step":
            if port == "a":
                return source_type == "float"
            return source_type in TYPE_COMPONENTS
        return source_type in TYPE_COMPONENTS
    return False


def _parse_camera(raw: Any, graph: ParsedGraph) -> CameraSpec:
    camera = CameraSpec()
    if raw is None:
        return camera
    if not isinstance(raw, dict):
        graph.add_error("CAMERA_INVALID", "camera must be an object.")
        return camera
    position = _triple(raw.get("position", camera.position))
    target = _triple(raw.get("target", camera.target))
    up = _triple(raw.get("up", camera.up))
    fov = _finite_number(raw.get("fov_degrees", camera.fov_degrees))
    if position is None or target is None or up is None:
        graph.add_error(
            "CAMERA_INVALID", "camera position, target and up must be vec3 values."
        )
        return camera
    if fov is None or not 1.0 <= fov <= 179.0:
        graph.add_error(
            "CAMERA_INVALID", "camera fov_degrees must be finite and between 1 and 179."
        )
        return camera
    if position == target:
        graph.add_error("CAMERA_INVALID", "camera position and target must differ.")
        return camera
    forward = tuple(target[index] - position[index] for index in range(3))
    cross_up = (
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    )
    if math.sqrt(sum(item * item for item in cross_up)) < 1e-8:
        graph.add_error("CAMERA_INVALID", "camera up must not be parallel to view direction.")
        return camera
    return CameraSpec(position, target, up, fov)


def _parse_light(raw: Any, graph: ParsedGraph) -> LightSpec:
    light = LightSpec()
    if raw is None:
        return light
    if not isinstance(raw, dict):
        graph.add_error("LIGHT_INVALID", "light must be an object.")
        return light
    direction = _triple(raw.get("direction", light.direction))
    color = _triple(raw.get("color", light.color))
    intensity = _finite_number(raw.get("intensity", light.intensity))
    if direction is None or math.sqrt(sum(item * item for item in direction)) < 1e-8:
        graph.add_error("LIGHT_INVALID", "light direction must be a non-zero vec3.")
        return light
    if color is None or any(item < 0.0 for item in color):
        graph.add_error("LIGHT_INVALID", "light color must be three non-negative numbers.")
        return light
    if intensity is None or intensity < 0.0:
        graph.add_error("LIGHT_INVALID", "light intensity must be non-negative.")
        return light
    return LightSpec(direction, color, intensity)


def load_graph_file(path: Path) -> ParsedGraph:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        graph = ParsedGraph(
            graph_id=path.stem,
            source_name=str(path),
            document={},
        )
        graph.add_error(
            "GRAPH_JSON_INVALID",
            f"Failed to parse JSON: {exc}",
            details={"exception": type(exc).__name__},
        )
        return graph
    return parse_graph_document(document, source_name=str(path))


def remap_channels(values: dict[str, Any]) -> list[Any]:
    return [values[channel] for channel in CHANNELS]

