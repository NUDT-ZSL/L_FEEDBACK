"""GLSL generation for validated material graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .graph import (
    CHANNEL_INDEX,
    CHANNELS,
    TYPE_COMPONENTS,
    Node,
    ParsedGraph,
)


VERTEX_SHADER = """#version 330 core
out vec2 v_uv;

void main() {
    vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 position = positions[gl_VertexID];
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


@dataclass(frozen=True)
class ShaderSources:
    vertex: str
    fragment: str


@dataclass(frozen=True)
class UniformLayout:
    offsets: dict[str, int]
    size: int
    glsl_types: dict[str, str]


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "0.0"
    text = f"{float(value):.9g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}e{int(exponent)}"
    if "." not in text:
        text += ".0"
    return text


def _glsl_value(value: float | list[float] | tuple[float, ...], glsl_type: str) -> str:
    if glsl_type == "float":
        return _fmt(float(value))  # type: ignore[arg-type]
    items = list(value)  # type: ignore[arg-type]
    return f"{glsl_type}(" + ", ".join(_fmt(float(item)) for item in items) + ")"


def material_uniform_layout(graph: ParsedGraph) -> UniformLayout:
    offsets: dict[str, int] = {}
    types: dict[str, str] = {}
    offset = 0
    for index, name in enumerate(sorted(graph.parameters)):
        spec = graph.parameters[name]
        alignment = {"float": 4, "vec2": 8, "vec3": 16, "vec4": 16}[
            spec.glsl_type
        ]
        offset = (offset + alignment - 1) // alignment * alignment
        offsets[name] = offset
        types[name] = spec.glsl_type
        offset += {"float": 4, "vec2": 8, "vec3": 12, "vec4": 16}[
            spec.glsl_type
        ]
    size = (offset + 15) // 16 * 16
    return UniformLayout(offsets, max(size, 16), types)


class GLSLGenerator:
    def __init__(self, graph: ParsedGraph):
        if not graph.valid:
            raise ValueError("Cannot generate GLSL for an invalid graph")
        self.graph = graph
        self._node_names = {
            node_id: f"n{index}" for index, node_id in enumerate(graph.topo_order)
        }
        self._var_names: dict[tuple[str, str], str] = {}
        for node_id in graph.topo_order:
            info = graph.node_info[node_id]
            for port in info.output_types:
                self._var_names[(node_id, port)] = (
                    f"{self._node_names[node_id]}_{_safe(port)}"
                )

    def generate(self) -> ShaderSources:
        return ShaderSources(VERTEX_SHADER, self._fragment())

    def _fragment(self) -> str:
        graph = self.graph
        lines: list[str] = [
            "#version 330 core",
            "// Generated deterministically from material graph "
            + repr(graph.graph_id),
            "precision highp float;",
            "precision highp int;",
            "",
            "layout(std140) uniform MaterialParameters {",
        ]
        if graph.parameters:
            for name in sorted(graph.parameters):
                spec = graph.parameters[name]
                lines.append(f"    {spec.glsl_type} {graph.parameter_uniform(name)};")
        else:
            lines.append("    float _material_padding;")
        lines.extend(
            [
                "};",
                "",
                "layout(std140) uniform SceneParameters {",
                "    vec4 u_camera_position;",
                "    vec4 u_camera_forward;",
                "    vec4 u_camera_right;",
                "    vec4 u_camera_up;",
                "    vec4 u_lens;",
                "    vec4 u_light_direction;",
                "    vec4 u_light_color_intensity;",
                "};",
                "",
                "uniform int u_channel;",
                "in vec2 v_uv;",
                "layout(location = 0) out vec4 out_channel;",
                "layout(location = 1) out vec4 out_roughness;",
                "layout(location = 2) out vec4 out_metallic;",
                "layout(location = 3) out vec4 out_normal;",
                "",
                "struct MaterialChannels {",
                "    vec3 base_color;",
                "    float roughness;",
                "    float metallic;",
                "    vec3 normal;",
                "    vec3 emissive;",
                "};",
                "",
            ]
        )
        textures = sorted(self._referenced_textures())
        for texture in textures:
            lines.append(f"uniform sampler2D {graph.texture_uniform(texture)};")
        if textures:
            lines.append("")

        lines.extend(
            [
                "vec2 _sphereUV(vec3 p) {",
                "    return vec2(atan(p.z, p.x) / 6.28318530718 + 0.5,",
                "                acos(clamp(p.y, -1.0, 1.0)) / 3.14159265359);",
                "}",
                "",
                "MaterialChannels evaluateMaterial(vec3 P, vec3 N, vec2 UV) {",
                "    MaterialChannels material;",
            ]
        )
        lines.extend(self._node_code())
        for channel in CHANNELS:
            node_id, port = graph.outputs[channel]
            var = self._var_names[(node_id, port)]
            lines.append(f"    material.{channel} = {var};")
        lines.extend(
            [
                "    return material;",
                "}",
                "",
                "vec3 shadeBase(MaterialChannels m, vec3 N, vec3 P, vec3 rayDir) {",
                "    vec3 L = normalize(-u_light_direction.xyz);",
                "    vec3 V = normalize(-rayDir);",
                "    vec3 H = normalize(L + V);",
                "    float ndl = max(dot(normalize(N), L), 0.0);",
                "    float shininess = mix(96.0, 6.0, clamp(m.roughness, 0.0, 1.0));",
                "    float specular = pow(max(dot(normalize(N), H), 0.0), shininess);",
                "    vec3 diffuseAlbedo = m.base_color * (1.0 - clamp(m.metallic, 0.0, 1.0));",
                "    vec3 specularColor = mix(vec3(1.0), m.base_color, clamp(m.metallic, 0.0, 1.0));",
                "    vec3 lightColor = u_light_color_intensity.rgb * u_light_color_intensity.a;",
                "    return diffuseAlbedo * lightColor * (0.12 + 0.88 * ndl)",
                "         + specularColor * lightColor * specular * 0.35;",
                "}",
                "",
                "vec3 backgroundChannel(int channel, vec2 uv) {",
                "    vec2 cell = floor(uv * 12.0);",
                "    float checker = mod(cell.x + cell.y, 2.0);",
                "    if (channel == 0) {",
                "        return mix(vec3(0.018, 0.024, 0.034), vec3(0.055, 0.070, 0.090), checker);",
                "    } else if (channel == 1 || channel == 2) {",
                "        return vec3(0.0);",
                "    } else if (channel == 3) {",
                "        return vec3(0.5, 0.5, 1.0);",
                "    }",
                "    vec3 horizon = mix(vec3(0.01, 0.008, 0.02), vec3(0.08, 0.02, 0.01), uv.y);",
                "    return horizon * (0.35 + 0.25 * checker);",
                "}",
                "",
                "void main() {",
                "    vec2 ndc = v_uv * 2.0 - 1.0;",
                "    vec3 ro = u_camera_position.xyz;",
                "    vec3 rd = normalize(u_camera_forward.xyz",
                "        + u_camera_right.xyz * (ndc.x * u_lens.x * u_lens.y)",
                "        + u_camera_up.xyz * (ndc.y * u_lens.x));",
                "    float b = dot(rd, ro);",
                "    float c = dot(ro, ro) - 1.0;",
                "    float discriminant = b * b - c;",
                "    bool hit = false;",
                "    float t = 0.0;",
                "    if (discriminant >= 0.0) {",
                "        float root = sqrt(discriminant);",
                "        t = -b - root;",
                "        if (t <= 0.0) t = -b + root;",
                "        hit = t > 0.0;",
                "    }",
                "    vec3 P = ro + rd * max(t, 0.0);",
                "    vec3 N = hit ? normalize(P) : vec3(0.0, 0.0, 1.0);",
                "    vec2 UV = hit ? _sphereUV(P) : v_uv;",
                "    MaterialChannels m = evaluateMaterial(P, N, UV);",
                "    vec3 color;",
                "    if (!hit) {",
                "        color = backgroundChannel(u_channel, v_uv);",
                "    } else if (u_channel == 0) {",
                "        color = shadeBase(m, N, P, rd);",
                "    } else if (u_channel == 1) {",
                "        color = vec3(clamp(m.roughness, 0.0, 1.0));",
                "    } else if (u_channel == 2) {",
                "        color = vec3(clamp(m.metallic, 0.0, 1.0));",
                "    } else if (u_channel == 3) {",
                "        color = normalize(m.normal) * 0.5 + 0.5;",
                "    } else {",
                "        color = clamp(m.emissive, 0.0, 1.0);",
                "    }",
                "    out_channel = vec4(clamp(color, 0.0, 1.0), 1.0);",
                "    out_roughness = vec4(hit ? m.roughness : 0.0, 0.0, 0.0, 1.0);",
                "    out_metallic = vec4(hit ? m.metallic : 0.0, 0.0, 0.0, 1.0);",
                "    out_normal = vec4(normalize(hit ? m.normal : N), 1.0);",
                "}",
                "",
            ]
        )
        return "\n".join(lines)

    def _referenced_textures(self) -> set[str]:
        result: set[str] = set()
        for node_id in self.graph.topo_order:
            node = self.graph.nodes[node_id]
            if node.node_type == "texture":
                result.add(str(node.params.get("texture")))
        return result

    def _node_code(self) -> list[str]:
        lines: list[str] = []
        for node_id in self.graph.topo_order:
            node = self.graph.nodes[node_id]
            info = self.graph.node_info[node_id]
            lines.append(f"    // node[{node_id}] type={node.node_type}")
            lines.extend(self._emit_node(node, info))
        return lines

    def _input(self, node_id: str, port: str) -> str:
        for edge in self.graph.edges:
            if edge.to_node == node_id:
                target_port = edge.to_port or _default_input_port(self.graph.nodes[node_id])
                if target_port == port:
                    source_port = edge.from_port or self._only_output(edge.from_node)
                    return self._var_names[(edge.from_node, source_port)]
        raise ValueError(f"Missing input {node_id}.{port}")

    def _only_output(self, node_id: str) -> str:
        info = self.graph.node_info[node_id]
        if "out" in info.output_types:
            return "out"
        if len(info.output_types) == 1:
            return next(iter(info.output_types))
        raise ValueError(f"Ambiguous output for node {node_id}")

    def _emit_node(self, node: Node, info) -> list[str]:
        node_id = node.node_id
        kind = node.node_type
        params = node.params
        output_lines: list[str] = []

        def declaration(port: str, expression: str) -> None:
            glsl_type = info.output_types[port]
            var = self._var_names[(node_id, port)]
            output_lines.append(f"    {glsl_type} {var} = {expression};")

        if kind == "constant_float":
            declaration("out", _fmt(float(params.get("value", 0.0))))
        elif kind in {"constant_vec2", "constant_vec3", "constant_vec4"}:
            glsl_type = kind.removeprefix("constant_")
            value = params.get("value", [0.0] * TYPE_COMPONENTS[glsl_type])
            declaration("out", _glsl_value(value, glsl_type))
        elif kind == "constant_color":
            value = params.get("value", params.get("color", [0.0, 0.0, 0.0]))
            declaration("out", _glsl_value(value, "vec3"))
        elif kind == "parameter":
            declaration("out", self.graph.parameter_uniform(str(params["name"])))
        elif kind == "uv":
            declaration("out", "UV")
        elif kind == "world_position":
            declaration("out", "P")
        elif kind == "world_normal":
            declaration("out", "normalize(N)")
        elif kind == "texture":
            uv = self._input(node_id, "uv")
            texture = str(params["texture"])
            rgba = self._var_names[(node_id, "rgba")]
            output_lines.append(
                f"    vec4 {rgba} = texture({self.graph.texture_uniform(texture)}, {uv});"
            )
            output_lines.append(f"    vec3 {self._var_names[(node_id, 'color')]} = {rgba}.rgb;")
            output_lines.append(f"    float {self._var_names[(node_id, 'alpha')]} = {rgba}.a;")
        elif kind == "math":
            output_lines.extend(self._emit_math(node, info))
        elif kind == "swizzle":
            source = self._input(node_id, "in")
            components = str(params["components"])
            declaration("out", f"{source}.{components}")
        elif kind == "combine":
            size = int(params.get("size", 3))
            ports = ["x", "y", "z", "w"][:size]
            args = ", ".join(self._input(node_id, port) for port in ports)
            declaration("out", f"vec{size}({args})")
        elif kind == "normal_map":
            tangent = self._input(node_id, "normal")
            strength = (
                self._input(node_id, "strength")
                if self._has_input(node_id, "strength")
                else "1.0"
            )
            prefix = self._node_names[node_id]
            output_lines.extend(
                [
                    f"    vec3 {prefix}_geo = normalize(N);",
                    f"    vec3 {prefix}_dp1 = dFdx(P);",
                    f"    vec3 {prefix}_dp2 = dFdy(P);",
                    f"    vec3 {prefix}_tangent = normalize({prefix}_dp1 - "
                    f"{prefix}_geo * dot({prefix}_geo, {prefix}_dp1));",
                    f"    vec3 {prefix}_bitangent = normalize(cross({prefix}_geo, "
                    f"{prefix}_tangent));",
                    f"    vec3 {prefix}_ts = normalize({tangent});",
                    f"    vec3 {prefix}_mapped = normalize("
                    f"{prefix}_tangent * {prefix}_ts.x + "
                    f"{prefix}_bitangent * {prefix}_ts.y + "
                    f"{prefix}_geo * {prefix}_ts.z);",
                    f"    vec3 {self._var_names[(node_id, 'out')]} = normalize("
                    f"mix({prefix}_geo, {prefix}_mapped, clamp({strength}, 0.0, 1.0)));",
                ]
            )
        elif kind == "quantize":
            value = self._input(node_id, "value")
            steps = _fmt(float(params.get("steps", 255.0)))
            declaration("out", f"floor(({value}) * {steps} + 0.5) / {steps}")
        elif kind == "channel_switch":
            value_type = str(params.get("type", "float"))
            values = params.get("values", {})
            expression = _glsl_value(params.get("default", 0.0), value_type)
            for channel in reversed(CHANNELS):
                if channel in values:
                    literal = _glsl_value(values[channel], value_type)
                    expression = (
                        f"(u_channel == {CHANNEL_INDEX[channel]} ? "
                        f"{literal} : {expression})"
                    )
            declaration("out", expression)
        else:
            raise ValueError(f"Unsupported node type {kind}")
        return output_lines

    def _has_input(self, node_id: str, port: str) -> bool:
        for edge in self.graph.edges:
            if edge.to_node == node_id:
                target = edge.to_port or _default_input_port(self.graph.nodes[node_id])
                if target == port:
                    return True
        return False

    def _emit_math(self, node: Node, info) -> list[str]:
        node_id = node.node_id
        op = str(node.params["op"])
        output_type = info.output_types["out"]
        a = self._input(node_id, "a")
        if op in {"add", "sub", "mul", "div"}:
            b = self._input(node_id, "b")
            a_type = info.input_types.get("a")
            b_type = info.input_types.get("b")
            if a_type == "float" and b_type in {"vec2", "vec3", "vec4"}:
                a = f"{b_type}({a})"
            elif b_type == "float" and a_type in {"vec2", "vec3", "vec4"}:
                b = f"{a_type}({b})"
            expression = f"{a} {_binary_operator(op)} {b}"
        elif op in {"pow", "min", "max"}:
            a = self._broadcast_to(a, info, "a", output_type)
            b = self._input(node_id, "b")
            b = self._broadcast_to(b, info, "b", output_type)
            expression = f"{op}({a}, {b})"
        elif op in {"dot", "cross"}:
            b = self._input(node_id, "b")
            expression = f"{op}({a}, {b})"
        elif op in {
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
            "normalize",
            "length",
        }:
            expression = f"{op}({a})"
        elif op == "negate":
            expression = f"-({a})"
        elif op == "clamp":
            b = self._self_broadcast(self._input(node_id, "b"), info, "a", "b")
            c = self._self_broadcast(self._input(node_id, "c"), info, "a", "c")
            expression = f"clamp({a}, {b}, {c})"
        elif op == "mix":
            a = self._broadcast_to(a, info, "a", output_type)
            b = self._broadcast_to(self._input(node_id, "b"), info, "b", output_type)
            c = self._broadcast_to(self._input(node_id, "c"), info, "c", output_type)
            expression = f"mix({a}, {b}, {c})"
        elif op == "smoothstep":
            b = self._input(node_id, "b")
            c = self._input(node_id, "c")
            expression = f"smoothstep({a}, {b}, {c})"
        elif op == "step":
            b = self._input(node_id, "b")
            expression = f"step({a}, {b})"
        else:
            raise ValueError(f"Unsupported math op {op}")
        var = self._var_names[(node_id, "out")]
        return [f"    {output_type} {var} = {expression};"]

    def _self_broadcast(
        self, expression: str, info, anchor_port: str, actual_port: str
    ) -> str:
        anchor_type = info.input_types.get(anchor_port)
        actual_type = info.input_types.get(actual_port)
        if anchor_type in {"vec2", "vec3", "vec4"} and actual_type == "float":
            return f"{anchor_type}({expression})"
        return expression

    def _broadcast_to(
        self, expression: str, info, actual_port: str, target_type: str
    ) -> str:
        actual_type = info.input_types.get(actual_port)
        if actual_type == "float" and target_type in {"vec2", "vec3", "vec4"}:
            return f"{target_type}({expression})"
        return expression


def _binary_operator(op: str) -> str:
    return {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
    }[op]


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in value)


def generate_shaders(graph: ParsedGraph) -> ShaderSources:
    return GLSLGenerator(graph).generate()

