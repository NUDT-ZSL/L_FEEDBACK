from __future__ import annotations

import hashlib
import heapq
import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import PipelineError, graph_error
from .limits import DEFAULT_LIMITS, Limits
from .texture import TextureAsset, load_texture, sample_texture

SUPPORTED_NODE_TYPES = {
    "constant",
    "uv",
    "texture_sample",
    "add",
    "multiply",
    "mix",
    "fresnel",
    "normal_map",
}

OUTPUT_TYPES = {
    "base_color": 3,
    "normal": 3,
    "roughness": 1,
    "metallic": 1,
    "emissive": 3,
}

INPUT_KEYS = {
    "constant": set(),
    "uv": set(),
    "texture_sample": {"uv"},
    "add": {"a", "b"},
    "multiply": {"a", "b"},
    "mix": {"a", "b", "factor"},
    "fresnel": {"normal", "ior"},
    "normal_map": {"normal", "strength"},
}

MAX_CONSTANT_ABS = 1_000_000.0


@dataclass(frozen=True)
class Operand:
    kind: str
    dim: int
    value: Any = None
    name: Optional[str] = None

    @classmethod
    def constant(cls, value: Any, dim: int) -> "Operand":
        return cls(kind="constant", dim=dim, value=value)

    @classmethod
    def temp(cls, name: str, dim: int) -> "Operand":
        return cls(kind="temp", dim=dim, name=name)

    def to_dict(self) -> Dict[str, Any]:
        if self.kind == "temp":
            return {"kind": "temp", "dim": self.dim, "name": self.name}
        return {"kind": "constant", "dim": self.dim, "value": self.value}


@dataclass(frozen=True)
class Instruction:
    node_index: int
    node_id: str
    op: str
    dst: int
    operands: Tuple[Tuple[str, Operand], ...]
    attrs: Tuple[Tuple[str, Any], ...]
    result_dim: int

    @property
    def temp_name(self) -> str:
        return f"t{self.dst}"

    def operand_dict(self) -> Dict[str, Operand]:
        return dict(self.operands)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_index": self.node_index,
            "node_id": self.node_id,
            "op": self.op,
            "dst": self.temp_name,
            "result_dim": self.result_dim,
            "operands": {key: operand.to_dict() for key, operand in self.operands},
            "attrs": {key: value for key, value in self.attrs},
        }


@dataclass(frozen=True)
class FoldedNode:
    node_index: int
    node_id: str
    op: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_index": self.node_index,
            "node_id": self.node_id,
            "op": self.op,
        }


@dataclass(frozen=True)
class CompiledMaterial:
    schema: str
    name: str
    instructions: Tuple[Instruction, ...]
    outputs: Dict[str, Operand]
    folded_nodes: Tuple[FoldedNode, ...]
    textures: Dict[str, TextureAsset]
    node_count: int
    limits: Limits

    def instruction_order(self) -> List[str]:
        return [instruction.node_id for instruction in self.instructions]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "material-ir/v1",
            "source_schema": self.schema,
            "name": self.name,
            "node_count": self.node_count,
            "instruction_count": len(self.instructions),
            "folded_node_count": len(self.folded_nodes),
            "instruction_order": self.instruction_order(),
            "folded_nodes": [folded.to_dict() for folded in self.folded_nodes],
            "outputs": {
                key: operand.to_dict() for key, operand in sorted(self.outputs.items())
            },
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "textures": {
                key: asset.metadata() for key, asset in sorted(self.textures.items())
            },
            "limits": self.limits.as_dict(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Node:
    index: int
    node_id: str
    op: str
    inputs: Dict[str, str]
    raw: Mapping[str, Any]
    dim: int = 0
    constant_value: Any = None


def _node_error(
    code: str,
    message: str,
    node: _Node,
    field: Optional[str] = None,
) -> PipelineError:
    return graph_error(
        code,
        message,
        node_index=node.index,
        node_id=node.node_id,
        field=field,
    )


def _validate_number(value: Any, node: _Node, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise _node_error("E_CONSTANT_TYPE", "value must be a finite number", node, field)
    result = float(value)
    if not math.isfinite(result):
        raise _node_error("E_CONSTANT_TYPE", "value must be finite", node, field)
    if abs(result) > MAX_CONSTANT_ABS:
        raise _node_error(
            "E_LIMIT",
            f"absolute constant value exceeds {MAX_CONSTANT_ABS:g}",
            node,
            field,
        )
    return result


def _constant_operand(value: Any, node: _Node) -> Operand:
    if isinstance(value, list):
        if len(value) not in (2, 3):
            raise _node_error(
                "E_CONSTANT_TYPE",
                "constant vector must contain 2 or 3 numbers",
                node,
                "value",
            )
        numbers_out = tuple(
            _validate_number(component, node, f"value[{index}]")
            for index, component in enumerate(value)
        )
        return Operand.constant(numbers_out, len(numbers_out))
    numeric = _validate_number(value, node, "value")
    return Operand.constant(numeric, 1)


def _validate_id(value: Any, index: int) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise graph_error(
            "E_NODE_ID",
            "node id must be a non-empty string of at most 128 characters",
            node_index=index,
            field="id",
        )
    return value


def _parse_nodes(graph: Mapping[str, Any], limits: Limits) -> List[_Node]:
    raw_nodes = graph.get("nodes")
    if not isinstance(raw_nodes, list):
        raise graph_error("E_NODE_SCHEMA", "nodes must be an array", field="nodes")
    if not raw_nodes:
        raise graph_error("E_EMPTY_GRAPH", "nodes must not be empty", field="nodes")
    if len(raw_nodes) > limits.max_nodes:
        raise graph_error(
            "E_LIMIT",
            f"node count {len(raw_nodes)} exceeds limit {limits.max_nodes}",
            node_index=limits.max_nodes,
            field="nodes",
        )

    nodes: List[_Node] = []
    seen_ids: Dict[str, int] = {}
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            raise graph_error(
                "E_NODE_SCHEMA",
                "node must be a JSON object",
                node_index=index,
            )
        node_id = _validate_id(raw.get("id"), index)
        if node_id in seen_ids:
            raise graph_error(
                "E_DUPLICATE_ID",
                f"duplicate node id {node_id!r}; first used at node {seen_ids[node_id]}",
                node_index=index,
                node_id=node_id,
                field="id",
            )
        seen_ids[node_id] = index

        op = raw.get("type")
        if not isinstance(op, str):
            raise graph_error(
                "E_NODE_TYPE",
                "node type must be a string",
                node_index=index,
                node_id=node_id,
                field="type",
            )
        if op not in SUPPORTED_NODE_TYPES:
            raise graph_error(
                "E_NODE_TYPE",
                f"unsupported node type {op!r}",
                node_index=index,
                node_id=node_id,
                field="type",
            )

        raw_inputs = raw.get("inputs", {})
        if not isinstance(raw_inputs, Mapping):
            raise graph_error(
                "E_NODE_SCHEMA",
                "inputs must be an object mapping input names to node ids",
                node_index=index,
                node_id=node_id,
                field="inputs",
            )
        inputs: Dict[str, str] = {}
        for key in sorted(raw_inputs):
            reference = raw_inputs[key]
            if not isinstance(key, str) or not isinstance(reference, str) or not reference:
                raise graph_error(
                    "E_NODE_SCHEMA",
                    "input names and references must be non-empty strings",
                    node_index=index,
                    node_id=node_id,
                    field=f"inputs.{key}",
                )
            inputs[key] = reference

        expected = INPUT_KEYS[op]
        unknown = sorted(set(inputs) - expected)
        if unknown:
            raise graph_error(
                "E_INPUT_UNKNOWN",
                f"unknown input {unknown[0]!r} for node type {op!r}",
                node_index=index,
                node_id=node_id,
                field=f"inputs.{unknown[0]}",
            )
        missing = sorted(expected - set(inputs))
        if missing:
            raise graph_error(
                "E_MISSING_INPUT",
                f"missing required input {missing[0]!r} for node type {op!r}",
                node_index=index,
                node_id=node_id,
                field=f"inputs.{missing[0]}",
            )

        node = _Node(
            index=index,
            node_id=node_id,
            op=op,
            inputs=inputs,
            raw=raw,
        )
        if op == "constant":
            if "value" not in raw:
                raise _node_error(
                    "E_CONSTANT_TYPE",
                    "constant node requires a value",
                    node,
                    "value",
                )
            operand = _constant_operand(raw["value"], node)
            node = _Node(
                index=index,
                node_id=node_id,
                op=op,
                inputs=inputs,
                raw=raw,
                dim=operand.dim,
                constant_value=operand.value,
            )
        if op == "texture_sample":
            texture_id = raw.get("texture")
            if not isinstance(texture_id, str) or not texture_id:
                raise _node_error(
                    "E_TEXTURE_REF",
                    "texture_sample requires a non-empty texture id",
                    node,
                    "texture",
                )
        nodes.append(node)
    return nodes


def _validate_references(nodes: Sequence[_Node]) -> Dict[str, _Node]:
    by_id = {node.node_id: node for node in nodes}
    for node in nodes:
        for input_name in sorted(node.inputs):
            reference = node.inputs[input_name]
            if reference not in by_id:
                raise _node_error(
                    "E_INPUT_NOT_FOUND",
                    f"input {input_name!r} references unknown node {reference!r}",
                    node,
                    f"inputs.{input_name}",
                )
    return by_id


def _check_cycles_and_depth(
    nodes: Sequence[_Node],
    by_id: Mapping[str, _Node],
    limits: Limits,
) -> None:
    state = {node.node_id: 0 for node in nodes}
    depth = {node.node_id: 1 for node in nodes}

    def visit(node: _Node, stack: List[str]) -> int:
        if state[node.node_id] == 1:
            cycle_start = stack.index(node.node_id) if node.node_id in stack else 0
            cycle = stack[cycle_start:] + [node.node_id]
            raise _node_error(
                "E_CYCLE",
                "graph contains a cycle: " + " -> ".join(cycle),
                node,
            )
        if state[node.node_id] == 2:
            return depth[node.node_id]
        state[node.node_id] = 1
        stack.append(node.node_id)
        maximum = 1
        for input_name in sorted(node.inputs):
            child_depth = visit(by_id[node.inputs[input_name]], stack) + 1
            maximum = max(maximum, child_depth)
            if maximum > limits.max_graph_depth:
                raise _node_error(
                    "E_GRAPH_TOO_DEEP",
                    f"graph depth exceeds limit {limits.max_graph_depth}",
                    node,
                )
        stack.pop()
        state[node.node_id] = 2
        depth[node.node_id] = maximum
        return maximum

    for node in sorted(nodes, key=lambda item: item.index):
        if state[node.node_id] == 0:
            visit(node, [])


def _validate_outputs(
    graph: Mapping[str, Any],
    by_id: Mapping[str, _Node],
) -> Dict[str, str]:
    outputs = graph.get("outputs")
    if not isinstance(outputs, Mapping):
        raise graph_error(
            "E_OUTPUT_MISSING",
            "outputs must be an object with five required channels",
            field="outputs",
        )
    unknown = sorted(set(outputs) - set(OUTPUT_TYPES))
    if unknown:
        raise graph_error(
            "E_OUTPUT_UNKNOWN",
            f"unknown output channel {unknown[0]!r}",
            field=f"outputs.{unknown[0]}",
        )
    result: Dict[str, str] = {}
    for output_name in sorted(OUTPUT_TYPES):
        if output_name not in outputs:
            raise graph_error(
                "E_OUTPUT_MISSING",
                f"missing required output {output_name!r}",
                field=f"outputs.{output_name}",
            )
        reference = outputs[output_name]
        if not isinstance(reference, str) or reference not in by_id:
            raise graph_error(
                "E_OUTPUT_REF",
                f"output {output_name!r} must reference an existing node",
                field=f"outputs.{output_name}",
            )
        result[output_name] = reference
    return result


def _validate_reachability(
    nodes: Sequence[_Node],
    by_id: Mapping[str, _Node],
    output_refs: Mapping[str, str],
) -> None:
    reachable = set()
    stack = list(output_refs.values())
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        node = by_id[node_id]
        stack.extend(node.inputs.values())
    for node in sorted(nodes, key=lambda item: item.index):
        if node.node_id not in reachable:
            raise _node_error(
                "E_NO_OUTPUT_PATH",
                "node does not contribute to any material output",
                node,
            )


def _dims_compatible(left: int, right: int) -> bool:
    return left == right or left == 1 or right == 1


def _infer_types(
    by_id: Mapping[str, _Node],
    topological: Sequence[_Node],
    output_refs: Mapping[str, str],
    texture_specs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    inferred: Dict[str, int] = {}

    for node in topological:
        input_dims = {name: inferred[by_id[reference].node_id] for name, reference in node.inputs.items()}

        def type_error(message: str, input_name: str) -> PipelineError:
            return _node_error("E_INPUT_TYPE", message, node, f"inputs.{input_name}")

        if node.op == "constant":
            result_dim = node.dim
        elif node.op == "uv":
            result_dim = 2
        elif node.op == "texture_sample":
            if input_dims["uv"] != 2:
                raise type_error("texture_sample.uv must be a vec2", "uv")
            texture_id = node.raw["texture"]
            if texture_id not in texture_specs:
                raise _node_error(
                    "E_TEXTURE_REF",
                    f"texture {texture_id!r} is not declared in the textures map",
                    node,
                    "texture",
                )
            result_dim = 3
        elif node.op in ("add", "multiply"):
            left = input_dims["a"]
            right = input_dims["b"]
            if not _dims_compatible(left, right):
                raise _node_error(
                    "E_INPUT_DIMENSION",
                    f"{node.op} inputs have incompatible dimensions {left} and {right}",
                    node,
                    "inputs",
                )
            result_dim = max(left, right)
        elif node.op == "mix":
            if input_dims["a"] != input_dims["b"]:
                raise _node_error(
                    "E_INPUT_DIMENSION",
                    "mix inputs a and b must have identical dimensions",
                    node,
                    "inputs",
                )
            if input_dims["factor"] != 1:
                raise type_error("mix.factor must be a scalar", "factor")
            result_dim = input_dims["a"]
        elif node.op == "fresnel":
            if input_dims["normal"] != 3:
                raise type_error("fresnel.normal must be a vec3", "normal")
            if input_dims["ior"] != 1:
                raise type_error("fresnel.ior must be a scalar", "ior")
            result_dim = 1
        elif node.op == "normal_map":
            if input_dims["normal"] != 3:
                raise type_error("normal_map.normal must be a vec3", "normal")
            if input_dims["strength"] != 1:
                raise type_error("normal_map.strength must be a scalar", "strength")
            result_dim = 3
        else:
            raise _node_error("E_NODE_TYPE", f"unsupported node type {node.op!r}", node)

        inferred[node.node_id] = result_dim

    for output_name, expected_dim in OUTPUT_TYPES.items():
        actual_dim = inferred[output_refs[output_name]]
        if actual_dim != expected_dim:
            referenced = by_id[output_refs[output_name]]
            raise graph_error(
                "E_OUTPUT_TYPE",
                (
                    f"output {output_name!r} must have dimension {expected_dim}, "
                    f"but referenced node has dimension {actual_dim}"
                ),
                node_index=referenced.index,
                node_id=referenced.node_id,
                field=f"outputs.{output_name}",
            )
    return inferred


def _stable_topological_order(
    nodes: Sequence[_Node],
    by_id: Mapping[str, _Node],
) -> List[_Node]:
    indegree: Dict[str, int] = {}
    dependents: Dict[str, List[str]] = {node.node_id: [] for node in nodes}
    for node in nodes:
        dependencies = set(node.inputs.values())
        indegree[node.node_id] = len(dependencies)
        for dependency in dependencies:
            dependents[dependency].append(node.node_id)
    for values in dependents.values():
        values.sort()
    ready = [node_id for node_id, count in indegree.items() if count == 0]
    heapq.heapify(ready)
    ordered: List[_Node] = []
    while ready:
        node_id = heapq.heappop(ready)
        ordered.append(by_id[node_id])
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(ordered) != len(nodes):
        remaining = sorted(
            (node for node in nodes if indegree[node.node_id] > 0),
            key=lambda item: item.index,
        )
        raise _node_error("E_CYCLE", "graph contains a cycle", remaining[0])
    return ordered


def _constant_array(operand: Operand) -> np.ndarray:
    value = operand.value
    if operand.dim == 1:
        return np.asarray([value], dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _fold_binary(
    node: _Node,
    left: Operand,
    right: Operand,
    op: str,
) -> Operand:
    left_value = _constant_array(left)
    right_value = _constant_array(right)
    with np.errstate(over="ignore", invalid="ignore"):
        if op == "add":
            result = left_value + right_value
        elif op == "multiply":
            result = left_value * right_value
        else:
            raise AssertionError(op)
    if not np.all(np.isfinite(result)):
        raise _node_error(
            "E_NUMERIC",
            "constant folding produced a non-finite value",
            node,
        )
    dim = max(left.dim, right.dim)
    if dim == 1:
        return Operand.constant(float(result[0]), 1)
    return Operand.constant(tuple(float(component) for component in result), dim)


def _fold_mix(node: _Node, a: Operand, b: Operand, factor: Operand) -> Operand:
    a_value = _constant_array(a)
    b_value = _constant_array(b)
    f = float(factor.value)
    result = a_value * (1.0 - f) + b_value * f
    if not np.all(np.isfinite(result)):
        raise _node_error("E_NUMERIC", "constant folding produced a non-finite value", node)
    if a.dim == 1:
        return Operand.constant(float(result[0]), 1)
    return Operand.constant(tuple(float(component) for component in result), a.dim)


def _texture_specs(graph: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    raw = graph.get("textures", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise graph_error("E_TEXTURE_SCHEMA", "textures must be an object", field="textures")
    result: Dict[str, Mapping[str, Any]] = {}
    for texture_id in sorted(raw):
        if not isinstance(texture_id, str) or not texture_id:
            raise graph_error(
                "E_TEXTURE_SCHEMA",
                "texture ids must be non-empty strings",
                field="textures",
            )
        spec = raw[texture_id]
        if not isinstance(spec, Mapping):
            raise graph_error(
                "E_TEXTURE_SCHEMA",
                f"texture {texture_id!r} must be an object",
                field=f"textures.{texture_id}",
            )
        result[texture_id] = spec
    return result


def compile_graph(
    graph: Mapping[str, Any],
    base_dir: Path,
    limits: Limits = DEFAULT_LIMITS,
) -> CompiledMaterial:
    if not isinstance(graph, Mapping):
        raise graph_error("E_ROOT_TYPE", "material graph root must be a JSON object")
    if graph.get("schema") != "material-graph/v1":
        raise graph_error(
            "E_SCHEMA",
            "material graph schema must be 'material-graph/v1'",
            field="schema",
        )
    name_raw = graph.get("name", "")
    if not isinstance(name_raw, str):
        raise graph_error("E_SCHEMA", "material graph name must be a string", field="name")

    texture_specs = _texture_specs(graph)
    nodes = _parse_nodes(graph, limits)
    by_id = _validate_references(nodes)
    _check_cycles_and_depth(nodes, by_id, limits)
    output_refs = _validate_outputs(graph, by_id)
    _validate_reachability(nodes, by_id, output_refs)
    topological = _stable_topological_order(nodes, by_id)
    dims = _infer_types(by_id, topological, output_refs, texture_specs)

    texture_cache: Dict[str, TextureAsset] = {}

    def get_texture(node: _Node) -> TextureAsset:
        texture_id = node.raw["texture"]
        if texture_id not in texture_specs:
            raise _node_error(
                "E_TEXTURE_REF",
                f"texture {texture_id!r} is not declared in the textures map",
                node,
                "texture",
            )
        if texture_id not in texture_cache:
            texture_cache[texture_id] = load_texture(
                texture_id,
                texture_specs[texture_id],
                base_dir,
                limits,
                node_index=node.index,
                node_id=node.node_id,
            )
        return texture_cache[texture_id]

    refs: Dict[str, Operand] = {}
    instructions: List[Instruction] = []
    folded: List[FoldedNode] = []

    def mark_folded(node: _Node) -> None:
        folded.append(FoldedNode(node.index, node.node_id, node.op))

    def emit(
        node: _Node,
        operands: Iterable[Tuple[str, Operand]],
        attrs: Iterable[Tuple[str, Any]] = (),
    ) -> Operand:
        dst = len(instructions)
        instruction = Instruction(
            node_index=node.index,
            node_id=node.node_id,
            op=node.op,
            dst=dst,
            operands=tuple(operands),
            attrs=tuple(attrs),
            result_dim=dims[node.node_id],
        )
        instructions.append(instruction)
        return Operand.temp(instruction.temp_name, instruction.result_dim)

    for node in topological:
        inputs = {key: refs[reference] for key, reference in node.inputs.items()}
        all_constant = all(operand.kind == "constant" for operand in inputs.values())

        if node.op == "constant":
            refs[node.node_id] = Operand.constant(node.constant_value, node.dim)
            mark_folded(node)
        elif node.op in ("add", "multiply") and all_constant:
            refs[node.node_id] = _fold_binary(node, inputs["a"], inputs["b"], node.op)
            mark_folded(node)
        elif node.op == "mix" and all_constant:
            refs[node.node_id] = _fold_mix(
                node,
                inputs["a"],
                inputs["b"],
                inputs["factor"],
            )
            mark_folded(node)
        elif node.op == "texture_sample" and inputs["uv"].kind == "constant":
            asset = get_texture(node)
            uv_value = tuple(float(component) for component in inputs["uv"].value)
            try:
                sampled = sample_texture(asset, np.asarray(uv_value, dtype=np.float64))
            except ValueError as exc:
                raise _node_error("E_NUMERIC", str(exc), node, "inputs.uv")
            if not np.all(np.isfinite(sampled)):
                raise _node_error("E_NUMERIC", "texture sample is non-finite", node)
            refs[node.node_id] = Operand.constant(
                tuple(float(component) for component in sampled),
                3,
            )
            mark_folded(node)
        elif node.op == "uv":
            refs[node.node_id] = emit(node, ())
        elif node.op == "texture_sample":
            get_texture(node)
            refs[node.node_id] = emit(
                node,
                (("uv", inputs["uv"]),),
                (("texture", node.raw["texture"]),),
            )
        elif node.op in ("add", "multiply"):
            refs[node.node_id] = emit(
                node,
                (("a", inputs["a"]), ("b", inputs["b"])),
            )
        elif node.op == "mix":
            refs[node.node_id] = emit(
                node,
                (
                    ("a", inputs["a"]),
                    ("b", inputs["b"]),
                    ("factor", inputs["factor"]),
                ),
            )
        elif node.op == "fresnel":
            refs[node.node_id] = emit(
                node,
                (("normal", inputs["normal"]), ("ior", inputs["ior"])),
            )
        elif node.op == "normal_map":
            refs[node.node_id] = emit(
                node,
                (("normal", inputs["normal"]), ("strength", inputs["strength"])),
            )
        else:
            raise _node_error("E_NODE_TYPE", f"unsupported node type {node.op!r}", node)

    outputs = {key: refs[reference] for key, reference in output_refs.items()}
    return CompiledMaterial(
        schema="material-graph/v1",
        name=name_raw,
        instructions=tuple(instructions),
        outputs=outputs,
        folded_nodes=tuple(folded),
        textures=dict(sorted(texture_cache.items())),
        node_count=len(nodes),
        limits=limits,
    )


def load_graph_json(path: Path, limits: Limits = DEFAULT_LIMITS) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise graph_error("E_IO", f"cannot read material graph: {type(exc).__name__}")
    if size > limits.max_json_bytes:
        raise graph_error("E_LIMIT", f"material graph exceeds {limits.max_json_bytes} bytes")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise graph_error(
            "E_INVALID_JSON",
            f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",
        )
    except UnicodeError:
        raise graph_error("E_INVALID_JSON", "material graph is not valid UTF-8")
    except OSError as exc:
        raise graph_error("E_IO", f"cannot read material graph: {type(exc).__name__}")
    return value


def load_and_compile(
    path: Path,
    limits: Limits = DEFAULT_LIMITS,
) -> CompiledMaterial:
    graph = load_graph_json(path, limits)
    return compile_graph(graph, path.resolve().parent, limits)
