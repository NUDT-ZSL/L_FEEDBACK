from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .diagnostics import Diagnostic, has_errors
from .gl_context import create_headless_context
from .manifest import CANONICAL_LAYOUT, MaterialSpec


RESERVED_UNIFORMS: dict[str, str] = {
    "u_model": "mat4",
    "u_view_proj": "mat4",
    "u_normal_matrix": "mat3",
    "u_camera_position": "vec3",
    "u_ambient_color": "vec3",
    "u_normal_fallback": "vec3",
    "u_background_color": "vec4",
    "u_light_count": "int",
    "u_light_position": "vec3",
    "u_light_color": "vec3",
    "u_light_intensity": "float",
    "u_light_radius": "float",
}
OUTPUT_CONTRACT = {
    0: "frag_world_position",
    1: "frag_world_normal",
    2: "frag_albedo",
    3: "frag_metallic_roughness",
    4: "frag_color",
}
GL_TYPE_NAMES = {
    5120: "byte",
    5121: "ubyte",
    5122: "short",
    5123: "ushort",
    5124: "int",
    5125: "uint",
    5126: "float",
    35664: "vec2",
    35665: "vec3",
    35666: "vec4",
    35667: "ivec2",
    35668: "ivec3",
    35669: "ivec4",
    35670: "bool",
    35671: "bvec2",
    35672: "bvec3",
    35673: "bvec4",
    35674: "mat2",
    35675: "mat3",
    35676: "mat4",
    36294: "uvec2",
    36295: "uvec3",
    36296: "uvec4",
    35678: "sampler2D",
}
UNIFORM_DECL_RE = re.compile(
    r"^\s*(?:layout\s*\([^)]*\)\s*)?uniform\s+"
    r"(?:(?:highp|mediump|lowp)\s+)?"
    r"([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*(?:\[(\d*)\])?\s*;"
)
OUTPUT_RE = re.compile(
    r"^\s*(?:layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*)?"
    r"out\s+(?:(?:highp|mediump|lowp)\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;"
)
INPUT_DECL_RE = re.compile(
    r"^\s*(?:(flat|smooth|noperspective)\s+)?in\s+"
    r"(?:(?:highp|mediump|lowp)\s+)?([A-Za-z_]\w*)\s+([A-Za-z_]\w*)\s*;"
)
STUB_FRAGMENT = """\
#version 330 core
layout(location = 0) out vec4 frag_color;
void main() {
    frag_color = vec4(1.0);
}
"""
LINE_PATTERNS = (
    re.compile(r"(?:^|\s)\d+:(\d+)(?:\((\d+)\))?\s*:\s*(.*)$"),
    re.compile(r"(?:^|\s)\d+\((\d+)\)\s*:\s*(.*)$"),
    re.compile(r"(?:^|\s)(\d+):(\d+)\s*:\s*(.*)$"),
)


@dataclass
class UniformDecl:
    name: str
    type: str
    stage: str
    line: int
    array_length: int | None = None


@dataclass
class ActiveUniform:
    name: str
    type: str
    array_length: int


@dataclass
class StageResult:
    stage: str
    status: str
    source_path: str
    line_count: int
    diagnostics: list[Diagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "source_path": self.source_path,
            "line_count": self.line_count,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass
class MaterialCompileResult:
    material: str
    channels: list[str]
    status: str
    stages: list[StageResult]
    active_uniforms: list[dict[str, Any]]
    outputs: dict[str, str]
    attributes: list[str]
    diagnostics: list[Diagnostic]

    def to_dict(self) -> dict[str, Any]:
        return {
            "material": self.material,
            "channels": list(self.channels),
            "status": self.status,
            "stages": [item.to_dict() for item in self.stages],
            "active_uniforms": list(self.active_uniforms),
            "outputs": dict(self.outputs),
            "attributes": list(self.attributes),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


def _strip_comments(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), source, flags=re.S)
    return re.sub(r"//[^\n]*", "", source)


def _prepare_source(source: str, stage: str, material: str, channels: tuple[str, ...]) -> tuple[str, list[Diagnostic]]:
    diagnostics: list[Diagnostic] = []
    if not re.search(r"^\s*#version\b", source, flags=re.M):
        source = "#version 330 core\n" + source
        diagnostics.append(
            Diagnostic(
                "warning",
                "default_shader_version",
                "Shader has no #version directive; #version 330 core was prepended",
                material=material,
                channels=channels,
                stage=stage,
                line=1,
            )
        )
    return source, diagnostics


def _parse_declarations(source: str, stage: str) -> list[UniformDecl]:
    result: list[UniformDecl] = []
    for line_number, line in enumerate(_strip_comments(source).splitlines(), start=1):
        match = UNIFORM_DECL_RE.match(line)
        if not match:
            continue
        array_raw = match.group(3)
        array_length = int(array_raw) if array_raw else (0 if array_raw == "" else None)
        result.append(
            UniformDecl(
                name=match.group(2),
                type=match.group(1),
                stage=stage,
                line=line_number,
                array_length=array_length,
            )
        )
    return result


def _parse_outputs(source: str) -> dict[int, str]:
    outputs: dict[int, str] = {}
    next_location = 0
    for line in _strip_comments(source).splitlines():
        match = OUTPUT_RE.match(line)
        if not match:
            continue
        if match.group(1) is not None:
            location = int(match.group(1))
        else:
            while next_location in outputs:
                next_location += 1
            location = next_location
        next_location = max(next_location, location + 1)
        outputs[location] = match.group(3)
    return outputs


def _stub_vertex_for_fragment(fragment_source: str) -> str:
    declarations: list[str] = []
    for line in _strip_comments(fragment_source).splitlines():
        match = INPUT_DECL_RE.match(line)
        if not match:
            continue
        qualifier = f"{match.group(1)} " if match.group(1) else ""
        declarations.append(f"{qualifier}out {match.group(2)} {match.group(3)};")
    body = "\n".join(declarations)
    return (
        "#version 330 core\n"
        f"{body}\n"
        "void main() {\n"
        "    gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "}\n"
    )


def _is_compile_failure(log: str) -> bool:
    lower = log.lower()
    if "linker failed" in lower or "link failed" in lower:
        return False
    return "compiler failed" in lower or "compilation failed" in lower or "error" in lower


def _parse_error_line(line: str) -> tuple[int | None, int | None, str]:
    stripped = line.strip()
    for pattern_index, pattern in enumerate(LINE_PATTERNS):
        match = pattern.search(stripped)
        if not match:
            continue
        if pattern_index == 0:
            return int(match.group(1)), int(match.group(2)) if match.group(2) else None, match.group(3)
        if pattern_index == 1:
            return int(match.group(1)), None, match.group(2)
        return int(match.group(1)), int(match.group(2)), match.group(3)
    return None, None, stripped


def _diagnostics_from_log(
    log: str,
    stage: str,
    material: str,
    channels: tuple[str, ...],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    seen: set[str] = set()
    for raw_line in log.splitlines():
        line = raw_line.strip()
        if not line or line in seen:
            continue
        lower = line.lower()
        if "error" not in lower and "warning" not in lower:
            continue
        seen.add(line)
        line_number, column, message = _parse_error_line(line)
        severity = "warning" if "warning" in lower and "error" not in lower else "error"
        diagnostics.append(
            Diagnostic(
                severity=severity,
                code="glsl_compile_error" if severity == "error" else "glsl_compile_warning",
                message=message or line,
                material=material,
                channels=channels,
                stage=stage,
                line=line_number,
                column=column,
                raw=raw_line,
            )
        )
    if not diagnostics and log.strip():
        diagnostics.append(
            Diagnostic(
                "error",
                "shader_compile_failed",
                log.strip(),
                material=material,
                channels=channels,
                stage=stage,
                raw=log,
            )
        )
    return diagnostics


def _active_uniforms(program: Any) -> dict[str, ActiveUniform]:
    result: dict[str, ActiveUniform] = {}
    for name, member in program._members.items():
        if member.__class__.__name__ != "Uniform":
            continue
        gl_type = getattr(member, "gl_type", None)
        if gl_type not in GL_TYPE_NAMES:
            continue
        result[name] = ActiveUniform(
            name=name,
            type=GL_TYPE_NAMES[gl_type],
            array_length=int(getattr(member, "array_length", 1)),
        )
    return result


def _active_attributes(program: Any) -> list[str]:
    result: list[str] = []
    for name, member in program._members.items():
        if member.__class__.__name__ == "Attribute":
            result.append(name)
    return sorted(result)


def _uniform_type_matches(declared: str, active: ActiveUniform) -> bool:
    return declared == active.type


def _channel_tuple(material: MaterialSpec) -> tuple[str, ...]:
    return tuple(material.channel_names)


def _compare_uniforms(
    material: MaterialSpec,
    active: dict[str, ActiveUniform],
    declarations: list[UniformDecl],
    policy: str,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    channels = _channel_tuple(material)
    declared_by_name = {item.name: item for item in declarations}
    declared_names = set(material.uniforms)
    for name, uniform in active.items():
        if name in RESERVED_UNIFORMS:
            expected = RESERVED_UNIFORMS[name]
            if expected != uniform.type:
                diagnostics.append(
                    Diagnostic(
                        "error",
                        "reserved_uniform_type_mismatch",
                        f"Reserved uniform {name} must have GLSL type {expected}, found {uniform.type}",
                        material=material.name,
                        channels=channels,
                        details={"active_type": uniform.type, "expected_type": expected},
                    )
                )
            continue
        if name not in declared_names:
            declaration = declared_by_name.get(name)
            source_path = None
            if declaration is not None:
                source_path = (
                    material.vertex_shader
                    if declaration.stage == "vertex"
                    else material.fragment_shader
                )
            diagnostics.append(
                Diagnostic(
                    "error",
                    "active_uniform_not_declared",
                    f"Shader reads uniform {name} but the manifest does not declare it",
                    source_path,
                    material=material.name,
                    channels=channels,
                    stage=declaration.stage if declaration else None,
                    line=declaration.line if declaration else None,
                    details={
                        "uniform": name,
                        "shader_type": uniform.type,
                        "array_length": uniform.array_length,
                    },
                )
            )
            continue
        spec = material.uniforms[name]
        if not _uniform_type_matches(spec.type, uniform):
            declaration = declared_by_name.get(name)
            diagnostics.append(
                Diagnostic(
                    "error",
                    "uniform_type_mismatch",
                    f"Manifest type {spec.type} for {name} does not match shader type {uniform.type}",
                    spec.path,
                    material=material.name,
                    channels=channels,
                    stage=declaration.stage if declaration else None,
                    line=declaration.line if declaration else None,
                    details={"manifest_type": spec.type, "shader_type": uniform.type},
                )
            )
        if uniform.array_length and uniform.array_length > 1:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "array_uniform_unsupported",
                    f"Manifest uniform {name} is bound as an array but material uniform arrays are unsupported",
                    spec.path,
                    material=material.name,
                    channels=channels,
                    details={"array_length": uniform.array_length},
                )
            )
    if policy != "ignore":
        for name, spec in material.uniforms.items():
            if name in active:
                continue
            severity = "error" if policy == "error" else "warning"
            diagnostics.append(
                Diagnostic(
                    severity,
                    "declared_uniform_unused",
                    f"Manifest declares {name} but no linked shader stage uses it",
                    spec.path,
                    material=material.name,
                    channels=channels,
                    details={"policy": policy},
                )
            )
    return diagnostics


def _validate_outputs(
    material: MaterialSpec,
    outputs: dict[int, str],
    channels: tuple[str, ...],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    for location, expected_name in OUTPUT_CONTRACT.items():
        actual_name = outputs.get(location)
        if actual_name is None:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "missing_shader_output",
                    f"Fragment shader has no output at location {location}; expected {expected_name}",
                    material.fragment_shader,
                    material=material.name,
                    channels=channels,
                    stage="fragment",
                    details={"location": location, "expected": expected_name, "actual": None},
                )
            )
        elif actual_name != expected_name:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "shader_output_layout_mismatch",
                    f"Fragment output location {location} is {actual_name}; expected {expected_name}",
                    None,
                    material=material.name,
                    channels=channels,
                    stage="fragment",
                    details={"location": location, "expected": expected_name, "actual": actual_name},
                )
            )
    return diagnostics


def _stage_error(
    stage: str,
    source_path: Path,
    material: MaterialSpec,
    message: str,
) -> tuple[StageResult, list[Diagnostic]]:
    diagnostic = Diagnostic(
        "error",
        "shader_source_error",
        message,
        material=material.name,
        channels=_channel_tuple(material),
        stage=stage,
    )
    result = StageResult(
        stage=stage,
        status="error",
        source_path=str(source_path),
        line_count=0,
        diagnostics=[diagnostic],
    )
    return result, [diagnostic]


def compile_material_with_context(
    context: Any,
    material: MaterialSpec,
    unused_uniform_policy: str = "warn",
) -> MaterialCompileResult:
    channels = _channel_tuple(material)
    diagnostics: list[Diagnostic] = []
    stages: list[StageResult] = []
    stage_ok: dict[str, bool] = {}
    sources: dict[str, str] = {}
    all_declarations: list[UniformDecl] = []
    stage_inputs = (
        ("vertex", material.vertex_path),
        ("fragment", material.fragment_path),
    )
    for stage, source_path in stage_inputs:
        stage_diagnostics: list[Diagnostic] = []
        try:
            source = source_path.read_text(encoding="utf-8")
        except OSError as exc:
            result, stage_diagnostics = _stage_error(stage, source_path, material, f"Cannot read shader: {exc}")
            stages.append(result)
            diagnostics.extend(stage_diagnostics)
            continue
        prepared, version_diagnostics = _prepare_source(source, stage, material.name, channels)
        stage_diagnostics.extend(version_diagnostics)
        diagnostics.extend(version_diagnostics)
        sources[stage] = prepared
        all_declarations.extend(_parse_declarations(prepared, stage))
        try:
            if stage == "vertex":
                isolated_program = context.program(vertex_shader=prepared, fragment_shader=STUB_FRAGMENT)
            else:
                isolated_program = context.program(
                    vertex_shader=_stub_vertex_for_fragment(prepared),
                    fragment_shader=prepared,
                )
        except Exception as exc:
            log = str(exc.args[0]) if exc.args else str(exc)
            if _is_compile_failure(log):
                parsed = _diagnostics_from_log(log, stage, material.name, channels)
                stage_diagnostics.extend(parsed)
                diagnostics.extend(parsed)
                stage_ok[stage] = False
                status = "error"
            else:
                stage_ok[stage] = True
                status = "ok"
            stages.append(
                StageResult(
                    stage=stage,
                    status=status,
                    source_path=str(source_path),
                    line_count=len(prepared.splitlines()),
                    diagnostics=stage_diagnostics,
                )
            )
            continue
        isolated_program.release()
        stage_ok[stage] = True
        stages.append(
            StageResult(
                stage=stage,
                status="ok",
                source_path=str(source_path),
                line_count=len(prepared.splitlines()),
                diagnostics=stage_diagnostics,
            )
        )
    program = None
    if all(stage_ok.get(stage, False) for stage in ("vertex", "fragment")):
        link_diagnostics: list[Diagnostic] = []
        try:
            program = context.program(
                vertex_shader=sources["vertex"],
                fragment_shader=sources["fragment"],
            )
        except Exception as exc:
            log = str(exc.args[0]) if exc.args else str(exc)
            link_diagnostics = _diagnostics_from_log(log, "link", material.name, channels)
            diagnostics.extend(link_diagnostics)
            stages.append(
                StageResult(
                    stage="link",
                    status="error",
                    source_path=str(material.fragment_path),
                    line_count=0,
                    diagnostics=link_diagnostics,
                )
            )
        else:
            stages.append(
                StageResult(
                    stage="link",
                    status="ok",
                    source_path=str(material.fragment_path),
                    line_count=0,
                )
            )
    active: dict[str, ActiveUniform] = {}
    attributes: list[str] = []
    outputs: dict[int, str] = {}
    if program is not None:
        active = _active_uniforms(program)
        attributes = _active_attributes(program)
        outputs = _parse_outputs(sources.get("fragment", ""))
        diagnostics.extend(_compare_uniforms(material, active, all_declarations, unused_uniform_policy))
        diagnostics.extend(_validate_outputs(material, outputs, channels))
        for name in ("in_position", "in_normal"):
            if name not in attributes:
                diagnostics.append(
                    Diagnostic(
                        "error",
                        "missing_vertex_attribute",
                        f"Vertex shader does not consume required attribute {name}",
                        material=material.name,
                        channels=channels,
                        stage="vertex",
                    )
                )
        program.release()
    elif stage_ok.get("fragment", False):
        outputs = _parse_outputs(sources.get("fragment", ""))
        diagnostics.extend(_validate_outputs(material, outputs, channels))
    status = "error" if has_errors(diagnostics) else "ok"
    active_rows = [
        {"name": item.name, "type": item.type, "array_length": item.array_length}
        for item in sorted(active.values(), key=lambda item: item.name)
    ]
    return MaterialCompileResult(
        material=material.name,
        channels=list(channels),
        status=status,
        stages=stages,
        active_uniforms=active_rows,
        outputs={str(key): value for key, value in sorted(outputs.items())},
        attributes=attributes,
        diagnostics=diagnostics,
    )


def _context_error_result(material: MaterialSpec, exc: Exception) -> MaterialCompileResult:
    diagnostic = Diagnostic(
        "error",
        "gl_context_error",
        f"Cannot create a compiler OpenGL context: {type(exc).__name__}: {exc}",
        material=material.name,
        channels=_channel_tuple(material),
        stage="context",
    )
    return MaterialCompileResult(
        material=material.name,
        channels=list(_channel_tuple(material)),
        status="error",
        stages=[],
        active_uniforms=[],
        outputs={},
        attributes=[],
        diagnostics=[diagnostic],
    )


def _compile_isolated(material: MaterialSpec, policy: str) -> MaterialCompileResult:
    try:
        context, _ = create_headless_context()
    except Exception as exc:
        return _context_error_result(material, exc)
    try:
        return compile_material_with_context(context, material, policy)
    finally:
        context.release()


def compile_materials(
    materials: Iterable[MaterialSpec],
    unused_uniform_policy: str = "warn",
    workers: int = 1,
) -> list[MaterialCompileResult]:
    material_list = list(materials)
    if workers <= 1:
        try:
            context, _ = create_headless_context()
        except Exception as exc:
            return [_context_error_result(material, exc) for material in material_list]
        try:
            return [
                compile_material_with_context(context, material, unused_uniform_policy)
                for material in material_list
            ]
        finally:
            context.release()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="material-compile") as executor:
        return list(
            executor.map(
                lambda material: _compile_isolated(material, unused_uniform_policy),
                material_list,
            )
        )
