"""Decode calls and turn them into name-based graph evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .binary import (
    ImportSymbol,
    OP_CALL_IMPORT,
    OP_CALL_LOCAL,
    OP_CMP_IMM,
    OP_JCC,
    OP_JMP,
    OP_NOP,
    OP_PUSH_IMM,
    OP_RET,
    PeImage,
    decode_body,
)
from .manifests import Manifest, region_bytes


@dataclass(frozen=True)
class EffectiveInstruction:
    offset: int
    kind: str
    signature: str
    opcode: int

    def change_record(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "kind": self.kind,
            "offset": self.offset,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class CallSite:
    function: str
    kind: str
    offset: int
    instruction_index: int
    callee: str | None = None
    import_symbol: ImportSymbol | None = None
    branch_number: int | None = None
    branch_offset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "function": self.function,
            "kind": self.kind,
            "offset": self.offset,
            "instruction_index": self.instruction_index,
            "branch_number": self.branch_number,
            "branch_offset": self.branch_offset,
        }
        if self.kind == "local" and self.callee is not None:
            result["callee"] = self.callee
        if self.kind == "import" and self.import_symbol is not None:
            result["import"] = self.import_symbol.to_dict()
        return result


@dataclass
class ParsedFunction:
    function: str
    invalid_bytes: int
    parse_issues: list[dict[str, object]]
    branch_count: int
    calls: list[CallSite]
    broken_calls: list[dict[str, Any]]
    effective: list[EffectiveInstruction]

    @property
    def complete(self) -> bool:
        return self.invalid_bytes == 0 and not self.broken_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "function": self.function,
            "invalid_bytes": self.invalid_bytes,
            "parse_issues": self.parse_issues,
            "branch_count": self.branch_count,
            "calls": [call.to_dict() for call in self.calls],
            "broken_calls": self.broken_calls,
        }


@dataclass
class CallAnalysis:
    manifest: Manifest
    imports: list[ImportSymbol]
    functions: dict[str, ParsedFunction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "imports": [item.to_dict() for item in self.imports],
            "functions": [
                self.functions[name].to_dict()
                for name in sorted(self.functions)
            ],
        }


def _instruction_signature(
    opcode: int,
    value: int,
    branch_number: int | None,
    *,
    callee_name: str | None = None,
    import_symbol: ImportSymbol | None = None,
    local_index: int | None = None,
) -> tuple[str, str]:
    """Return ``(kind, layout-independent signature)`` for an instruction."""

    if opcode == OP_PUSH_IMM:
        return "insn", f"push_imm:{value}"
    if opcode == OP_CMP_IMM:
        return "insn", f"cmp_imm:{value}"
    if opcode == OP_NOP:
        return "insn", "nop"
    if opcode == OP_RET:
        return "insn", "ret"
    if opcode == OP_JCC:
        return "branch", f"jcc:{branch_number}"
    if opcode == OP_JMP:
        return "branch", "jmp"
    if opcode == OP_CALL_LOCAL:
        if callee_name is not None:
            return "insn", f"call_local:{callee_name}"
        return "insn", f"broken_call_local:{local_index}"
    if opcode == OP_CALL_IMPORT:
        if import_symbol is not None:
            return "insn", (
                f"call_import:{import_symbol.module}!{import_symbol.name}"
            )
        return "insn", "broken_call_import"
    return "insn", f"opcode:{opcode}:{value}"


def parse_function(
    *,
    body: bytes,
    function: str,
    local_names: list[str],
    imports: list[ImportSymbol],
    known_functions: set[str],
) -> ParsedFunction:
    decoded = decode_body(body, function)
    effective: list[EffectiveInstruction] = []
    calls: list[CallSite] = []
    broken: list[dict[str, Any]] = []
    branch_count = 0
    active_branch: tuple[int, int] | None = None

    for instruction in decoded.instructions:
        callee: str | None = None
        import_symbol: ImportSymbol | None = None
        local_index: int | None = None
        if instruction.opcode == OP_CALL_LOCAL:
            local_index = instruction.value
            if local_index >= len(local_names):
                broken.append(
                    {
                        "function": function,
                        "offset": instruction.offset,
                        "kind": "local",
                        "local_index": local_index,
                        "reason": "local_name_index_out_of_range",
                    }
                )
            else:
                callee = local_names[local_index]
                if callee not in known_functions:
                    broken.append(
                        {
                            "function": function,
                            "offset": instruction.offset,
                            "kind": "local",
                            "callee": callee,
                            "reason": "unknown_function_boundary",
                        }
                    )
                else:
                    calls.append(
                        CallSite(
                            function=function,
                            kind="local",
                            offset=instruction.offset,
                            instruction_index=len(effective),
                            callee=callee,
                            branch_number=(
                                active_branch[0] if active_branch else None
                            ),
                            branch_offset=(
                                active_branch[1] if active_branch else None
                            ),
                        )
                    )
        elif instruction.opcode == OP_CALL_IMPORT:
            local_index = instruction.value
            if local_index >= len(imports):
                broken.append(
                    {
                        "function": function,
                        "offset": instruction.offset,
                        "kind": "import",
                        "import_index": local_index,
                        "reason": "import_index_out_of_range",
                    }
                )
            else:
                import_symbol = imports[local_index]
                calls.append(
                    CallSite(
                        function=function,
                        kind="import",
                        offset=instruction.offset,
                        instruction_index=len(effective),
                        import_symbol=import_symbol,
                        branch_number=(
                            active_branch[0] if active_branch else None
                        ),
                        branch_offset=(
                            active_branch[1] if active_branch else None
                        ),
                    )
                )
        elif instruction.opcode == OP_JCC:
            branch_count += 1
            active_branch = (
                instruction.branch_number
                if instruction.branch_number is not None
                else branch_count - 1,
                instruction.offset,
            )
        elif instruction.opcode == OP_RET:
            active_branch = None

        kind, signature = _instruction_signature(
            instruction.opcode,
            instruction.value,
            instruction.branch_number,
            callee_name=callee,
            import_symbol=import_symbol,
            local_index=local_index,
        )
        effective.append(
            EffectiveInstruction(
                offset=instruction.offset,
                kind=kind,
                signature=signature,
                opcode=instruction.opcode,
            )
        )

    calls.sort(key=lambda call: (call.offset, call.instruction_index))
    broken.sort(
        key=lambda item: (
            int(item.get("offset", 0)),
            str(item.get("reason", "")),
        )
    )
    return ParsedFunction(
        function=function,
        invalid_bytes=decoded.invalid_bytes,
        parse_issues=decoded.parse_issues,
        branch_count=branch_count,
        calls=calls,
        broken_calls=broken,
        effective=effective,
    )


def analyze_calls(image: PeImage, manifest: Manifest) -> CallAnalysis:
    parsed: dict[str, ParsedFunction] = {}
    known_functions = set(manifest.regions)
    for name in sorted(manifest.regions):
        region = manifest.regions[name]
        body = region_bytes(image, region)
        parsed[name] = parse_function(
            body=body,
            function=name,
            local_names=image.local_names,
            imports=image.imports,
            known_functions=known_functions,
        )
    return CallAnalysis(manifest, image.imports, parsed)
