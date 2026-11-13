from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .disasm import (
    Instruction,
    Operand,
    _canonical_register,
    decode_function,
    resolve_call_target,
)
from .pe import PEInfo
from .pdb import FunctionSymbol, PDBInfo


Taint = frozenset


API_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "memcpy": {"params": 3, "convention": "cdecl", "taintable": [0, 1, 2]},
    "strcpy": {"params": 2, "convention": "cdecl", "taintable": [0, 1]},
    "CreateProcessW": {"params": 10, "convention": "stdcall", "taintable": [0, 1, 2, 3, 4, 5, 6, 7, 8]},
    "WinExec": {"params": 2, "convention": "stdcall", "taintable": [0]},
    "LoadLibraryW": {"params": 1, "convention": "stdcall", "taintable": [0]},
    "CreateFileW": {"params": 7, "convention": "stdcall", "taintable": [0]},
}
DEFAULT_DANGEROUS = [
    "memcpy", "strcpy", "CreateProcessW", "WinExec", "LoadLibraryW", "CreateFileW"
]


@dataclass
class CallEdge:
    call_rva: int
    kind: str
    callee_name: str
    callee_rva: Optional[int]
    args: List[Taint]


@dataclass
class FunctionSummary:
    function: FunctionSymbol
    edges: List[CallEdge] = field(default_factory=list)
    return_taint: Taint = field(default_factory=frozenset)
    decode_error: Optional[str] = None


@dataclass
class SinkSite:
    symbol: str
    apis: Tuple[str, ...]
    call_rva: int
    taintable: Tuple[int, ...]


@dataclass
class TaintHit:
    source: str
    path: List[str]
    tainted_params: Tuple[int, ...]
    call_rva: int


def _empty() -> Taint:
    return frozenset()


def _union(*values: Iterable[int]) -> Taint:
    result: Set[int] = set()
    for value in values:
        result.update(value)
    return frozenset(result)


def _canon_reg(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return _canonical_register(name)


class FunctionEmulator:
    def __init__(
        self,
        pe: PEInfo,
        pdb: PDBInfo,
        functions_by_rva: Dict[int, FunctionSymbol],
        dangerous: Set[str],
        sanitizers: Set[str],
    ):
        self.pe = pe
        self.pdb = pdb
        self.is64 = pe.magic == 0x20B
        self.functions_by_rva = functions_by_rva
        self.dangerous = dangerous
        self.sanitizers = sanitizers
        self.unknown_propagation = 0
        self.recursion_warnings: List[Dict[str, Any]] = []
        self.summaries: Dict[int, FunctionSummary] = {}

    def ensure_summary(
        self,
        function: FunctionSymbol,
        stack: Tuple[int, ...] = (),
    ) -> Optional[FunctionSummary]:
        existing = self.summaries.get(function.rva)
        if existing is not None:
            return existing
        if function.rva in stack:
            self.recursion_warnings.append(
                {
                    "code": "RECURSION_CUT",
                    "function": function.name,
                    "message": "recursive call graph edge was pruned",
                }
            )
            return None
        if len(stack) >= 8:
            return None

        instructions, decode_error = decode_function(
            self.pe, function.rva, function.size, self.is64
        )
        for instruction in instructions:
            if instruction.mnemonic != "call":
                continue
            resolved = self._resolve_instruction_call(instruction)
            if resolved is None:
                continue
            kind, callee = resolved
            if kind == "function" and callee.name not in self.dangerous and callee.name not in self.sanitizers:
                if callee.rva in stack:
                    self.recursion_warnings.append(
                        {
                            "code": "RECURSION_CUT",
                            "function": function.name,
                            "message": "recursive call graph edge was pruned",
                        }
                    )
                    continue
                self.ensure_summary(callee, stack + (function.rva,))

        summary = self._emulate(function, instructions, decode_error)
        self.summaries[function.rva] = summary
        return summary

    def _resolve_instruction_call(self, instruction: Instruction):
        if instruction.op1 is None:
            return None
        if instruction.op1.kind == "imm":
            return resolve_call_target(
                self.pe, instruction.op1.value, self.functions_by_rva
            )
        if instruction.op1.kind == "mem" and instruction.op1.rip_relative:
            slot_rva = instruction.rva + instruction.size + instruction.op1.disp
            if slot_rva in self.pe.import_by_iat:
                return ("import", self.pe.import_by_iat[slot_rva])
        if (
            instruction.op1.kind == "mem"
            and instruction.op1.rip_relative is False
            and instruction.op1.base is None
            and instruction.op1.index is None
        ):
            candidates = [instruction.op1.disp]
            if instruction.op1.disp >= self.pe.image_base:
                candidates.append(instruction.op1.disp - self.pe.image_base)
            for candidate in candidates:
                if candidate in self.pe.import_by_iat:
                    return ("import", self.pe.import_by_iat[candidate])
        return None

    def _stack_address(self, operand: Operand, addresses: Dict[str, Optional[int]]) -> Optional[int]:
        if operand.kind != "mem" or operand.rip_relative or operand.index is not None:
            return None
        base = _canon_reg(operand.base)
        if base not in ("rsp", "rbp"):
            return None
        value = addresses.get(base)
        if value is None:
            return None
        return value + operand.disp

    def _read_operand(
        self,
        operand: Optional[Operand],
        registers: Dict[str, Taint],
        memory: Dict[int, Taint],
        addresses: Dict[str, Optional[int]],
    ) -> Taint:
        if operand is None:
            return _empty()
        if operand.kind == "imm":
            return _empty()
        if operand.kind == "reg":
            return registers.get(_canon_reg(operand.reg) or "", _empty())
        if operand.kind == "mem":
            key = self._stack_address(operand, addresses)
            if key is not None:
                return memory.get(key, _empty())
            base = _canon_reg(operand.base)
            index = _canon_reg(operand.index)
            if base is not None and base in registers:
                return registers[base]
            if index is not None and index in registers:
                return registers[index]
        return _empty()

    def _write_operand(
        self,
        operand: Optional[Operand],
        taint: Taint,
        registers: Dict[str, Taint],
        memory: Dict[int, Taint],
        addresses: Dict[str, Optional[int]],
    ) -> None:
        if operand is None:
            return
        if operand.kind == "reg":
            registers[_canon_reg(operand.reg) or ""] = taint
        elif operand.kind == "mem":
            key = self._stack_address(operand, addresses)
            if key is not None:
                memory[key] = taint

    def _argument_values(
        self,
        registers: Dict[str, Taint],
        memory: Dict[int, Taint],
        addresses: Dict[str, Optional[int]],
        count: int,
    ) -> List[Taint]:
        result: List[Taint] = []
        rsp = addresses.get("rsp")
        if self.is64:
            arg_regs = ["rcx", "rdx", "r8", "r9"]
            for index in range(count):
                if index < 4:
                    result.append(registers.get(arg_regs[index], _empty()))
                else:
                    key = None if rsp is None else rsp + 0x20 + 8 * (index - 4)
                    result.append(memory.get(key, _empty()) if key is not None else _empty())
        else:
            for index in range(count):
                key = None if rsp is None else rsp + 4 * index
                result.append(memory.get(key, _empty()) if key is not None else _empty())
        return result

    def _emulate(
        self,
        function: FunctionSymbol,
        instructions: List[Instruction],
        decode_error: Optional[str],
    ) -> FunctionSummary:
        registers: Dict[str, Taint] = {}
        memory: Dict[int, Taint] = {}
        addresses: Dict[str, Optional[int]] = {"rsp": 0, "rbp": None}
        word = 8 if self.is64 else 4
        param_count = max(8, function.param_count or 0)
        if self.is64:
            for index, reg in enumerate(["rcx", "rdx", "r8", "r9"]):
                registers[reg] = frozenset({index})
            for index in range(4, param_count):
                memory[0x28 + word * (index - 4)] = frozenset({index})
        else:
            for index in range(param_count):
                memory[4 + word * index] = frozenset({index})
        for reg in (
            "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"
        ):
            registers.setdefault(reg, _empty())

        summary = FunctionSummary(function=function, decode_error=decode_error)
        for instruction in instructions:
            mnemonic = instruction.mnemonic
            if mnemonic == "mov":
                source = self._read_operand(instruction.op2, registers, memory, addresses)
                self._write_operand(instruction.op1, source, registers, memory, addresses)
                dest = instruction.op1
                src = instruction.op2
                if dest and dest.kind == "reg" and src and src.kind == "reg":
                    dest_reg = _canon_reg(dest.reg)
                    src_reg = _canon_reg(src.reg)
                    if dest_reg in addresses and src_reg in addresses:
                        addresses[dest_reg] = addresses[src_reg]
                elif dest and dest.kind == "reg" and src and src.kind == "imm":
                    dest_reg = _canon_reg(dest.reg)
                    if dest_reg in addresses:
                        addresses[dest_reg] = src.value
            elif mnemonic == "lea":
                value = _empty()
                if instruction.op2 is not None:
                    if instruction.op2.base:
                        value = _union(value, registers.get(_canon_reg(instruction.op2.base) or "", _empty()))
                    if instruction.op2.index:
                        value = _union(value, registers.get(_canon_reg(instruction.op2.index) or "", _empty()))
                self._write_operand(instruction.op1, value, registers, memory, addresses)
                if (
                    instruction.op1 is not None
                    and instruction.op1.kind == "reg"
                    and instruction.op2 is not None
                    and instruction.op2.kind == "mem"
                ):
                    key = self._stack_address(instruction.op2, addresses)
                    reg = _canon_reg(instruction.op1.reg)
                    if key is not None and reg in addresses:
                        addresses[reg] = key
            elif mnemonic == "push":
                addresses["rsp"] = (addresses.get("rsp") or 0) - word
                value = self._read_operand(instruction.op1, registers, memory, addresses)
                memory[addresses["rsp"]] = value
            elif mnemonic == "pop":
                rsp = addresses.get("rsp")
                value = memory.get(rsp or 0, _empty()) if rsp is not None else _empty()
                self._write_operand(instruction.op1, value, registers, memory, addresses)
                addresses["rsp"] = (rsp or 0) + word
            elif mnemonic in ("add", "sub"):
                dest = instruction.op1
                src = instruction.op2
                if dest is not None and dest.kind == "reg":
                    dest_reg = _canon_reg(dest.reg)
                    if dest_reg in addresses and src is not None and src.kind == "imm":
                        delta = src.value if src.value is not None else 0
                        addresses[dest_reg] = (addresses.get(dest_reg) or 0) + (delta if mnemonic == "add" else -delta)
                    else:
                        old = self._read_operand(dest, registers, memory, addresses)
                        new = _union(old, self._read_operand(src, registers, memory, addresses))
                        registers[dest_reg or ""] = new
            elif mnemonic in ("and", "or", "xor", "adc", "sbb", "imul"):
                dest = instruction.op1
                if dest is not None and dest.kind == "reg":
                    dest_reg = _canon_reg(dest.reg)
                    src = instruction.op2
                    same = (
                        src is not None
                        and src.kind == "reg"
                        and _canon_reg(src.reg) == dest_reg
                        and mnemonic == "xor"
                    )
                    if same:
                        registers[dest_reg or ""] = _empty()
                    else:
                        old = self._read_operand(dest, registers, memory, addresses)
                        new = _union(old, self._read_operand(src, registers, memory, addresses))
                        registers[dest_reg or ""] = new
            elif mnemonic == "movsxd":
                self._write_operand(
                    instruction.op2,
                    self._read_operand(instruction.op1, registers, memory, addresses),
                    registers,
                    memory,
                    addresses,
                )
            elif mnemonic == "call":
                resolved = self._resolve_instruction_call(instruction)
                callee_name = ""
                callee_rva = None
                kind = "unknown"
                if resolved is not None:
                    kind, info = resolved
                    if kind == "function":
                        callee_name = info.name
                        callee_rva = info.rva
                        if callee_name in self.dangerous:
                            kind = "sink"
                    else:
                        callee_name = info
                        if callee_name in self.dangerous:
                            kind = "sink"
                if kind == "sink":
                    arg_count = API_SIGNATURES.get(callee_name, {}).get("params", 8)
                elif kind == "function":
                    child = self.summaries.get(callee_rva or -1)
                    arg_count = max(
                        child.function.param_count if child else 0,
                        info.param_count or 0,
                        4 if self.is64 else 0,
                    )
                else:
                    arg_count = 4 if self.is64 else 8
                args = self._argument_values(registers, memory, addresses, arg_count)
                edge = CallEdge(
                    call_rva=instruction.rva,
                    kind=kind,
                    callee_name=callee_name,
                    callee_rva=callee_rva,
                    args=args,
                )
                summary.edges.append(edge)
                if kind == "function" and callee_name not in self.sanitizers:
                    child = self.summaries.get(callee_rva or -1)
                    if child is not None:
                        ret = _empty()
                        for dependency in child.return_taint:
                            if dependency < len(args):
                                ret = _union(ret, args[dependency])
                        registers["rax"] = ret
                    else:
                        registers["rax"] = _empty()
                elif callee_name in self.sanitizers:
                    registers["rax"] = _empty()
                else:
                    registers["rax"] = _empty()
                if any(args):
                    if kind == "unknown":
                        self.unknown_propagation += 1
                    elif kind == "function":
                        child = self.summaries.get(callee_rva or -1)
                        if child is None:
                            self.unknown_propagation += 1
                volatile = (
                    ["rcx", "rdx", "r8", "r9", "r10", "r11"]
                    if self.is64
                    else ["rcx", "rdx"]
                )
                for reg in volatile:
                    registers[reg] = _empty()
        summary.return_taint = registers.get("rax", _empty())
        return summary


def analyze(
    pe: PEInfo,
    pdb: PDBInfo,
    config: Dict[str, Any],
    start_time: Optional[float] = None,
) -> Dict[str, Any]:
    started = time.monotonic() if start_time is None else start_time
    sources_raw = config.get("sources", [])
    if not isinstance(sources_raw, list):
        raise ValueError("config.sources must be a list")
    dangerous = set(DEFAULT_DANGEROUS)
    for name in config.get("dangerous_apis", []) or []:
        if isinstance(name, str):
            dangerous.add(name)
    sanitizers = {"sanitize"}
    sanitizers.update(
        str(name) for name in (config.get("sanitizers", []) or []) if isinstance(name, str)
    )
    function_by_name = {}
    functions_by_rva: Dict[int, FunctionSymbol] = {}
    for function in pdb.functions:
        function_by_name.setdefault(function.name, function)
        functions_by_rva.setdefault(function.rva, function)
    for public in pdb.public_symbols:
        function_by_name.setdefault(public.name, FunctionSymbol(
            name=public.name,
            raw_name=public.raw_name,
            rva=public.rva,
            size=0,
            segment=public.segment,
            offset=public.offset,
            type_index=0,
            module="public",
            kind="S_PUB32",
        ))
    exports_by_name = {item["name"]: item for item in pe.exports}

    warnings: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    source_functions: List[FunctionSymbol] = []
    for ordinal, source in enumerate(sources_raw):
        if not isinstance(source, dict) or not isinstance(source.get("function"), str):
            errors.append({"code": "INVALID_CONFIG", "message": "invalid source entry"})
            continue
        name = source["function"]
        try:
            param_index = int(source.get("param_index", 0))
        except (TypeError, ValueError):
            errors.append({"code": "INVALID_CONFIG", "message": "invalid param_index for %s" % name})
            continue
        function = function_by_name.get(name)
        if function is not None and function.rva not in functions_by_rva:
            functions_by_rva[function.rva] = function
        if function is None and name in exports_by_name:
            export_rva = exports_by_name[name]["rva"]
            for candidate in pdb.functions:
                if candidate.rva <= export_rva < candidate.rva + max(1, candidate.size):
                    function = candidate
                    break
            if function is None:
                covering = None
                for candidate in pe.functions:
                    if candidate["rva"] <= export_rva < candidate["end_rva"]:
                        covering = candidate
                        break
                if covering is not None:
                    function = FunctionSymbol(
                        name=name,
                        raw_name=name,
                        rva=covering["rva"],
                        size=covering["size"],
                        segment=0,
                        offset=0,
                        type_index=0,
                        module="export",
                        kind="EXPORT",
                    )
                    functions_by_rva[function.rva] = function
        if function is None:
            warnings.append(
                {
                    "code": "SOURCE_NOT_FOUND",
                    "function": name,
                    "message": "source function is absent from both PDB functions and PE exports",
                }
            )
            continue
        if param_index < 0:
            errors.append({"code": "INVALID_CONFIG", "message": "negative param_index for %s" % name})
            continue
        if function.param_count is not None and param_index >= function.param_count:
            errors.append(
                {
                    "code": "INVALID_CONFIG",
                    "function": name,
                    "message": "param_index %d exceeds declared parameter count %d"
                    % (param_index, function.param_count),
                }
            )
            continue
        sources.append(
            {
                "ordinal": ordinal,
                "name": name,
                "function": function,
                "param_index": param_index,
                "type": source.get("type", "ptr"),
            }
        )
        source_functions.append(function)

    user_modules = {source["function"].module for source in sources}
    hints = config.get("module_hints", []) or []
    if isinstance(hints, str):
        hints = [hints]
    for hint in hints:
        if not isinstance(hint, str):
            continue
        for function in pdb.functions:
            if hint.lower() in function.module.lower():
                user_modules.add(function.module)
    user_functions = [
        function for function in pdb.functions if function.module in user_modules
    ]
    if not user_functions and function_by_name:
        for function in pdb.functions:
            if function.name in {item.get("function") for item in sources_raw if isinstance(item, dict)}:
                user_functions.append(function)
                user_modules.add(function.module)
    user_functions.sort(key=lambda item: (item.rva, item.name))
    function_scan_truncated = max(0, len(user_functions) - 20000)

    emulator = FunctionEmulator(pe, pdb, functions_by_rva, dangerous, sanitizers)
    for function in user_functions[:20000]:
        emulator.ensure_summary(function)
    warnings.extend(emulator.recursion_warnings)

    summaries = {
        rva: summary
        for rva, summary in emulator.summaries.items()
        if summary.function.module in user_modules
    }
    user_names = {summary.function.name for summary in summaries.values()}

    call_edges_by_function = {
        summary.function.name: [
            edge for edge in summary.edges
            if edge.kind == "function" and edge.callee_name in user_names
        ]
        for summary in summaries.values()
    }

    def structural_paths(source: Dict[str, Any]) -> Dict[str, List[str]]:
        start = source["function"].name
        queue = deque([(start, [start], 0)])
        best: Dict[str, List[str]] = {start: [start]}
        while queue:
            current, path, depth = queue.popleft()
            if depth >= 3:
                continue
            neighbors = sorted(
                {edge.callee_name for edge in call_edges_by_function.get(current, [])}
            )
            for neighbor in neighbors:
                candidate = path + [neighbor]
                previous = best.get(neighbor)
                if previous is None or (len(candidate), candidate) < (len(previous), previous):
                    best[neighbor] = candidate
                    queue.append((neighbor, candidate, depth + 1))
        return best

    structural_by_source = {
        source["name"]: structural_paths(source) for source in sources
    }

    sink_map: Dict[str, SinkSite] = {}
    for summary in sorted(summaries.values(), key=lambda item: item.function.rva):
        for edge in summary.edges:
            if edge.kind != "sink":
                continue
            signature = API_SIGNATURES.get(edge.callee_name, {})
            taintable = tuple(signature.get("taintable", range(max(1, len(edge.args)))))
            previous = sink_map.get(summary.function.name)
            if previous is None:
                sink_map[summary.function.name] = SinkSite(
                    symbol=summary.function.name,
                    apis=(edge.callee_name,),
                    call_rva=edge.call_rva,
                    taintable=taintable,
                )
            else:
                sink_map[summary.function.name] = SinkSite(
                    symbol=previous.symbol,
                    apis=tuple(sorted(set(previous.apis + (edge.callee_name,)))), 
                    call_rva=min(previous.call_rva, edge.call_rva),
                    taintable=tuple(sorted(set(previous.taintable + taintable))),
                )

    hits: Dict[str, List[TaintHit]] = {}
    evaluated_paths = 0
    truncated_count = 0
    for source in sources:
        source_name = source["name"]
        function = source["function"]
        start_taints = {function.name: {source["param_index"]: frozenset({source["ordinal"]})}}
        queue = deque([(function.name, start_taints[function.name], [function.name], 0)])
        visited = set()
        while queue:
            fn_name, param_taints, path, depth = queue.popleft()
            key = (fn_name, tuple((i, frozenset(v)) for i, v in sorted(param_taints.items()) if v))
            if key in visited:
                continue
            visited.add(key)
            summary = next(
                (item for item in summaries.values() if item.function.name == fn_name),
                None,
            )
            if summary is None:
                continue
            for edge in summary.edges:
                evaluated_paths += 1
                projected: List[Set[int]] = []
                for dependencies in edge.args:
                    values: Set[int] = set()
                    for dependency in dependencies:
                        values.update(param_taints.get(dependency, frozenset()))
                    projected.append(values)
                if edge.kind == "sink":
                    site = sink_map.get(summary.function.name)
                    taintable = site.taintable if site else tuple(range(len(projected)))
                    tainted_params = tuple(
                        index
                        for index, values in enumerate(projected)
                        if values and index in taintable
                    )
                    if tainted_params:
                        hits.setdefault(summary.function.name, []).append(
                            TaintHit(
                                source=source_name,
                                path=path,
                                tainted_params=tainted_params,
                                call_rva=edge.call_rva,
                            )
                        )
                elif edge.kind == "function" and edge.callee_name not in sanitizers:
                    next_taints: Dict[int, Set[int]] = {}
                    for index, values in enumerate(projected):
                        if values:
                            next_taints[index] = values
                    if next_taints:
                        if depth >= 3:
                            truncated_count += 1
                            continue
                        next_path = path + [edge.callee_name]
                        queue.append((edge.callee_name, next_taints, next_path, depth + 1))
                elif edge.kind == "unknown":
                    if any(projected):
                        emulator.unknown_propagation += 1

    function_map = {}
    for summary in summaries.values():
        function_map.setdefault(summary.function.name, summary.function)
    for function in source_functions:
        function_map.setdefault(function.name, function)

    sinks = []
    for symbol, site in sorted(sink_map.items()):
        candidates = hits.get(symbol, [])
        structural_choice = None
        structural_source = None
        for source in sources:
            path = structural_by_source.get(source["name"], {}).get(symbol)
            if path is None:
                continue
            candidate = (len(path), tuple(path), source["name"])
            if structural_choice is None or candidate < structural_choice[0]:
                structural_choice = (candidate, path)
                structural_source = source["name"]
        if candidates:
            ordered = sorted(
                candidates,
                key=lambda hit: (len(hit.path), tuple(hit.path), hit.source),
            )
            chosen = ordered[0]
            path = chosen.path
            chosen_source = chosen.source
            tainted_params = sorted(
                {index for hit in candidates for index in hit.tainted_params}
            )
            tainted_sources = sorted({hit.source for hit in candidates})
            path_has_sanitizer = any(name in sanitizers for name in path)
            if len(path) <= 2 and not path_has_sanitizer:
                exploitability = "DIRECT"
            else:
                exploitability = "CONDITIONAL"
        else:
            chosen = None
            chosen_source = structural_source
            path = structural_choice[1] if structural_choice else []
            tainted_params = []
            tainted_sources = []
            exploitability = "UNREACHABLE"
        path_rvas = []
        for name in path:
            function = function_map.get(name)
            path_rvas.append(function.rva if function is not None else 0)
        path_items = [
            {"name": name, "rva": path_rvas[index]} for index, name in enumerate(path)
        ]
        sinks.append(
            {
                "symbol": symbol,
                "api": site.apis[0],
                "apis": list(site.apis),
                "call_rva": site.call_rva,
                "exploitability": exploitability,
                "tainted_params": tainted_params,
                "tainted_sources": tainted_sources,
                "shortest_path_source": chosen_source,
                "path": path,
                "shortest_path": path,
                "path_rvas": path_rvas,
                "path_items": path_items,
                "shortest_path_items": path_items,
            }
        )

    import_names = {item["name"] for item in pe.imports if item.get("name")}
    for function in pdb.functions:
        if function.name in import_names:
            warnings.append(
                {
                    "code": "NAME_CONFLICT",
                    "function": function.name,
                    "pdb_name": function.name,
                    "import_name": function.name,
                    "message": "PDB function name conflicts with an imported function name",
                }
            )

    if pe.magic == 0x10B:
        for function in pdb.functions:
            if function.module not in user_modules:
                continue
            signature = API_SIGNATURES.get(function.name)
            if signature is None:
                continue
            actual_params = function.param_count
            actual_conv = function.convention
            if (
                (actual_params is not None and actual_params != signature["params"])
                or (actual_conv is not None and actual_conv != signature["convention"])
            ):
                warnings.append(
                    {
                        "code": "CONV_MISMATCH",
                        "function": function.name,
                        "expected_params": signature["params"],
                        "actual_params": actual_params,
                        "expected_convention": signature["convention"],
                        "actual_convention": actual_conv,
                        "message": "PDB signature disagrees with the built-in Windows API table",
                    }
                )
    else:
        for function in pdb.functions:
            if function.module not in user_modules:
                continue
            signature = API_SIGNATURES.get(function.name)
            if signature is not None and function.param_count is not None and function.param_count != signature["params"]:
                warnings.append(
                    {
                        "code": "CONV_MISMATCH",
                        "function": function.name,
                        "expected_params": signature["params"],
                        "actual_params": function.param_count,
                        "expected_convention": signature["convention"],
                        "actual_convention": function.convention,
                        "message": "PDB parameter count disagrees with the built-in Windows API table",
                    }
                )

    warnings = _dedupe_sorted(warnings)
    errors = _dedupe_sorted(errors)
    elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
    passed = not errors
    return {
        "schema_version": "1.0",
        "pe_info": _pe_info(pe),
        "pdb_info": _pdb_info(pdb),
        "analysis_stats": {
            "scanned_functions": len(summaries),
            "evaluated_paths": evaluated_paths,
            "elapsed_ms": elapsed_ms,
            "truncated_count": truncated_count + function_scan_truncated,
            "unknown_propagation": int(emulator.unknown_propagation),
        },
        "sinks": sinks,
        "warnings": warnings,
        "errors": errors,
        "passed": passed,
    }


def _dedupe_sorted(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        unique[key] = item
    return [unique[key] for key in sorted(unique)]


def _pe_info(pe: PEInfo) -> Dict[str, Any]:
    return {
        "machine": pe.machine,
        "magic": "PE32+" if pe.is_pe32_plus else "PE32",
        "image_base": pe.image_base,
        "entry_point_rva": pe.entry_point,
        "size_of_image": pe.size_of_image,
        "timestamp": pe.timestamp,
        "sha256": hashlib.sha256(pe.data).hexdigest(),
        "sections": [
            {
                "name": section.name,
                "rva": section.rva,
                "virtual_size": section.virtual_size,
                "raw_size": section.raw_size,
                "raw_ptr": section.raw_ptr,
                "characteristics": section.characteristics,
            }
            for section in pe.sections
        ],
        "imports": list(pe.imports),
        "exports": list(pe.exports),
        "function_count": len(pe.functions),
    }


def _pdb_info(pdb: PDBInfo) -> Dict[str, Any]:
    return {
        "guid": pdb.guid,
        "age": pdb.age,
        "signature": pdb.signature,
        "block_size": pdb.block_size,
        "block_count": pdb.block_count,
        "stream_count": pdb.stream_count,
        "module_count": pdb.module_count,
        "function_count": len(pdb.functions),
        "public_symbol_count": len(pdb.public_symbols),
    }


def diff_reports(
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    before_sinks = {item["symbol"]: item for item in before.get("sinks", [])}
    after_sinks = {item["symbol"]: item for item in after.get("sinks", [])}
    improved = []
    regressed = []
    unchanged = []
    rank = {"UNREACHABLE": 0, "CONDITIONAL": 1, "DIRECT": 2}
    for symbol in sorted(set(before_sinks) | set(after_sinks)):
        b = before_sinks.get(symbol)
        a = after_sinks.get(symbol)
        if b is None:
            b_null = {
                "symbol": symbol,
                "exploitability": "UNREACHABLE",
                "path": [],
                "tainted_params": [],
            }
            b_expl = "UNREACHABLE"
        else:
            b_null = b
            b_expl = b.get("exploitability", "UNREACHABLE")
        if a is None:
            a_null = {
                "symbol": symbol,
                "exploitability": "UNREACHABLE",
                "path": [],
                "tainted_params": [],
            }
            a_expl = "UNREACHABLE"
        else:
            a_null = a
            a_expl = a.get("exploitability", "UNREACHABLE")
        record = {
            "symbol": symbol,
            "before": {
                "exploitability": b_expl,
                "path": b_null.get("path", []),
                "tainted_params": b_null.get("tainted_params", []),
            },
            "after": {
                "exploitability": a_expl,
                "path": a_null.get("path", []),
                "tainted_params": a_null.get("tainted_params", []),
            },
        }
        if rank[a_expl] < rank[b_expl]:
            record["evidence"] = "exploitability decreased"
            improved.append(record)
        elif rank[a_expl] > rank[b_expl]:
            record["evidence"] = "exploitability increased"
            regressed.append(record)
        else:
            unchanged.append(record)
    errors = list(before.get("errors", [])) + list(after.get("errors", []))
    warnings = list(before.get("warnings", [])) + list(after.get("warnings", []))
    stats = dict(after.get("analysis_stats", {}))
    stats["before_evaluated_paths"] = before.get("analysis_stats", {}).get("evaluated_paths", 0)
    return {
        "schema_version": "1.0",
        "analysis_stats": stats,
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "warnings": _dedupe_sorted(warnings),
        "errors": _dedupe_sorted(errors),
        "passed": not regressed and not errors,
    }

