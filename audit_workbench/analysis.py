"""Interprocedural reachability and taint analysis for parsed sample roots."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Iterable

from .cparse import (
    Assignment,
    CallRecord,
    CToken,
    DANGEROUS_RULES,
    FunctionRecord,
    ParsedUnit,
)
from .config import AuditConfig
from .hashing import canonical_json


SOURCE_FUNCTIONS = {
    "read_input",
    "recv_from_socket",
    "getenv",
    "getopt",
    "readline",
    "recv",
}

RELEVANT_ARGS = {
    "strcpy": [1],
    "strcat": [1],
    "sprintf": [1, 2, 3, 4, 5, 6, 7, 8],
    "vsprintf": [1, 2, 3, 4, 5, 6, 7, 8],
    "memcpy": [1],
    "memmove": [1],
    "system": [0],
    "gets": [0],
    "scanf": [0, 1, 2, 3, 4, 5, 6, 7, 8],
}


def _merge_state(left: str, right: str) -> str:
    if left == "tainted" or right == "tainted":
        return "tainted"
    if left == "unknown" or right == "unknown":
        return "unknown"
    return "untainted"


class TaintEngine:
    def __init__(self, unit: ParsedUnit) -> None:
        self.unit = unit
        self.param_states: dict[str, dict[str, str]] = {
            name: {param: "untainted" for param in function.params}
            for name, function in unit.functions.items()
        }
        self._return_cache: dict[tuple[str, tuple[tuple[str, str], ...]], str] = {}

    @staticmethod
    def _tokens_text(tokens: Iterable[CToken]) -> str:
        return "".join(token.text for token in tokens)

    @staticmethod
    def _has_unresolved(tokens: Iterable[CToken]) -> bool:
        return any("__AUDIT_UNRESOLVED_MACRO" in token.text for token in tokens)

    @staticmethod
    def _is_literal(tokens: list[CToken]) -> bool:
        if not tokens:
            return False
        text = "".join(token.text for token in tokens)
        return bool(
            __import__("re").fullmatch(
                r'(?:(?:L?)"(?:\\.|[^"\\])*"|L?\'(?:\\.|[^\'\\])*\'|0[xX][0-9a-fA-F]+|[0-9]+(?:\.[0-9]+)?|\-?[0-9]+)',
                text,
            )
        )

    def _param_state(self, function: FunctionRecord, name: str) -> str:
        return self.param_states.get(function.name, {}).get(name, "unknown")

    def _assignment_state(
        self,
        function: FunctionRecord,
        assignment: Assignment,
        seen: set[str],
    ) -> str:
        if assignment.lhs in seen:
            return "unknown"
        return self.eval_tokens(
            function,
            assignment.rhs_tokens,
            assignment.relative_file,
            seen | {assignment.lhs},
        )

    def _variable_state(
        self,
        function: FunctionRecord,
        name: str,
        seen: set[str],
    ) -> str:
        if name in function.params:
            return self._param_state(function, name)
        assignments = [
            assignment for assignment in function.assignments if assignment.lhs == name
        ]
        if not assignments:
            return "unknown"
        state = "untainted"
        for assignment in assignments:
            state = _merge_state(
                state,
                self._assignment_state(function, assignment, seen),
            )
        return state

    def _return_state(
        self,
        function: FunctionRecord,
        seen: set[str],
    ) -> str:
        cache_key = (
            function.name,
            tuple(sorted(self.param_states.get(function.name, {}).items())),
        )
        cached = self._return_cache.get(cache_key)
        if cached is not None:
            return cached
        if not function.returns:
            return "unknown"
        state = "untainted"
        for expression in function.returns:
            state = _merge_state(
                state,
                self.eval_tokens(function, expression, function.relative_file, seen),
            )
        self._return_cache[cache_key] = state
        return state

    def _call_state(
        self,
        function: FunctionRecord,
        callee: str,
        args: list[list[CToken]],
        relative_file: str,
        seen: set[str],
    ) -> str:
        if callee in SOURCE_FUNCTIONS:
            return "tainted"
        target = self.unit.functions.get(callee)
        if target is not None:
            if target.name in seen:
                return "unknown"
            # Parameter state for this specific call is captured by the
            # interprocedural fixed point.  Return-value propagation is a
            # conservative one-level extension of that same model.
            return self._return_state(target, seen | {target.name})
        return "unknown"

    def eval_tokens(
        self,
        function: FunctionRecord,
        tokens: list[CToken],
        relative_file: str,
        seen: set[str] | None = None,
    ) -> str:
        seen = seen or set()
        if not tokens:
            return "untainted"
        if self._has_unresolved(tokens):
            return "unknown"
        if self._is_literal(tokens):
            return "untainted"

        # Strip one or more simple C casts and redundant outer parentheses.
        while tokens and tokens[0].text == "(":
            close = None
            depth = 0
            for index, token in enumerate(tokens, 1):
                if token.text == "(":
                    depth += 1
                elif token.text == ")":
                    depth -= 1
                    if depth == 0:
                        close = index
                        break
            if close is None:
                break
            inner = tokens[1:close]
            type_words = {"char", "const", "void", "int", "long", "short", "unsigned", "signed", "struct", "*"}
            if inner and all(token.text in type_words for token in inner):
                tokens = tokens[close + 1 :]
                continue
            if close == len(tokens):
                tokens = inner
                continue
            break

        if tokens and tokens[0].text in {"&", "*", "+", "-", "!"}:
            if tokens[0].text in {"&", "*"} and len(tokens) > 1:
                return self.eval_tokens(function, tokens[1:], relative_file, seen)
            # Arithmetic is outside the supported data-flow subset.
            return "unknown"

        # argv[n] is a command-line source.
        if tokens[0].text in {"argv", "argc"}:
            return "tainted"

        first = tokens[0]
        if __import__("re").fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", first.text):
            name = first.text
            # Function call expression.
            if len(tokens) >= 3 and tokens[1].text == "(":
                depth = 0
                close_index = None
                for index in range(1, len(tokens)):
                    if tokens[index].text == "(":
                        depth += 1
                    elif tokens[index].text == ")":
                        depth -= 1
                        if depth == 0:
                            close_index = index
                            break
                if close_index is not None:
                    arg_tokens = tokens[2:close_index]
                    args: list[list[CToken]] = [[]]
                    depth = 0
                    for token in arg_tokens:
                        if token.text in {"(", "[", "{"}:
                            depth += 1
                        elif token.text in {")", "]", "}"}:
                            depth = max(0, depth - 1)
                        if token.text == "," and depth == 0:
                            args.append([])
                        else:
                            args[-1].append(token)
                    return self._call_state(
                        function,
                        name,
                        args,
                        relative_file,
                        seen,
                    )
            if name in {"NULL", "nullptr"}:
                return "untainted"
            # Variable-like expressions: identifier, index, or member access.
            state = self._variable_state(function, name, seen)
            if any(token.text in {"+", "-", "*", "/", "%", "&", "|", "^"} for token in tokens[1:]):
                if state == "tainted":
                    return "unknown"
            return state

        return "unknown"

    def refresh_parameters(self) -> None:
        # A deliberately small fixed point (one-level semantics are a subset):
        # each call contributes its caller-side argument state to callee params.
        for _ in range(max(1, len(self.unit.functions) * 2)):
            changed = False
            alias_targets: dict[str, set[str]] = defaultdict(set)
            for function in self.unit.functions.values():
                for variable, targets in function.alias_targets.items():
                    alias_targets[variable].update(
                        target for target in targets if target in self.unit.functions
                    )
            for caller_name, caller in self.unit.functions.items():
                for call in caller.calls:
                    targets: list[FunctionRecord] = []
                    if call.kind == "indirect_table":
                        table = self.unit.tables.get(call.table_name)
                        if table:
                            targets = [
                                self.unit.functions[name]
                                for name in sorted(table.targets)
                                if name in self.unit.functions
                            ]
                    else:
                        direct = self.unit.functions.get(call.callee)
                        if direct is not None:
                            targets = [direct]
                        elif call.callee in alias_targets:
                            targets = [
                                self.unit.functions[name]
                                for name in sorted(alias_targets[call.callee])
                            ]
                    for target in targets:
                        target_states = self.param_states.setdefault(
                            target.name,
                            {param: "untainted" for param in target.params},
                        )
                        for index, arg in enumerate(call.args):
                            if index >= len(target.params):
                                break
                            state = self.eval_tokens(caller, arg, call.relative_file)
                            param = target.params[index]
                            merged = _merge_state(target_states.get(param, "untainted"), state)
                            if merged != target_states.get(param):
                                target_states[param] = merged
                                changed = True
            self._return_cache.clear()
            if not changed:
                break

    def call_taint(self, call: CallRecord) -> dict[str, object]:
        function = self.unit.functions[call.enclosing_function]
        relevant = set(RELEVANT_ARGS.get(call.callee, range(len(call.args))))
        arg_states: list[dict[str, object]] = []
        overall = "untainted"
        reason_bits: list[str] = []
        for index, arg in enumerate(call.args):
            state = self.eval_tokens(function, arg, call.relative_file)
            is_relevant = index in relevant
            arg_states.append(
                {
                    "index": index,
                    "arg": self._tokens_text(arg),
                    "state": state,
                    "relevant": is_relevant,
                }
            )
            if is_relevant:
                overall = _merge_state(overall, state)
                if state == "tainted":
                    reason_bits.append(f"arg{index} traces to an external input source")
                elif state == "unknown":
                    reason_bits.append(f"arg{index} contains unresolved or unsupported syntax")
        if overall == "untainted":
            reason = "all relevant arguments are literals, local constants, or untainted locals"
        elif overall == "tainted":
            reason = "; ".join(reason_bits) or "external input reaches a relevant argument"
        else:
            reason = "; ".join(reason_bits) or "taint cannot be decided"
        return {
            "taint": overall,
            "taint_reason": reason,
            "argument_states": arg_states,
        }


def _dangerous(call: CallRecord) -> bool:
    return call.callee in DANGEROUS_RULES


def analyze(
    unit: ParsedUnit,
    config: AuditConfig,
    *,
    max_hops: int = 8,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    entries = sorted(set(config.entries))
    edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
    alias_targets: dict[str, set[str]] = defaultdict(set)
    for function in unit.functions.values():
        for variable, targets in function.alias_targets.items():
            alias_targets[variable].update(
                target for target in targets if target in unit.functions
            )
        for call in function.calls:
            if call.kind == "direct" and call.callee in unit.functions:
                target_file = unit.functions[call.callee].relative_file
                if target_file != function.relative_file and call.callee in unit.prototypes:
                    via = "cross-file-declaration-definition"
                else:
                    via = "direct-call"
                edges[function.name].append((call.callee, via))
            elif call.kind == "direct" and call.callee in alias_targets:
                for target in sorted(alias_targets[call.callee]):
                    edges[function.name].append(
                        (target, f"function-pointer-assignment:{call.callee}")
                    )
            elif call.kind == "indirect_table":
                table = unit.tables.get(call.table_name)
                if table:
                    for target in sorted(table.targets):
                        if target in unit.functions:
                            edges[function.name].append(
                                (target, f"function-pointer-table:{call.table_name}")
                            )
    for function_name in edges:
        edges[function_name] = sorted(set(edges[function_name]))

    # BFS with a deterministic shortest-path tie-break.
    reachable: set[str] = set()
    path_hops: dict[str, list[dict[str, str]]] = {}
    queue: deque[str] = deque()
    for entry in entries:
        if entry in unit.functions:
            reachable.add(entry)
            path_hops[entry] = []
            queue.append(entry)
    while queue:
        current = queue.popleft()
        for target, via in edges.get(current, []):
            candidate = path_hops[current] + [{"from": current, "to": target, "via": via}]
            if target not in reachable:
                reachable.add(target)
                path_hops[target] = candidate
                queue.append(target)
    present_entries = [entry for entry in entries if entry in unit.functions]
    missing_entries = [] if present_entries else list(entries)

    taint_engine = TaintEngine(unit)
    taint_engine.refresh_parameters()

    findings: list[dict[str, object]] = []
    for call in unit.calls:
        if not _dangerous(call):
            continue
        function = unit.functions[call.enclosing_function]
        if function.name not in reachable:
            reachability = "unreachable"
            reach_reason = "no path from the configured entry functions"
            path: list[dict[str, str]] = []
            truncated = False
        else:
            full_path = path_hops.get(function.name, [])
            truncated = len(full_path) > max_hops
            path = full_path[:max_hops]
            if call.kind == "indirect_unknown":
                reachability = "unknown"
                reach_reason = call.reason or "unresolved indirect call target"
            elif call.kind == "direct":
                reachability = "reachable"
                reach_reason = "direct dangerous call in a reachable function"
            elif call.kind == "indirect_table":
                reachability = "reachable"
                reach_reason = f"resolved function-pointer table {call.table_name}"
            else:
                reachability = "unknown"
                reach_reason = call.reason or "unresolved call target"

        taint = taint_engine.call_taint(call)
        high = reachability == "reachable" and taint["taint"] == "tainted"
        findings.append(
            {
                "id": call.callsite_id,
                "fingerprint": call.fingerprint,
                "callee": call.callee,
                "rule": DANGEROUS_RULES[call.callee],
                "function": call.enclosing_function,
                "location": call.location(),
                "reachability": reachability,
                "reachability_reason": reach_reason,
                "taint": taint["taint"],
                "taint_reason": taint["taint_reason"],
                "argument_states": taint["argument_states"],
                "high_risk": high,
                "path": path,
                "path_truncated": truncated,
                "truncated": truncated,
                "path_hops": len(path),
            }
        )

    findings.sort(
        key=lambda item: (
            item["location"]["file"],
            item["location"]["line"],
            item["location"]["column"],
            item["id"],
        )
    )
    counts = {
        "reachable": sum(1 for item in findings if item["reachability"] == "reachable"),
        "unreachable": sum(1 for item in findings if item["reachability"] == "unreachable"),
        "unknown": sum(1 for item in findings if item["reachability"] == "unknown"),
    }
    taint_counts = {
        "tainted": sum(1 for item in findings if item["taint"] == "tainted"),
        "untainted": sum(1 for item in findings if item["taint"] == "untainted"),
        "unknown": sum(1 for item in findings if item["taint"] == "unknown"),
    }
    summary = {
        "calls_total": len(findings),
        "reachability": counts,
        "taint": taint_counts,
        "high_risk": sum(1 for item in findings if item["high_risk"]),
        "entry_missing": missing_entries,
    }
    return findings, summary

