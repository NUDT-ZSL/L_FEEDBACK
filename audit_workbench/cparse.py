"""Lightweight C tokenizer and procedure/call extractor."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .hashing import canonical_json
from .preprocessor import ParseFailure, PreprocessResult, VirtualLine


DANGEROUS_RULES = {
    "strcpy": "CWE-120/STRCPY",
    "strcat": "CWE-120/STRCAT",
    "sprintf": "CWE-134/SPRINTF",
    "vsprintf": "CWE-134/VSPRINTF",
    "memcpy": "CWE-787/MEMCPY",
    "memmove": "CWE-787/MEMMOVE",
    "system": "CWE-78/SYSTEM",
    "gets": "CWE-242/GETS",
    "scanf": "CWE-676/SCANF",
}

_LEX_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<string>L?"(?:\\.|[^"\\])*")
  | (?P<char>L?'(?:\\.|[^'\\])*')
  | (?P<number>0[xX][0-9a-fA-F]+|[0-9]+(?:\.[0-9]+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><<|>>|<=|>=|==|!=|&&|\|\||->|\+\+|--|\.\.\.)
  | (?P<punct>.)
    """,
    re.VERBOSE | re.DOTALL,
)

_CONTROL_WORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "_Alignof",
    "alignof",
    "typeof",
    "defined",
    "__attribute__",
    "__declspec",
    "asm",
    "__asm__",
}


@dataclass(frozen=True)
class CToken:
    text: str
    file: str
    relative_file: str
    line: int
    column: int

    def sort_key(self) -> tuple[str, int, int]:
        return (self.relative_file, self.line, self.column)


@dataclass
class Assignment:
    lhs: str
    rhs_tokens: list[CToken]
    line: int
    relative_file: str

    @property
    def rhs_text(self) -> str:
        return "".join(token.text for token in self.rhs_tokens)


@dataclass
class CallRecord:
    callee: str
    args: list[list[CToken]]
    line: int
    column: int
    relative_file: str
    enclosing_function: str
    kind: str
    reason: str = ""
    table_name: str = ""
    explicit_id: str | None = None
    callsite_id: str = ""
    fingerprint: str = ""
    token_index: int = 0

    def args_text(self) -> list[str]:
        return ["".join(token.text for token in arg).strip() for arg in self.args]

    def location(self) -> dict[str, object]:
        return {
            "file": self.relative_file,
            "line": self.line,
            "column": self.column,
        }


@dataclass
class FunctionRecord:
    name: str
    params: list[str]
    file: str
    relative_file: str
    line: int
    calls: list[CallRecord] = field(default_factory=list)
    assignments: list[Assignment] = field(default_factory=list)
    returns: list[list[CToken]] = field(default_factory=list)
    alias_targets: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class PrototypeRecord:
    name: str
    params: list[str]
    file: str
    relative_file: str
    line: int


@dataclass
class GlobalTable:
    name: str
    targets: list[str]
    relative_file: str
    line: int


@dataclass
class ParsedUnit:
    functions: dict[str, FunctionRecord] = field(default_factory=dict)
    prototypes: dict[str, list[PrototypeRecord]] = field(default_factory=dict)
    tables: dict[str, GlobalTable] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)
    included_files: list[str] = field(default_factory=list)
    macro_expansions: int = 0

    def merge(self, other: "ParsedUnit") -> None:
        for name, function in other.functions.items():
            existing = self.functions.get(name)
            if existing is None:
                self.functions[name] = function
            else:
                # Translation units may share inline/header functions.  Keep
                # one deterministic definition and union its calls.
                existing.calls.extend(function.calls)
                existing.assignments.extend(function.assignments)
                existing.returns.extend(function.returns)
                for key, values in function.alias_targets.items():
                    existing.alias_targets.setdefault(key, set()).update(values)
        for name, records in other.prototypes.items():
            self.prototypes.setdefault(name, []).extend(records)
        for name, table in other.tables.items():
            if name not in self.tables:
                self.tables[name] = table
            else:
                self.tables[name].targets = sorted(
                    set(self.tables[name].targets) | set(table.targets)
                )
        self.calls.extend(other.calls)
        self.failures.extend(other.failures)
        self.included_files.extend(other.included_files)
        self.macro_expansions += other.macro_expansions


def _lex(preprocessed: PreprocessResult) -> list[CToken]:
    tokens: list[CToken] = []
    for virtual in preprocessed.lines:
        text = virtual.text
        for match in _LEX_RE.finditer(text):
            kind = match.lastgroup
            if kind in {"space", "comment"}:
                continue
            value = match.group(0)
            if kind == "comment":
                continue
            column = match.start() + 1
            tokens.append(
                CToken(
                    text=value,
                    file=virtual.file,
                    relative_file=virtual.relative_file,
                    line=virtual.line,
                    column=column,
                )
            )
    return tokens


def _find_matching(tokens: list[CToken], open_index: int, opening: str, closing: str) -> int | None:
    depth = 0
    for index in range(open_index, len(tokens)):
        token = tokens[index].text
        if token == opening:
            depth += 1
        elif token == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _find_previous_matching(tokens: list[CToken], close_index: int, opening: str, closing: str) -> int | None:
    depth = 0
    for index in range(close_index, -1, -1):
        token = tokens[index].text
        if token == closing:
            depth += 1
        elif token == opening:
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_commas(tokens: list[CToken]) -> list[list[CToken]]:
    if not tokens:
        return []
    result: list[list[CToken]] = [[]]
    depth = 0
    for token in tokens:
        if token.text in {"(", "[", "{"}:
            depth += 1
        elif token.text in {")", "]", "}"}:
            depth = max(0, depth - 1)
        if token.text == "," and depth == 0:
            result.append([])
        else:
            result[-1].append(token)
    return result


def _parameter_names(tokens: list[CToken]) -> list[str]:
    names: list[str] = []
    for segment in _split_commas(tokens):
        if not segment:
            continue
        if len(segment) == 1 and segment[0].text in {"void", "..."}:
            continue
        identifiers = [token.text for token in segment if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token.text)]
        if not identifiers:
            continue
        # The final identifier is the parameter name in the declarations used
        # by this workbench; function-pointer parameter names are skipped.
        candidates = [name for name in identifiers if name not in {"const", "volatile", "restrict", "struct", "enum", "union"}]
        if candidates:
            names.append(candidates[-1])
    return names


def _normalize_arg(tokens: list[CToken]) -> str:
    return "".join(token.text for token in tokens)


def _semantic_fingerprint(
    relative_file: str,
    function: str,
    callee: str,
    args: list[str],
    kind: str,
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "file": relative_file,
                "function": function,
                "callee": callee,
                "args": args,
                "kind": kind,
            }
        ).encode("utf-8")
    ).hexdigest()


def _marker_map(preprocessed: PreprocessResult) -> dict[tuple[str, int], str]:
    markers: dict[tuple[str, int], str] = {}
    marker_re = re.compile(r"@call:([A-Za-z0-9_.-]+)")
    for virtual in preprocessed.lines:
        match = marker_re.search(virtual.source_text) or marker_re.search(virtual.text)
        if match:
            markers[(virtual.relative_file, virtual.line)] = match.group(1)
    return markers


def _classify_call(
    tokens: list[CToken],
    index: int,
    close_index: int,
    function: str,
    known_functions: set[str],
    tables: dict[str, GlobalTable],
) -> tuple[str, str, str, str] | None:
    """Return (callee, kind, table_name, reason) or None for non-call parentheses."""

    previous = tokens[index - 1] if index > 0 else None
    before_previous = tokens[index - 2] if index > 1 else None
    if previous is None:
        return None
    if previous.text in _CONTROL_WORDS:
        return None
    if previous.text in {"(", "[", "{", ",", ";", "=", "?", ":", "!", "~"}:
        return None
    if previous.text == ")":
        # Function pointer cast/returning-function forms are not statically
        # resolvable by this restricted parser.
        return ("__unknown_function_pointer__" if before_previous else "__unknown__", "indirect_unknown", "", "callee expression is not a named function")

    if previous.text == "]":
        opening = _find_previous_matching(tokens, index - 1, "[", "]")
        if opening is not None and opening > 0 and re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", tokens[opening - 1].text
        ):
            base = tokens[opening - 1].text
            if base in tables:
                return (base, "indirect_table", base, "")
            return (base, "indirect_unknown", "", f"unresolved function-pointer array {base}")
        return ("__unknown_index__", "indirect_unknown", "", "callee expression uses an unresolved index")

    if previous.text in {".", "->"}:
        member = tokens[index - 1].text
        # For `x -> member (`, previous is member and before_previous is `->`.
        member = previous.text
        if before_previous and before_previous.text in {".", "->"}:
            member = previous.text
        # Field member name is the identifier immediately before '('. The
        # punctuation form is already excluded by the branch above.
        return (
            member,
            "indirect_unknown",
            "",
            "callee is an external structure field function pointer",
        )

    # `base.member(` / `base->member(` lexes as id, punctuation, id, '('.
    if before_previous and before_previous.text in {".", "->"}:
        return (
            previous.text,
            "indirect_unknown",
            "",
            "callee is an external structure field function pointer",
        )

    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", previous.text):
        name = previous.text
        if name in known_functions:
            return (name, "direct", "", "")
        # An identifier call may still be a known table or a variable holding a
        # function pointer; resolution happens in the interprocedural pass.
        return (name, "direct", "", "")

    if previous.text == "*":
        return ("__function_pointer__", "indirect_unknown", "", "dereferenced function pointer")
    return None


def parse_unit(preprocessed: PreprocessResult, translation_unit: str) -> ParsedUnit:
    unit = ParsedUnit()
    unit.failures.extend(preprocessed.failures)
    unit.included_files.extend(preprocessed.included_files)
    unit.macro_expansions = preprocessed.macro_expansions
    tokens = _lex(preprocessed)
    markers = _marker_map(preprocessed)

    # First pass: function definitions, prototypes, and global tables.
    known_functions: set[str] = set()
    function_ranges: list[tuple[int, int]] = []
    index = 0
    segment_start = 0
    while index < len(tokens):
        token = tokens[index]
        if token.text == ";":
            segment_start = index + 1
            index += 1
            continue
        if token.text != "{":
            index += 1
            continue
        close = _find_matching(tokens, index, "{", "}")
        if close is None:
            unit.failures.append(
                ParseFailure(
                    token.relative_file,
                    token.line,
                    "unmatched opening brace",
                    token.text,
                )
            )
            break
        header_tokens = tokens[segment_start:index]
        # Find the final parameter list in a candidate function header.
        last_close = None
        for candidate in range(len(header_tokens) - 1, -1, -1):
            if header_tokens[candidate].text == ")":
                last_close = candidate
                break
        function_name: str | None = None
        params: list[str] = []
        if last_close is not None:
            global_close = segment_start + last_close
            open_index = _find_previous_matching(tokens, global_close, "(", ")")
            if open_index is not None and open_index > 0:
                name_token = tokens[open_index - 1]
                if (
                    re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name_token.text)
                    and name_token.text not in _CONTROL_WORDS
                    and "=" not in [item.text for item in header_tokens]
                ):
                    function_name = name_token.text
                    params = _parameter_names(tokens[open_index + 1 : global_close])
        if function_name:
            record = FunctionRecord(
                name=function_name,
                params=params,
                file=token.file,
                relative_file=token.relative_file,
                line=tokens[segment_start].line if segment_start < len(tokens) else token.line,
            )
            if function_name not in unit.functions:
                unit.functions[function_name] = record
                known_functions.add(function_name)
            else:
                # Keep deterministic first definition; duplicate definitions
                # are represented by a parse failure rather than silently
                # overwriting one.
                unit.failures.append(
                    ParseFailure(
                        token.relative_file,
                        token.line,
                        f"duplicate function definition {function_name}",
                        function_name,
                    )
                )
            function_ranges.append((index + 1, close))
        else:
            # Global function-pointer table initializer: `name[] = { ... };`
            equality = None
            for candidate in range(len(header_tokens) - 1, -1, -1):
                if header_tokens[candidate].text == "=":
                    equality = candidate
                    break
            if equality is not None:
                left = header_tokens[:equality]
                ids = [
                    item.text
                    for item in left
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.text)
                ]
                if ids:
                    table_name = ids[-1]
                    body_ids = [
                        item.text
                        for item in tokens[index + 1 : close]
                        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.text)
                    ]
                    unit.tables[table_name] = GlobalTable(
                        name=table_name,
                        targets=sorted(set(body_ids)),
                        relative_file=token.relative_file,
                        line=token.line,
                    )
        segment_start = close + 1
        index = close + 1

    # Prototypes at top level (and declarations visible in headers).
    depth = 0
    for idx, token in enumerate(tokens):
        if token.text == "{":
            depth += 1
        elif token.text == "}":
            depth = max(0, depth - 1)
        elif token.text == "(" and depth == 0:
            open_index = idx
            close_index = _find_matching(tokens, open_index, "(", ")")
            if close_index is None:
                continue
            if open_index > 0:
                name_token = tokens[open_index - 1]
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name_token.text):
                    after = close_index + 1
                    while after < len(tokens) and tokens[after].text not in {";", "{"}:
                        after += 1
                    if after < len(tokens) and tokens[after].text == ";":
                        name = name_token.text
                        if name not in known_functions:
                            unit.prototypes.setdefault(name, []).append(
                                PrototypeRecord(
                                    name=name,
                                    params=_parameter_names(tokens[open_index + 1 : close_index]),
                                    file=token.file,
                                    relative_file=token.relative_file,
                                    line=token.line,
                                )
                            )

    # Second pass: calls, assignments, and returns inside each function body.
    for function_index, (body_start, body_end) in enumerate(function_ranges):
        # Locate the function record by body range line file/name mapping.
        # function_ranges and definitions were appended in lock-step.
        definitions = list(unit.functions.values())
        if function_index >= len(definitions):
            continue
        function = definitions[function_index]
        body_tokens = tokens[body_start:body_end]
        depth = 0
        for local_index, token in enumerate(body_tokens):
            if token.text == "{":
                depth += 1
            elif token.text == "}":
                depth = max(0, depth - 1)
            if token.text == "=" and local_index > 0:
                lhs_candidates = [
                    item for item in body_tokens[max(0, local_index - 5) : local_index]
                    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item.text)
                ]
                if lhs_candidates and depth == 0:
                    lhs = lhs_candidates[-1].text
                    rhs: list[CToken] = []
                    cursor = local_index + 1
                    paren_depth = 0
                    while cursor < len(body_tokens):
                        rhs_token = body_tokens[cursor]
                        if rhs_token.text in {"(", "[", "{"}:
                            paren_depth += 1
                        elif rhs_token.text in {")", "]", "}"}:
                            paren_depth = max(0, paren_depth - 1)
                        if rhs_token.text == ";" and paren_depth == 0:
                            break
                        rhs.append(rhs_token)
                        cursor += 1
                    function.assignments.append(
                        Assignment(
                            lhs=lhs,
                            rhs_tokens=rhs,
                            line=token.line,
                            relative_file=token.relative_file,
                        )
                    )
                    if rhs and len(rhs) == 1 and re.fullmatch(
                        r"[A-Za-z_][A-Za-z0-9_]*", rhs[0].text
                    ):
                        function.alias_targets.setdefault(lhs, set()).add(rhs[0].text)
            if token.text == "return":
                rhs = []
                cursor = local_index + 1
                paren_depth = 0
                while cursor < len(body_tokens):
                    rhs_token = body_tokens[cursor]
                    if rhs_token.text in {"(", "[", "{"}:
                        paren_depth += 1
                    elif rhs_token.text in {")", "]", "}"}:
                        paren_depth = max(0, paren_depth - 1)
                    if rhs_token.text == ";" and paren_depth == 0:
                        break
                    rhs.append(rhs_token)
                    cursor += 1
                if rhs:
                    function.returns.append(rhs)

        # Calls are recognized from every '(' in the body.
        local_index = 0
        explicit_ordinal = 0
        while local_index < len(body_tokens):
            token = body_tokens[local_index]
            if token.text != "(":
                local_index += 1
                continue
            close_index = _find_matching(body_tokens, local_index, "(", ")")
            if close_index is None:
                unit.failures.append(
                    ParseFailure(
                        token.relative_file,
                        token.line,
                        "unmatched call parentheses",
                        token.text,
                    )
                )
                local_index += 1
                continue
            classification = _classify_call(
                body_tokens,
                local_index,
                close_index,
                function.name,
                known_functions,
                unit.tables,
            )
            if classification is None:
                local_index += 1
                continue
            callee, kind, table_name, reason = classification
            if callee in {"__unknown__", "__unknown_function_pointer__", "__unknown_index__", "__function_pointer__"}:
                # Keep a non-dangerous unknown call visible for diagnostics but
                # do not turn it into a dangerous-call finding by itself.
                pass
            args_tokens = body_tokens[local_index + 1 : close_index]
            args = _split_commas(args_tokens)
            marker = markers.get((token.relative_file, token.line))
            if marker is None:
                marker = markers.get((token.relative_file, token.line - 1))
            if marker is None:
                marker = markers.get((token.relative_file, token.line - 2))
            record = CallRecord(
                callee=callee,
                args=args,
                line=token.line,
                column=body_tokens[local_index - 1].column,
                relative_file=token.relative_file,
                enclosing_function=function.name,
                kind=kind,
                reason=reason,
                table_name=table_name,
                explicit_id=marker,
                token_index=local_index,
            )
            function.calls.append(record)
            unit.calls.append(record)
            local_index = close_index + 1

    # Stable IDs: explicit markers are strongest; auto IDs deliberately omit
    # line/column offsets and use semantic content plus a deterministic ordinal.
    auto_ordinals: dict[tuple[str, str, str, tuple[str, ...], str], int] = {}
    for call in sorted(
        unit.calls,
        key=lambda item: (
            item.relative_file,
            item.enclosing_function,
            item.line,
            item.column,
            item.callee,
        ),
    ):
        args = call.args_text()
        if call.explicit_id:
            call.callsite_id = call.explicit_id
        else:
            base_signature = (
                call.relative_file,
                call.enclosing_function,
                call.callee,
                tuple(args),
                call.kind,
            )
            ordinal = auto_ordinals.get(base_signature, 0)
            auto_ordinals[base_signature] = ordinal + 1
            call.callsite_id = "auto:" + hashlib.sha256(
                canonical_json([list(base_signature), ordinal]).encode("utf-8")
            ).hexdigest()[:24]
        call.fingerprint = _semantic_fingerprint(
            call.relative_file,
            call.enclosing_function,
            call.callee,
            args,
            call.kind,
        )
    return unit

