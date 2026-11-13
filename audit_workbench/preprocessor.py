"""Restricted, bounded C preprocessor used by the audit parser.

This is deliberately not a compiler preprocessor.  It implements the subset
needed for admission auditing, while imposing hard limits so malformed input
cannot consume unbounded CPU or memory.
"""

from __future__ import annotations

import ast
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import AuditConfig, MacroSetting
from .errors import WorkbenchError


MAX_INCLUDE_DEPTH = 12
MAX_VIRTUAL_LINES = 30000
MAX_MACRO_EXPANSIONS = 20000
MAX_EXPANSION_DEPTH = 48


@dataclass(frozen=True)
class Macro:
    name: str
    params: tuple[str, ...] | None
    variadic: bool
    body: str
    file: str
    line: int

    def signature(self) -> tuple[tuple[str, ...] | None, bool, str]:
        return (self.params, self.variadic, " ".join(self.body.split()))


@dataclass
class VirtualLine:
    file: str
    relative_file: str
    line: int
    text: str
    source_text: str


@dataclass
class ParseFailure:
    file: str
    line: int
    reason: str
    snippet: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "reason": self.reason,
            "snippet": self.snippet,
        }


@dataclass
class PreprocessResult:
    lines: list[VirtualLine] = field(default_factory=list)
    failures: list[ParseFailure] = field(default_factory=list)
    macro_expansions: int = 0
    included_files: list[str] = field(default_factory=list)


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN_RE = re.compile(
    r"""
    (?P<space>\s+)
  | (?P<string>L?"(?:\\.|[^"\\])*")
  | (?P<char>L?'(?:\\.|[^'\\])*')
  | (?P<number>(?:0[xX][0-9a-fA-F]+|[0-9]+)(?:[uUlL]+)?)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><<|>>|<=|>=|==|!=|&&|\|\||\#\#|->|\+\+|--)
  | (?P<punct>.)
    """,
    re.VERBOSE,
)


def _tokenize(text: str) -> list[str]:
    return [match.group(0) for match in _TOKEN_RE.finditer(text)]


def _stringify(argument: str) -> str:
    escaped = argument.strip().replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


def _split_source_segments(text: str) -> list[tuple[bool, str]]:
    """Split into code and literal/comment segments; macros only expand in code."""

    segments: list[tuple[bool, str]] = []
    code_start = 0
    index = 0

    def flush_code(end: int) -> None:
        nonlocal code_start
        if code_start < end:
            segments.append((True, text[code_start:end]))

    while index < len(text):
        if text.startswith("//", index):
            flush_code(index)
            segments.append((False, text[index:]))
            return segments
        if text.startswith("/*", index):
            flush_code(index)
            end = text.find("*/", index + 2)
            if end < 0:
                segments.append((False, text[index:]))
                return segments
            segments.append((False, text[index : end + 2]))
            index = end + 2
            code_start = index
            continue
        if text[index] in {'"', "'"}:
            quote = text[index]
            cursor = index + 1
            escaped = False
            while cursor < len(text):
                char = text[cursor]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    cursor += 1
                    break
                cursor += 1
            index = cursor
            continue
        index += 1
    flush_code(len(text))
    return segments


class Preprocessor:
    def __init__(self, root: Path, config: AuditConfig, *, fixture_hang: bool = True) -> None:
        self.root = Path(root).resolve()
        self.config = config
        self.fixture_hang = fixture_hang
        self.result = PreprocessResult()
        self.macros: dict[str, Macro] = {}
        self.pragma_once: set[str] = set()
        self.include_stack: list[str] = []
        self._included_seen: set[str] = set()
        self._line_count = 0
        self._expansions = 0
        for name, setting in config.macros.items():
            if setting.defined:
                self.macros[name] = Macro(
                    name=name,
                    params=None,
                    variadic=False,
                    body=setting.value if setting.value is not None else "1",
                    file="<config>",
                    line=1,
                )
        self.include_paths = []
        for include_path in config.include_paths:
            resolved = self._safe_resolve_base(include_path)
            if resolved is not None:
                self.include_paths.append(resolved)

    def _safe_resolve_base(self, include_path: str) -> Path | None:
        candidate = Path(include_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.root / candidate).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return None
        return resolved

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def _failure(self, file: Path | str, line: int, reason: str, snippet: str = "") -> None:
        if isinstance(file, Path):
            display = self._relative(file)
        else:
            display = file
        self.result.failures.append(
            ParseFailure(display, line, reason, snippet.strip()[:240])
        )

    def _add_virtual_line(
        self,
        file: Path,
        line_number: int,
        text: str,
        source_text: str,
    ) -> None:
        self._line_count += 1
        if self._line_count > MAX_VIRTUAL_LINES:
            raise WorkbenchError(
                "E_PARSE_LIMIT",
                "virtual-line limit exceeded",
                {
                    "file": self._relative(file),
                    "line": line_number,
                    "limit": MAX_VIRTUAL_LINES,
                },
            )
        self.result.lines.append(
            VirtualLine(
                file=str(file.resolve()),
                relative_file=self._relative(file),
                line=line_number,
                text=text,
                source_text=source_text,
            )
        )

    def _record_expansion(self, file: Path, line: int, snippet: str) -> None:
        self._expansions += 1
        self.result.macro_expansions += 1
        if self._expansions > MAX_MACRO_EXPANSIONS:
            raise WorkbenchError(
                "E_PARSE_LIMIT",
                "macro-expansion limit exceeded",
                {
                    "file": self._relative(file),
                    "line": line,
                    "limit": MAX_MACRO_EXPANSIONS,
                },
            )

    @staticmethod
    def _split_macro_args(tokens: list[str], open_index: int) -> tuple[list[str], int] | None:
        depth = 0
        args: list[list[str]] = [[]]
        index = open_index
        while index < len(tokens):
            token = tokens[index]
            if token == "(":
                depth += 1
                if depth > 1:
                    args[-1].append(token)
            elif token == ")":
                depth -= 1
                if depth == 0:
                    return ["".join(arg).strip() for arg in args], index + 1
                args[-1].append(token)
            elif token == "," and depth == 1:
                args.append([])
            else:
                if depth >= 1:
                    args[-1].append(token)
            index += 1
        return None

    def _expand_tokens(
        self,
        tokens: list[str],
        *,
        file: Path,
        line: int,
        snippet: str,
        active_names: frozenset[str] = frozenset(),
        depth: int = 0,
    ) -> str:
        if depth > MAX_EXPANSION_DEPTH:
            self._failure(file, line, "macro expansion depth limit exceeded", snippet)
            return "__AUDIT_UNRESOLVED_MACRO_DEPTH__"
        output: list[str] = []
        index = 0
        while index < len(tokens):
            token = tokens[index]
            macro = self.macros.get(token) if _IDENT_RE.fullmatch(token) else None
            if macro is None:
                output.append(token)
                index += 1
                continue
            if token in active_names:
                self._failure(
                    file,
                    line,
                    f"unresolvable recursive macro {token}",
                    snippet,
                )
                output.append(f"__AUDIT_UNRESOLVED_MACRO_{token}__")
                index += 1
                continue
            if macro.params is None:
                self._record_expansion(file, line, snippet)
                expanded = self._expand_tokens(
                    _tokenize(macro.body),
                    file=file,
                    line=line,
                    snippet=snippet,
                    active_names=active_names | {token},
                    depth=depth + 1,
                )
                output.append(expanded)
                index += 1
                continue

            # Function-like macro: require an immediately following '(' after
            # optional whitespace (whitespace is irrelevant in our token list).
            open_index = index + 1
            while open_index < len(tokens) and tokens[open_index].isspace():
                open_index += 1
            if open_index >= len(tokens) or tokens[open_index] != "(":
                output.append(token)
                index += 1
                continue
            parsed = self._split_macro_args(tokens, open_index)
            if parsed is None:
                self._failure(file, line, f"unterminated macro invocation {token}", snippet)
                output.append(token)
                index += 1
                continue
            args, end_index = parsed
            expected = len(macro.params)
            if expected == 0 and not macro.variadic and args == [""]:
                args = []
            if macro.variadic:
                if len(args) < expected:
                    self._failure(
                        file,
                        line,
                        f"macro {token} expects at least {expected} arguments",
                        snippet,
                    )
                    output.append(f"__AUDIT_UNRESOLVED_MACRO_{token}__")
                    index = end_index
                    continue
                fixed = args[:expected]
                rest = ",".join(args[expected:])
                args = fixed + [rest]
            elif len(args) != expected:
                self._failure(
                    file,
                    line,
                    f"macro {token} expects {expected} arguments, got {len(args)}",
                    snippet,
                )
                output.append(f"__AUDIT_UNRESOLVED_MACRO_{token}__")
                index = end_index
                continue
            self._record_expansion(file, line, snippet)

            body_tokens = _tokenize(macro.body)
            substituted: list[str] = []
            param_names = list(macro.params) + (["__VA_ARGS__"] if macro.variadic else [])
            arg_map = dict(zip(param_names, args))
            param_index = 0
            while param_index < len(body_tokens):
                current = body_tokens[param_index]
                if current == "#" and param_index + 1 < len(body_tokens):
                    name = body_tokens[param_index + 1]
                    if name in arg_map:
                        substituted.append(_stringify(arg_map[name]))
                        param_index += 2
                        continue
                if current == "##" and substituted and param_index + 1 < len(body_tokens):
                    next_token = body_tokens[param_index + 1]
                    if next_token in arg_map:
                        next_token = arg_map[next_token].strip()
                    substituted[-1] = substituted[-1].rstrip() + next_token
                    param_index += 2
                    continue
                if current in arg_map:
                    raw_arg = arg_map[current]
                    # Arguments are macro-expanded before substitution unless
                    # adjacent to stringification/paste, which were handled above.
                    arg_expanded = self._expand_tokens(
                        _tokenize(raw_arg),
                        file=file,
                        line=line,
                        snippet=snippet,
                        active_names=active_names,
                        depth=depth + 1,
                    )
                    substituted.append(arg_expanded)
                else:
                    substituted.append(current)
                param_index += 1
            expanded_body = self._expand_tokens(
                _tokenize("".join(substituted)),
                file=file,
                line=line,
                snippet=snippet,
                active_names=active_names | {token},
                depth=depth + 1,
            )
            output.append(expanded_body)
            index = end_index
        return "".join(output)

    def _expand_text(self, text: str, file: Path, line: int, source_text: str) -> str:
        parts: list[str] = []
        for is_code, segment in _split_source_segments(text):
            if not is_code:
                parts.append(segment)
                continue
            expanded = self._expand_tokens(
                _tokenize(segment),
                file=file,
                line=line,
                snippet=source_text,
            )
            parts.append(expanded)
        return "".join(parts)

    @staticmethod
    def _macro_from_define(
        text: str,
        file: str,
        line: int,
    ) -> Macro | None:
        match = re.match(r"\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        if not match:
            return None
        name = match.group(1)
        position = match.end()
        params: tuple[str, ...] | None = None
        variadic = False
        if position < len(text) and text[position] == "(":
            close = text.find(")", position + 1)
            if close < 0:
                return None
            raw_params = text[position + 1 : close].strip()
            if raw_params == "...":
                params = ()
                variadic = True
            elif raw_params:
                names = [item.strip() for item in raw_params.split(",")]
                if names and names[-1] == "...":
                    variadic = True
                    names = names[:-1]
                params = tuple(names)
            else:
                params = ()
            position = close + 1
        body = text[position:].strip()
        return Macro(name, params, variadic, body, file, line)

    def _resolve_include(self, current_file: Path, include_name: str, quote: bool) -> Path | None:
        candidates: list[Path] = []
        if quote:
            candidates.append(current_file.parent / include_name)
        candidates.extend(path / include_name for path in self.include_paths)
        if not quote:
            # Current-directory fallback is permitted for deterministic fixtures.
            candidates.append(current_file.parent / include_name)
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                resolved.relative_to(self.root)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None

    @staticmethod
    def _eval_if_expression(
        expression: str,
        macros: dict[str, Macro],
    ) -> bool:
        expression = re.sub(r"/\*.*?\*/", " ", expression)
        expression = re.sub(r"//.*$", "", expression)
        expression = expression.strip()

        def defined_replacer(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            return "1" if name in macros else "0"

        expression = re.sub(
            r"defined\s*(?:\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)|\s*([A-Za-z_][A-Za-z0-9_]*))",
            defined_replacer,
            expression,
        )

        def ident_replacer(match: re.Match[str]) -> str:
            name = match.group(0)
            if name in {"and", "or", "not"}:
                return name
            macro = macros.get(name)
            if macro is None:
                return "0"
            body = macro.body.strip()
            if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", body):
                return body
            return "1"

        expression = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", ident_replacer, expression)
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return False

        def evaluate(node: ast.AST) -> int | float:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.UnaryOp):
                value = evaluate(node.operand)
                if isinstance(node.op, ast.Not):
                    return 0 if value else 1
                if isinstance(node.op, ast.USub):
                    return -value
                if isinstance(node.op, ast.UAdd):
                    return +value
                if isinstance(node.op, ast.Invert):
                    return ~int(value)
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left // right if right else 0
                if isinstance(node.op, ast.Mod):
                    return left % right if right else 0
                if isinstance(node.op, ast.LShift):
                    return int(left) << int(right)
                if isinstance(node.op, ast.RShift):
                    return int(left) >> int(right)
                if isinstance(node.op, ast.BitAnd):
                    return int(left) & int(right)
                if isinstance(node.op, ast.BitOr):
                    return int(left) | int(right)
                if isinstance(node.op, ast.BitXor):
                    return int(left) ^ int(right)
            if isinstance(node, ast.BoolOp):
                values = [evaluate(item) for item in node.values]
                if isinstance(node.op, ast.And):
                    return 1 if all(values) else 0
                if isinstance(node.op, ast.Or):
                    return 1 if any(values) else 0
            if isinstance(node, ast.Compare):
                left = evaluate(node.left)
                for op, comparator in zip(node.ops, node.comparators):
                    right = evaluate(comparator)
                    if isinstance(op, ast.Eq) and not left == right:
                        return 0
                    if isinstance(op, ast.NotEq) and not left != right:
                        return 0
                    if isinstance(op, ast.Lt) and not left < right:
                        return 0
                    if isinstance(op, ast.LtE) and not left <= right:
                        return 0
                    if isinstance(op, ast.Gt) and not left > right:
                        return 0
                    if isinstance(op, ast.GtE) and not left >= right:
                        return 0
                    left = right
                return 1
            raise ValueError("unsupported preprocessor expression")

        try:
            return bool(evaluate(tree.body))
        except (ValueError, TypeError, ZeroDivisionError):
            return False

    def _process_file(self, path: Path) -> None:
        canonical = str(path.resolve())
        if canonical in self.pragma_once and canonical in self._included_seen:
            return
        if path.resolve() in [Path(item).resolve() for item in self.include_stack]:
            raise WorkbenchError(
                "E_INCLUDE_CYCLE",
                f"include cycle detected while loading {self._relative(path)}",
                {
                    "file": self._relative(path),
                    "line": 1,
                    "include": self._relative(path),
                    "stack": [self._relative(Path(item)) for item in self.include_stack],
                },
            )
        if len(self.include_stack) >= MAX_INCLUDE_DEPTH:
            raise WorkbenchError(
                "E_PARSE_LIMIT",
                "include-depth limit exceeded",
                {
                    "file": self._relative(path),
                    "line": 1,
                    "limit": MAX_INCLUDE_DEPTH,
                },
            )
        self.include_stack.append(str(path.resolve()))
        self._included_seen.add(canonical)
        self.result.included_files.append(self._relative(path))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self._failure(path, 1, f"cannot read source file: {exc}")
            self.include_stack.pop()
            return

        cond_stack: list[dict[str, bool]] = []

        def is_active() -> bool:
            return all(frame["active"] for frame in cond_stack)

        physical_lines = text.splitlines()
        physical_index = 0
        while physical_index < len(physical_lines):
            line_number = physical_index + 1
            raw_line = physical_lines[physical_index]
            if self.fixture_hang and "@audit-hang" in raw_line and is_active():
                while True:
                    time.sleep(0.05)
            physical_index += 1
            consumed_lines = 1
            while (
                raw_line.rstrip().endswith("\\")
                and physical_index < len(physical_lines)
            ):
                continuation = physical_lines[physical_index]
                if self.fixture_hang and "@audit-hang" in continuation and is_active():
                    while True:
                        time.sleep(0.05)
                physical_index += 1
                consumed_lines += 1
                raw_line = raw_line.rstrip()[:-1] + " " + continuation.lstrip()
            self._line_count += consumed_lines
            if self._line_count > MAX_VIRTUAL_LINES:
                raise WorkbenchError(
                    "E_PARSE_LIMIT",
                    "virtual-line limit exceeded",
                    {
                        "file": self._relative(path),
                        "line": line_number,
                        "limit": MAX_VIRTUAL_LINES,
                    },
                )
            stripped = raw_line.lstrip()
            directive_match = re.match(r"#\s*([A-Za-z_]+)", stripped)
            directive = directive_match.group(1).lower() if directive_match else ""
            active_before = is_active()

            if directive in {"if", "ifdef", "ifndef"}:
                parent_active = active_before
                condition = False
                rest = stripped[directive_match.end() :].strip() if directive_match else ""
                if directive == "ifdef":
                    name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", rest)
                    condition = bool(name_match and name_match.group(1) in self.macros)
                elif directive == "ifndef":
                    name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", rest)
                    condition = bool(name_match and name_match.group(1) not in self.macros)
                else:
                    condition = self._eval_if_expression(rest, self.macros)
                effective = parent_active and condition
                cond_stack.append(
                    {
                        "parent": parent_active,
                        "active": effective,
                        "taken": effective,
                        "else_seen": False,
                    }
                )
                continue

            if directive == "elif":
                if not cond_stack:
                    self._failure(path, line_number, "unmatched #elif", raw_line)
                    continue
                frame = cond_stack[-1]
                if frame["else_seen"]:
                    self._failure(path, line_number, "#elif after #else", raw_line)
                    continue
                rest = stripped[directive_match.end() :].strip() if directive_match else ""
                condition = self._eval_if_expression(rest, self.macros)
                frame["active"] = frame["parent"] and not frame["taken"] and condition
                frame["taken"] = frame["taken"] or frame["active"]
                continue

            if directive == "else":
                if not cond_stack:
                    self._failure(path, line_number, "unmatched #else", raw_line)
                    continue
                frame = cond_stack[-1]
                if frame["else_seen"]:
                    self._failure(path, line_number, "duplicate #else", raw_line)
                    continue
                frame["else_seen"] = True
                frame["active"] = frame["parent"] and not frame["taken"]
                frame["taken"] = True
                continue

            if directive == "endif":
                if not cond_stack:
                    self._failure(path, line_number, "unmatched #endif", raw_line)
                else:
                    cond_stack.pop()
                continue

            if not active_before:
                continue

            if directive == "include":
                include_match = re.match(
                    r'\s*#\s*include\s*(?:"([^"]+)"|<([^>]+)>)',
                    raw_line,
                )
                if not include_match:
                    operand = re.sub(
                        r"^\s*#\s*include\s*",
                        "",
                        raw_line,
                        flags=re.IGNORECASE,
                    )
                    expanded_operand = self._expand_text(
                        operand,
                        path,
                        line_number,
                        raw_line,
                    )
                    include_match = re.match(
                        r'\s*(?:"([^"]+)"|<([^>]+)>)',
                        expanded_operand,
                    )
                    if not include_match:
                        self._failure(
                            path,
                            line_number,
                            "malformed or unresolvable #include directive",
                            raw_line,
                        )
                        continue
                quote = include_match.group(1) is not None
                include_name = include_match.group(1) or include_match.group(2) or ""
                resolved = self._resolve_include(path, include_name, quote)
                if resolved is None:
                    self._failure(
                        path,
                        line_number,
                        f"include not found: {include_name}",
                        raw_line,
                    )
                    continue
                if str(resolved.resolve()) in self.include_stack:
                    self._failure(
                        path,
                        line_number,
                        f"include cycle: {self._relative(resolved)}",
                        raw_line,
                    )
                    raise WorkbenchError(
                        "E_INCLUDE_CYCLE",
                        f"include cycle detected: {self._relative(resolved)}",
                        {
                            "file": self._relative(path),
                            "line": line_number,
                            "include": self._relative(resolved),
                            "stack": [
                                self._relative(Path(item)) for item in self.include_stack
                            ],
                        },
                    )
                if str(resolved.resolve()) in self.pragma_once:
                    continue
                self._process_file(resolved)
                continue

            if directive == "pragma":
                if re.match(r"\s*#\s*pragma\s+once\b", raw_line, re.IGNORECASE):
                    self.pragma_once.add(canonical)
                continue

            if directive == "define":
                macro = self._macro_from_define(raw_line, self._relative(path), line_number)
                if macro is None:
                    self._failure(path, line_number, "malformed #define directive", raw_line)
                    continue
                previous = self.macros.get(macro.name)
                if previous is not None and previous.signature() != macro.signature():
                    raise WorkbenchError(
                        "E_MACRO_CONFLICT",
                        f"macro {macro.name!r} is redefined with a different value",
                        {
                            "file": self._relative(path),
                            "line": line_number,
                            "macro": macro.name,
                            "previous_file": previous.file,
                            "previous_line": previous.line,
                        },
                    )
                self.macros[macro.name] = macro
                continue

            if directive == "undef":
                name_match = re.match(r"\s*#\s*undef\s+([A-Za-z_][A-Za-z0-9_]*)", raw_line)
                if name_match:
                    self.macros.pop(name_match.group(1), None)
                continue

            if directive == "error":
                self._failure(path, line_number, "#error directive reached", raw_line)
                continue

            if directive in {"line", "warning", "pragma", "ident", "sccs", "assert"}:
                continue

            expanded = self._expand_text(raw_line, path, line_number, raw_line)
            self._add_virtual_line(path, line_number, expanded, raw_line)

        self.include_stack.pop()

    def process(self, translation_unit: Path) -> PreprocessResult:
        self._process_file(Path(translation_unit))
        return self.result

