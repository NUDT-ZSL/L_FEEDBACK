"""Seed-only deterministic source sample and acceptance-fixture generator."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from .config import BUILTIN_CONFIGS, config_from_dict
from .errors import WorkbenchError
from .hashing import canonical_json, deterministic_token, sha256_text


GENERATED_MARKER = ".audit_generated"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _line_column(text: str, needle: str, line: int | None = None) -> tuple[int, int]:
    lines = text.splitlines()
    candidates = range(len(lines)) if line is None else [line - 1]
    for index in candidates:
        if index < 0 or index >= len(lines):
            continue
        column = lines[index].find(needle)
        if column >= 0:
            return index + 1, column + 1
    return line or 1, 1


def _hidden_entries() -> str:
    return "\n".join(
        [
            "entry_unit_a",
            "entry_unit_b",
            "entry_unit_c",
        ]
    )


def _config_text(name: str, include_paths: list[str], seed: int) -> str:
    base = dict(BUILTIN_CONFIGS[name])
    base["include_paths"] = include_paths
    base["entries"] = ["entry_unit_a", "entry_unit_b", "entry_unit_c"]
    base["_generated_seed"] = seed
    # Build through the canonical model so fixture configs and builtins share semantics.
    cfg = config_from_dict(base, source=f"<generated:{name}>")
    return cfg.serialized()


def _root_files(
    slug: str,
    seed: int,
    root_name: str,
    include_dir: str,
    src_dir: str,
    nested_dir: str,
    deep_dir: str,
    file_names: dict[str, str],
    hang_in_bundle: bool,
) -> dict[str, str]:
    token = deterministic_token(seed, f"root-{slug}-content", 10)
    seed_line = f"/* deterministic sample seed={seed} root={slug} token={token} */"
    guard = f"UNIT_{slug.upper()}_BUNDLE_H"
    function_tail = f"_{slug}"
    copy_fn = f"copy_fn{function_tail}"
    external_type = f"external_ops{function_tail}"

    include_bundle = f'#include "{file_names["bundle"]}"'
    include_ops = f'#include "{file_names["ops"]}"'
    include_dispatch = f'#include "{file_names["dispatch_header"]}"'

    hang_lines = ""
    if hang_in_bundle:
        hang_lines = (
            "#if defined(HANG_MODE)\n"
            f'#include "{file_names["hang"]}"\n'
            "#endif\n"
        )

    bundle = f"""\
{seed_line}
#ifndef {guard}
#define {guard}

#define AUDIT_SEED {seed}
#define STRINGIFY_VALUE{function_tail}(x) #x
#define NESTED_TEXT{function_tail}(x) STRINGIFY_VALUE{function_tail}(x)
#define FORWARD_FORMAT{function_tail}(fn, out, fmt, value) fn(out, fmt, value)
#define RECURSE_INPUT{function_tail} RECURSE_INPUT{function_tail}

#ifndef AUDIT_SEED
#error deterministic seed macro must be defined
#endif

typedef int (*{copy_fn})(char *dst, const char *src, int n);
struct {external_type} {{
    {copy_fn} memcpy;
}};
extern struct {external_type} g_external;
extern char *g_sink;

char *read_input(int limit);
char *recv_from_socket(int fd, int limit);
char *getenv(const char *name);
int dispatch_request(char *payload, int n);
int relay_input(char *payload);
int safe_copy(char *dst, const char *src, int n);
int bounded_copy(char *dst, const char *src, int n);
int orphan_copy(char *dst, char *src);
int local_banner(char *out);

{hang_lines}#endif
"""

    ops = f"""\
{seed_line}
#pragma once
{include_bundle}

struct local_ops{function_tail} {{
    {copy_fn} entries[2];
}};
"""

    dispatch_header = f"""\
{seed_line}
#ifndef UNIT_{slug.upper()}_DISPATCH_H
#define UNIT_{slug.upper()}_DISPATCH_H
{include_bundle}
int dispatch_request(char *payload, int n);
int relay_input(char *payload);
int safe_copy(char *dst, const char *src, int n);
int bounded_copy(char *dst, const char *src, int n);
int orphan_copy(char *dst, char *src);
#endif
"""

    worker = f"""\
{seed_line}
{include_bundle}
{include_ops}
{include_dispatch}

char *g_sink;

int safe_copy(char *dst, const char *src, int n) {{
    return memcpy(dst, src, n); /* @call:unit_{slug}.safe_memcpy */
}}

int bounded_copy(char *dst, const char *src, int n) {{
    return memcpy(dst, src, n); /* @call:unit_{slug}.fp_memcpy */
}}

int orphan_copy(char *dst, char *src) {{
    return strcpy(dst, src); /* @call:unit_{slug}.orphan_strcpy */
}}

int relay_input(char *payload) {{
    return memcpy(g_sink, payload, 16); /* @call:unit_{slug}.param_memcpy */
}}
"""

    dispatch = f"""\
{seed_line}
{include_bundle}
{include_ops}
{include_dispatch}

static {copy_fn} hooks[] = {{ bounded_copy, safe_copy }};
struct {external_type} g_external;

int dispatch_request(char *payload, int n) {{
    char codebuf[64];
    hooks[0](codebuf, payload, n);
    char *net_payload = recv_from_socket(n, 16);
    safe_copy(codebuf, net_payload, 8);
    strcpy(codebuf, RECURSE_INPUT{function_tail}); /* @call:unit_{slug}.unresolved_strcpy */
    return n;
}}
"""

    main = f"""\
{seed_line}
{include_bundle}
{include_ops}
{include_dispatch}

static int helper_literal_sprintf(char *out) {{
    return FORWARD_FORMAT{function_tail}(sprintf, out, "literal-%d", 7); /* @call:unit_{slug}.literal_sprintf */
}}

static int helper_external(char *dst, char *src) {{
    return g_external.memcpy(dst, src, 8); /* @call:unit_{slug}.external_unknown */
}}

static int helper_system_literal(void) {{
    return system("echo audit"); /* @call:unit_{slug}.system_literal */
}}

int local_banner(char *out) {{
    const char *banner = NESTED_TEXT{function_tail}(AUDIT_SEED);
    out[0] = banner[0];
    return 0;
}}

int entry_unit_{slug}(int argc, char **argv) {{
    char buffer[128];
    char *payload = read_input(32);
    helper_literal_sprintf(buffer);
    helper_system_literal();
    local_banner(buffer);
    relay_input(payload);
    dispatch_request(payload, 32);
    helper_external(buffer, payload);
#ifdef ENABLE_UNSAFE
    strcpy(buffer, getenv("AUDIT_SECRET")); /* @call:unit_{slug}.cond_strcpy */
    system(argv[1]); /* @call:unit_{slug}.argv_system */
#endif
#if defined(FEATURE_ALT)
    sprintf(buffer, "%s", getenv("ALT_SECRET")); /* @call:unit_{slug}.alt_sprintf */
#endif
#if defined(FEATURE_B_ONLY)
    sprintf(buffer, "b:%s", getenv("B_ONLY_SECRET")); /* @call:unit_{slug}.b_only_sprintf */
#endif
    return argc + payload[0];
}}
"""

    hang = f"""\
{seed_line}
#pragma once
// @audit-hang deterministically generated timeout probe
#define HANG_PROBE{function_tail}(x) x
"""

    rel = lambda name: (Path(root_name) / name).as_posix()
    files = {
        rel(f"{include_dir}/{file_names['bundle']}"): bundle,
        rel(f"{include_dir}/{file_names['ops']}"): ops,
        rel(f"{src_dir}/{file_names['main']}"): main,
        rel(
            f"{src_dir}/{nested_dir}/{file_names['worker']}"
        ): worker,
        rel(
            f"{src_dir}/{nested_dir}/{deep_dir}/{file_names['dispatch']}"
        ): dispatch,
        rel(
            f"{src_dir}/{nested_dir}/{deep_dir}/{file_names['dispatch_header']}"
        ): dispatch_header,
    }
    if hang_in_bundle:
        files[rel(f"{include_dir}/{file_names['hang']}")] = hang
    return files


def _fixture_includes_cycle(seed: int, fixture_root: str) -> tuple[dict[str, str], dict[str, Any]]:
    prefix = f"/* seed={seed} include-cycle fixture */\n"
    files = {
        f"{fixture_root}/main.c": prefix
        + '#include "loop_a.h"\nint main(void) { return 0; }\n',
        f"{fixture_root}/loop_a.h": prefix
        + '#include "loop_b.h"\n#ifndef LOOP_A_H\n#define LOOP_A_H\nint loop_a(void);\n#endif\n',
        f"{fixture_root}/loop_b.h": prefix
        + '#ifndef LOOP_B_H\n#define LOOP_B_H\n#include "loop_a.h"\nint loop_b(void);\n#endif\n',
    }
    config = {
        "name": "fixture_include_cycle",
        "macros": [],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["main"],
    }
    return files, config


def _fixture_macro_conflict(seed: int, fixture_root: str) -> tuple[dict[str, str], dict[str, Any]]:
    text = (
        f"/* seed={seed} macro-conflict fixture */\n"
        "int main(void) { return 0; }\n"
    )
    config = {
        "name": "fixture_macro_conflict",
        "macros": [
            {"name": "CONFLICT_MODE", "defined": True, "value": "1"},
            {"name": "CONFLICT_MODE", "defined": True, "value": "2"},
        ],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["main"],
    }
    return {f"{fixture_root}/main.c": text}, config


def _fixture_parse_failure(seed: int, fixture_root: str) -> tuple[dict[str, str], dict[str, Any]]:
    text = f"""\
/* seed={seed} parse-failure fixture */
#define RECUR(x) RECUR(x)
int main(void) {{
    char buf[32];
    char *guess = RECUR(secret);
    strcpy(buf, RECUR(secret)); /* @call:parse_failure.unresolved */
    sprintf(buf, "still-visible"); /* @call:parse_failure.visible */
    return 0;
}}
"""
    config = {
        "name": "fixture_parse_failure",
        "macros": [],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["main"],
    }
    return {f"{fixture_root}/main.c": text}, config


def _fixture_parse_limit(seed: int, fixture_root: str) -> tuple[dict[str, str], dict[str, Any]]:
    files: dict[str, str] = {
        f"{fixture_root}/main.c": (
            f"/* seed={seed} parse-limit fixture */\n"
            '#include "chain_0.h"\n'
            "int main(void) { return 0; }\n"
        )
    }
    for index in range(24):
        next_include = (
            f'#include "chain_{index + 1}.h"\n' if index < 23 else ""
        )
        files[f"{fixture_root}/chain_{index}.h"] = (
            f"/* chain {index} */\n#ifndef CHAIN_{index}_H\n#define CHAIN_{index}_H\n"
            f"{next_include}#endif\n"
        )
    config = {
        "name": "fixture_parse_limit",
        "macros": [],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["main"],
    }
    return files, config


def _fixture_deep_path(seed: int, fixture_root: str) -> tuple[dict[str, str], dict[str, Any]]:
    functions = []
    for index in range(12):
        if index == 0:
            body = "    return hop_1(input);\n"
        elif index < 11:
            body = f"    return hop_{index + 1}(input);\n"
        else:
            body = (
                "    char *out = g_sink;\n"
                "    return memcpy(out, input, 8); /* @call:deep_path.memcpy */\n"
            )
        functions.append(
            f"static int hop_{index}(char *input) {{\n{body}}}\n"
        )
    text = (
        f"/* seed={seed} deep-path fixture */\n"
        "extern char *g_sink;\n"
        + "".join(functions)
        + "int entry_deep(int argc, char **argv) {\n"
        "    (void)argc; (void)argv;\n"
        "    return hop_0(\"literal\");\n"
        "}\n"
    )
    config = {
        "name": "fixture_deep_path",
        "macros": [],
        "include_paths": [],
        "language_standard": "c11",
        "entries": ["entry_deep"],
    }
    return {f"{fixture_root}/main.c": text}, config


def _build_fixtures(
    seed: int,
    out: Path,
    inc_dir: str,
    src_dir: str,
) -> dict[str, Any]:
    fixture_specs: dict[str, tuple[dict[str, str], dict[str, Any]]] = {}
    root_rel = f"fixtures/include_cycle/{inc_dir}"
    files, cfg = _fixture_includes_cycle(seed, f"fixtures/include_cycle/{src_dir}")
    fixture_specs["include_cycle"] = (files, cfg)

    root_rel = f"fixtures/macro_conflict/{src_dir}"
    files, cfg = _fixture_macro_conflict(seed, f"fixtures/macro_conflict/{src_dir}")
    fixture_specs["macro_conflict"] = (files, cfg)

    files, cfg = _fixture_parse_failure(seed, f"fixtures/parse_failure/{src_dir}")
    fixture_specs["parse_failure"] = (files, cfg)

    files, cfg = _fixture_parse_limit(seed, f"fixtures/parse_limit/{src_dir}")
    fixture_specs["parse_limit"] = (files, cfg)

    files, cfg = _fixture_deep_path(seed, f"fixtures/deep_path/{src_dir}")
    fixture_specs["deep_path"] = (files, cfg)

    manifest: dict[str, Any] = {}
    for name, (files, config) in fixture_specs.items():
        for relative, text in sorted(files.items()):
            _write(out / relative, text)
        config_path = out / "fixtures" / name / "config.json"
        _write(
            config_path,
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        )
        root_dir = next(iter(sorted(files))).split("/")[0:3]
        root_path = "/".join(root_dir)
        manifest[name] = {
            "root": root_path,
            "config": config_path.relative_to(out).as_posix(),
            "files": sorted(files),
        }
    return manifest


def generate(seed: int, out: Path, force: bool = False) -> dict[str, Any]:
    """Generate a complete deterministic bundle and return its manifest."""

    out = Path(out)
    if out.exists() and not out.is_dir():
        raise WorkbenchError(
            "E_IO",
            "generation output exists and is not a directory",
            {"file": str(out), "line": 1},
        )
    if out.exists() and any(out.iterdir()):
        generated = (out / GENERATED_MARKER).exists()
        if generated or force:
            shutil.rmtree(out)
        else:
            raise WorkbenchError(
                "E_IO",
                "refusing to overwrite an existing non-generated directory (use --force)",
                {"file": str(out), "line": 1},
            )
    out.mkdir(parents=True, exist_ok=True)

    inc_dir = f"include_{deterministic_token(seed, 'include-dir')}"
    src_dir = f"src_{deterministic_token(seed, 'source-dir')}"
    nested_dir = f"nested_{deterministic_token(seed, 'nested-dir')}"
    deep_dir = f"deep_{deterministic_token(seed, 'deep-dir')}"
    file_token = deterministic_token(seed, "file-names", 10)
    file_names = {
        "bundle": f"bundle_{file_token}.h",
        "ops": f"ops_{file_token}.h",
        "main": f"main_{file_token}.c",
        "worker": f"worker_{file_token}.c",
        "dispatch": f"dispatch_{file_token}.c",
        "dispatch_header": f"dispatch_{file_token}.h",
        "hang": f"hang_{file_token}.h",
    }
    include_paths = [
        inc_dir,
        src_dir,
        f"{src_dir}/{nested_dir}",
        f"{src_dir}/{nested_dir}/{deep_dir}",
    ]

    roots: list[dict[str, Any]] = []
    all_files: dict[str, str] = {}
    for offset, slug in enumerate(("a", "b", "c")):
        root_name = f"root_{deterministic_token(seed, f'root-dir-{slug}')}_{slug}"
        files = _root_files(
            slug=slug,
            seed=seed,
            root_name=root_name,
            include_dir=inc_dir,
            src_dir=src_dir,
            nested_dir=nested_dir,
            deep_dir=deep_dir,
            file_names=file_names,
            hang_in_bundle=(slug == "c"),
        )
        all_files.update(files)
        roots.append(
            {
                "name": f"unit_{slug}",
                "path": root_name,
                "entry": f"entry_unit_{slug}",
                "files": sorted(files),
                "file_count": len(files),
                "max_directory_depth": max(
                    len(Path(relative).parts) - 2
                    for relative in files
                ),
            }
        )

    for relative, text in sorted(all_files.items()):
        _write(out / relative, text)

    for config_name in ("cfgA", "cfgB", "cfgC"):
        config_text = _config_text(config_name, include_paths, seed)
        _write(out / "configs" / f"{config_name}.json", config_text)

    fixtures = _build_fixtures(seed, out, inc_dir, src_dir)

    # Compute expected marker locations from the generated files on disk.
    expected: dict[str, Any] = {}
    for root in roots:
        slug = root["name"].split("_")[1]
        marker_file = next(
            file
            for file in root["files"]
            if file.endswith(file_names["main"])
        )
        worker_file = next(
            file for file in root["files"] if file.endswith(file_names["worker"])
        )
        dispatch_file = next(
            file
            for file in root["files"]
            if file.endswith(file_names["dispatch"])
        )
        main_text = (out / marker_file).read_text(encoding="utf-8")
        worker_text = (out / worker_file).read_text(encoding="utf-8")
        dispatch_text = (out / dispatch_file).read_text(encoding="utf-8")
        prefix = root["path"] + "/"
        marker_file_report = (
            marker_file[len(prefix) :] if marker_file.startswith(prefix) else marker_file
        )
        worker_file_report = (
            worker_file[len(prefix) :] if worker_file.startswith(prefix) else worker_file
        )
        dispatch_file_report = (
            dispatch_file[len(prefix) :] if dispatch_file.startswith(prefix) else dispatch_file
        )
        def location(text: str, marker: str) -> tuple[int, int]:
            for line_number, line in enumerate(text.splitlines(), 1):
                if marker in line:
                    needle = marker.split(".")[-1]
                    if needle == "unresolved_strcpy":
                        needle = "strcpy"
                    elif needle in {"param_memcpy", "fp_memcpy", "safe_memcpy"}:
                        needle = "memcpy"
                    elif needle == "orphan_strcpy":
                        needle = "strcpy"
                    elif needle == "literal_sprintf":
                        needle = "FORWARD_FORMAT"
                    elif needle == "external_unknown":
                        needle = "memcpy"
                    elif needle == "system_literal":
                        needle = "system"
                    elif needle == "cond_strcpy":
                        needle = "strcpy"
                    elif needle == "argv_system":
                        needle = "system"
                    elif needle == "alt_sprintf":
                        needle = "sprintf"
                    return line_number, line.find(needle) + 1
            return 1, 1

        expected[f"unit_{slug}"] = {
            "conditional_call": {
                "id": f"unit_{slug}.cond_strcpy",
                "file": marker_file_report,
                "line": location(main_text, f"@call:unit_{slug}.cond_strcpy")[0],
                "column": location(main_text, f"@call:unit_{slug}.cond_strcpy")[1],
                "exists_in": ["cfgA"],
                "absent_in": ["cfgB"],
            },
            "b_only_call": {
                "id": f"unit_{slug}.b_only_sprintf",
                "file": marker_file_report,
                "line": location(main_text, f"@call:unit_{slug}.b_only_sprintf")[0],
                "column": location(main_text, f"@call:unit_{slug}.b_only_sprintf")[1],
                "exists_in": ["cfgB"],
                "absent_in": ["cfgA"],
            },
            "literal_sprintf": {
                "id": f"unit_{slug}.literal_sprintf",
                "file": marker_file_report,
                "line": location(main_text, f"@call:unit_{slug}.literal_sprintf")[0],
                "column": location(main_text, f"@call:unit_{slug}.literal_sprintf")[1],
            },
            "unreachable": {
                "id": f"unit_{slug}.orphan_strcpy",
                "file": worker_file_report,
                "line": location(worker_text, f"@call:unit_{slug}.orphan_strcpy")[0],
                "column": location(worker_text, f"@call:unit_{slug}.orphan_strcpy")[1],
            },
            "fp_reachable": {
                "id": f"unit_{slug}.fp_memcpy",
                "file": worker_file_report,
                "line": location(worker_text, f"@call:unit_{slug}.fp_memcpy")[0],
                "column": location(worker_text, f"@call:unit_{slug}.fp_memcpy")[1],
            },
            "indirect_unknown": {
                "id": f"unit_{slug}.external_unknown",
                "file": marker_file_report,
                "line": location(main_text, f"@call:unit_{slug}.external_unknown")[0],
                "column": location(main_text, f"@call:unit_{slug}.external_unknown")[1],
            },
            "literal_untainted": {
                "id": f"unit_{slug}.literal_sprintf",
            },
            "param_tainted": {
                "id": f"unit_{slug}.param_memcpy",
                "file": worker_file_report,
                "line": location(worker_text, f"@call:unit_{slug}.param_memcpy")[0],
                "column": location(worker_text, f"@call:unit_{slug}.param_memcpy")[1],
            },
            "unresolved_taint": {
                "id": f"unit_{slug}.unresolved_strcpy",
                "file": dispatch_file_report,
                "line": location(dispatch_text, f"@call:unit_{slug}.unresolved_strcpy")[0],
                "column": location(dispatch_text, f"@call:unit_{slug}.unresolved_strcpy")[1],
            },
        }

    manifest = {
        "format": "audit-samples-v1",
        "seed": seed,
        "layout": {
            "include_dir": inc_dir,
            "source_dir": src_dir,
            "nested_dir": nested_dir,
            "deep_dir": deep_dir,
            "include_paths": include_paths,
        },
        "roots": roots,
        "configs": ["configs/cfgA.json", "configs/cfgB.json", "configs/cfgC.json"],
        "expected": expected,
        "fixtures": fixtures,
        "notes": {
            "timeout": "unit_c with cfgC includes the @audit-hang fixture",
            "unreachable_transition": "call orphan_copy from entry_unit_<slug> to make it reachable",
        },
    }
    _write(out / "roots.json", json.dumps(manifest, indent=2, sort_keys=True))
    _write(out / GENERATED_MARKER, "generated by audit_workbench; do not edit\n")
    return manifest

