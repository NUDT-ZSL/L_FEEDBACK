from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from . import SCHEMA_VERSION
from .analysis import analyze, diff_reports
from .job import run_isolated
from .pdb import PDBFormatError, PDBMismatch, load_pdb
from .pe import PEFormatError, load_pe


DEFAULT_MEM_LIMIT = 512 * 1024 * 1024
DEFAULT_CPU_LIMIT_MS = 120000


def _emit(report: Dict[str, Any], out_path: Optional[str]) -> None:
    text = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    sys.stdout.write(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


def _error_report(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pe_info": None,
        "pdb_info": None,
        "analysis_stats": {
            "scanned_functions": 0,
            "evaluated_paths": 0,
            "elapsed_ms": 0,
            "truncated_count": 0,
            "unknown_propagation": 0,
        },
        "sinks": [],
        "warnings": [],
        "errors": [{"code": code, "message": message}],
        "passed": False,
    }
    report.update(extra)
    return report


def _pdb_mismatch_report(fields: Dict[str, Any]) -> Dict[str, Any]:
    errors = [dict(fields)]
    report = _error_report(fields.get("code", "PDB_MISMATCH"), fields.get("message", "PDB mismatch"))
    report["errors"] = errors
    report.pop("sinks", None)
    for key in ("pe_guid", "pdb_guid", "pe_age", "pdb_age"):
        report[key] = fields.get(key)
    return report


def _section_dicts(pe) -> List[Dict[str, Any]]:
    return [
        {
            "name": section.name,
            "rva": section.rva,
            "virtual_size": section.virtual_size,
            "raw_size": section.raw_size,
            "raw_ptr": section.raw_ptr,
            "characteristics": section.characteristics,
        }
        for section in sorted(pe.sections, key=lambda item: item.rva)
    ]


def _parse_report(pe, pdb=None) -> Dict[str, Any]:
    sections = _section_dicts(pe)
    functions = []
    pdata = sorted(pe.functions, key=lambda item: (item["rva"], item["end_rva"]))
    if pdb is not None and pdata:
        for item in pdata:
            match = None
            for function in pdb.functions:
                if function.rva == item["rva"]:
                    match = function
                    break
            if match is None:
                for function in pdb.functions:
                    if function.rva <= item["rva"] < function.rva + max(1, function.size):
                        match = function
                        break
            functions.append(
                {
                    "name": match.name if match else "sub_%X" % item["rva"],
                    "rva": item["rva"],
                    "end_rva": item["end_rva"],
                    "size": item["size"],
                    "unwind_rva": item["unwind_rva"],
                }
            )
    elif pdata:
        functions = [
            {
                "name": "sub_%X" % item["rva"],
                "rva": item["rva"],
                "end_rva": item["end_rva"],
                "size": item["size"],
                "unwind_rva": item["unwind_rva"],
            }
            for item in pdata
        ]
    elif pdb is not None:
        functions = [
            {
                "name": function.name,
                "rva": function.rva,
                "end_rva": function.rva + function.size,
                "size": function.size,
                "type_index": function.type_index,
                "param_count": function.param_count,
                "convention": function.convention,
            }
            for function in sorted(pdb.functions, key=lambda item: (item.rva, item.size, item.name))
        ]
    errors = [dict(item) for item in pe.errors]
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "pe_info": {
            "machine": pe.machine,
            "magic": "PE32+" if pe.is_pe32_plus else "PE32",
            "image_base": pe.image_base,
            "entry_point_rva": pe.entry_point,
            "size_of_image": pe.size_of_image,
            "timestamp": pe.timestamp,
            "codeview_guid": pe.codeview_guid,
            "codeview_age": pe.codeview_age,
            "codeview_path": pe.codeview_path,
        },
        "pe_sections": sections,
        "imports": list(pe.imports),
        "exports": list(pe.exports),
        "functions": functions,
        "base_relocations": list(pe.base_relocations),
        "errors": errors,
        "passed": not errors,
    }
    if pdb is not None:
        report["pdb_info"] = {
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
        report["pdb_functions"] = [
            {
                "name": function.name,
                "rva": function.rva,
                "end_rva": function.rva + function.size,
                "size": function.size,
                "segment": function.segment,
                "offset": function.offset,
                "type_index": function.type_index,
                "param_count": function.param_count,
                "convention": function.convention,
                "kind": function.kind,
            }
            for function in sorted(pdb.functions, key=lambda item: (item.rva, item.size, item.name))
        ]
        report["pdb_public_symbols"] = [
            {
                "name": symbol.name,
                "rva": symbol.rva,
                "segment": symbol.segment,
                "offset": symbol.offset,
                "flags": symbol.flags,
            }
            for symbol in sorted(pdb.public_symbols, key=lambda item: (item.rva, item.name))
        ]
    return report


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("config root must be an object")
    return value


def _analysis_report(pe_path: str, pdb_path: str, config_path: str) -> Tuple[Optional[Dict[str, Any]], int]:
    try:
        pe = load_pe(pe_path)
        pdb = load_pdb(pdb_path, pe)
    except PDBMismatch as exc:
        return _pdb_mismatch_report(exc.fields), 2
    except PEFormatError as exc:
        return _error_report(exc.code, str(exc)), 2
    except PDBFormatError as exc:
        return _error_report(exc.code, str(exc)), 2
    try:
        config = _load_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _error_report("INVALID_CONFIG", "cannot read config: %s" % exc), 2
    try:
        report = analyze(pe, pdb, config)
    except ValueError as exc:
        return _error_report("INVALID_CONFIG", str(exc)), 2
    return report, 0 if report.get("passed", False) else 2


def _infer_pdb(exe_path: str, explicit: Optional[str], other: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    for candidate in (os.path.splitext(exe_path)[0] + ".pdb",):
        if os.path.exists(candidate):
            return candidate
    return other


def _diff_report(args) -> Tuple[Dict[str, Any], int]:
    if args.baseline:
        try:
            with open(args.baseline, "r", encoding="utf-8-sig") as handle:
                baseline = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return (
                _error_report("BASELINE_INVALID", "cannot read baseline: %s" % exc),
                2,
            )
        if not isinstance(baseline, dict) or baseline.get("schema_version") != SCHEMA_VERSION:
            report = _error_report(
                "BASELINE_VERSION_MISMATCH",
                "baseline schema_version is not compatible with %s" % SCHEMA_VERSION,
            )
            report["baseline_schema_version"] = (
                baseline.get("schema_version") if isinstance(baseline, dict) else None
            )
            return report, 2
        if not args.after:
            return _error_report("INVALID_ARGUMENTS", "--after is required with --baseline"), 2
        after_pdb = _infer_pdb(args.after, args.after_pdb, args.pdb)
        if not after_pdb:
            return _error_report("INVALID_PDB", "cannot infer the after-image PDB"), 2
        after_report, after_code = _analysis_report(args.after, after_pdb, args.config)
        if after_code == 2:
            return after_report or _error_report("ANALYSIS_FAILED", "after analysis failed"), 2
        result = diff_reports(baseline, after_report or {})
        return result, 1 if result["regressed"] else 0

    if not args.before or not args.after:
        return _error_report("INVALID_ARGUMENTS", "diff requires --before and --after"), 2
    before_pdb = _infer_pdb(args.before, args.pdb, None)
    after_pdb = _infer_pdb(args.after, args.after_pdb, before_pdb)
    if not before_pdb or not after_pdb:
        return _error_report("INVALID_PDB", "cannot infer PDB paths for diff"), 2
    before_report, before_code = _analysis_report(args.before, before_pdb, args.config)
    if before_code == 2:
        return before_report or _error_report("ANALYSIS_FAILED", "before analysis failed"), 2
    after_report, after_code = _analysis_report(args.after, after_pdb, args.config)
    if after_code == 2:
        return after_report or _error_report("ANALYSIS_FAILED", "after analysis failed"), 2
    result = diff_reports(before_report or {}, after_report or {})
    return result, 1 if result["regressed"] else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfaudit")
    sub = parser.add_subparsers(dest="command", required=True)

    parse = sub.add_parser("parse")
    parse.add_argument("--pe", required=True)
    parse.add_argument("--pdb")
    parse.add_argument("--out")

    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--pe", required=True)
    analyze_parser.add_argument("--pdb", required=True)
    analyze_parser.add_argument("--config", required=True)
    analyze_parser.add_argument("--out")
    analyze_parser.add_argument("--mem-limit", type=int, default=DEFAULT_MEM_LIMIT)
    analyze_parser.add_argument("--cpu-limit-ms", type=int, default=DEFAULT_CPU_LIMIT_MS)

    diff = sub.add_parser("diff")
    diff.add_argument("--before")
    diff.add_argument("--after")
    diff.add_argument("--pdb")
    diff.add_argument("--after-pdb")
    diff.add_argument("--config")
    diff.add_argument("--baseline")
    diff.add_argument("--out")
    diff.add_argument("--mem-limit", type=int, default=DEFAULT_MEM_LIMIT)
    diff.add_argument("--cpu-limit-ms", type=int, default=DEFAULT_CPU_LIMIT_MS)
    return parser


def _execute(args, worker: bool = False) -> Tuple[Dict[str, Any], int]:
    if args.command == "parse":
        try:
            pe = load_pe(args.pe)
        except PEFormatError as exc:
            return _error_report(exc.code, str(exc)), 2
        pdb = None
        if args.pdb:
            try:
                pdb = load_pdb(args.pdb, pe)
            except PDBMismatch as exc:
                return _pdb_mismatch_report(exc.fields), 2
            except PDBFormatError as exc:
                return _error_report(exc.code, str(exc)), 2
        report = _parse_report(pe, pdb)
        return report, 0 if report["passed"] else 2
    if args.command == "analyze":
        return _analysis_report(args.pe, args.pdb, args.config)
    if args.command == "diff":
        if args.config is None:
            return _error_report("INVALID_ARGUMENTS", "--config is required"), 2
        return _diff_report(args)
    return _error_report("INVALID_ARGUMENTS", "unknown command"), 2


def _strip_out(argv: List[str]) -> List[str]:
    result = []
    index = 0
    while index < len(argv):
        if argv[index] == "--out":
            index += 2
            continue
        result.append(argv[index])
        index += 1
    return result


def _limits_for_args(args) -> Tuple[int, int]:
    mem_limit = int(getattr(args, "mem_limit", DEFAULT_MEM_LIMIT) or DEFAULT_MEM_LIMIT)
    cpu_limit_ms = int(getattr(args, "cpu_limit_ms", DEFAULT_CPU_LIMIT_MS) or DEFAULT_CPU_LIMIT_MS)
    return max(1024 * 1024, mem_limit), max(1, cpu_limit_ms)


def _worker_main(argv: List[str]) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        report, code = _execute(args, worker=True)
    except MemoryError:
        from .job import _resource_report

        sys.stdout.write(_resource_report("memory") + "\n")
        return 3
    except Exception as exc:
        report = _error_report("INTERNAL_ERROR", str(exc))
        code = 2
    _emit(report, None)
    return code


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return _worker_main(argv[1:])
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if args.command in ("analyze", "diff"):
        worker_args = _strip_out(argv)
        mem_limit, cpu_limit_ms = _limits_for_args(args)
        module_file = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        try:
            return run_isolated(module_file, worker_args, mem_limit, cpu_limit_ms, args.out)
        except Exception as exc:
            report = _error_report("INTERNAL_ERROR", str(exc))
            _emit(report, args.out)
            return 2
    try:
        report, code = _execute(args, worker=False)
    except Exception as exc:
        report = _error_report("INTERNAL_ERROR", str(exc))
        code = 2
    _emit(report, args.out)
    return code

