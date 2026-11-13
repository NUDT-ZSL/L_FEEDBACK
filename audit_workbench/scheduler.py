"""Real subprocess isolation and bounded parallel scheduling."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from .config import AuditConfig
from .errors import AuditError, WorkbenchError
from .hashing import sha256_text


_active_lock = threading.Lock()
_active_children = 0


def _increment_active() -> None:
    global _active_children
    with _active_lock:
        _active_children += 1


def _decrement_active() -> None:
    global _active_children
    with _active_lock:
        _active_children -= 1


def active_child_count() -> int:
    with _active_lock:
        return _active_children


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=0.8)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def _empty_summary() -> dict[str, Any]:
    return {
        "calls_total": 0,
        "reachability": {"reachable": 0, "unreachable": 0, "unknown": 0},
        "taint": {"tainted": 0, "untainted": 0, "unknown": 0},
        "high_risk": 0,
        "entry_missing": [],
    }


def _timeout_result(label: str, config: AuditConfig, timeout: float) -> dict[str, Any]:
    combination = f"{label}/{config.name}"
    error = AuditError(
        code="E_TIMEOUT",
        message=f"combination exceeded {timeout:g}s wall-clock timeout",
        context={
            "combination": combination,
            "root": label,
            "config": config.name,
            "timeout_seconds": timeout,
        },
    ).to_dict()
    return {
        "status": "timed_out",
        "calls": [],
        "summary": _empty_summary(),
        "parse_failures": [],
        "parse_failure_count": 0,
        "included_files": [],
        "macro_expansions": 0,
        "translation_units": [],
        "errors": [error],
    }


def run_combination_subprocess(
    *,
    root_path: Path,
    label: str,
    config: AuditConfig,
    timeout: float,
    state_dir: Path,
    max_hops: int,
    ordinal: int,
) -> dict[str, Any]:
    tmp_dir = Path(state_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_dir / f"combo-{ordinal}-{os.getpid()}.payload.json"
    result_path = tmp_dir / f"combo-{ordinal}-{os.getpid()}.result.json"
    stdout_path = tmp_dir / f"combo-{ordinal}-{os.getpid()}.stdout.log"
    stderr_path = tmp_dir / f"combo-{ordinal}-{os.getpid()}.stderr.log"
    payload = {
        "root": str(Path(root_path).resolve()),
        "root_label": label,
        "config": config.to_dict(),
        "max_hops": max_hops,
    }
    try:
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        return {
            "status": "error",
            "calls": [],
            "summary": _empty_summary(),
            "parse_failures": [],
            "parse_failure_count": 0,
            "included_files": [],
            "macro_expansions": 0,
            "translation_units": [],
            "errors": [
                AuditError(
                    "E_IO",
                    f"cannot write worker payload: {exc}",
                    {"combination": f"{label}/{config.name}", "file": str(payload_path), "line": 1},
                ).to_dict()
            ],
        }

    command = [
        sys.executable,
        "-m",
        "audit_workbench",
        "_worker",
        str(payload_path),
        str(result_path),
    ]
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        start_new_session = True
    result: dict[str, Any]
    _increment_active()
    try:
        with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
            proc = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc)
                result = _timeout_result(label, config, timeout)
                result_path.unlink(missing_ok=True)
                return result
        if result_path.is_file():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                result = {
                    "status": "error",
                    "calls": [],
                    "summary": _empty_summary(),
                    "parse_failures": [],
                    "parse_failure_count": 0,
                    "included_files": [],
                    "macro_expansions": 0,
                    "translation_units": [],
                    "errors": [
                        AuditError(
                            "E_WORKER",
                            f"worker result is unreadable: {exc}",
                            {"combination": f"{label}/{config.name}", "line": 1},
                        ).to_dict()
                    ],
                }
        else:
            result = {
                "status": "error",
                "calls": [],
                "summary": _empty_summary(),
                "parse_failures": [],
                "parse_failure_count": 0,
                "included_files": [],
                "macro_expansions": 0,
                "translation_units": [],
                "errors": [
                    AuditError(
                        "E_WORKER",
                        f"worker exited without a result (exit={proc.returncode})",
                        {"combination": f"{label}/{config.name}", "line": 1},
                    ).to_dict()
                ],
            }
        return result
    except OSError as exc:
        return {
            "status": "error",
            "calls": [],
            "summary": _empty_summary(),
            "parse_failures": [],
            "parse_failure_count": 0,
            "included_files": [],
            "macro_expansions": 0,
            "translation_units": [],
            "errors": [
                AuditError(
                    "E_IO",
                    f"cannot start worker subprocess: {exc}",
                    {"combination": f"{label}/{config.name}", "line": 1},
                ).to_dict()
            ],
        }
    finally:
        _decrement_active()
        for path in (payload_path, result_path, stdout_path, stderr_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def run_combinations(
    combinations: Iterable[tuple[str, Path, AuditConfig]],
    *,
    jobs: int,
    timeout: float,
    state_dir: Path,
    max_hops: int,
) -> list[tuple[str, Path, AuditConfig, dict[str, Any]]]:
    combos = list(combinations)
    results: list[tuple[str, Path, AuditConfig, dict[str, Any]] | None] = [None] * len(combos)

    def task(index: int) -> None:
        label, root_path, config = combos[index]
        try:
            combo_result = run_combination_subprocess(
                root_path=root_path,
                label=label,
                config=config,
                timeout=timeout,
                state_dir=state_dir,
                max_hops=max_hops,
                ordinal=index,
            )
        except Exception as exc:  # pragma: no cover - defensive isolation boundary
            combo_result = {
                "status": "error",
                "calls": [],
                "summary": _empty_summary(),
                "parse_failures": [],
                "parse_failure_count": 0,
                "included_files": [],
                "macro_expansions": 0,
                "translation_units": [],
                "errors": [
                    AuditError(
                        "E_WORKER",
                        f"supervisor failed: {type(exc).__name__}: {exc}",
                        {"combination": f"{label}/{config.name}", "line": 1},
                    ).to_dict()
                ],
            }
        results[index] = (label, root_path, config, combo_result)

    with ThreadPoolExecutor(max_workers=max(1, jobs), thread_name_prefix="audit-supervisor") as pool:
        list(pool.map(task, range(len(combos))))
    return [item for item in results if item is not None]

