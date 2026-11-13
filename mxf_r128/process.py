"""Subprocess execution with hard timeouts and whole-process-tree cleanup."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool


def _popen_options() -> dict[str, object]:
    if os.name == "nt":
        flags = 0
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"creationflags": flags}
    return {"start_new_session": True}


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass

    try:
        process.kill()
    except OSError:
        pass


def run_command(
    args: Sequence[str],
    *,
    timeout: float,
    env: Mapping[str, str] | None = None,
    capture_output: bool = True,
) -> CommandResult:
    """Run a command and kill the complete process group on timeout."""

    popen_kwargs = _popen_options()
    if capture_output:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE
    else:
        popen_kwargs["stdout"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = subprocess.DEVNULL

    try:
        process = subprocess.Popen(
            list(args),
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=dict(env) if env is not None else None,
            **popen_kwargs,
        )
    except OSError as exc:
        return CommandResult(
            returncode=None,
            stdout="",
            stderr=type(exc).__name__,
            timed_out=False,
        )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return CommandResult(
            returncode=None,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=True,
        )

    return CommandResult(
        returncode=process.returncode,
        stdout=stdout or "",
        stderr=stderr or "",
        timed_out=False,
    )


def run_filtered_command(
    args: Iterable[str],
    *,
    timeout: float,
    env: Mapping[str, str] | None = None,
) -> CommandResult:
    return run_command(args, timeout=timeout, env=env, capture_output=False)
