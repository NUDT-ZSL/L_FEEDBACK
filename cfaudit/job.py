from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from typing import List, Optional, Tuple


JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodUserTime", ctypes.c_int64),
        ("ThisPeriodKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


def _resource_report(resource: str) -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
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
            "errors": [
                {
                    "code": "RESOURCE_LIMIT",
                    "resource": resource,
                    "message": "analysis process was terminated by the Windows job object",
                }
            ],
            "passed": False,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _valid_json(stdout: bytes) -> Optional[dict]:
    try:
        value = json.loads(stdout.decode("utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


class WindowsJob:
    def __init__(self, mem_limit: int, cpu_limit_ms: int):
        if os.name != "nt":
            raise OSError("WindowsJob is only available on Windows")
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.job = self.kernel32.CreateJobObjectW(None, None)
        if not self.job:
            raise OSError("CreateJobObjectW failed")
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_JOB_TIME
            | JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        info.BasicLimitInformation.PerJobUserTimeLimit = max(1, cpu_limit_ms) * 10000
        info.ProcessMemoryLimit = mem_limit
        ok = self.kernel32.SetInformationJobObject(
            self.job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            self.close()
            raise OSError("SetInformationJobObject failed")
        self.mem_limit = mem_limit
        self.cpu_limit_ms = cpu_limit_ms
        self.assigned = False

    def assign(self, process) -> None:
        handle = wintypes.HANDLE(int(process._handle))
        if not self.kernel32.AssignProcessToJobObject(self.job, handle):
            self.close()
            raise OSError("AssignProcessToJobObject failed")
        self.assigned = True

    def accounting(self) -> Optional[_JOBOBJECT_BASIC_ACCOUNTING_INFORMATION]:
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        returned = ctypes.c_uint32(0)
        ok = self.kernel32.QueryInformationJobObject(
            self.job,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        )
        return info if ok else None

    def peaks(self) -> Tuple[int, int]:
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        returned = ctypes.c_uint32(0)
        ok = self.kernel32.QueryInformationJobObject(
            self.job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info),
            ctypes.sizeof(info),
            ctypes.byref(returned),
        )
        if not ok:
            return (0, 0)
        return (int(info.PeakProcessMemoryUsed), int(info.PeakJobMemoryUsed))

    def terminate(self, exit_code: int = 3) -> None:
        if self.job:
            self.kernel32.TerminateJobObject(self.job, exit_code)

    def close(self) -> None:
        if getattr(self, "job", None):
            self.kernel32.CloseHandle(self.job)
            self.job = None


def _posix_limits(mem_limit: int, cpu_limit_ms: int):
    def apply_limits():
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
        except (ValueError, OSError):
            pass
        cpu_seconds = max(1, int((cpu_limit_ms + 999) // 1000))
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except (ValueError, OSError):
            pass

    return apply_limits


def run_isolated(
    package_root: str,
    command_args: List[str],
    mem_limit: int,
    cpu_limit_ms: int,
    output_path: Optional[str],
) -> int:
    worker_args = [sys.executable, "-m", "cfaudit", "_worker"] + list(command_args)
    env = os.environ.copy()
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if os.name == "nt":
        return _run_windows_job(worker_args, env, mem_limit, cpu_limit_ms, output_path)
    return _run_posix(worker_args, env, mem_limit, cpu_limit_ms, output_path)


def _write_output(stdout: bytes, output_path: Optional[str]) -> None:
    if not output_path:
        return
    with open(output_path, "wb") as handle:
        handle.write(stdout)


def _run_windows_job(
    worker_args: List[str],
    env: dict,
    mem_limit: int,
    cpu_limit_ms: int,
    output_path: Optional[str],
) -> int:
    job = WindowsJob(mem_limit, cpu_limit_ms)
    process = None
    try:
        process = subprocess.Popen(
            worker_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=env,
        )
        job.assign(process)
        reason = None
        started_wall = time.monotonic()
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.02)
                break
            except subprocess.TimeoutExpired:
                accounting = job.accounting()
                peaks = job.peaks()
                job_user_time = accounting.TotalUserTime if accounting is not None else 0
                peak_memory = max(peaks[0], peaks[1])
                if job_user_time >= max(1, cpu_limit_ms) * 10000:
                    reason = "time"
                if peak_memory >= mem_limit:
                    reason = "memory"
                if reason is not None:
                    job.terminate(3)
                    try:
                        stdout, stderr = process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                    break
                if time.monotonic() - started_wall > max(30.0, cpu_limit_ms / 1000.0 * 10.0):
                    reason = "time"
                    job.terminate(3)
                    try:
                        stdout, stderr = process.communicate(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                    break
        del stderr
        parsed = _valid_json(stdout)
        if parsed is not None:
            _write_output(stdout, output_path)
            sys.stdout.write(stdout.decode("utf-8", "replace"))
            return int(process.returncode)
        accounting = job.accounting()
        peaks = job.peaks()
        peak_memory = max(peaks[0], peaks[1])
        user_time = accounting.TotalUserTime if accounting is not None else 0
        if reason is None:
            if peak_memory >= mem_limit:
                reason = "memory"
            elif user_time >= max(1, cpu_limit_ms) * 10000:
                reason = "time"
            else:
                reason = "time"
        payload = _resource_report(reason).encode("utf-8") + b"\n"
        _write_output(payload, output_path)
        sys.stdout.write(payload.decode("utf-8"))
        return 3
    finally:
        if process is not None and process.poll() is None:
            job.terminate(3)
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        job.close()


def _run_posix(
    worker_args: List[str],
    env: dict,
    mem_limit: int,
    cpu_limit_ms: int,
    output_path: Optional[str],
) -> int:
    process = subprocess.Popen(
        worker_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=_posix_limits(mem_limit, cpu_limit_ms),
        env=env,
    )
    try:
        stdout, stderr = process.communicate(timeout=max(30.0, cpu_limit_ms / 1000.0 * 2.0))
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        payload = _resource_report("time").encode("utf-8") + b"\n"
        _write_output(payload, output_path)
        sys.stdout.write(payload.decode("utf-8"))
        return 3
    del stderr
    parsed = _valid_json(stdout)
    if parsed is not None:
        _write_output(stdout, output_path)
        sys.stdout.write(stdout.decode("utf-8", "replace"))
        return int(process.returncode)
    reason = "time" if process.returncode < 0 else "memory"
    payload = _resource_report(reason).encode("utf-8") + b"\n"
    _write_output(payload, output_path)
    sys.stdout.write(payload.decode("utf-8"))
    return 3

