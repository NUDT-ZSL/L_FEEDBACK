"""Cross-platform process peak-memory and elapsed-time monitoring."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from pathlib import Path

from .errors import perf_error


class PerformanceMonitor:
    def __init__(
        self,
        memory_limit_mb: float = 512.0,
        time_limit_sec: float = 30.0,
        *,
        start: float | None = None,
    ) -> None:
        self.memory_limit_bytes = int(memory_limit_mb * 1024 * 1024)
        self.time_limit_sec = float(time_limit_sec)
        self.started_at = time.perf_counter() if start is None else start
        self.peak_memory_bytes = 0
        self._last_memory_bytes = 0

    def elapsed_seconds(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)

    def _read_peak_memory(self) -> int:
        if sys.platform == "win32":
            return self._read_windows_peak()
        try:
            status = Path("/proc/self/status").read_text(encoding="ascii", errors="ignore")
            for line in status.splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return int(usage) if sys.platform == "darwin" else int(usage) * 1024
        except (AttributeError, ImportError, OSError, ValueError):
            return self._read_current_rss()

    def _read_windows_peak(self) -> int:
        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        )
        if not ok:
            return self._read_current_rss()
        return int(counters.PeakWorkingSetSize)

    def _read_current_rss(self) -> int:
        try:
            statm = Path("/proc/self/statm").read_text(encoding="ascii").split()
            return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError, AttributeError):
            return 0

    def sample(self) -> int:
        measured = int(self._read_peak_memory())
        self._last_memory_bytes = max(self._last_memory_bytes, measured)
        self.peak_memory_bytes = max(self.peak_memory_bytes, measured)
        return self.peak_memory_bytes

    def check(self) -> None:
        peak = self.sample()
        elapsed = self.elapsed_seconds()
        if peak > self.memory_limit_bytes:
            raise perf_error(
                f"peak process memory {peak} bytes exceeds limit "
                f"{self.memory_limit_bytes} bytes (elapsed {elapsed:.3f}s)",
                peak_memory_bytes=peak,
                memory_limit_bytes=self.memory_limit_bytes,
                elapsed_seconds=elapsed,
                limit_kind="memory",
            )
        if elapsed > self.time_limit_sec:
            raise perf_error(
                f"elapsed time {elapsed:.3f}s exceeds limit "
                f"{self.time_limit_sec:.3f}s (peak memory {peak} bytes)",
                peak_memory_bytes=peak,
                memory_limit_bytes=self.memory_limit_bytes,
                elapsed_seconds=elapsed,
                time_limit_seconds=self.time_limit_sec,
                limit_kind="time",
            )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "peak_memory_bytes": self.peak_memory_bytes,
            "peak_memory_mb": round(self.peak_memory_bytes / (1024.0 * 1024.0), 3),
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_limit_mb": round(self.memory_limit_bytes / (1024.0 * 1024.0), 3),
            "elapsed_seconds": round(self.elapsed_seconds(), 6),
            "time_limit_seconds": round(self.time_limit_sec, 6),
        }

