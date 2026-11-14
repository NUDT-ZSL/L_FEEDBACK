from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass


def _linux_peak_bytes() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _linux_current_bytes() -> int | None:
    try:
        with open("/proc/self/status", "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _resource_bytes() -> tuple[int | None, int | None]:
    try:
        import resource
    except ImportError:
        return None, None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    return peak, peak


class _ProcessMemoryCounters(ctypes.Structure):
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
    ]


def _windows_bytes() -> tuple[int | None, int | None]:
    if os.name != "nt":
        return None, None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        function = psapi.GetProcessMemoryInfo
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        function.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        ok = function(process, ctypes.byref(counters), counters.cb)
    except (AttributeError, OSError):
        return None, None
    if not ok:
        return None, None
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def current_rss_bytes() -> int | None:
    if os.name == "nt":
        return _windows_bytes()[0]
    current = _linux_current_bytes()
    if current is not None:
        return current
    return _resource_bytes()[0]


def process_peak_rss_bytes() -> int | None:
    if os.name == "nt":
        return _windows_bytes()[1]
    peak = _linux_peak_bytes()
    if peak is not None:
        return peak
    return _resource_bytes()[1]


@dataclass
class PeakMemoryTracker:
    current_bytes: int = 0
    peak_bytes: int = 0

    def __post_init__(self) -> None:
        current = current_rss_bytes()
        peak = process_peak_rss_bytes()
        if current is not None:
            self.current_bytes = max(self.current_bytes, current)
        if peak is not None:
            self.peak_bytes = max(self.peak_bytes, peak)
        self.sample()

    def sample(self) -> None:
        current = current_rss_bytes()
        if current is not None:
            self.current_bytes = max(self.current_bytes, current)
        peak = process_peak_rss_bytes()
        if peak is not None:
            self.peak_bytes = max(self.peak_bytes, peak)
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024.0 * 1024.0)
