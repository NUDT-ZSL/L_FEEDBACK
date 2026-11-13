"""Re-entrant single-process lock for a synthesis output directory.

The lock is an exclusively created file containing JSON owner metadata.
Stale locks whose owner process no longer exists can be safely taken over;
live owners cause a non-zero exit with a readable message instead of any
overwrite of an in-progress checkpoint.
"""

import json
import os
import sys
import tempfile

LOCK_FILE_NAME = ".voiceforge.lock"


class LockUnavailableError(RuntimeError):
    pass


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return True
            STILL_ACTIVE = 259
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ProcessLock:
    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, LOCK_FILE_NAME)
        self.token = ""
        self._held = False

    def acquire(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        host = os.environ.get("COMPUTERNAME")
        if not host:
            host = os.uname().nodename if hasattr(os, "uname") else "unknown"
        owner = {
            "pid": os.getpid(),
            "host": host,
            "platform": sys.platform,
            "program": "voiceforge",
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            self._handle_existing_lock()
            return self.acquire()
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(owner, handle, indent=2)
            self._held = True
            self.token = str(owner["pid"])
        except Exception:
            os.unlink(self.path)
            raise

    def _handle_existing_lock(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                owner = json.load(handle)
            owner_pid = int(owner.get("pid", -1))
        except (OSError, ValueError, KeyError):
            owner_pid = -1
        if _process_alive(owner_pid):
            raise LockUnavailableError(
                f"输出目录已被进程 PID={owner_pid} 锁定（{self.path}）。"
                "若确认没有其他合成进程在运行，请手动删除该锁文件后重试。")
        os.unlink(self.path)

    def release(self) -> None:
        if self._held and os.path.exists(self.path):
            try:
                os.unlink(self.path)
            except OSError:
                pass
        self._held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.release()
        return False


def atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

