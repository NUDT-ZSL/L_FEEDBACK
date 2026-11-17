"""Content-hash cache, persistent state file, locking, atomic writes.

- Cache segments are content-addressed: the file name is the hash of the
  source bytes plus the render parameters, so concurrent writers always
  write identical bytes.
- Every file is published with write-temp-then-os.replace, so a reader
  never observes a half-written cache segment or state file.
- The shared state index is mutated under a cross-process lock file
  (atomic O_EXCL create), so concurrent workers cannot corrupt it.
- A corrupted or unreadable state file is treated as empty, which safely
  falls back to a full re-render.
"""

import hashlib
import json
import os
import time

from .errors import BusinessError, ErrorCode, ExitCode

STATE_VERSION = 1


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, data):
    """Publish bytes at path atomically (temp file in same dir + replace)."""
    tmp = f"{path}.tmp-{os.getpid()}-{time.time_ns()}"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


class FileLock:
    """Cross-process advisory lock via atomic lock-file creation."""

    def __init__(self, path, timeout_s=30.0, stale_s=120.0):
        self.path = path
        self.timeout_s = timeout_s
        self.stale_s = stale_s
        self.acquired = False

    def acquire(self):
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("ascii"))
                os.close(fd)
                self.acquired = True
                return
            except FileExistsError:
                self._break_if_stale()
                if time.monotonic() > deadline:
                    raise BusinessError(
                        ErrorCode.E_STATE_LOCK,
                        f"timed out acquiring state lock: {self.path}",
                        ExitCode.INTERNAL,
                    )
                time.sleep(0.05)

    def _break_if_stale(self):
        try:
            age = time.time() - os.path.getmtime(self.path)
            if age > self.stale_s:
                os.remove(self.path)
        except OSError:
            pass

    def release(self):
        if self.acquired:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def _empty_state():
    return {"version": STATE_VERSION, "entries": {}}


class StateStore:
    """Persistent hash index mapping render keys to cache segment files."""

    def __init__(self, state_dir):
        self.state_dir = state_dir
        self.state_path = os.path.join(state_dir, "state.json")
        self.lock_path = os.path.join(state_dir, "state.lock")
        self.segments_dir = os.path.join(state_dir, "segments")
        os.makedirs(self.segments_dir, exist_ok=True)

    def load(self):
        """Load state; any corruption safely falls back to an empty state."""
        try:
            with open(self.state_path, "rb") as fh:
                state = json.loads(fh.read().decode("utf-8"))
            if not isinstance(state, dict) or not isinstance(state.get("entries"), dict):
                return _empty_state()
            if state.get("version") != STATE_VERSION:
                return _empty_state()
            return state
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return _empty_state()

    def get_entry(self, key):
        return self.load()["entries"].get(key)

    def update_entries(self, new_entries):
        """Merge entries into the shared index under the file lock."""
        with FileLock(self.lock_path):
            state = self.load()
            state["entries"].update(new_entries)
            atomic_write_json(self.state_path, state)

    def segment_path(self, key):
        return os.path.join(self.segments_dir, f"{key}.f32")

    def segment_meta_path(self, key):
        return os.path.join(self.segments_dir, f"{key}.json")
