"""Versioned, fingerprint-checked checkpoint store with atomic writes."""

import hashlib
import json
import os

from .errors import CheckpointVersionError

CHECKPOINT_VERSION = 2
CHECKPOINT_FILENAME = "checkpoints.v%d.json" % CHECKPOINT_VERSION


def parameter_fingerprint(params):
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _discover_legacy_files(directory):
    return [
        name
        for name in os.listdir(directory)
        if name.startswith("checkpoints.v") and name.endswith(".json")
        and name != CHECKPOINT_FILENAME
    ]


class CheckpointStore:
    """JSON dict keyed by clip token; format version stored inside the file."""

    def __init__(self, directory):
        self.directory = directory
        self.path = os.path.join(directory, CHECKPOINT_FILENAME)
        os.makedirs(directory, exist_ok=True)
        self._data = self._load()

    def _load(self):
        # Refuse to interpret an older/other version by byte offset or layout.
        for legacy in _discover_legacy_files(self.directory):
            raise CheckpointVersionError(
                "checkpoint file version mismatch: %s" % legacy,
                details={"expected": CHECKPOINT_VERSION, "found_file": legacy},
            )
        if not os.path.exists(self.path):
            return {"version": CHECKPOINT_VERSION, "clips": {}}
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION:
            raise CheckpointVersionError(
                "checkpoint content version mismatch",
                details={"expected": CHECKPOINT_VERSION, "found": data.get("version")},
            )
        return data

    def is_current(self, token, input_hash, param_fp):
        rec = self._data["clips"].get(token)
        if not rec:
            return False
        return (
            rec.get("input_hash") == input_hash
            and rec.get("param_fingerprint") == param_fp
            and rec.get("status") == "done"
        )

    def get(self, token):
        return self._data["clips"].get(token, {})

    def record(self, token, input_hash, param_fp, outputs, degradation=None):
        self._data["clips"][token] = {
            "status": "done",
            "input_hash": input_hash,
            "param_fingerprint": param_fp,
            "outputs": outputs,
            "recoverable": (degradation or {}).get("recoverable", []),
            "fatal": (degradation or {}).get("fatal", []),
        }

    def save_atomic(self):
        tmp = self.path + ".tmp"
        payload = json.dumps(self._data, sort_keys=True, indent=2)
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
