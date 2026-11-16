from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import BusinessError
from .io_utils import (
    read_json,
    remove_path,
    remove_partial,
    sha256_file,
    utc_now,
    write_json_atomic,
)


def render_state_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".state.json")


def sample_state_path(audio_root: Path) -> Path:
    return audio_root / ".cutscene_dub_samples_state.json"


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    try:
        write_json_atomic(path, state)
    except BusinessError as exc:
        raise BusinessError.from_other(
            "E_STATE_WRITE",
            f"Unable to persist recovery state: {exc.message}",
            exc,
        ) from exc


def _best_effort_unlink(path_value: Any) -> None:
    if not isinstance(path_value, str):
        return
    try:
        Path(path_value).unlink()
    except (FileNotFoundError, OSError):
        pass


class RenderState:
    def __init__(self, path: Path) -> None:
        self.path = path
        loaded = read_json(path)
        self.data: dict[str, Any] = loaded if isinstance(loaded, dict) else {}

    def _artifact_hashes_match(
        self, artifacts: dict[str, Any], expected_paths: dict[str, Path]
    ) -> bool:
        for key, path in expected_paths.items():
            artifact = artifacts.get(key)
            if not isinstance(artifact, dict):
                return False
            if artifact.get("path") != str(path.resolve()):
                return False
            expected_hash = artifact.get("sha256")
            if not isinstance(expected_hash, str) or not path.is_file():
                return False
            try:
                if sha256_file(path) != expected_hash:
                    return False
            except BusinessError:
                return False
        return True

    def is_reusable(
        self,
        *,
        fingerprint: str,
        output_path: Path,
        report_path: Path,
    ) -> bool:
        if self.data.get("schema_version") != "1.0":
            return False
        if self.data.get("status") != "complete":
            return False
        if self.data.get("fingerprint") != fingerprint:
            return False
        artifacts = self.data.get("artifacts")
        if not isinstance(artifacts, dict):
            return False
        return self._artifact_hashes_match(
            artifacts,
            {"audio": output_path, "report": report_path},
        )

    def begin(
        self,
        *,
        fingerprint: str,
        output_path: Path,
        report_path: Path,
    ) -> None:
        self.data = {
            "schema_version": "1.0",
            "status": "writing",
            "fingerprint": fingerprint,
            "artifacts": {
                "audio": {
                    "path": str(output_path.resolve()),
                    "status": "writing",
                    "sha256": None,
                },
                "report": {
                    "path": str(report_path.resolve()),
                    "status": "pending",
                    "sha256": None,
                },
            },
            "error": None,
        }
        _save_state(self.path, self.data)
        remove_path(output_path)
        remove_partial(output_path)
        remove_path(report_path)
        remove_partial(report_path)

    def audio_published(self, sha256: str, frame_count: int) -> None:
        artifact = self.data.setdefault("artifacts", {}).setdefault("audio", {})
        artifact["status"] = "complete"
        artifact["sha256"] = sha256
        artifact["frames"] = frame_count
        _save_state(self.path, self.data)

    def report_published(self, sha256: str) -> None:
        artifact = self.data.setdefault("artifacts", {}).setdefault("report", {})
        artifact["status"] = "complete"
        artifact["sha256"] = sha256
        _save_state(self.path, self.data)

    def complete(self, *, frame_count: int, audio_sha256: str, report_sha256: str) -> None:
        artifacts = self.data.setdefault("artifacts", {})
        artifacts["audio"].update(
            {"status": "complete", "sha256": audio_sha256, "frames": frame_count}
        )
        artifacts["report"].update({"status": "complete", "sha256": report_sha256})
        self.data["status"] = "complete"
        self.data["error"] = None
        self.data["summary"] = {
            "frame_count": frame_count,
            "audio_sha256": audio_sha256,
            "report_sha256": report_sha256,
        }
        _save_state(self.path, self.data)

    def failed(self, exc: BusinessError, *, report_published: bool = False) -> None:
        self.data["status"] = "failed"
        self.data["error"] = exc.as_dict()
        artifacts = self.data.setdefault("artifacts", {})
        _best_effort_unlink(artifacts.get("audio", {}).get("path"))
        if report_published:
            report = artifacts.setdefault("report", {})
            report["status"] = "complete"
        else:
            _best_effort_unlink(artifacts.get("report", {}).get("path"))
            report = artifacts.setdefault("report", {})
            report["status"] = "incomplete"
            report["sha256"] = None
        audio = artifacts.setdefault("audio", {})
        audio["status"] = "incomplete"
        audio["sha256"] = None
        _save_state(self.path, self.data)

    def interrupted(self, stage: str) -> None:
        self.data["status"] = "interrupted"
        self.data["error"] = {
            "code": "E_INTERRUPTED",
            "message": "Processing was interrupted before all artifacts were committed.",
            "entry_id": None,
            "details": {"stage": stage},
        }
        artifacts = self.data.setdefault("artifacts", {})
        for artifact in artifacts.values():
            if isinstance(artifact, dict):
                _best_effort_unlink(artifact.get("path"))
                artifact["status"] = "incomplete"
                artifact["sha256"] = None
        _save_state(self.path, self.data)

    @property
    def summary(self) -> dict[str, Any]:
        summary = self.data.get("summary")
        return summary if isinstance(summary, dict) else {}


class SampleState:
    def __init__(self, path: Path) -> None:
        self.path = path
        loaded = read_json(path)
        self.data: dict[str, Any] = loaded if isinstance(loaded, dict) else {}

    def is_reusable(
        self,
        *,
        fingerprint: str,
        expected_paths: list[Path],
    ) -> bool:
        if self.data.get("schema_version") != "1.0":
            return False
        if self.data.get("status") != "complete":
            return False
        if self.data.get("fingerprint") != fingerprint:
            return False
        files = self.data.get("files")
        if not isinstance(files, dict):
            return False
        for path in expected_paths:
            record = files.get(str(path.resolve()))
            if not isinstance(record, dict) or not path.is_file():
                return False
            expected_hash = record.get("sha256")
            if not isinstance(expected_hash, str):
                return False
            try:
                if sha256_file(path) != expected_hash:
                    return False
            except BusinessError:
                return False
        return True

    def begin(self, *, fingerprint: str, expected_paths: list[Path]) -> None:
        self.data = {
            "schema_version": "1.0",
            "status": "writing",
            "fingerprint": fingerprint,
            "files": {
                str(path.resolve()): {"status": "pending", "sha256": None}
                for path in expected_paths
            },
            "error": None,
        }
        _save_state(self.path, self.data)
        for path in expected_paths:
            remove_path(path)
            remove_partial(path)

    def file_published(self, path: Path, sha256: str) -> None:
        record = self.data.setdefault("files", {}).setdefault(
            str(path.resolve()), {"status": "pending", "sha256": None}
        )
        record["status"] = "complete"
        record["sha256"] = sha256
        _save_state(self.path, self.data)

    def complete(self) -> None:
        self.data["status"] = "complete"
        self.data["error"] = None
        _save_state(self.path, self.data)

    def failed(self, exc: BusinessError) -> None:
        self.data["status"] = "failed"
        self.data["error"] = exc.as_dict()
        files = self.data.setdefault("files", {})
        for path_value, record in files.items():
            _best_effort_unlink(path_value)
            if isinstance(record, dict):
                record["status"] = "incomplete"
                record["sha256"] = None
        _save_state(self.path, self.data)

    def interrupted(self, stage: str) -> None:
        self.data["status"] = "interrupted"
        self.data["error"] = {
            "code": "E_INTERRUPTED",
            "message": "Sample generation was interrupted before completion.",
            "entry_id": None,
            "details": {"stage": stage},
        }
        files = self.data.setdefault("files", {})
        for path_value, record in files.items():
            _best_effort_unlink(path_value)
            if isinstance(record, dict):
                record["status"] = "incomplete"
                record["sha256"] = None
        _save_state(self.path, self.data)

