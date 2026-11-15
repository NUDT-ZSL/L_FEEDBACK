from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .util import (
    atomic_write,
    canonical_json_bytes,
    load_json_bytes,
    sha256_bytes,
)


STAGE_ORDER = ("parse", "spatialize", "mix", "loudness", "package")
STAGE_VERSIONS = {
    "parse": 1,
    "spatialize": 1,
    "mix": 1,
    "loudness": 1,
    "package": 1,
}


@dataclass
class LoadedSnapshot:
    stage: str
    version: int
    input_digest: str
    snapshot_digest: str
    meta: Dict[str, Any]
    files: Dict[str, bytes]


class SnapshotStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.root = out_dir / ".bake_state"

    def stage_dir(self, stage: str) -> Path:
        return self.root / stage

    def expected_input_digest(
        self,
        stage: str,
        global_input_digest: str,
        upstream_snapshot_digest: str,
    ) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "global_input_digest": global_input_digest,
                    "stage": stage,
                    "upstream_snapshot_digest": upstream_snapshot_digest,
                    "version": STAGE_VERSIONS[stage],
                }
            )
        )

    def load(
        self,
        stage: str,
        expected_input_digest: str,
    ) -> tuple[Optional[LoadedSnapshot], str]:
        directory = self.stage_dir(stage)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return None, "missing"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None, "corrupt_manifest"
        if manifest.get("stage") != stage:
            return None, "stage_mismatch"
        version = int(manifest.get("version", -1))
        if version != STAGE_VERSIONS[stage]:
            return None, "version_mismatch"
        if manifest.get("input_digest") != expected_input_digest:
            return None, "digest_mismatch"
        files = manifest.get("files")
        if not isinstance(files, dict):
            return None, "corrupt_manifest"
        body = {
            "stage": stage,
            "version": version,
            "input_digest": expected_input_digest,
            "files": files,
        }
        recomputed_digest = sha256_bytes(canonical_json_bytes(body))
        if manifest.get("snapshot_digest") != recomputed_digest:
            return None, "digest_mismatch"
        loaded_files: Dict[str, bytes] = {}
        for name in sorted(files):
            path = directory / name
            if not path.is_file():
                return None, "missing_file"
            data = path.read_bytes()
            if sha256_bytes(data) != files[name]:
                return None, "digest_mismatch"
            loaded_files[name] = data
        try:
            meta = load_json_bytes(loaded_files["meta.json"], f"{stage}/meta.json")
        except KeyError:
            return None, "missing_meta"
        except Exception:
            return None, "corrupt_meta"
        return (
            LoadedSnapshot(
                stage=stage,
                version=version,
                input_digest=expected_input_digest,
                snapshot_digest=recomputed_digest,
                meta=meta,
                files=loaded_files,
            ),
            "loaded",
        )

    def save(
        self,
        stage: str,
        input_digest: str,
        meta: Mapping[str, Any],
        data_files: Mapping[str, bytes],
    ) -> str:
        directory = self.stage_dir(stage)
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

        files: Dict[str, bytes] = {"meta.json": canonical_json_bytes(meta)}
        for name, data in data_files.items():
            if name == "meta.json":
                raise ValueError("data_files may not replace meta.json")
            files[name] = data

        file_hashes: Dict[str, str] = {}
        for name in sorted(files):
            atomic_write(directory / name, files[name])
            file_hashes[name] = sha256_bytes(files[name])

        body = {
            "stage": stage,
            "version": STAGE_VERSIONS[stage],
            "input_digest": input_digest,
            "files": file_hashes,
        }
        snapshot_digest = sha256_bytes(canonical_json_bytes(body))
        manifest = dict(body)
        manifest["snapshot_digest"] = snapshot_digest
        atomic_write(directory / "manifest.json", canonical_json_bytes(manifest))
        return snapshot_digest

