"""Business error codes shared by the CLI, pipeline and snapshot layer."""

from __future__ import annotations

from dataclasses import dataclass


E_SCENE_INVALID = "E_SCENE_INVALID"
E_TILE_FAILED = "E_TILE_FAILED"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
E_SNAPSHOT_IO = "E_SNAPSHOT_IO"


@dataclass
class PipelineError(Exception):
    code: str
    message: str
    stage: str = "pipeline"

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "stage": self.stage}


class SnapshotCorruptError(PipelineError):
    def __init__(self, message: str) -> None:
        super().__init__(E_SNAPSHOT_CORRUPT, message, "snapshot")
