"""工程状态文件、恢复校验与工程构建入口。"""

from __future__ import annotations

import json
from pathlib import Path

from . import STATE_VERSION
from .errors import ErrorCode, make_error_record
from .fingerprint import file_fingerprint
from .mixer import Mixer, SnapshotManager, TrackConfig
from .sources import SourceLibrary
from .timeline import Timeline, build_timeline


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Project:
    """聚合素材库、时间轴与混音参数，并支持状态持久化/恢复。"""

    def __init__(
        self,
        root_dir: Path,
        project_rate: int,
        library: SourceLibrary,
        timeline: Timeline,
        tracks: dict[str, TrackConfig],
        alignment_report: list[dict],
        script: dict,
        mix_config: dict,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.project_rate = int(project_rate)
        self.library = library
        self.timeline = timeline
        self.tracks = tracks
        self.alignment_report = alignment_report
        self.script = script
        self.mix_config = mix_config
        self.mixer = Mixer(timeline, library, tracks)
        self.snapshots = SnapshotManager(self.mixer)

    @classmethod
    def build(
        cls,
        root_dir: Path,
        script_path: Path,
        mix_path: Path,
        default_fade_in: float = 0.0,
        default_fade_out: float = 0.0,
    ) -> "Project":
        root_dir = Path(root_dir)
        script = _load_json(Path(script_path))
        mix_config = _load_json(Path(mix_path))
        project_rate = int(script.get("project_rate", 48000))
        library = SourceLibrary(project_rate, root_dir)
        for entry in script.get("sources", []):
            library.load(str(entry["source_id"]), entry["path"])
        timeline, alignment_report = build_timeline(
            script, library, default_fade_in, default_fade_out
        )
        tracks = {
            name: TrackConfig.from_dict(cfg)
            for name, cfg in mix_config.get("tracks", {}).items()
        }
        return cls(
            root_dir=root_dir,
            project_rate=project_rate,
            library=library,
            timeline=timeline,
            tracks=tracks,
            alignment_report=alignment_report,
            script=script,
            mix_config=mix_config,
        )

    def state_payload(self) -> dict:
        return {
            "version": STATE_VERSION,
            "project_rate": self.project_rate,
            "sources": [
                {
                    "source_id": source_id,
                    "path": _relative(self.root_dir, source.path),
                    "fingerprint": source.fingerprint,
                    "original_rate": source.original_rate,
                }
                for source_id, source in sorted(self.library.items())
            ],
            "timeline": self.timeline.to_list(),
            "tracks": {
                name: cfg.to_dict() for name, cfg in sorted(self.tracks.items())
            },
            "errors": self.library.errors,
        }

    def save(self, state_path: Path) -> Path:
        path = Path(state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.state_payload(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path

    @classmethod
    def restore(cls, root_dir: Path, state_path: Path) -> "RestoredProject":
        """依据素材内容指纹恢复工程；指纹不一致的素材对应片段被拒绝。"""
        root_dir = Path(root_dir)
        state = _load_json(Path(state_path))
        project_rate = int(state["project_rate"])
        library = SourceLibrary(project_rate, root_dir)
        verify_errors: list[dict] = []
        rejected_sources: set[str] = set()
        for entry in state.get("sources", []):
            source_id = str(entry["source_id"])
            path = library.resolve(entry["path"])
            if not path.exists():
                verify_errors.append(
                    make_error_record(path, ErrorCode.E_MISSING, "恢复校验时文件缺失")
                )
                rejected_sources.add(source_id)
                continue
            current = file_fingerprint(path)
            if current != entry["fingerprint"]:
                verify_errors.append(
                    {
                        "path": str(path),
                        "code": "E_CHANGED",
                        "detail": "素材内容指纹与工程状态不一致，拒绝恢复对应片段",
                        "expected": entry["fingerprint"],
                        "actual": current,
                    }
                )
                rejected_sources.add(source_id)
                continue
            library.load(source_id, entry["path"])
        timeline = Timeline.from_list(project_rate, state.get("timeline", []))
        rejected_clips = [
            clip.clip_id
            for clip in timeline.clips
            if clip.source_id in rejected_sources
        ]
        timeline.clips = [
            clip
            for clip in timeline.clips
            if clip.source_id not in rejected_sources
        ]
        tracks = {
            name: TrackConfig.from_dict(cfg)
            for name, cfg in state.get("tracks", {}).items()
        }
        project = cls(
            root_dir=root_dir,
            project_rate=project_rate,
            library=library,
            timeline=timeline,
            tracks=tracks,
            alignment_report=[],
            script={},
            mix_config={},
        )
        return RestoredProject(
            project=project,
            verify_errors=verify_errors,
            rejected_sources=sorted(rejected_sources),
            rejected_clips=rejected_clips,
            state_version=int(state.get("version", 0)),
        )


class RestoredProject:
    """恢复结果：工程本体与校验差异。"""

    def __init__(
        self,
        project: Project,
        verify_errors: list[dict],
        rejected_sources: list[str],
        rejected_clips: list[str],
        state_version: int,
    ) -> None:
        self.project = project
        self.verify_errors = verify_errors
        self.rejected_sources = rejected_sources
        self.rejected_clips = rejected_clips
        self.state_version = state_version


def _relative(root_dir: Path, path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)

