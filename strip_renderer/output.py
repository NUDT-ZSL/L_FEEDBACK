from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .errors import EXIT_ASSET_OR_ENVIRONMENT, RenderError
from .render import RenderResult
from .report import CHANNEL_FILES, REPORT_FILE, build_report, quantize_rgb
from .scene import Scene


def _probe_directory(directory: Path) -> None:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RenderError(
            code="E_OUTPUT_NOT_WRITABLE",
            message=f"cannot create output directory {directory}: {exc}",
            exit_code=EXIT_ASSET_OR_ENVIRONMENT,
            details={"path": str(directory)},
        ) from exc
    if not directory.is_dir():
        raise RenderError(
            code="E_OUTPUT_NOT_WRITABLE",
            message=f"output path is not a directory: {directory}",
            exit_code=EXIT_ASSET_OR_ENVIRONMENT,
            details={"path": str(directory)},
        )
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".strip_renderer_probe_",
            dir=directory,
            delete=False,
        ) as handle:
            handle.write(b"probe")
            probe_path = Path(handle.name)
    except OSError as exc:
        raise RenderError(
            code="E_OUTPUT_NOT_WRITABLE",
            message=f"output directory is not writable: {directory}: {exc}",
            exit_code=EXIT_ASSET_OR_ENVIRONMENT,
            details={"path": str(directory)},
        ) from exc
    finally:
        if probe_path is not None:
            try:
                probe_path.unlink()
            except OSError:
                pass


def prepare_output_directory(scene: Scene) -> None:
    _probe_directory(scene.output.directory)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.write_text(text + "\n", encoding="utf-8")


def _commit_staged_files(
    staging: Path,
    staged_names: list[str],
    target_directory: Path,
) -> None:
    targets = [target_directory / name for name in staged_names]
    backups: dict[Path, Path] = {}
    committed: set[Path] = set()
    try:
        for target in targets:
            if target.exists():
                backup = staging / f".backup_{target.name}"
                os.replace(target, backup)
                backups[target] = backup
        for name in staged_names:
            target = target_directory / name
            os.replace(staging / name, target)
            committed.add(target)
    except Exception as exc:
        for target in committed:
            try:
                if target.exists():
                    target.unlink()
            except OSError:
                pass
        for target, backup in backups.items():
            try:
                if target.exists():
                    target.unlink()
                if backup.exists():
                    os.replace(backup, target)
            except OSError:
                pass
        raise RenderError(
            code="E_OUTPUT_NOT_WRITABLE",
            message=f"cannot commit output files in {target_directory}: {exc}",
            exit_code=EXIT_ASSET_OR_ENVIRONMENT,
            details={"path": str(target_directory)},
        ) from exc


def write_render_outputs(scene: Scene, result: RenderResult) -> Path:
    _probe_directory(scene.output.directory)
    report = build_report(scene, result)
    quantized = {
        key: quantize_rgb(channel)
        for key, channel in (
            ("base_color", result.base_color),
            ("direct", result.direct_light),
            ("ambient", result.ambient_light),
            ("composite", result.composite),
        )
    }

    staging = Path(
        tempfile.mkdtemp(
            prefix=".strip_renderer_stage_",
            dir=scene.output.directory,
        )
    )
    staged_names = list(CHANNEL_FILES.values()) + [REPORT_FILE]
    try:
        for key, filename in CHANNEL_FILES.items():
            image = Image.fromarray(quantized[key], mode="RGB")
            image.save(staging / filename, format="PNG")
        _write_json(staging / REPORT_FILE, report)
        _commit_staged_files(staging, staged_names, scene.output.directory)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return scene.output.directory / REPORT_FILE


def write_error_report(
    error: RenderError,
    candidate_directories: list[Path],
) -> Path | None:
    for index, directory in enumerate(candidate_directories):
        if directory is None:
            continue
        filename = REPORT_FILE if index == 0 else "render_error_report.json"
        staging: Path | None = None
        try:
            _probe_directory(directory)
            staging = Path(
                tempfile.mkdtemp(
                    prefix=".strip_renderer_error_",
                    dir=directory,
                )
            )
            staged = staging / filename
            _write_json(staged, error.as_dict())
            os.replace(staged, directory / filename)
            return directory / filename
        except (OSError, RenderError):
            continue
        finally:
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)
    return None
