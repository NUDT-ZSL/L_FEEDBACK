"""Best-effort all-or-nothing multi-file replacement without kill handlers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _write_staged_file(path: Path, data: bytes) -> Path:
    parent = path.parent
    if not parent.is_dir():
        raise NotADirectoryError(f"output directory does not exist: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage-", suffix=".tmp", dir=str(parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        return temporary
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _backup_name(path: Path) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.backup-", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def commit_outputs(files: list[tuple[Path, bytes]]) -> None:
    """Stage every file, then replace targets; restore old targets on failure."""

    if not files:
        return
    absolute_targets = [
        os.path.normcase(os.path.realpath(os.fspath(path))) for path, _ in files
    ]
    if len(set(absolute_targets)) != len(absolute_targets):
        raise ValueError("duplicate output target paths")

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for target, data in files:
            if target.exists() and target.is_dir():
                raise IsADirectoryError(f"output target is a directory: {target}")
            staged[target] = _write_staged_file(target, data)

        for target, _ in files:
            if os.path.lexists(target):
                backup = _backup_name(target)
                os.replace(target, backup)
                backups[target] = backup
            os.replace(staged[target], target)
            committed.append(target)
    except Exception:
        rollback_errors: list[str] = []
        for target in reversed(committed):
            try:
                _remove_file(target)
            except OSError as exc:
                rollback_errors.append(f"cannot remove new target {target}: {exc}")
        for target, backup in backups.items():
            try:
                if os.path.lexists(backup):
                    os.replace(backup, target)
            except OSError as exc:
                rollback_errors.append(f"cannot restore {target}: {exc}")
        for temporary in staged.values():
            try:
                _remove_file(temporary)
            except OSError as exc:
                rollback_errors.append(f"cannot remove staged file {temporary}: {exc}")
        for backup in backups.values():
            try:
                _remove_file(backup)
            except OSError as exc:
                rollback_errors.append(f"cannot remove backup file {backup}: {exc}")
        if rollback_errors:
            raise OSError(
                "output commit failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise

    for backup in backups.values():
        try:
            _remove_file(backup)
        except OSError:
            pass

