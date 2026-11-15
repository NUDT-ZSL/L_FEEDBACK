"""Parsing and validation of the JSON project description."""

import json
import os
import tempfile

from .errors import ConfigError, ParameterRangeError

REQUIRED_FIELDS = (
    "project_name",
    "input_files",
    "loop_start",
    "loop_end",
    "crossfade_duration",
    "output_dir",
)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_project(config_path):
    """Load and validate a project JSON file.

    Relative input/output paths are resolved against the directory that
    contains the project file.

    Returns a normalised project dictionary.  Raises :class:`ConfigError`
    (exit code 2) for structural problems and :class:`ParameterRangeError`
    (exit code 4) for out-of-range numeric parameters.
    """
    if not os.path.isfile(config_path):
        raise ConfigError("project file not found: %s" % config_path)

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            "invalid JSON in project file %s: %s" % (config_path, exc)
        ) from exc
    except OSError as exc:
        raise ConfigError(
            "cannot open project file %s: %s" % (config_path, exc)
        ) from exc

    if not isinstance(raw, dict):
        raise ConfigError("project file must contain a JSON object at the top")

    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ConfigError(
            "missing required project field(s): %s" % ", ".join(missing)
        )

    project_name = raw["project_name"]
    if not isinstance(project_name, str) or not project_name.strip():
        raise ConfigError("project_name must be a non-empty string")

    input_files = raw["input_files"]
    if not isinstance(input_files, list) or not input_files:
        raise ConfigError("input_files must be a non-empty list of paths")
    if not all(isinstance(item, str) and item for item in input_files):
        raise ConfigError("every entry in input_files must be a string path")

    base_dir = os.path.dirname(os.path.abspath(config_path))
    resolved_inputs = [
        item if os.path.isabs(item) else os.path.normpath(
            os.path.join(base_dir, item)
        )
        for item in input_files
    ]

    output_dir = raw["output_dir"]
    if not isinstance(output_dir, str) or not output_dir:
        raise ConfigError("output_dir must be a non-empty string path")
    resolved_output = (
        output_dir
        if os.path.isabs(output_dir)
        else os.path.normpath(os.path.join(base_dir, output_dir))
    )

    loop_start = raw["loop_start"]
    loop_end = raw["loop_end"]
    crossfade_duration = raw["crossfade_duration"]
    target_duration = raw.get("target_duration")

    if not _is_number(loop_start) or loop_start < 0:
        raise ParameterRangeError(
            "loop_start must be a non-negative number of seconds"
        )
    if not _is_number(loop_end) or loop_end <= 0:
        raise ParameterRangeError(
            "loop_end must be a positive number of seconds"
        )
    if loop_end <= loop_start:
        raise ParameterRangeError(
            "loop_end (%.6fs) must be greater than loop_start (%.6fs)"
            % (loop_end, loop_start)
        )
    if not _is_number(crossfade_duration):
        raise ParameterRangeError(
            "crossfade_duration must be a number of seconds"
        )
    if crossfade_duration <= 0:
        raise ParameterRangeError(
            "crossfade_duration must be greater than zero, got %r"
            % crossfade_duration
        )
    if target_duration is not None:
        if not _is_number(target_duration) or target_duration <= 0:
            raise ParameterRangeError(
                "target_duration must be a positive number of seconds"
            )

    overwrite = raw.get("overwrite", True)
    if not isinstance(overwrite, bool):
        raise ConfigError("overwrite must be a boolean (true/false)")

    _ensure_output_writable(resolved_output)

    return {
        "project_name": project_name.strip(),
        "input_files": resolved_inputs,
        "loop_start": float(loop_start),
        "loop_end": float(loop_end),
        "crossfade_duration": float(crossfade_duration),
        "target_duration": (
            None if target_duration is None else float(target_duration)
        ),
        "output_dir": resolved_output,
        "overwrite": overwrite,
        "config_path": os.path.abspath(config_path),
    }


def _ensure_output_writable(output_dir):
    """Create the output directory if needed and prove it is writable."""
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise ConfigError(
            "cannot create output directory %s: %s" % (output_dir, exc)
        ) from exc
    try:
        fd, probe = tempfile.mkstemp(prefix=".write_probe_", dir=output_dir)
    except OSError as exc:
        raise ConfigError(
            "output directory %s is not writable: %s" % (output_dir, exc)
        ) from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"0")
    finally:
        try:
            os.remove(probe)
        except OSError:
            pass
